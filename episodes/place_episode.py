"""
Run scripted Jenga placement episodes through normal Viam APIs.

One episode places one block: reset the tower, pick the next block from the
supply region, carry it above its slot, lower and release it, adjust it if it
landed off target, and retreat. Like `oracle_episode.py`, this is an oracle: it
plans from exact sim poses.

Training data cannot mask frames, so every deliberate error happens before
capture starts and only the recovery is recorded:
  C  the arm starts from a perturbed pose,
  D  the last tower block starts misplaced; the episode straightens it first,
  E  the arm starts hovering empty, as after a missed grasp, beside the block.
Set F (`--chain`) builds a whole tower without resets, keeping the oracle's own
small errors.

The tower blocks must exist as props in the sim world config; print them with
`--print-props`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from viam.components.arm import Arm, JointPositions
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Pose as ViamPose, PoseInFrame
from viam.proto.service.motion import Constraints
from viam.services.motion import MotionClient

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__:
    from . import jenga_tower as jt
    from .oracle_episode import (
        APPROACH_WAYPOINT_MIN_DURATION_SEC,
        CENTER_MOVE_CONSTRAINTS,
        FINAL_DESCENT_MIN_DURATION_SEC,
        HOME_JOINT_DEGREES,
        LINEAR_PICK_CONSTRAINTS,
        MIN_SUBTASK_DURATION_SEC,
        OPEN_SETTLE_MAX_SEC,
        PRE_GRASP_CLEARANCE_M,
        _ElapsedClock,
        _PhaseTimer,
        _destination,
        _execute_trajectory,
        _move,
        _plan_move,
        _prepare_trajectory,
        _wait_for_firm_grasp,
        _wait_for_gripper_settled,
        _start_episode_capture,
        _stop_episode_capture,
        _wait_sim_duration,
        connect,
    )
else:
    import jenga_tower as jt
    from oracle_episode import (
        APPROACH_WAYPOINT_MIN_DURATION_SEC,
        CENTER_MOVE_CONSTRAINTS,
        FINAL_DESCENT_MIN_DURATION_SEC,
        HOME_JOINT_DEGREES,
        LINEAR_PICK_CONSTRAINTS,
        MIN_SUBTASK_DURATION_SEC,
        OPEN_SETTLE_MAX_SEC,
        PRE_GRASP_CLEARANCE_M,
        _ElapsedClock,
        _PhaseTimer,
        _destination,
        _execute_trajectory,
        _move,
        _plan_move,
        _prepare_trajectory,
        _wait_for_firm_grasp,
        _wait_for_gripper_settled,
        _start_episode_capture,
        _stop_episode_capture,
        _wait_sim_duration,
        connect,
    )

DEFAULT_PROMPT = "place the next block on the tower"
# Phase labels for the result record. Capture uses one "base" stage, as the
# pick oracle does, so these are not written into the episode.
STAGE_STRAIGHTEN = "straighten the top block of the tower"
STAGE_PICK = "pick up the next block"
STAGE_CARRY = "carry the block above its slot on the tower"
STAGE_LOWER = "lower the block into its slot and open the gripper"
STAGE_ADJUST = "adjust the block into its slot"
STAGE_RETREAT = "lift the gripper away from the tower"
STAGE_HOLD = "hold still"

# Tower reset: blocks are set this far above their rest height per layer so
# they never start interpenetrating, then allowed to settle.
RESET_CLEARANCE_PER_LAYER_M = 0.0005
RESET_SETTLE_SEC = 1.0
RESET_STABLE_XY_M = 0.003
RESET_STABLE_YAW_DEG = 2.0
RESET_STABLE_Z_M = 0.004
# A grasp counts if the block rose at least this far with the gripper.
MIN_LIFT_TO_COUNT_AS_HELD_M = 0.02


@dataclass(frozen=True)
class GraspConfig:
    """How the gripper holds a block.

    `axis` "end" closes the jaws along the block's long axis (fingers on the
    75 mm ends, free when blocks sit side by side; needs the Gripper G2's
    84 mm stroke). "side" closes across the 25 mm width (fits the Lite6
    wide jaw; the fingers need room beside neighbouring blocks).
    `jaw_yaw_offset_deg` is the tool yaw at which the jaws close along world
    X; calibrate it once per gripper asset."""

    axis: str = "end"
    jaw_yaw_offset_deg: float = 0.0
    # TCP height relative to the block centre while grasping. Keep the
    # fingertips inside the block's thickness so they clear the layer below.
    grasp_z_offset_m: float = 0.0
    approach_clearance_m: float = 0.10
    carry_clearance_m: float = 0.06
    lift_height_m: float = 0.08
    release_clearance_m: float = 0.002
    release_settle_sec: float = 0.3
    gripper_settle_sec: float = 3.0
    # Corrective re-grasp: lift this much, shift, lower, release.
    adjust_lift_m: float = 0.003
    max_adjust_attempts: int = 2

    def __post_init__(self) -> None:
        if self.axis not in ("end", "side"):
            raise ValueError("grasp axis must be 'end' or 'side'")


@dataclass(frozen=True)
class NoiseConfig:
    """Set C: perturb the arm before capture, and the pre-release arrival."""

    start_xy_stdev_m: float = 0.03
    start_z_stdev_m: float = 0.02
    arrival_xy_stdev_m: float = 0.004
    arrival_yaw_stdev_deg: float = 4.0


@dataclass(frozen=True)
class MachineNames:
    world: str = "sim-world"
    arm: str = "pick-arm"
    gripper: str = "pick-gripper"
    motion: str = "builtin"
    fixed_camera: str = "scene-cam"
    fixed_camera_2: str = "third-cam"
    wrist_camera: str = "wrist-cam"


def tool_yaw_deg(block_yaw_deg: float, grasp: GraspConfig) -> float:
    across = 0.0 if grasp.axis == "end" else 90.0
    return jt.wrap_deg(block_yaw_deg + across + grasp.jaw_yaw_offset_deg, 180.0)


def release_target(
    slot: jt.Pose,
    lifted: jt.Pose,
    tcp_xy: tuple[float, float],
    pick_tool_yaw_deg: float,
) -> tuple[float, float, float, dict[str, float]]:
    """TCP x, y and tool yaw that put the held block, not the TCP, on `slot`.

    The block's offset and yaw relative to the tool are read after the lift
    and carried rigidly with the tool to the slot."""
    off_x = lifted.position[0] - tcp_xy[0]
    off_y = lifted.position[1] - tcp_xy[1]
    # Offset in the tool frame at pick time.
    c, s = math.cos(math.radians(-pick_tool_yaw_deg)), math.sin(math.radians(-pick_tool_yaw_deg))
    tool_x, tool_y = c * off_x - s * off_y, s * off_x + c * off_y
    block_in_tool_deg = jt.wrap_deg(lifted.yaw_deg - pick_tool_yaw_deg, 180.0)
    place_yaw = jt.wrap_deg(slot.yaw_deg - block_in_tool_deg, 180.0)
    c, s = math.cos(math.radians(place_yaw)), math.sin(math.radians(place_yaw))
    world_x, world_y = c * tool_x - s * tool_y, s * tool_x + c * tool_y
    return (
        slot.position[0] - world_x,
        slot.position[1] - world_y,
        place_yaw,
        {"x_m": tool_x, "y_m": tool_y, "yaw_deg": block_in_tool_deg},
    )


def _pose_from_reading(reading: Mapping[str, Any]) -> jt.Pose:
    pose = reading.get("pose", reading)
    return jt.Pose(
        tuple(float(v) for v in pose["position"]),  # type: ignore[arg-type]
        jt.quaternion_wxyz_to_yaw_deg(pose["orientation_wxyz"]),
    )


class _Episode:
    """One placement episode's plan-and-move helpers and capture state."""

    def __init__(
        self,
        world: Any,
        arm: Any,
        gripper: Any,
        motion: Any,
        names: MachineNames,
        grasp: GraspConfig,
    ) -> None:
        self.world = world
        self.arm = arm
        self.gripper = gripper
        self.motion = motion
        self.names = names
        self.grasp = grasp
        self.timer = _PhaseTimer()
        self.capturing = False
        self.phases: list[str] = []

    async def read_pose(self, prop: str) -> jt.Pose:
        reading = await self.world.do_command({"command": "get_prop_pose", "prop": prop})
        return _pose_from_reading(reading)

    async def read_poses(self, props: Sequence[str]) -> dict[str, jt.Pose]:
        return {name: await self.read_pose(name) for name in props}

    async def mark(self, phase: str) -> None:
        """Record that a phase began. Capture keeps the oracle's single base
        stage, so this only feeds the result record."""
        if phase not in self.phases:
            self.phases.append(phase)

    async def go(
        self,
        label: str,
        x: float,
        y: float,
        z: float,
        yaw: float,
        constraints: Constraints,
        minimum_duration_sec: float,
    ) -> None:
        """Plan with physics paused (while capturing), then stream the move."""
        destination = _destination(x, y, z, yaw)
        started = time.monotonic()
        if self.capturing:
            await self.world.do_command({"command": "pause"})
        try:
            plan = await _plan_move(
                self.motion, self.names.gripper, destination, constraints=constraints
            )
            trajectory = await _prepare_trajectory(
                self.arm, self.names.arm, plan, minimum_duration_sec
            )
        finally:
            if self.capturing:
                await self.world.do_command({"command": "play"})
        plan_sec = time.monotonic() - started
        self.timer.add("plan", plan_sec)
        execute_started = time.monotonic()
        await _execute_trajectory(self.arm, trajectory)
        execute_sec = time.monotonic() - execute_started
        self.timer.add("execute", execute_sec)
        self.timer.moves.append(
            {
                "label": label,
                "plan_sec": plan_sec,
                "execute_sec": execute_sec,
                "planned_sec": trajectory[-1].time.total_seconds(),
            }
        )

    async def open(self, label: str) -> None:
        with self.timer.phase(label):
            await self.gripper.open()
            self.timer.waits[label] = await _wait_for_gripper_settled(
                self.world, self.gripper, OPEN_SETTLE_MAX_SEC, f"gripper {label} wait"
            )

    async def grab(self) -> bool:
        with self.timer.phase("grab"):
            if not await self.gripper.grab():
                return False
            grabbed, _meta = await _wait_for_firm_grasp(
                self.world, self.gripper, self.grasp.gripper_settle_sec
            )
            return grabbed

    async def tcp(self) -> tuple[float, float, float, float]:
        """Current TCP x, y, z (m) and tool yaw (deg) in the world frame."""
        pose = (await self.motion.get_pose(self.names.gripper, "world")).pose
        return pose.x / 1000.0, pose.y / 1000.0, pose.z / 1000.0, -pose.theta

    async def transit(
        self, label: str, x: float, y: float, yaw: float, clear_z: float
    ) -> float:
        """Rise straight up to at least `clear_z`, then cross at that height to
        above (x, y). The tower is not an obstacle in the motion planner's
        world, so every sideways move happens above it. Returns the height."""
        cx, cy, cz, cyaw = await self.tcp()
        height = max(cz, clear_z)
        if height - cz > 1e-4:
            await self.go(
                f"{label}-rise", cx, cy, height, cyaw,
                LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
            )
        await self.go(
            f"{label}-cross", x, y, height, yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        return height

    async def pick(
        self, block: jt.Pose, label: str, clear_z: float, lift_z: float
    ) -> bool:
        """Cross above the tower, descend onto the block, grab, and lift
        straight up to `lift_z`. Returns whether the gripper *reports* holding
        the block; callers should check the block's measured height too."""
        x, y, z = block.position
        yaw = tool_yaw_deg(block.yaw_deg, self.grasp)
        grasp_z = z + self.grasp.grasp_z_offset_m
        await self.transit(
            f"{label}-approach", x, y, yaw,
            max(clear_z, z + self.grasp.approach_clearance_m),
        )
        await self.go(
            f"{label}-pre-grasp", x, y, grasp_z + PRE_GRASP_CLEARANCE_M, yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        await self.go(
            f"{label}-grasp", x, y, grasp_z, yaw,
            LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC,
        )
        grabbed = await self.grab()
        await self.go(
            f"{label}-lift", x, y, max(lift_z, grasp_z + self.grasp.lift_height_m), yaw,
            LINEAR_PICK_CONSTRAINTS, MIN_SUBTASK_DURATION_SEC,
        )
        if not grabbed:
            return False
        status = await self.gripper.is_holding_something()
        return bool(status.is_holding_something)


    async def adjust(
        self, block: str, target: jt.Pose, label: str, clear_z: float
    ) -> jt.Pose:
        """Re-grasp a block where it lies, lift it a few mm, move it onto its
        target, release. Returns the block's pose afterwards."""
        current = await self.read_pose(block)
        x, y, z = current.position
        tx, ty, _tz = target.position
        start_yaw = tool_yaw_deg(current.yaw_deg, self.grasp)
        # Turn the tool by the block's yaw error, taking the short way round.
        target_yaw = start_yaw + jt.wrap_deg(target.yaw_deg - current.yaw_deg, 180.0)
        grasp_z = z + self.grasp.grasp_z_offset_m
        lifted = grasp_z + self.grasp.adjust_lift_m
        await self.transit(f"{label}-approach", x, y, start_yaw, clear_z)
        await self.go(
            f"{label}-pre-grasp", x, y, grasp_z + PRE_GRASP_CLEARANCE_M, start_yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        await self.go(
            f"{label}-grasp", x, y, grasp_z, start_yaw,
            LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC,
        )
        # The gripper's firm-grasp flag can miss a held flat block, so carry
        # on regardless; the block's measured pose afterwards is the truth.
        await self.grab()
        await self.go(
            f"{label}-lift", x, y, lifted, start_yaw,
            LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC,
        )
        await self.go(
            f"{label}-shift", tx, ty, lifted, target_yaw,
            LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC,
        )
        await self.go(
            f"{label}-lower", tx, ty, grasp_z + self.grasp.release_clearance_m,
            target_yaw, LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC,
        )
        await self.open(f"{label}-release")
        await _wait_sim_duration(self.world, self.grasp.release_settle_sec)
        await self.go(
            f"{label}-clear", tx, ty, grasp_z + PRE_GRASP_CLEARANCE_M, target_yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        return await self.read_pose(block)


async def reset_tower(
    world: Any,
    tower: jt.TowerConfig,
    spec: jt.EpisodeSpec,
) -> dict[str, Any]:
    """Pause, place the tower and the supply block, park every other block,
    let physics settle, and check nothing moved. Returns the settled poses
    and whether the tower is stable."""
    await world.do_command({"command": "pause"})
    try:
        unused = [
            name
            for name in tower.block_names()
            if name not in spec.tower_poses
            and name != spec.next_block
            and name not in spec.scattered_poses
        ]
        if unused:
            await world.do_command({"command": "park_props", "props": unused})
        # tower_poses is in placement order; props are shuffled per seed.
        for placement, (name, pose) in enumerate(spec.tower_poses.items()):
            if name == spec.misplaced_block and spec.misplaced_pose is not None:
                pose = spec.misplaced_pose
            layer = jt.slot_ref(tower, placement).layer
            lifted = jt.Pose(
                (
                    pose.position[0],
                    pose.position[1],
                    pose.position[2] + RESET_CLEARANCE_PER_LAYER_M * (layer + 1),
                ),
                pose.yaw_deg,
            )
            await world.do_command(lifted.as_prop_command(name))
        await world.do_command(spec.supply_pose.as_prop_command(spec.next_block))
        for name, pose in spec.scattered_poses.items():
            await world.do_command(pose.as_prop_command(name))
    finally:
        await world.do_command({"command": "play"})
    await _wait_sim_duration(world, RESET_SETTLE_SEC)

    settled: dict[str, jt.Pose] = {}
    unstable: list[str] = []
    for name, commanded in spec.tower_poses.items():
        if name == spec.misplaced_block and spec.misplaced_pose is not None:
            commanded = spec.misplaced_pose
        reading = await world.do_command({"command": "get_prop_pose", "prop": name})
        actual = _pose_from_reading(reading)
        settled[name] = actual
        error = jt.pose_error(actual, commanded)
        if (
            error["xy_m"] > RESET_STABLE_XY_M
            or error["yaw_deg"] > RESET_STABLE_YAW_DEG
            or error["z_m"] > RESET_STABLE_Z_M
        ):
            unstable.append(name)
    return {"stable": not unstable, "unstable_blocks": unstable, "poses": settled}


async def prepare_supply_only(
    world: Any, spec: jt.EpisodeSpec
) -> None:
    """Chained builds keep the tower as it stands; only the next block moves."""
    await world.do_command(spec.supply_pose.as_prop_command(spec.next_block))
    await _wait_sim_duration(world, RESET_SETTLE_SEC / 2.0)


async def execute_place(
    world: Any,
    gripper: Any,
    motion: Any,
    arm: Any,
    spec: jt.EpisodeSpec,
    tower: jt.TowerConfig,
    *,
    names: MachineNames = MachineNames(),
    grasp: GraspConfig = GraspConfig(),
    noise: NoiseConfig = NoiseConfig(),
    tolerance: jt.PlacementTolerance = jt.PlacementTolerance(),
    episode_output_dir: str | None = None,
    episode_id: str | None = None,
    cosmos_output_dir: str | None = None,
    task: str = DEFAULT_PROMPT,
    reset: bool = True,
    home_joint_degrees: Sequence[float] = HOME_JOINT_DEGREES,
) -> dict[str, Any]:
    """Run one placement episode from `spec`. With `reset=False` (chained
    builds) the tower is left as it stands."""
    ep = _Episode(world, arm, gripper, motion, names, grasp)
    sim_clock = await _ElapsedClock.start(world, "episode")
    result: dict[str, Any] = {"spec": spec.summary(), "outcome": "incomplete"}

    await arm.move_to_joint_positions(JointPositions(values=list(home_joint_degrees)))
    await ep.open("open")

    if reset:
        with ep.timer.phase("tower_reset"):
            reset_report = await reset_tower(world, tower, spec)
        result["reset"] = {
            "stable": reset_report["stable"],
            "unstable_blocks": reset_report["unstable_blocks"],
        }
        if not reset_report["stable"]:
            result["outcome"] = "reset_unstable"
            return result
    else:
        await prepare_supply_only(world, spec)

    tower_names = list(spec.tower_poses)
    before = await ep.read_poses(tower_names)
    supply = await ep.read_pose(spec.next_block)
    # Empty-handed TCP height that clears the tower by the carry clearance,
    # with a block thickness of margin for the fingertips below the TCP.
    tower_top = jt.tower_top_z(tower, spec.placed_count)
    clear_z = tower_top + tower.block_thickness_m + grasp.carry_clearance_m
    # Holding a block: its bottom clears the tower by the carry clearance.
    carry_estimate_z = (
        tower_top + grasp.carry_clearance_m + tower.block_thickness_m / 2.0
        + grasp.grasp_z_offset_m
    )

    # Pre-capture setup: the deliberate error happens here, unrecorded.
    if spec.perturbed_start:
        start = await motion.get_pose(names.gripper, "world")
        rng = random.Random(f"place-start:{spec.seed}")
        moved = PoseInFrame(
            reference_frame=start.reference_frame,
            pose=ViamPose(
                x=start.pose.x + rng.gauss(0.0, noise.start_xy_stdev_m) * 1000.0,
                y=start.pose.y + rng.gauss(0.0, noise.start_xy_stdev_m) * 1000.0,
                z=start.pose.z + rng.gauss(0.0, noise.start_z_stdev_m) * 1000.0,
                o_x=start.pose.o_x, o_y=start.pose.o_y, o_z=start.pose.o_z,
                theta=start.pose.theta,
            ),
        )
        with ep.timer.phase("perturbed_start"):
            await _move(motion, names.gripper, moved, constraints=CENTER_MOVE_CONSTRAINTS)
    if spec.failed_grasp_aim_xy_m is not None:
        ax, ay = spec.failed_grasp_aim_xy_m
        hover_z = supply.position[2] + grasp.grasp_z_offset_m + grasp.lift_height_m
        aim_yaw = tool_yaw_deg(supply.yaw_deg, grasp)
        with ep.timer.phase("failed_grasp_start"):
            await ep.transit("failed-grasp", ax, ay, aim_yaw, clear_z)
            await ep.go(
                "failed-grasp-hover", ax, ay, hover_z, aim_yaw,
                LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
            )

    if episode_output_dir is not None:
        start_command: dict[str, Any] = {
            "command": "start_episode",
            "output_dir": episode_output_dir,
            "episode_id": episode_id or f"{spec.seed:06d}",
            "seed": spec.seed,
            "task": task,
            "arm": names.arm,
            "gripper": names.gripper,
            "fixed_camera": names.fixed_camera,
            "fixed_camera_2": names.fixed_camera_2,
            "wrist_camera": names.wrist_camera,
            "cosmos_output_dir": cosmos_output_dir,
            "target": {
                "prop_name": spec.next_block,
                "position_m": list(supply.position),
                "rotation_wxyz": list(supply.orientation_wxyz),
            },
        }
        with ep.timer.phase("capture_start"):
            await _start_episode_capture(world, start_command)
        ep.capturing = True

    capture = None
    adjustments: list[dict[str, Any]] = []
    try:
        if spec.misplaced_block is not None:
            await ep.mark(STAGE_STRAIGHTEN)
            target = spec.tower_poses[spec.misplaced_block]
            straightened = await ep.adjust(
                spec.misplaced_block, target, "straighten", clear_z
            )
            adjustments.append(
                {"block": spec.misplaced_block, **jt.pose_error(straightened, target)}
            )
            before[spec.misplaced_block] = straightened

        await ep.mark(STAGE_PICK)
        reported = await ep.pick(supply, "pick", clear_z, carry_estimate_z)
        tcp_x, tcp_y, tcp_z, pick_yaw = await ep.tcp()
        lifted = await ep.read_pose(spec.next_block)
        # The sim's own block pose decides whether the grasp worked; the
        # gripper's firm-grasp flag can miss a flat block it is holding.
        holding = lifted.position[2] - supply.position[2] >= MIN_LIFT_TO_COUNT_AS_HELD_M
        result["holding_after_lift"] = holding
        result["gripper_reported_holding"] = reported
        if not holding:
            result["outcome"] = "failure"
            result["failure"] = "grasp"
            return result

        slot = spec.next_slot_pose
        _slot_x, _slot_y, sz = slot.position
        # Measured after the lift, so any slip in the grasp is accounted for:
        # the TCP's height above the block centre, and where the block sits in
        # the hand. Release puts the block, not the TCP, on the slot.
        tcp_above_center = tcp_z - lifted.position[2]
        sx, sy, yaw, in_hand = release_target(
            slot, lifted, (tcp_x, tcp_y), pick_yaw
        )
        result["in_hand_offset"] = in_hand
        carry_z = max(
            tcp_z,
            tower_top + grasp.carry_clearance_m + tower.block_thickness_m / 2.0
            + tcp_above_center,
        )
        release_z = sz + grasp.release_clearance_m + tcp_above_center

        await ep.mark(STAGE_CARRY)
        await ep.transit("carry", sx, sy, yaw, carry_z)
        await ep.mark(STAGE_LOWER)
        pre_x, pre_y, pre_yaw = sx, sy, yaw
        if spec.perturbed_start:
            rng = random.Random(f"place-arrival:{spec.seed}")
            pre_x += rng.gauss(0.0, noise.arrival_xy_stdev_m)
            pre_y += rng.gauss(0.0, noise.arrival_xy_stdev_m)
            pre_yaw += rng.gauss(0.0, noise.arrival_yaw_stdev_deg)
        await ep.go(
            "pre-release", pre_x, pre_y, release_z + PRE_GRASP_CLEARANCE_M, pre_yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        await ep.go(
            "release", sx, sy, release_z, yaw,
            LINEAR_PICK_CONSTRAINTS, FINAL_DESCENT_MIN_DURATION_SEC,
        )
        await ep.open("release")
        await _wait_sim_duration(world, grasp.release_settle_sec)
        await ep.go(
            "release-clear", sx, sy, release_z + PRE_GRASP_CLEARANCE_M, yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )

        placed = await ep.read_pose(spec.next_block)
        error = jt.pose_error(placed, slot)
        attempts = 0
        while not jt.placement_within(error, tolerance) and attempts < grasp.max_adjust_attempts:
            await ep.mark(STAGE_ADJUST)
            placed = await ep.adjust(spec.next_block, slot, f"adjust-{attempts}", clear_z)
            error = jt.pose_error(placed, slot)
            adjustments.append({"block": spec.next_block, **error})
            attempts += 1

        await ep.mark(STAGE_RETREAT)
        rx, ry, _rz, ryaw = await ep.tcp()
        await ep.go(
            "retreat", rx, ry, max(carry_z, clear_z), ryaw,
            LINEAR_PICK_CONSTRAINTS, MIN_SUBTASK_DURATION_SEC,
        )

        after = await ep.read_poses(tower_names)
        moved = jt.disturbed_blocks(before, after, tolerance)
        success = jt.placement_within(error, tolerance) and not moved
        result.update(
            {
                "placement_error": error,
                "placed_pose": {"position_m": list(placed.position), "yaw_deg": placed.yaw_deg},
                "disturbed_blocks": moved,
                "adjustments": adjustments,
            }
        )
        result["outcome"] = "success" if success else "failure"
        if not success:
            result["failure"] = "disturbed" if moved else "placement"
        else:
            await ep.mark(STAGE_HOLD)
    finally:
        if ep.capturing:
            with ep.timer.phase("capture_stop"):
                capture = await _stop_episode_capture(
                    world, {"command": "stop_episode", "outcome": result["outcome"]}
                )
        timing = ep.timer.summary()
        timing["sim_elapsed_sec"] = (
            await sim_clock.elapsed(check_playing=False) if sim_clock.simulated else None
        )
        result["phases"] = list(ep.phases)
        result["capture"] = capture
        result["timing"] = timing
    return result


async def arm_to_world_xy(machine: Any, arm_name: str, x_m: float, y_m: float) -> tuple[float, float]:
    transformed = await machine.transform_pose(
        PoseInFrame(
            reference_frame=f"{arm_name}:base",
            pose=ViamPose(x=x_m * 1000.0, y=y_m * 1000.0, z=0.0, o_z=1.0),
        ),
        "world",
    )
    return transformed.pose.x / 1000.0, transformed.pose.y / 1000.0


def _xy(text: str) -> tuple[float, float]:
    parts = [float(v) for v in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y in metres")
    return parts[0], parts[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address", nargs="?", help="machine address from the Viam CONNECT tab")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--count", type=int, default=1, help="episodes (or chained towers)")
    parser.add_argument(
        "--set", dest="episode_set", choices=jt.EPISODE_SETS,
        help="force one episode set; default samples A-E by the plan's weights",
    )
    parser.add_argument("--placed", type=int, help="force the starting tower size")
    parser.add_argument(
        "--scatter", action="store_true",
        help="lay blocks not in the tower at random spots on the table instead of parking them",
    )
    parser.add_argument("--scatter-min-xy", type=_xy, default=jt.ScatterConfig.region_min_xy_m, help="scatter region corner X,Y (world, m)")
    parser.add_argument("--scatter-max-xy", type=_xy, default=jt.ScatterConfig.region_max_xy_m)
    parser.add_argument("--chain", action="store_true", help="set F: build whole towers without resets")
    parser.add_argument("--layers", type=int, default=jt.TowerConfig.layers)
    parser.add_argument("--tower-xy", type=_xy, default=jt.TowerConfig.base_xy_m, help="tower centre X,Y in the arm base frame (m)")
    parser.add_argument("--tower-yaw-deg", type=float, default=0.0, help="tower yaw in the world frame")
    parser.add_argument(
        "--block-scale", type=float, default=1.0,
        help="uniform scale for the Jenga blocks (must match the sim props), e.g. 0.416 so a Lite6 can grip one by its ends",
    )
    parser.add_argument(
        "--slot-gap-m", type=float, default=jt.TowerConfig.slot_gap_m,
        help="extra spacing between blocks in a layer, e.g. room for a side-grip finger",
    )
    parser.add_argument("--supply-min-xy", type=_xy, default=jt.SupplyConfig.region_min_xy_m, help="supply region corner X,Y in the arm base frame (m)")
    parser.add_argument("--supply-max-xy", type=_xy, default=jt.SupplyConfig.region_max_xy_m)
    parser.add_argument("--grasp-axis", choices=("end", "side"), default="end")
    parser.add_argument("--jaw-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--grasp-z-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--home-joints", type=lambda s: [float(v) for v in s.split(",")],
        default=list(HOME_JOINT_DEGREES),
        help="comma-separated home joint angles in degrees; the default is the Lite6 home",
    )
    parser.add_argument("--task", default=DEFAULT_PROMPT)
    parser.add_argument("--world", default=MachineNames.world)
    parser.add_argument("--arm", default=MachineNames.arm)
    parser.add_argument("--gripper", default=MachineNames.gripper)
    parser.add_argument("--motion", default=MachineNames.motion)
    parser.add_argument("--fixed-camera", default=MachineNames.fixed_camera)
    parser.add_argument("--fixed-camera-2", default=MachineNames.fixed_camera_2)
    parser.add_argument("--wrist-camera", default=MachineNames.wrist_camera)
    parser.add_argument("--viam-config", type=Path, default=Path("/etc/viam.json"))
    parser.add_argument("--disable-webrtc", action="store_true")
    parser.add_argument("--run-dir", type=Path, help="local directory for results.jsonl; also records episodes under RUN_DIR/episodes unless --episode-dir is set")
    parser.add_argument(
        "--episode-dir",
        help="record episodes into this directory on the sim machine (e.g. a data manager capture_dir); defaults to RUN_DIR/episodes",
    )
    parser.add_argument("--cosmos-output", action="store_true", help="also write CosmosWriter output next to the episode directory, under cosmoswriter/")
    parser.add_argument(
        "--park-z-m", type=float, default=0.0,
        help="with --print-props: height of the surface parked blocks rest on (e.g. a floor below the table)",
    )
    parser.add_argument(
        "--print-props", action="store_true",
        help="print the sim world `props` entries for the tower and exit",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    names = MachineNames(
        world=args.world, arm=args.arm, gripper=args.gripper, motion=args.motion,
        fixed_camera=args.fixed_camera, fixed_camera_2=args.fixed_camera_2,
        wrist_camera=args.wrist_camera,
    )
    grasp = GraspConfig(
        axis=args.grasp_axis,
        jaw_yaw_offset_deg=args.jaw_yaw_offset_deg,
        grasp_z_offset_m=args.grasp_z_offset_m,
    )
    results_file = None
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        results_file = (args.run_dir / "results.jsonl").open("a", encoding="utf-8")
    failures = 0
    try:
        async with await connect(
            args.address, viam_config_path=args.viam_config, disable_webrtc=args.disable_webrtc
        ) as machine:
            tower_xy = await arm_to_world_xy(machine, names.arm, *args.tower_xy)
            supply_corners = [
                await arm_to_world_xy(machine, names.arm, x, y)
                for x in (args.supply_min_xy[0], args.supply_max_xy[0])
                for y in (args.supply_min_xy[1], args.supply_max_xy[1])
            ]
            xs, ys = zip(*supply_corners)
            tower = jt.TowerConfig(
                base_xy_m=tower_xy, base_yaw_deg=args.tower_yaw_deg, layers=args.layers,
                slot_gap_m=args.slot_gap_m, **jt.scaled_block_dims(args.block_scale),
            )
            supply = jt.SupplyConfig(
                region_min_xy_m=(min(xs), min(ys)), region_max_xy_m=(max(xs), max(ys))
            )
            scatter = (
                jt.ScatterConfig(region_min_xy_m=args.scatter_min_xy, region_max_xy_m=args.scatter_max_xy)
                if args.scatter else None
            )
            world = Generic.from_robot(machine, names.world)
            gripper = Gripper.from_robot(machine, names.gripper)
            motion = MotionClient.from_robot(machine, names.motion)
            arm = Arm.from_robot(machine, names.arm)

            # Each group runs in order and stops at its first failure: a single
            # episode, or one chained tower (set F) built without resets.
            groups: list[list[tuple[jt.EpisodeSpec, str, bool]]] = []
            for seed in range(args.seed, args.seed + args.count):
                if args.chain:
                    first = args.placed or 0
                    chain = []
                    for k in range(first, tower.block_count):
                        spec = jt.build_episode_spec(
                            seed, tower, supply, episode_set="F", placed_count=k
                        )
                        # Each chained step gets its own supply pose.
                        spec = replace(
                            spec,
                            supply_pose=jt.sample_supply_pose(
                                tower, supply, seed, f"supply-chain-{k}"
                            ),
                        )
                        chain.append((spec, f"chain{seed:06d}-{k:02d}", k == first))
                    groups.append(chain)
                else:
                    spec = jt.build_episode_spec(
                        seed, tower, supply,
                        episode_set=args.episode_set, placed_count=args.placed,
                        scatter=scatter,
                    )
                    groups.append([(spec, f"{seed:06d}", True)])

            jobs = [job for group in groups for job in group]
            episode_dir = args.episode_dir or (
                str(args.run_dir / "episodes") if args.run_dir else None
            )
            skip_until_reset = False
            for spec, episode_id, reset in jobs:
                if skip_until_reset and not reset:
                    continue
                skip_until_reset = False
                result = await execute_place(
                    world, gripper, motion, arm, spec, tower,
                    names=names, grasp=grasp, task=args.task, reset=reset,
                    home_joint_degrees=args.home_joints,
                    episode_output_dir=episode_dir,
                    cosmos_output_dir=(
                        str(Path(episode_dir).parent / "cosmoswriter" / episode_id)
                        if episode_dir and args.cosmos_output else None
                    ),
                    episode_id=episode_id,
                )
                result["episode_id"] = episode_id
                line = json.dumps(result, sort_keys=True, default=str)
                print(line, flush=True)
                if results_file is not None:
                    results_file.write(line + "\n")
                    results_file.flush()
                if result["outcome"] == "reset_unstable":
                    # Not the oracle's failure: the starting tower didn't stand.
                    skip_until_reset = args.chain
                elif result["outcome"] != "success":
                    failures += 1
                    # A chained tower can't continue from a failed step; move
                    # on to the next tower, which starts with a reset.
                    skip_until_reset = args.chain
    finally:
        if results_file is not None:
            results_file.close()
    return 0 if failures == 0 else 1


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.print_props:
        tower = jt.TowerConfig(
            layers=args.layers, slot_gap_m=args.slot_gap_m, **jt.scaled_block_dims(args.block_scale)
        )
        print(json.dumps(jt.tower_props(tower, park_z_m=args.park_z_m), indent=2))
        return 0
    if not args.address:
        parser.error("address is required unless --print-props is given")
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.chain and args.episode_set not in (None, "F"):
        parser.error("--chain always runs set F")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
