"""
Run scripted Jenga placement episodes through normal Viam APIs.

One episode places one block: reset the tower and scatter the loose blocks on
the table, pick the loose block nearest the tower whose ends are clear, carry
it above its slot, lower and release it, adjust it if it landed off target,
and retreat. Like `oracle_episode.py`, this is an oracle: it
plans from exact sim poses.

Training data cannot mask frames, so every deliberate error happens before
capture starts and only the recovery is recorded:
  C  the arm starts from a perturbed pose,
  D  the last tower block starts misplaced; the episode straightens it first,
  E  the arm starts hovering empty, as after a missed grasp, beside the block.
Set F (`--chain`) builds a whole tower without resets, keeping the oracle's own
small errors; loose blocks stay where they lie between steps.

`--loose-blocks N` lays only N blocks on the table (`--loose-blocks 1`: one
block to pick, no distractors) and parks the rest off the table. Chained
builds lay a fresh block from the parked ones before each step.

Each episode is recorded with one prompt for the whole placement (`--task`)
and a prompt per stage (pick, carry, lower, ...) through `mark_episode_stage`.
A stage needs at least one 33-frame action chunk to yield a training sample;
the sim module holds a shorter stage open (the arm waits) until it has one,
so the carry and retreat moves alone last that long and the arm rarely waits.

About half the episodes (`--nudge-fraction`) are nudged with the pick recipe
that trained our best policy (dn-best-1): the sim pushes the arm off its
commanded joints and records the clean targets. The gripper is only ever
opened or closed.

The tower blocks must exist as props in the sim world config; print them with
`--print-props`.

Every episode ends the same way, whatever its outcome: the arm lifts about
10 cm above the new tower top and holds still with the gripper open for 3 s.
The training loader never starts a sample in an episode's last 32 frames
(~2.1 s at 15 Hz), so a shorter final hold would never teach the policy to
stop; and the end pose differs from the home start, so the last frame never
looks like the next episode's first.

Defaults match our UFACTORY 850 + Gripper G2 sims: components `arm`,
`gripper`, `overview-cam`, `front-cam`, `wrist-cam`, and armfarm01's saved
home pose. The sim config should label the embodiment for the episode
manifest and policy checks: arm `arm_model: "xarm850"`, gripper
`gripper_model: "xarm_gripper_g2"`.
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
    from .batch_episodes import _nudge_deg
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
    from batch_episodes import _nudge_deg
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
# Embodiment names for the UFACTORY 850 + Gripper G2, shared with the dataset
# export and the real controller.
ARM_MODEL = "xarm850"
GRIPPER_MODEL = "xarm_gripper_g2"
# armfarm01's saved home pose (its `home-pose` component), in degrees. The
# oracle's default home is the Lite6's, whose joint 3 (+90 deg) is outside the
# 850's range.
UF850_HOME_JOINT_DEGREES = (0.0, 22.57, -50.69, 0.04, -74.04, -0.04)
# Stage prompts, written into the episode with mark_episode_stage and listed
# in the result record.
STAGE_STRAIGHTEN = "straighten the top block of the tower"
STAGE_PICK = "pick up the next block"
STAGE_CARRY = "carry the block above its slot on the tower"
STAGE_LOWER = "lower the block into its slot and open the gripper"
STAGE_ADJUST = "adjust the block into its slot"
STAGE_RETREAT = "lift the gripper away from the tower"
STAGE_HOLD = "hold still"
# The nudges of the pick recipe that trained dn-best-1 (cosmos-spike
# sim-tools/record_all.sh), used in about half the episodes.
DN_BEST_NUDGE_DEG = (2.5, 3.3, 2.9, 1.5, 3.2, 5.0)
DN_BEST_NUDGE_INTERVAL_SEC = 1.2
DN_BEST_NUDGE_DURATION_SEC = 0.6
DN_BEST_NUDGE_FRACTION = 0.5
# Nudged episodes are nudged only while these stages run: the arm is away from
# the tower. Every stage that touches the tower (straighten, lower, adjust,
# retreat, hold) is recorded un-nudged, so a nudge can't knock the tower over
# and the end hold is truly still. Needs the sim module's set_nudge_paused.
NUDGED_STAGES = frozenset({STAGE_PICK, STAGE_CARRY})
# Extra wait after a stopped nudge's reported fade-out: one 15 Hz frame.
NUDGE_FADE_MARGIN_SEC = 1.0 / 15.0

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
    X; calibrate it once per gripper asset. The Gripper G2 needs 90: at tool
    yaw 0 its jaws close along world Y (checked with the wrist camera on our
    850 + G2 sim, Nov 19)."""

    axis: str = "end"
    jaw_yaw_offset_deg: float = 90.0
    # TCP height relative to the block centre while grasping. Keep the
    # fingertips inside the block's thickness so they clear the layer below.
    # The G2's TCP (172 mm) lies 4.9 mm past its fingertips at a 75 mm grip,
    # so 9.4 mm below the centre puts the fingertips 3 mm above the block's
    # bottom: the grip our 850 + G2 sim lifted a block with (Nov 19).
    grasp_z_offset_m: float = -0.0094
    approach_clearance_m: float = 0.10
    carry_clearance_m: float = 0.08
    lift_height_m: float = 0.08
    release_clearance_m: float = 0.002
    release_settle_sec: float = 0.3
    # Before the final lower: hover this far above the release height, let
    # the arm settle, then re-aim from where the held block actually is.
    reaim_height_m: float = 0.02
    reaim_settle_sec: float = 0.5
    # End of every episode: lift this far above the new tower top, then hold.
    end_lift_m: float = 0.10
    end_hold_sec: float = 3.0
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
    arm: str = "arm"
    gripper: str = "gripper"
    motion: str = "builtin"
    fixed_camera: str = "overview-cam"
    fixed_camera_2: str = "front-cam"
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
    and carried rigidly with the tool to the slot. A block looks the same
    turned 180 deg, so the tool turns whichever way is shorter: at most 90
    deg from `pick_tool_yaw_deg`."""
    off_x = lifted.position[0] - tcp_xy[0]
    off_y = lifted.position[1] - tcp_xy[1]
    # Offset in the tool frame at pick time.
    c, s = math.cos(math.radians(-pick_tool_yaw_deg)), math.sin(math.radians(-pick_tool_yaw_deg))
    tool_x, tool_y = c * off_x - s * off_y, s * off_x + c * off_y
    block_in_tool_deg = jt.wrap_deg(lifted.yaw_deg - pick_tool_yaw_deg, 180.0)
    place_yaw = jt.wrap_deg(slot.yaw_deg - block_in_tool_deg, 180.0)
    place_yaw = pick_tool_yaw_deg + jt.wrap_deg(place_yaw - pick_tool_yaw_deg, 180.0)
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
        self.nudged = False
        # Stages a nudged episode is nudged in; layer mode sets its own.
        self.nudged_stages: frozenset[str] = NUDGED_STAGES
        self.phases: list[str] = []
        # The stage being recorded, and every stage with its first frame.
        self.stage: str | None = None
        self.stages: list[dict[str, Any]] = []

    async def read_pose(self, prop: str) -> jt.Pose:
        reading = await self.world.do_command({"command": "get_prop_pose", "prop": prop})
        return _pose_from_reading(reading)

    async def read_poses(self, props: Sequence[str]) -> dict[str, jt.Pose]:
        return {name: await self.read_pose(name) for name in props}

    async def begin_capture(self, command: Mapping[str, Any], stage: str) -> None:
        """Start capture with `stage` as its first stage, at frame 0."""
        await _start_episode_capture(self.world, command, stage)
        self.capturing = True
        self.stage = stage
        self.stages.append({"task": stage, "start_frame": 0})
        if stage not in self.phases:
            self.phases.append(stage)
        await self.set_nudges(stage)

    async def set_nudges(self, stage: str) -> None:
        """Nudge only in `nudged_stages`. A nudge under way when they stop
        fades out first; wait for it so the next stage starts un-nudged
        (callers pause before marking that stage)."""
        if not (self.capturing and self.nudged):
            return
        paused = stage not in self.nudged_stages
        reply = await self.world.do_command({"command": "set_nudge_paused", "paused": paused})
        remaining = float(reply.get("remaining_s") or 0.0)
        if paused and remaining > 0.0:
            await _wait_sim_duration(self.world, remaining + NUDGE_FADE_MARGIN_SEC)

    async def mark(self, stage: str) -> None:
        """Begin a stage: list it in the result and, while capturing, write it
        into the episode. The sim module holds the previous stage open until
        it has one action chunk of frames."""
        if stage not in self.phases:
            self.phases.append(stage)
        if not self.capturing or stage == self.stage:
            return
        # Stop nudging before the stage begins, so a nudge's fade-out is
        # recorded in the stage it started in; resume once it has begun.
        nudged = stage in self.nudged_stages
        if not nudged:
            await self.set_nudges(stage)
        reply = await self.world.do_command({"command": "mark_episode_stage", "task": stage})
        self.stage = stage
        self.stages.append({"task": stage, "start_frame": reply.get("start_frame")})
        if nudged:
            await self.set_nudges(stage)

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
        self,
        label: str,
        x: float,
        y: float,
        yaw: float,
        clear_z: float,
        cross_min_sec: float = APPROACH_WAYPOINT_MIN_DURATION_SEC,
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
            f"{label}-cross", x, y, height, yaw, LINEAR_PICK_CONSTRAINTS, cross_min_sec
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

    async def finish(self, end_top_z: float) -> None:
        """End every episode the same way: lift straight up to `end_lift_m`
        above `end_top_z` with the gripper open, then hold still."""
        await self.mark(STAGE_RETREAT)
        x, y, z, yaw = await self.tcp()
        await self.go(
            "retreat", x, y, max(z, end_top_z + self.grasp.end_lift_m), yaw,
            LINEAR_PICK_CONSTRAINTS, MIN_SUBTASK_DURATION_SEC,
        )
        await self.mark(STAGE_HOLD)
        with self.timer.phase("end_hold"):
            await _wait_sim_duration(self.world, self.grasp.end_hold_sec)


async def reset_tower(
    world: Any,
    tower: jt.TowerConfig,
    spec: jt.EpisodeSpec,
) -> dict[str, Any]:
    """Pause, place the tower, lay the loose blocks on the table, park any
    other block, let physics settle, and check the tower stood. Returns the
    settled poses and whether the tower is stable."""
    table = table_poses(spec)
    await world.do_command({"command": "pause"})
    try:
        unused = [
            name
            for name in tower.block_names()
            if name not in spec.tower_poses and name not in table
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
        for name, pose in table.items():
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


def table_poses(spec: Any) -> dict[str, jt.Pose]:
    """Blocks the reset lays on the table: the loose blocks, and in layer mode
    the layer on the pad."""
    return {**spec.loose_poses, **getattr(spec, "pad_poses", {})}


async def lay_props(world: Any, poses: Mapping[str, jt.Pose]) -> None:
    """Chained builds: lay these blocks on the table and leave the tower and
    the other loose blocks as they stand."""
    await world.do_command({"command": "pause"})
    try:
        for name, pose in poses.items():
            await world.do_command(pose.as_prop_command(name))
    finally:
        await world.do_command({"command": "play"})
    await _wait_sim_duration(world, RESET_SETTLE_SEC)


async def read_loose_poses(world: Any, names: Sequence[str]) -> dict[str, jt.Pose]:
    """Where the loose blocks lie now, for the next step of a chained build."""
    poses = {}
    for name in names:
        reading = await world.do_command({"command": "get_prop_pose", "prop": name})
        poses[name] = _pose_from_reading(reading)
    return poses


def episode_nudge(
    key: str,
    fraction: float,
    amplitude_deg: float | Sequence[float],
    interval_sec: float,
    duration_sec: float,
) -> dict[str, Any] | None:
    """The start_episode nudge for one episode, or None: a fixed share of
    episodes, chosen by `key` (seed and step) so reruns nudge the same ones."""
    amplitudes = amplitude_deg if isinstance(amplitude_deg, Sequence) else [amplitude_deg]
    if not max(amplitudes) or random.Random(f"nudge:{key}").random() >= fraction:
        return None
    return {
        "amplitude_deg": (
            list(amplitude_deg) if isinstance(amplitude_deg, Sequence) else amplitude_deg
        ),
        "interval_s": interval_sec,
        "duration_s": duration_sec,
    }


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
    home_joint_degrees: Sequence[float] = UF850_HOME_JOINT_DEGREES,
    nudge: Mapping[str, Any] | None = None,
    upload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one placement episode from `spec`. With `reset=False` (chained
    builds) the tower and the loose blocks are left as they lie. `nudge` and
    `upload` go to start_episode and stop_episode as the pick oracle sends
    them."""
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
    elif spec.fresh_poses:
        with ep.timer.phase("lay_blocks"):
            await lay_props(world, spec.fresh_poses)

    tower_names = list(spec.tower_poses)
    before = await ep.read_poses(tower_names)
    block = await ep.read_pose(spec.next_block)
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
        await perturb_start(ep, spec.seed, noise)
    if spec.failed_grasp_aim_xy_m is not None:
        ax, ay = spec.failed_grasp_aim_xy_m
        hover_z = block.position[2] + grasp.grasp_z_offset_m + grasp.lift_height_m
        aim_yaw = tool_yaw_deg(block.yaw_deg, grasp)
        with ep.timer.phase("failed_grasp_start"):
            await ep.transit("failed-grasp", ax, ay, aim_yaw, clear_z)
            await ep.go(
                "failed-grasp-hover", ax, ay, hover_z, aim_yaw,
                LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
            )

    if episode_output_dir is not None:
        first_stage = STAGE_STRAIGHTEN if spec.misplaced_block is not None else STAGE_PICK
        await begin_episode_capture(
            ep, first_stage, spec.seed, spec.next_block, block,
            episode_output_dir=episode_output_dir, episode_id=episode_id,
            cosmos_output_dir=cosmos_output_dir, task=task, nudge=nudge,
        )
    result["nudge"] = dict(nudge) if nudge is not None and ep.capturing else None

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
        reported = await ep.pick(block, "pick", clear_z, carry_estimate_z)
        tcp_x, tcp_y, tcp_z, pick_yaw = await ep.tcp()
        lifted = await ep.read_pose(spec.next_block)
        # The sim's own block pose decides whether the grasp worked; the
        # gripper's firm-grasp flag can miss a flat block it is holding.
        holding = lifted.position[2] - block.position[2] >= MIN_LIFT_TO_COUNT_AS_HELD_M
        result["holding_after_lift"] = holding
        result["gripper_reported_holding"] = reported
        if not holding:
            result["outcome"] = "failure"
            result["failure"] = "grasp"
            await ep.open("open-after-miss")
            await ep.finish(jt.tower_top_z(tower, spec.placed_count))
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
        await ep.transit(
            "carry", sx, sy, yaw, carry_z, cross_min_sec=MIN_SUBTASK_DURATION_SEC
        )
        await ep.mark(STAGE_LOWER)
        pre_x, pre_y, pre_yaw = sx, sy, yaw
        if spec.perturbed_start:
            rng = random.Random(f"place-arrival:{spec.seed}")
            pre_x += rng.gauss(0.0, noise.arrival_xy_stdev_m)
            pre_y += rng.gauss(0.0, noise.arrival_xy_stdev_m)
            pre_yaw += rng.gauss(0.0, noise.arrival_yaw_stdev_deg)
        hover_z = release_z + grasp.reaim_height_m
        await ep.go(
            "pre-release", pre_x, pre_y, hover_z, pre_yaw,
            LINEAR_PICK_CONSTRAINTS, APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        # Settle, then re-aim from where the block actually hangs: the arm's
        # last few mm of settling would otherwise drag the layer below.
        await _wait_sim_duration(world, grasp.reaim_settle_sec)
        hx, hy, _hz, hyaw = await ep.tcp()
        hanging = await ep.read_pose(spec.next_block)
        sx, sy, yaw, _aim = release_target(slot, hanging, (hx, hy), hyaw)
        await ep.go(
            "re-aim", sx, sy, hover_z, yaw,
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

        await ep.finish(jt.tower_top_z(tower, spec.placed_count + 1))

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
    finally:
        await end_episode_capture(ep, result, sim_clock, upload)
    return result


async def perturb_start(ep: _Episode, seed: int, noise: NoiseConfig) -> None:
    """Set C's deliberate start error, before capture: move the TCP off home
    by the noise config's spread. Shared by block and layer mode."""
    start = await ep.motion.get_pose(ep.names.gripper, "world")
    rng = random.Random(f"place-start:{seed}")
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
        await _move(ep.motion, ep.names.gripper, moved, constraints=CENTER_MOVE_CONSTRAINTS)


async def begin_episode_capture(
    ep: _Episode,
    first_stage: str,
    seed: int,
    target_prop: str,
    target_pose: jt.Pose,
    *,
    episode_output_dir: str,
    episode_id: str | None,
    cosmos_output_dir: str | None,
    task: str,
    nudge: Mapping[str, Any] | None,
) -> None:
    """Open the episode's capture with `first_stage` at frame 0. Shared by
    block and layer mode."""
    start_command: dict[str, Any] = {
        "command": "start_episode",
        "output_dir": episode_output_dir,
        "episode_id": episode_id or f"{seed:06d}",
        "seed": seed,
        "task": task,
        "arm": ep.names.arm,
        "gripper": ep.names.gripper,
        "fixed_camera": ep.names.fixed_camera,
        "fixed_camera_2": ep.names.fixed_camera_2,
        "wrist_camera": ep.names.wrist_camera,
        "cosmos_output_dir": cosmos_output_dir,
        "target": {
            "prop_name": target_prop,
            "position_m": list(target_pose.position),
            "rotation_wxyz": list(target_pose.orientation_wxyz),
        },
    }
    if nudge is not None:
        start_command["nudge"] = dict(nudge)
        ep.nudged = True
    with ep.timer.phase("capture_start"):
        await ep.begin_capture(start_command, first_stage)


async def end_episode_capture(
    ep: _Episode,
    result: dict[str, Any],
    sim_clock: Any,
    upload: Mapping[str, Any] | None,
) -> None:
    """Close any open capture with the result's outcome and fill in the
    result's phases, stages, capture and timing. Shared by block and layer
    mode; runs even when the episode raised."""
    capture = None
    if ep.capturing:
        stop_command: dict[str, Any] = {
            "command": "stop_episode", "outcome": result["outcome"],
        }
        if upload is not None:
            stop_command["upload"] = dict(upload)
        with ep.timer.phase("capture_stop"):
            capture = await _stop_episode_capture(ep.world, stop_command)
    timing = ep.timer.summary()
    timing["sim_elapsed_sec"] = (
        await sim_clock.elapsed(check_playing=False) if sim_clock.simulated else None
    )
    result["phases"] = list(ep.phases)
    result["stages"] = list(ep.stages)
    result["capture"] = capture
    result["timing"] = timing


async def close_open_episode(world: Any) -> None:
    """Best-effort stop of a capture an interrupted episode left running, so
    it can't block every later episode."""
    try:
        await asyncio.wait_for(
            world.do_command({"command": "stop_episode", "outcome": "incomplete"}),
            timeout=60,
        )
    except Exception:
        # Usually the episode already closed its capture; nothing left to do.
        pass


async def run_episode_safely(
    world: Any,
    gripper: Any,
    motion: Any,
    arm: Any,
    spec: jt.EpisodeSpec,
    tower: jt.TowerConfig,
    *,
    timeout_sec: float = 0.0,
    execute: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """`execute_place`, but an exception or a timeout ends only this episode:
    it is recorded as outcome "error" and the batch carries on, as
    batch_episodes.py does for picks."""
    try:
        run = (execute or execute_place)(world, gripper, motion, arm, spec, tower, **kwargs)
        if timeout_sec > 0:
            return await asyncio.wait_for(run, timeout=timeout_sec)
        return await run
    except Exception as exc:
        await close_open_episode(world)
        return {
            "spec": spec.summary(),
            "outcome": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


async def arm_to_world_xy(machine: Any, arm_name: str, x_m: float, y_m: float) -> tuple[float, float]:
    """(x, y) in the arm's base frame to world. RDK names a component's static
    origin frame `<name>_origin`; its model's link frames are `<name>:<link>`,
    and the 850's base link is not called `base`."""
    transformed = await machine.transform_pose(
        PoseInFrame(
            reference_frame=f"{arm_name}_origin",
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
    parser.add_argument(
        "--build", choices=("block", "layer"), default="block",
        help="block: place one block per episode; layer: the reset lays the next "
        "layer on the pad and the episode places all of it with a side grip",
    )
    parser.add_argument(
        "--placed", type=int,
        help="force the starting tower size: blocks (--build block) or whole layers (--build layer)",
    )
    parser.add_argument(
        "--pad-xy", type=_xy, default=jt.PadConfig.center_xy_m,
        help="--build layer: where the reset lays the next layer, X,Y in the arm base frame (m)",
    )
    parser.add_argument(
        "--scatter-min-xy", type=_xy, default=jt.ScatterConfig.region_min_xy_m,
        help="corner X,Y of the table region the loose blocks lie in, in the arm base frame (m)",
    )
    parser.add_argument("--scatter-max-xy", type=_xy, default=jt.ScatterConfig.region_max_xy_m)
    parser.add_argument(
        "--loose-blocks", type=int,
        help="--build block: lay only this many blocks on the table (1: just the block "
        "to pick) and park the rest; default every block not in the tower",
    )
    parser.add_argument("--chain", action="store_true", help="set F: build whole towers without resets")
    parser.add_argument("--layers", type=int, default=jt.TowerConfig.layers)
    parser.add_argument("--tower-xy", type=_xy, default=jt.TowerConfig.base_xy_m, help="tower centre X,Y in the arm base frame (m)")
    parser.add_argument("--tower-yaw-deg", type=float, default=0.0, help="tower yaw in the world frame")
    parser.add_argument(
        "--support-z-m", type=float, default=jt.TowerConfig.support_z_m,
        help="world height of the table top the tower and loose blocks rest on (m), e.g. -0.0127 for armfarm01",
    )
    parser.add_argument(
        "--block-scale", type=float, default=1.0,
        help="uniform scale for the Jenga blocks (must match the sim props), e.g. 0.416 so a Lite6 can grip one by its ends",
    )
    parser.add_argument(
        "--slot-gap-m", type=float, default=jt.TowerConfig.slot_gap_m,
        help="extra spacing between blocks in a layer, e.g. room for a side-grip finger",
    )
    parser.add_argument("--grasp-axis", choices=("end", "side"), default="end")
    parser.add_argument(
        "--jaw-yaw-offset-deg", type=float, default=GraspConfig.jaw_yaw_offset_deg,
        help="tool yaw at which the jaws close along world X (G2: 90)",
    )
    parser.add_argument(
        "--grasp-z-offset-m", type=float, default=GraspConfig.grasp_z_offset_m,
        help="TCP height relative to the block centre while grasping (G2: -0.0094)",
    )
    parser.add_argument(
        "--home-joints", type=lambda s: [float(v) for v in s.split(",")],
        default=list(UF850_HOME_JOINT_DEGREES),
        help="comma-separated home joint angles in degrees; the default is armfarm01's 850 home "
        f"(the Lite6 home is {','.join(f'{v:g}' for v in HOME_JOINT_DEGREES)})",
    )
    parser.add_argument(
        "--episode-timeout-sec", type=float, default=0.0,
        help="end an episode that runs longer than this (0: no limit); it is recorded as an error",
    )
    parser.add_argument(
        "--nudge-fraction", type=float, default=DN_BEST_NUDGE_FRACTION,
        help="share of episodes that are nudged (0: none), chosen per seed",
    )
    parser.add_argument(
        "--nudge-deg", type=_nudge_deg, default=list(DN_BEST_NUDGE_DEG),
        help="nudge amplitude per joint in degrees: one value or six comma-separated "
        "(default: the dn-best-1 pick recipe)",
    )
    parser.add_argument("--nudge-interval-sec", type=float, default=DN_BEST_NUDGE_INTERVAL_SEC)
    parser.add_argument("--nudge-duration-sec", type=float, default=DN_BEST_NUDGE_DURATION_SEC)
    parser.add_argument(
        "--upload-dataset-id",
        help="have the machine upload each recorded episode to this Viam dataset",
    )
    parser.add_argument(
        "--upload-tag", action="append", default=[],
        help="extra tag for uploaded episodes; may be repeated",
    )
    parser.add_argument("--app-url", help="Viam app host the machine uploads to (default: app.viam.com)")
    parser.add_argument(
        "--task", default=None,
        help=f"episode prompt; default {DEFAULT_PROMPT!r}, or for --build layer the layer prompt",
    )
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
    if args.loose_blocks is not None and args.build != "block":
        raise SystemExit("--loose-blocks is for --build block only")
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
            scatter_corners = [
                await arm_to_world_xy(machine, names.arm, x, y)
                for x in (args.scatter_min_xy[0], args.scatter_max_xy[0])
                for y in (args.scatter_min_xy[1], args.scatter_max_xy[1])
            ]
            xs, ys = zip(*scatter_corners)
            tower = jt.TowerConfig(
                base_xy_m=tower_xy, base_yaw_deg=args.tower_yaw_deg, layers=args.layers,
                support_z_m=args.support_z_m,
                arm_xy_m=await arm_to_world_xy(machine, names.arm, 0.0, 0.0),
                slot_gap_m=args.slot_gap_m, **jt.scaled_block_dims(args.block_scale),
            )
            scatter = jt.ScatterConfig(
                region_min_xy_m=(min(xs), min(ys)), region_max_xy_m=(max(xs), max(ys))
            )
            layer_mode = args.build == "layer"
            if layer_mode:
                if __package__:
                    from . import layer_episode as le
                else:
                    import layer_episode as le
                pad = jt.PadConfig(
                    center_xy_m=await arm_to_world_xy(machine, names.arm, *args.pad_xy)
                )
                tower = jt.layer_mode_tower(tower, pad)
                execute, task = le.execute_place_layer, args.task or le.LAYER_PROMPT
            else:
                execute, task = execute_place, args.task or DEFAULT_PROMPT
            world = Generic.from_robot(machine, names.world)
            gripper = Gripper.from_robot(machine, names.gripper)
            motion = MotionClient.from_robot(machine, names.motion)
            arm = Arm.from_robot(machine, names.arm)
            episode_dir = args.episode_dir or (
                str(args.run_dir / "episodes") if args.run_dir else None
            )
            upload = None
            if args.upload_dataset_id:
                tags = ["isaac-episode", "jenga-place"]
                if args.run_dir is not None:
                    tags.insert(0, args.run_dir.name)
                upload = {"dataset_id": args.upload_dataset_id, "tags": [*tags, *args.upload_tag]}
                if args.app_url:
                    upload["app_url"] = args.app_url

            def emit(record: dict[str, Any]) -> None:
                line = json.dumps(record, sort_keys=True, default=str)
                print(line, flush=True)
                if results_file is not None:
                    results_file.write(line + "\n")
                    results_file.flush()

            async def run(spec: Any, episode_id: str, reset: bool) -> dict[str, Any]:
                result = await run_episode_safely(
                    world, gripper, motion, arm, spec, tower,
                    timeout_sec=args.episode_timeout_sec, execute=execute,
                    names=names, grasp=grasp, task=task, reset=reset,
                    home_joint_degrees=args.home_joints,
                    episode_output_dir=episode_dir,
                    cosmos_output_dir=(
                        str(Path(episode_dir).parent / "cosmoswriter" / episode_id)
                        if episode_dir and args.cosmos_output else None
                    ),
                    episode_id=episode_id,
                    nudge=episode_nudge(
                        episode_id, args.nudge_fraction, args.nudge_deg,
                        args.nudge_interval_sec, args.nudge_duration_sec,
                    ),
                    upload=upload,
                )
                result["episode_id"] = episode_id
                result["build"] = args.build
                result["embodiment"] = {"arm_model": ARM_MODEL, "gripper_model": GRIPPER_MODEL}
                emit(result)
                return result

            for seed in range(args.seed, args.seed + args.count):
                if layer_mode:
                    failures += await _run_layers(args, seed, tower, pad, scatter, world, run, emit)
                    continue
                if not args.chain:
                    spec = jt.build_episode_spec(
                        seed, tower, scatter,
                        episode_set=args.episode_set, placed_count=args.placed,
                        loose_blocks=args.loose_blocks,
                    )
                    result = await run(spec, f"{seed:06d}", True)
                    # A tower that didn't stand at reset is not the oracle's failure.
                    if result["outcome"] not in ("success", "reset_unstable"):
                        failures += 1
                    continue
                # Set F: one tower, built step by step without resets. Each
                # step picks from where the loose blocks lie after the last.
                spec = jt.build_episode_spec(
                    seed, tower, scatter, episode_set="F", placed_count=args.placed or 0,
                    loose_blocks=args.loose_blocks,
                )
                built = list(spec.tower_poses)
                reset = True
                while True:
                    k = spec.placed_count
                    result = await run(spec, f"chain{seed:06d}-{k:02d}", reset)
                    if result["outcome"] != "success":
                        # A chained tower can't continue from a failed step.
                        if result["outcome"] != "reset_unstable":
                            failures += 1
                        break
                    built.append(spec.next_block)
                    if len(built) == tower.block_count:
                        break
                    loose_names = [n for n in spec.loose_poses if n != spec.next_block]
                    loose = await read_loose_poses(world, loose_names)
                    next_spec = jt.chain_step_spec(
                        seed, tower, scatter, built, loose, args.loose_blocks
                    )
                    if next_spec is None:
                        failures += 1
                        emit({
                            "episode_id": f"chain{seed:06d}-{len(built):02d}",
                            "outcome": "failure", "failure": "no_clear_block",
                        })
                        break
                    spec, reset = next_spec, False
    finally:
        if results_file is not None:
            results_file.close()
    return 0 if failures == 0 else 1


async def _run_layers(
    args: argparse.Namespace,
    seed: int,
    tower: jt.TowerConfig,
    pad: jt.PadConfig,
    scatter: jt.ScatterConfig,
    world: Any,
    run: Any,
    emit: Any,
) -> int:
    """One layer-mode episode, or with --chain one tower built layer by layer
    (set F) without resets: the pad is laid afresh before each step. Returns
    the failures to count."""
    if not args.chain:
        spec = jt.build_layer_spec(
            seed, tower, pad, scatter,
            episode_set=args.episode_set, placed_layers=args.placed,
        )
        result = await run(spec, f"layer{seed:06d}", True)
        return 0 if result["outcome"] in ("success", "reset_unstable") else 1
    spec = jt.build_layer_spec(
        seed, tower, pad, scatter, episode_set="A", placed_layers=args.placed or 0
    )
    spec = replace(spec, episode_set="F")
    built = list(spec.tower_poses)
    reset = True
    while True:
        layer = spec.placed_layers
        result = await run(spec, f"layerchain{seed:06d}-{layer:02d}", reset)
        if result["outcome"] != "success":
            return 0 if result["outcome"] == "reset_unstable" else 1
        built.extend(spec.pad_poses)
        if len(built) == tower.block_count:
            return 0
        loose = await read_loose_poses(world, list(spec.loose_poses))
        next_spec = jt.chain_layer_spec(seed, tower, pad, built, loose)
        if next_spec is None:
            emit({
                "episode_id": f"layerchain{seed:06d}-{layer + 1:02d}",
                "build": "layer", "outcome": "failure", "failure": "no_layer_blocks",
            })
            return 1
        spec, reset = next_spec, False


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
    if args.build == "layer" and args.episode_set not in (None, "F", *jt.LAYER_EPISODE_SETS):
        parser.error(f"--build layer runs sets {', '.join(jt.LAYER_EPISODE_SETS)} (F with --chain)")
    if not 0.0 <= args.nudge_fraction <= 1.0:
        parser.error("--nudge-fraction must be between 0 and 1")
    if args.nudge_interval_sec <= 0 or args.nudge_duration_sec <= 0:
        parser.error("--nudge-interval-sec and --nudge-duration-sec must be positive")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
