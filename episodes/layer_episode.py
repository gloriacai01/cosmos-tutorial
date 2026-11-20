"""Place a whole Jenga layer at once: the layer-at-a-time episodes.

The reset lays the next layer's blocks side by side on a pad on the table
(`jenga_tower.build_layer_spec`). The arm grips all of them at once with a
side grip, the jaws closing across the block widths (the G2's ~30 mm pads
would only reach the middle block's ends in an end grip), carries the layer
above the tower, lowers it onto the top layer, releases, and ends like every
block-mode episode: lifted above the new top, gripper open, holding still.

Run it through `place_episode.py --build layer`; this module holds only what
differs from placing one block. Everything else (motion helpers, capture,
the end hold, error handling) is `place_episode`'s.
"""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from viam.components.arm import JointPositions

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if __package__:
    from . import jenga_tower as jt
    from . import place_episode as pe
else:
    import jenga_tower as jt
    import place_episode as pe

LAYER_PROMPT = "place the layer on the tower"
STAGE_PICK_LAYER = "pick up the layer"
STAGE_CARRY_LAYER = "carry the layer above the tower"
STAGE_LOWER_LAYER = "lower the layer onto the tower and open the gripper"


async def lay_pad(world: Any, spec: jt.LayerSpec) -> None:
    """Chained builds: lay the next layer on the pad and leave the tower and
    the other loose blocks as they stand."""
    await pe.lay_props(world, spec.pad_poses)


async def execute_place_layer(
    world: Any,
    gripper: Any,
    motion: Any,
    arm: Any,
    spec: jt.LayerSpec,
    tower: jt.TowerConfig,
    *,
    names: pe.MachineNames = pe.MachineNames(),
    grasp: pe.GraspConfig = pe.GraspConfig(),
    noise: pe.NoiseConfig = pe.NoiseConfig(),
    tolerance: jt.PlacementTolerance = jt.PlacementTolerance(),
    episode_output_dir: str | None = None,
    episode_id: str | None = None,
    cosmos_output_dir: str | None = None,
    task: str = LAYER_PROMPT,
    reset: bool = True,
    home_joint_degrees: Sequence[float] = pe.UF850_HOME_JOINT_DEGREES,
    nudge: Mapping[str, Any] | None = None,
    upload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one place-the-layer episode from `spec`. With `reset=False`
    (chained builds) only the pad is laid; the tower stands as built."""
    # The bundle is held across the block widths; single-block corrections
    # after the release keep the configured (end) grip.
    side = replace(grasp, axis="side")
    ep = pe._Episode(world, arm, gripper, motion, names, grasp)
    sim_clock = await pe._ElapsedClock.start(world, "episode")
    result: dict[str, Any] = {"spec": spec.summary(), "outcome": "incomplete"}

    await arm.move_to_joint_positions(JointPositions(values=list(home_joint_degrees)))
    await ep.open("open")

    if reset:
        with ep.timer.phase("tower_reset"):
            reset_report = await pe.reset_tower(world, tower, spec)
        result["reset"] = {
            "stable": reset_report["stable"],
            "unstable_blocks": reset_report["unstable_blocks"],
        }
        if not reset_report["stable"]:
            result["outcome"] = "reset_unstable"
            return result
    else:
        with ep.timer.phase("pad_reset"):
            await lay_pad(world, spec)

    layer_names = list(spec.pad_poses)
    middle = spec.middle_block
    tower_names = list(spec.tower_poses)
    before = await ep.read_poses(tower_names)
    on_pad = await ep.read_poses(layer_names)
    tower_top = jt.tower_top_z(tower, spec.placed_count)
    clear_z = tower_top + tower.block_thickness_m + grasp.carry_clearance_m
    carry_estimate_z = (
        tower_top + grasp.carry_clearance_m + tower.block_thickness_m / 2.0
        + grasp.grasp_z_offset_m
    )

    if spec.perturbed_start:
        await pe.perturb_start(ep, spec.seed, noise)

    if episode_output_dir is not None:
        await pe.begin_episode_capture(
            ep, STAGE_PICK_LAYER, spec.seed, middle, on_pad[middle],
            episode_output_dir=episode_output_dir, episode_id=episode_id,
            cosmos_output_dir=cosmos_output_dir, task=task, nudge=nudge,
        )
    result["nudge"] = dict(nudge) if nudge is not None and ep.capturing else None

    adjustments: list[dict[str, Any]] = []
    try:
        await ep.mark(STAGE_PICK_LAYER)
        ep.grasp = side
        try:
            reported = await ep.pick(on_pad[middle], "pick-layer", clear_z, carry_estimate_z)
        finally:
            ep.grasp = grasp
        tcp_x, tcp_y, tcp_z, pick_yaw = await ep.tcp()
        lifted = await ep.read_poses(layer_names)
        # Every block of the layer must have come up; a side grip that only
        # caught some of them has dropped the rest.
        dropped = [
            name for name in layer_names
            if lifted[name].position[2] - on_pad[name].position[2] < pe.MIN_LIFT_TO_COUNT_AS_HELD_M
        ]
        result["holding_after_lift"] = not dropped
        result["gripper_reported_holding"] = reported
        if dropped:
            result["outcome"] = "failure"
            result["failure"] = "grasp"
            result["dropped_blocks"] = dropped
            await ep.open("open-after-miss")
            await ep.finish(tower_top)
            return result

        # The layer is carried by its middle block: put that block on the
        # middle slot, the rest follow rigidly.
        slot = spec.layer_targets[middle]
        sz = slot.position[2]
        tcp_above_center = tcp_z - lifted[middle].position[2]
        sx, sy, yaw, in_hand = pe.release_target(slot, lifted[middle], (tcp_x, tcp_y), pick_yaw)
        result["in_hand_offset"] = in_hand
        carry_z = max(
            tcp_z,
            tower_top + grasp.carry_clearance_m + tower.block_thickness_m / 2.0
            + tcp_above_center,
        )
        release_z = sz + grasp.release_clearance_m + tcp_above_center

        await ep.mark(STAGE_CARRY_LAYER)
        await ep.transit(
            "carry", sx, sy, yaw, carry_z, cross_min_sec=pe.MIN_SUBTASK_DURATION_SEC
        )
        await ep.mark(STAGE_LOWER_LAYER)
        pre_x, pre_y, pre_yaw = sx, sy, yaw
        if spec.perturbed_start:
            rng = random.Random(f"place-arrival:{spec.seed}")
            pre_x += rng.gauss(0.0, noise.arrival_xy_stdev_m)
            pre_y += rng.gauss(0.0, noise.arrival_xy_stdev_m)
            pre_yaw += rng.gauss(0.0, noise.arrival_yaw_stdev_deg)
        hover_z = release_z + grasp.reaim_height_m
        await ep.go(
            "pre-release", pre_x, pre_y, hover_z, pre_yaw,
            pe.LINEAR_PICK_CONSTRAINTS, pe.APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        await pe._wait_sim_duration(world, grasp.reaim_settle_sec)
        hx, hy, _hz, hyaw = await ep.tcp()
        hanging = await ep.read_pose(middle)
        sx, sy, yaw, _aim = pe.release_target(slot, hanging, (hx, hy), hyaw)
        await ep.go(
            "re-aim", sx, sy, hover_z, yaw,
            pe.LINEAR_PICK_CONSTRAINTS, pe.APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )
        await ep.go(
            "release", sx, sy, release_z, yaw,
            pe.LINEAR_PICK_CONSTRAINTS, pe.FINAL_DESCENT_MIN_DURATION_SEC,
        )
        await ep.open("release")
        await pe._wait_sim_duration(world, grasp.release_settle_sec)
        await ep.go(
            "release-clear", sx, sy, release_z + pe.PRE_GRASP_CLEARANCE_M, yaw,
            pe.LINEAR_PICK_CONSTRAINTS, pe.APPROACH_WAYPOINT_MIN_DURATION_SEC,
        )

        # Fix any block of the new layer that landed outside tolerance, one
        # at a time, with the single-block (end) grip.
        errors: dict[str, dict[str, float]] = {}
        for name in layer_names:
            target = spec.layer_targets[name]
            placed = await ep.read_pose(name)
            error = jt.pose_error(placed, target)
            attempts = 0
            while not jt.placement_within(error, tolerance) and attempts < grasp.max_adjust_attempts:
                await ep.mark(pe.STAGE_ADJUST)
                placed = await ep.adjust(name, target, f"adjust-{name}-{attempts}", clear_z)
                error = jt.pose_error(placed, target)
                adjustments.append({"block": name, **error})
                attempts += 1
            errors[name] = error

        await ep.finish(jt.tower_top_z(tower, spec.placed_count + len(layer_names)))

        after = await ep.read_poses(tower_names)
        moved = jt.disturbed_blocks(before, after, tolerance)
        misplaced = [n for n, e in errors.items() if not jt.placement_within(e, tolerance)]
        result.update(
            {
                "placement_errors": errors,
                "misplaced_blocks": misplaced,
                "disturbed_blocks": moved,
                "adjustments": adjustments,
            }
        )
        success = not misplaced and not moved
        result["outcome"] = "success" if success else "failure"
        if not success:
            result["failure"] = "disturbed" if moved else "placement"
    finally:
        await pe.end_episode_capture(ep, result, sim_clock, upload)
    return result
