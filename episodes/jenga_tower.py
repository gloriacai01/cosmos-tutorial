"""Pure geometry and episode sampling for scripted Jenga tower placement episodes.

Nothing here talks to a machine: it decides, from a seed and a config, which
tower to build, which block to place next, where it goes, and how the episode
starts. `place_episode.py` turns that into Viam calls.

Frames: every pose is in the world frame, metres, with orientation as a WXYZ
quaternion about world Z (blocks only ever yaw). The tower stands on a support
plane at `support_z_m`.

Layout: layer `i` holds `slots_per_layer` blocks side by side. Even layers lay
their long axis along the tower's local X, odd layers along local Y, so each
layer crosses the one below at 90 degrees. Placement `k` fills layer
`k // slots_per_layer`, in the layer's fill order (`layer_slot_order`): by
default the middle slot first, then the side nearer the arm, then the far
side. Which prop fills
which placement is shuffled per seed (`prop_order`), so each block's colour
lands somewhere different in every tower.

Blocks not in the tower lie flat at random spots on the table; there is no
supply spot. Each episode picks the loose block nearest the tower whose ends
are clear for the fingers (`choose_pick_block`), a rule the cameras can see.

Real towers show seams: blocks differ slightly in shade, sit a fraction of a
millimetre apart, and are never perfectly aligned. Every tower here has a
small slot gap and per-block jitter for that reason, and `tower_props` gives
each block its own wood shade.

Seeds follow the oracle's convention: each random draw uses its own stream,
`random.Random(f"{stream}:{seed}")`, so changing one draw never shifts another.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

Position = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]

# Standard Jenga block, lying flat: long x wide x thick.
BLOCK_LENGTH_M = 0.075
BLOCK_WIDTH_M = 0.025
BLOCK_THICKNESS_M = 0.015
BLOCK_PROP_PREFIX = "jenga_block"

# Episode sets from the training plan. F (full chained builds) is not sampled
# per seed: it is a batch mode that places every block of one tower in turn.
EPISODE_SET_WEIGHTS: Mapping[str, float] = {
    "A": 0.35,  # clean tower, clean placement
    "B": 0.20,  # drifted tower
    "C": 0.20,  # perturbed start, recorded recovery
    "D": 0.15,  # top block misplaced before capture, recorded correction
    "E": 0.05,  # grasp failed before capture, recorded re-grasp
}
EPISODE_SETS = ("A", "B", "C", "D", "E", "F")


@dataclass(frozen=True)
class Pose:
    position: Position
    yaw_deg: float

    @property
    def orientation_wxyz(self) -> Quaternion:
        return yaw_to_quaternion_wxyz(self.yaw_deg)

    def as_prop_command(self, prop: str) -> dict[str, Any]:
        return {
            "command": "set_prop_pose",
            "prop": prop,
            "position": list(self.position),
            "orientation_wxyz": list(self.orientation_wxyz),
        }


@dataclass(frozen=True)
class DriftConfig:
    """Tower error that accumulates up the layers, plus per-block jitter.

    Real towers lean and drift as a whole; independent per-block noise does
    not look like that. Each layer's offset is a random walk from the layer
    below, clipped so the tower stays buildable."""

    layer_step_xy_m: float = 0.0008
    layer_step_yaw_deg: float = 0.4
    block_jitter_xy_m: float = 0.0005
    block_jitter_yaw_deg: float = 0.5
    # Jitter every tower gets, drifted or not, so seams show between blocks.
    seam_jitter_xy_m: float = 0.0003
    seam_jitter_yaw_deg: float = 0.3
    max_layer_offset_xy_m: float = 0.004
    max_layer_offset_yaw_deg: float = 2.0


@dataclass(frozen=True)
class TowerConfig:
    base_xy_m: tuple[float, float] = (0.30, 0.10)
    base_yaw_deg: float = 0.0
    support_z_m: float = 0.0
    # Our towers are 10 layers (30 blocks, about 150 mm), not a full Jenga set.
    layers: int = 10
    slots_per_layer: int = 3
    # Fill order within a layer, as slot indices (0 = negative side of the
    # layer, slots_per_layer - 1 = positive side). None: middle first, then
    # the side nearer the arm, then the far side, worked out per layer
    # direction from `arm_xy_m`. Fixed per tower so identical images never
    # get conflicting labels.
    slot_order: tuple[int, ...] | None = None
    # The arm base in world XY, for the default fill order.
    arm_xy_m: tuple[float, float] = (0.0, 0.0)
    block_length_m: float = BLOCK_LENGTH_M
    block_width_m: float = BLOCK_WIDTH_M
    block_thickness_m: float = BLOCK_THICKNESS_M
    # Centre-to-centre spacing beyond the block width. The Gripper G2's pads
    # are about 29.6 mm wide against a 25 mm block, so in an end grip each pad
    # overhangs its block by ~2.3 mm toward the neighbours; 3 mm keeps them
    # clear. Use more (e.g. 0.012) for room for a side-grip finger.
    slot_gap_m: float = 0.003
    prop_prefix: str = BLOCK_PROP_PREFIX
    drift: DriftConfig = field(default_factory=DriftConfig)

    def __post_init__(self) -> None:
        if self.layers < 1 or self.slots_per_layer < 1:
            raise ValueError("tower needs at least one layer and one slot")
        if self.slot_order is not None and sorted(self.slot_order) != list(
            range(self.slots_per_layer)
        ):
            raise ValueError("slot_order must be a permutation of the slot indices")
        if self.slot_gap_m < 0.0:
            raise ValueError("slot_gap_m must be non-negative")

    @property
    def block_count(self) -> int:
        return self.layers * self.slots_per_layer

    def block_name(self, placement_index: int) -> str:
        """Placement k always uses the same prop, so a tower of n blocks is
        exactly props 0..n-1 and the next block to place is prop n."""
        if not 0 <= placement_index < self.block_count:
            raise ValueError(f"placement index {placement_index} outside the tower")
        return f"{self.prop_prefix}_{placement_index:02d}"

    def block_names(self) -> list[str]:
        return [self.block_name(k) for k in range(self.block_count)]


def prop_order(config: TowerConfig, seed: int) -> list[str]:
    """Which prop fills each placement, shuffled per seed. Placement k uses
    prop_order(...)[k]; chained builds share one seed, so one order."""
    names = config.block_names()
    _rng("prop-order", seed).shuffle(names)
    return names


def scaled_block_dims(scale: float) -> dict[str, float]:
    """Block dimensions for a uniformly scaled Jenga block, as TowerConfig
    keyword arguments. Uniform scaling keeps three blocks to a square layer."""
    if not 0.0 < scale <= 1.0:
        raise ValueError("block scale must be in (0, 1]")
    return {
        "block_length_m": BLOCK_LENGTH_M * scale,
        "block_width_m": BLOCK_WIDTH_M * scale,
        "block_thickness_m": BLOCK_THICKNESS_M * scale,
    }


@dataclass(frozen=True)
class ScatterConfig:
    """Where loose blocks lie on the table, and the rule for which one to
    pick next. The region is in world XY."""

    region_min_xy_m: tuple[float, float] = (0.20, -0.40)
    region_max_xy_m: tuple[float, float] = (0.60, 0.40)
    # Clear space between any two loose blocks' footprints.
    min_gap_m: float = 0.006
    # Loose blocks stay this far outside the tower's footprint, so the hand
    # placing a bottom-layer block never brushes them.
    tower_clearance_m: float = 0.03
    # Pick rule: blocks closer than this to the tower centre are skipped,
    # leaving room for the whole hand beside the tower.
    min_pick_distance_m: float = 0.13
    # Pick rule: each end of the block needs this box clear for a finger:
    # 10 mm between the block end and the G2's open pad (95 mm gap) plus an
    # estimated 15 mm finger, by the 29.6 mm pad width plus 4 mm a side.
    finger_clear_length_m: float = 0.025
    finger_clear_width_m: float = 0.038


@dataclass(frozen=True)
class EpisodeSamplingConfig:
    misplace_xy_range_m: tuple[float, float] = (0.002, 0.005)
    misplace_yaw_range_deg: tuple[float, float] = (2.0, 5.0)
    # Set E: the block is found this far from where the failed grasp was aimed.
    regrasp_offset_range_m: tuple[float, float] = (0.01, 0.03)
    set_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(EPISODE_SET_WEIGHTS)
    )


@dataclass(frozen=True)
class SlotRef:
    placement_index: int
    layer: int
    slot: int


@dataclass(frozen=True)
class EpisodeSpec:
    """Everything an episode needs that is decided before the robot moves."""

    seed: int
    episode_set: str
    placed_count: int
    # Poses the tower blocks are reset to, by prop name (placement order).
    tower_poses: Mapping[str, Pose]
    next_slot: SlotRef
    next_block: str
    next_slot_pose: Pose
    # Where the next block lies on the table before the episode.
    pick_pose: Pose
    # Set D: the last tower block starts here instead of its tower pose.
    misplaced_block: str | None = None
    misplaced_pose: Pose | None = None
    # Set E: the arm starts hovering empty above here, as after a missed grasp.
    failed_grasp_aim_xy_m: tuple[float, float] | None = None
    # Set C: start from a perturbed arm pose before capture.
    perturbed_start: bool = False
    # The blocks lying on the table, by prop name; the next block is one of
    # them. Every block not in the tower, unless the build limits how many
    # lie out (--loose-blocks).
    loose_poses: Mapping[str, Pose] = field(default_factory=dict)
    # Chained builds with a loose-block limit: blocks laid on the table
    # before this step (a subset of loose_poses), the rest stay as they lie.
    fresh_poses: Mapping[str, Pose] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "episode_set": self.episode_set,
            "placed_count": self.placed_count,
            "next_block": self.next_block,
            "next_slot": {
                "placement_index": self.next_slot.placement_index,
                "layer": self.next_slot.layer,
                "slot": self.next_slot.slot,
            },
            "next_slot_pose": _pose_json(self.next_slot_pose),
            "pick_pose": _pose_json(self.pick_pose),
            "misplaced_block": self.misplaced_block,
            "misplaced_pose": (
                _pose_json(self.misplaced_pose) if self.misplaced_pose else None
            ),
            "failed_grasp_aim_xy_m": (
                list(self.failed_grasp_aim_xy_m) if self.failed_grasp_aim_xy_m else None
            ),
            "perturbed_start": self.perturbed_start,
            "loose_blocks": len(self.loose_poses),
        }


def _pose_json(pose: Pose) -> dict[str, Any]:
    return {"position_m": list(pose.position), "yaw_deg": pose.yaw_deg}


def yaw_to_quaternion_wxyz(yaw_deg: float) -> Quaternion:
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def quaternion_wxyz_to_yaw_deg(rotation_wxyz: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in rotation_wxyz)
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_deg(angle_deg: float, period_deg: float = 360.0) -> float:
    """Wrap into [-period/2, period/2)."""
    half = period_deg / 2.0
    return (angle_deg + half) % period_deg - half


def yaw_error_deg(actual_deg: float, target_deg: float) -> float:
    """A block looks and grasps the same turned 180 degrees."""
    return abs(wrap_deg(actual_deg - target_deg, 180.0))


def _rng(stream: str, seed: int) -> random.Random:
    return random.Random(f"{stream}:{int(seed)}")


def slot_ref(config: TowerConfig, placement_index: int) -> SlotRef:
    if not 0 <= placement_index < config.block_count:
        raise ValueError(f"placement index {placement_index} outside the tower")
    layer, position_in_layer = divmod(placement_index, config.slots_per_layer)
    return SlotRef(placement_index, layer, layer_slot_order(config, layer)[position_in_layer])


def layer_slot_order(config: TowerConfig, layer: int) -> tuple[int, ...]:
    """The order a layer's slots are filled in: `slot_order` if set, else
    from the middle outward, the side nearer the arm before the far side.

    Even and odd layers run in different directions, so which side is nearer
    the arm can differ between them; ties go to the lower slot index."""
    if config.slot_order is not None:
        return config.slot_order
    centre = (config.slots_per_layer - 1) / 2.0

    def key(slot: int) -> tuple[float, float, int]:
        pose = nominal_slot_pose(config, SlotRef(0, layer, slot))
        return (abs(slot - centre), math.dist(pose.position[:2], config.arm_xy_m), slot)

    return tuple(sorted(range(config.slots_per_layer), key=key))


# --- Slot names and prompts ----------------------------------------------------
#
# Slots are named from the robot's side: standing at the arm base and looking
# at the tower centre. A layer whose slots run across that line of sight has
# right / middle / left slots; one whose slots run along it has near / middle /
# far. Layers are numbered from 1 at the bottom. Episodes and evals share these
# functions, so a prompt always names the slot the oracle filled.

SLOT_TASK_TEMPLATE = "place the block in the {slot} slot on layer {layer}"
SLOT_STAGE_TEMPLATES = {
    "carry": "carry the block above the {slot} slot on layer {layer}",
    "lower": "lower the block into the {slot} slot on layer {layer} and open the gripper",
    "adjust": "adjust the block into the {slot} slot on layer {layer}",
}


def _sight_axes(config: TowerConfig) -> tuple[tuple[float, float], tuple[float, float]]:
    """Unit vectors from the arm base toward the tower centre, and to the
    robot's left of that line, in world XY."""
    dx = config.base_xy_m[0] - config.arm_xy_m[0]
    dy = config.base_xy_m[1] - config.arm_xy_m[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        raise ValueError("the tower centre is on the arm base, so its slots have no sides")
    forward = (dx / norm, dy / norm)
    return forward, (-forward[1], forward[0])


def slot_names(config: TowerConfig, layer: int) -> tuple[str, str, str]:
    """The name of each slot index in `layer`, seen from the arm: near /
    middle / far when the layer's slots run toward the arm, right / middle /
    left when they run across. Exactly 45 deg counts as toward."""
    if config.slots_per_layer != 3:
        raise ValueError("slot names need three slots per layer")
    forward, left = _sight_axes(config)
    first = nominal_slot_pose(config, SlotRef(0, layer, 0)).position
    last = nominal_slot_pose(config, SlotRef(0, layer, 2)).position
    ax, ay = last[0] - first[0], last[1] - first[1]
    along = ax * forward[0] + ay * forward[1]
    across = ax * left[0] + ay * left[1]
    if abs(along) >= abs(across):
        return ("near", "middle", "far") if along > 0 else ("far", "middle", "near")
    return ("right", "middle", "left") if across > 0 else ("left", "middle", "right")


def slot_name(config: TowerConfig, ref: SlotRef) -> str:
    return slot_names(config, ref.layer)[ref.slot]


def layer_number(ref: SlotRef) -> int:
    """The layer as prompts count it: 1 is the bottom."""
    return ref.layer + 1


def slot_prompt(config: TowerConfig, ref: SlotRef, template: str = SLOT_TASK_TEMPLATE) -> str:
    """The prompt for placing a block in `ref`, e.g. "place the block in the
    near slot on layer 2"."""
    return template.format(slot=slot_name(config, ref), layer=layer_number(ref))


def slot_stage_prompts(config: TowerConfig, ref: SlotRef) -> dict[str, str]:
    """The slot-naming stage prompts (carry, lower, adjust) for `ref`."""
    return {stage: slot_prompt(config, ref, t) for stage, t in SLOT_STAGE_TEMPLATES.items()}


def slot_ref_named(config: TowerConfig, layer: int, name: str) -> SlotRef:
    """The slot a prompt names, by its 1-based layer number and slot name;
    for evals that turn a prompt back into a target."""
    if not 1 <= layer <= config.layers:
        raise ValueError(f"layer {layer} outside 1..{config.layers}")
    names = slot_names(config, layer - 1)
    if name not in names:
        raise ValueError(f"layer {layer} has slots {names}, not {name!r}")
    slot = names.index(name)
    order = layer_slot_order(config, layer - 1)
    return SlotRef((layer - 1) * config.slots_per_layer + order.index(slot), layer - 1, slot)


def _rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (c * x - s * y, s * x + c * y)


def layer_center_z(config: TowerConfig, layer: int) -> float:
    return config.support_z_m + config.block_thickness_m * (layer + 0.5)


def layer_support_z(config: TowerConfig, layer: int) -> float:
    """Height of the surface a block in this layer rests on."""
    return config.support_z_m + config.block_thickness_m * layer


def tower_top_z(config: TowerConfig, placed_count: int) -> float:
    layers_started = math.ceil(placed_count / config.slots_per_layer)
    return config.support_z_m + config.block_thickness_m * layers_started


def nominal_slot_pose(
    config: TowerConfig,
    ref: SlotRef,
    layer_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    layer_offset_yaw_deg: float = 0.0,
) -> Pose:
    """Slot centre for a perfect tower, optionally shifted by a layer offset."""
    pitch = config.block_width_m + config.slot_gap_m
    across = (ref.slot - (config.slots_per_layer - 1) / 2.0) * pitch
    crossed = ref.layer % 2 == 1
    # Even layers: long axis along local X, blocks spread along local Y.
    local_x, local_y = (across, 0.0) if crossed else (0.0, across)
    local_yaw = 90.0 if crossed else 0.0
    layer_yaw = config.base_yaw_deg + layer_offset_yaw_deg
    dx, dy = _rotate_xy(local_x, local_y, layer_yaw)
    return Pose(
        position=(
            config.base_xy_m[0] + layer_offset_xy_m[0] + dx,
            config.base_xy_m[1] + layer_offset_xy_m[1] + dy,
            layer_center_z(config, ref.layer),
        ),
        yaw_deg=wrap_deg(layer_yaw + local_yaw, 180.0),
    )


def layer_offsets(
    config: TowerConfig, seed: int, drifted: bool
) -> list[tuple[tuple[float, float], float]]:
    """Per-layer (xy offset, yaw offset): a clipped random walk, or zeros."""
    if not drifted:
        return [((0.0, 0.0), 0.0)] * config.layers
    drift = config.drift
    rng = _rng("tower-drift", seed)
    offsets: list[tuple[tuple[float, float], float]] = []
    x = y = yaw = 0.0
    for _layer in range(config.layers):
        x += rng.gauss(0.0, drift.layer_step_xy_m)
        y += rng.gauss(0.0, drift.layer_step_xy_m)
        yaw += rng.gauss(0.0, drift.layer_step_yaw_deg)
        norm = math.hypot(x, y)
        if norm > drift.max_layer_offset_xy_m:
            scale = drift.max_layer_offset_xy_m / norm
            x, y = x * scale, y * scale
        yaw = max(-drift.max_layer_offset_yaw_deg, min(drift.max_layer_offset_yaw_deg, yaw))
        offsets.append(((x, y), yaw))
    return offsets


def tower_poses(
    config: TowerConfig,
    seed: int,
    placed_count: int,
    drifted: bool,
    names: Sequence[str] | None = None,
) -> dict[str, Pose]:
    """Reset poses for the first `placed_count` blocks of the tower, keyed by
    prop name in placement order. `names` defaults to `prop_order`."""
    if not 0 <= placed_count <= config.block_count:
        raise ValueError(f"placed_count {placed_count} outside 0..{config.block_count}")
    names = list(names) if names is not None else prop_order(config, seed)
    offsets = layer_offsets(config, seed, drifted)
    jitter = _rng("tower-block-jitter", seed)
    drift = config.drift
    xy_sd = drift.block_jitter_xy_m if drifted else drift.seam_jitter_xy_m
    yaw_sd = drift.block_jitter_yaw_deg if drifted else drift.seam_jitter_yaw_deg
    poses: dict[str, Pose] = {}
    by_layer: dict[int, list[Pose]] = {}
    for k in range(placed_count):
        ref = slot_ref(config, k)
        (ox, oy), oyaw = offsets[ref.layer]
        nominal = nominal_slot_pose(config, ref, (ox, oy), oyaw)
        jx, jy = jitter.gauss(0.0, xy_sd), jitter.gauss(0.0, xy_sd)
        jyaw = jitter.gauss(0.0, yaw_sd)
        pose = Pose(
            (nominal.position[0] + jx, nominal.position[1] + jy, nominal.position[2]),
            wrap_deg(nominal.yaw_deg + jyaw, 180.0),
        )
        neighbours = by_layer.setdefault(ref.layer, [])
        if any(blocks_overlap(config, pose, other, BLOCK_CLEARANCE_M) for other in neighbours):
            # The jitter would push this block into a neighbour: keep its
            # layer's pose instead, so a reset never starts interpenetrating.
            pose = nominal
        neighbours.append(pose)
        poses[names[k]] = pose
    return poses


def next_slot_target(
    config: TowerConfig, seed: int, placed_count: int, drifted: bool
) -> Pose:
    """Where the next block should go: its slot, following its layer's drift.

    A block in a layer that already has blocks follows that layer's offset; a
    block starting a new layer centres on the layer below's offset, since it
    should sit on the tower that is actually there."""
    ref = slot_ref(config, placed_count)
    offsets = layer_offsets(config, seed, drifted)
    if placed_count % config.slots_per_layer:
        (ox, oy), oyaw = offsets[ref.layer]
    elif ref.layer == 0:
        (ox, oy), oyaw = (0.0, 0.0), 0.0
    else:
        (ox, oy), oyaw = offsets[ref.layer - 1]
    return nominal_slot_pose(config, ref, (ox, oy), oyaw)


def tower_footprint_radius(config: TowerConfig) -> float:
    half_long = config.block_length_m / 2.0
    pitch = config.block_width_m + config.slot_gap_m
    half_across = (config.slots_per_layer * pitch) / 2.0
    return math.hypot(max(half_long, half_across), max(half_long, half_across))


# Blocks side by side in a layer keep at least this much clear space between
# them when a tower is generated, so resets never start interpenetrating.
BLOCK_CLEARANCE_M = 0.0005


def _rect_corners(
    xy: Sequence[float], yaw_deg: float, length_m: float, width_m: float
) -> list[tuple[float, float]]:
    """The four corners of a rectangle centred at `xy`, long axis at `yaw_deg`."""
    corners = []
    for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
        dx, dy = _rotate_xy(sx * length_m / 2.0, sy * width_m / 2.0, yaw_deg)
        corners.append((xy[0] + dx, xy[1] + dy))
    return corners


def _rects_overlap(
    a: Sequence[tuple[float, float]],
    b: Sequence[tuple[float, float]],
    margin_m: float = 0.0,
) -> bool:
    """Separating-axis test on two rectangles given by their corners: whether
    they overlap or come within `margin_m` of each other."""
    for corners in (a, b):
        for i in range(2):
            ex = corners[i + 1][0] - corners[i][0]
            ey = corners[i + 1][1] - corners[i][1]
            norm = math.hypot(ex, ey)
            axis = (ex / norm, ey / norm)
            pa = [px * axis[0] + py * axis[1] for px, py in a]
            pb = [px * axis[0] + py * axis[1] for px, py in b]
            if max(pa) + margin_m <= min(pb) or max(pb) + margin_m <= min(pa):
                return False
    return True


def block_corners_xy(config: TowerConfig, pose: Pose) -> list[tuple[float, float]]:
    """The four corners of a flat block's footprint in world XY."""
    return _rect_corners(
        pose.position[:2], pose.yaw_deg, config.block_length_m, config.block_width_m
    )


def blocks_overlap(config: TowerConfig, a: Pose, b: Pose, margin_m: float = 0.0) -> bool:
    """Whether two flat blocks' footprints overlap, or come within `margin_m`
    of each other."""
    diagonal = math.hypot(config.block_length_m, config.block_width_m)
    if math.dist(a.position[:2], b.position[:2]) > diagonal + margin_m:
        return False  # too far apart to touch; skips the full test
    return _rects_overlap(block_corners_xy(config, a), block_corners_xy(config, b), margin_m)


def finger_zones(
    config: TowerConfig, scatter: ScatterConfig, pose: Pose
) -> list[list[tuple[float, float]]]:
    """Where the open fingers come down for an end grip: a box beyond each end
    of the block, along its long axis."""
    reach = config.block_length_m / 2.0 + scatter.finger_clear_length_m / 2.0
    zones = []
    for sign in (1.0, -1.0):
        dx, dy = _rotate_xy(sign * reach, 0.0, pose.yaw_deg)
        centre = (pose.position[0] + dx, pose.position[1] + dy)
        zones.append(
            _rect_corners(
                centre, pose.yaw_deg,
                scatter.finger_clear_length_m, scatter.finger_clear_width_m,
            )
        )
    return zones


# A loose block counts as lying flat on the table within this height error;
# one on its side or on another block sits 5 mm or more higher.
FLAT_Z_TOLERANCE_M = 0.003


def choose_pick_block(
    config: TowerConfig, scatter: ScatterConfig, loose: Mapping[str, Pose]
) -> str | None:
    """The next block to pick: the loose block nearest the tower centre that
    lies flat, is at least `min_pick_distance_m` from the tower centre, and has
    both ends clear of every other loose block for the fingers. None if no
    block qualifies.

    The policy has no memory, so the choice must follow from what the cameras
    show; ties go to the prop name so it is deterministic."""
    flat_z = config.support_z_m + config.block_thickness_m / 2.0
    footprints = {name: block_corners_xy(config, pose) for name, pose in loose.items()}
    best: tuple[float, str] | None = None
    for name, pose in loose.items():
        if abs(pose.position[2] - flat_z) > FLAT_Z_TOLERANCE_M:
            continue
        distance = math.dist(pose.position[:2], config.base_xy_m)
        if distance < scatter.min_pick_distance_m:
            continue
        zones = finger_zones(config, scatter, pose)
        if any(
            _rects_overlap(zone, corners)
            for zone in zones
            for other, corners in footprints.items()
            if other != name
        ):
            continue
        if best is None or (distance, name) < best:
            best = (distance, name)
    return None if best is None else best[1]


# Each scatter layout gets this many draws per block, and each seed this many
# layouts, before giving up.
SCATTER_DRAWS_PER_BLOCK = 400
SCATTER_LAYOUTS = 20


def _scatter_once(
    tower: TowerConfig,
    scatter: ScatterConfig,
    rng: random.Random,
    names: Sequence[str],
    keep_clear: Sequence[Pose] = (),
) -> dict[str, Pose] | None:
    lo_x, lo_y = scatter.region_min_xy_m
    hi_x, hi_y = scatter.region_max_xy_m
    half_diag = math.hypot(tower.block_length_m, tower.block_width_m) / 2.0
    tower_reach = tower_footprint_radius(tower) + scatter.tower_clearance_m + half_diag
    z = tower.support_z_m + tower.block_thickness_m / 2.0
    poses: dict[str, Pose] = {}
    for name in names:
        for _draw in range(SCATTER_DRAWS_PER_BLOCK):
            yaw = wrap_deg(rng.uniform(-90.0, 90.0), 180.0)
            c, s = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
            half_x = (c * tower.block_length_m + s * tower.block_width_m) / 2.0
            half_y = (s * tower.block_length_m + c * tower.block_width_m) / 2.0
            if lo_x + half_x > hi_x - half_x or lo_y + half_y > hi_y - half_y:
                raise ValueError("scatter region cannot hold a block")
            x = rng.uniform(lo_x + half_x, hi_x - half_x)
            y = rng.uniform(lo_y + half_y, hi_y - half_y)
            if math.dist((x, y), tower.base_xy_m) < tower_reach:
                continue
            pose = Pose((x, y, z), yaw)
            if any(
                blocks_overlap(tower, pose, other, scatter.tower_clearance_m)
                for other in keep_clear
            ):
                continue
            if not any(
                blocks_overlap(tower, pose, other, scatter.min_gap_m)
                for other in poses.values()
            ):
                poses[name] = pose
                break
        else:
            return None
    return poses


def sample_loose_poses(
    tower: TowerConfig,
    scatter: ScatterConfig,
    seed: int,
    names: Sequence[str],
    *,
    keep_clear: Sequence[Pose] = (),
    require_pick: bool = True,
    stream: str = "scatter",
) -> dict[str, Pose]:
    """Flat random poses on the table for every block in `names`: inside the
    region, `min_gap_m` apart, clear of the tower and of the `keep_clear`
    blocks (the layer pad, or loose blocks already down) by
    `tower_clearance_m`, and, with `require_pick`, with at least one block the
    pick rule accepts. `stream` keys the draws, so each step of a chained
    build gets its own layout. Raises if the region can't hold them all."""
    for layout in range(SCATTER_LAYOUTS):
        poses = _scatter_once(
            tower, scatter, _rng(f"{stream}:{layout}", seed), names, keep_clear
        )
        if poses is None:
            continue
        if not require_pick or choose_pick_block(tower, scatter, poses) is not None:
            return poses
    raise ValueError(
        f"could not lay {len(names)} loose blocks out in the scatter region; widen it"
    )


def sample_episode_set(seed: int, weights: Mapping[str, float]) -> str:
    names = [name for name, weight in weights.items() if weight > 0.0]
    if not names:
        raise ValueError("episode set weights must have a positive entry")
    sampled = set(EPISODE_SET_WEIGHTS)
    if not set(names) <= sampled:
        raise ValueError(
            f"only sets {sorted(sampled)} are sampled per seed; got {sorted(names)}"
        )
    return _rng("episode-set", seed).choices(names, [weights[n] for n in names])[0]


def _misplaced(
    config: TowerConfig,
    target: Pose,
    neighbours: Sequence[Pose],
    sampling: EpisodeSamplingConfig,
    seed: int,
) -> Pose:
    """Set D's starting error: the block shifted 2-5 mm and turned 2-5 deg,
    drawn until it clears the other blocks in its layer.

    A side block can't move far toward its neighbour (the gap is only a few
    mm) and a 5 deg turn alone swings its ends by ~3 mm, so draws that would
    overlap are rejected. If none fits, the block slides along its own long
    axis, where a layer has no neighbours."""
    rng = _rng("misplace", seed)
    x, y, z = target.position
    for _attempt in range(64):
        distance = rng.uniform(*sampling.misplace_xy_range_m)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        yaw = rng.uniform(*sampling.misplace_yaw_range_deg) * rng.choice((-1.0, 1.0))
        pose = Pose(
            (x + distance * math.cos(heading), y + distance * math.sin(heading), z),
            wrap_deg(target.yaw_deg + yaw, 180.0),
        )
        if not any(blocks_overlap(config, pose, n, BLOCK_CLEARANCE_M) for n in neighbours):
            return pose
    dx, dy = _rotate_xy(sampling.misplace_xy_range_m[0], 0.0, target.yaw_deg)
    return Pose((x + dx, y + dy, z), target.yaw_deg)


def sample_tower_height(tower: TowerConfig, seed: int, episode_set: str) -> int:
    """Blocks already placed, uniform over 0..block_count-1 (from 1 for set D)."""
    low = 1 if episode_set == "D" else 0
    return _rng("tower-height", seed).randint(low, tower.block_count - 1)


def build_episode_spec(
    seed: int,
    tower: TowerConfig,
    scatter: ScatterConfig = ScatterConfig(),
    sampling: EpisodeSamplingConfig = EpisodeSamplingConfig(),
    *,
    episode_set: str | None = None,
    placed_count: int | None = None,
    loose_blocks: int | None = None,
) -> EpisodeSpec:
    """Decide one episode from its seed.

    Tower height is uniform over 0..block_count-1, so every slot is the next
    slot equally often and the top layers get as much data as the bottom.
    Set D needs a block to misplace, so it never starts from an empty tower.
    Every block not in the tower is scattered on the table, or only the first
    `loose_blocks` of them (the reset parks the rest off the table)."""
    if loose_blocks is not None and loose_blocks < 1:
        raise ValueError("loose_blocks must be at least 1")
    chosen = episode_set or sample_episode_set(seed, sampling.set_weights)
    if chosen not in EPISODE_SETS:
        raise ValueError(f"unknown episode set {chosen!r}")
    if placed_count is None:
        placed_count = sample_tower_height(tower, seed, chosen)
    if not 0 <= placed_count < tower.block_count:
        raise ValueError(f"placed_count {placed_count} leaves no slot to fill")
    if chosen == "D" and placed_count == 0:
        raise ValueError("set D needs at least one placed block")

    # F targets the clean tower: its drift is the oracle's own accumulated error.
    drifted = chosen in ("B", "D")
    names = prop_order(tower, seed)
    poses = tower_poses(tower, seed, placed_count, drifted, names)
    ref = slot_ref(tower, placed_count)
    off_tower = names[placed_count:]
    if loose_blocks is not None:
        off_tower = off_tower[:loose_blocks]
    loose = sample_loose_poses(tower, scatter, seed, off_tower)
    next_block = choose_pick_block(tower, scatter, loose)
    assert next_block is not None  # sample_loose_poses guarantees one
    pick_pose = loose[next_block]

    misplaced_block = misplaced_pose = None
    if chosen == "D":
        misplaced_block = names[placed_count - 1]
        layer = slot_ref(tower, placed_count - 1).layer
        neighbours = [
            poses[names[k]] for k in range(placed_count - 1)
            if slot_ref(tower, k).layer == layer
        ]
        misplaced_pose = _misplaced(
            tower, poses[misplaced_block], neighbours, sampling, seed
        )

    failed_aim = None
    if chosen == "E":
        rng = _rng("failed-grasp", seed)
        distance = rng.uniform(*sampling.regrasp_offset_range_m)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        failed_aim = (
            pick_pose.position[0] + distance * math.cos(heading),
            pick_pose.position[1] + distance * math.sin(heading),
        )

    return EpisodeSpec(
        seed=int(seed),
        episode_set=chosen,
        placed_count=placed_count,
        tower_poses=poses,
        next_slot=ref,
        next_block=next_block,
        next_slot_pose=next_slot_target(tower, seed, placed_count, drifted),
        pick_pose=pick_pose,
        misplaced_block=misplaced_block,
        misplaced_pose=misplaced_pose,
        failed_grasp_aim_xy_m=failed_aim,
        perturbed_start=chosen == "C",
        loose_poses=loose,
    )


def chain_step_spec(
    seed: int,
    tower: TowerConfig,
    scatter: ScatterConfig,
    tower_names: Sequence[str],
    loose: Mapping[str, Pose],
    loose_blocks: int | None = None,
) -> EpisodeSpec | None:
    """The next step of a chained build (set F), from the tower built so far
    (prop names in placement order) and where the loose blocks lie now. The
    loose blocks stay where they are; with `loose_blocks`, parked blocks are
    laid out (fresh_poses) to top the table back up to that many. None if no
    loose block can be picked."""
    placed_count = len(tower_names)
    if placed_count >= tower.block_count:
        raise ValueError("the tower is already complete")
    fresh: dict[str, Pose] = {}
    if loose_blocks is not None and len(loose) < loose_blocks:
        parked = [
            name for name in prop_order(tower, seed)
            if name not in tower_names and name not in loose
        ]
        fresh = sample_loose_poses(
            tower, scatter, seed, parked[: loose_blocks - len(loose)],
            keep_clear=list(loose.values()), require_pick=False,
            stream=f"scatter-step{placed_count}",
        )
        loose = {**loose, **fresh}
    next_block = choose_pick_block(tower, scatter, loose)
    if next_block is None:
        return None
    return EpisodeSpec(
        seed=int(seed),
        episode_set="F",
        placed_count=placed_count,
        tower_poses=tower_poses(tower, seed, placed_count, False, tower_names),
        next_slot=slot_ref(tower, placed_count),
        next_block=next_block,
        next_slot_pose=next_slot_target(tower, seed, placed_count, False),
        pick_pose=loose[next_block],
        loose_poses=dict(loose),
        fresh_poses=fresh,
    )


# --- Layer at a time ---------------------------------------------------------
#
# The training plan's other way to build: the reset lays a whole layer (one
# block per slot, side by side) on a pad on the table, and the robot places
# all of it on the tower at once with a side grip across the blocks. Twelve
# placements per tower instead of 36.

# Layer-mode episode sets: clean tower, drifted tower, perturbed start. D and
# E (a misplaced top block, a missed grasp) are block-mode only for now.
LAYER_EPISODE_SETS = ("A", "B", "C")


@dataclass(frozen=True)
class PadConfig:
    """Where the reset lays the next layer, ready to pick, in world XY.

    The layer points the way the tower's next layer does, so the carry is
    nearly a translation. Its centre and yaw are jittered per seed so the
    policy has to find it in the image rather than memorise one spot."""

    center_xy_m: tuple[float, float] = (0.30, -0.15)
    # Gap between neighbouring blocks on the pad. Nothing goes between them
    # (the side grip closes across all of them), so it is a seam, not room.
    seam_m: float = 0.001
    xy_jitter_m: float = 0.015
    yaw_jitter_deg: float = 10.0


@dataclass(frozen=True)
class LayerSpec:
    """Everything a place-the-layer episode needs before the robot moves."""

    seed: int
    episode_set: str
    # Whole layers already on the tower; the next layer is this index.
    placed_layers: int
    placed_count: int
    # Tower blocks by prop name, in placement order.
    tower_poses: Mapping[str, Pose]
    # The layer on the pad and where each of its blocks goes, both keyed by
    # prop name and listed by slot (0 = negative side of the layer).
    pad_poses: Mapping[str, Pose]
    layer_targets: Mapping[str, Pose]
    # Blocks neither in the tower nor on the pad, lying on the table.
    loose_poses: Mapping[str, Pose] = field(default_factory=dict)
    perturbed_start: bool = False
    # Block mode's set D fields; never used in layer mode, read by the reset.
    misplaced_block: str | None = None
    misplaced_pose: Pose | None = None

    @property
    def middle_block(self) -> str:
        """The block the grip centres on and the carry aims with."""
        names = list(self.pad_poses)
        return names[len(names) // 2]

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "episode_set": self.episode_set,
            "build": "layer",
            "placed_layers": self.placed_layers,
            "placed_count": self.placed_count,
            "pad_blocks": list(self.pad_poses),
            "pad_poses": {n: _pose_json(p) for n, p in self.pad_poses.items()},
            "layer_targets": {n: _pose_json(p) for n, p in self.layer_targets.items()},
            "perturbed_start": self.perturbed_start,
            "loose_blocks": len(self.loose_poses),
        }


def next_layer_targets(
    config: TowerConfig, seed: int, layer: int, drifted: bool
) -> list[Pose]:
    """Slot poses for a whole new layer, by slot index. A new layer centres on
    the layer below's drift, as `next_slot_target` does for its first block."""
    if not 0 <= layer < config.layers:
        raise ValueError(f"layer {layer} outside the tower")
    offsets = layer_offsets(config, seed, drifted)
    (ox, oy), oyaw = ((0.0, 0.0), 0.0) if layer == 0 else offsets[layer - 1]
    order = layer_slot_order(config, layer)
    start = layer * config.slots_per_layer
    return [
        nominal_slot_pose(
            config, SlotRef(start + order.index(slot), layer, slot), (ox, oy), oyaw
        )
        for slot in range(config.slots_per_layer)
    ]


def pad_layer_poses(
    config: TowerConfig, pad: PadConfig, seed: int, layer: int, key: str = "pad"
) -> list[Pose]:
    """The layer as the reset lays it on the pad, by slot index: the tower's
    layer `layer` (undrifted) moved to the pad centre, turned by the seed's
    yaw jitter, with the tower's seam jitter per block. `config` is the
    layer-mode tower (`layer_mode_tower`), so the blocks are `seam_m` apart."""
    rng = _rng(key, seed)
    nominal = next_layer_targets(config, seed, layer, drifted=False)
    centre_x = pad.center_xy_m[0] + rng.uniform(-pad.xy_jitter_m, pad.xy_jitter_m)
    centre_y = pad.center_xy_m[1] + rng.uniform(-pad.xy_jitter_m, pad.xy_jitter_m)
    turn = rng.uniform(-pad.yaw_jitter_deg, pad.yaw_jitter_deg)
    z = config.support_z_m + config.block_thickness_m / 2.0
    drift = config.drift
    poses: list[Pose] = []
    for target in nominal:
        dx, dy = _rotate_xy(
            target.position[0] - config.base_xy_m[0],
            target.position[1] - config.base_xy_m[1],
            turn,
        )
        nominal_pose = Pose(
            (centre_x + dx, centre_y + dy, z), wrap_deg(target.yaw_deg + turn, 180.0)
        )
        jittered = Pose(
            (
                nominal_pose.position[0] + rng.gauss(0.0, drift.seam_jitter_xy_m),
                nominal_pose.position[1] + rng.gauss(0.0, drift.seam_jitter_xy_m),
                z,
            ),
            wrap_deg(nominal_pose.yaw_deg + rng.gauss(0.0, drift.seam_jitter_yaw_deg), 180.0),
        )
        if any(blocks_overlap(config, jittered, other, BLOCK_CLEARANCE_M) for other in poses):
            jittered = nominal_pose
        poses.append(jittered)
    return poses


def layer_mode_tower(config: TowerConfig, pad: PadConfig) -> TowerConfig:
    """The tower as layer mode builds it. A side grip squeezes the layer
    together, so placed layers sit `pad.seam_m` apart, not the block-mode gap
    that leaves room for end-grip pads between neighbours."""
    return replace(config, slot_gap_m=pad.seam_m)


def _check_pad_clear_of_tower(config: TowerConfig, pad: PadConfig) -> None:
    reach = 2.0 * tower_footprint_radius(config) + math.sqrt(2.0) * pad.xy_jitter_m
    if math.dist(pad.center_xy_m, config.base_xy_m) < reach:
        raise ValueError(
            f"layer pad at {pad.center_xy_m} is within {reach:.3f} m of the tower; move it"
        )


def _pad_names_by_slot(
    config: TowerConfig, layer: int, names_in_placement_order: Sequence[str]
) -> list[str]:
    """The pad's prop names by slot index, from the next layer's names in
    placement order (so slot s gets the prop block mode would place there)."""
    order = layer_slot_order(config, layer)
    by_slot = [""] * config.slots_per_layer
    for position, slot in enumerate(order):
        by_slot[slot] = names_in_placement_order[position]
    return by_slot


def sample_layer_episode_set(seed: int) -> str:
    weights = {name: EPISODE_SET_WEIGHTS[name] for name in LAYER_EPISODE_SETS}
    return sample_episode_set(seed, weights)


def build_layer_spec(
    seed: int,
    tower: TowerConfig,
    pad: PadConfig = PadConfig(),
    scatter: ScatterConfig = ScatterConfig(),
    *,
    episode_set: str | None = None,
    placed_layers: int | None = None,
) -> LayerSpec:
    """Decide one place-the-layer episode from its seed.

    Tower height is uniform over 0..layers-1 whole layers. The next layer's
    blocks lie on the pad; every other block not in the tower is scattered
    on the table, clear of the pad, as distractors. The tower is the
    layer-mode tower (`layer_mode_tower`)."""
    tower = layer_mode_tower(tower, pad)
    chosen = episode_set or sample_layer_episode_set(seed)
    if chosen not in LAYER_EPISODE_SETS:
        raise ValueError(
            f"layer mode runs sets {LAYER_EPISODE_SETS} (F with --chain); got {chosen!r}"
        )
    _check_pad_clear_of_tower(tower, pad)
    if placed_layers is None:
        placed_layers = _rng("tower-layers", seed).randint(0, tower.layers - 1)
    if not 0 <= placed_layers < tower.layers:
        raise ValueError(f"placed_layers {placed_layers} leaves no layer to place")
    n = tower.slots_per_layer
    placed_count = placed_layers * n
    drifted = chosen == "B"
    names = prop_order(tower, seed)
    pad_names = _pad_names_by_slot(tower, placed_layers, names[placed_count:placed_count + n])
    pad_poses = pad_layer_poses(tower, pad, seed, placed_layers)
    targets = next_layer_targets(tower, seed, placed_layers, drifted)
    loose_names = names[placed_count + n:]
    loose = (
        sample_loose_poses(
            tower, scatter, seed, loose_names, keep_clear=pad_poses, require_pick=False
        )
        if loose_names else {}
    )
    return LayerSpec(
        seed=int(seed),
        episode_set=chosen,
        placed_layers=placed_layers,
        placed_count=placed_count,
        tower_poses=tower_poses(tower, seed, placed_count, drifted, names),
        pad_poses=dict(zip(pad_names, pad_poses)),
        layer_targets=dict(zip(pad_names, targets)),
        loose_poses=loose,
        perturbed_start=chosen == "C",
    )


def chain_layer_spec(
    seed: int,
    tower: TowerConfig,
    pad: PadConfig,
    tower_names: Sequence[str],
    loose: Mapping[str, Pose],
) -> LayerSpec | None:
    """The next step of a chained layer build (set F): the tower built so far
    (prop names in placement order) stays as it stands, the next layer is laid
    on the pad from the loose blocks (in prop-name order), and the other loose
    blocks stay where they lie. None if fewer than a layer's worth of loose
    blocks are left, or one of the rest lies on the pad."""
    tower = layer_mode_tower(tower, pad)
    n = tower.slots_per_layer
    placed_count = len(tower_names)
    if placed_count % n:
        raise ValueError("a chained layer build only ever has whole layers")
    layer = placed_count // n
    if layer >= tower.layers:
        raise ValueError("the tower is already complete")
    available = sorted(loose)
    if len(available) < n:
        return None
    pad_poses = pad_layer_poses(tower, pad, seed, layer, key=f"pad:{layer}")
    rest = {name: loose[name] for name in available[n:]}
    if any(
        blocks_overlap(tower, pose, block, 0.0) for pose in pad_poses for block in rest.values()
    ):
        return None
    pad_names = _pad_names_by_slot(tower, layer, available[:n])
    return LayerSpec(
        seed=int(seed),
        episode_set="F",
        placed_layers=layer,
        placed_count=placed_count,
        tower_poses=tower_poses(tower, seed, placed_count, False, tower_names),
        pad_poses=dict(zip(pad_names, pad_poses)),
        layer_targets=dict(zip(pad_names, next_layer_targets(tower, seed, layer, False))),
        loose_poses=rest,
    )


@dataclass(frozen=True)
class PlacementTolerance:
    # Our sim towers stood with random per-block errors up to ~4 mm / 2 deg
    # (Oct 29 test); 2 mm / 2 deg is the success bar and the re-grasp trigger.
    xy_m: float = 0.002
    yaw_deg: float = 2.0
    z_m: float = 0.003
    # How far any other tower block may move during the episode.
    disturbed_xy_m: float = 0.002
    disturbed_z_m: float = 0.003


def pose_error(actual: Pose, target: Pose) -> dict[str, float]:
    return {
        "xy_m": math.dist(actual.position[:2], target.position[:2]),
        "z_m": abs(actual.position[2] - target.position[2]),
        "yaw_deg": yaw_error_deg(actual.yaw_deg, target.yaw_deg),
    }


def placement_within(error: Mapping[str, float], tolerance: PlacementTolerance) -> bool:
    return (
        error["xy_m"] <= tolerance.xy_m
        and error["yaw_deg"] <= tolerance.yaw_deg
        and error["z_m"] <= tolerance.z_m
    )


def disturbed_blocks(
    before: Mapping[str, Pose],
    after: Mapping[str, Pose],
    tolerance: PlacementTolerance,
) -> list[str]:
    """Tower blocks that moved more than tolerance between two readings."""
    moved = []
    for name, start in before.items():
        end = after[name]
        if (
            math.dist(start.position[:2], end.position[:2]) > tolerance.disturbed_xy_m
            or abs(start.position[2] - end.position[2]) > tolerance.disturbed_z_m
        ):
            moved.append(name)
    return moved


def park_position(index: int, origin_xy_m: tuple[float, float], z_m: float) -> Position:
    """A grid of parking spots for blocks not in use, 10 per row."""
    row, column = divmod(index, 10)
    return (origin_xy_m[0] + 0.1 * column, origin_xy_m[1] - 0.05 * row, z_m)


# Light-wood base colour (sRGB 0-1) and how far each block's shade varies.
BLOCK_BASE_COLOR = (0.86, 0.70, 0.50)
BLOCK_SHADE_RANGE = (0.82, 1.08)
BLOCK_HUE_JITTER = 0.03


def block_color(index: int) -> list[float]:
    """A fixed light-wood shade per prop: brightness varies most, with a small
    per-channel drift so neighbouring blocks read as different pieces."""
    rng = _rng("block-color", index)
    shade = rng.uniform(*BLOCK_SHADE_RANGE)
    return [
        round(min(1.0, max(0.0, c * shade + rng.uniform(-BLOCK_HUE_JITTER, BLOCK_HUE_JITTER))), 4)
        for c in BLOCK_BASE_COLOR
    ]


def tower_props(
    config: TowerConfig,
    *,
    park_origin_xy_m: tuple[float, float] = (-1.0, -1.0),
    park_z_m: float = 0.0,
    static_friction: float = 0.5,
    dynamic_friction: float = 0.5,
) -> list[dict[str, Any]]:
    """The `props` entries a sim world needs for this tower, parked off-table
    on a surface at `park_z_m`.

    Each block gets its own `color` instead of the sim's "wood" visual
    material, which binds one colour to every block and hides the seams.
    Size, friction and mass are fixed when the sim boots, so per-episode
    physics randomization has to come from the sim module, not this list."""
    props = []
    for k, name in enumerate(config.block_names()):
        props.append(
            {
                "name": name,
                "type": "cube",
                "size": 1.0,
                "scale": [
                    config.block_length_m,
                    config.block_width_m,
                    config.block_thickness_m,
                ],
                "position": list(
                    park_position(
                        k, park_origin_xy_m, park_z_m + config.block_thickness_m / 2.0
                    )
                ),
                "color": block_color(k),
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
            }
        )
    return props
