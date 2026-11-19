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
`k // slots_per_layer`, slot `slot_order[k % slots_per_layer]`. Which prop fills
which placement is shuffled per seed (`prop_order`), so each block's colour
lands somewhere different in every tower.

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
from dataclasses import dataclass, field
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
    layers: int = 18
    slots_per_layer: int = 3
    # Fill order within a layer, as slot indices (0 = negative side of the
    # layer, slots_per_layer - 1 = positive side). Fixed so identical images
    # never get conflicting labels.
    slot_order: tuple[int, ...] = (0, 1, 2)
    block_length_m: float = BLOCK_LENGTH_M
    block_width_m: float = BLOCK_WIDTH_M
    block_thickness_m: float = BLOCK_THICKNESS_M
    # Centre-to-centre spacing beyond the block width. The default is a real
    # tower's seam; use more (e.g. 0.012) for room for a side-grip finger.
    slot_gap_m: float = 0.0008
    prop_prefix: str = BLOCK_PROP_PREFIX
    drift: DriftConfig = field(default_factory=DriftConfig)

    def __post_init__(self) -> None:
        if self.layers < 1 or self.slots_per_layer < 1:
            raise ValueError("tower needs at least one layer and one slot")
        if sorted(self.slot_order) != list(range(self.slots_per_layer)):
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
class SupplyConfig:
    """Where the next block waits to be picked up, lying flat, in world XY."""

    region_min_xy_m: tuple[float, float] = (0.22, -0.20)
    region_max_xy_m: tuple[float, float] = (0.34, -0.06)
    yaw_range_deg: tuple[float, float] = (-90.0, 90.0)
    support_z_m: float = 0.0
    # The supply block keeps at least this much XY clearance from the tower's
    # footprint so the gripper never brushes the tower while picking.
    tower_clearance_m: float = 0.06


@dataclass(frozen=True)
class ScatterConfig:
    """Blocks not in the tower lie flat at random spots on the table instead
    of being parked out of view. The next block to place is one of them."""

    region_min_xy_m: tuple[float, float] = (0.12, -0.32)
    region_max_xy_m: tuple[float, float] = (0.42, -0.04)
    # Centre-to-centre spacing. None: the block's footprint diagonal plus 6 mm.
    min_separation_m: float | None = None


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
    supply_pose: Pose
    # Set D: the last tower block starts here instead of its tower pose.
    misplaced_block: str | None = None
    misplaced_pose: Pose | None = None
    # Set E: the arm starts hovering empty above here, as after a missed grasp.
    failed_grasp_aim_xy_m: tuple[float, float] | None = None
    # Set C: start from a perturbed arm pose before capture.
    perturbed_start: bool = False
    # Blocks not yet in the tower that lie on the table, by prop name.
    scattered_poses: Mapping[str, Pose] = field(default_factory=dict)

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
            "supply_pose": _pose_json(self.supply_pose),
            "misplaced_block": self.misplaced_block,
            "misplaced_pose": (
                _pose_json(self.misplaced_pose) if self.misplaced_pose else None
            ),
            "failed_grasp_aim_xy_m": (
                list(self.failed_grasp_aim_xy_m) if self.failed_grasp_aim_xy_m else None
            ),
            "perturbed_start": self.perturbed_start,
            "scattered_blocks": len(self.scattered_poses),
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
    return SlotRef(placement_index, layer, config.slot_order[position_in_layer])


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
    for k in range(placed_count):
        ref = slot_ref(config, k)
        (ox, oy), oyaw = offsets[ref.layer]
        pose = nominal_slot_pose(config, ref, (ox, oy), oyaw)
        jx, jy = jitter.gauss(0.0, xy_sd), jitter.gauss(0.0, xy_sd)
        jyaw = jitter.gauss(0.0, yaw_sd)
        pose = Pose(
            (pose.position[0] + jx, pose.position[1] + jy, pose.position[2]),
            wrap_deg(pose.yaw_deg + jyaw, 180.0),
        )
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


def sample_supply_pose(
    tower: TowerConfig, supply: SupplyConfig, seed: int, stream: str = "supply"
) -> Pose:
    """A flat supply-block pose inside the region, clear of the tower."""
    reach = tower_footprint_radius(tower) + supply.tower_clearance_m
    block_radius = math.hypot(tower.block_length_m, tower.block_width_m) / 2.0
    for attempt in range(256):
        rng = _rng(f"{stream}:{attempt}", seed)
        yaw = rng.uniform(*supply.yaw_range_deg)
        c, s = abs(math.cos(math.radians(yaw))), abs(math.sin(math.radians(yaw)))
        half_x = (c * tower.block_length_m + s * tower.block_width_m) / 2.0
        half_y = (s * tower.block_length_m + c * tower.block_width_m) / 2.0
        lo_x, hi_x = supply.region_min_xy_m[0] + half_x, supply.region_max_xy_m[0] - half_x
        lo_y, hi_y = supply.region_min_xy_m[1] + half_y, supply.region_max_xy_m[1] - half_y
        if lo_x > hi_x or lo_y > hi_y:
            raise ValueError("supply region cannot hold a block")
        x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
        if math.dist((x, y), tower.base_xy_m) >= reach + block_radius:
            z = supply.support_z_m + tower.block_thickness_m / 2.0
            return Pose((x, y, z), wrap_deg(yaw, 180.0))
    raise ValueError("could not place the supply block clear of the tower")


def sample_scatter_poses(
    tower: TowerConfig,
    scatter: ScatterConfig,
    seed: int,
    names: Sequence[str],
    occupied_xy: Sequence[tuple[float, float]],
) -> dict[str, Pose]:
    """Random flat poses for `names`, inside the region, apart from each
    other, from every `occupied_xy` point, and clear of the tower. Blocks
    that don't fit after many tries are left out (they stay parked)."""
    rng = _rng("scatter", seed)
    half = math.hypot(tower.block_length_m, tower.block_width_m) / 2.0
    separation = scatter.min_separation_m or (2.0 * half + 0.006)
    tower_reach = tower_footprint_radius(tower) + separation / 2.0
    lo_x, lo_y = scatter.region_min_xy_m[0] + half, scatter.region_min_xy_m[1] + half
    hi_x, hi_y = scatter.region_max_xy_m[0] - half, scatter.region_max_xy_m[1] - half
    if lo_x > hi_x or lo_y > hi_y:
        raise ValueError("scatter region cannot hold a block")
    taken = list(occupied_xy)
    poses: dict[str, Pose] = {}
    z = tower.support_z_m + tower.block_thickness_m / 2.0
    for name in names:
        for _attempt in range(200):
            x, y = rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)
            if math.dist((x, y), tower.base_xy_m) < tower_reach + half:
                continue
            if all(math.dist((x, y), other) >= separation for other in taken):
                poses[name] = Pose((x, y, z), wrap_deg(rng.uniform(-90.0, 90.0), 180.0))
                taken.append((x, y))
                break
    return poses


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
    target: Pose, sampling: EpisodeSamplingConfig, seed: int
) -> Pose:
    rng = _rng("misplace", seed)
    distance = rng.uniform(*sampling.misplace_xy_range_m)
    heading = rng.uniform(0.0, 2.0 * math.pi)
    yaw = rng.uniform(*sampling.misplace_yaw_range_deg) * rng.choice((-1.0, 1.0))
    x, y, z = target.position
    return Pose(
        (x + distance * math.cos(heading), y + distance * math.sin(heading), z),
        wrap_deg(target.yaw_deg + yaw, 180.0),
    )


def build_episode_spec(
    seed: int,
    tower: TowerConfig,
    supply: SupplyConfig,
    sampling: EpisodeSamplingConfig = EpisodeSamplingConfig(),
    *,
    episode_set: str | None = None,
    placed_count: int | None = None,
    scatter: ScatterConfig | None = None,
) -> EpisodeSpec:
    """Decide one episode from its seed.

    Tower height is uniform over 0..block_count-1, so every slot is the next
    slot equally often and the top layers get as much data as the bottom.
    Set D needs a block to misplace, so it never starts from an empty tower."""
    chosen = episode_set or sample_episode_set(seed, sampling.set_weights)
    if chosen not in EPISODE_SETS:
        raise ValueError(f"unknown episode set {chosen!r}")
    if placed_count is None:
        low = 1 if chosen == "D" else 0
        placed_count = _rng("tower-height", seed).randint(low, tower.block_count - 1)
    if not 0 <= placed_count < tower.block_count:
        raise ValueError(f"placed_count {placed_count} leaves no slot to fill")
    if chosen == "D" and placed_count == 0:
        raise ValueError("set D needs at least one placed block")

    # F targets the clean tower: its drift is the oracle's own accumulated error.
    drifted = chosen in ("B", "D")
    names = prop_order(tower, seed)
    poses = tower_poses(tower, seed, placed_count, drifted, names)
    ref = slot_ref(tower, placed_count)
    supply_pose = sample_supply_pose(tower, supply, seed)

    misplaced_block = misplaced_pose = None
    if chosen == "D":
        misplaced_block = names[placed_count - 1]
        misplaced_pose = _misplaced(poses[misplaced_block], sampling, seed)

    failed_aim = None
    if chosen == "E":
        rng = _rng("failed-grasp", seed)
        distance = rng.uniform(*sampling.regrasp_offset_range_m)
        heading = rng.uniform(0.0, 2.0 * math.pi)
        failed_aim = (
            supply_pose.position[0] + distance * math.cos(heading),
            supply_pose.position[1] + distance * math.sin(heading),
        )

    scattered: dict[str, Pose] = {}
    if scatter is not None:
        unused = names[placed_count + 1:]
        avoid = [supply_pose.position[:2]]
        if failed_aim is not None:
            avoid.append(failed_aim)
        scattered = sample_scatter_poses(tower, scatter, seed, unused, avoid)

    return EpisodeSpec(
        seed=int(seed),
        episode_set=chosen,
        placed_count=placed_count,
        tower_poses=poses,
        next_slot=ref,
        next_block=names[placed_count],
        next_slot_pose=next_slot_target(tower, seed, placed_count, drifted),
        supply_pose=supply_pose,
        misplaced_block=misplaced_block,
        misplaced_pose=misplaced_pose,
        failed_grasp_aim_xy_m=failed_aim,
        perturbed_start=chosen == "C",
        scattered_poses=scattered,
    )


@dataclass(frozen=True)
class PlacementTolerance:
    xy_m: float = 0.001
    yaw_deg: float = 1.0
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
