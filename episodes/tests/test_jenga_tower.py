import math
from collections import Counter

import pytest

import jenga_tower as jt


TOWER = jt.TowerConfig()
SCATTER = jt.ScatterConfig()


def test_block_names_follow_placement_order():
    assert TOWER.block_count == 30  # 10 layers of 3
    assert TOWER.block_name(0) == "jenga_block_00"
    assert TOWER.block_name(29) == "jenga_block_29"
    with pytest.raises(ValueError):
        TOWER.block_name(30)


def test_slot_ref_fills_layers_in_slot_order():
    config = jt.TowerConfig(slot_order=(2, 0, 1))
    refs = [jt.slot_ref(config, k) for k in range(6)]
    assert [(r.layer, r.slot) for r in refs] == [
        (0, 2), (0, 0), (0, 1), (1, 2), (1, 0), (1, 1),
    ]


def test_default_fill_order_is_middle_then_near_side_then_far_side():
    # Default spot (0.30, 0.10), arm at the origin: slot 0 is nearer in both
    # layer directions. With the tower on the other side of the arm, even
    # layers' near side flips.
    assert [jt.layer_slot_order(TOWER, layer) for layer in (0, 1)] == [(1, 0, 2), (1, 0, 2)]
    other_side = jt.TowerConfig(base_xy_m=(0.30, -0.10))
    assert [jt.layer_slot_order(other_side, layer) for layer in (0, 1)] == [(1, 2, 0), (1, 0, 2)]
    for base, yaw, arm in [
        ((0.30, 0.10), 0.0, (0.0, 0.0)),
        ((0.30, -0.10), 0.0, (0.0, 0.0)),
        ((0.25, 0.20), 35.0, (0.0, 0.0)),
        ((0.30, 0.10), 0.0, (0.50, 0.40)),
        ((-0.30, 0.10), 0.0, (0.0, 0.0)),
    ]:
        config = jt.TowerConfig(base_xy_m=base, base_yaw_deg=yaw, arm_xy_m=arm)
        for layer in (0, 1, 4, 9):
            refs = [jt.slot_ref(config, layer * 3 + i) for i in range(3)]
            assert {r.layer for r in refs} == {layer}
            assert refs[0].slot == 1
            near, far = (
                math.dist(jt.nominal_slot_pose(config, r).position[:2], arm) for r in refs[1:]
            )
            assert near < far, (base, yaw, arm, layer)


def test_slot_order_must_be_a_permutation():
    with pytest.raises(ValueError):
        jt.TowerConfig(slot_order=(0, 0, 1))


def test_slots_are_named_from_the_arm_and_prompts_round_trip():
    # Arm at the origin looking toward (0.30, 0.10): even layers run across
    # that line (slot 0 on the robot's right), odd layers along it.
    assert jt.slot_names(TOWER, 0) == ("right", "middle", "left")
    assert jt.slot_names(TOWER, 1) == ("near", "middle", "far")
    turned = jt.TowerConfig(base_yaw_deg=90.0)
    assert jt.slot_names(turned, 0) == ("far", "middle", "near")
    assert jt.slot_names(turned, 1) == ("right", "middle", "left")
    ref = jt.slot_ref(TOWER, 4)  # layer 2, filled second: the near side
    assert jt.slot_prompt(TOWER, ref) == "place the block in the near slot on layer 2"
    assert jt.slot_stage_prompts(TOWER, ref)["lower"] == (
        "lower the block into the near slot on layer 2 and open the gripper"
    )
    for k in range(TOWER.block_count):
        ref = jt.slot_ref(TOWER, k)
        assert jt.slot_ref_named(TOWER, jt.layer_number(ref), jt.slot_name(TOWER, ref)) == ref
    with pytest.raises(ValueError):
        jt.slot_ref_named(TOWER, 2, "left")  # layer 2 runs near/far
    # The rec-sim twin builds every layer middle first, then right or near.
    twin = jt.TowerConfig(base_xy_m=(0.327, 0.011), layers=3)
    assert [jt.slot_name(twin, jt.slot_ref(twin, k)) for k in range(9)] == [
        "middle", "right", "left", "middle", "near", "far", "middle", "right", "left",
    ]


def test_layers_alternate_ninety_degrees_and_stack_by_thickness():
    # Each layer's first placement is its middle slot.
    even = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 0))
    odd = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 3))
    assert jt.yaw_error_deg(odd.yaw_deg, even.yaw_deg + 90.0) == pytest.approx(0.0)
    assert even.position[2] == pytest.approx(jt.BLOCK_THICKNESS_M / 2)
    assert odd.position[2] == pytest.approx(1.5 * jt.BLOCK_THICKNESS_M)
    # Middle slots sit on the tower axis.
    assert even.position[:2] == pytest.approx(TOWER.base_xy_m)
    assert odd.position[:2] == pytest.approx(TOWER.base_xy_m)


def test_adjacent_slots_are_one_block_width_plus_the_seam_apart():
    a = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 0))
    b = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 1))
    assert TOWER.slot_gap_m > 0.0
    assert math.dist(a.position, b.position) == pytest.approx(
        jt.BLOCK_WIDTH_M + TOWER.slot_gap_m
    )


def test_tower_yaw_rotates_the_whole_layout():
    turned = jt.TowerConfig(base_yaw_deg=90.0)
    plain = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 0))
    rotated = jt.nominal_slot_pose(turned, jt.slot_ref(turned, 0))
    dx, dy = (plain.position[i] - TOWER.base_xy_m[i] for i in range(2))
    rx, ry = (rotated.position[i] - turned.base_xy_m[i] for i in range(2))
    assert (rx, ry) == pytest.approx((-dy, dx))


def test_clean_tower_has_only_seam_jitter():
    """No layer drift, but every block is a fraction of a millimetre off so
    seams show between blocks."""
    poses = jt.tower_poses(TOWER, seed=3, placed_count=27, drifted=False)
    drift = TOWER.drift
    errors = []
    for k, pose in enumerate(poses.values()):
        nominal = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, k))
        error = jt.pose_error(pose, nominal)
        assert error["z_m"] == 0.0
        assert error["xy_m"] < 5 * drift.seam_jitter_xy_m
        assert error["yaw_deg"] < 5 * drift.seam_jitter_yaw_deg
        errors.append(error["xy_m"])
    assert max(errors) > 0.0


def test_props_are_shuffled_per_seed_and_cover_the_tower():
    order = jt.prop_order(TOWER, 4)
    assert sorted(order) == TOWER.block_names()
    assert order == jt.prop_order(TOWER, 4)
    assert order != jt.prop_order(TOWER, 5)
    assert order != TOWER.block_names()
    poses = jt.tower_poses(TOWER, seed=4, placed_count=10, drifted=False)
    assert list(poses) == order[:10]


def test_drift_is_bounded_reproducible_and_correlated():
    offsets = jt.layer_offsets(TOWER, seed=11, drifted=True)
    assert offsets == jt.layer_offsets(TOWER, seed=11, drifted=True)
    assert offsets != jt.layer_offsets(TOWER, seed=12, drifted=True)
    drift = TOWER.drift
    for (x, y), yaw in offsets:
        assert math.hypot(x, y) <= drift.max_layer_offset_xy_m + 1e-12
        assert abs(yaw) <= drift.max_layer_offset_yaw_deg + 1e-12
    # A random walk: consecutive layers differ by far less than the clip.
    steps = [
        math.dist(offsets[i][0], offsets[i + 1][0]) for i in range(len(offsets) - 1)
    ]
    assert max(steps) < drift.max_layer_offset_xy_m


def _at(x, y, yaw=0.0, z=jt.BLOCK_THICKNESS_M / 2):
    return jt.Pose((x, y, z), yaw)


@pytest.mark.parametrize("placed", [0, 20, 29])
def test_every_loose_block_lies_flat_apart_in_the_region_and_clear_of_the_tower(placed):
    for seed in range(15):
        spec = jt.build_episode_spec(seed, TOWER, SCATTER, episode_set="A", placed_count=placed)
        assert list(spec.loose_poses) == jt.prop_order(TOWER, seed)[placed:]
        poses = list(spec.loose_poses.values())
        for i, pose in enumerate(poses):
            for x, y in jt.block_corners_xy(TOWER, pose):
                assert SCATTER.region_min_xy_m[0] - 1e-9 <= x <= SCATTER.region_max_xy_m[0] + 1e-9
                assert SCATTER.region_min_xy_m[1] - 1e-9 <= y <= SCATTER.region_max_xy_m[1] + 1e-9
            assert pose.position[2] == pytest.approx(jt.BLOCK_THICKNESS_M / 2)
            reach = jt.tower_footprint_radius(TOWER) + SCATTER.tower_clearance_m
            assert all(math.dist(c, TOWER.base_xy_m) >= reach for c in jt.block_corners_xy(TOWER, pose))
            for other in poses[i + 1:]:
                assert not jt.blocks_overlap(TOWER, pose, other, SCATTER.min_gap_m - 1e-9)


def test_pick_rule_takes_the_nearest_flat_block_with_both_ends_clear():
    tower = jt.TowerConfig(base_xy_m=(0.3, 0.1))
    loose = {
        "too_close": _at(0.42, 0.1),  # 12 cm from the tower centre
        "end_blocked": _at(0.3, -0.04),  # 14 cm, but a block lies across its +x end
        "blocker": _at(0.36, -0.04, 90.0),  # 15.2 cm, ends clear
        "on_its_side": _at(0.3, 0.235, z=jt.BLOCK_WIDTH_M / 2),  # 13.5 cm, not flat
        "clear": _at(0.155, 0.1),  # 14.5 cm, ends clear
        "farther": _at(0.3, 0.3),  # 20 cm
    }
    assert jt.choose_pick_block(tower, SCATTER, loose) == "clear"
    del loose["clear"]
    assert jt.choose_pick_block(tower, SCATTER, loose) == "blocker"
    assert jt.choose_pick_block(
        tower, SCATTER, {k: loose[k] for k in ("too_close", "on_its_side")}
    ) is None


def test_finger_zones_sit_beyond_each_end_of_the_block():
    zones = jt.finger_zones(TOWER, SCATTER, _at(0.4, 0.0, 90.0))
    ys = sorted(sum(y for _x, y in zone) / 4 for zone in zones)
    reach = jt.BLOCK_LENGTH_M / 2 + SCATTER.finger_clear_length_m / 2
    assert ys == pytest.approx([-reach, reach])


def test_spec_picks_its_next_block_by_the_rule():
    for seed in range(20):
        spec = jt.build_episode_spec(seed, TOWER, SCATTER, episode_set="A")
        assert spec.next_block == jt.choose_pick_block(TOWER, SCATTER, spec.loose_poses)
        assert spec.pick_pose == spec.loose_poses[spec.next_block]
        assert math.dist(spec.pick_pose.position[:2], TOWER.base_xy_m) >= SCATTER.min_pick_distance_m


def test_a_region_too_small_for_the_loose_blocks_is_an_error():
    cramped = jt.ScatterConfig(region_min_xy_m=(0.40, -0.10), region_max_xy_m=(0.55, 0.10))
    with pytest.raises(ValueError):
        jt.build_episode_spec(0, TOWER, cramped, episode_set="A", placed_count=0)


def test_chain_step_builds_on_the_tower_and_picks_from_where_blocks_lie():
    first = jt.build_episode_spec(4, TOWER, SCATTER, episode_set="F", placed_count=0)
    built = [first.next_block]
    loose = {n: p for n, p in first.loose_poses.items() if n != first.next_block}
    step = jt.chain_step_spec(4, TOWER, SCATTER, built, loose)
    assert step.placed_count == 1 and list(step.tower_poses) == built
    assert step.next_block == jt.choose_pick_block(TOWER, SCATTER, loose)
    assert step.loose_poses == loose  # nothing is moved for the next step
    assert step.next_slot == jt.slot_ref(TOWER, 1)
    assert jt.chain_step_spec(4, TOWER, SCATTER, built, {}) is None


def test_loose_blocks_lays_only_that_many_and_chains_lay_a_fresh_one():
    for seed in range(10):
        spec = jt.build_episode_spec(seed, TOWER, SCATTER, episode_set="A", loose_blocks=1)
        assert list(spec.loose_poses) == [spec.next_block]
        assert spec.next_block == jt.prop_order(TOWER, seed)[spec.placed_count]
    first = jt.build_episode_spec(4, TOWER, SCATTER, episode_set="F", placed_count=0, loose_blocks=1)
    built, picks = [first.next_block], [first.pick_pose]
    for _ in range(3):
        step = jt.chain_step_spec(4, TOWER, SCATTER, built, {}, loose_blocks=1)
        assert list(step.fresh_poses) == list(step.loose_poses) == [step.next_block]
        assert step.next_block not in built
        built.append(step.next_block)
        picks.append(step.pick_pose)
    assert len({p.position for p in picks}) == len(picks)  # a new spot each step


def test_episode_sets_follow_weights():
    counts = Counter(jt.sample_episode_set(s, jt.EPISODE_SET_WEIGHTS) for s in range(5000))
    assert "F" not in counts
    for name, weight in jt.EPISODE_SET_WEIGHTS.items():
        assert counts[name] / 5000 == pytest.approx(weight, abs=0.03)


def test_set_f_is_not_sampled_per_seed():
    with pytest.raises(ValueError):
        jt.sample_episode_set(0, {"A": 0.5, "F": 0.5})


def test_tower_height_is_uniform_over_slots():
    heights = Counter(jt.sample_tower_height(TOWER, s, "A") for s in range(5400))
    assert jt.build_episode_spec(3, TOWER, SCATTER, episode_set="A").placed_count == (
        jt.sample_tower_height(TOWER, 3, "A")
    )
    assert set(heights) == set(range(TOWER.block_count))
    assert max(heights.values()) < 3 * min(heights.values())


def test_spec_is_reproducible():
    assert jt.build_episode_spec(7, TOWER, SCATTER) == jt.build_episode_spec(7, TOWER, SCATTER)


def test_next_block_and_slot_match_placed_count():
    spec = jt.build_episode_spec(5, TOWER, SCATTER, episode_set="A", placed_count=4)
    order = jt.prop_order(TOWER, 5)
    assert list(spec.tower_poses) == order[:4]
    assert spec.next_block in order[4:]
    assert spec.next_block not in spec.tower_poses
    # Layer 1's second block: the side nearer the arm (slot 0 at the default spot).
    assert (spec.next_slot.layer, spec.next_slot.slot) == (1, 0)


def test_set_d_misplaces_the_last_block_within_range():
    sampling = jt.EpisodeSamplingConfig()
    for seed in range(100):
        spec = jt.build_episode_spec(seed, TOWER, SCATTER, episode_set="D")
        assert spec.placed_count >= 1
        assert spec.misplaced_block == list(spec.tower_poses)[-1]
        error = jt.pose_error(spec.misplaced_pose, spec.tower_poses[spec.misplaced_block])
        lo, hi = sampling.misplace_xy_range_m
        assert lo - 1e-12 <= error["xy_m"] <= hi + 1e-12
        lo, hi = sampling.misplace_yaw_range_deg
        assert lo - 1e-9 <= error["yaw_deg"] <= hi + 1e-9
        assert error["z_m"] == 0.0


def test_set_d_rejects_empty_tower():
    with pytest.raises(ValueError):
        jt.build_episode_spec(0, TOWER, SCATTER, episode_set="D", placed_count=0)


def test_set_flags():
    c = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="C")
    e = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="E")
    a = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A")
    assert c.perturbed_start and not a.perturbed_start
    assert e.failed_grasp_aim_xy_m is not None and a.failed_grasp_aim_xy_m is None
    offset = math.dist(e.failed_grasp_aim_xy_m, e.pick_pose.position[:2])
    assert 0.01 <= offset <= 0.03


def test_new_layer_targets_the_layer_below():
    seed, placed = 2, 6  # block 6 starts layer 2
    offsets = jt.layer_offsets(TOWER, seed, drifted=True)
    target = jt.next_slot_target(TOWER, seed, placed, drifted=True)
    expected = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, placed), *offsets[1])
    assert target == expected


def test_yaw_error_treats_half_turn_as_identical():
    assert jt.yaw_error_deg(179.5, -0.5) == pytest.approx(0.0)
    assert jt.yaw_error_deg(3.0, 0.0) == pytest.approx(3.0)


def test_placement_tolerance_and_disturbance():
    tol = jt.PlacementTolerance()
    target = jt.Pose((0.3, 0.1, 0.0075), 0.0)
    assert jt.placement_within(jt.pose_error(jt.Pose((0.3005, 0.1, 0.0075), 0.5), target), tol)
    assert not jt.placement_within(jt.pose_error(jt.Pose((0.302, 0.1, 0.0075), 0.0), target), tol)
    before = {"a": target, "b": target}
    after = {"a": target, "b": jt.Pose((0.3, 0.1, 0.0), 0.0)}
    assert jt.disturbed_blocks(before, after, tol) == ["b"]


def test_quaternion_round_trip():
    for yaw in (-170.0, -45.0, 0.0, 30.0, 120.0):
        assert jt.quaternion_wxyz_to_yaw_deg(jt.yaw_to_quaternion_wxyz(yaw)) == pytest.approx(yaw)


def test_tower_props_cover_every_block_and_are_parked_apart():
    props = jt.tower_props(TOWER)
    assert [p["name"] for p in props] == TOWER.block_names()
    assert props[0]["scale"] == [0.075, 0.025, 0.015]
    positions = {tuple(p["position"]) for p in props}
    assert len(positions) == len(props)


def test_scaled_tower_keeps_square_layers_and_scales_props():
    small = jt.TowerConfig(**jt.scaled_block_dims(0.416))
    assert small.block_length_m == pytest.approx(3 * small.block_width_m)
    a = jt.nominal_slot_pose(small, jt.slot_ref(small, 0))
    b = jt.nominal_slot_pose(small, jt.slot_ref(small, 1))
    assert math.dist(a.position, b.position) == pytest.approx(
        small.block_width_m + small.slot_gap_m
    )
    assert jt.tower_props(small)[0]["scale"] == pytest.approx([0.0312, 0.0104, 0.00624])
    spec = jt.build_episode_spec(2, small, SCATTER, episode_set="A", placed_count=20)
    assert len(spec.loose_poses) == small.block_count - 20
    with pytest.raises(ValueError):
        jt.scaled_block_dims(0.0)


def test_props_have_their_own_wood_shades_and_park_on_the_given_surface():
    props = jt.tower_props(TOWER, park_z_m=-0.04)
    colors = [tuple(p["color"]) for p in props]
    assert len(set(colors)) == len(colors)
    for color in colors:
        assert all(0.0 <= c <= 1.0 for c in color)
        assert color[0] > color[1] > color[2]  # warm wood, red > green > blue
    assert all("visual_material" not in p for p in props)
    assert props[0]["position"][2] == pytest.approx(-0.04 + jt.BLOCK_THICKNESS_M / 2)


def test_defaults_match_our_850_tower():
    assert TOWER.layers == 10 and TOWER.block_count == 30
    assert TOWER.slot_gap_m == pytest.approx(0.003)
    tolerance = jt.PlacementTolerance()
    assert tolerance.xy_m == pytest.approx(0.002) and tolerance.yaw_deg == pytest.approx(2.0)


def test_blocks_overlap_on_their_footprints():
    a = jt.Pose((0.0, 0.0, 0.0075), 0.0)
    pitch = TOWER.block_width_m + TOWER.slot_gap_m
    assert not jt.blocks_overlap(TOWER, a, jt.Pose((0.0, pitch, 0.0075), 0.0))
    assert jt.blocks_overlap(TOWER, a, jt.Pose((0.0, 0.024, 0.0075), 0.0))
    # A 5 deg turn swings the ends ~3.3 mm toward a neighbour 3 mm away.
    assert jt.blocks_overlap(TOWER, a, jt.Pose((0.0, pitch, 0.0075), 5.0))
    assert jt.blocks_overlap(TOWER, a, jt.Pose((0.0, pitch, 0.0075), 0.0), margin_m=0.0031)


def _same_layer_pairs(spec):
    poses = dict(spec.tower_poses)
    if spec.misplaced_block is not None:
        poses[spec.misplaced_block] = spec.misplaced_pose
    names = list(spec.tower_poses)
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            if jt.slot_ref(TOWER, i).layer == jt.slot_ref(TOWER, j).layer:
                yield poses[a], poses[names[j]]


@pytest.mark.parametrize("episode_set", ["A", "B", "D"])
def test_generated_towers_never_start_overlapping(episode_set):
    for seed in range(300):
        spec = jt.build_episode_spec(seed, TOWER, SCATTER, episode_set=episode_set)
        for a, b in _same_layer_pairs(spec):
            assert not jt.blocks_overlap(TOWER, a, b), (seed, a, b)


def test_set_d_still_misplaces_its_block():
    moved = 0
    for seed in range(100):
        spec = jt.build_episode_spec(seed, TOWER, SCATTER, episode_set="D")
        error = jt.pose_error(spec.misplaced_pose, spec.tower_poses[spec.misplaced_block])
        assert error["xy_m"] >= 0.002 - 1e-12
        moved += error["yaw_deg"] >= 2.0 - 1e-9
    # Most draws keep a 2-5 deg turn; only blocks boxed in by neighbours lose it.
    assert moved >= 50
