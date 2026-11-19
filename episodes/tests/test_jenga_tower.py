import math
from collections import Counter

import pytest

import jenga_tower as jt


TOWER = jt.TowerConfig()
SUPPLY = jt.SupplyConfig()


def test_block_names_follow_placement_order():
    assert TOWER.block_count == 54
    assert TOWER.block_name(0) == "jenga_block_00"
    assert TOWER.block_name(53) == "jenga_block_53"
    with pytest.raises(ValueError):
        TOWER.block_name(54)


def test_slot_ref_fills_layers_in_slot_order():
    config = jt.TowerConfig(slot_order=(2, 0, 1))
    refs = [jt.slot_ref(config, k) for k in range(6)]
    assert [(r.layer, r.slot) for r in refs] == [
        (0, 2), (0, 0), (0, 1), (1, 2), (1, 0), (1, 1),
    ]


def test_slot_order_must_be_a_permutation():
    with pytest.raises(ValueError):
        jt.TowerConfig(slot_order=(0, 0, 1))


def test_layers_alternate_ninety_degrees_and_stack_by_thickness():
    even = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 1))
    odd = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 4))
    assert jt.yaw_error_deg(odd.yaw_deg, even.yaw_deg + 90.0) == pytest.approx(0.0)
    assert even.position[2] == pytest.approx(jt.BLOCK_THICKNESS_M / 2)
    assert odd.position[2] == pytest.approx(1.5 * jt.BLOCK_THICKNESS_M)
    # Middle slots sit on the tower axis.
    assert even.position[:2] == pytest.approx(TOWER.base_xy_m)
    assert odd.position[:2] == pytest.approx(TOWER.base_xy_m)


def test_adjacent_slots_are_one_block_width_apart():
    a = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 0))
    b = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 1))
    assert math.dist(a.position, b.position) == pytest.approx(jt.BLOCK_WIDTH_M)


def test_tower_yaw_rotates_the_whole_layout():
    turned = jt.TowerConfig(base_yaw_deg=90.0)
    plain = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, 0))
    rotated = jt.nominal_slot_pose(turned, jt.slot_ref(turned, 0))
    dx, dy = (plain.position[i] - TOWER.base_xy_m[i] for i in range(2))
    rx, ry = (rotated.position[i] - turned.base_xy_m[i] for i in range(2))
    assert (rx, ry) == pytest.approx((-dy, dx))


def test_clean_tower_has_no_drift():
    poses = jt.tower_poses(TOWER, seed=3, placed_count=9, drifted=False)
    for k, name in enumerate(TOWER.block_names()[:9]):
        nominal = jt.nominal_slot_pose(TOWER, jt.slot_ref(TOWER, k))
        assert poses[name] == nominal


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


def test_supply_pose_is_inside_region_and_clear_of_tower():
    for seed in range(200):
        pose = jt.sample_supply_pose(TOWER, SUPPLY, seed)
        assert SUPPLY.region_min_xy_m[0] <= pose.position[0] <= SUPPLY.region_max_xy_m[0]
        assert SUPPLY.region_min_xy_m[1] <= pose.position[1] <= SUPPLY.region_max_xy_m[1]
        assert pose.position[2] == pytest.approx(jt.BLOCK_THICKNESS_M / 2)
        assert math.dist(pose.position[:2], TOWER.base_xy_m) > jt.tower_footprint_radius(TOWER)


def test_episode_sets_follow_weights():
    counts = Counter(jt.sample_episode_set(s, jt.EPISODE_SET_WEIGHTS) for s in range(5000))
    assert "F" not in counts
    for name, weight in jt.EPISODE_SET_WEIGHTS.items():
        assert counts[name] / 5000 == pytest.approx(weight, abs=0.03)


def test_set_f_is_not_sampled_per_seed():
    with pytest.raises(ValueError):
        jt.sample_episode_set(0, {"A": 0.5, "F": 0.5})


def test_tower_height_is_uniform_over_slots():
    heights = Counter(
        jt.build_episode_spec(s, TOWER, SUPPLY, episode_set="A").placed_count
        for s in range(5400)
    )
    assert set(heights) == set(range(TOWER.block_count))
    assert max(heights.values()) < 3 * min(heights.values())


def test_spec_is_reproducible():
    assert jt.build_episode_spec(7, TOWER, SUPPLY) == jt.build_episode_spec(7, TOWER, SUPPLY)


def test_next_block_and_slot_match_placed_count():
    spec = jt.build_episode_spec(5, TOWER, SUPPLY, episode_set="A", placed_count=4)
    assert list(spec.tower_poses) == TOWER.block_names()[:4]
    assert spec.next_block == "jenga_block_04"
    assert (spec.next_slot.layer, spec.next_slot.slot) == (1, 1)


def test_set_d_misplaces_the_last_block_within_range():
    sampling = jt.EpisodeSamplingConfig()
    for seed in range(100):
        spec = jt.build_episode_spec(seed, TOWER, SUPPLY, episode_set="D")
        assert spec.placed_count >= 1
        assert spec.misplaced_block == TOWER.block_name(spec.placed_count - 1)
        error = jt.pose_error(spec.misplaced_pose, spec.tower_poses[spec.misplaced_block])
        lo, hi = sampling.misplace_xy_range_m
        assert lo - 1e-12 <= error["xy_m"] <= hi + 1e-12
        lo, hi = sampling.misplace_yaw_range_deg
        assert lo - 1e-9 <= error["yaw_deg"] <= hi + 1e-9
        assert error["z_m"] == 0.0


def test_set_d_rejects_empty_tower():
    with pytest.raises(ValueError):
        jt.build_episode_spec(0, TOWER, SUPPLY, episode_set="D", placed_count=0)


def test_set_flags():
    c = jt.build_episode_spec(1, TOWER, SUPPLY, episode_set="C")
    e = jt.build_episode_spec(1, TOWER, SUPPLY, episode_set="E")
    a = jt.build_episode_spec(1, TOWER, SUPPLY, episode_set="A")
    assert c.perturbed_start and not a.perturbed_start
    assert e.failed_grasp_aim_xy_m is not None and a.failed_grasp_aim_xy_m is None
    offset = math.dist(e.failed_grasp_aim_xy_m, e.supply_pose.position[:2])
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
