"""Layer mode (`place_episode.py --build layer`) against the kinematic fake."""

import asyncio
import math
from dataclasses import replace

import pytest

import jenga_tower as jt
import layer_episode as le
from test_place_episode import FakeSim, _components

TOWER = jt.TowerConfig(layers=3)
PAD = jt.PadConfig()
SCATTER = jt.ScatterConfig()


class LayerSim(FakeSim):
    """A side grip (jaws across the block widths) also carries every block
    between the jaws, as the real G2 does with a layer."""

    def __init__(self, tower):
        super().__init__(tower)
        self.extra = []
        self.bundle_grips = 0

    async def grab(self):
        await super().grab()
        if not self.held:
            return True
        name = self.held[0]
        if abs(jt.wrap_deg(self.tool_yaw - self.props[name].yaw_deg, 180.0)) > 15.0:
            return True  # an end grip: one block only
        self.bundle_grips += 1
        jaw_z = self.tcp[2] + self.grasp_height
        c, s = math.cos(math.radians(-self.tool_yaw)), math.sin(math.radians(-self.tool_yaw))
        for other, pose in self.props.items():
            if other == name or abs(pose.position[2] - jaw_z) > 0.006:
                continue
            if math.dist(pose.position[:2], self.tcp[:2]) < 0.045:
                dx, dy, dz = (pose.position[i] - self.tcp[i] for i in range(3))
                yaw_offset = jt.wrap_deg(self.tool_yaw - pose.yaw_deg, 180.0)
                self.extra.append((other, (c * dx - s * dy, s * dx + c * dy, dz), yaw_offset))
        return True

    def _carry(self):
        super()._carry()
        c, s = math.cos(math.radians(self.tool_yaw)), math.sin(math.radians(self.tool_yaw))
        for name, (tx, ty, tz), yaw_offset in self.extra:
            self.props[name] = jt.Pose(
                (self.tcp[0] + c * tx - s * ty, self.tcp[1] + s * tx + c * ty, self.tcp[2] + tz),
                jt.wrap_deg(self.tool_yaw - yaw_offset, 180.0),
            )

    async def open(self):
        released = [name for name, *_ in self.extra]
        self.extra = []
        await super().open()
        for name in released:
            self._settle(name)


def _place(sim, spec, reset=True):
    world, gripper, motion, arm = _components(sim)
    return asyncio.run(
        le.execute_place_layer(
            world, gripper, motion, arm, spec, jt.layer_mode_tower(TOWER, PAD),
            episode_output_dir="out", reset=reset,
        )
    )


def test_the_pad_layer_follows_the_next_layer_and_stays_clear():
    tower = jt.TowerConfig(layers=12)
    packed = jt.layer_mode_tower(tower, PAD)
    for seed in range(25):
        spec = jt.build_layer_spec(seed, tower, PAD, SCATTER)
        pad, targets = list(spec.pad_poses.values()), list(spec.layer_targets.values())
        assert len(pad) == len(targets) == 3
        for block, target in zip(pad, targets):
            # Same way round as its slot, within the pad's yaw jitter.
            assert abs(jt.yaw_error_deg(block.yaw_deg, target.yaw_deg)) <= PAD.yaw_jitter_deg + 1.5
            assert target.position[2] == pytest.approx(jt.layer_center_z(tower, spec.placed_layers))
        # Packed to the pad seam, and nothing scattered touches the pad.
        assert math.dist(pad[0].position[:2], pad[1].position[:2]) == pytest.approx(
            tower.block_width_m + PAD.seam_m, abs=0.002
        )
        for loose in spec.loose_poses.values():
            assert not any(jt.blocks_overlap(packed, loose, b, SCATTER.tower_clearance_m) for b in pad)
    with pytest.raises(ValueError, match="layer pad"):
        jt.build_layer_spec(0, tower, jt.PadConfig(center_xy_m=tower.base_xy_m))


@pytest.mark.parametrize("placed_layers", [0, 2])
def test_a_layer_lands_in_its_slots_with_one_side_grip(placed_layers):
    sim = LayerSim(TOWER)
    spec = jt.build_layer_spec(3, TOWER, PAD, SCATTER, episode_set="A", placed_layers=placed_layers)
    result = _place(sim, spec)
    assert result["outcome"] == "success", result
    assert sim.bundle_grips == 1
    assert not result["adjustments"]
    for name, target in spec.layer_targets.items():
        assert jt.pose_error(sim.props[name], target)["xy_m"] < 1e-3
    assert sim.stages[:4] == [
        le.STAGE_PICK_LAYER, le.STAGE_CARRY_LAYER, le.STAGE_LOWER_LAYER,
        "lift the gripper away from the tower",
    ]
    assert sim.start_command["task"] == le.LAYER_PROMPT


def test_a_chained_layer_build_finishes_the_tower():
    sim = LayerSim(TOWER)
    spec = replace(
        jt.build_layer_spec(5, TOWER, PAD, SCATTER, episode_set="A", placed_layers=0),
        episode_set="F",
    )
    built, reset = [], True
    while True:
        result = _place(sim, spec, reset=reset)
        assert result["outcome"] == "success", result
        built.extend(spec.pad_poses)
        if len(built) == TOWER.block_count:
            break
        loose = {name: sim.props[name] for name in spec.loose_poses}
        spec, reset = jt.chain_layer_spec(5, TOWER, PAD, built, loose), False
        assert spec is not None
    for k, name in enumerate(built):
        layer = k // TOWER.slots_per_layer
        assert sim.props[name].position[2] == pytest.approx(jt.layer_center_z(TOWER, layer), abs=1e-6)
