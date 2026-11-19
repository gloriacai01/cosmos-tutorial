"""Drive `execute_place` against a small kinematic fake of the sim.

The fake is not physics: a grasped block follows the TCP exactly and a
released block drops straight down onto whatever is under it. That is enough
to check the episode's sequencing, targets, stage marks and success logic.
"""

import asyncio
import json
import math
from types import SimpleNamespace

import pytest

import jenga_tower as jt
import place_episode as pe

TOWER = jt.TowerConfig(layers=3)
SUPPLY = jt.SupplyConfig()


class FakeSim:
    def __init__(self, tower: jt.TowerConfig) -> None:
        self.tower = tower
        self.props: dict[str, jt.Pose] = {
            name: jt.Pose(tuple(p["position"]), 0.0)
            for name, p in zip(tower.block_names(), jt.tower_props(tower))
        }
        self.park = dict(self.props)
        self.sim_time = 0.0
        self.tcp = (0.2, 0.0, 0.3)
        self.tool_yaw = 0.0
        self.held: tuple[str, tuple[float, float, float], float] | None = None
        self.planned: tuple[tuple[float, float, float], float] | None = None
        self.stages: list[str] = []
        self.capturing = False
        self.reset_drift: dict[str, tuple[float, float]] = {}
        # Where the block ends up in the hand relative to where it was aimed:
        # tool-frame x, y (m) and yaw (deg).
        self.grasp_slip = (0.0, 0.0, 0.0)
        self.moves: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []

    # -- world ---------------------------------------------------------
    async def do_command(self, command):
        cmd = command["command"]
        if cmd == "status":
            self.sim_time += 0.5
            return {"sim_time": self.sim_time, "playing": True}
        if cmd in ("pause", "play"):
            return {}
        if cmd == "park_props":
            for name in command["props"]:
                self.props[name] = self.park[name]
            return {}
        if cmd == "set_prop_pose":
            name = command["prop"]
            x, y, z = command["position"]
            dx, dy = self.reset_drift.get(name, (0.0, 0.0))
            self.props[name] = jt.Pose(
                (x + dx, y + dy, z),
                jt.quaternion_wxyz_to_yaw_deg(command["orientation_wxyz"]),
            )
            self._settle(name)
            return {"position": command["position"], "orientation_wxyz": command["orientation_wxyz"]}
        if cmd == "get_prop_pose":
            pose = self.props[command["prop"]]
            return {
                "pose": {
                    "position": list(pose.position),
                    "orientation_wxyz": list(pose.orientation_wxyz),
                }
            }
        if cmd == "start_episode":
            self.capturing = True
            return {}
        if cmd == "mark_episode_stage":
            self.stages.append(command["task"])
            return {}
        if cmd == "stop_episode":
            self.capturing = False
            return {"outcome": command["outcome"]}
        raise AssertionError(f"unexpected world command {cmd}")

    def _support_z(self, name: str, x: float, y: float, z: float) -> float:
        """Top of the highest other block below (x, y, z), else the table."""
        top = 0.0
        thickness = self.tower.block_thickness_m
        for other, pose in self.props.items():
            if other == name or (self.held and other == self.held[0]):
                continue
            below = pose.position[2] < z - thickness / 2
            if below and math.dist(pose.position[:2], (x, y)) < 0.04:
                top = max(top, pose.position[2] + thickness / 2)
        return top

    def _settle(self, name: str) -> None:
        pose = self.props[name]
        if pose.position[0] < -0.5:
            return  # parked
        x, y, z = pose.position
        rest = self._support_z(name, x, y, z) + self.tower.block_thickness_m / 2
        self.props[name] = jt.Pose((x, y, rest), pose.yaw_deg)

    # -- motion / arm --------------------------------------------------
    name = "builtin"

    async def plan(self, command):
        request = json.loads(command["plan"])
        pose = request["destination"]["pose"]
        target = (pose.get("x", 0.0) / 1000, pose.get("y", 0.0) / 1000, pose.get("z", 0.0) / 1000)
        self.planned = (target, -pose.get("theta", 0.0))
        return {"plan": [{"pick-arm": [0.0] * 6}, {"pick-arm": [0.1] * 6}]}

    async def move(self, _component, destination, constraints=None):
        pose = destination.pose
        self.tcp = (pose.x / 1000, pose.y / 1000, pose.z / 1000)
        self.tool_yaw = -pose.theta
        return True

    async def get_pose(self, _component, _frame):
        from viam.proto.common import Pose, PoseInFrame

        x, y, z = (v * 1000 for v in self.tcp)
        return PoseInFrame(
            reference_frame="world",
            pose=Pose(x=x, y=y, z=z, o_z=-1.0, theta=-self.tool_yaw),
        )

    async def arm_command(self, command):
        assert command["command"] == "dynamics_metadata"
        return {"max_velocity_rad_s": [3.0] * 6, "max_acceleration_rad_s2": [10.0] * 6}

    async def stream(self, batches):
        async for _ in batches:
            pass
        assert self.planned is not None
        start = self.tcp
        self.tcp, self.tool_yaw = self.planned
        self.moves.append((start, self.tcp))
        self._carry()
        yield SimpleNamespace()

    def _carry(self) -> None:
        """A held block moves rigidly with the tool."""
        if not self.held:
            return
        name, (tx, ty, tz), yaw_offset = self.held
        c, s = math.cos(math.radians(self.tool_yaw)), math.sin(math.radians(self.tool_yaw))
        self.props[name] = jt.Pose(
            (self.tcp[0] + c * tx - s * ty, self.tcp[1] + s * tx + c * ty, self.tcp[2] + tz),
            jt.wrap_deg(self.tool_yaw - yaw_offset, 180.0),
        )

    # -- gripper -------------------------------------------------------
    async def grab(self):
        for name, pose in self.props.items():
            if math.dist(pose.position[:2], self.tcp[:2]) < 0.006 and abs(pose.position[2] - self.tcp[2]) < 0.006:
                dx, dy, dz = (pose.position[i] - self.tcp[i] for i in range(3))
                c = math.cos(math.radians(-self.tool_yaw))
                s = math.sin(math.radians(-self.tool_yaw))
                slip_x, slip_y, slip_yaw = self.grasp_slip
                offset = (c * dx - s * dy + slip_x, s * dx + c * dy + slip_y, dz)
                yaw_offset = jt.wrap_deg(self.tool_yaw - pose.yaw_deg - slip_yaw, 180.0)
                self.held = (name, offset, yaw_offset)
                self._carry()
                return True
        return True

    async def open(self):
        if self.held:
            name = self.held[0]
            self.held = None
            self._settle(name)

    async def is_holding_something(self):
        holding = self.held is not None
        return SimpleNamespace(is_holding_something=holding, meta={"firm_grasp": holding})

    async def is_moving(self):
        raise NotImplementedError


def _components(sim: FakeSim):
    world = SimpleNamespace(do_command=sim.do_command)
    motion = SimpleNamespace(
        name="builtin", do_command=sim.plan, move=sim.move, get_pose=sim.get_pose
    )

    async def move_to_joint_positions(_positions):
        return None

    arm = SimpleNamespace(
        do_command=sim.arm_command,
        move_through_joint_positions_streamed=sim.stream,
        move_to_joint_positions=move_to_joint_positions,
    )
    gripper = SimpleNamespace(
        grab=sim.grab, open=sim.open,
        is_holding_something=sim.is_holding_something, is_moving=sim.is_moving,
    )
    return world, gripper, motion, arm


def _run(sim, spec, **kwargs):
    world, gripper, motion, arm = _components(sim)
    return asyncio.run(
        pe.execute_place(world, gripper, motion, arm, spec, TOWER, episode_output_dir="out", **kwargs)
    )


@pytest.mark.parametrize("placed", [0, 2, 3, 7])
def test_clean_placement_lands_in_its_slot(placed):
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(1, TOWER, SUPPLY, episode_set="A", placed_count=placed)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["placement_error"]["xy_m"] < 1e-6
    assert result["placement_error"]["z_m"] < 1e-6
    assert result["disturbed_blocks"] == []
    assert result["phases"] == [pe.STAGE_PICK, pe.STAGE_CARRY, pe.STAGE_LOWER, pe.STAGE_RETREAT, pe.STAGE_HOLD]
    # Capture records one base stage, like the pick oracle.
    assert sim.stages == ["base"]
    assert not sim.capturing


@pytest.mark.parametrize("episode_set", ["A", "C", "D", "E"])
def test_sideways_moves_only_happen_above_the_tower(episode_set):
    """The tower is not an obstacle in the planner's world, so any move that
    travels sideways must start and end above the tower top."""
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(8, TOWER, SUPPLY, episode_set=episode_set, placed_count=7)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    safe_z = jt.tower_top_z(TOWER, spec.placed_count) + TOWER.block_thickness_m
    for start, end in sim.moves:
        if math.dist(start[:2], end[:2]) > 0.01:
            assert min(start[2], end[2]) >= safe_z, (start, end)


def test_release_corrects_for_where_the_block_sits_in_the_hand():
    sim = FakeSim(TOWER)
    sim.grasp_slip = (0.004, -0.002, 3.0)
    spec = jt.build_episode_spec(1, TOWER, SUPPLY, episode_set="A", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["adjustments"] == []
    assert result["in_hand_offset"]["x_m"] == pytest.approx(0.004)
    assert result["in_hand_offset"]["yaw_deg"] == pytest.approx(3.0)


def test_side_grasp_also_places():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(4, TOWER, SUPPLY, episode_set="B", placed_count=5)
    result = _run(sim, spec, grasp=pe.GraspConfig(axis="side"))
    assert result["outcome"] == "success", result


def test_set_d_straightens_the_misplaced_block_first():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(9, TOWER, SUPPLY, episode_set="D", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["phases"][0] == pe.STAGE_STRAIGHTEN
    straightened = result["adjustments"][0]
    assert straightened["block"] == spec.misplaced_block
    assert straightened["xy_m"] < 1e-6 and straightened["yaw_deg"] < 1e-6


def test_set_e_starts_hovering_beside_the_block_before_capture():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(3, TOWER, SUPPLY, episode_set="E", placed_count=1)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert "failed_grasp_start" in result["timing"]["phases_sec"]


def test_set_c_perturbs_before_capture_and_still_lands():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(6, TOWER, SUPPLY, episode_set="C", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert "perturbed_start" in result["timing"]["phases_sec"]
    labels = [move["label"] for move in result["timing"]["moves"]]
    assert "pre-release" in labels and "release" in labels


def test_unstable_reset_skips_capture():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(2, TOWER, SUPPLY, episode_set="A", placed_count=3)
    sim.reset_drift[TOWER.block_name(1)] = (0.01, 0.0)
    result = _run(sim, spec)
    assert result["outcome"] == "reset_unstable"
    assert result["reset"]["unstable_blocks"] == [TOWER.block_name(1)]
    assert sim.stages == [] and not sim.capturing


def test_chained_step_keeps_the_tower_and_moves_only_the_next_block():
    sim = FakeSim(TOWER)
    first = jt.build_episode_spec(5, TOWER, SUPPLY, episode_set="F", placed_count=0)
    assert _run(sim, first)["outcome"] == "success"
    placed_block = dict(sim.props)[TOWER.block_name(0)]
    second = jt.build_episode_spec(5, TOWER, SUPPLY, episode_set="F", placed_count=1)
    result = _run(sim, second, reset=False)
    assert result["outcome"] == "success", result
    assert "reset" not in result
    assert sim.props[TOWER.block_name(0)] == placed_block
