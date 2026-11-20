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
SCATTER = jt.ScatterConfig()


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
        self.start_command: dict | None = None
        self.stop_command: dict | None = None
        self.capturing = False
        self.reset_drift: dict[str, tuple[float, float]] = {}
        # Where the block ends up in the hand relative to where it was aimed:
        # tool-frame x, y (m) and yaw (deg).
        self.grasp_slip = (0.0, 0.0, 0.0)
        # The real sim's firm-grasp flag can miss a held flat block.
        self.report_holding = True
        # Close on nothing, as a missed grasp would.
        self.miss_grasp = False
        # The jaws close on a block whose centre is this far above the TCP.
        self.grasp_height = -pe.GraspConfig().grasp_z_offset_m
        # Make the n-th streamed move raise, as a dropped connection would.
        self.fail_on_move: int | None = None
        self.moves: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        self.nudge_calls: list[tuple[str | None, bool]] = []

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
            self.start_command = dict(command)
            self.capture_started = self.sim_time
            return {}
        if cmd == "set_nudge_paused":
            # (stage being recorded, paused) per call; a nudge is always mid-way.
            self.nudge_calls.append((self.stages[-1] if self.stages else None, command["paused"]))
            return {"nudging": True, "paused": command["paused"], "remaining_s": 0.3}
        if cmd == "mark_episode_stage":
            self.stages.append(command["task"])
            frame = round((self.sim_time - self.capture_started) * 15)
            return {"start_frame": frame, "task": command["task"]}
        if cmd == "stop_episode":
            self.capturing = False
            self.stop_command = dict(command)
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
        return {"plan": [{"arm": [0.0] * 6}, {"arm": [0.1] * 6}]}

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
        async for batch in batches:
            # The arm takes the trajectory's planned time on the sim clock.
            self.sim_time += batch[-1].time.total_seconds()
        if self.fail_on_move is not None and len(self.moves) >= self.fail_on_move:
            raise RuntimeError("stream dropped")
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
        if self.miss_grasp:
            return True
        for name, pose in self.props.items():
            jaw_z = self.tcp[2] + self.grasp_height
            if math.dist(pose.position[:2], self.tcp[:2]) < 0.006 and abs(pose.position[2] - jaw_z) < 0.006:
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
        holding = self.held is not None and self.report_holding
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
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=placed)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["placement_error"]["xy_m"] < 1e-6
    assert result["placement_error"]["z_m"] < 1e-6
    assert result["disturbed_blocks"] == []
    stages = [pe.STAGE_PICK, pe.STAGE_CARRY, pe.STAGE_LOWER, pe.STAGE_RETREAT, pe.STAGE_HOLD]
    assert result["phases"] == stages
    # Each stage is written into the episode, starting with the first frame.
    assert sim.stages == stages
    assert result["stages"][0] == {"task": pe.STAGE_PICK, "start_frame": 0}
    assert not sim.capturing


@pytest.mark.parametrize("episode_set", ["A", "C", "D", "E"])
def test_sideways_moves_only_happen_above_the_tower(episode_set):
    """The tower is not an obstacle in the planner's world, so any move that
    travels sideways must start and end above the tower top."""
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(8, TOWER, SCATTER, episode_set=episode_set, placed_count=7)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    safe_z = jt.tower_top_z(TOWER, spec.placed_count) + TOWER.block_thickness_m
    for start, end in sim.moves:
        if math.dist(start[:2], end[:2]) > 0.01:
            assert min(start[2], end[2]) >= safe_z, (start, end)


def test_release_corrects_for_where_the_block_sits_in_the_hand():
    sim = FakeSim(TOWER)
    sim.grasp_slip = (0.004, -0.002, 3.0)
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["adjustments"] == []
    assert result["in_hand_offset"]["x_m"] == pytest.approx(0.004)
    # The block's yaw in the tool frame: the slip plus the jaw offset.
    expected = jt.wrap_deg(3.0 - pe.GraspConfig().jaw_yaw_offset_deg, 180.0)
    assert result["in_hand_offset"]["yaw_deg"] == pytest.approx(expected)


def test_held_block_counts_even_when_the_gripper_says_otherwise():
    sim = FakeSim(TOWER)
    sim.report_holding = False
    spec = jt.build_episode_spec(4, TOWER, SCATTER, episode_set="D", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["holding_after_lift"] and not result["gripper_reported_holding"]


def test_side_grasp_also_places():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(4, TOWER, SCATTER, episode_set="B", placed_count=5)
    result = _run(sim, spec, grasp=pe.GraspConfig(axis="side"))
    assert result["outcome"] == "success", result


def test_set_d_straightens_the_misplaced_block_first():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(9, TOWER, SCATTER, episode_set="D", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["phases"][0] == pe.STAGE_STRAIGHTEN
    straightened = result["adjustments"][0]
    assert straightened["block"] == spec.misplaced_block
    assert straightened["xy_m"] < 1e-6 and straightened["yaw_deg"] < 1e-6


def test_set_e_starts_hovering_beside_the_block_before_capture():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(3, TOWER, SCATTER, episode_set="E", placed_count=1)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert "failed_grasp_start" in result["timing"]["phases_sec"]


def test_set_c_perturbs_before_capture_and_still_lands():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(6, TOWER, SCATTER, episode_set="C", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert "perturbed_start" in result["timing"]["phases_sec"]
    labels = [move["label"] for move in result["timing"]["moves"]]
    assert "pre-release" in labels and "release" in labels


def test_unstable_reset_skips_capture():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(2, TOWER, SCATTER, episode_set="A", placed_count=3)
    sim.reset_drift[TOWER.block_name(1)] = (0.01, 0.0)
    result = _run(sim, spec)
    assert result["outcome"] == "reset_unstable"
    assert result["reset"]["unstable_blocks"] == [TOWER.block_name(1)]
    assert sim.stages == [] and not sim.capturing


def test_chained_step_leaves_the_tower_and_the_loose_blocks_where_they_lie():
    sim = FakeSim(TOWER)
    first = jt.build_episode_spec(5, TOWER, SCATTER, episode_set="F", placed_count=0)
    assert _run(sim, first)["outcome"] == "success"
    built = [first.next_block]
    loose_names = [n for n in first.loose_poses if n != first.next_block]
    world = _components(sim)[0]
    loose = asyncio.run(pe.read_loose_poses(world, loose_names))
    second = jt.chain_step_spec(5, TOWER, SCATTER, built, loose)
    before = dict(sim.props)
    result = _run(sim, second, reset=False)
    assert result["outcome"] == "success", result
    assert "reset" not in result
    assert sim.props[first.next_block] == before[first.next_block]
    for name in loose_names:
        if name != second.next_block:
            assert sim.props[name] == before[name]


@pytest.mark.parametrize(
    "episode_set,first", [("A", pe.STAGE_PICK), ("D", pe.STAGE_STRAIGHTEN), ("E", pe.STAGE_PICK)]
)
def test_capture_opens_with_the_first_real_stage(episode_set, first):
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(9, TOWER, SCATTER, episode_set=episode_set, placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert sim.stages[0] == first and "base" not in sim.stages
    assert [s["task"] for s in result["stages"]] == sim.stages


def test_carry_and_retreat_last_one_action_chunk_on_their_own():
    """The sim module holds a stage open until it has 33 frames; these two
    one-move stages are long enough that the arm doesn't wait."""
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=4)
    result = _run(sim, spec)
    planned = {m["label"]: m["planned_sec"] for m in result["timing"]["moves"]}
    chunk_sec = 33 / 15
    assert planned["carry-cross"] >= chunk_sec - 1e-9
    assert planned["retreat"] >= chunk_sec - 1e-9


def test_nudge_and_upload_reach_the_module():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=2)
    nudge = {"amplitude_deg": list(pe.DN_BEST_NUDGE_DEG), "interval_s": 1.2, "duration_s": 0.6}
    upload = {"dataset_id": "ds", "tags": ["isaac-episode"]}
    result = _run(sim, spec, nudge=nudge, upload=upload)
    assert result["outcome"] == "success", result
    assert sim.start_command["nudge"] == nudge and result["nudge"] == nudge
    assert sim.stop_command["upload"] == upload
    # Nudges run in the pick and carry only, off for the final grasp descent
    # and the grab. They stop while the carry is still being recorded,
    # before the lowering is marked, and stay off.
    assert sim.nudge_calls == [
        (pe.STAGE_PICK, False),
        (pe.STAGE_PICK, True),
        (pe.STAGE_PICK, False),
        (pe.STAGE_CARRY, False),
        (pe.STAGE_CARRY, True),
        (pe.STAGE_LOWER, True),
        (pe.STAGE_RETREAT, True),
    ]
    plain = FakeSim(TOWER)
    _run(plain, spec)
    assert "nudge" not in plain.start_command and "upload" not in plain.stop_command
    assert plain.nudge_calls == []


def test_a_nudged_straighten_starts_paused_and_resumes_for_the_pick():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(4, TOWER, SCATTER, episode_set="D", placed_count=4)
    assert spec.misplaced_block is not None
    nudge = {"amplitude_deg": list(pe.DN_BEST_NUDGE_DEG), "interval_s": 1.2, "duration_s": 0.6}
    _run(sim, spec, nudge=nudge)
    assert sim.nudge_calls[:2] == [(pe.STAGE_STRAIGHTEN, True), (pe.STAGE_PICK, False)]
    assert (pe.STAGE_CARRY, True) in sim.nudge_calls


def test_slot_prompts_name_the_slot_and_keep_the_nudge_stages():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=4, loose_blocks=1)
    nudge = {"amplitude_deg": list(pe.DN_BEST_NUDGE_DEG), "interval_s": 1.2, "duration_s": 0.6}
    result = _run(sim, spec, nudge=nudge, slot_prompts=True)
    assert result["outcome"] == "success", result
    prompt = "place the block in the near slot on layer 2"
    assert result["prompt"] == sim.start_command["task"] == prompt
    assert result["slot"] == jt.slot_metadata(TOWER, spec.next_slot)
    assert sim.start_command["target"]["slot"] == result["slot"]
    assert sim.stages == [
        pe.STAGE_PICK,
        "carry the block above the near slot on layer 2",
        "lower the block into the near slot on layer 2 and open the gripper",
        pe.STAGE_RETREAT,
        pe.STAGE_HOLD,
    ]
    # Nudging follows the stage, not its prompt: pick and carry only.
    assert [paused for _stage, paused in sim.nudge_calls] == [False, True, False, False, True, True, True]


def test_about_half_the_episodes_are_nudged_and_reruns_agree():
    def nudge(key, fraction=pe.DN_BEST_NUDGE_FRACTION, amplitude=pe.DN_BEST_NUDGE_DEG):
        return pe.episode_nudge(key, fraction, list(amplitude), 1.2, 0.6)

    nudged = [nudge(f"{seed:06d}") is not None for seed in range(2000)]
    assert 0.45 < sum(nudged) / len(nudged) < 0.55
    assert nudged == [nudge(f"{seed:06d}") is not None for seed in range(2000)]
    assert nudge("000001", fraction=0.0) is None
    assert all(nudge(f"{s}", amplitude=[0.0] * 6) is None for s in range(50))
    picked = next(nudge(f"{s}") for s in range(50) if nudge(f"{s}"))
    assert picked == {"amplitude_deg": list(pe.DN_BEST_NUDGE_DEG), "interval_s": 1.2, "duration_s": 0.6}


def test_the_gripper_is_only_opened_or_closed():
    sim = FakeSim(TOWER)
    calls = []
    world, gripper, motion, arm = _components(sim)

    def record(name, fn):
        async def wrapped(*args, **kwargs):
            calls.append(name)
            return await fn(*args, **kwargs)
        return wrapped

    gripper = SimpleNamespace(
        **{name: record(name, getattr(gripper, name)) for name in vars(gripper)}
    )
    spec = jt.build_episode_spec(9, TOWER, SCATTER, episode_set="D", placed_count=4)
    result = asyncio.run(
        pe.execute_place(world, gripper, motion, arm, spec, TOWER, episode_output_dir="out")
    )
    assert result["outcome"] == "success", result
    commands = {c for c in calls if c not in ("is_holding_something", "is_moving")}
    assert commands == {"open", "grab"}


def test_every_episode_ends_lifted_above_the_tower_holding_still():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    assert result["phases"][-2:] == [pe.STAGE_RETREAT, pe.STAGE_HOLD]
    end_z = jt.tower_top_z(TOWER, spec.placed_count + 1) + pe.GraspConfig().end_lift_m
    assert sim.tcp[2] >= end_z - 1e-9
    assert sim.held is None  # the gripper is open
    assert "end_hold" in result["timing"]["phases_sec"]
    assert [m["label"] for m in result["timing"]["moves"]][-1] == "retreat"


def test_a_missed_grasp_still_ends_lifted_open_and_holding():
    sim = FakeSim(TOWER)
    sim.miss_grasp = True
    spec = jt.build_episode_spec(2, TOWER, SCATTER, episode_set="A", placed_count=3)
    result = _run(sim, spec)
    assert result["outcome"] == "failure" and result["failure"] == "grasp"
    assert result["phases"][-2:] == [pe.STAGE_RETREAT, pe.STAGE_HOLD]
    assert sim.tcp[2] >= jt.tower_top_z(TOWER, spec.placed_count) + pe.GraspConfig().end_lift_m - 1e-9
    assert not sim.capturing


def test_release_turns_the_wrist_the_short_way():
    slot = jt.Pose((0.3, 0.1, 0.05), -88.0)
    lifted = jt.Pose((0.2, -0.1, 0.1), 85.0)
    for pick_yaw in (85.0, -95.0, 175.0, 10.0):
        _x, _y, place_yaw, _ = pe.release_target(slot, lifted, (0.2, -0.1), pick_yaw)
        assert abs(place_yaw - pick_yaw) <= 90.0 + 1e-9
        # The block still lands on the slot's axis (a block looks the same turned 180).
        in_hand = jt.wrap_deg(lifted.yaw_deg - pick_yaw, 180.0)
        assert jt.yaw_error_deg(place_yaw + in_hand, slot.yaw_deg) < 1e-9


def test_release_re_aims_from_a_hover_before_the_final_lower():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(6, TOWER, SCATTER, episode_set="C", placed_count=4)
    result = _run(sim, spec)
    assert result["outcome"] == "success", result
    labels = [m["label"] for m in result["timing"]["moves"]]
    i = labels.index("pre-release")
    assert labels[i:i + 3] == ["pre-release", "re-aim", "release"]


def test_an_error_ends_only_its_episode_and_closes_capture():
    sim = FakeSim(TOWER)
    sim.fail_on_move = 3
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=2)
    world, gripper, motion, arm = _components(sim)
    result = asyncio.run(
        pe.run_episode_safely(world, gripper, motion, arm, spec, TOWER, episode_output_dir="out")
    )
    assert result["outcome"] == "error"
    assert "stream dropped" in result["error"]
    assert not sim.capturing


def test_a_tiny_move_toppra_cannot_time_still_gets_a_trajectory():
    # The Nov 20 straighten lowering TOPP-RA returned nothing for: 0.25 deg.
    import oracle_episode as oe

    start = [0.030750, -0.390441, -0.462837, 0.000125, -0.072492, 1.635386]
    end = [0.030749, -0.393715, -0.463259, 0.000147, -0.069542, 1.635360]
    limits = [math.radians(60.0)] * 6
    points = oe._trajectory_points([{"arm": start}, {"arm": end}], "arm", limits, limits, 0.7333)
    assert points[-1].time.total_seconds() >= 0.7333 - 1e-3
    assert points[-1].positions == pytest.approx([math.degrees(v) for v in end], abs=1e-6)
    assert points[0].positions == pytest.approx([math.degrees(v) for v in start], abs=1e-6)
    for point in points:
        assert max(map(abs, point.constraints.velocities)) <= 60.0
        assert max(map(abs, point.constraints.accelerations)) <= 60.0
    assert points[-1].constraints.velocities == [0.0] * 6


def test_every_episode_records_its_slot_even_without_slot_prompts():
    sim = FakeSim(TOWER)
    spec = jt.build_episode_spec(1, TOWER, SCATTER, episode_set="A", placed_count=4, loose_blocks=1)
    result = _run(sim, spec)
    assert result["prompt"] == sim.start_command["task"] == pe.DEFAULT_PROMPT
    assert result["slot"]["slot_prompt"] == "place the block in the near slot on layer 2"
    assert sim.start_command["target"]["slot"] == result["slot"]
