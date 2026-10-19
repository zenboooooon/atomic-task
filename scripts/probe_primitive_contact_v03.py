"""Check that the source player renders above ground and requires panel contact."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.smoke import _rgb_at
from task_probe import _advance_motion, _route_action, _route_waypoints


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/primitive_contact_v03/probe"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    player = ROOT / "unity/Build/TpsPretrainThree.x86_64"
    manifest = ROOT / "manifests/primitive_contact_v03_layouts.json"
    layout_manifest_sha256(player, manifest)
    log = OUTPUT / "player.log"
    layouts = json.loads(manifest.read_text())["layouts"]
    avoid_layouts = {item["layout_id"]: item for item in layouts
                     if item["task"] == "avoid" and item["split"] == "validation"}
    environment = UnityEnvironment(
        file_name=str(player), worker_id=82, seed=0, no_graphics=False,
        additional_args=["-logFile", str(log),
                         "--tps-task=pretrain", "--tps-split=validation"],
    )
    screenshot = OUTPUT / "avoid_pitch60.png"
    approach_success = False
    avoid_success = False
    camera_captured = False
    initial_counts = {}
    avoid_ids = []
    route_states = {}
    try:
        environment.reset()
        behaviors = {name.split("?", 1)[0].removeprefix("TPS_").lower(): name
                     for name in environment.behavior_specs}
        if set(behaviors) != {"look", "approach", "avoid"}:
            raise AssertionError(behaviors)
        for task, behavior in behaviors.items():
            decisions, terminals = environment.get_steps(behavior)
            initial_counts[task] = len(decisions)
            if len(decisions) != 4 or len(terminals):
                raise AssertionError(f"bad initial {task} arena")
            if task == "avoid":
                avoid_ids = sorted(map(int, decisions.agent_id))
        for step in range(220):
            pending_avoid_actions = {}
            for task, behavior in behaviors.items():
                decisions, terminals = environment.get_steps(behavior)
                if task == "approach" and step < 20 and len(terminals):
                    raise AssertionError("approach terminated without moving")
                if task == "approach" and len(terminals):
                    approach_success |= any(
                        not interrupted and reward > 0.5
                        for interrupted, reward in zip(terminals.interrupted, terminals.reward)
                    )
                if task == "avoid" and len(terminals):
                    for agent_id, interrupted, reward in zip(
                            terminals.agent_id, terminals.interrupted, terminals.reward):
                        if int(agent_id) in route_states:
                            avoid_success |= bool(not interrupted and reward > 0.5)
                            route_states[int(agent_id)]["done"] = True
                if task == "avoid" and step == 20:
                    if not len(decisions):
                        raise AssertionError("no avoid camera observation")
                    rgb = _rgb_at(decisions.obs, 0)
                    image = cv2.cvtColor(np.rint(rgb * 255).clip(0, 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR)
                    if not cv2.imwrite(str(screenshot), image):
                        raise RuntimeError("failed to capture avoid camera")
                    camera_captured = True
                    checkpoint_lines = [line for line in log.read_text(errors="replace").splitlines()
                                        if "TPS_EPISODE " in line and " task=avoid " in line
                                        and " reason=checkpoint ticks=100 " in line]
                    if len(checkpoint_lines) != 4:
                        raise AssertionError("cannot map avoid agents to layouts")
                    for agent_id, line in zip(avoid_ids, checkpoint_lines):
                        layout_id = next(token.split("=", 1)[1] for token in line.split()
                                         if token.startswith("layout="))
                        layout = avoid_layouts[layout_id]
                        route_states[agent_id] = {
                            "layout": layout,
                            "position": np.array((layout["player_x"], layout["player_z"]),
                                                 dtype=np.float64),
                            "velocity": np.zeros(2, dtype=np.float64),
                            "yaw": float(layout["camera_yaw_degrees"]),
                            "waypoints": _route_waypoints(layout)
                                         + [np.array((layout["target_x"], layout["target_z"]))],
                            "waypoint_index": 0,
                            "done": False,
                        }
                actions = np.zeros((len(decisions), 4), dtype=np.float32)
                if task == "avoid" and step < 20:
                    actions[:, 3] = 1.0
                if task == "avoid" and step >= 20:
                    for row, agent_id in enumerate(decisions.agent_id):
                        state = route_states.get(int(agent_id))
                        if state is None or state["done"]:
                            continue
                        action, state["waypoint_index"] = _route_action(
                            state["position"], state["yaw"], state["waypoints"],
                            state["waypoint_index"])
                        actions[row] = action
                        pending_avoid_actions[int(agent_id)] = action
                if task == "approach" and step >= 20:
                    for row in range(len(decisions)):
                        rgb = _rgb_at(decisions.obs, row)
                        red = rgb[..., 0]
                        mask = ((red > 0.5) & (red > 2.0 * rgb[..., 1])
                                & (red > 2.0 * rgb[..., 2]))
                        columns = np.nonzero(mask)[1]
                        error = (math.degrees(math.atan(
                            (float(columns.mean()) - 111.5) /
                            (112.0 / math.tan(math.radians(35.0)))
                        )) if len(columns) else 0.0)
                        actions[row, 1] = 1.0
                        actions[row, 2] = np.clip(error / 12.0, -1.0, 1.0)
                environment.set_actions(behavior, ActionTuple(continuous=actions))
            if approach_success and avoid_success and camera_captured:
                break
            environment.step()
            if step >= 20:
                for agent_id, action in pending_avoid_actions.items():
                    state = route_states[agent_id]
                    state["position"], state["velocity"], state["yaw"] = _advance_motion(
                        state["position"], state["velocity"], state["yaw"],
                        action,
                    )
    finally:
        environment.close()

    success_lines = [line for line in log.read_text(errors="replace").splitlines()
                     if "TPS_EPISODE " in line and " task=approach " in line
                     and " reason=success " in line]
    distances = []
    for line in success_lines:
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        distances.append(float(fields["distance"]))
    if not approach_success or not distances or max(distances) > 0.9:
        raise AssertionError(f"panel contact unverified: successes={approach_success}, distances={distances}")
    avoid_lines = [line for line in log.read_text(errors="replace").splitlines()
                   if "TPS_EPISODE " in line and " task=avoid " in line
                   and " reason=success " in line]
    avoid_distances = []
    for line in avoid_lines:
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        avoid_distances.append(float(fields["distance"]))
    if not avoid_success or not avoid_distances or max(avoid_distances) > 0.9:
        raise AssertionError(f"avoid contact unverified: successes={avoid_success}, distances={avoid_distances}")
    if not camera_captured:
        raise AssertionError("avoid pitch camera was not captured")
    result = {"source_player": str(player), "initial_agents": initial_counts,
              "approach_success_distances_m": distances,
              "avoid_success_distances_m": avoid_distances,
              "avoid_pitch60_image": str(screenshot), "result": "passed"}
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
