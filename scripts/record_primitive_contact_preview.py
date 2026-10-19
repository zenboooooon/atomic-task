"""Record scripted camera previews from the corrected primitive-task player.

These are environment demonstrations, not rollouts of a trained policy.
"""

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
OUT = ROOT / "runs/primitive_contact_v03/preview"
TASKS = ("look", "approach", "avoid")


def video_frame(rgb: np.ndarray, task: str, step: int, status: str = "") -> np.ndarray:
    image = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (448, 448))
    canvas = np.zeros((492, 448, 3), dtype=np.uint8)
    canvas[44:] = image
    cv2.putText(canvas, f"{task.upper()}  |  scripted environment demo", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"decision {step:03d}  {status}", (8, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (90, 220, 220), 1, cv2.LINE_AA)
    return canvas


def red_center(rgb: np.ndarray) -> tuple[float, float] | None:
    mask = ((rgb[..., 0] > 0.5) & (rgb[..., 0] > 2 * rgb[..., 1])
            & (rgb[..., 0] > 2 * rgb[..., 2]))
    rows, columns = np.nonzero(mask)
    if len(rows) < 5:
        return None
    focal = 112.0 / math.tan(math.radians(35.0))
    yaw_error = math.degrees(math.atan((float(columns.mean()) - 111.5) / focal))
    pitch_error = math.degrees(math.atan((float(rows.mean()) - 111.5) / focal))
    return yaw_error, pitch_error


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    player = ROOT / "unity/Build/TpsPretrainThree.x86_64"
    manifest = ROOT / "manifests/primitive_contact_v03_layouts.json"
    layout_manifest_sha256(player, manifest)
    layouts = json.loads(manifest.read_text())["layouts"]
    avoid_layouts = {item["layout_id"]: item for item in layouts
                     if item["task"] == "avoid" and item["split"] == "validation"}
    player_log = OUT / "player.log"
    env = UnityEnvironment(
        file_name=str(player), worker_id=87, seed=0, no_graphics=False,
        additional_args=["-logFile", str(player_log),
                         "--tps-task=pretrain", "--tps-split=validation"],
    )
    writers: dict[tuple[str, int], cv2.VideoWriter] = {}
    counts: dict[tuple[str, int], int] = {}
    outcomes: dict[tuple[str, int], bool] = {}
    avoid_states: dict[int, dict] = {}
    first_ids: dict[str, set[int]] = {}
    try:
        env.reset()
        behaviors = {name.split("?", 1)[0].removeprefix("TPS_").lower(): name
                     for name in env.behavior_specs}
        assert set(behaviors) == set(TASKS), behaviors
        for task in TASKS:
            decisions, terminals = env.get_steps(behaviors[task])
            assert len(decisions) == 4 and len(terminals) == 0
            first_ids[task] = set(map(int, decisions.agent_id))
        for step in range(240):
            pending_avoid = {}
            for task in TASKS:
                behavior = behaviors[task]
                decisions, terminals = env.get_steps(behavior)
                for row, agent_id in enumerate(terminals.agent_id):
                    key = (task, int(agent_id))
                    if key not in writers or key in outcomes:
                        continue
                    success = bool(not terminals.interrupted[row]
                                   and terminals.reward[row] > 0.5)
                    outcomes[key] = success
                    end = video_frame(_rgb_at(terminals.obs, row), task, step,
                                      "SUCCESS" if success else "FAILURE")
                    for _ in range(10):
                        writers[key].write(end)
                    writers[key].release()
                    del writers[key]
                if task == "avoid" and step >= 20 and not avoid_states:
                    lines = [line for line in player_log.read_text(errors="replace").splitlines()
                             if "TPS_EPISODE " in line and " task=avoid " in line
                             and " reason=checkpoint ticks=100 " in line]
                    if len(lines) >= 4:
                        selected_lines = lines[:4]
                    else:
                        selected_lines = []
                    for agent_id, line in zip(sorted(first_ids["avoid"]), selected_lines):
                        layout_id = next(part.split("=", 1)[1] for part in line.split()
                                         if part.startswith("layout="))
                        layout = avoid_layouts[layout_id]
                        avoid_states[agent_id] = {
                            "position": np.array((layout["player_x"], layout["player_z"]),
                                                 dtype=np.float64),
                            "velocity": np.zeros(2, dtype=np.float64),
                            "yaw": float(layout["camera_yaw_degrees"]),
                            "waypoints": _route_waypoints(layout)
                                         + [np.array((layout["target_x"], layout["target_z"]))],
                            "index": 0,
                        }
                actions = np.zeros((len(decisions), 4), dtype=np.float32)
                for row, agent_id in enumerate(decisions.agent_id):
                    agent_id = int(agent_id)
                    key = (task, agent_id)
                    if agent_id not in first_ids[task] or key in outcomes:
                        continue
                    rgb = _rgb_at(decisions.obs, row)
                    if key not in writers:
                        writer = cv2.VideoWriter(str(OUT / f"{task}_agent{agent_id:02d}.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"), 10, (448, 492))
                        if not writer.isOpened():
                            raise RuntimeError(f"cannot write video for {key}")
                        writers[key] = writer
                        counts[key] = 0
                    writers[key].write(video_frame(rgb, task, step))
                    counts[key] += 1
                    if task == "look":
                        center = red_center(rgb)
                        if center is None:
                            actions[row, 2] = 0.6
                        else:
                            actions[row, 2] = np.clip(center[0] / 12, -1, 1)
                            actions[row, 3] = np.clip(center[1] / 9, -1, 1)
                    elif task == "approach" and step >= 20:
                        center = red_center(rgb)
                        actions[row, 1] = 1.0
                        if center is not None:
                            actions[row, 2] = np.clip(center[0] / 12, -1, 1)
                    elif task == "avoid":
                        if agent_id in avoid_states:
                            state = avoid_states[agent_id]
                            action, state["index"] = _route_action(
                                state["position"], state["yaw"],
                                state["waypoints"], state["index"])
                            actions[row] = action
                            pending_avoid[agent_id] = action
                env.set_actions(behavior, ActionTuple(continuous=actions))
            if all(any(outcomes.get((task, agent), False) for agent in first_ids[task])
                   for task in TASKS):
                break
            env.step()
            for agent_id, action in pending_avoid.items():
                state = avoid_states[agent_id]
                state["position"], state["velocity"], state["yaw"] = _advance_motion(
                    state["position"], state["velocity"], state["yaw"], action)
            if step % 40 == 0:
                print(f"step={step} successes="
                      f"{[task for task in TASKS if any(outcomes.get((task, a), False) for a in first_ids[task])]}"
                      , flush=True)
    finally:
        env.close()
        for writer in writers.values():
            writer.release()
    report = {"note": "scripted environment demonstration, not trained policy",
              "outcomes": [{"task": task, "agent": agent, "success": success,
                            "frames": counts[(task, agent)],
                            "file": f"{task}_agent{agent:02d}.mp4"}
                           for (task, agent), success in sorted(outcomes.items())]}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
