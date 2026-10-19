"""Replay diagnostic actions against one explicit Unity task layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.episode import EpisodeTracker
from tps_transfer.smoke import _rgb_at, _steps

ROOT = Path(__file__).resolve().parents[1]


def _look_action(
    layout: dict[str, object], yaw: float, pitch: float, position: np.ndarray
) -> np.ndarray:
    radians = math.radians(yaw)
    camera_x = float(position[0]) - 3.5 * math.sin(radians)
    camera_z = float(position[1]) - 3.5 * math.cos(radians)
    dx = float(layout["target_x"]) - camera_x
    dz = float(layout["target_z"]) - camera_z
    desired_yaw = math.degrees(math.atan2(dx, dz))
    desired_pitch = math.degrees(math.atan2(3.2 - 1.6, math.hypot(dx, dz)))
    yaw_error = (desired_yaw - yaw + 180) % 360 - 180
    return np.array([
        0.0, 0.0,
        np.clip(yaw_error / 12.0, -1.0, 1.0),
        np.clip((desired_pitch - pitch) / 9.0, -1.0, 1.0),
    ], dtype=np.float32)


def _visual_look_action(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0]
    mask = (red > 0.5) & (red > 2.0 * rgb[..., 1]) & (red > 2.0 * rgb[..., 2])
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        raise RuntimeError("target marker is not visible in diagnostic RGB")
    focal_pixels = 112.0 / math.tan(math.radians(35.0))
    yaw_error = math.degrees(math.atan((float(columns.mean()) - 111.5) / focal_pixels))
    pitch_error = math.degrees(math.atan((float(rows.mean()) - 111.5) / focal_pixels))
    return np.array([
        np.clip(yaw_error / 12.0, -1.0, 1.0),
        np.clip(pitch_error / 9.0, -1.0, 1.0),
    ], dtype=np.float32)


def _stop_action(
    layout: dict[str, object], position: np.ndarray, velocity: np.ndarray, yaw: float
) -> np.ndarray:
    target = np.array([layout["target_x"], layout["target_z"]], dtype=np.float64)
    delta = target - position
    distance = np.linalg.norm(delta)
    direction = delta / max(distance, 1e-6)
    desired_speed = float(np.clip(2.0 * (distance - 1.5), -3.0, 3.0))
    radians = math.radians(yaw)
    right = np.array([math.cos(radians), -math.sin(radians)])
    forward = np.array([math.sin(radians), math.cos(radians)])
    world_input = direction * (desired_speed / 3.0)
    return np.array([
        float(world_input @ right),
        float(world_input @ forward),
        0.0, 0.0,
    ], dtype=np.float32)


def _route_waypoints(layout: dict[str, object]) -> list[np.ndarray]:
    start = np.array([layout["player_x"], layout["player_z"]], dtype=np.float64)
    target = np.array([layout["target_x"], layout["target_z"]], dtype=np.float64)
    direction = target - start
    direction /= np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    candidates = []
    for side in (-1, 1):
        offset = lateral * side * 1.6
        route = [
            np.array([item["x"], item["z"]], dtype=np.float64) + offset
            for item in layout["obstacles"]
        ]
        route.append(target - direction * 0.6 + offset)
        boundary = max(float(np.abs(point).max()) for point in route)
        candidates.append((boundary, route))
    return min(candidates, key=lambda item: item[0])[1]


def _route_action(
    position: np.ndarray, yaw: float, waypoints: list[np.ndarray], index: int
) -> tuple[np.ndarray, int]:
    while index < len(waypoints) - 1 and np.linalg.norm(waypoints[index] - position) < 0.35:
        index += 1
    delta = waypoints[index] - position
    distance = np.linalg.norm(delta)
    speed = float(np.clip(2.0 * distance, 0.0, 3.0))
    world_input = delta / max(distance, 1e-6) * (speed / 3.0)
    radians = math.radians(yaw)
    right = np.array([math.cos(radians), -math.sin(radians)])
    forward = np.array([math.sin(radians), math.cos(radians)])
    action = np.array([
        float(world_input @ right),
        float(world_input @ forward),
        0.0, 0.0,
    ], dtype=np.float32)
    return action, index


def _advance_motion(
    position: np.ndarray, velocity: np.ndarray, yaw: float, action: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    position = position.copy()
    velocity = velocity.copy()
    for _ in range(5):
        yaw += float(action[2]) * 120.0 * 0.02
        radians = math.radians(yaw)
        right = np.array([math.cos(radians), -math.sin(radians)])
        forward = np.array([math.sin(radians), math.cos(radians)])
        local = action[:2].astype(np.float64)
        local /= max(1.0, np.linalg.norm(local))
        desired = (right * local[0] + forward * local[1]) * 3.0
        delta = desired - velocity
        norm = np.linalg.norm(delta)
        velocity += delta * min(1.0, 0.12 / max(norm, 1e-9))
        position += velocity * 0.02
    return position, velocity, yaw


def run(
    task: str, split: str, index: int, policy: str, max_steps: int,
    capture_dir: Path | None = None, worker_id: int = 0,
) -> dict[str, object]:
    if capture_dir is not None:
        capture_dir = (ROOT / capture_dir).resolve()
    manifest = json.loads((ROOT / "manifests/pilot_layouts.json").read_text())
    layouts = [
        item for item in manifest["layouts"]
        if item["task"] == task and item["split"] == split
    ]
    if not 0 <= index < len(layouts):
        raise ValueError("layout index outside split")
    layout = layouts[index]
    executable = ROOT / "unity/Build/TpsTask.x86_64"
    layout_manifest_sha256(executable, ROOT / "manifests/pilot_layouts.json")
    tracker = EpisodeTracker(worker_id=worker_id)
    yaw = float(layout["camera_yaw_degrees"])
    pitch = 15.0
    position = np.array([layout["player_x"], layout["player_z"]], dtype=np.float64)
    target = np.array([layout["target_x"], layout["target_z"]], dtype=np.float64)
    direction = target - position
    direction /= np.linalg.norm(direction)
    velocity = direction * float(layout["initial_speed_mps"])
    action_digest = hashlib.sha256()
    captures: list[str] = []
    waypoints = _route_waypoints(layout) if policy in ("avoid", "composite", "treasure") else []
    route_index = 0
    treasure_seen = False
    decisions_seen = 0
    transitions_seen = 0
    cumulative_reward = 0.0
    terminal_reward = None
    log_path = ROOT / "runs/phase1/episode_logs" / f"{task}_{split}_{index:03d}_{policy}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    interrupted = None
    environment = UnityEnvironment(
        file_name=str(executable),
        worker_id=worker_id,
        seed=0,
        no_graphics=False,
        additional_args=[
            "-logFile", str(log_path),
            f"--tps-task={task}",
            f"--tps-split={split}",
            f"--tps-layout-start={index}",
        ],
    )
    try:
        environment.reset()
        if len(environment.behavior_specs) != 1:
            raise RuntimeError("expected one task behavior")
        behavior = next(iter(environment.behavior_specs))
        for iteration in range(max_steps):
            decisions, terminals = environment.get_steps(behavior)
            try:
                observed = tracker.observe(
                    behavior, _steps(decisions, False), _steps(terminals, True)
                )
                transitions_seen += len(observed)
                cumulative_reward += sum(item.reward for item in observed)
            except ValueError as exc:
                raise RuntimeError(
                    f"iteration={iteration} decisions={decisions.reward.tolist()} "
                    f"terminals={terminals.reward.tolist()} "
                    f"interrupted={terminals.interrupted.tolist()}"
                ) from exc
            if len(terminals):
                terminal_reward = float(terminals.reward[0])
                interrupted = bool(terminals.interrupted[0])
                break
            if len(decisions) != 1:
                raise RuntimeError(f"expected one decision, got {len(decisions)}")
            if capture_dir is not None and split == "train" and decisions_seen % 5 == 0 and decisions_seen <= 55:
                capture_dir.mkdir(parents=True, exist_ok=True)
                rgb = _rgb_at(decisions.obs, 0)
                file = capture_dir / f"{task}_{split}_{index:03d}_{decisions_seen:03d}.png"
                color = np.rint(rgb * 255.0).clip(0, 255).astype(np.uint8)
                if not cv2.imwrite(str(file), cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"failed to save {file}")
                captures.append(str(file.relative_to(ROOT)))
            if policy == "look":
                try:
                    look = _visual_look_action(_rgb_at(decisions.obs, 0))
                except RuntimeError:
                    look = _look_action(layout, yaw, pitch, position)[2:]
                action = np.concatenate((np.zeros(2, dtype=np.float32), look))
            elif policy == "stop":
                action = _stop_action(layout, position, velocity, yaw)
            elif policy in ("avoid", "composite"):
                action, route_index = _route_action(position, yaw, waypoints, route_index)
                if policy == "composite":
                    action[2:] = _visual_look_action(_rgb_at(decisions.obs, 0))
            elif policy == "treasure":
                try:
                    _visual_look_action(_rgb_at(decisions.obs, 0))
                    treasure_seen = True
                except RuntimeError:
                    pass
                if not treasure_seen and decisions_seen < 25:
                    action = _look_action(layout, yaw, pitch, position)
                else:
                    action, route_index = _route_action(position, yaw, waypoints, route_index)
                    action[2:] = _look_action(layout, yaw, pitch, position)[2:]
            elif policy == "forward":
                action = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
            elif policy == "zero":
                action = np.zeros(4, dtype=np.float32)
            else:
                raise ValueError("unknown probe policy")
            decisions_seen += 1
            tracker.record_action(behavior, int(decisions.agent_id[0]), action)
            action_digest.update(action.tobytes())
            environment.set_actions(behavior, ActionTuple(continuous=action[None]))
            environment.step()
            position, velocity, yaw = _advance_motion(position, velocity, yaw, action)
            pitch = float(np.clip(pitch + float(action[3]) * 9.0, -60, 60))
        else:
            decisions, terminals = environment.get_steps(behavior)
            observed = tracker.observe(
                behavior, _steps(decisions, False), _steps(terminals, True)
            )
            transitions_seen += len(observed)
            cumulative_reward += sum(item.reward for item in observed)
            if len(terminals):
                terminal_reward = float(terminals.reward[0])
                interrupted = bool(terminals.interrupted[0])
    finally:
        environment.close()
    diagnostics = {}
    if log_path.is_file():
        states = [line for line in log_path.read_text(errors="replace").splitlines()
                  if "TPS_EPISODE " in line and " task=" in line]
        if states:
            tokens = states[-1].split()
            diagnostics = {key: value for key, value in
                           (token.split("=", 1) for token in tokens if "=" in token)}
    reward_keys = (
        "reward_time", "reward_view", "reward_distance", "reward_band",
        "penalty_look_movement", "penalty_stop_speed", "penalty_invisible",
        "reward_terminal", "reward_milestone", "penalty_band_distance",
        "penalty_look_command", "penalty_stop_view", "penalty_stop_camera_command",
    )
    component_total = None
    if all(key in diagnostics for key in reward_keys):
        component_total = sum(float(diagnostics[key]) for key in reward_keys)
        if terminal_reward is not None and not np.isclose(
            component_total, cumulative_reward, rtol=0.0, atol=0.002
        ):
            raise RuntimeError(
                f"Unity reward ledger {component_total} differs from LLAPI {cumulative_reward}"
            )
    return {
        "task": task,
        "split": split,
        "layout_index": index,
        "layout_id": layout["layout_id"],
        "policy": policy,
        "treasure_seen_in_rgb": treasure_seen,
        "action_sha256": action_digest.hexdigest(),
        "decisions_sent": decisions_seen,
        "transitions_seen": transitions_seen,
        "terminal_reward": terminal_reward,
        "cumulative_reward": cumulative_reward,
        "component_total": component_total,
        "unity_log": str(log_path.relative_to(ROOT)),
        "diagnostics": diagnostics,
        "interrupted": interrupted,
        "success": terminal_reward is not None and terminal_reward > 0.5 and not interrupted,
        "captures": captures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("look", "approach", "stop", "avoid", "composite", "treasure"), default="look")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--policy", choices=("zero", "look", "forward", "stop", "avoid", "composite", "treasure"), default="look")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()
    result = run(args.task, args.split, args.index, args.policy, args.max_steps,
                 args.capture_dir, args.worker_id)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
