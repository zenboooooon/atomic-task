"""Oracle smoke test: finding treasure is insufficient until the red panel is touched."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from task_probe import _advance_motion, _look_action
from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]
SPACING = 0.25
LIMIT = 8.5


def _cell(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint((point + LIMIT) / SPACING).astype(int))


def _point(cell: tuple[int, int]) -> np.ndarray:
    return np.array(cell, dtype=float) * SPACING - LIMIT


def _path(layout: dict[str, object], destination: np.ndarray) -> list[np.ndarray]:
    obstacles = [np.array((item["x"], item["z"])) for item in layout["obstacles"]]
    start = _cell(np.array((layout["player_x"], layout["player_z"])))
    goal = _cell(destination)

    def free(cell: tuple[int, int]) -> bool:
        point = _point(cell)
        return (np.max(np.abs(point)) <= LIMIT
                and all(np.linalg.norm(point - item) > 1.05 for item in obstacles))

    frontier = [(0.0, start)]
    costs = {start: 0.0}
    previous = {}
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dz == 0:
                    continue
                next_cell = current[0] + dx, current[1] + dz
                if not free(next_cell):
                    continue
                if dx and dz and (not free((current[0] + dx, current[1]))
                                  or not free((current[0], current[1] + dz))):
                    continue
                cost = costs[current] + math.hypot(dx, dz)
                if cost >= costs.get(next_cell, math.inf):
                    continue
                costs[next_cell] = cost
                previous[next_cell] = current
                heapq.heappush(frontier, (cost + math.dist(next_cell, goal), next_cell))
    if goal not in costs:
        raise RuntimeError("no oracle path to stand-off point")
    cells = [goal]
    while cells[-1] != start:
        cells.append(previous[cells[-1]])
    cells.reverse()
    return [_point(item) for item in cells[1:]] + [destination]


def _motion(position: np.ndarray, yaw: float, waypoint: np.ndarray) -> np.ndarray:
    delta = waypoint - position
    speed = min(2.5, 2.0 * np.linalg.norm(delta))
    world = delta / max(np.linalg.norm(delta), 1e-6) * speed / 3.0
    radians = math.radians(yaw)
    right = np.array((math.cos(radians), -math.sin(radians)))
    forward = np.array((math.sin(radians), math.cos(radians)))
    return np.array((world @ right, world @ forward), dtype=np.float32)


def _video_frame(rgb: np.ndarray, phase: str, frame_number: int,
                 layout: dict[str, object], position: np.ndarray,
                 trail: list[np.ndarray]) -> np.ndarray:
    camera = cv2.cvtColor(np.rint(rgb * 255).clip(0, 255).astype(np.uint8),
                          cv2.COLOR_RGB2BGR)
    canvas = np.full((504, 672, 3), (28, 28, 28), dtype=np.uint8)
    canvas[56:, :448] = cv2.resize(camera, (448, 448))
    cv2.putText(canvas, "TREASURE v0.2  |  18 m arena", (9, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"oracle demonstration  |  {phase}  |  {frame_number:03d}",
                (9, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (170, 230, 170), 1, cv2.LINE_AA)
    cv2.putText(canvas, "LAYOUT MAP", (470, 88), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (230, 230, 230), 1, cv2.LINE_AA)
    left, top, size = 470, 111, 178
    cv2.rectangle(canvas, (left, top), (left + size, top + size),
                  (160, 160, 160), 1)

    def marker(point: np.ndarray) -> tuple[int, int]:
        return (round(left + (point[0] + 9) / 18 * size),
                round(top + (9 - point[1]) / 18 * size))

    if len(trail) > 1:
        cv2.polylines(canvas, [np.array([marker(item) for item in trail],
                                         dtype=np.int32)], False, (90, 190, 90), 1)
    for obstacle in layout["obstacles"]:
        cv2.circle(canvas, marker(np.array((obstacle["x"], obstacle["z"]))),
                   6, (135, 135, 135), -1)
    cv2.circle(canvas, marker(np.array((layout["target_x"], layout["target_z"]))),
               6, (50, 50, 240), -1)
    cv2.circle(canvas, marker(position), 6, (245, 120, 55), -1)
    cv2.putText(canvas, "blue: player", (470, 318), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (245, 160, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, "red: treasure", (470, 339), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (120, 120, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "gray: obstacles", (470, 360), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "green: route", (470, 381), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (100, 200, 100), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    video_path = args.video.resolve() if args.video else None
    if video_path is not None and video_path.exists():
        raise FileExistsError(video_path)
    staging_path = (video_path.with_name(video_path.stem + ".staging.mp4")
                    if video_path is not None else None)
    writer = None
    manifest = json.loads((ROOT / "manifests/pilot_layouts.json").read_text())
    assert manifest["protocol_version"] == "treasure-v0.2"
    layout = next(item for item in manifest["layouts"]
                  if item["task"] == "treasure" and item["split"] == "validation")
    executable = ROOT / "unity/Build/TpsTreasure.x86_64"
    layout_manifest_sha256(executable, ROOT / "manifests/pilot_layouts.json")
    start = np.array((layout["player_x"], layout["player_z"]), dtype=float)
    target = np.array((layout["target_x"], layout["target_z"]), dtype=float)
    direction = (target - start) / np.linalg.norm(target - start)
    stand_off = target - direction * 1.2
    path = _path(layout, stand_off)
    position = start.copy()
    trail = [start.copy()]
    velocity = np.zeros(2)
    yaw = float(layout["camera_yaw_degrees"])
    pitch = 15.0
    waypoint_index = 0
    phase = "search"
    hold_ticks = 0
    first_agent = None
    agent_order = None
    log_path = ROOT / "probe-treasure-contact-v02.log"
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=71, seed=0, no_graphics=False,
        additional_args=["-logFile", str(log_path), "--tps-task=treasure",
                         "--tps-split=validation", "--tps-layout-start=0"],
    )
    try:
        environment.reset()
        behavior = next(iter(environment.behavior_specs))
        for step in range(250):
            decisions, terminals = environment.get_steps(behavior)
            if agent_order is None:
                agent_order = sorted(map(int, decisions.agent_id))
            if step == 20:
                # Checkpoint logs are emitted in the same order as sorted ML-Agents
                # IDs. Resolve the chosen layout rather than assuming scene order.
                entries = [line for line in log_path.read_text(errors="replace").splitlines()
                           if "TPS_EPISODE" in line and "ticks=100 " in line]
                if len(entries) != len(agent_order):
                    raise AssertionError("could not map Unity agent IDs to layouts")
                layout_ids = [next(token.split("=", 1)[1] for token in line.split()
                                   if token.startswith("layout=")) for line in entries]
                first_agent = agent_order[layout_ids.index(layout["layout_id"])]
            if first_agent is None:
                environment.set_actions(behavior, ActionTuple(
                    continuous=np.zeros((len(decisions), 4), dtype=np.float32)))
                environment.step()
                continue
            terminal_ids = list(map(int, terminals.agent_id))
            if first_agent in terminal_ids:
                result = float(terminals.reward[terminal_ids.index(first_agent)])
                if phase != "collect" or result <= 0.5:
                    raise AssertionError(f"unexpected terminal: phase={phase} reward={result}")
                states = [line for line in log_path.read_text(errors="replace").splitlines()
                          if "TPS_EPISODE" in line and f"layout={layout['layout_id']}" in line]
                if not states or "reason=success" not in states[-1]:
                    raise AssertionError("Unity did not log contact success for selected layout")
                fields = dict(token.split("=", 1) for token in states[-1].split() if "=" in token)
                distance = float(fields["distance"])
                if distance > 0.9:
                    raise AssertionError(f"success before panel contact: distance={distance}")
                if writer is not None:
                    final_rgb = _rgb_at(terminals.obs, terminal_ids.index(first_agent))
                    frame = _video_frame(final_rgb, "SUCCESS: PANEL CONTACT",
                                         step - 20, layout, position, trail)
                    for _ in range(10):
                        writer.write(frame)
                    writer.release()
                    writer = None
                    staging_path.replace(video_path)
                print(json.dumps({"result": "passed", "layout": layout["layout_id"],
                                  "steps": step, "stand_off_checked": True,
                                  "success_distance": distance,
                                  "reward_version": fields["reward_version"],
                                  "video": str(video_path) if video_path else None}), flush=True)
                return
            actions = np.zeros((len(decisions), 4), dtype=np.float32)
            selected = list(map(int, decisions.agent_id)).index(first_agent)
            if video_path is not None:
                if writer is None:
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(staging_path),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             10, (672, 504))
                    if not writer.isOpened():
                        raise RuntimeError("could not open treasure MP4 writer")
                writer.write(_video_frame(_rgb_at(decisions.obs, selected), phase,
                                          step - 20, layout, position, trail))
            look = _look_action(layout, yaw, pitch, position)
            if step >= 40 and phase == "search":
                phase = "navigate"
            if phase == "navigate":
                while waypoint_index < len(path) - 1 and np.linalg.norm(
                        path[waypoint_index] - position) < 0.42:
                    waypoint_index += 1
                actions[selected, :2] = _motion(position, yaw, path[waypoint_index])
                if waypoint_index == len(path) - 1 and np.linalg.norm(
                        position - stand_off) < 0.16:
                    phase = "hold"
            if phase == "hold":
                actions[selected, :2] = 0
                hold_ticks += 1
                if hold_ticks >= 12:
                    distance = float(np.linalg.norm(position - target))
                    if not 0.95 < distance < 1.5:
                        raise AssertionError(f"unsafe stand-off distance: {distance}")
                    phase = "collect"
            if phase == "collect":
                actions[selected, :2] = _motion(position, yaw, target)
            actions[selected, 2:] = look[2:]
            environment.set_actions(behavior, ActionTuple(continuous=actions))
            environment.step()
            position, velocity, yaw = _advance_motion(position, velocity, yaw,
                                                       actions[selected])
            trail.append(position.copy())
            pitch = float(np.clip(pitch + float(actions[selected, 3]) * 9.0, -60, 60))
        raise AssertionError("oracle did not touch the panel within 250 decisions")
    finally:
        environment.close()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    main()
