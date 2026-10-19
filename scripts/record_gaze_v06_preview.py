"""Play the revised environments with a scripted controller, without a learned policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from probe_treasure_contact_v02 import _motion, _path
from record_gaze_v05_look1p5_contact import advance
from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/gaze_v06_layouts.json"
SIZE = (448, 528)


def target_pixels(rgb: np.ndarray, treasure: bool) -> tuple[np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    if treasure:
        mask = (r > .4) & (b > .4) & (g < .55 * np.minimum(r, b))
    else:
        mask = (r > .25) & (r > 1.4 * g) & (g > 1.5 * b)
    return np.nonzero(mask)


def target_center(rgb: np.ndarray, treasure: bool) -> tuple[float, float] | None:
    rows, columns = target_pixels(rgb, treasure)
    if len(rows) < 5:
        return None
    focal = 112 / math.tan(math.radians(35))
    return (math.degrees(math.atan((float(columns.mean()) - 111.5) / focal)),
            math.degrees(math.atan((float(rows.mean()) - 111.5) / focal)))


def wrap(degrees: float) -> float:
    return (degrees + 180) % 360 - 180


def frame(rgb: np.ndarray, name: str, step: int, status: str, treasure: bool) -> np.ndarray:
    canvas = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    image = cv2.cvtColor(np.rint(rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    canvas[80:] = cv2.resize(image, (448, 448))
    lines = [f"v0.6 {name.upper()} | SCRIPTED CONTROLLER",
             "Magenta diamond / yellow box" if treasure else "Orange ellipsoid / cyan cylinder",
             f"decision {step:03d} | {status}"]
    for y, text in zip((21, 44, 68), lines):
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (225, 230, 230), 1, cv2.LINE_AA)
    return canvas


def swept_layout(layout: dict, moving: bool) -> dict:
    if not moving:
        return layout
    route = np.array((layout["target_x"] - layout["player_x"],
                      layout["target_z"] - layout["player_z"]))
    sideways = np.array((-route[1], route[0])) / np.linalg.norm(route)
    result = dict(layout)
    result["obstacles"] = [dict(obstacle, x=obstacle["x"] + offset * sideways[0],
                                z=obstacle["z"] + offset * sideways[1])
                           for obstacle in layout["obstacles"]
                           for offset in (-.65, -.325, 0, .325, .65)]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("look", "treasure-static", "treasure-moving"), required=True)
    parser.add_argument("--layout-start", type=int, default=0)
    parser.add_argument("--name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=160)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    name = args.name or args.mode
    treasure = args.mode.startswith("treasure")
    moving = args.mode.endswith("moving")
    task = "treasure" if treasure else "look"
    player = "TpsTreasurePreview" if treasure else "TpsLookPreview"
    executable = ROOT / "unity/Build" / f"{player}.x86_64"
    layout_manifest_sha256(executable, MANIFEST)
    candidates = [r for r in json.loads(MANIFEST.read_text())["layouts"]
                  if r["task"] == task and r["split"] == "validation"]
    layout = candidates[args.layout_start]
    position = np.array((layout["player_x"], layout["player_z"]), dtype=float)
    target = np.array((layout["target_x"], layout["target_z"]), dtype=float)
    velocity = np.zeros(2)
    yaw = float(layout["camera_yaw_degrees"])
    path = []
    if treasure:
        direction = (target - position) / np.linalg.norm(target - position)
        path = _path(swept_layout(layout, moving), target - direction * 1.2) + [target]
    waypoint = 0
    navigating = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video = args.output_dir / f"{name}.mp4"
    log = args.output_dir / f"{name}.player.log"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, SIZE)
    if not writer.isOpened():
        raise RuntimeError("cannot open video writer")
    env = UnityEnvironment(file_name=str(executable), worker_id=args.worker_id,
                           seed=0, no_graphics=False,
                           additional_args=["-logFile", str(log),
                                            f"--tps-task={'treasure' if treasure else 'pretrain'}",
                                            "--tps-split=validation",
                                            f"--tps-layout-start={args.layout_start}",
                                            f"--tps-obstacle-motion={'moving' if moving else 'static'}"])
    report = None
    first_view = None
    try:
        env.reset()
        behavior = next(iter(env.behavior_specs))
        for step in range(300):
            decisions, terminals = env.get_steps(behavior)
            if len(terminals):
                success = bool(not terminals.interrupted[0] and terminals.reward[0] > .5)
                entries = [line for line in log.read_text(errors="replace").splitlines()
                           if "TPS_EPISODE" in line and f"layout={layout['layout_id']}" in line]
                fields = dict(token.split("=", 1) for token in entries[-1].split() if "=" in token)
                reason = fields["reason"]
                final = frame(_rgb_at(terminals.obs, 0), name, step,
                              "SUCCESS" if success else f"FAILURE: {reason}", treasure)
                for _ in range(15):
                    writer.write(final)
                report = {"mode": args.mode, "scripted": True, "trained_policy": False,
                          "layout_id": layout["layout_id"], "layout_index": args.layout_start,
                          "first_frame_target": first_view, "success": success,
                          "terminal_reason": reason, "terminal_ticks": int(fields["ticks"]),
                          "contact_distance_m": float(fields["distance"]) if treasure else None,
                          "decision_frames": step, "video": video.name,
                          "build_sha256": unity_build_sha256(executable)}
                break
            if len(decisions) != 1:
                raise RuntimeError("expected a single preview agent")
            rgb = _rgb_at(decisions.obs, 0)
            center = target_center(rgb, treasure)
            if first_view is None:
                cv2.imwrite(str(args.output_dir / f"{name}-initial.png"),
                            cv2.cvtColor(np.rint(rgb * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                rows, columns = target_pixels(rgb, treasure)
                first_view = {"visible": center is not None,
                              "center_x": float(columns.mean() / 224) if len(columns) else None,
                              "bbox_x": [int(columns.min()), int(columns.max())] if len(columns) else None}
                if not treasure:
                    x = first_view["center_x"]
                    if x is None or not (.08 < x < .18 or .82 < x < .92):
                        raise RuntimeError(f"look target did not start at a visible edge: {first_view}")
                    if columns.min() == 0 or columns.max() == 223:
                        raise RuntimeError("look target is clipped at reset")
            action = np.zeros(4, dtype=np.float32)
            status = "INITIAL VIEW" if step < 10 else "CENTER + HOLD 1.5s"
            if step >= 10:
                if center is not None:
                    action[2] = np.clip(.7 * center[0] / 9, -.35, .35)
                    action[3] = np.clip(.7 * center[1] / 9, -.25, .25)
                else:
                    bearing = math.degrees(math.atan2(target[0] - position[0], target[1] - position[1]))
                    action[2] = np.clip(wrap(bearing - yaw) / 9, -.35, .35)
                if treasure:
                    navigating = navigating or (center is not None and abs(center[0]) < 7)
                    status = "NAVIGATE TO CONTACT" if navigating else "SEARCH"
                    if navigating:
                        while waypoint < len(path) - 1 and np.linalg.norm(path[waypoint] - position) < .42:
                            waypoint += 1
                        action[:2] = _motion(position, yaw, path[waypoint])
            writer.write(frame(rgb, name, step, status, treasure))
            env.set_actions(behavior, ActionTuple(continuous=action.reshape(1, 4)))
            env.step()
            position, velocity, yaw = advance(position, velocity, yaw, action)
        if report is None:
            raise RuntimeError("scripted episode did not finish")
    finally:
        env.close()
        writer.release()
    (args.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not report["success"]:
        raise RuntimeError(f"scripted episode failed: {report['terminal_reason']}")


if __name__ == "__main__":
    main()
