"""Record scripted pretraining and treasure environment previews before training."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/gaze_v04_layouts.json"
SIZE = (448, 504)


def red_center(rgb: np.ndarray) -> tuple[float, float] | None:
    mask = ((rgb[..., 0] > 0.5) & (rgb[..., 0] > 2 * rgb[..., 1])
            & (rgb[..., 0] > 2 * rgb[..., 2]))
    rows, columns = np.nonzero(mask)
    if len(rows) < 5:
        return None
    focal = 112.0 / math.tan(math.radians(35.0))
    return (math.degrees(math.atan((float(columns.mean()) - 111.5) / focal)),
            math.degrees(math.atan((float(rows.mean()) - 111.5) / focal)))


def wrap(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def frame(rgb: np.ndarray, mode: str, number: int, status: str = "") -> np.ndarray:
    image = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (448, 448))
    canvas = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    canvas[56:] = image
    cv2.putText(canvas, f"{mode.upper()}  |  scripted environment preview",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"decision {number:03d}  {status}", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.49, (95, 220, 220), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("look", "approach", "treasure-static",
                                            "treasure-moving"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=84)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = args.mode.split("-", 1)[0]
    motion = args.mode.split("-", 1)[1] if task == "treasure" else "static"
    executable = ROOT / "unity/Build" / {
        "look": "TpsLookPreview.x86_64",
        "approach": "TpsApproachPreview.x86_64",
        "treasure": "TpsTreasurePreview.x86_64",
    }[task]
    layout_manifest_sha256(executable, MANIFEST)
    build_sha = unity_build_sha256(executable)
    layouts = json.loads(MANIFEST.read_text())["layouts"]
    layout = next(item for item in layouts
                  if item["task"] == task and item["split"] == "validation")
    target_yaw = math.degrees(math.atan2(
        layout["target_x"] - layout["player_x"],
        layout["target_z"] - layout["player_z"]))
    yaw = float(layout["camera_yaw_degrees"])
    player_log = args.output_dir / f"{args.mode}.player.log"
    output = args.output_dir / f"{args.mode}.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 10, SIZE)
    if not writer.isOpened():
        raise RuntimeError("cannot open video writer")
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=args.worker_id, seed=0,
        no_graphics=False, additional_args=[
            "-logFile", str(player_log),
            "--tps-task=treasure" if task == "treasure" else "--tps-task=pretrain",
            "--tps-split=validation", f"--tps-obstacle-motion={motion}",
        ],
    )
    first_visible = None
    selected = None
    terminal = None
    count = 0
    limit = 100 if task == "treasure" else 170
    try:
        environment.reset()
        behaviors = {name.split("?", 1)[0].removeprefix("TPS_").lower(): name
                     for name in environment.behavior_specs}
        if task not in behaviors:
            raise RuntimeError(f"unexpected Unity behaviors: {behaviors}")
        initial, _ = environment.get_steps(behaviors[task])
        selected = min(map(int, initial.agent_id))
        for number in range(limit):
            pending = {}
            for current_task, behavior in behaviors.items():
                decisions, terminals = environment.get_steps(behavior)
                if current_task == task and selected in map(int, terminals.agent_id):
                    row = list(map(int, terminals.agent_id)).index(selected)
                    terminal = {"success": bool(not terminals.interrupted[row]
                                                and terminals.reward[row] > 0.5),
                                "reward": float(terminals.reward[row]), "decision": number}
                    final = frame(_rgb_at(terminals.obs, row), args.mode, number,
                                  "SUCCESS" if terminal["success"] else "FAILURE")
                    for _ in range(10):
                        writer.write(final)
                    break
                actions = np.zeros((len(decisions), 4), dtype=np.float32)
                if current_task == task:
                    ids = list(map(int, decisions.agent_id))
                    if selected not in ids:
                        raise RuntimeError("selected agent has no decision")
                    row = ids.index(selected)
                    rgb = _rgb_at(decisions.obs, row)
                    center = red_center(rgb)
                    if first_visible is None:
                        first_visible = center is not None
                    if task == "look":
                        if center is None:
                            actions[row, 2] = np.clip(wrap(target_yaw - yaw) / 9.0, -1, 1)
                        else:
                            actions[row, 2] = np.clip(center[0] / 9.0, -1, 1)
                            actions[row, 3] = np.clip(center[1] / 9.0, -1, 1)
                    elif task == "approach":
                        # The camera is over the avatar's shoulder, so image
                        # centering would steer the body beside the panel.
                        actions[row, 2] = np.clip(wrap(target_yaw - yaw) / 9.0, -1, 1)
                        actions[row, 1] = 1.0
                    else:
                        # Identical fixed camera actions make static and moving
                        # obstacle layouts easy to compare visually.
                        actions[row, 2] = np.clip(wrap(target_yaw - yaw) / 9.0, -1, 1)
                    writer.write(frame(rgb, args.mode, number))
                    count += 1
                    yaw += float(actions[row, 2]) * 9.0
                pending[behavior] = actions
            if terminal is not None:
                break
            for behavior, actions in pending.items():
                environment.set_actions(behavior, ActionTuple(continuous=actions))
            environment.step()
    finally:
        environment.close()
        writer.release()
    if count < 2:
        raise RuntimeError("video contains too few frames")
    if task == "look" and first_visible is not False:
        raise RuntimeError("look target should begin outside the field of view")
    if task == "approach" and first_visible is not True:
        raise RuntimeError("approach target should begin inside the field of view")
    report = {"mode": args.mode, "scripted": True, "trained_policy": False,
              "build_sha256": build_sha, "layout_id": layout["layout_id"],
              "first_frame_target_visible": first_visible,
              "decision_frames": count, "terminal": terminal,
              "video": output.name}
    (args.output_dir / f"{args.mode}.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
