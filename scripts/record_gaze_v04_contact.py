"""Record a scripted treasure contact in the static-obstacle preview scene."""

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
from record_gaze_v04_preview import frame, wrap
from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/gaze_v04/preview"


def advance(position: np.ndarray, velocity: np.ndarray, yaw: float,
            action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    position = position.copy()
    velocity = velocity.copy()
    for _ in range(5):
        yaw += float(action[2]) * 90.0 * 0.02
        angle = math.radians(yaw)
        right = np.array((math.cos(angle), -math.sin(angle)))
        forward = np.array((math.sin(angle), math.cos(angle)))
        local = action[:2].astype(float)
        local /= max(1.0, np.linalg.norm(local))
        desired = (right * local[0] + forward * local[1]) * 3.0
        delta = desired - velocity
        velocity += delta * min(1.0, 0.12 / max(np.linalg.norm(delta), 1e-9))
        position += velocity * 0.02
    return position, velocity, yaw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-away", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "manifests/gaze_v04_layouts.json"
    layout = next(item for item in json.loads(manifest.read_text())["layouts"]
                  if item["task"] == "treasure" and item["split"] == "validation")
    executable = ROOT / "unity/Build/TpsTreasurePreview.x86_64"
    layout_manifest_sha256(executable, manifest)
    position = np.array((layout["player_x"], layout["player_z"]), dtype=float)
    target = np.array((layout["target_x"], layout["target_z"]), dtype=float)
    direction = (target - position) / np.linalg.norm(target - position)
    stand_off = target - direction * 1.2
    path = _path(layout, stand_off) + [target]
    velocity = np.zeros(2)
    yaw = float(layout["camera_yaw_degrees"])
    waypoint = 0
    name = ("treasure-static-contact-no-gaze" if args.face_away
            else "treasure-static-contact")
    video = OUT / f"{name}.mp4"
    log = OUT / f"{name}.player.log"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (448, 504))
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")
    env = UnityEnvironment(
        file_name=str(executable), worker_id=87, seed=0, no_graphics=False,
        additional_args=["-logFile", str(log), "--tps-task=treasure",
                         "--tps-split=validation", "--tps-obstacle-motion=static"],
    )
    report = None
    try:
        env.reset()
        behavior = next(iter(env.behavior_specs))
        for step in range(260):
            decisions, terminals = env.get_steps(behavior)
            if len(terminals):
                reward = float(terminals.reward[0])
                lines = [line for line in log.read_text(errors="replace").splitlines()
                         if "TPS_EPISODE" in line and "reason=success" in line]
                if reward <= 0.5 or not lines:
                    raise RuntimeError(f"scripted route failed at decision {step}: reward={reward}")
                fields = dict(part.split("=", 1) for part in lines[-1].split()
                              if "=" in part)
                distance = float(fields["distance"])
                if distance > 0.9:
                    raise RuntimeError(f"success before contact: distance={distance}")
                found = fields["treasure_found"] == "True"
                if args.face_away and found:
                    raise RuntimeError("face-away contact unexpectedly saw the treasure")
                final = frame(_rgb_at(terminals.obs, 0), name, step,
                              "SUCCESS: PANEL CONTACT")
                for _ in range(10):
                    writer.write(final)
                report = {"mode": name, "scripted": True, "trained_policy": False,
                          "layout_id": layout["layout_id"], "terminal": "success",
                          "contact_distance_m": distance, "decision_frames": step,
                          "treasure_found_before_contact": found,
                          "video": video.name}
                break
            if len(decisions) != 1:
                raise RuntimeError("expected one preview agent")
            status = ("FACE AWAY" if args.face_away else "SEARCH") if step < 15 else "NAVIGATE"
            writer.write(frame(_rgb_at(decisions.obs, 0), name, step, status))
            action = np.zeros(4, dtype=np.float32)
            desired_yaw = math.degrees(math.atan2(target[0] - position[0],
                                                  target[1] - position[1]))
            if args.face_away:
                desired_yaw += 180.0
            action[2] = np.clip(wrap(desired_yaw - yaw) / 9.0, -1.0, 1.0)
            if step >= 15:
                while waypoint < len(path) - 1 and np.linalg.norm(
                        path[waypoint] - position) < 0.42:
                    waypoint += 1
                action[:2] = _motion(position, yaw, path[waypoint])
            env.set_actions(behavior, ActionTuple(continuous=action.reshape(1, 4)))
            env.step()
            position, velocity, yaw = advance(position, velocity, yaw, action)
        if report is None:
            raise RuntimeError("scripted route did not touch treasure in 260 decisions")
    finally:
        env.close()
        writer.release()
    (OUT / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
