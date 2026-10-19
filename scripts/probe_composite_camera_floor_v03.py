"""Capture extreme camera pitches in the fixed-boom treasure/composite players."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256
from tps_transfer.smoke import _rgb_at


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/primitive_contact_v03/camera_probe"


def snapshot(env: UnityEnvironment, behavior: str, agent_id: int,
             task: str, label: str) -> float:
    decisions, _ = env.get_steps(behavior)
    ids = list(map(int, decisions.agent_id))
    if agent_id not in ids:
        raise RuntimeError(f"{task} agent {agent_id} vanished before {label}")
    rgb = _rgb_at(decisions.obs, ids.index(agent_id))
    image = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
    path = OUT / f"{task}_{label}.png"
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cannot write {path}")
    sky = ((rgb[..., 0] > 0.42) & (rgb[..., 0] < 0.72)
           & (rgb[..., 1] > 0.63) & (rgb[..., 2] > 0.8))
    return float(sky.mean())


def one(task: str, worker_id: int) -> dict[str, object]:
    player = ROOT / "unity/Build" / (
        "TpsTreasure.x86_64" if task == "treasure" else "TpsComposite.x86_64")
    manifest = ROOT / "manifests/pilot_layouts.json"
    layout_manifest_sha256(player, manifest)
    env = UnityEnvironment(
        file_name=str(player), worker_id=worker_id, seed=0, no_graphics=False,
        additional_args=["-logFile", str(OUT / f"{task}.player.log"),
                         f"--tps-task={task}", "--tps-split=validation"],
    )
    try:
        env.reset()
        behavior = next(iter(env.behavior_specs))
        if not behavior.startswith(f"TPS_{task}"):
            raise RuntimeError(f"unexpected behavior: {behavior}")
        decisions, terminals = env.get_steps(behavior)
        if len(decisions) != 16 or len(terminals):
            raise RuntimeError(f"unexpected initial {task} agents")
        agent_id = int(decisions.agent_id[0])
        fractions = {"initial": snapshot(env, behavior, agent_id, task, "initial")}
        for direction, label, count in ((1.0, "pitch_plus60", 6),
                                         (-1.0, "pitch_minus60", 14)):
            for _ in range(count):
                decisions, _ = env.get_steps(behavior)
                actions = np.zeros((len(decisions), 4), dtype=np.float32)
                actions[:, 3] = direction
                env.set_actions(behavior, ActionTuple(continuous=actions))
                env.step()
            fractions[label] = snapshot(env, behavior, agent_id, task, label)
        if fractions["pitch_minus60"] < 0.2:
            raise AssertionError(f"{task} camera may be under floor: {fractions}")
        return {"task": task, "agent_id": agent_id,
                "sky_fraction": fractions,
                "result": "passed"}
    finally:
        env.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = [one("treasure", 90), one("composite", 91)]
    (OUT / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
