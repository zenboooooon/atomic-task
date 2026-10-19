"""Replay deterministic treasure validation and record six agent-camera videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.episode import EpisodeTracker
from tps_transfer.model import ActorHead, ModelConfig, SharedEncoder
from tps_transfer.observation import FrameProcessor, history_batch
from tps_transfer.preprocess import DepthCalibration
from tps_transfer.smoke import _rgb_at, _steps


ROOT = Path(__file__).resolve().parents[1]
SELECTED = (
    ((1, 0), True),
    ((4, 0), True),
    ((0, 1), True),
    ((2, 0), False),
    ((13, 0), False),
    ((1, 1), False),
)
FRAME_SIZE = (448, 504)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(rgb: np.ndarray, label: str, agent: int, ordinal: int,
           frame_number: int, final: bool = False) -> np.ndarray:
    image = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    canvas[56:, :] = image
    color = (70, 200, 70) if label == "SUCCESS" else (60, 80, 230)
    cv2.rectangle(canvas, (0, 56), (447, 503), color, 3)
    cv2.putText(canvas, f"{label}  |  agent {agent:02d}  episode {ordinal}",
                (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"P update 2000  |  decision {frame_number:03d}",
                (9, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (230, 230, 230), 1, cv2.LINE_AA)
    if final:
        cv2.rectangle(canvas, (0, 440), (447, 503), (0, 0, 0), -1)
        cv2.putText(canvas, label, (95, 488), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, color, 3, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=52)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()
    staging = output_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference = json.loads(reference_path.read_text())
    executable = ROOT / "unity/Build/TpsTreasure.x86_64"
    calibration_path = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    if checkpoint.get("stage") != "downstream_pilot" or checkpoint.get("condition") != "P":
        raise ValueError("expected the P downstream checkpoint")
    if checkpoint.get("update") != 2000 or checkpoint.get("downstream_task") != "treasure":
        raise ValueError("expected the final treasure checkpoint")
    if reference.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("reference and checkpoint do not match")
    if reference.get("environment_seed") != 9000 or reference.get("episodes_per_agent") != 2:
        raise ValueError("reference must use the standard deterministic validation protocol")
    if checkpoint["unity_build_sha256"] != unity_build_sha256(executable):
        raise ValueError("Unity build differs from checkpoint")
    if checkpoint["layout_manifest_sha256"] != layout_manifest_sha256(
        executable, ROOT / "manifests/pilot_layouts.json"
    ):
        raise ValueError("layout manifest differs from checkpoint")
    if checkpoint["calibration_sha256"] != sha256(calibration_path):
        raise ValueError("depth calibration differs from checkpoint")

    expected = {
        (row["agent_id"], row["episode_ordinal_for_agent"]): bool(row["success"])
        for row in reference["results"]["treasure"]
    }
    for key, success in SELECTED:
        if expected.get(key) is not success:
            raise ValueError(f"selected case does not match reference: {key}")
    selected = dict(SELECTED)
    names = {
        key: f"{'success' if success else 'failure'}_{number:02d}_agent{key[0]:02d}_episode{key[1]}.mp4"
        for number, (key, success) in enumerate(SELECTED, 1)
    }

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model replay")
    config = ModelConfig(**checkpoint["model_config"])
    shared = SharedEncoder(config).to(device).eval()
    shared.load_state_dict(checkpoint["shared"])
    actor = ActorHead(config.width).to(device).eval()
    actor.load_state_dict(checkpoint["actor"])
    shared.requires_grad_(False)
    actor.requires_grad_(False)
    processor = FrameProcessor(DepthEstimator(392), DepthCalibration.load(calibration_path))
    player_log = staging / "replay.player.log"
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=args.worker_id, seed=9000,
        no_graphics=False, additional_args=[
            "-logFile", str(player_log), "--tps-task=treasure", "--tps-split=validation"
        ],
    )
    tracker = EpisodeTracker(worker_id=args.worker_id)
    writers: dict[tuple[int, int], cv2.VideoWriter] = {}
    frame_counts = {key: 0 for key in selected}
    outcomes: dict[tuple[int, int], bool] = {}
    completed: dict[int, int] = {}
    try:
        environment.reset()
        (behavior,) = environment.behavior_specs
        initial, terminals = environment.get_steps(behavior)
        if len(initial) != 16 or len(terminals):
            raise RuntimeError("expected 16 initial treasure agents")
        completed = {int(agent_id): 0 for agent_id in initial.agent_id}
        calls = 0
        while any(ordinal < 2 for ordinal in completed.values()):
            if calls >= 700:
                raise RuntimeError("validation replay did not finish")
            decisions, terminals = environment.get_steps(behavior)
            terminal_rgb = {
                int(agent_id): _rgb_at(terminals.obs, row)
                for row, agent_id in enumerate(terminals.agent_id)
            }
            transitions = tracker.observe(
                behavior, _steps(decisions, False), _steps(terminals, True)
            )
            for transition in transitions:
                if not (transition.terminated or transition.truncated):
                    continue
                agent_id = transition.key.agent_id
                ordinal = completed[agent_id]
                key = (agent_id, ordinal)
                if ordinal < 2:
                    outcomes[key] = bool(transition.terminated and transition.reward > 0.5)
                if key in selected:
                    writer = writers.get(key)
                    if writer is None:
                        raise RuntimeError(f"no initial video frame for {key}")
                    frame = render(terminal_rgb[agent_id],
                                   "SUCCESS" if selected[key] else "FAILURE",
                                   agent_id, ordinal, frame_counts[key], final=True)
                    for _ in range(10):
                        writer.write(frame)
                    writer.release()
                    del writers[key]
                completed[agent_id] = ordinal + 1
            if len(decisions):
                ids = [int(agent_id) for agent_id in decisions.agent_id]
                active_rows = [row for row, agent_id in enumerate(ids)
                               if completed[agent_id] < 2]
                actions = np.zeros((len(ids), 4), dtype=np.float32)
                if active_rows:
                    images, previous_actions, valid = history_batch(
                        [tracker.history_for(behavior, ids[row]) for row in active_rows],
                        processor, device,
                    )
                    with torch.inference_mode():
                        mean, _ = actor(shared(images, previous_actions, valid))
                        actions[active_rows] = torch.tanh(mean).cpu().numpy().astype(np.float32)
                for row, agent_id in enumerate(ids):
                    key = (agent_id, completed[agent_id])
                    if key in selected:
                        writer = writers.get(key)
                        if writer is None:
                            writer = cv2.VideoWriter(str(staging / names[key]),
                                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                                     10, FRAME_SIZE)
                            if not writer.isOpened():
                                raise RuntimeError(f"could not open MP4 writer for {key}")
                            writers[key] = writer
                        frame = render(_rgb_at(decisions.obs, row),
                                       "SUCCESS" if selected[key] else "FAILURE",
                                       agent_id, key[1], frame_counts[key])
                        writer.write(frame)
                        frame_counts[key] += 1
                    tracker.record_action(behavior, agent_id, actions[row])
                environment.set_actions(behavior, ActionTuple(continuous=actions))
            environment.step()
            calls += 1
            if calls % 100 == 0:
                print(f"replay calls={calls} completed={sum(min(x,2) for x in completed.values())}/32",
                      flush=True)
    finally:
        environment.close()
        for writer in writers.values():
            writer.release()

    if outcomes != expected:
        differences = {str(key): (expected.get(key), outcomes.get(key))
                       for key in expected if expected.get(key) != outcomes.get(key)}
        raise RuntimeError(f"replay outcomes differ from reference: {differences}")
    for key, filename in names.items():
        path = staging / filename
        if frame_counts[key] < 2 or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing recorded frames for {key}")
        capture = cv2.VideoCapture(str(path))
        readable = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) >= frame_counts[key]
        capture.release()
        if not readable:
            raise RuntimeError(f"MP4 could not be read: {path}")
        path.replace(output_dir / filename)
    manifest = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "all_32_outcomes_match": True,
        "videos": [
            {"file": names[key], "agent_id": key[0], "episode_ordinal": key[1],
             "success": success, "decision_frames": frame_counts[key]}
            for key, success in SELECTED
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
