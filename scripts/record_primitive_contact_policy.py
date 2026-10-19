"""Record learned primitive-contact-v03 policy on deterministic validation episodes."""

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
CASES = {
    ("look", 8, 0): True,
    ("look", 8, 1): False,
    ("approach", 0, 0): True,
    ("approach", 0, 2): False,
    ("avoid", 2, 0): True,
    ("avoid", 9, 0): False,
}
SIZE = (448, 504)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(rgb: np.ndarray, task: str, success: bool, agent: int,
          ordinal: int, number: int, final: bool = False) -> np.ndarray:
    image = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    canvas[56:] = image
    label = "SUCCESS" if success else "FAILURE"
    color = (70, 200, 70) if success else (60, 80, 230)
    cv2.rectangle(canvas, (0, 56), (447, 503), color, 3)
    cv2.putText(canvas, f"{task.upper()}  {label}  |  agent {agent:02d} ep {ordinal}",
                (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.49, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Primitive contact v03  |  decision {number:03d}",
                (9, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (230, 230, 230), 1, cv2.LINE_AA)
    if final:
        cv2.rectangle(canvas, (0, 440), (447, 503), (0, 0, 0), -1)
        cv2.putText(canvas, label, (95, 488), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, color, 3, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker-id", type=int, default=74)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    reference_path = args.reference.resolve()
    output_dir = args.output_dir.resolve()
    staging = output_dir / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reference = json.loads(reference_path.read_text())
    executable = ROOT / "unity/Build/TpsPretrainThree.x86_64"
    manifest = ROOT / "manifests/primitive_contact_v03_layouts.json"
    calibration = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    tasks = ("look", "approach", "avoid")
    if checkpoint.get("stage") != "pretraining_pilot" or tuple(checkpoint.get("pretraining_tasks", ())) != tasks:
        raise ValueError("expected a three-task pretraining checkpoint")
    if checkpoint.get("agent_decisions") != 511872 or reference.get("checkpoint_sha256") != sha256(checkpoint_path):
        raise ValueError("checkpoint and reference do not match")
    if reference.get("environment_seed") != 9000 or reference.get("episodes_per_agent") != 8:
        raise ValueError("expected seed-9000, eight-episode validation reference")
    if checkpoint["unity_build_sha256"] != unity_build_sha256(executable):
        raise ValueError("Unity build differs from checkpoint")
    if checkpoint["layout_manifest_sha256"] != layout_manifest_sha256(executable, manifest):
        raise ValueError("layout manifest differs from checkpoint")
    if checkpoint["calibration_sha256"] != sha256(calibration):
        raise ValueError("depth calibration differs from checkpoint")
    expected = {
        (task, row["agent_id"], row["episode_ordinal_for_agent"]): bool(row["success"])
        for task in tasks for row in reference["results"][task]
    }
    if any(expected.get(key) is not value for key, value in CASES.items()):
        raise ValueError("selected video outcomes differ from reference")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model replay")
    device = torch.device("cuda")
    config = ModelConfig(**checkpoint["model_config"])
    shared = SharedEncoder(config).to(device).eval()
    shared.load_state_dict(checkpoint["shared"])
    shared.requires_grad_(False)
    actors = {}
    for task in tasks:
        actor = ActorHead(config.width).to(device).eval()
        actor.load_state_dict(checkpoint["actors"][task])
        actor.requires_grad_(False)
        actors[task] = actor
    processor = FrameProcessor(DepthEstimator(392), DepthCalibration.load(calibration))
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=args.worker_id, seed=9000, no_graphics=False,
        additional_args=["-logFile", str(staging / "replay.player.log"),
                         "--tps-task=pretrain", "--tps-split=validation"],
    )
    tracker = EpisodeTracker(worker_id=args.worker_id)
    filenames = {
        key: f"{key[0]}_{'success' if value else 'failure'}_agent{key[1]:02d}_episode{key[2]}.mp4"
        for key, value in CASES.items()
    }
    writers: dict[tuple[str, int, int], cv2.VideoWriter] = {}
    counts = {key: 0 for key in CASES}
    observed: dict[tuple[str, int, int], bool] = {}
    completed: dict[str, dict[int, int]] = {}
    try:
        environment.reset()
        behaviors = {behavior.split("?", 1)[0].removeprefix("TPS_").lower(): behavior
                     for behavior in environment.behavior_specs}
        if set(behaviors) != set(tasks):
            raise RuntimeError(f"unexpected behaviors: {behaviors}")
        for task in tasks:
            decisions, terminals = environment.get_steps(behaviors[task])
            if len(decisions) != 4 or len(terminals):
                raise RuntimeError(f"expected four initial {task} agents")
            completed[task] = {int(agent_id): 0 for agent_id in decisions.agent_id}
        calls = 0
        while set(observed) != set(CASES):
            if calls >= 800:
                raise RuntimeError(f"selected episodes did not finish: {observed}")
            for task in tasks:
                behavior = behaviors[task]
                decisions, terminals = environment.get_steps(behavior)
                terminal_rgb = {int(agent_id): _rgb_at(terminals.obs, row)
                                for row, agent_id in enumerate(terminals.agent_id)}
                transitions = tracker.observe(
                    behavior, _steps(decisions, False), _steps(terminals, True)
                )
                for transition in transitions:
                    if not (transition.terminated or transition.truncated):
                        continue
                    agent_id = transition.key.agent_id
                    ordinal = completed[task][agent_id]
                    key = (task, agent_id, ordinal)
                    if key in CASES:
                        outcome = bool(transition.terminated and transition.reward > 0.5)
                        observed[key] = outcome
                        if outcome != CASES[key]:
                            raise RuntimeError(f"reference outcome mismatch: {key}: {outcome}")
                        writer = writers.pop(key, None)
                        if writer is None:
                            raise RuntimeError(f"no video frames for {key}")
                        last = frame(terminal_rgb[agent_id], task, outcome,
                                     agent_id, ordinal, counts[key], final=True)
                        for _ in range(10):
                            writer.write(last)
                        writer.release()
                    completed[task][agent_id] = ordinal + 1
                if len(decisions):
                    ids = [int(agent_id) for agent_id in decisions.agent_id]
                    active = [row for row, agent_id in enumerate(ids)
                              if completed[task][agent_id] < 8]
                    actions = np.zeros((len(ids), 4), dtype=np.float32)
                    if active:
                        images, previous_actions, valid = history_batch(
                            [tracker.history_for(behavior, ids[row]) for row in active],
                            processor, device,
                        )
                        with torch.inference_mode():
                            mean, _ = actors[task](shared(images, previous_actions, valid))
                            actions[active] = torch.tanh(mean).cpu().numpy().astype(np.float32)
                    for row, agent_id in enumerate(ids):
                        key = (task, agent_id, completed[task][agent_id])
                        if key in CASES and key not in observed:
                            writer = writers.get(key)
                            if writer is None:
                                writer = cv2.VideoWriter(str(staging / filenames[key]),
                                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                                         10, SIZE)
                                if not writer.isOpened():
                                    raise RuntimeError(f"cannot open video writer: {key}")
                                writers[key] = writer
                            writer.write(frame(_rgb_at(decisions.obs, row), task,
                                               CASES[key], agent_id, key[2], counts[key]))
                            counts[key] += 1
                        tracker.record_action(behavior, agent_id, actions[row])
                    environment.set_actions(behavior, ActionTuple(continuous=actions))
            environment.step()
            calls += 1
            if calls % 100 == 0:
                print(f"replay calls={calls} videos={len(observed)}/{len(CASES)}", flush=True)
    finally:
        environment.close()
        for writer in writers.values():
            writer.release()

    videos = []
    for key, filename in filenames.items():
        path = staging / filename
        capture = cv2.VideoCapture(str(path))
        readable = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) >= counts[key]
        capture.release()
        if counts[key] < 2 or not readable:
            raise RuntimeError(f"recorded video is unreadable: {path}")
        path.replace(output_dir / filename)
        videos.append({"task": key[0], "agent_id": key[1], "episode_ordinal": key[2],
                       "success": CASES[key], "decision_frames": counts[key], "file": filename})
    report = {"checkpoint_sha256": sha256(checkpoint_path),
              "reference_sha256": sha256(reference_path),
              "selected_outcomes_match_reference": True,
              "validation_summary": {task: reference["summary"][task] for task in tasks},
              "videos": videos}
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
