"""Record first look success/failure using a frozen checkpoint and training sampling."""

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
from tps_transfer.model import ActorHead, ModelConfig, SharedEncoder, TanhGaussian
from tps_transfer.observation import FrameProcessor, history_batch
from tps_transfer.preprocess import DepthCalibration
from tps_transfer.smoke import _rgb_at, _steps

ROOT = Path(__file__).resolve().parents[1]
SIZE = (448, 528)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(rgb: np.ndarray, seed: int, update: int, decision: int,
          outcome: str, reason: str, final: bool = False) -> np.ndarray:
    canvas = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    canvas[80:] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (448, 448))
    color = (75, 220, 100) if outcome == "success" else (80, 115, 245)
    lines = (f"LOOK v0.7 | seed {seed} | update {update}",
             "LEARNED POLICY | stochastic | train layouts",
             f"{outcome.upper()} | decision {decision:03d}")
    for y, text in zip((21, 44, 68), lines):
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    color if y == 68 else (230, 230, 230), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (224, 304), (220, 220, 220), cv2.MARKER_CROSS, 12, 1)
    if final:
        cv2.rectangle(canvas, (0, 470), (447, 527), (0, 0, 0), -1)
        cv2.putText(canvas, reason.replace("_", " ").upper(), (12, 505),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=220)
    parser.add_argument("--sampling-seed", type=int, default=20261017)
    parser.add_argument("--max-calls", type=int, default=1200)
    args = parser.parse_args()
    path = args.checkpoint.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "pretraining_pilot"
    assert tuple(checkpoint["pretraining_tasks"]) == ("look", "approach")
    executable = ROOT / "unity/Build/TpsPretrainTwo.x86_64"
    manifest = ROOT / "manifests/gaze_v07_layouts.json"
    calibration = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    build_hash = unity_build_sha256(executable)
    assert checkpoint["unity_build_sha256"] == build_hash
    assert checkpoint["layout_manifest_sha256"] == layout_manifest_sha256(executable, manifest)
    assert checkpoint["calibration_sha256"] == sha256(calibration)
    seed = int(checkpoint["seed"])
    update = int(checkpoint["agent_decisions"]) // 256
    torch.manual_seed(args.sampling_seed)
    torch.cuda.manual_seed_all(args.sampling_seed)
    device = torch.device("cuda")
    config = ModelConfig(**checkpoint["model_config"])
    shared = SharedEncoder(config).to(device).eval()
    shared.load_state_dict(checkpoint["shared"])
    shared.requires_grad_(False)
    actors = {}
    for task in ("look", "approach"):
        actor = ActorHead(config.width).to(device).eval()
        actor.load_state_dict(checkpoint["actors"][task])
        actor.requires_grad_(False)
        actors[task] = actor
    processor = FrameProcessor(DepthEstimator(392), DepthCalibration.load(calibration))
    log = output / "replay.player.log"
    env = UnityEnvironment(file_name=str(executable), worker_id=args.worker_id, seed=seed,
                           no_graphics=False, additional_args=["-logFile", str(log),
                                                              "--tps-task=pretrain", "--tps-split=train"])
    tracker = EpisodeTracker(worker_id=args.worker_id)
    buffers = {}
    action_buffers = {}
    returns = {}
    episodes = []
    selected = {}
    try:
        env.reset()
        behaviors = {b.split("?", 1)[0].removeprefix("TPS_"): b for b in env.behavior_specs}
        assert set(behaviors) == set(actors)
        for calls in range(args.max_calls):
            for task, behavior in behaviors.items():
                decisions, terminals = env.get_steps(behavior)
                terminal_rgb = {int(agent): _rgb_at(terminals.obs, row)
                                for row, agent in enumerate(terminals.agent_id)}
                transitions = tracker.observe(behavior, _steps(decisions, False), _steps(terminals, True))
                for transition in transitions:
                    if task != "look":
                        continue
                    key = transition.key
                    returns[key] = returns.get(key, 0.0) + transition.reward
                    if not (transition.terminated or transition.truncated):
                        continue
                    success = bool(transition.terminated and transition.reward > .5)
                    outcome = "success" if success else "failure"
                    # Look has only success, timeout, and arena-boundary terminals.
                    reason = "central_hold_0.5s" if success else (
                        "time_limit" if transition.truncated else "out_of_bounds")
                    images = buffers.pop(key)
                    episode_actions = action_buffers.pop(key)
                    info = {"agent_id": key.agent_id, "episode_ordinal": key.episode_id,
                            "success": success, "interrupted": transition.truncated,
                            "reason": reason, "decision_frames": len(images),
                            "cumulative_reward": returns.pop(key)}
                    episodes.append(info)
                    if outcome not in selected:
                        filename = f"look_{outcome}_seed{seed}_update{update:04d}.mp4"
                        writer = cv2.VideoWriter(str(output / filename), cv2.VideoWriter_fourcc(*"mp4v"), 10, SIZE)
                        if not writer.isOpened():
                            raise RuntimeError("video writer failed")
                        for number, rgb in enumerate(images):
                            writer.write(frame(rgb, seed, update, number, outcome, reason))
                        last = np.rint(terminal_rgb[key.agent_id] * 255).clip(0, 255).astype(np.uint8)
                        for _ in range(15):
                            writer.write(frame(last, seed, update, len(images), outcome, reason, final=True))
                        writer.release()
                        np.save(output / f"look_{outcome}_actions.npy", np.stack(episode_actions))
                        selected[outcome] = dict(info, file=filename)
                        print(json.dumps({"selected": outcome, **info}), flush=True)
                actions = np.zeros((len(decisions), 4), dtype=np.float32)
                ids = [int(agent) for agent in decisions.agent_id]
                if ids:
                    images, previous, valid = history_batch(
                        [tracker.history_for(behavior, agent) for agent in ids], processor, device)
                    with torch.inference_mode():
                        mean, log_std = actors[task](shared(images, previous, valid))
                        sampled, _ = TanhGaussian(mean, log_std).sample()
                        actions = sampled.cpu().numpy().astype(np.float32)
                    for row, agent in enumerate(ids):
                        key = tracker.key_for(behavior, agent)
                        if task == "look":
                            rgb = np.rint(_rgb_at(decisions.obs, row) * 255).clip(0, 255).astype(np.uint8)
                            buffers.setdefault(key, []).append(rgb)
                            action_buffers.setdefault(key, []).append(actions[row].copy())
                        tracker.record_action(behavior, agent, actions[row])
                    env.set_actions(behavior, ActionTuple(continuous=actions))
            if len(selected) == 2:
                break
            env.step()
            if (calls + 1) % 50 == 0:
                processor.clear()
                print(json.dumps({"calls": calls + 1, "look_episodes": len(episodes),
                                  "successes": sum(e['success'] for e in episodes),
                                  "selected": list(selected)}), flush=True)
        if len(selected) != 2:
            raise RuntimeError(f"did not find both outcomes after {args.max_calls} calls")
    finally:
        env.close()
    report = {"checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": checkpoint_hash,
              "seed": seed, "update": update, "build_sha256": build_hash,
              "split": "train", "environment_seed": seed, "sampling_seed": args.sampling_seed,
              "action_policy": "TanhGaussian.sample, same as training collector",
              "recording_type": "fresh rollout from frozen training checkpoint",
              "selection": "first completed success and first completed failure",
              "training_continued_separately": True, "weight_updates_during_recording": False,
              "episodes": episodes, "videos": list(selected.values())}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
