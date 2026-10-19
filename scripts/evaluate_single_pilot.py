"""Evaluate a single-task pilot checkpoint on untouched validation layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.episode import AgentKey, EpisodeTracker
from tps_transfer.model import ActorHead, CriticHead, ModelConfig, SharedEncoder
from tps_transfer.observation import FrameProcessor, history_batch
from tps_transfer.preprocess import DepthCalibration
from tps_transfer.smoke import _steps

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    task = checkpoint["task"]
    if not 1 <= args.episodes <= 32:
        parser.error("episodes must be in [1, 32] for the pilot validation split")
    calibration_path = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    executable = ROOT / "unity/Build/TpsTask.x86_64"
    if checkpoint["calibration_sha256"] != _sha256(calibration_path):
        raise ValueError("checkpoint calibration mismatch")
    if checkpoint["unity_build_sha256"] != unity_build_sha256(executable):
        raise ValueError("checkpoint Unity build mismatch")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pilot evaluation")
    processor = FrameProcessor(DepthEstimator(392), DepthCalibration.load(calibration_path))
    config = ModelConfig(**checkpoint["model_config"])
    shared = SharedEncoder(config).to(device).eval()
    actor = ActorHead(config.width).to(device).eval()
    shared.load_state_dict(checkpoint["shared"])
    actor.load_state_dict(checkpoint["actor"])
    shared.requires_grad_(False)
    actor.requires_grad_(False)
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=0, seed=checkpoint["seed"],
        no_graphics=False,
        additional_args=[f"--tps-task={task}", "--tps-split=validation"],
    )
    tracker = EpisodeTracker(worker_id=0)
    results = []
    episode_returns: dict[AgentKey, float] = {}
    started = time.perf_counter()
    try:
        environment.reset()
        if len(environment.behavior_specs) != 1:
            raise RuntimeError("expected one behavior")
        behavior = next(iter(environment.behavior_specs))
        while len(results) < args.episodes:
            decisions, terminals = environment.get_steps(behavior)
            transitions = tracker.observe(
                behavior, _steps(decisions, False), _steps(terminals, True)
            )
            for transition in transitions:
                key = transition.key
                episode_returns[key] = episode_returns.get(key, 0.0) + transition.reward
                if transition.terminated or transition.truncated:
                    result = {
                        "layout_index": len(results),
                        "cumulative_reward": episode_returns.pop(key),
                        "final_transition_reward": transition.reward,
                        "interrupted": transition.truncated,
                        "success": transition.reward > 0.5 and transition.terminated,
                    }
                    results.append(result)
                    if len(results) >= args.episodes:
                        break
            if len(results) >= args.episodes:
                break
            if len(decisions):
                ids = [int(agent_id) for agent_id in decisions.agent_id]
                images, actions, valid = history_batch(
                    [tracker.history_for(behavior, agent_id) for agent_id in ids],
                    processor, device,
                )
                with torch.inference_mode():
                    mean, _ = actor(shared(images, actions, valid))
                    selected = torch.tanh(mean).cpu().numpy().astype(np.float32)
                for row, agent_id in enumerate(ids):
                    tracker.record_action(behavior, agent_id, selected[row])
                environment.set_actions(behavior, ActionTuple(continuous=selected))
            environment.step()
    finally:
        environment.close()
    report = {
        "task": task,
        "split": "validation",
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "agent_decisions_trained": checkpoint["agent_decisions"],
        "episodes": args.episodes,
        "successes": sum(item["success"] for item in results),
        "success_rate": sum(item["success"] for item in results) / args.episodes,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
