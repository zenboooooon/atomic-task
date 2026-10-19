"""Deterministically evaluate one checkpoint across all fixed validation arenas."""

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

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.episode import AgentKey, EpisodeTracker
from tps_transfer.model import ActorHead, ModelConfig, SharedEncoder
from tps_transfer.observation import FrameProcessor, history_batch
from tps_transfer.preprocess import DepthCalibration
from tps_transfer.smoke import _steps

from tps_transfer.run_metadata import git_state
from tps_transfer.primitive_variants import pretraining_player, task_metadata

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _treasure_diagnostics(player_log: Path, manifest_path: Path, split: str,
                          episodes_per_agent: int) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    layouts = [item for item in manifest["layouts"]
               if item["task"] == "treasure" and item["split"] == split]
    selected = {
        layouts[agent + 16 * ordinal]["layout_id"]: agent + 16 * ordinal
        for agent in range(16) for ordinal in range(episodes_per_agent)
    }
    outcomes = {}
    for line in player_log.read_text(errors="replace").splitlines():
        if "TPS_EPISODE " not in line or " task=treasure " not in line:
            continue
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        layout_id = fields.get("layout")
        reason = fields.get("reason", "")
        if layout_id in selected and layout_id not in outcomes and reason != "checkpoint":
            outcomes[layout_id] = fields
    if set(outcomes) != set(selected):
        raise RuntimeError("missing treasure terminal diagnostics for evaluation layouts")

    def summarize(ids: list[str]) -> dict[str, int]:
        rows = [outcomes[item] for item in ids]
        return {
            "episodes": len(rows),
            "found": sum(row.get("treasure_found", "False").lower() == "true" for row in rows),
            "successes": sum(row.get("reason") == "success" for row in rows),
            "collisions": sum(row.get("reason", "").startswith("collision_") for row in rows),
        }

    return {
        "all": summarize(list(selected)),
        "near_start_yaw": summarize([key for key, index in selected.items() if index % 2 == 0]),
        "wide_start_yaw": summarize([key for key, index in selected.items() if index % 2 == 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes-per-agent", type=int, default=2)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--layout-manifest", type=Path,
                        default=ROOT / "manifests/pilot_layouts.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.episodes_per_agent <= 8:
        parser.error("episodes per agent must be between 1 and 8 for pilot validation")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    player_log = output.with_suffix(".player.log")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stage = checkpoint.get("stage")
    if stage == "pretraining_pilot":
        if args.split == "test":
            parser.error("primitive source tasks have no test layouts in this pilot")
        task_names = tuple(checkpoint.get(
            "pretraining_tasks", ("look", "approach", "stop", "avoid")
        ))
        player, agent_count = pretraining_player(task_names)
        task_argument = "pretrain"
        layout_stride = agent_count
    elif stage == "downstream_pilot":
        task_argument = checkpoint.get("downstream_task", "composite")
        if task_argument not in ("composite", "treasure"):
            raise ValueError("unknown downstream task")
        player = "TpsTreasure" if task_argument == "treasure" else "TpsComposite"
        task_names = (task_argument,)
        agent_count = 16
        layout_stride = 16
    elif stage == "single_task_pilot" and checkpoint.get("player") in ("multi", "composite"):
        player = "TpsComposite"
        task_names = (checkpoint["task"],)
        task_argument = checkpoint["task"]
        agent_count = 16
        layout_stride = 16
    else:
        raise ValueError("checkpoint is not a multi-arena pilot")
    executable = ROOT / "unity/Build" / f"{player}.x86_64"
    if stage == "downstream_pilot" and checkpoint.get("unity_executable"):
        executable = ROOT / checkpoint["unity_executable"]
    if checkpoint.get("layout_manifest_sha256") != layout_manifest_sha256(
        executable, args.layout_manifest.resolve()
    ):
        raise ValueError("checkpoint and layout manifest differ")
    if checkpoint["unity_build_sha256"] != unity_build_sha256(executable):
        raise ValueError("checkpoint and Unity build differ")
    calibration_path = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    if checkpoint["calibration_sha256"] != _sha256(calibration_path):
        raise ValueError("checkpoint and depth calibration differ")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for visual evaluation")
    config = ModelConfig(**checkpoint["model_config"])
    shared = SharedEncoder(config).to(device).eval()
    shared.load_state_dict(checkpoint["shared"])
    shared.requires_grad_(False)
    actors = {}
    unique_actors = {}
    head_map = checkpoint.get("task_head_map", {task: task for task in task_names})
    for task in task_names:
        head = head_map[task] if stage == "pretraining_pilot" else task
        if head in unique_actors:
            actors[task] = unique_actors[head]
            continue
        actor = ActorHead(config.width).to(device).eval()
        actor.load_state_dict(checkpoint["actors"][head] if stage == "pretraining_pilot"
                              else checkpoint["actor"])
        actor.requires_grad_(False)
        actors[task] = actor
        unique_actors[head] = actor
    processor = FrameProcessor(DepthEstimator(392), DepthCalibration.load(calibration_path))
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=args.worker_id, seed=args.seed, no_graphics=False,
        additional_args=["-logFile", str(player_log),
                         f"--tps-task={task_argument}", f"--tps-split={args.split}"],
    )
    tracker = EpisodeTracker(worker_id=args.worker_id)
    behavior_map = {}
    completed: dict[str, dict[int, int]] = {}
    episode_returns: dict[AgentKey, float] = {}
    results: dict[str, list[dict[str, object]]] = {task: [] for task in task_names}
    action_sum = {task: np.zeros(5, dtype=np.float64) for task in task_names}
    action_abs_sum = {task: np.zeros(5, dtype=np.float64) for task in task_names}
    action_count = {task: 0 for task in task_names}
    started = time.perf_counter()
    calls = 0
    max_decisions = 400 if stage == "downstream_pilot" and task_argument == "composite" else 300
    max_calls = args.episodes_per_agent * max_decisions + 100
    try:
        environment.reset()
        for behavior in environment.behavior_specs:
            task = (task_names[0] if stage == "single_task_pilot" else
                    behavior.split("?", 1)[0].removeprefix("TPS_").lower())
            if task not in task_names:
                raise RuntimeError(f"unexpected behavior {behavior}")
            behavior_map[task] = behavior
            decisions, _ = environment.get_steps(behavior)
            ids = [int(item) for item in decisions.agent_id]
            if len(ids) != agent_count:
                raise RuntimeError(f"{task}: expected {agent_count} initial agents, got {ids}")
            completed[task] = {agent_id: 0 for agent_id in ids}
        if set(behavior_map) != set(task_names):
            raise RuntimeError("missing task behavior")
        while any(any(value < args.episodes_per_agent for value in completed[task].values())
                  for task in task_names):
            if calls >= max_calls:
                raise RuntimeError(f"validation did not finish after {calls} calls: {completed}")
            for task in task_names:
                behavior = behavior_map[task]
                decisions, terminals = environment.get_steps(behavior)
                transitions = tracker.observe(
                    behavior, _steps(decisions, False), _steps(terminals, True)
                )
                for transition in transitions:
                    key = transition.key
                    episode_returns[key] = episode_returns.get(key, 0.0) + transition.reward
                    if transition.terminated or transition.truncated:
                        ordinal = completed[task][key.agent_id]
                        if ordinal < args.episodes_per_agent:
                            results[task].append({
                                "agent_id": key.agent_id,
                                "episode_ordinal_for_agent": ordinal,
                                "cumulative_reward": episode_returns[key],
                                "final_transition_reward": transition.reward,
                                "interrupted": transition.truncated,
                                "success": transition.terminated and transition.reward > 0.5,
                            })
                        completed[task][key.agent_id] = ordinal + 1
                        episode_returns.pop(key)
                if len(decisions):
                    ids = [int(item) for item in decisions.agent_id]
                    active_rows = [row for row, agent_id in enumerate(ids)
                                   if completed[task][agent_id] < args.episodes_per_agent]
                    selected = np.zeros((len(ids), 5), dtype=np.float32)
                    if active_rows:
                        images, actions, valid = history_batch(
                            [tracker.history_for(behavior, ids[row]) for row in active_rows],
                            processor, device,
                        )
                        with torch.inference_mode():
                            mean, _ = actors[task](shared(images, actions, valid))
                            selected[active_rows] = torch.tanh(mean).cpu().numpy().astype(np.float32)
                    for row, agent_id in enumerate(ids):
                        tracker.record_action(behavior, agent_id, selected[row])
                    for row in active_rows:
                        action_sum[task] += selected[row]
                        action_abs_sum[task] += np.abs(selected[row])
                        action_count[task] += 1
                    environment.set_actions(behavior, ActionTuple(continuous=selected))
            environment.step()
            calls += 1
            if calls % 100 == 0:
                print(json.dumps({"env_step_calls": calls,
                                  "completed": completed}), flush=True)
    finally:
        environment.close()
    summary = {
        task: {
            "episodes": len(results[task]),
            "successes": sum(bool(item["success"]) for item in results[task]),
            "success_rate": sum(bool(item["success"]) for item in results[task]) / len(results[task]),
            "decision_count": action_count[task],
            "mean_action": (action_sum[task] / max(action_count[task], 1)).tolist(),
            "mean_abs_action": (action_abs_sum[task] / max(action_count[task], 1)).tolist(),
        }
        for task in task_names
    }
    treasure_diagnostics = (
        _treasure_diagnostics(player_log, args.layout_manifest.resolve(),
                              args.split, args.episodes_per_agent)
        if stage == "downstream_pilot" and task_argument == "treasure" else None
    )
    if treasure_diagnostics is not None:
        if treasure_diagnostics["all"]["successes"] != summary["treasure"]["successes"]:
            raise RuntimeError("treasure Unity logs and LLAPI success counts disagree")
    report = {
        "environment_variant": checkpoint.get("environment_variant"),
        "unity_executable": str(executable.relative_to(ROOT)),
        "stage": stage,
        "task_head_map": checkpoint.get("task_head_map"),
        "pretraining_head_keys": checkpoint.get("pretraining_head_keys"),
        **git_state(ROOT),
        "condition": checkpoint.get("condition"),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "agent_decisions_trained": checkpoint["agent_decisions"],
        "split": args.split,
        "layout_rule": f"arena start index 0..{agent_count - 1}; stride {layout_stride}",
        "episodes_per_agent": args.episodes_per_agent,
        "unity_build_sha256": unity_build_sha256(executable),
        "layout_manifest_sha256": checkpoint["layout_manifest_sha256"],
        "calibration_sha256": _sha256(calibration_path),
        "environment_seed": args.seed,
        "player_log": str(player_log.relative_to(ROOT)),
        "env_step_calls": calls,
        "wall_seconds": time.perf_counter() - started,
        "summary": summary,
        "task_variants": task_metadata(task_names) if stage == "pretraining_pilot" else None,
        "treasure_diagnostics": treasure_diagnostics,
        "results": results,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
