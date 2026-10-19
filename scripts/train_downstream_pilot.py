"""Branch one pretraining checkpoint into matched S/R/P/F downstream pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.model import (
    ModelConfig, SharedEncoder, make_downstream_heads, module_hash,
)
from tps_transfer.observation import FrameProcessor
from tps_transfer.pilot import PilotCollector, update_pilot
from tps_transfer.ppo import downstream_optimizer
from tps_transfer.preprocess import DepthCalibration

from tps_transfer.run_metadata import git_state
from tps_transfer.primitive_variants import COLOR_TASKS, pretraining_player

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("S", "R", "P", "F")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(state: dict, path: Path) -> None:
    """Publish only complete checkpoints to the concurrent evaluator."""
    pending = path.with_suffix(path.suffix + ".pending")
    torch.save(state, pending)
    pending.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--pretrain-layout-manifest", type=Path,
                        help="source manifest for an unchanged pretraining player")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--conditions", default="S,R,P,F")
    parser.add_argument("--downstream-task", choices=("composite", "treasure"),
                        default="composite")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--rollout", type=int, default=256)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--shared-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--minibatch", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("runs/phase3/downstream_pilot"))
    parser.add_argument("--summary-output", type=Path,
                        help="separate summary path for parallel condition jobs")
    parser.add_argument("--resume", type=Path, help="continue one condition and seed from a checkpoint in a new output directory")
    args = parser.parse_args()
    seeds = [int(item) for item in args.seeds.split(",")]
    conditions = args.conditions.split(",")
    if not seeds or not conditions or any(item not in CONDITIONS for item in conditions):
        parser.error("invalid seeds or conditions")
    if args.resume is not None and (len(seeds) != 1 or len(conditions) != 1):
        parser.error("resume requires exactly one seed and one condition")
    if min(args.rollout, args.updates, args.checkpoint_every,
           args.epochs, args.minibatch) < 1:
        parser.error("all count options must be positive")
    if args.rollout % args.minibatch:
        parser.error("rollout must be divisible by minibatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for downstream pilot")
    pretrained_path = args.pretrained.resolve()
    pretrained = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    if pretrained.get("stage") != "pretraining_pilot":
        raise ValueError("expected a pretraining pilot checkpoint")
    source_tasks = pretrained.get("pretraining_tasks", ["look", "approach", "stop", "avoid"])
    expected_tasks = (["look", "approach", "jump"] if args.downstream_task == "treasure"
                      else ["look", "approach", "stop", "avoid"])
    if source_tasks != expected_tasks and not (
        args.downstream_task == "treasure" and tuple(source_tasks) == COLOR_TASKS
    ):
        raise ValueError("pretraining task set does not match downstream protocol")
    pretrain_player, _ = pretraining_player(tuple(source_tasks))
    pretrain_build = ROOT / "unity/Build" / f"{pretrain_player}.x86_64"
    downstream_manifest = ROOT / "manifests/pilot_layouts.json"
    pretrain_manifest = (args.pretrain_layout_manifest.resolve()
                         if args.pretrain_layout_manifest is not None else downstream_manifest)
    pretrain_layout_sha = layout_manifest_sha256(pretrain_build, pretrain_manifest)
    if pretrained.get("layout_manifest_sha256") != pretrain_layout_sha:
        raise ValueError("pretraining layout manifest changed since checkpoint")
    if pretrained["unity_build_sha256"] != unity_build_sha256(pretrain_build):
        raise ValueError("pretraining Unity build changed since checkpoint")
    calibration_path = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    calibration_sha = _sha256(calibration_path)
    if pretrained["calibration_sha256"] != calibration_sha:
        raise ValueError("depth calibration mismatch")
    downstream_player = "TpsTreasure" if args.downstream_task == "treasure" else "TpsComposite"
    executable = ROOT / "unity/Build" / f"{downstream_player}.x86_64"
    composite_build_sha = unity_build_sha256(executable)
    layout_sha = layout_manifest_sha256(executable, downstream_manifest)
    if args.pretrain_layout_manifest is None and layout_sha != pretrain_layout_sha:
        raise ValueError("composite and pretraining layouts differ")
    if args.pretrain_layout_manifest is not None:
        if args.downstream_task != "treasure":
            raise ValueError("separate source manifest is only supported for treasure")
        source = json.loads(pretrain_manifest.read_text())
        downstream = json.loads(downstream_manifest.read_text())
        for task in expected_tasks:
            old = [{key: value for key, value in item.items() if key != "layout_id"}
                   for item in source["layouts"] if item["task"] == task]
            new = [{key: value for key, value in item.items() if key != "layout_id"}
                   for item in downstream["layouts"] if item["task"] == task]
            if old != new:
                raise ValueError(f"source-task geometry changed for {task}")
    device = torch.device("cuda")
    config = ModelConfig(**pretrained["model_config"])
    depth = DepthEstimator(pretrained["depth_input_size"])
    calibration = DepthCalibration.load(calibration_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_state = git_state(ROOT)
    all_reports = []
    pretrained_sha = _sha256(pretrained_path)
    training_config = {
        "downstream_task": args.downstream_task,
        "rollout": args.rollout, "epochs": args.epochs,
        "minibatch": args.minibatch, "shared_lr": args.shared_lr,
        "head_lr": args.head_lr,
    }
    for seed in seeds:
        network_seed = 20_000 + seed
        head_seed = 30_000 + seed
        environment_seed = 10_000 + seed
        action_seed = 40_000 + seed
        baseline_distribution = None
        for condition in conditions:
            torch.manual_seed(network_seed)
            torch.cuda.manual_seed_all(network_seed)
            shared = SharedEncoder(config).to(device)
            if condition in ("P", "F"):
                shared.load_state_dict(pretrained["shared"])
            actor, critic = make_downstream_heads(head_seed, config.width)
            actor = actor.to(device)
            critic = critic.to(device)
            train_shared = condition in ("S", "F")
            optimizer = downstream_optimizer(
                shared, actor, critic, train_shared,
                shared_lr=args.shared_lr, head_lr=args.head_lr,
            )
            initial_shared_hash = module_hash(shared)
            with torch.inference_mode():
                mean, log_std = actor(torch.zeros(1, config.width, device=device))
            distribution = {
                "mean": mean[0].cpu().tolist(),
                "log_std": log_std[0].cpu().tolist(),
            }
            if baseline_distribution is None:
                baseline_distribution = distribution
            elif distribution != baseline_distribution:
                raise RuntimeError("matched conditions have different initial action distributions")
            torch.manual_seed(action_seed)
            torch.cuda.manual_seed_all(action_seed)
            processor = FrameProcessor(depth, calibration)
            run_dir = output / f"{condition}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            first_update = 0
            resume_source = None
            if args.resume is not None:
                resume_path = args.resume.resolve()
                if resume_path.parent == run_dir:
                    raise ValueError("resume output must be a new directory")
                previous = torch.load(resume_path, map_location="cpu", weights_only=False)
                required = {
                    "stage": "downstream_pilot", "condition": condition,
                    "seed": seed, "model_config": config.__dict__,
                    "pretrained_sha256": pretrained_sha,
                    "calibration_sha256": calibration_sha,
                    "unity_build_sha256": composite_build_sha,
                    "layout_manifest_sha256": layout_sha,
                    "pretrain_layout_manifest_sha256": pretrain_layout_sha,
                    "downstream_task": args.downstream_task,
                }
                if any(previous.get(key) != value for key, value in required.items()):
                    raise ValueError("resume checkpoint does not match downstream configuration or assets")
                if (previous.get("training_config") is not None
                        and previous["training_config"] != training_config):
                    raise ValueError("resume training hyperparameters differ from checkpoint")
                first_update = int(previous["update"])
                if first_update < 0 or first_update >= args.updates:
                    raise ValueError("total requested updates must exceed resume update")
                if int(previous["agent_decisions"]) != first_update * args.rollout:
                    raise ValueError("resume decisions do not match rollout size")
                shared.load_state_dict(previous["shared"])
                actor.load_state_dict(previous["actor"])
                critic.load_state_dict(previous["critic"])
                optimizer.load_state_dict(previous["optimizer"])
                initial_shared_hash = previous["initial_shared_hash"]
                if not train_shared and module_hash(shared) != initial_shared_hash:
                    raise ValueError("frozen shared encoder differs from initial checkpoint")
                torch.set_rng_state(previous["torch_rng"])
                torch.cuda.set_rng_state_all(previous["cuda_rng"])
                resume_source = {
                    "checkpoint": str(resume_path.relative_to(ROOT)),
                    "sha256": _sha256(resume_path),
                    "environment_restarted": True,
                    "previous_agent_decisions": int(previous["agent_decisions"]),
                    "legacy_training_config_unverified": "training_config" not in previous,
                }
            else:
                _save_checkpoint({
                    "stage": "downstream_pilot", "condition": condition,
                    "downstream_task": args.downstream_task,
                    "seed": seed, "update": 0, "agent_decisions": 0,
                    **code_state,
                    "shared": shared.state_dict(), "actor": actor.state_dict(),
                    "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                    "model_config": config.__dict__,
                    "training_config": training_config,
                    "pretrained_sha256": pretrained_sha,
                    "calibration_sha256": calibration_sha,
                    "unity_build_sha256": composite_build_sha,
                    "layout_manifest_sha256": layout_sha,
                    "pretrain_layout_manifest_sha256": pretrain_layout_sha,
                    "initial_shared_hash": initial_shared_hash,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                }, run_dir / "checkpoint_0000.pt")
            started = time.perf_counter()
            records = []
            collector = PilotCollector(executable, args.downstream_task, environment_seed,
                                       processor, shared, actor, critic, device,
                                       worker_id=args.worker_id)
            try:
                for update in range(first_update, args.updates):
                    rollout_started = time.perf_counter()
                    rollout = collector.collect(args.rollout, gamma=0.99, gae_lambda=0.95)
                    rollout_seconds = time.perf_counter() - rollout_started
                    training_started = time.perf_counter()
                    metrics = update_pilot(
                        rollout, processor, shared, actor, critic, optimizer,
                        device, args.epochs, args.minibatch,
                    )
                    training_seconds = time.perf_counter() - training_started
                    processor.clear()
                    if not train_shared and module_hash(shared) != initial_shared_hash:
                        raise RuntimeError("frozen shared encoder changed")
                    record = {
                        "update": update + 1,
                        "agent_decisions_trained": (update + 1) * args.rollout,
                        "rollout_seconds": rollout_seconds,
                        "training_seconds": training_seconds,
                        "successes": rollout.terminal_successes,
                        "failures": rollout.terminal_failures,
                        "interruptions": rollout.interruptions,
                        "surplus_transitions_discarded": rollout.surplus_transitions_discarded,
                        **metrics,
                    }
                    records.append(record)
                    print(json.dumps({"condition": condition, "seed": seed, **record}), flush=True)
                    if (update + 1) % args.checkpoint_every == 0 or update + 1 == args.updates:
                        _save_checkpoint({
                            "stage": "downstream_pilot", "condition": condition,
                            "downstream_task": args.downstream_task,
                            "seed": seed, "update": update + 1,
                            **code_state,
                            "agent_decisions": record["agent_decisions_trained"],
                            "shared": shared.state_dict(), "actor": actor.state_dict(),
                            "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                            "model_config": config.__dict__,
                            "training_config": training_config,
                            "pretrained_sha256": pretrained_sha,
                            "calibration_sha256": calibration_sha,
                            "unity_build_sha256": composite_build_sha,
                            "layout_manifest_sha256": layout_sha,
                            "pretrain_layout_manifest_sha256": pretrain_layout_sha,
                            "initial_shared_hash": initial_shared_hash,
                            "resume_source": resume_source,
                            "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(),
                        }, run_dir / f"checkpoint_{update + 1:04d}.pt")
            finally:
                collector.close()
            report = {
                "condition": condition,
                "downstream_task": args.downstream_task,
                "seed": seed,
                **code_state,
                "network_seed": network_seed,
                "head_seed": head_seed,
                "environment_seed": environment_seed,
                "action_seed": action_seed,
                "pretrained_checkpoint": str(pretrained_path.relative_to(ROOT)),
                "pretrained_sha256": pretrained_sha,
                "initial_action_distribution": distribution,
                "initial_shared_hash": initial_shared_hash,
                "final_shared_hash": module_hash(shared),
                "resume_source": resume_source,
                "shared_frozen": not train_shared,
                "shared_lr": args.shared_lr, "head_lr": args.head_lr,
                "training_config": training_config,
                "calibration_sha256": calibration_sha,
                "unity_build_sha256": composite_build_sha,
                "layout_manifest_sha256": layout_sha,
                "pretrain_layout_manifest_sha256": pretrain_layout_sha,
                "wall_seconds": time.perf_counter() - started,
                "updates": records,
            }
            (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            all_reports.append(report)
            del collector, processor, optimizer, actor, critic, shared
    summary = {
        "stage": "downstream_pilot", "pretrained_sha256": pretrained_sha,
        "seeds": seeds, "conditions": conditions,
        "runs": [{"condition": item["condition"], "seed": item["seed"],
                  "final_shared_hash": item["final_shared_hash"],
                  "wall_seconds": item["wall_seconds"]} for item in all_reports],
    }
    summary_path = args.summary_output.resolve() if args.summary_output else output / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
