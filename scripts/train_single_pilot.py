"""Run a bounded single-task PPO integration pilot on the rendered Unity player."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.model import ActorHead, CriticHead, ModelConfig, SharedEncoder, module_hash
from tps_transfer.observation import FrameProcessor
from tps_transfer.pilot import PilotCollector, update_pilot
from tps_transfer.ppo import downstream_optimizer, pretraining_optimizer
from tps_transfer.preprocess import DepthCalibration

from tps_transfer.run_metadata import git_state

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("look", "approach", "stop", "avoid", "composite"), default="look")
    parser.add_argument("--player", choices=("task", "multi", "composite"), default="task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--shared-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--move-log-std", type=float, default=-0.5)
    parser.add_argument("--camera-log-std", type=float, default=-0.5)
    parser.add_argument("--minibatch", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("runs/phase2/single_pilot"))
    parser.add_argument("--resume", type=Path, help="checkpoint to continue a pilot after a fresh Unity reset")
    parser.add_argument("--shared-from", type=Path, help="initialize the shared encoder from a four-task checkpoint")
    parser.add_argument("--freeze-shared", action="store_true", help="train only the task heads")
    parser.add_argument("--zero-head-output", action="store_true", help="match new-head initial action distributions")
    args = parser.parse_args()
    if min(args.rollout, args.updates, args.epochs, args.minibatch) < 1:
        parser.error("rollout, updates, epochs, and minibatch must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for visual PPO pilot")
    calibration_path = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    calibration = DepthCalibration.load(calibration_path)
    processor = FrameProcessor(DepthEstimator(input_size=392), calibration)
    config = ModelConfig()
    shared = SharedEncoder(config).to(device)
    actor = ActorHead(config.width, zero_output=args.zero_head_output).to(device)
    initial_log_std = [args.move_log_std] * 2 + [args.camera_log_std] * 2
    with torch.no_grad():
        actor.log_std.copy_(torch.tensor(initial_log_std, device=device))
    critic = CriticHead(config.width, zero_output=args.zero_head_output).to(device)
    executable = ROOT / "unity/Build" / (
        "TpsComposite.x86_64" if args.player != "task" else "TpsTask.x86_64"
    )
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_state = git_state(ROOT)
    build_sha256 = unity_build_sha256(executable)
    layout_sha256 = layout_manifest_sha256(executable, ROOT / "manifests/pilot_layouts.json")
    calibration_sha256 = _sha256(calibration_path)
    shared_source_sha256 = None
    shared_source_checkpoint = None
    if args.shared_from is not None:
        source_path = args.shared_from.resolve()
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        expected = {
            "stage": "pretraining_pilot",
            "model_config": config.__dict__,
            "calibration_sha256": calibration_sha256,
            "layout_manifest_sha256": layout_sha256,
            "unity_build_sha256": unity_build_sha256(ROOT / "unity/Build/TpsPretrain.x86_64"),
        }
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("shared initialization checkpoint does not match pilot assets")
        shared.load_state_dict(source["shared"])
        shared_source_sha256 = _sha256(source_path)
        shared_source_checkpoint = str(source_path.relative_to(ROOT))
    initialization = {
        "shared_source_checkpoint": shared_source_checkpoint,
        "shared_source_sha256": shared_source_sha256,
        "freeze_shared": args.freeze_shared,
        "zero_head_output": args.zero_head_output,
    }
    if args.freeze_shared:
        optimizer = downstream_optimizer(
            shared, actor, critic, train_shared=False,
            shared_lr=args.shared_lr, head_lr=args.head_lr,
        )
    else:
        optimizer = pretraining_optimizer(
            shared, {args.task: actor}, {args.task: critic},
            shared_lr=args.shared_lr, head_lr=args.head_lr,
        )
    initial_shared_hash = module_hash(shared)
    resume_source = None
    first_update = 0
    total_decisions = 0
    if args.resume is not None:
        resume_path = args.resume.resolve()
        previous = torch.load(resume_path, map_location="cpu", weights_only=False)
        required = {
            "stage": "single_task_pilot", "task": args.task, "player": args.player,
            "seed": args.seed, "model_config": config.__dict__,
            "shared_lr": args.shared_lr, "head_lr": args.head_lr,
            "initial_log_std": initial_log_std,
            "initialization": initialization,
            "calibration_sha256": calibration_sha256,
            "unity_build_sha256": build_sha256,
            "layout_manifest_sha256": layout_sha256,
        }
        if any((previous.get(key, [-0.5] * 4) if key == "initial_log_std"
                 else previous.get(key, {"shared_source_checkpoint": None, "shared_source_sha256": None,
                                         "freeze_shared": False, "zero_head_output": False}) if key == "initialization"
                 else previous.get(key)) != value
               for key, value in required.items()):
            raise ValueError("resume checkpoint does not match pilot configuration or assets")
        total_decisions = int(previous["agent_decisions"])
        if total_decisions % args.rollout:
            raise ValueError("resume decisions must align with rollout size")
        first_update = total_decisions // args.rollout
        if first_update >= args.updates:
            raise ValueError("total requested updates must exceed resume checkpoint")
        shared.load_state_dict(previous["shared"])
        actor.load_state_dict(previous["actor"])
        critic.load_state_dict(previous["critic"])
        optimizer.load_state_dict(previous["optimizer"])
        initial_shared_hash = previous.get("initial_shared_hash", module_hash(shared))
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        resume_source = {
            "checkpoint": str(resume_path.relative_to(ROOT)),
            "sha256": _sha256(resume_path),
            "environment_restarted": True,
            "previous_agent_decisions": total_decisions,
        }
    records = []
    collector = PilotCollector(executable, args.task, args.seed,
                               processor, shared, actor, critic, device)
    started = time.perf_counter()
    try:
        for update in range(first_update, args.updates):
            rollout_started = time.perf_counter()
            rollout = collector.collect(args.rollout, gamma=0.99, gae_lambda=0.95)
            rollout_seconds = time.perf_counter() - rollout_started
            total_decisions += len(rollout.samples)
            training_started = time.perf_counter()
            metrics = update_pilot(rollout, processor, shared, actor, critic,
                                   optimizer, device, args.epochs, args.minibatch)
            training_seconds = time.perf_counter() - training_started
            processor.clear()
            if args.freeze_shared and module_hash(shared) != initial_shared_hash:
                raise RuntimeError("frozen shared encoder changed")
            record = {
                "update": update + 1,
                "agent_decisions": total_decisions,
                "rollout_seconds": rollout_seconds,
                "training_seconds": training_seconds,
                "successes": rollout.terminal_successes,
                "failures": rollout.terminal_failures,
                "interruptions": rollout.interruptions,
                "surplus_transitions_discarded": rollout.surplus_transitions_discarded,
                **metrics,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            torch.save({
                "shared": shared.state_dict(), "actor": actor.state_dict(),
                "critic": critic.state_dict(), "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "stage": "single_task_pilot",
                **code_state,
                "task": args.task, "player": args.player, "seed": args.seed,
                "agent_decisions": total_decisions,
                "model_config": config.__dict__,
                "shared_lr": args.shared_lr, "head_lr": args.head_lr,
                "initial_log_std": initial_log_std,
                "initialization": initialization,
                "initial_shared_hash": initial_shared_hash,
                "calibration_sha256": calibration_sha256,
                "resume_source": resume_source,
                "unity_build_sha256": build_sha256,
                "layout_manifest_sha256": layout_sha256,
            }, output / f"checkpoint_{update + 1:04d}.pt")
    finally:
        collector.close()
    report = {
        "stage": "single_task_pilot",
        **code_state,
        "task": args.task,
        "player": args.player,
        "seed": args.seed,
        "depth_input_size": 392,
        "shared_lr": args.shared_lr, "head_lr": args.head_lr,
        "initial_log_std": initial_log_std,
        "initialization": initialization,
        "initial_shared_hash": initial_shared_hash,
        "final_shared_hash": module_hash(shared),
        "calibration_sha256": calibration_sha256,
        "resume_source": resume_source,
        "unity_build_sha256": build_sha256,
        "layout_manifest_sha256": layout_sha256,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "wall_seconds": time.perf_counter() - started,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "updates": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
