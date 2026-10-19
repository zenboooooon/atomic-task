"""Run balanced visual PPO pretraining on the selected primitive tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.model import ModelConfig, SharedEncoder, make_primitive_heads, module_hash
from tps_transfer.multitask import TASKS, TREASURE_SOURCE_TASKS, GAZE_SOURCE_TASKS, MultiTaskCollector, update_balanced
from tps_transfer.observation import FrameProcessor
from tps_transfer.ppo import pretraining_optimizer
from tps_transfer.preprocess import DepthCalibration

from tps_transfer.run_metadata import git_state
from tps_transfer.shared_primitive_heads import share_color_heads, heads_for_groups
from tps_transfer.primitive_variants import (
    COLOR_TASKS, base_task, pretraining_player, task_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-width", type=int, default=256)
    parser.add_argument("--rollout-per-task", type=int, default=128)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--shared-lr", type=float, default=3e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--look-move-log-std", type=float, default=-0.5)
    parser.add_argument("--stop-camera-log-std", type=float, default=-0.5)
    parser.add_argument("--minibatch-per-task", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/phase2/pretrain_pilot"))
    parser.add_argument("--resume", type=Path, help="checkpoint to continue after a fresh Unity reset")
    parser.add_argument("--shared-from", type=Path,
                        help="source of the shared encoder at this stage's start")
    parser.add_argument("--stage-index", type=int, default=1)
    parser.add_argument("--freeze-shared", action="store_true")
    parser.add_argument("--task-set", choices=("all4", "treasure3", "gaze2", "gaze6", "jump3"), default="gaze6")
    parser.add_argument("--layout-manifest", type=Path,
                        default=ROOT / "manifests/pilot_layouts.json")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--share-heads-by-task", action="store_true")
    args = parser.parse_args()
    if args.share_heads_by_task and args.task_set != "gaze6":
        parser.error("shared color heads require --task-set gaze6")
    tasks = (TASKS if args.task_set == "all4" else
             COLOR_TASKS if args.task_set == "gaze6" else
             ("look", "approach", "jump") if args.task_set == "jump3" else
             GAZE_SOURCE_TASKS if args.task_set == "gaze2" else TREASURE_SOURCE_TASKS)
    variants = task_metadata(tasks)
    if args.stage_index < 1:
        parser.error("stage index must be positive")
    if args.stage_index > 1 and args.shared_from is None:
        parser.error("later stages require --shared-from")
    if args.freeze_shared and args.shared_from is None:
        parser.error("--freeze-shared requires --shared-from")
    if min(args.rollout_per_task, args.updates, args.epochs,
           args.minibatch_per_task, args.checkpoint_every) < 1:
        parser.error("all counts must be positive")
    if args.rollout_per_task % args.minibatch_per_task:
        parser.error("rollout per task must be divisible by minibatch per task")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pretraining pilot")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    device = torch.device("cuda")
    if args.model_width < 1 or args.model_width % ModelConfig().heads:
        parser.error("model width must be positive and divisible by attention heads")
    config = ModelConfig(width=args.model_width)
    shared = SharedEncoder(config).to(device)
    # Later stages discard every old actor/critic and the old optimizer state.
    if args.shared_from is not None:
        torch.manual_seed(args.seed + 10_000 * args.stage_index)
        torch.cuda.manual_seed_all(args.seed + 10_000 * args.stage_index)
    actors, critics = make_primitive_heads(
        tasks, config.width, device, zero_output=args.shared_from is not None,
    )
    initial_log_std_by_task = {task: [-0.5] * 5 for task in tasks}
    for task in tasks:
        if base_task(task) == "look":
            initial_log_std_by_task[task][:2] = [args.look_move_log_std] * 2
    if "stop" in tasks:
        initial_log_std_by_task["stop"][2:] = [args.stop_camera_log_std] * 2
    with torch.no_grad():
        for task in tasks:
            actors[task].log_std.copy_(
                torch.tensor(initial_log_std_by_task[task], device=device)
            )
    task_head_map = {task: task for task in tasks}
    if args.share_heads_by_task:
        actors, critics, task_head_map = share_color_heads(actors, critics)
    head_keys = tuple(actors)
    calibration_path = ROOT / "manifests/phase1/depth_selected_v02_normalization.json"
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    processor = FrameProcessor(DepthEstimator(392), DepthCalibration.load(calibration_path))
    player, _ = pretraining_player(tasks)
    executable = ROOT / "unity/Build" / f"{player}.x86_64"
    build_sha = unity_build_sha256(executable)
    layout_sha = layout_manifest_sha256(executable, args.layout_manifest.resolve())
    shared_source = None
    if args.shared_from is not None:
        source_path = args.shared_from.resolve()
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        required_source = {
            "task_head_map": task_head_map,
            "stage": "pretraining_pilot", "model_config": config.__dict__,
            "depth_input_size": 392, "calibration_sha256": calibration_sha,
            "unity_build_sha256": build_sha, "layout_manifest_sha256": layout_sha,
            "pretraining_tasks": list(tasks), "task_variants": variants,
        }
        if any(source.get(key) != value for key, value in required_source.items()):
            raise ValueError("source checkpoint does not match model or assets")
        if source.get("stage_index", 1) != args.stage_index - 1:
            raise ValueError("source checkpoint must come from the previous stage")
        shared.load_state_dict(source["shared"])
        shared_source = {
            "checkpoint": str(source_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "previous_stage": source.get("stage_index", 1),
            "previous_agent_decisions": int(source["agent_decisions"]),
        }
    initial_shared_hash = module_hash(shared)
    initial_head_hashes = {
        task: {"actor": module_hash(actors[task]), "critic": module_hash(critics[task])}
        for task in head_keys
    }
    if args.freeze_shared:
        shared.requires_grad_(False)
        optimizer = torch.optim.Adam(
            [parameter for task in head_keys
             for module in (actors[task], critics[task])
             for parameter in module.parameters()], lr=args.head_lr,
        )
    else:
        optimizer = pretraining_optimizer(shared, actors, critics,
                                          shared_lr=args.shared_lr, head_lr=args.head_lr)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_state = git_state(ROOT)
    code_state.update(task_head_map=task_head_map, pretraining_head_keys=list(head_keys),
                      head_sharing="by_base_task" if args.share_heads_by_task else "none")
    first_update = 0
    resume_source = None
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if resume_path.parent == output:
            raise ValueError("resume output must be a new directory")
        previous = torch.load(resume_path, map_location="cpu", weights_only=False)
        required = {
            "task_head_map": task_head_map,
            "stage": "pretraining_pilot", "seed": args.seed,
            "model_config": config.__dict__, "depth_input_size": 392,
            "shared_lr": args.shared_lr, "head_lr": args.head_lr,
            "initial_log_std_by_task": initial_log_std_by_task,
            "calibration_sha256": calibration_sha,
            "unity_build_sha256": build_sha,
            "layout_manifest_sha256": layout_sha,
            "pretraining_tasks": list(tasks), "task_variants": variants,
        }
        if args.stage_index > 1:
            required.update({
                "stage_index": args.stage_index,
                "shared_source": shared_source,
                "shared_frozen": args.freeze_shared,
                "initial_shared_hash": initial_shared_hash,
            })
        if any((previous.get(key, {task: [-0.5] * 5 for task in tasks})
                if key == "initial_log_std_by_task" else previous.get(key)) != value
               for key, value in required.items()):
            raise ValueError("resume checkpoint does not match configuration or assets")
        decisions = int(previous["agent_decisions"])
        decisions_per_update = args.rollout_per_task * len(tasks)
        if decisions % decisions_per_update:
            raise ValueError("resume decisions must align with rollout size")
        first_update = decisions // decisions_per_update
        if first_update >= args.updates:
            raise ValueError("total requested updates must exceed resume checkpoint")
        if "minibatch_rng" not in previous:
            raise ValueError("checkpoint has no minibatch RNG state")
        shared.load_state_dict(previous["shared"])
        for task in head_keys:
            actors[task].load_state_dict(previous["actors"][task])
            critics[task].load_state_dict(previous["critics"][task])
        optimizer.load_state_dict(previous["optimizer"])
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        generator.set_state(previous["minibatch_rng"])
        initial_head_hashes = previous.get("initial_head_hashes", initial_head_hashes)
        resume_source = {
            "checkpoint": str(resume_path.relative_to(ROOT)),
            "sha256": hashlib.sha256(resume_path.read_bytes()).hexdigest(),
            "environment_restarted": True,
            "previous_agent_decisions": decisions,
        }
    else:
        torch.save({
            "stage": "pretraining_pilot", "stage_index": args.stage_index,
            "pretraining_tasks": list(tasks), "task_variants": variants,
            "seed": args.seed, "agent_decisions": 0, **code_state,
            "shared": shared.state_dict(),
            "actors": {task: actors[task].state_dict() for task in head_keys},
            "critics": {task: critics[task].state_dict() for task in head_keys},
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "minibatch_rng": generator.get_state(),
            "resume_source": None,
            "shared_source": shared_source,
            "shared_frozen": args.freeze_shared,
            "initial_shared_hash": initial_shared_hash,
            "initial_head_hashes": initial_head_hashes,
            "model_config": config.__dict__, "depth_input_size": 392,
            "shared_lr": args.shared_lr, "head_lr": args.head_lr,
            "initial_log_std_by_task": initial_log_std_by_task,
            "calibration_sha256": calibration_sha,
            "unity_build_sha256": build_sha,
            "layout_manifest_sha256": layout_sha,
        }, output / "checkpoint_0000.pt")
    collector = MultiTaskCollector(str(executable), args.seed, processor, shared,
                                   heads_for_groups(actors, task_head_map),
                                   heads_for_groups(critics, task_head_map),
                                   device, tasks=tasks, worker_id=args.worker_id)
    started = time.perf_counter()
    records = []
    try:
        for update in range(first_update, args.updates):
            rollout_started = time.perf_counter()
            rollouts, surplus = collector.collect(args.rollout_per_task)
            rollout_seconds = time.perf_counter() - rollout_started
            training_started = time.perf_counter()
            metrics = update_balanced(
                rollouts, processor, shared, heads_for_groups(actors, task_head_map),
                heads_for_groups(critics, task_head_map), optimizer,
                device, args.epochs, args.minibatch_per_task, generator, tasks=tasks,
            )
            training_seconds = time.perf_counter() - training_started
            processor.clear()
            if args.freeze_shared and module_hash(shared) != initial_shared_hash:
                raise RuntimeError("frozen shared encoder changed")
            record = {
                "update": update + 1,
                "agent_decisions_trained": (update + 1) * args.rollout_per_task * len(tasks),
                "rollout_seconds": rollout_seconds,
                "training_seconds": training_seconds,
                "surplus_transitions_discarded": surplus,
                **metrics["_summary"],
                "tasks": {
                    task: {
                        "successes": rollouts[task].terminal_successes,
                        "failures": rollouts[task].terminal_failures,
                        "interruptions": rollouts[task].interruptions,
                        **metrics[task],
                    }
                    for task in tasks
                },
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            if (update + 1) % args.checkpoint_every == 0 or update + 1 == args.updates:
                torch.save({
                    "stage": "pretraining_pilot", "seed": args.seed,
                    "pretraining_tasks": list(tasks), "task_variants": variants,
                    **code_state,
                    "agent_decisions": record["agent_decisions_trained"],
                    "shared": shared.state_dict(),
                    "actors": {task: actors[task].state_dict() for task in head_keys},
                    "critics": {task: critics[task].state_dict() for task in head_keys},
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "minibatch_rng": generator.get_state(),
                    "resume_source": resume_source,
                    "model_config": config.__dict__,
                    "depth_input_size": 392,
                    "shared_lr": args.shared_lr, "head_lr": args.head_lr,
                    "initial_log_std_by_task": initial_log_std_by_task,
                    "calibration_sha256": calibration_sha,
                    "unity_build_sha256": build_sha,
                    "layout_manifest_sha256": layout_sha,
                    "stage_index": args.stage_index,
                    "shared_source": shared_source,
                    "shared_frozen": args.freeze_shared,
                    "initial_shared_hash": initial_shared_hash,
                    "initial_head_hashes": initial_head_hashes,
                }, output / f"checkpoint_{update + 1:04d}.pt")
    finally:
        collector.close()
    report = {
        "stage": "pretraining_pilot", "seed": args.seed,
        "pretraining_tasks": list(tasks), "task_variants": variants,
        **code_state,
        "resume_source": resume_source,
        "unity_build_sha256": build_sha,
        "layout_manifest_sha256": layout_sha,
        "calibration_sha256": calibration_sha,
        "depth_input_size": 392,
        "shared_lr": args.shared_lr, "head_lr": args.head_lr,
        "initial_log_std_by_task": initial_log_std_by_task,
        "stage_index": args.stage_index,
        "shared_source": shared_source,
        "shared_frozen": args.freeze_shared,
        "initial_shared_hash": initial_shared_hash,
        "final_shared_hash": module_hash(shared),
        "initial_head_hashes": initial_head_hashes,
        "wall_seconds": time.perf_counter() - started,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "updates": records,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
