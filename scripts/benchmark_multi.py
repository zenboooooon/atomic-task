"""Verify 16 rendered arenas, task behavior groups, and decision throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.episode import EpisodeTracker
from tps_transfer.smoke import _rgb_at, _steps

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("pretrain", "composite"), required=True)
    parser.add_argument("--calls", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    player = "TpsPretrain" if args.kind == "pretrain" else "TpsComposite"
    executable = ROOT / "unity/Build" / f"{player}.x86_64"
    tracker = EpisodeTracker(worker_id=0)
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=0, seed=0, no_graphics=False,
        additional_args=[f"--tps-task={args.kind if args.kind == 'pretrain' else 'composite'}"],
    )
    decisions = transitions = terminals = 0
    behavior_counts = {}
    behavior_terminals = {}
    first_images = []
    try:
        environment.reset()
        started = time.perf_counter()
        for call in range(args.calls):
            for behavior, spec in environment.behavior_specs.items():
                if spec.action_spec.continuous_size != 4:
                    raise ValueError(f"{behavior} has wrong action size")
                current, terminal = environment.get_steps(behavior)
                transitions += len(tracker.observe(behavior, _steps(current, False),
                                                   _steps(terminal, True)))
                terminals += len(terminal)
                behavior_terminals[behavior] = behavior_terminals.get(behavior, 0) + len(terminal)
                decisions += len(current)
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + len(current)
                if call == 0:
                    first_images.extend(hashlib.sha256(_rgb_at(current.obs, row).tobytes()).hexdigest()
                                        for row in range(len(current)))
                if len(current):
                    action = np.zeros((len(current), 4), dtype=np.float32)
                    for row, agent_id in enumerate(current.agent_id):
                        tracker.record_action(behavior, int(agent_id), action[row])
                    environment.set_actions(behavior, ActionTuple(continuous=action))
            environment.step()
        elapsed = time.perf_counter() - started
    finally:
        environment.close()
    expected = 4 if args.kind == "pretrain" else 1
    if len(behavior_counts) != expected or len(first_images) != 16:
        raise RuntimeError(f"expected {expected} behaviors and 16 arenas, got "
                           f"{len(behavior_counts)} and {len(first_images)}")
    if len(set(first_images)) < 8:
        raise RuntimeError("too few distinct initial arena images")
    report = {
        "kind": args.kind,
        "build_sha256": unity_build_sha256(executable),
        "layout_manifest_sha256": layout_manifest_sha256(
            executable, ROOT / "manifests/pilot_layouts.json"
        ),
        "env_step_calls": args.calls,
        "agent_decisions": decisions,
        "agent_transitions": transitions,
        "terminal_events": terminals,
        "behavior_decisions": behavior_counts,
        "behavior_terminals": behavior_terminals,
        "distinct_first_images": len(set(first_images)),
        "rollout_seconds": elapsed,
        "decisions_per_second": decisions / elapsed,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
