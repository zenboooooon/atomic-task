"""Verify a rendered Unity TPS build through the ML-Agents low-level API."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from .episode import EpisodeTracker, Step


def _rgb_at(observations: list[np.ndarray], row: int) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for array in observations:
        if array.ndim != 4:
            continue
        if array.shape[1:] == (3, 224, 224):
            candidates.append(np.moveaxis(array[row], 0, -1))
        elif array.shape[1:] == (224, 224, 3):
            candidates.append(array[row])
    if len(candidates) != 1:
        raise ValueError(
            f"expected one 224x224 RGB visual observation, found {len(candidates)}; "
            f"observation shapes: {[array.shape for array in observations]}"
        )
    image = candidates[0]
    if not np.isfinite(image).all():
        raise ValueError("RGB observation contains nonfinite values")
    if image.min() < 0 or image.max() > 1:
        raise ValueError("RGB observation must be normalized to [0, 1]")
    return image.copy()


def _steps(raw: object, terminal: bool) -> list[Step]:
    result: list[Step] = []
    for row, agent_id in enumerate(raw.agent_id):
        result.append(
            Step(
                agent_id=int(agent_id),
                rgb=_rgb_at(raw.obs, row),
                reward=float(raw.reward[row]),
                interrupted=bool(raw.interrupted[row]) if terminal else False,
            )
        )
    return result


def run(
    executable: Path,
    worker_id: int,
    steps: int,
    seed: int,
    sample_path: Path | None = None,
) -> dict[str, object]:
    if not executable.is_file():
        raise FileNotFoundError(f"Unity executable does not exist: {executable}")
    tracker = EpisodeTracker(worker_id=worker_id)
    started = time.perf_counter()
    environment = UnityEnvironment(
        file_name=str(executable.resolve()),
        worker_id=worker_id,
        seed=seed,
        no_graphics=False,
    )
    decisions_seen = 0
    transitions_seen = 0
    terminals_seen = 0
    interrupted_seen = 0
    behaviors_seen: set[str] = set()
    first_rgb: np.ndarray | None = None
    try:
        environment.reset()
        reset_done = time.perf_counter()
        for _ in range(steps):
            for behavior, specification in environment.behavior_specs.items():
                if specification.action_spec.continuous_size != 4:
                    raise ValueError(f"{behavior}: expected 4 continuous actions")
                if specification.action_spec.discrete_size != 0:
                    raise ValueError(f"{behavior}: unexpected discrete actions")
                decisions, terminals = environment.get_steps(behavior)
                transitions = tracker.observe(
                    behavior,
                    _steps(decisions, terminal=False),
                    _steps(terminals, terminal=True),
                )
                transitions_seen += len(transitions)
                terminals_seen += len(terminals)
                interrupted_seen += sum(bool(value) for value in terminals.interrupted)
                decisions_seen += len(decisions)
                if len(decisions):
                    behaviors_seen.add(behavior)
                    if first_rgb is None:
                        first_rgb = _rgb_at(decisions.obs, 0)
                    actions = np.zeros((len(decisions), 4), dtype=np.float32)
                    for row, agent_id in enumerate(decisions.agent_id):
                        tracker.record_action(behavior, int(agent_id), actions[row])
                    environment.set_actions(behavior, ActionTuple(continuous=actions))
            environment.step()
        # The final step produced observations and rewards too. Consume them
        # before reporting counts, without scheduling another action.
        for behavior in environment.behavior_specs:
            decisions, terminals = environment.get_steps(behavior)
            transitions_seen += len(
                tracker.observe(
                    behavior,
                    _steps(decisions, terminal=False),
                    _steps(terminals, terminal=True),
                )
            )
            terminals_seen += len(terminals)
            interrupted_seen += sum(bool(value) for value in terminals.interrupted)
        rollout_done = time.perf_counter()
    finally:
        environment.close()
    closed = time.perf_counter()
    if decisions_seen == 0:
        raise RuntimeError("Unity returned no decisions")
    if terminals_seen == 0:
        raise RuntimeError("Unity returned no terminal event; run more steps")
    if sample_path is not None and first_rgb is not None:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_u8 = np.clip(np.rint(first_rgb * 255.0), 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(sample_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)):
            raise OSError(f"could not write sample image: {sample_path}")
    digest = hashlib.sha256()
    with executable.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return {
        "build_sha256": digest.hexdigest(),
        "worker_id": worker_id,
        "seed": seed,
        "env_step_calls": steps,
        "agent_decisions": decisions_seen,
        "agent_transitions": transitions_seen,
        "terminal_events": terminals_seen,
        "interrupted_events": interrupted_seen,
        "behaviors": sorted(behaviors_seen),
        "rgb_shape": [224, 224, 3],
        "first_rgb_min": float(first_rgb.min()),
        "first_rgb_max": float(first_rgb.max()),
        "first_rgb_std": float(first_rgb.std()),
        "sample_path": str(sample_path.resolve()) if sample_path is not None else None,
        "graphics_enabled": True,
        "startup_seconds": reset_done - started,
        "rollout_seconds": rollout_done - reset_done,
        "shutdown_seconds": closed - rollout_done,
        "decisions_per_second": decisions_seen / (rollout_done - reset_done),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-path", type=Path, default=Path("runs/smoke/sample.png"))
    parser.add_argument("--report-path", type=Path, default=Path("runs/phase0/smoke.json"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    report = run(args.executable, args.worker_id, args.steps, args.seed, args.sample_path)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
