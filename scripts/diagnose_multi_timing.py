"""Print per-call visual/reward changes for each ML-Agents behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.smoke import _rgb_at

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    log = ROOT / "runs/phase2/diagnose_multi_timing_player.log"
    environment = UnityEnvironment(
        file_name=str(ROOT / "unity/Build/TpsPretrain.x86_64"), worker_id=0,
        seed=0, no_graphics=False,
        additional_args=["-logFile", str(log), "--tps-task=pretrain", "--tps-split=train"],
    )
    previous = {}
    try:
        environment.reset()
        for call in range(20):
            rows = []
            for behavior in sorted(environment.behavior_specs):
                task = behavior.split("?", 1)[0].removeprefix("TPS_")
                decisions, terminals = environment.get_steps(behavior)
                if len(decisions):
                    image = _rgb_at(decisions.obs, 0)
                    digest = hashlib.sha256(image.tobytes()).hexdigest()[:8]
                    changed = digest != previous.get(task)
                    previous[task] = digest
                    action = np.zeros((len(decisions), 4), dtype=np.float32)
                    if task == "look":
                        action[:, 2] = 1.0
                    environment.set_actions(behavior, ActionTuple(continuous=action))
                    rows.append(f"{task}:{len(decisions)}:{float(decisions.reward[0]):+.4f}:{digest}:{int(changed)}")
                else:
                    rows.append(f"{task}:0")
                if len(terminals):
                    rows.append(f"{task}:term={len(terminals)}")
            print(call, " ".join(rows), flush=True)
            environment.step()
    finally:
        environment.close()
    print("unity_log", log)


if __name__ == "__main__":
    main()
