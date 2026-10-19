"""Check final multi-arena gaze players without starting training."""

from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/gaze_v04_layouts.json"


def check(name: str, task: str, expected: dict[str, int], worker: int) -> None:
    executable = ROOT / "unity/Build" / f"{name}.x86_64"
    layout_manifest_sha256(executable, MANIFEST)
    env = UnityEnvironment(
        file_name=str(executable), worker_id=worker, seed=0, no_graphics=False,
        additional_args=["-logFile", str(ROOT / f"smoke-{name}-gaze-v04.log"),
                         f"--tps-task={task}", "--tps-split=validation"],
    )
    try:
        env.reset()
        observed = {}
        for behavior, spec in env.behavior_specs.items():
            decisions, terminals = env.get_steps(behavior)
            assert len(terminals) == 0
            assert spec.action_spec.continuous_size == 4
            assert len(decisions.obs) == 1
            assert decisions.obs[0].shape[1:] in ((224, 224, 3), (3, 224, 224))
            observed[behavior.split("?", 1)[0]] = len(decisions)
            env.set_actions(behavior, ActionTuple(
                continuous=np.zeros((len(decisions), 4), dtype=np.float32)))
        assert observed == expected, (observed, expected)
        env.step()
        print(name, observed, "OK", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    check("TpsPretrainTwo", "pretrain", {"TPS_look": 6, "TPS_approach": 6}, 91)
    check("TpsTreasure", "treasure", {"TPS_treasure": 16}, 92)
