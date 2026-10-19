"""Check the new Unity players and their observation/action contracts."""

import json
from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment

from tps_transfer.build_identity import layout_manifest_sha256


ROOT = Path(__file__).resolve().parents[1]


def check(player: str, task: str, expected: dict[str, int], worker_id: int) -> None:
    executable = ROOT / "unity/Build" / f"{player}.x86_64"
    source_manifest = ROOT / "manifests/pretrain3_v01_layouts.json"
    manifest = (source_manifest if player == "TpsPretrainThree"
                and source_manifest.exists() else ROOT / "manifests/pilot_layouts.json")
    layout_manifest_sha256(executable, manifest)
    environment = UnityEnvironment(
        file_name=str(executable), worker_id=worker_id, seed=0,
        no_graphics=False, additional_args=[
            "-logFile", str(ROOT / f"smoke-{player}.log"),
            f"--tps-task={task}", "--tps-split=validation",
        ],
    )
    try:
        environment.reset()
        observed = {}
        for behavior, spec in environment.behavior_specs.items():
            name = behavior.split("?", 1)[0]
            decisions, terminals = environment.get_steps(behavior)
            assert not len(terminals)
            assert spec.action_spec.continuous_size == 4
            assert spec.action_spec.discrete_size == 0
            assert len(decisions.obs) == 1
            print(player, name, "observation_shape", decisions.obs[0].shape, flush=True)
            assert decisions.obs[0].shape[1:] in ((224, 224, 3), (3, 224, 224))
            observed[name] = len(decisions)
        assert observed == expected, (player, observed, expected)
        for behavior in environment.behavior_specs:
            decisions, _ = environment.get_steps(behavior)
            environment.set_actions(behavior, ActionTuple(
                continuous=np.zeros((len(decisions), 4), dtype=np.float32)
            ))
        environment.step()
        if task == "treasure":
            behavior = next(iter(environment.behavior_specs))
            decisions, terminals = environment.get_steps(behavior)
            rewards = np.concatenate((decisions.reward, terminals.reward))
            protocol = json.loads((ROOT / "manifests/pilot_layouts.json").read_text())["protocol_version"]
            if protocol == "treasure-v0.2":
                assert not np.any(rewards > 0.1), "treasure was found without a camera search"
            else:
                assert np.any(rewards > 0.1), "initially visible treasure did not reward discovery"
        print(player, observed, flush=True)
    finally:
        environment.close()


def main() -> None:
    check("TpsPretrainThree", "pretrain",
          {"TPS_look": 4, "TPS_approach": 4, "TPS_avoid": 4}, 47)
    check("TpsTreasure", "treasure", {"TPS_treasure": 16}, 48)


if __name__ == "__main__":
    main()
