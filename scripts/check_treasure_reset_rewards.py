"""Check that treasure discovery is rewarded only after an agent acts."""

from pathlib import Path

import numpy as np
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    environment = UnityEnvironment(
        file_name=str(ROOT / "unity/Build/TpsTreasure.x86_64"),
        worker_id=49,
        seed=0,
        no_graphics=False,
        additional_args=[
            "-logFile", str(ROOT / "treasure-reset-rewards.player.log"),
            "--tps-task=treasure", "--tps-split=train",
        ],
    )
    try:
        environment.reset()
        (behavior,) = environment.behavior_specs
        decisions, terminals = environment.get_steps(behavior)
        assert len(terminals) == 0
        assert np.all(decisions.reward == 0), "initial observation has reward"
        initial_agents = len(decisions)
        pending_reset: set[int] = set()
        resets_checked = 0
        discovery_rewards = 0
        for _ in range(330):
            environment.set_actions(behavior, ActionTuple(
                continuous=np.zeros((len(decisions), 4), dtype=np.float32)
            ))
            environment.step()
            decisions, terminals = environment.get_steps(behavior)
            terminal_ids = set(map(int, terminals.agent_id))
            for index, agent_id in enumerate(decisions.agent_id):
                if int(agent_id) in pending_reset or int(agent_id) in terminal_ids:
                    assert decisions.reward[index] == 0, (
                        f"reset observation has reward: agent={agent_id} "
                        f"reward={decisions.reward[index]}"
                    )
                    resets_checked += 1
            pending_reset = (pending_reset | terminal_ids) - set(map(int, decisions.agent_id))
            discovery_rewards += int(np.count_nonzero(decisions.reward > 0.1))
            if resets_checked >= initial_agents:
                break
        assert resets_checked >= initial_agents, (
            f"only {resets_checked}/{initial_agents} agents reset"
        )
        assert discovery_rewards > 0, "discovery reward was never delivered after an action"
        print(f"treasure reset rewards: {resets_checked} zero; "
              f"discovery rewards after action: {discovery_rewards}")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
