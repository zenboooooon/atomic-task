"""Verify all six fresh heads survive saving, update, and deterministic eval."""

import json
from pathlib import Path

import torch

from tps_transfer.model import ActorHead, CriticHead, module_hash
from tps_transfer.primitive_variants import COLOR_TASKS, task_metadata

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "runs/gaze_v08/smoke_final"
    initial = torch.load(directory / "checkpoint_0000.pt", map_location="cpu", weights_only=False)
    final = torch.load(directory / "checkpoint_0002.pt", map_location="cpu", weights_only=False)
    validation = json.loads((directory / "validation.json").read_text())
    assert initial["agent_decisions"] == 0 and final["agent_decisions"] == 192
    assert tuple(final["pretraining_tasks"]) == COLOR_TASKS
    assert final["task_variants"] == task_metadata(COLOR_TASKS)
    assert any(not torch.equal(initial["shared"][k], value)
               for k, value in final["shared"].items())
    hashes = {"actor": set(), "critic": set()}
    for task in COLOR_TASKS:
        for key, kind, cls in (("actors", "actor", ActorHead), ("critics", "critic", CriticHead)):
            module = cls(final["model_config"]["width"])
            module.load_state_dict(initial[key][task])
            digest = module_hash(module)
            assert digest == final["initial_head_hashes"][task][kind]
            hashes[kind].add(digest)
            assert any(not torch.equal(initial[key][task][k], value)
                       for k, value in final[key][task].items()), (task, kind)
    assert all(len(items) == 6 for items in hashes.values())
    assert set(validation["summary"]) == set(COLOR_TASKS)
    assert validation["environment_seed"] == 9000
    assert all(row["episodes"] == 6 for row in validation["summary"].values())
    assert validation["task_variants"] == task_metadata(COLOR_TASKS)
    report = {"fresh_unique_actor_heads": 6, "fresh_unique_critic_heads": 6,
              "all_six_actor_and_critic_heads_updated": True, "shared_encoder_updated": True,
              "saved_and_reloaded": True, "deterministic_validation_episodes_per_head": 6,
              "smoke_updates": 2, "smoke_agent_decisions": 192,
              "full_training_started": False}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
