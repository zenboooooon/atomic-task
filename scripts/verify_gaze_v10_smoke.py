"""Verify task changes, five-action PPO, matched transfer heads, and physics checks."""
import hashlib
import json
import math
from pathlib import Path
import torch
from tps_transfer.build_identity import unity_build_sha256, layout_manifest_sha256

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "runs/v10_smoke"
TASKS = ["look", "approach", "jump"]


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    report = json.loads((SMOKE / "ppo/report.json").read_text())
    assert finite(report) and len(report["updates"]) == 2
    assert report["updates"][-1]["agent_decisions_trained"] == 1536
    assert report["pretraining_head_keys"] == TASKS
    initial = torch.load(SMOKE / "ppo/checkpoint_0000.pt", map_location="cpu", weights_only=False)
    final = torch.load(SMOKE / "ppo/checkpoint_0002.pt", map_location="cpu", weights_only=False)
    assert final["model_config"]["action_size"] == 5
    assert list(final["actors"]) == list(final["critics"]) == TASKS
    assert report["initial_shared_hash"] != report["final_shared_hash"]
    for task in TASKS:
        assert final["actors"][task]["mean_output.weight"].shape[0] == 5
        assert any(not torch.equal(initial["actors"][task][k], v) for k,v in final["actors"][task].items())
    ids = [v for g in final["optimizer"]["param_groups"] for v in g["params"]]
    assert len(ids) == len(set(ids))
    evaluation = json.loads((SMOKE / "primitive_validation.json").read_text())
    assert evaluation["environment_seed"] == 9000
    assert set(evaluation["summary"]) == set(TASKS)
    assert all(x["episodes"] == 6 for x in evaluation["summary"].values())
    heads=[]
    for condition in ["P","S","F"]:
        p=SMOKE/f"downstream/{condition}_seed42"
        d=json.loads((p/'report.json').read_text())
        assert len(d['updates']) == 1 and finite(d)
        assert d['shared_frozen'] == (condition == 'P')
        assert (d['initial_shared_hash'] == d['final_shared_hash']) == (condition == 'P')
        state=torch.load(p/'checkpoint_0000.pt',map_location='cpu',weights_only=False)
        heads.append(state['actor'])
    assert all(all(torch.equal(heads[0][k], h[k]) for k in heads[0]) for h in heads[1:])
    for filename, tasks in [('probe_primitive.player.log',TASKS),('probe_composite.player.log',['treasure'])]:
        lines=(SMOKE/filename).read_text(errors='replace').splitlines()
        for task in tasks:
            successes=[x for x in lines if 'TPS_EPISODE ' in x and f' task={task} ' in x and ' reason=success ' in x]
            assert successes, (filename,task,'scripted success missing')
            if task in ('jump','treasure'):
                assert all('landed_platform=True' in x for x in successes)
    walk=(SMOKE/'probe_walk.player.log').read_text(errors='replace').splitlines()
    assert not any(' task=jump ' in x and ' reason=success ' in x for x in walk)
    assert any(' task=jump ' in x and ' reason=time_limit ' in x for x in walk)
    hashes={name:unity_build_sha256(ROOT/'unity/Build'/f'{name}.x86_64') for name in ['TpsPretrainJump','TpsTreasure']}
    layout_hash=layout_manifest_sha256(ROOT/'unity/Build/TpsPretrainJump.x86_64',ROOT/'manifests/gaze_v10_layouts.json')
    assert layout_manifest_sha256(ROOT/'unity/Build/TpsTreasure.x86_64',ROOT/'manifests/gaze_v10_layouts.json') == layout_hash
    assert (ROOT/'python/tps_transfer/ppo.py').read_bytes() == Path('/home/s6323038/atomic-task-shared-heads-v09/python/tps_transfer/ppo.py').read_bytes()
    evidence=dict(all_checks_passed=True,checkpoint_head_keys=TASKS,gpu_smoke_updates=2,gpu_smoke_decisions=1536,
                  action_size=5,unity_build_sha256=hashes,layout_manifest_sha256=layout_hash,
                  physics_scripted_probes_passed=True,transfer_conditions_checked=['P','S','F'],
                  extra_losses=[],training_objective='unchanged PPO')
    (SMOKE/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
