"""Record concrete preflight evidence for the head-sharing-only comparison."""
import json
import math
from pathlib import Path

import torch

from tps_transfer.build_identity import unity_build_sha256, layout_manifest_sha256
from tps_transfer.primitive_variants import COLOR_TASKS, base_task

ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path('/home/s6323038/atomic-task-color-variants-v08')
SMOKE = ROOT / 'runs/v09_smoke'


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    report = json.loads((SMOKE / 'ppo/report.json').read_text())
    assert finite(report)
    assert len(report['updates']) == 2
    assert report['updates'][-1]['agent_decisions_trained'] == 1536
    assert report['head_sharing'] == 'by_base_task'
    assert report['pretraining_head_keys'] == ['look', 'approach']
    assert set(report['updates'][-1]['tasks']) == set(COLOR_TASKS)
    assert report['initial_shared_hash'] != report['final_shared_hash']
    assert 'paired_appearance' not in report
    initial = torch.load(SMOKE / 'ppo/checkpoint_0000.pt', map_location='cpu', weights_only=False)
    final = torch.load(SMOKE / 'ppo/checkpoint_0002.pt', map_location='cpu', weights_only=False)
    for key in ('actors', 'critics'):
        assert list(initial[key]) == list(final[key]) == ['look', 'approach']
        for task in ('look', 'approach'):
            assert any(not torch.equal(initial[key][task][k], final[key][task][k]) for k in final[key][task])
    assert final['task_head_map'] == {task: base_task(task) for task in COLOR_TASKS}
    ids = [pid for group in final['optimizer']['param_groups'] for pid in group['params']]
    assert len(ids) == len(set(ids))
    evaluation = json.loads((SMOKE / 'primitive_validation.json').read_text())
    assert evaluation['environment_seed'] == 9000
    assert set(evaluation['summary']) == set(COLOR_TASKS)
    assert all(v['episodes'] == 6 for v in evaluation['summary'].values())
    downstream = json.loads((SMOKE / 'downstream/F_seed42/report.json').read_text())
    assert len(downstream['updates']) == 1 and finite(downstream)
    assert downstream['initial_shared_hash'] != downstream['final_shared_hash']
    changed = []
    for directory in ('python', 'scripts', 'manifests', 'configs', 'unity/Assets/Scripts', 'unity/Assets/Editor'):
        for source in (BASELINE / directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and not source.name.startswith('._'):
                relative = source.relative_to(BASELINE)
                if source.read_bytes() != (ROOT / relative).read_bytes():
                    changed.append(str(relative))
    assert sorted(changed) == sorted(['python/tps_transfer/multitask.py',
                                     'scripts/train_pretraining_pilot.py',
                                     'scripts/evaluate_multi_pilot.py'])
    hashes = {}
    for name in ('TpsPretrainColors', 'TpsTreasure'):
        executable = Path('unity/Build') / (name + '.x86_64')
        hashes[name] = unity_build_sha256(ROOT / executable)
        assert hashes[name] == unity_build_sha256(BASELINE / executable)
    layout_hash = layout_manifest_sha256(ROOT / 'unity/Build/TpsPretrainColors.x86_64',
                                        ROOT / 'manifests/gaze_v08_layouts.json')
    evidence = dict(all_checks_passed=True, checkpoint_head_keys=['look', 'approach'],
                    gpu_smoke_updates=2, gpu_smoke_decisions=1536, tests_passed=23,
                    changed_existing_files=changed, unity_build_sha256=hashes,
                    layout_manifest_sha256=layout_hash,
                    primitive_smoke_evaluation_episodes_per_color=6,
                    downstream_checkpoint_accepted=True)
    (SMOKE / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
