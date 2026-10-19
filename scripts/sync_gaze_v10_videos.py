"""Fetch and package completed video batches; report only undelivered milestones."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import shlex
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/s6323038/atomic-task-jump-v10'
HOST = 's6323038@10.66.84.33'
OUTPUT = ROOT / 'runs/qualitative/every200'


def atomic(path, data):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(data, indent=2) + '\n')
    pending.replace(path)


def fetch(relative, target=None):
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a path inside the remote experiment')
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST,
               'cat -- ' + shlex.quote(f'{REMOTE}/{path}')]
    if target is None:
        return subprocess.run(command, check=True, capture_output=True, timeout=30).stdout
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + '.pending')
    with pending.open('wb') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.PIPE, timeout=120)
    pending.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mark-delivered', nargs='+')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'sync.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        delivered_path = OUTPUT / 'delivered.json'
        delivered = json.loads(delivered_path.read_text()) if delivered_path.exists() else {}
        if args.mark_delivered:
            for batch_id in args.mark_delivered:
                artifact = OUTPUT / batch_id / 'delivery.json'
                if not artifact.exists():
                    raise FileNotFoundError(f'Cannot mark an unpackaged batch delivered: {batch_id}')
                delivered[batch_id] = datetime.now().astimezone().isoformat(timespec='seconds')
            atomic(delivered_path, delivered)
            print(json.dumps({'marked_delivered': args.mark_delivered}))
            return
        index = json.loads(fetch('runs/qualitative/every200/index.json'))
        atomic(OUTPUT / 'remote_index.json', index)
        new = []
        for batch_id, batch in index['batches'].items():
            if batch['state'] not in ('complete', 'partial') or batch_id in delivered:
                continue
            if not batch_id.replace('_', '').isalnum():
                raise ValueError('Invalid batch id')
            out = OUTPUT / batch_id
            out.mkdir(parents=True, exist_ok=True)
            delivery_path = out / 'delivery.json'
            if not delivery_path.exists():
                selection = batch['selection']
                for task, cases in selection.items():
                    if task not in ('look', 'approach', 'jump', 'treasure'):
                        raise ValueError('Unexpected video task')
                    for outcome, item in cases.items():
                        if outcome not in ('success', 'failure'):
                            raise ValueError('Unexpected video outcome')
                        if item['update'] != batch['update']:
                            raise ValueError('Checkpoint update mismatch')
                        target = out / 'source' / f'{task}_{outcome}.mp4'
                        fetch(item['remote_file'], target)
                        item['local_file'] = str(target)
                selection_path = out / 'selection.json'
                atomic(selection_path, selection)
                subprocess.run([sys.executable, str(ROOT / 'scripts/package_training_example_videos.py'),
                                '--selection', str(selection_path), '--output-dir', str(out),
                                '--update', str(batch['update']), '--group', batch['group']],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                atomic(out / 'recording_batch.json', batch)
            delivery = json.loads(delivery_path.read_text())
            new.append({'id': batch_id, 'group': batch['group'], 'update': batch['update'],
                        'state': batch['state'], 'missing': batch.get('missing', {}),
                        'combined': str(out / delivery['combined']),
                        'seconds': delivery['combined_seconds'],
                        'tasks': {task: str(out / f'{task}_success_and_failure.mp4')
                                  for task in batch['tasks']},
                        'seeds': {task: {outcome: item['seed'] for outcome, item in cases.items()}
                                  for task, cases in batch['selection'].items()}})
        print(json.dumps({'pipeline_state': index.get('state'), 'updated': index.get('updated'),
                          'active_batch': index.get('active_batch'), 'pipeline_error': index.get('error'),
                          'recording_errors': [{'batch': b['id'], **a} for b in index['batches'].values()
                                               for a in b['attempts'] if a.get('error')],
                          'new_deliveries': new}, indent=2))


if __name__ == '__main__':
    main()
