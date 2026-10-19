"""Serial, restartable qualitative recording queue; never controls training."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def now():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')


def atomic_json(path, value):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def specifications(config):
    for update in range(config['checkpoint_interval'], config['updates'] + 1,
                        config['checkpoint_interval']):
        yield f'primitives_update{update:04d}', 'primitives', update, config['primitive_tasks']
        for condition in config['downstream_conditions']:
            yield f'{condition}_update{update:04d}', condition, update, ['treasure']


def checkpoint(config, group, seed, update):
    base = ROOT / config['source_run']
    job = (base / f'seed{seed}/pretrain3_seed{seed}' if group == 'primitives'
           else ROOT / config.get('downstream_run', str(Path(config['source_run']) / 'downstream'))
                / f'{group}_seed{seed}')
    return job / f'checkpoint_{update:04d}.pt'


def ready(path, settle):
    try:
        stat = path.stat()
        return stat.st_size > 0 and time.time() - stat.st_mtime >= settle
    except FileNotFoundError:
        return False


def missing(batch):
    return {task: [case for case in ('success', 'failure') if case not in cases]
            for task, cases in batch['selection'].items()
            if any(case not in cases for case in ('success', 'failure'))}


def execute(config, group, update, seed, needed, output):
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / f'seed{seed}'
    if attempt.exists():
        # A previous process may have died after writing an episode manifest.
        manifest = attempt / 'manifest.json'
        if manifest.exists():
            return json.loads(manifest.read_text()), None
        attempt = output / f'seed{seed}_retry_{int(time.time())}'
    cp = checkpoint(config, group, seed, update)
    command = [str(ROOT / 'scripts/run_gaze_v10_training_examples.sh'),
               '--checkpoint', str(cp), '--output-dir', str(attempt),
               '--worker-id', str(config['worker_id']),
               '--sampling-seed', str(20261019 + update * 10 + seed),
               '--max-calls', str(config['max_calls_per_attempt']),
               '--max-seconds', str(config['max_seconds_per_attempt']),
               '--capture-tasks', *needed, '--needed-cases', json.dumps(needed)]
    error = None
    with (output / f'seed{seed}.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        atomic_json(output / 'active.json', {'pid': child.pid, 'started': now(),
                                            'checkpoint': str(cp), 'command': command})
        deadline = time.monotonic() + config['max_seconds_per_attempt'] + 90
        while child.poll() is None:
            if STOP or time.monotonic() > deadline:
                error = 'recorder_stopped' if STOP else 'recorder_timeout'
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                break
            time.sleep(1)
        if child.returncode and error is None:
            error = f'recorder_exit_{child.returncode}'
    (output / 'active.json').unlink(missing_ok=True)
    manifest = attempt / 'manifest.json'
    return (json.loads(manifest.read_text()) if manifest.exists() else None), error


def collect_one(config, index, base):
    for batch_id, group, update, tasks in specifications(config):
        batch = index['batches'].setdefault(batch_id, {
            'id': batch_id, 'group': group, 'update': update, 'tasks': tasks,
            'state': 'waiting', 'selection': {task: {} for task in tasks}, 'attempts': []})
        if batch['state'] in ('complete', 'partial', 'error'):
            continue
        needed = missing(batch)
        if not needed:
            batch.update(state='complete', finished=now(), missing={})
            continue
        tried = {item['seed'] for item in batch['attempts']}
        seed = next((s for s in config['seed_order'] if s not in tried), None)
        if seed is None:
            batch.update(state='partial', finished=now(), missing=needed)
            continue
        cp = checkpoint(config, group, seed, update)
        if not ready(cp, config['checkpoint_settle_seconds']):
            continue
        batch.update(state='recording', active_seed=seed)
        index.update(updated=now(), active_batch=batch_id)
        atomic_json(base / 'index.json', index)
        report, error = execute(config, group, update, seed, needed, base / batch_id)
        attempt = {'seed': seed, 'finished': now(), 'error': error,
                   'report': str((base / batch_id / f'seed{seed}/manifest.json').relative_to(ROOT))}
        if report:
            if report['update'] != update or report['seed'] != seed:
                raise ValueError('recording provenance does not match queued checkpoint')
            report_path = Path(report['checkpoint'])
            if report_path != cp.relative_to(ROOT):
                raise ValueError('recorder used an unexpected source checkpoint')
            for item in report['videos']:
                task, outcome = item['task'], item['outcome']
                if task in needed and outcome in needed[task] and outcome not in batch['selection'][task]:
                    # File location comes from this attempt's report directory.
                    candidates = list((base / batch_id).glob(f'seed{seed}*/{item["file"]}'))
                    if not candidates:
                        raise FileNotFoundError(item['file'])
                    batch['selection'][task][outcome] = {
                        **item, 'seed': seed, 'update': update,
                        'condition': report.get('condition'),
                        'remote_file': str(candidates[-1].relative_to(ROOT)),
                        'checkpoint': report['checkpoint'],
                        'checkpoint_sha256': report['checkpoint_sha256'],
                        'sampling_seed': report['sampling_seed'],
                        'recording_type': report['recording_type']}
            attempt.update(episodes=len(report['episodes']), complete=report['complete'])
        batch['attempts'].append(attempt)
        batch.pop('active_seed', None)
        batch['missing'] = missing(batch)
        all_tried = len(batch['attempts']) == len(config['seed_order'])
        batch['state'] = ('complete' if not batch['missing'] else 'partial' if all_tried else 'waiting')
        if batch['state'] in ('complete', 'partial'):
            batch['finished'] = now()
        index.update(updated=now(), active_batch=None)
        atomic_json(base / 'index.json', index)
        print(json.dumps({'time': now(), 'batch': batch_id, 'seed': seed,
                          'state': batch['state'], 'missing': batch['missing'], 'error': error}), flush=True)
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/video_pipeline_v10.json')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['selection_mode'] == 'one_pair_per_task'
    base = ROOT / config['output']
    base.mkdir(parents=True, exist_ok=True)
    with (base / 'pipeline.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A video pipeline is already running')
        (base / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
        index_path = base / 'index.json'
        index = (json.loads(index_path.read_text()) if index_path.exists() else
                 {'version': 1, 'source_run': config['source_run'], 'batches': {}, 'created': now()})
        def stop(signum, frame):
            global STOP
            STOP = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            while not STOP:
                worked = collect_one(config, index, base)
                index.update(updated=now(), state='running' if worked else 'waiting_for_checkpoints')
                atomic_json(index_path, index)
                (base / 'pipeline.status').write_text(f'{now()} {index["state"]}\n')
                if all(b['state'] in ('complete', 'partial', 'error') for b in index['batches'].values()) \
                        and len(index['batches']) == len(list(specifications(config))):
                    index.update(state='finished', finished=now())
                    atomic_json(index_path, index)
                    (base / 'pipeline.status').write_text(f'{now()} finished\n')
                    break
                if args.once:
                    break
                if not worked:
                    for _ in range(config['poll_seconds']):
                        if STOP:
                            break
                        time.sleep(1)
        except Exception as exc:
            index.update(state='error', error=repr(exc), updated=now())
            atomic_json(index_path, index)
            (base / 'pipeline.status').write_text(f'{now()} error {exc!r}\n')
            raise
        finally:
            if STOP:
                (base / 'pipeline.status').write_text(f'{now()} stopped_recording_only\n')


if __name__ == '__main__':
    main()
