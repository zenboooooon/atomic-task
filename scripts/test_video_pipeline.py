"""Queue safety tests with synthetic checkpoint files; no Unity or training jobs."""
import importlib.util
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('video_queue', Path(__file__).with_name('watch_gaze_v10_videos.py'))
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


class QueueSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(source_run='run', checkpoint_interval=200, updates=200,
                           primitive_tasks=['look'], downstream_conditions=[],
                           seed_order=[0, 1, 2], checkpoint_settle_seconds=15)
        self.base = self.root / 'videos'
        self.base.mkdir()
        self.index = {'batches': {}}
        self.root_patch = patch.object(queue, 'ROOT', self.root)
        self.root_patch.start()
        for seed in range(3):
            cp = queue.checkpoint(self.config, 'primitives', seed, 200)
            cp.parent.mkdir(parents=True)
            cp.write_bytes(b'fake checkpoint')
            os.utime(cp, (time.time()-60, time.time()-60))

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def report(self, seed, cases):
        folder = self.base / 'primitives_update0200' / f'seed{seed}'
        folder.mkdir(parents=True, exist_ok=True)
        videos = []
        for case in cases:
            name = f'look_{case}.mp4'
            (folder / name).write_bytes(b'video')
            videos.append(dict(task='look', outcome=case, file=name))
        return dict(seed=seed, update=200,
                    checkpoint=str(queue.checkpoint(self.config, 'primitives', seed, 200).relative_to(self.root)),
                    checkpoint_sha256='hash', sampling_seed=123, recording_type='test',
                    videos=videos, episodes=[], complete=len(cases)==2)

    def test_seed_cascade_and_no_duplicate_completed_jobs(self):
        with patch.object(queue, 'execute', return_value=(self.report(0, ['success']), None)) as record:
            queue.collect_one(self.config, self.index, self.base)
            self.assertEqual(record.call_args.args[3:5], (0, {'look': ['success', 'failure']}))
        with patch.object(queue, 'execute', return_value=(self.report(1, ['failure']), None)) as record:
            queue.collect_one(self.config, self.index, self.base)
            self.assertEqual(record.call_args.args[3:5], (1, {'look': ['failure']}))
        batch = self.index['batches']['primitives_update0200']
        self.assertEqual(batch['state'], 'complete')
        with patch.object(queue, 'execute') as record:
            queue.collect_one(self.config, self.index, self.base)
            record.assert_not_called()
        self.assertEqual(batch['selection']['look']['success']['seed'], 0)
        self.assertEqual(batch['selection']['look']['failure']['seed'], 1)

    def test_no_success_is_explicit_partial_not_fabricated(self):
        for seed in range(3):
            with patch.object(queue, 'execute', return_value=(self.report(seed, ['failure']), None)):
                queue.collect_one(self.config, self.index, self.base)
        batch = self.index['batches']['primitives_update0200']
        self.assertEqual(batch['state'], 'partial')
        self.assertEqual(batch['missing'], {'look': ['success']})
        self.assertNotIn('success', batch['selection']['look'])

    def test_do_not_read_checkpoint_still_being_saved(self):
        cp = queue.checkpoint(self.config, 'primitives', 0, 200)
        os.utime(cp, None)
        with patch.object(queue, 'execute') as record:
            queue.collect_one(self.config, self.index, self.base)
            record.assert_not_called()

    def test_reject_mixed_checkpoint_updates(self):
        report = self.report(0, ['success'])
        report['update'] = 400
        with patch.object(queue, 'execute', return_value=(report, None)):
            with self.assertRaisesRegex(ValueError, 'provenance'):
                queue.collect_one(self.config, self.index, self.base)


if __name__ == '__main__':
    unittest.main()
