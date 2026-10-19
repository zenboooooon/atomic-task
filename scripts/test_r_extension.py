"""Check R queue capacity and evaluator exclusivity without starting any jobs."""
import unittest
from types import SimpleNamespace

import run_gaze_v10_r_extension as r


class RQueueSafety(unittest.TestCase):
    def completed_training(self):
        return dict(trained_jobs=list(r.ORIGINAL_JOBS), running_jobs=[], queued_jobs=[],
                    validations_complete=80, active_validation={"condition": "F", "seed": 2, "update": 2000})

    def test_no_R_training_before_all_original_jobs_exit(self):
        progress = self.completed_training()
        self.assertTrue(r.training_gate(progress, False))
        self.assertFalse(r.training_gate(progress, True))
        progress["trained_jobs"].pop()
        self.assertFalse(r.training_gate(progress, False))

    def test_pending_or_running_original_job_blocks_R(self):
        for key in ("running_jobs", "queued_jobs"):
            progress = self.completed_training()
            progress[key] = ["F_seed2"]
            self.assertFalse(r.training_gate(progress, False))

    def test_R_evaluation_waits_for_original_evaluator(self):
        progress = self.completed_training()
        self.assertFalse(r.evaluation_gate(progress))
        progress["validations_complete"] = 90
        self.assertFalse(r.evaluation_gate(progress))
        progress["active_validation"] = None
        self.assertTrue(r.evaluation_gate(progress))

    def test_combined_counts_and_original_validation_are_preserved(self):
        progress = self.completed_training()
        combined = r.combined_progress(progress, {0}, {1: SimpleNamespace(pid=123)}, {0, 1}, {(0, 200)}, None, "training_R")
        self.assertEqual(combined["training_jobs_total"], 12)
        self.assertEqual(combined["validations_total"], 120)
        self.assertEqual(combined["validations_complete"], 81)
        self.assertEqual(len(combined["trained_jobs"]), 10)
        self.assertEqual(combined["queued_jobs"], ["R_seed2"])
        self.assertEqual(combined["active_validation"], progress["active_validation"])

    def test_R_recording_queue_adds_ten_batches(self):
        from watch_gaze_v10_videos import specifications
        config = dict(checkpoint_interval=200, updates=2000, primitive_tasks=["look", "approach", "jump"],
                      downstream_conditions=["P", "S", "F", "R"])
        jobs = list(specifications(config))
        self.assertEqual(len(jobs), 50)
        self.assertEqual(len({job[0] for job in jobs}), 50)
        self.assertEqual([j[2] for j in jobs if j[1] == "R"], list(range(200, 2001, 200)))


if __name__ == "__main__":
    unittest.main()
