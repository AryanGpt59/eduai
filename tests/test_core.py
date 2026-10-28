from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

from audio import analyze_wav_bytes, rms_dbfs
from core import AttendanceTracker, motion_activity_index
from history import clear_summaries, list_summaries, save_summary
from insights import teaching_prompts


class AttendanceTrackerTests(unittest.TestCase):
    def test_sustained_nearby_detections_accumulate_observed_time(self):
        tracker = AttendanceTracker(confirmation_seconds=2.0)
        for second in (0.0, 0.5, 1.0, 1.5, 2.0):
            tracker.update(second, [(10 + second, 10, 40, 40)])
        row = tracker.records()[0]
        self.assertEqual(row["attendance"], "Confirmed")
        self.assertAlmostEqual(row["observed_seconds"], 2.0)

    def test_long_gap_does_not_count_toward_presence(self):
        tracker = AttendanceTracker(confirmation_seconds=1.0)
        tracker.update(0.0, [(0, 0, 30, 30)])
        tracker.update(0.5, [(0, 0, 30, 30)])
        active_rows = tracker.update(3.0, [(0, 0, 30, 30)])
        rows = tracker.records()
        self.assertEqual(len(active_rows), 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["observed_seconds"], 0.5)
        self.assertEqual(rows[0]["attendance"], "Below threshold")

    def test_two_detections_do_not_share_one_track(self):
        tracker = AttendanceTracker()
        rows = tracker.update(0.0, [(0, 0, 20, 20), (200, 0, 20, 20)])
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["track_id"], rows[1]["track_id"])

    def test_bad_timestamps_are_rejected(self):
        tracker = AttendanceTracker()
        tracker.update(2.0, [])
        with self.assertRaises(ValueError):
            tracker.update(1.0, [])


class ActivityAndStorageTests(unittest.TestCase):
    def test_audio_level_and_windowing(self):
        self.assertEqual(rms_dbfs(np.zeros(10)), -120.0)
        self.assertAlmostEqual(rms_dbfs(np.full(10, 0.5)), -6.02, places=2)
        samples = np.full(20, 0.25, dtype=np.float32)
        buffer = BytesIO()
        sf.write(buffer, samples, 10, format="WAV")
        windows = analyze_wav_bytes(buffer.getvalue(), window_seconds=1.0)
        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0]["seconds"], 0.0)
        self.assertAlmostEqual(windows[1]["seconds"], 1.0)

    def test_motion_index_is_bounded(self):
        self.assertEqual(motion_activity_index(-4), 0.0)
        self.assertEqual(motion_activity_index(100), 1.0)

    def test_prompts_state_that_signals_are_not_learning_measures(self):
        prompts = teaching_prompts(
            confirmed_tracks=0,
            peak_faces=0,
            mean_motion=0.0,
            audio_activity=None,
        )
        self.assertTrue(any("not measures" in item for item in prompts))

    def test_local_history_round_trip_and_clear(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = Path(directory) / "history.sqlite3"
            save_summary({"confirmed_tracks": 2}, db)
            self.assertEqual(list_summaries(db)[0]["confirmed_tracks"], 2)
            clear_summaries(db)
            self.assertEqual(list_summaries(db), [])


if __name__ == "__main__":
    unittest.main()
