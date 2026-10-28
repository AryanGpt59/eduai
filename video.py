"""Video sampling and face-location analysis for a local classroom demo."""

from __future__ import annotations

import math
from pathlib import Path

from core import AttendanceTracker, motion_activity_index


def analyze_video(
    path: str | Path,
    *,
    confirmation_seconds: float = 3.0,
    max_samples: int = 900,
    samples_per_second: float = 2.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    """Sample a video locally and return timeline, anonymous tracks, and metadata."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is missing. Install the dependencies from requirements.txt.") from exc
    if max_samples < 1 or samples_per_second <= 0:
        raise ValueError("max_samples and samples_per_second must be positive")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The video could not be opened. Try a standard MP4, AVI, MOV, or MKV file.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_stride = max(1, int(round(fps / samples_per_second)))
    cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        capture.release()
        raise RuntimeError("OpenCV's face-detection model could not be loaded")

    tracker = AttendanceTracker(confirmation_seconds=confirmation_seconds)
    timeline: list[dict[str, object]] = []
    frame_index = 0
    sampled = 0
    try:
        while sampled < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_stride == 0:
                seconds = frame_index / fps
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(36, 36),
                )
                tracks = tracker.update(seconds, detections)
                timeline.append(
                    {
                        "seconds": round(seconds, 2),
                        "detected_faces": len(detections),
                        "active_tracks": len(tracks),
                        "mean_motion_pixels": round(tracker.last_motion_pixels, 2),
                        "motion_activity_proxy": round(motion_activity_index(tracker.last_motion_pixels), 3),
                    }
                )
                sampled += 1
            frame_index += 1
    finally:
        capture.release()

    if not timeline:
        raise ValueError("No video frames could be read from this file")
    duration = frame_count / fps if frame_count > 0 else float(timeline[-1]["seconds"])
    metadata = {
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "sampled_frames": float(sampled),
        "sample_interval_seconds": round(sample_stride / fps, 2),
    }
    return timeline, tracker.records(), metadata
