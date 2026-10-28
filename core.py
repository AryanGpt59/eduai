"""Session-local, anonymous presence tracking for EduAI.

This module deliberately tracks detected face locations, not identities. Track IDs
are temporary labels that are meaningful only during one video analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Sequence


Box = Sequence[float]


@dataclass
class _Track:
    track_id: int
    x: float
    y: float
    first_seen: float
    last_seen: float
    observed_seconds: float = 0.0
    observations: int = 1


class AttendanceTracker:
    """Match nearby detections and accumulate only short observed intervals."""

    def __init__(
        self,
        confirmation_seconds: float = 3.0,
        max_observation_gap_seconds: float = 1.0,
        retire_after_seconds: float = 1.5,
        max_match_distance: float = 100.0,
    ) -> None:
        if confirmation_seconds < 0:
            raise ValueError("confirmation_seconds must be non-negative")
        if max_observation_gap_seconds <= 0:
            raise ValueError("max_observation_gap_seconds must be positive")
        if retire_after_seconds < max_observation_gap_seconds:
            raise ValueError("retire_after_seconds must be at least the observation gap")
        if max_match_distance <= 0:
            raise ValueError("max_match_distance must be positive")
        self.confirmation_seconds = float(confirmation_seconds)
        self.max_observation_gap_seconds = float(max_observation_gap_seconds)
        self.retire_after_seconds = float(retire_after_seconds)
        self.max_match_distance = float(max_match_distance)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._last_timestamp: float | None = None
        self.last_motion_pixels = 0.0

    @staticmethod
    def _center(box: Box) -> tuple[float, float]:
        if len(box) != 4:
            raise ValueError("each detection must contain x, y, width, and height")
        x, y, width, height = (float(part) for part in box)
        return x + width / 2.0, y + height / 2.0

    def update(self, timestamp_seconds: float, boxes: Iterable[Box]) -> list[dict[str, object]]:
        """Update with detections at a video timestamp and return active track rows."""
        timestamp = float(timestamp_seconds)
        if timestamp < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        self._last_timestamp = timestamp

        centers = [self._center(box) for box in boxes]
        active = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen <= self.retire_after_seconds
        }

        candidates: list[tuple[float, int, int]] = []
        for track_id, track in active.items():
            for detection_index, (x, y) in enumerate(centers):
                distance = hypot(x - track.x, y - track.y)
                if distance <= self.max_match_distance:
                    candidates.append((distance, track_id, detection_index))
        candidates.sort()

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        movements: list[float] = []
        for distance, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self._tracks[track_id]
            gap = timestamp - track.last_seen
            if gap <= 0:
                movement = 0.0
            else:
                movement = distance
            if 0 < gap <= self.max_observation_gap_seconds:
                track.observed_seconds += gap
            track.x, track.y = centers[detection_index]
            track.last_seen = timestamp
            track.observations += 1
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            movements.append(movement)

        for detection_index, (x, y) in enumerate(centers):
            if detection_index in matched_detections:
                continue
            track = _Track(self._next_id, x, y, timestamp, timestamp)
            self._tracks[self._next_id] = track
            self._next_id += 1

        self.last_motion_pixels = sum(movements) / len(movements) if movements else 0.0
        return self.records(active_only=True)

    def records(self, active_only: bool = False) -> list[dict[str, object]]:
        """Return anonymous presence summaries; identity is never inferred."""
        tracks = self._tracks.values()
        rows = []
        for track in tracks:
            if active_only and self._last_timestamp is not None:
                if self._last_timestamp - track.last_seen > self.retire_after_seconds:
                    continue
            rows.append(
                {
                    "track_id": f"Guest {track.track_id}",
                    "first_seen_seconds": round(track.first_seen, 2),
                    "last_seen_seconds": round(track.last_seen, 2),
                    "observed_seconds": round(track.observed_seconds, 2),
                    "observations": track.observations,
                    "attendance": (
                        "Confirmed"
                        if track.observed_seconds >= self.confirmation_seconds
                        else "Below threshold"
                    ),
                }
            )
        return sorted(rows, key=lambda row: str(row["track_id"]))


def motion_activity_index(motion_pixels: float, reference_pixels: float = 40.0) -> float:
    """Return a bounded motion proxy, not an attention or learning score."""
    if reference_pixels <= 0:
        raise ValueError("reference_pixels must be positive")
    return max(0.0, min(1.0, float(motion_pixels) / reference_pixels))
