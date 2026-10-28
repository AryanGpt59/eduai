"""Local audio-level measurements. No speech recognition is performed."""

from __future__ import annotations

from io import BytesIO
import math

import numpy as np
import soundfile as sf


def rms_dbfs(samples: np.ndarray) -> float:
    """Measure signal level in dBFS; silence is represented as -120 dBFS."""
    values = np.asarray(samples, dtype=np.float64)
    if values.size == 0:
        return -120.0
    if not np.isfinite(values).all():
        raise ValueError("audio contains non-finite sample values")
    rms = float(np.sqrt(np.mean(np.square(values))))
    if rms <= 1e-6:
        return -120.0
    return max(-120.0, min(0.0, 20.0 * math.log10(rms)))


def activity_level(dbfs: float, noise_floor_dbfs: float = -45.0) -> float:
    """Map measured level to a 0..1 sound-activity proxy."""
    if noise_floor_dbfs >= 0:
        raise ValueError("noise_floor_dbfs must be below 0 dBFS")
    return max(0.0, min(1.0, (float(dbfs) - noise_floor_dbfs) / -noise_floor_dbfs))


def analyze_wav_bytes(data: bytes, window_seconds: float = 1.0) -> list[dict[str, float]]:
    """Analyze an uploaded audio file into timestamped RMS windows."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    try:
        samples, sample_rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc
    if samples.size == 0 or sample_rate <= 0:
        raise ValueError("The audio file is empty or has an invalid sample rate")
    mono = samples.mean(axis=1)
    window_size = max(1, int(round(sample_rate * window_seconds)))
    rows = []
    for start in range(0, len(mono), window_size):
        chunk = mono[start : start + window_size]
        dbfs = rms_dbfs(chunk)
        rows.append(
            {
                "seconds": round(start / sample_rate, 2),
                "rms_dbfs": round(dbfs, 2),
                "activity_level": round(activity_level(dbfs), 3),
            }
        )
    return rows
