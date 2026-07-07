"""Audio silence detection and segment splitting."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from ..config import AUDIO_TOP_DB
from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)


def detect_silence(
    audio_path: Path | str,
    sample_rate: int = 22050,
    top_db: float = AUDIO_TOP_DB,
    report: ExtractionReport | None = None,
) -> dict:
    """Detect silent and non-silent segments in audio.

    Returns dict with:
        - total_duration: float (seconds)
        - silent_duration: float
        - non_silent_duration: float
        - segments: list of {start_s, end_s, is_silent, confidence}
        - is_completely_silent: bool
    """
    audio_path = Path(audio_path)

    try:
        y, sr = librosa.load(str(audio_path), sr=sample_rate, duration=None)
    except Exception as exc:
        logger.error("librosa.load failed: %s", str(exc)[:200])
        if report:
            report.add_error(ErrorCode.LIBROSA_SILENCE_ERROR, str(exc), recoverable=False)
        return {"total_duration": 0, "is_completely_silent": True}

    # Check for completely silent audio (all zeros or near-zero)
    if np.allclose(y, 0, atol=1e-6):
        duration = len(y) / sr if sr > 0 else 0
        if report:
            report.add_warning(
                ErrorCode.AUDIO_SILENT,
                "Audio is completely silent, skipping feature extraction.",
                recoverable=True,
            )
        return {
            "total_duration": duration,
            "is_completely_silent": True,
            "silent_duration": duration,
            "non_silent_duration": 0,
            "segments": [],
        }

    # Use librosa.effects.trim to find non-silent boundaries
    try:
        y_trimmed, trim_indices = librosa.effects.trim(y, top_db=top_db, frame_length=2048)

        sr_val = sr if sr > 0 else sample_rate
        total_duration = len(y) / sr_val
        trim_start = trim_indices[0] / sr_val if len(trim_indices) > 0 else 0
        trim_end = trim_indices[1] / sr_val if len(trim_indices) > 0 else total_duration
        non_silent_duration = trim_end - trim_start
        silent_duration = total_duration - non_silent_duration

        segments = [
            {"start_s": 0.0, "end_s": trim_start, "is_silent": True, "confidence": 0.9},
            {"start_s": trim_start, "end_s": trim_end, "is_silent": False, "confidence": 0.9},
            {"start_s": trim_end, "end_s": total_duration, "is_silent": True, "confidence": 0.9},
        ]
        # Remove empty segments (zero-length)
        segments = [s for s in segments if s["end_s"] > s["start_s"]]

        # Trim the audio for feature extraction (only non-silent portion)
        trim_start_idx = int(trim_start * sr_val) if sr_val > 0 else 0
        trim_end_idx = int(trim_end * sr_val) if sr_val > 0 else len(y)
        y_for_features = y[trim_start_idx:trim_end_idx]

        result = {
            "total_duration": total_duration,
            "silent_duration": silent_duration,
            "non_silent_duration": non_silent_duration,
            "is_completely_silent": False,
            "segments": segments,
            "y_trimmed": y_for_features,
            "sr": sr_val,
        }

        if report:
            report.add_phase_report(
                "silence_detection",
                {
                    "total_duration": round(total_duration, 2),
                    "non_silent_duration": round(non_silent_duration, 2),
                    "silent_duration": round(silent_duration, 2),
                    "segments_count": len(segments),
                },
            )

        return result

    except Exception as exc:
        logger.warning("Silence detection failed, using full audio: %s", str(exc)[:200])
        if report:
            report.add_warning(
                ErrorCode.LIBROSA_SILENCE_ERROR,
                f"Silence detection failed: {str(exc)[:100]}",
                recoverable=True,
            )
        # Return full audio with no silence split
        return {
            "total_duration": len(y) / sr if sr > 0 else 0,
            "silent_duration": 0,
            "non_silent_duration": len(y) / sr if sr > 0 else 0,
            "is_completely_silent": False,
            "segments": [
                {
                    "start_s": 0,
                    "end_s": len(y) / sr if sr > 0 else 0,
                    "is_silent": False,
                    "confidence": 0.5,
                }
            ],
            "y_trimmed": y,  # use full audio
            "sr": sr,
        }
