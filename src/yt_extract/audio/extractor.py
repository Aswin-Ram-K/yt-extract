"""Audio extraction: normalize, slice, and prepare for feature computation."""

from __future__ import annotations

import logging
from pathlib import Path

from pydub import AudioSegment

from ..config import AUDIO_SAMPLE_RATE
from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)

# Supported audio formats from yt-dlp
SUPPORTED_FORMATS = {"m4a", "mp3", "wav", "opus", "flac", "webm"}


def extract_audio(
    video_path: Path | str,
    output_dir: Path,
    report: ExtractionReport,
    target_format: str = "mp3",
) -> Path | None:
    """Extract audio from video file using pydub (which uses ffmpeg internally).

    Returns path to extracted audio, or None on failure.
    """
    video_path = Path(video_path)
    output_path = output_dir / f"audio.{target_format}"

    if not video_path.exists():
        report.add_error(
            ErrorCode.AUDIO_EXTRACT_FAILED, f"Video file not found: {video_path}", recoverable=False
        )
        return None

    try:
        # Load video (pydub can load video files via ffmpeg)
        audio = AudioSegment.from_file(str(video_path))

        # Export as audio
        audio.export(str(output_path), format=target_format)

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(
                "Audio extracted: %s (%d channels, %d Hz)",
                output_path.name,
                audio.channels,
                audio.frame_rate,
            )
            return output_path
        else:
            report.add_error(
                ErrorCode.AUDIO_EXTRACT_FAILED,
                "Audio file is empty after extraction",
                recoverable=False,
            )
            return None

    except Exception as exc:
        logger.error("Audio extraction failed: %s", str(exc)[:200])
        report.add_error(ErrorCode.AUDIO_EXTRACT_FAILED, str(exc), recoverable=False)
        return None


def normalize_audio(
    input_path: Path | str,
    output_dir: Path,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> Path | None:
    """Normalize audio to mono at specified sample rate for librosa processing.

    Returns path to normalized WAV file.
    """
    input_path = Path(input_path)
    output_path = output_dir / "audio_normalized.wav"

    try:
        audio = AudioSegment.from_file(str(input_path))
        # Convert to mono and resample
        audio = audio.set_channels(1).set_frame_rate(sample_rate)
        audio.export(str(output_path), format="wav")

        if output_path.exists():
            logger.info("Audio normalized: %s → %s", input_path.name, output_path.name)
            return output_path
    except Exception as exc:
        logger.error("Audio normalization failed: %s", str(exc)[:200])

    return None
