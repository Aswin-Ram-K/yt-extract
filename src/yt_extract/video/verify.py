"""Post-download verification using ffprobe."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)


def verify_download(
    video_path: Path | str,
    audio_path: Path | str | None = None,
    thumb_path: Path | str | None = None,
    report: ExtractionReport | None = None,
) -> bool:  # noqa: PLR0915
    """Verify downloaded files are valid and contain expected streams."""
    video_path = Path(video_path)
    all_ok = True

    # Check video file
    if video_path.exists() and video_path.stat().st_size > 0:
        if not _probe_stream(video_path, "video"):
            report.add_error(
                ErrorCode.CORRUPT_FILE,
                f"Video file has no video stream: {video_path}",
                recoverable=False,
            )
            all_ok = False
        else:
            logger.info("Video verification OK: %s", video_path.name)
    elif video_path.exists():
        report.add_error(
            ErrorCode.CORRUPT_FILE,
            f"Video file is 0 bytes: {video_path}",
            recoverable=False,
        )
        all_ok = False
    else:
        report.add_error(
            ErrorCode.DOWNLOAD_FAILED,
            f"Video file not found: {video_path}",
            recoverable=False,
        )
        all_ok = False

    # Check audio file if provided
    if audio_path:
        audio_path = Path(audio_path)
        if audio_path.exists() and audio_path.stat().st_size > 0:
            if not _probe_stream(audio_path, "audio"):
                report.add_warning(
                    ErrorCode.CORRUPT_FILE,
                    f"Audio file has no audio stream: {audio_path}",
                    recoverable=True,
                )
        elif audio_path.exists():
            report.add_warning(
                ErrorCode.CORRUPT_FILE,
                f"Audio file is 0 bytes: {audio_path}",
                recoverable=True,
            )
        else:
            report.add_warning(
                ErrorCode.DOWNLOAD_FAILED,
                f"Audio file not found (video-only): {audio_path}",
                recoverable=True,
            )

    # Check thumbnail
    if thumb_path:
        thumb_path = Path(thumb_path)
        if thumb_path.exists() and thumb_path.stat().st_size > 100:  # minimal valid file
            logger.info("Thumbnail verification OK: %s", thumb_path.name)
        elif thumb_path.exists():
            report.add_warning(
                ErrorCode.CORRUPT_FILE,
                f"Thumbnail file too small: {thumb_path}",
                recoverable=True,
            )

    return all_ok


def _probe_stream(file_path: Path, stream_type: str, retries: int = 3) -> bool:
    """Use ffprobe to check if file has the expected stream type.

    Retries with short delay in case the file is still being written.
    """
    import json

    for attempt in range(retries):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-of",
                    "json",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=15,
            )
            data = json.loads(result.stdout) if result.stdout else {}
            streams = data.get("streams", [])
            video_streams = [s for s in streams if s.get("codec_type") == "video"]
            audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
            if stream_type == "video":
                return len(video_streams) > 0
            elif stream_type == "audio":
                return len(audio_streams) > 0
            return False
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
            json.JSONDecodeError,
        ):
            if attempt < retries - 1:
                import time

                time.sleep(0.2 * (attempt + 1))
                continue
            logger.error("ffprobe failed after %d retries: %s", retries, file_path.name)
            return False
        except Exception as exc:
            logger.error("ffprobe error: %s", str(exc)[:200])
            if attempt < retries - 1:
                import time

                time.sleep(0.2 * (attempt + 1))
                continue
            return False

    return False


def get_video_metadata(video_path: Path | str) -> dict:
    """Extract video metadata using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        import json

        return json.loads(result.stdout) if result.stdout else {}
    except Exception:
        return {}
