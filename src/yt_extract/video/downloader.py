"""yt-dlp wrapper with comprehensive error handling."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import yt_dlp

from ..config import (
    DOWNLOAD_RETRY_BASE_DELAY,
    DOWNLOAD_RETRY_COUNT,
)
from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Wraps yt-dlp with structured error handling and retry logic."""

    def __init__(
        self,
        output_dir: Path,
        report: ExtractionReport,
        audio_format: str = "m4a",
    ) -> None:
        self.output_dir = output_dir
        self.report = report
        self.audio_format = audio_format

    def _build_video_opts(self, info_only: bool = False) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "skip_download": info_only,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "extract_flat": False,
            "writethumbnail": True,
            "writeinfojson": True,
            # Download video + audio merged, no audio extraction post-processor
        }
        return opts

    def _build_audio_opts(self, info_only: bool = False) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "skip_download": info_only,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "extract_flat": False,
            "writethumbnail": False,
            "writeinfojson": False,
            "format": "bestaudio",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.audio_format,
                    "preferredquality": "192",
                },
            ],
        }
        return opts

    def _build_opts(self, info_only: bool = False) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
            "skip_download": info_only,
            "noplaylist": True,
            "quiet": False,
            "no_warnings": False,
            "extract_flat": False,
            "writethumbnail": True,
            "writeinfojson": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": self.audio_format,
                    "preferredquality": "192",
                },
            ],
        }
        return opts

    def extract_info(self, url: str) -> dict[str, Any] | None:
        """Extract video metadata without downloading."""
        opts = self._build_opts(info_only=True)
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as exc:
            self._handle_ytdl_error(exc, url)
            return None

    def download(self, url: str) -> Path | None:
        """Download video and audio separately.

        Returns path to video file (primary) or audio file (audio-only fallback).
        """
        video_path = self.output_dir / "video.mp4"
        audio_path = self.output_dir / "audio.m4a"
        info_path = self.output_dir / "info.json"
        temp_video = self.output_dir / "temp.%(ext)s"

        for attempt in range(1, DOWNLOAD_RETRY_COUNT + 1):
            logger.info("Download attempt %d/%d for %s", attempt, DOWNLOAD_RETRY_COUNT, url)
            try:
                # Step 1: Get metadata
                opts_info = self._build_video_opts(info_only=True)
                with yt_dlp.YoutubeDL(opts_info) as ydl:
                    info = ydl.extract_info(url, download=False)

                if not info:
                    logger.error("No metadata returned")
                    return None

                # Step 2: Download video stream
                opts_video = self._build_video_opts()
                opts_video["outtmpl"] = str(temp_video)
                opts_video["format"] = "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"
                with yt_dlp.YoutubeDL(opts_video) as ydl:
                    ydl.extract_info(url, download=True)

                # Find downloaded video file
                downloaded_video = None
                for f in self.output_dir.glob("temp.*"):
                    if f.suffix in (".mp4", ".mkv", ".webm"):
                        downloaded_video = f
                        break

                if downloaded_video:
                    downloaded_video.rename(video_path)
                    logger.info("Downloaded video: %s", video_path.name)

                # Step 3: Download and extract audio
                opts_audio = self._build_audio_opts()
                with yt_dlp.YoutubeDL(opts_audio) as ydl:
                    ydl.extract_info(url, download=True)

                # Find downloaded audio file
                for f in self.output_dir.glob("*.m4a"):
                    if f.name != "audio.m4a":
                        f.rename(audio_path)
                        logger.info("Downloaded audio: %s", audio_path.name)
                        break

                # Step 4: Write info.json
                safe_info = {
                    k: v
                    for k, v in info.items()
                    if k not in ("thumbnails", "automatic_captions", "subtitles")
                }
                with open(info_path, "w") as f:
                    json.dump(safe_info, f, indent=2, default=str)
                logger.info("Wrote info.json")

                # Return primary download path
                if video_path.exists() and video_path.stat().st_size > 0:
                    return video_path
                if audio_path.exists() and audio_path.stat().st_size > 0:
                    return audio_path

                # Clean up temp files
                for f in self.output_dir.glob("temp.*"):
                    f.unlink(missing_ok=True)

            except Exception as exc:
                error_msg = str(exc).lower()

                if "cannot download" in error_msg or "does not exist" in error_msg:
                    self.report.add_error(ErrorCode.VIDEO_NOT_FOUND, str(exc))
                    return None

                if (
                    "private video" in error_msg
                    or "login required" in error_msg
                    or "members only" in error_msg
                ):
                    self.report.add_error(ErrorCode.VIDEO_PRIVATE, str(exc), recoverable=False)
                    return None

                if "deleted" in error_msg or "content removed" in error_msg:
                    self.report.add_error(ErrorCode.VIDEO_DELETED, str(exc), recoverable=False)
                    return None

                if (
                    "geo" in error_msg
                    or "geoblocked" in error_msg
                    or "available only in" in error_msg
                ):
                    self.report.add_warning(ErrorCode.GEO_RESTRICTED, str(exc), recoverable=False)
                    # Continue — audio might still be available

                if "live" in error_msg or "is_live" in error_msg or "live stream" in error_msg:
                    self.report.add_error(ErrorCode.LIVE_STREAM, str(exc), recoverable=False)
                    return None

                if (
                    "rate limit" in error_msg
                    or "429" in error_msg
                    or "too many requests" in error_msg
                ):
                    delay = DOWNLOAD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning("Rate limited, waiting %.1fs before retry...", delay)
                    time.sleep(delay)
                    if attempt == DOWNLOAD_RETRY_COUNT:
                        self.report.add_error(ErrorCode.RATE_LIMITED, str(exc), recoverable=False)
                        return None
                    continue

                if "format not available" in error_msg or "no suitable format" in error_msg:
                    self.report.add_error(ErrorCode.FORMAT_NOT_SUPPORTED, str(exc))
                    if attempt == DOWNLOAD_RETRY_COUNT:
                        return None
                    continue

                # Unknown error — retry with backoff
                if attempt < DOWNLOAD_RETRY_COUNT:
                    delay = DOWNLOAD_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning("Download error: %s, retrying in %.1fs...", str(exc)[:80], delay)
                    time.sleep(delay)
                    continue
                else:
                    self.report.add_error(ErrorCode.DOWNLOAD_FAILED, str(exc), recoverable=False)
                    logger.error(
                        "Download failed after %d attempts: %s",
                        DOWNLOAD_RETRY_COUNT,
                        str(exc)[:200],
                    )
                    return None

        return None

    def _handle_ytdl_error(self, exc: Exception, url: str) -> None:
        """Classify and report yt-dlp errors."""
        error_msg = str(exc).lower()
        if "rate limit" in error_msg or "429" in error_msg:
            self.report.add_error(ErrorCode.RATE_LIMITED, str(exc))
        elif "private video" in error_msg or "login required" in error_msg:
            self.report.add_error(ErrorCode.VIDEO_PRIVATE, str(exc), recoverable=False)
        elif "geo" in error_msg or "geoblocked" in error_msg:
            self.report.add_warning(ErrorCode.GEO_RESTRICTED, str(exc))
        elif "live" in error_msg or "is_live" in error_msg:
            self.report.add_error(ErrorCode.LIVE_STREAM, str(exc), recoverable=False)
        else:
            self.report.add_error(ErrorCode.DOWNLOAD_FAILED, str(exc), recoverable=False)
            logger.error("yt-dlp error: %s", str(exc)[:500])


def get_ffprobe_path() -> str:
    """Find ffprobe binary, trying common paths."""

    # Try ffprobe directly
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe

    # Try ffmpeg (often has ffprobe bundled)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    raise FileNotFoundError(
        "ffprobe not found. Install ffmpeg: brew install ffmpeg (macOS), "
        "sudo apt install ffmpeg (Ubuntu), or winget install ffmpeg (Windows)"
    )
