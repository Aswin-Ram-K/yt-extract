"""Orchestrates the full extraction pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .audio.extractor import extract_audio, normalize_audio
from .audio.features import extract_features
from .audio.silence import detect_silence
from .errors import ErrorCode, ExtractionReport
from .ocr.detector import batch_ocr
from .ocr.engine import check_ocr_dependencies
from .output.manifest import build_manifest, generate_report_card, write_manifest
from .video.downloader import VideoDownloader
from .video.frames import extract_frames
from .video.verify import get_video_metadata, verify_download

logger = logging.getLogger(__name__)


def extract(
    url: str,
    output_dir: Path,
    frame_budget: int = 200,
    scdet_auto: bool = True,
    scdet_threshold: float | None = None,
    include_interval: bool = True,
    enable_ocr: bool = False,
    ocr_lang: str = "eng",
    frame_pixel: int = 720,
) -> dict:
    """Run full extraction pipeline.

    Returns dict with manifest, report, and timing info.
    """
    start_time = time.time()
    report = ExtractionReport()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === PHASE 1: Download ===
    logger.info("Phase 1: Download from %s", url)
    downloader = VideoDownloader(output_dir, report, audio_format="m4a")
    info = downloader.extract_info(url)
    if info is None:
        logger.error("Failed to get video info, aborting.")
        manifest = build_manifest(url, {}, {}, [], {}, None, report)
        write_manifest(manifest, output_dir)
        return {
            "manifest": manifest,
            "report": "Failed: could not fetch video info.",
            "elapsed": time.time() - start_time,
        }

    video_id = info.get("id", "unknown")
    video_path = output_dir / "video.mp4"
    audio_path = output_dir / "audio.m4a"
    thumb_path = output_dir / f"{video_id}.webp"
    info_path = output_dir / "info.json"

    # Download
    downloaded = downloader.download(url)
    if downloaded is None:
        logger.error("Download failed, aborting.")
        manifest = build_manifest(url, info, {}, [], {}, None, report)
        write_manifest(manifest, output_dir)
        return {
            "manifest": manifest,
            "report": "Failed: download error.",
            "elapsed": time.time() - start_time,
        }

    # Determine which file we got
    if video_path.exists():
        pass  # video downloaded
    elif audio_path.exists():
        pass  # audio-only (geo-restricted)
    else:
        logger.warning("Unknown download output, searching for files...")

    # Write info.json
    with open(info_path, "w") as f:
        safe_info = {
            k: v
            for k, v in info.items()
            if k not in ("thumbnails", "automatic_captions", "subtitles")
        }
        json.dump(safe_info, f, indent=2, default=str)

    # === PHASE 2: Verify ===
    logger.info("Phase 2: Verify downloads")
    verify_download(video_path, audio_path, thumb_path, report)

    # Get ffprobe metadata
    video_metadata = get_video_metadata(video_path) if video_path.exists() else {}

    # === PHASE 3: Extract frames ===
    logger.info("Phase 3: Extract frames")
    frames_path = output_dir / "frames"
    frames_path.mkdir(exist_ok=True)
    frames = extract_frames(
        str(video_path),
        output_dir,
        report,
        budget=frame_budget,
        auto_scdet=scdet_auto,
        scdet_threshold=scdet_threshold,
        include_interval=include_interval,
    )

    # === PHASE 4: Extract audio features ===
    logger.info("Phase 4: Extract audio features")
    audio_features_path = output_dir / "audio_features"
    audio_features_path.mkdir(exist_ok=True)

    # Normalize audio for librosa
    normalized_audio = (
        normalize_audio(video_path, audio_features_path) if video_path.exists() else None
    )
    if normalized_audio is None:
        # Try extracting audio from video
        normalized_audio = (
            extract_audio(video_path, audio_features_path, report) if video_path.exists() else None
        )

    audio_info: dict = {"duration": 0, "sample_rate": 0, "feature_count": 0, "segment_count": 0}
    if normalized_audio and normalized_audio.exists():
        # Silence detection
        silence_info = detect_silence(normalized_audio, report=report)

        # Feature extraction (on non-silent portion)
        y_trimmed = silence_info.get("y_trimmed")
        sr = silence_info.get("sr", 22050)

        if y_trimmed is not None and len(y_trimmed) > 1000:
            feat_info = extract_features(
                normalized_audio,
                audio_features_path,
                report,
                sample_rate=sr,
            )
            audio_info.update(
                {
                    "duration": silence_info.get("total_duration", 0),
                    "sample_rate": sr,
                    "non_silent_duration": silence_info.get("non_silent_duration", 0),
                    "silent_duration": silence_info.get("silent_duration", 0),
                    "feature_count": feat_info.get("feature_count", 0),
                    "segment_count": feat_info.get("segment_count", 0),
                }
            )
        else:
            report.add_warning(
                ErrorCode.AUDIO_SILENT,
                "Audio too short or empty, skipping feature extraction.",
                recoverable=True,
            )
            audio_info["duration"] = silence_info.get("total_duration", 0)
            audio_info["sample_rate"] = sr
    elif info.get("duration"):
        audio_info["duration"] = info["duration"]

    # === PHASE 5: OCR (optional) ===
    ocr_results: list[dict] = []
    if enable_ocr:
        logger.info("Phase 5: OCR")
        if check_ocr_dependencies(report):
            from .ocr.engine import TESSERACT_AVAILABLE

            if TESSERACT_AVAILABLE:
                frame_paths = []
                for f in frames:
                    fp = output_dir / f["filepath"]
                    if fp.exists():
                        frame_paths.append(fp)
                if frame_paths:
                    ocr_results = batch_ocr(
                        frame_paths,
                        output_dir,
                        report,
                        lang=ocr_lang,
                    )

    # === PHASE 6: Build manifest ===
    logger.info("Phase 6: Build manifest")
    manifest = build_manifest(
        url,
        info,
        video_metadata,
        frames,
        audio_info,
        ocr_results if enable_ocr else None,
        report,
    )

    report.add_phase_report(
        "timing",
        {
            "total_seconds": round(time.time() - start_time, 2),
        },
    )

    report_path = write_manifest(manifest, output_dir)

    # Generate report card
    report_card = generate_report_card(manifest)

    elapsed = time.time() - start_time

    logger.info("Extraction complete: %s", report_card)

    return {
        "manifest": manifest,
        "report": report_card,
        "report_path": str(report_path),
        "elapsed": elapsed,
    }
