"""Output module: manifest builder, report card, serialization."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ..config import MANIFEST_VERSION
from ..errors import ExtractionReport

logger = logging.getLogger(__name__)


def build_manifest(
    source_url: str,
    yt_info: dict,
    video_metadata: dict,
    frames: list[dict],
    audio_info: dict,
    ocr_results: list[dict] | None = None,
    report: ExtractionReport | None = None,
) -> dict:
    """Build the complete extraction manifest."""
    manifest: dict = {
        "version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "type": "youtube",
            "url": source_url,
        },
        "video": {},
        "frames": {
            "total": len(frames),
            "items": [],
        },
        "audio": audio_info,
        "ocr": {
            "enabled": ocr_results is not None,
            "total_frames_processed": len(ocr_results) if ocr_results else 0,
            "with_text": sum(1 for r in (ocr_results or [])) if ocr_results else 0,
        },
        "errors": [],
        "warnings": [],
        "phases": {},
    }

    # Source info from yt-dlp
    if yt_info:
        manifest["source"]["title"] = yt_info.get("title", "")
        manifest["source"]["channel"] = yt_info.get("channel", yt_info.get("uploader", ""))
        manifest["source"]["duration"] = yt_info.get("duration", 0)
        manifest["source"]["view_count"] = yt_info.get("view_count", 0)
        manifest["source"][
            "resolution"
        ] = f"{yt_info.get('width', '?')}x{yt_info.get('height', '?')}"
        manifest["source"]["fps"] = yt_info.get("fps", 0) or 0
        manifest["source"]["thumbnails"] = [
            t.get("url", "") for t in yt_info.get("thumbnails", [])[:3] if t.get("url")
        ]

        # Chapters
        chapters = yt_info.get("chapters", []) or yt_info.get("chapter_metadata", [])
        if chapters:
            manifest["source"]["chapters"] = [
                {
                    "title": c.get("title", f"Chapter {i}"),
                    "start_s": c.get("start_time", c.get("timestamp", 0)),
                    "end_s": c.get("end_time", ""),
                }
                for i, c in enumerate(chapters)
            ]

    # Video metadata from ffprobe
    if video_metadata:
        fmt = video_metadata.get("format", {})
        streams = video_metadata.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

        manifest["video"] = {
            "duration": float(fmt.get("duration", 0)),
            "size_bytes": int(fmt.get("size", 0)),
            "format": fmt.get("format_name", ""),
            "codec": video_stream.get("codec_name", ""),
            "width": video_stream.get("width", 0),
            "height": video_stream.get("height", 0),
            "fps": video_stream.get("r_frame_rate", ""),
        }

    # Frames
    for f in frames[:100]:  # Cap manifest size
        manifest["frames"]["items"].append(
            {
                "index": f.get("index", 0),
                "timestamp": f.get("timestamp", 0),
                "score": f.get("score", 0),
                "threshold": f.get("threshold", 0),
                "filepath": f.get("filepath", ""),
            }
        )
    manifest["frames"]["showing"] = min(100, len(frames))
    manifest["frames"]["total"] = len(frames)
    manifest["frames"]["truncated"] = len(frames) > 100

    # Errors and warnings
    if report:
        manifest["errors"] = [e.to_dict() for e in report.errors]
        manifest["warnings"] = [w.to_dict() for w in report.warnings]
        manifest["phases"] = report.phases

    return manifest


def write_manifest(manifest: dict, output_dir: Path) -> Path:
    """Write manifest.json to output directory."""
    manifest_path = output_dir / "manifest.json"
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Wrote manifest.json (%d bytes)", manifest_path.stat().st_size)
        return manifest_path
    except Exception as exc:
        logger.error("Failed to write manifest: %s", exc)
        raise


def generate_report_card(manifest: dict) -> str:
    """Generate a human-readable summary paragraph."""
    source = manifest.get("source", {})
    video = manifest.get("video", {})
    frames_info = manifest.get("frames", {})
    audio_info = manifest.get("audio", {})

    lines = [
        f"Extracted {frames_info.get('total', 0)} frames from '{source.get('title', 'Unknown')}'.",
    ]

    duration = video.get("duration", 0) or source.get("duration", 0) or 0
    resolution = video.get("resolution")
    if not resolution:
        w = video.get("width", "?")
        h = video.get("height", "?")
        resolution = f"{w}x{h}"
    if duration > 0:
        lines.append(f"Video: {int(duration // 60):02d}:{int(duration % 60):02d} ({resolution})")

    if frames_info.get("total", 0) > 0:
        fps_est = frames_info["total"] / duration if duration > 0 else 0
        lines.append(f"Frame rate: ~{fps_est:.1f} fps (scene cuts + intervals)")

    if audio_info:
        audio_duration = audio_info.get("duration", 0) or 0
        audio_sr = audio_info.get("sample_rate", 0)
        audio_features = audio_info.get("feature_count", 0)
        audio_segments = audio_info.get("segment_count", 0)
        lines.append(
            f"Audio: {int(audio_duration // 60):02d}:{int(audio_duration % 60):02d} at {audio_sr} Hz, {audio_features} features, {audio_segments} segments"
        )

    if manifest.get("ocr", {}).get("enabled"):
        ocr = manifest["ocr"]
        lines.append(
            f"OCR: {ocr.get('with_text', 0)}/{ocr.get('total_frames_processed', 0)} frames with text"
        )

    errors = manifest.get("errors", [])
    warnings = manifest.get("warnings", [])
    if errors:
        lines.append(f"Errors: {len(errors)}")
    if warnings:
        lines.append(f"Warnings: {len(warnings)}")

    return " ".join(lines)
