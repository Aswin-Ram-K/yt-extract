"""CLI entry point for yt-extract."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .extract import extract
from .output.manifest import write_manifest


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="yt-extract",
        description="Extract frames, audio features, and OCR from YouTube videos.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory (default: yt-extract-<video_id>/)",
    )
    parser.add_argument(
        "--frame-budget", "-n", type=int, default=200, help="Max frames to extract (default: 200)"
    )
    parser.add_argument(
        "--scdet-threshold",
        "-t",
        type=float,
        default=None,
        help="Scene detection threshold (default: auto-tune)",
    )
    parser.add_argument(
        "--scdet-auto",
        action="store_true",
        default=True,
        help="Auto-tune scdet threshold (default: true)",
    )
    parser.add_argument(
        "--no-scdet",
        dest="scdet_auto",
        action="store_false",
        help="Disable auto-tune, use manual threshold",
    )
    parser.add_argument(
        "--no-interval",
        dest="include_interval",
        action="store_false",
        default=True,
        help="Skip interval-based frame sampling",
    )
    parser.add_argument(
        "--ocr", action="store_true", default=False, help="Enable OCR on scene-cut frames"
    )
    parser.add_argument("--ocr-lang", type=str, default="eng", help="OCR language (default: eng)")
    parser.add_argument(
        "--compact",
        action="store_true",
        default=False,
        help="Compact output: omit full feature arrays from manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Generate manifest only, skip extraction",
    )
    parser.add_argument(
        "--frame-pixel", type=int, default=720, help="Frame width in pixels (default: 720)"
    )
    parser.add_argument(
        "--log-level",
        "-v",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        default=False,
        help="Output manifest to stdout as JSON",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create output directory
    from yt_extract.video.downloader import VideoDownloader
    from yt_extract.errors import ExtractionReport

    # For dry-run, we still need to know output dir
    try:
        # Try to get video ID first to create output dir
        downloader = VideoDownloader(Path("."), ExtractionReport())
        info = downloader.extract_info(args.url)
        if info is None:
            print("ERROR: Could not fetch video info. Check URL and network.", file=sys.stderr)
            sys.exit(1)
        video_id = info.get("id", "unknown")
        output_dir = args.output_dir or Path(f"yt-extract-{video_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        info_path = output_dir / "info.json"
        with open(info_path, "w") as f:
            safe_info = {
                k: v for k, v in info.items() if k not in ("thumbnails", "automatic_captions")
            }
            json.dump(safe_info, f, indent=2, default=str)
    except Exception as exc:
        print(f"WARNING: Could not pre-fetch video info: {exc}", file=sys.stderr)
        output_dir = args.output_dir or Path("yt-extract-unknown")
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Generate manifest from info only
        manifest = {
            "version": "1.0",
            "generated_at": "",
            "source": {"type": "youtube", "url": args.url},
            "frames": {"total": 0, "items": []},
            "audio": {"duration": 0},
            "ocr": {"enabled": False},
            "errors": [],
            "warnings": [],
        }
        if info:
            manifest["source"]["title"] = info.get("title", "")
            manifest["source"]["duration"] = info.get("duration", 0)
        if args.json_output:
            print(json.dumps(manifest, indent=2, default=str))
        else:
            manifest_path = write_manifest(manifest, output_dir)
            print(f"Dry run manifest: {manifest_path}")
        return

    # Full extraction
    report = extract(
        url=args.url,
        output_dir=output_dir,
        frame_budget=args.frame_budget,
        scdet_auto=args.scdet_auto,
        scdet_threshold=args.scdet_threshold,
        include_interval=args.include_interval,
        enable_ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        frame_pixel=args.frame_pixel,
    )

    manifest = report.get("manifest", {})
    report_text = report.get("report", "")

    if args.json_output:
        print(json.dumps(manifest, indent=2, default=str))
    elif report_text:
        print(report_text)

    # Print summary to stderr
    errors = manifest.get("errors", [])
    warnings = manifest.get("warnings", [])
    if errors:
        print(
            f"\n{len(errors)} error(s), {len(warnings)} warning(s). Check manifest.json.",
            file=sys.stderr,
        )
    else:
        print(
            f"\nDone. {manifest.get('frames', {}).get('total', 0)} frames, {manifest.get('audio', {}).get('segment_count', 0)} audio segments.",
            file=sys.stderr,
        )


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
