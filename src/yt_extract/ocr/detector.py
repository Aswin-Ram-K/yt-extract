"""OCR orchestration: batch text detection and extraction from frames."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..errors import ErrorCode, ExtractionReport
from .engine import TESSERACT_AVAILABLE, extract_ocr

logger = logging.getLogger(__name__)


def batch_ocr(
    frame_paths: list[Path],
    output_dir: Path,
    report: ExtractionReport,
    lang: str = "eng",
    density_threshold: float = 0.02,
    max_workers: int = 4,
) -> list[dict]:
    """Run OCR on multiple frames in parallel.

    Returns list of OCR results in order.
    Writes ocr.jsonl to output_dir.
    """
    if not TESSERACT_AVAILABLE:
        report.add_warning(
            ErrorCode.OCR_ENGINE_MISSING,
            "OCR module not available. Install with `pip install yt-extract[ocr]`.",
            recoverable=True,
        )
        return []

    results: list[dict] = [{} for _ in frame_paths]  # pre-allocate
    text_count = 0
    skip_count = 0

    def process_frame(index: int, path: Path) -> tuple[int, dict]:
        return index, extract_ocr(path, lang=lang, density_threshold=density_threshold)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_frame, i, p): i for i, p in enumerate(frame_paths)}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            if result.get("success"):
                text_count += 1
            elif result.get("skipped_density"):
                skip_count += 1

    # Write ocr.jsonl
    ocr_path = output_dir / "ocr.jsonl"
    try:
        with open(ocr_path, "w") as f:
            import json

            for r in results:
                f.write(json.dumps(r, default=_json_default) + "\n")
        logger.info("Wrote ocr.jsonl (%d with text, %d skipped)", text_count, skip_count)
    except Exception as exc:
        report.add_error(ErrorCode.OCR_PROCESSING_FAILED, str(exc), recoverable=True)

    report.add_phase_report(
        "ocr",
        {
            "total_frames": len(frame_paths),
            "with_text": text_count,
            "skipped_low_density": skip_count,
            "language": lang,
        },
    )

    return results


def _json_default(obj):
    """Handle numpy types in JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
