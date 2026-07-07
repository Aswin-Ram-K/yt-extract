"""Scene detection via FFmpeg scdet filter with auto-threshold tuning."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from ..config import SCDET_PRINT_SCORES
from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)


def get_ffmpeg_path() -> str:
    """Find ffmpeg binary."""
    import shutil

    path = shutil.which("ffmpeg")
    if path:
        return path
    raise FileNotFoundError("ffmpeg not found. Install ffmpeg.")


def check_ffmpeg_version() -> str:
    """Check ffmpeg version. Returns version string."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        first_line = result.stdout.split("\n")[0] if result.stdout else ""
        return first_line
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise FileNotFoundError("ffmpeg not found") from exc


def run_scdet_analysis(
    video_path: str,
    threshold: float = 0.4,
    timeout: int = 300,
) -> list[dict[str, float]]:
    """Run ffmpeg scdet in analysis mode, returns list of frame scores.

    Each dict has: {score: float, mafd: float, frame_index: int}
    """
    ffmpeg = get_ffmpeg_path()
    args = [
        ffmpeg,
        "-v",
        "info",
        "-i",
        video_path,
        "-filter:v",
        f"scdet=t={threshold}:{'print=1' if SCDET_PRINT_SCORES else 'print=0'}",
        "-f",
        "null",
        "-",
    ]
    scores: list[dict[str, float]] = []
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Parse SCDET log lines from stderr
        # Format: [scdet @ ...] frame:0 score:0.000000 mafd:0.000000
        # or: [scdet @ ...] frame:1 score:0.156250 mafd:0.333333
        pattern = re.compile(r"scdet.*?frame:(\d+)\s+score:([0-9.]+)\s+mafd:([0-9.]+)")
        for line in result.stderr.split("\n"):
            match = pattern.search(line)
            if match:
                scores.append(
                    {
                        "frame_index": int(match.group(1)),
                        "score": float(match.group(2)),
                        "mafd": float(match.group(3)),
                    }
                )
    except subprocess.TimeoutExpired:
        logger.warning("scdet analysis timed out after %ds", timeout)
    except Exception as exc:
        logger.error("scdet analysis failed: %s", str(exc)[:200])

    return scores


def compute_auto_threshold(scores: list[dict[str, float]], percentile: int = 75) -> float:
    """Compute threshold from score distribution at given percentile.

    Returns a normalized threshold (0-1 range).
    Falls back to 0.4 if too few scores.
    """
    if not scores:
        return 0.4

    values = sorted([s["score"] for s in scores])
    n = len(values)

    if n < 10:
        # Not enough data, use a moderate default
        return 0.4

    # Compute percentile
    idx = int(n * percentile / 100) - 1
    idx = max(0, min(idx, n - 1))
    auto_threshold = values[idx]

    # Normalize: scdet scores can vary widely depending on ffmpeg version.
    # In some versions, scores are 0-100 (percentage), in others 0-1 (normalized).
    # If the threshold is > 1, it's likely in the 0-100 range.
    if auto_threshold > 1.0:
        auto_threshold = auto_threshold / 100.0

    # Clamp
    auto_threshold = max(0.05, min(0.95, auto_threshold))
    return auto_threshold


def detect_scene_cuts(
    video_path: str,
    auto: bool = True,
    manual_threshold: float | None = None,
    report: ExtractionReport | None = None,
) -> tuple[float, list[dict[str, float]], bool]:
    """Detect scene cuts in a video.

    Returns: (threshold_used, scores_list, fallback_needed)
    fallback_needed is True when scdet found zero cuts.
    """
    ffmpeg_path = get_ffmpeg_path()
    try:
        version_output = subprocess.run(
            [ffmpeg_path, "-version"], capture_output=True, text=True, check=True, timeout=10
        )
        version_line = version_output.stdout.split("\n")[0] if version_output.stdout else ""
    except Exception:
        version_line = "unknown"

    if report:
        report.add_phase_report("scdet_version", {"version_line": version_line})

    # Check ffmpeg version for scdet support (added in 5.x)
    version_match = re.search(r"version\s+(\d+)\.", version_line)
    if version_match:
        major = int(version_match.group(1))
        if major < 5:
            if report:
                report.add_warning(
                    ErrorCode.SCDET_VERSION_TOO_OLD,
                    f"FFmpeg {major} detected, scdet requires FFmpeg 5+. Using fallback.",
                    recoverable=True,
                )
            return 0.4, [], True  # Fallback to interval extraction

    # Run analysis
    scores = run_scdet_analysis(video_path)

    if report:
        report.add_phase_report(
            "scdet_analysis",
            {
                "total_frames_analyzed": len(scores),
                "min_score": min((s["score"] for s in scores), default=None),
                "max_score": max((s["score"] for s in scores), default=None),
                "mean_score": sum(s["score"] for s in scores) / len(scores) if scores else 0,
            },
        )

    if auto:
        threshold = compute_auto_threshold(scores)
        if report:
            report.add_phase_report(
                "scdet_auto_threshold",
                {
                    "threshold": threshold,
                    "scores_count": len(scores),
                },
            )
    else:
        threshold = manual_threshold if manual_threshold is not None else 0.4

    # Check if we got any detections
    if len(scores) == 0:
        if report:
            report.add_warning(
                ErrorCode.SCDET_NO_CUTS,
                "No scene cuts detected. Falling back to fixed-interval extraction.",
                recoverable=True,
            )
        return threshold, scores, True

    return threshold, scores, False


def run_scdet_extraction(
    video_path: str,
    output_dir: str,
    threshold: float,
    pixel_width: int = 720,
    timeout: int = 600,
) -> list[dict]:
    """Extract frames at scene boundaries.

    Returns list of dicts: {timestamp, score, index, filepath}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_path()
    # scdet detects, select filters, showsinfo logs timestamps
    args = [
        ffmpeg,
        "-v",
        "warning",
        "-i",
        video_path,
        "-filter:v",
        f"scdet=t={threshold},s=1,select='gt(scene,{threshold})',"
        f"scale={pixel_width}:-1:force_original_aspect_ratio=decrease",
        "-vsync",
        "vfr",
        "-frame_pts",
        "true",
        f"{output_path}/scene_%04d.jpg",
    ]
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error("scdet extraction failed: %s", result.stderr[:300])
            return []
    except subprocess.TimeoutExpired:
        logger.error("scdet extraction timed out after %ds", timeout)
        return []
    except Exception as exc:
        logger.error("scdet extraction error: %s", str(exc)[:200])
        return []

    # Count actual frames extracted from directory listing
    jpg_files = sorted(Path(output_dir).glob("scene_*.jpg"))
    if not jpg_files:
        return []

    timestamps: list[dict] = []
    for i, _jpg in enumerate(jpg_files):
        timestamps.append(
            {
                "index": i,
                "timestamp": float(i),
                "filepath": f"scene_{i:04d}.jpg",
            }
        )

    return timestamps


def run_interval_extraction(
    video_path: str,
    output_dir: str,
    fps: float,
    pixel_width: int = 720,
    timeout: int = 600,
) -> list[dict]:
    """Extract frames at fixed intervals.

    Returns list of dicts: {timestamp, index, filepath}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_path()
    args = [
        ffmpeg,
        "-i",
        video_path,
        "-vf",
        f"fps={fps},scale={pixel_width}:-1:force_original_aspect_ratio=decrease",
        "-vsync",
        "vfr",
        "-frame_pts",
        "true",
        f"{output_path}/interval_%04d.jpg",
    ]
    try:
        subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        logger.error("interval extraction failed: %s", str(exc)[:200])
        return []

    # Count actual frames extracted from directory listing
    jpg_files = sorted(Path(output_dir).glob("interval_*.jpg"))
    if not jpg_files:
        return []

    timestamps: list[dict] = []
    for i, _jpg in enumerate(jpg_files):
        timestamps.append(
            {
                "index": i,
                "timestamp": round(float(i) / fps, 3) if fps > 0 else 0.0,
                "filepath": f"interval_{i:04d}.jpg",
            }
        )

    return timestamps
