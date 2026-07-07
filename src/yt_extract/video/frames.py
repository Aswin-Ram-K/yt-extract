"""Frame extraction orchestration: combines scene cuts and interval-based extraction."""

from __future__ import annotations

import logging
from pathlib import Path

from .scdet import run_interval_extraction, run_scdet_extraction
from ..config import FRAME_BUDGET_DEFAULT, FRAME_PIXEL_WIDTH, INTERVAL_MIN_FPS
from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    output_dir: Path,
    report: ExtractionReport,
    budget: int | None = None,
    auto_scdet: bool = True,
    scdet_threshold: float | None = None,
    include_interval: bool = True,
) -> list[dict]:
    """Extract frames from video using scene cuts and/or interval sampling.

    Returns a list of frame dicts with metadata for the manifest.
    """
    frames: list[dict] = []
    budget = budget or FRAME_BUDGET_DEFAULT

    # Get video duration from ffprobe for budget calculation
    import subprocess

    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0
    except Exception:
        duration = 0
        report.add_warning(
            ErrorCode.FRAME_FPS_INVALID,
            "Could not determine video duration, using default FPS.",
            recoverable=True,
        )

    # Phase 1: Scene-cut extraction (primary)
    scene_cutoffs, scores, fallback_needed = _detect_scene_cuts(
        video_path, auto_scdet, scdet_threshold, report, duration
    )

    if fallback_needed:
        # Fallback: use interval extraction only
        fps = max(INTERVAL_MIN_FPS, min(budget / max(duration, 1), 30))
        logger.info("scdet fallback: using fps=%.2f for %d budget", fps, budget)
        frames.extend(
            run_interval_extraction(video_path, str(output_dir / "frames"), fps, FRAME_PIXEL_WIDTH)
        )
    elif len(scene_cutoffs) > 0:
        # Scene cuts found, extract those frames
        frames.extend(scene_cutoffs)
        logger.info(
            "Extracted %d scene-cut frames at threshold=%.3f",
            len(scene_cutoffs),
            scene_cutoffs[0].get("threshold", 0.4),
        )
        # Store threshold for later
        if not any("threshold" in f for f in frames):
            frames[0]["threshold"] = (
                scene_cutoffs[0].get("threshold", 0.4) if scene_cutoffs else 0.4
            )
    else:
        # No scene cuts, use budget-based interval
        fps = max(INTERVAL_MIN_FPS, min(budget / max(duration, 1), 30))
        logger.info("No scene cuts found, using fps=%.2f for budget=%d", fps, budget)
        frames.extend(
            run_interval_extraction(video_path, str(output_dir / "frames"), fps, FRAME_PIXEL_WIDTH)
        )

    # Phase 2: Interval sampling for additional frames (optional)
    if include_interval and len(frames) < budget:
        remaining_budget = budget - len(frames)
        fps = max(INTERVAL_MIN_FPS, min(remaining_budget / max(duration, 1), 30))
        interval_frames = run_interval_extraction(
            video_path, str(output_dir / "frames"), fps, FRAME_PIXEL_WIDTH
        )
        # Filter out already-extracted timestamps (within 0.5s tolerance)
        existing_ts = {f["timestamp"] for f in frames}
        new_frames = [
            f
            for f in interval_frames
            if not any(abs(f["timestamp"] - et) < 0.5 for et in existing_ts)
        ]
        # Cap at remaining budget
        new_frames = new_frames[:remaining_budget]
        frames.extend(new_frames)
        logger.info("Added %d interval frames, total: %d", len(new_frames), len(frames))

    # Cap total frames at budget
    if len(frames) > budget:
        frames = frames[:budget]

    # Update report
    report.add_phase_report(
        "frame_extraction",
        {
            "total_frames": len(frames),
            "budget": budget,
            "duration_seconds": duration,
            "scene_cutoff_count": len(scene_cutoffs),
            "fallback_used": fallback_needed,
        },
    )

    return frames


def _detect_scene_cuts(
    video_path: str,
    auto_scdet: bool,
    scdet_threshold: float | None,
    report: ExtractionReport,
    duration: float,
) -> tuple[list[dict], bool, bool]:
    """Run scene detection and extract frames."""
    from .scdet import detect_scene_cuts

    threshold, scores, fallback_needed = detect_scene_cuts(
        video_path, auto=auto_scdet, manual_threshold=scdet_threshold, report=report
    )

    if fallback_needed:
        return [], True, True

    # Extract actual frames at detected boundaries
    cutoffs = run_scdet_extraction(
        video_path,
        str(Path(video_path).parent / "frames"),
        threshold,
        FRAME_PIXEL_WIDTH,
        timeout=600,
    )

    # Annotate with threshold
    for c in cutoffs:
        c["threshold"] = threshold
        c["score"] = next((s["score"] for s in scores if s["frame_index"] == c["index"]), 0)

    return cutoffs, False, fallback_needed
