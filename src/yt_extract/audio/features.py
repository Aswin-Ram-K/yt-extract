"""Audio feature extraction using librosa (signal processing, no ML models)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np

from ..config import (
    AUDIO_CHUNK_DURATION,
    AUDIO_SAMPLE_RATE,
    CHROMA_N_CHROMA,
    MFCC_N_MFCC,
    SPECTRAL_CONTRAST_N_BANDS,
    SPECTRAL_ROLLOFF_PERCENTILE,
)
from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)


def extract_features(
    audio_path: Path | str,
    output_dir: Path,
    report: ExtractionReport,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    chunk_duration: float = AUDIO_CHUNK_DURATION,
) -> dict:
    """Extract full librosa feature set from audio, chunked.

    Returns summary dict with feature stats and writes:
    - features.json: full feature arrays
    - features_summary.csv: per-feature statistics
    - segments.jsonl: per-chunk features
    """
    audio_path = Path(audio_path)

    # Load audio
    try:
        y, sr = librosa.load(str(audio_path), sr=sample_rate, duration=None)
    except Exception as exc:
        logger.error("librosa.load failed: %s", str(exc)[:200])
        report.add_error(ErrorCode.LIBROSA_FEATURE_ERROR, str(exc), recoverable=False)
        return {}

    sr = sr or sample_rate
    total_duration = len(y) / sr if sr > 0 else 0

    # Process in chunks
    segments_data: list[dict] = []
    all_features: dict[str, np.ndarray] = {}
    feature_summary: dict[str, dict] = {}

    chunk_size = int(chunk_duration * sr)
    if chunk_size <= 0:
        chunk_size = len(y)

    for i in range(0, len(y), chunk_size):
        chunk_y = y[i : i + chunk_size]
        if len(chunk_y) < 100:  # skip tiny final chunk
            continue

        chunk_start = i / sr
        chunk_features = _compute_chunk_features(chunk_y, sr, chunk_start)
        segments_data.append(chunk_features)

        # Accumulate full arrays
        for name, arr in chunk_features["arrays"].items():
            if name not in all_features:
                all_features[name] = arr
            else:
                all_features[name] = np.concatenate([all_features[name], arr], axis=-1)

    # Compute beat/tempo on full audio (not chunked)
    beat_info = _compute_beat_info(y, sr)

    # Convert numpy arrays to serializable format
    serializable_arrays: dict[str, dict] = {}
    for name, arr in all_features.items():
        shape = list(arr.shape)
        data = arr.flatten().tolist() if arr.size > 0 else []
        serializable_arrays[name] = {
            "shape": shape,
            "data": data,
        }

    # Compute summary statistics
    for name, arr in all_features.items():
        flat = arr.flatten()
        flat_valid = flat[~np.isnan(flat) & ~np.isinf(flat)]
        if len(flat_valid) > 0:
            feature_summary[name] = {
                "mean": float(np.mean(flat_valid)),
                "std": float(np.std(flat_valid)),
                "min": float(np.min(flat_valid)),
                "max": float(np.max(flat_valid)),
            }
        else:
            feature_summary[name] = {
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }

    # Write features.json (full arrays)
    features_data = {
        "sample_rate": sr,
        "duration_seconds": round(total_duration, 2),
        "chunk_duration": chunk_duration,
        "features": serializable_arrays,
        "summary": feature_summary,
    }
    if beat_info:
        features_data["beat"] = beat_info
    features_path = output_dir / "features.json"
    try:
        with open(features_path, "w") as f:
            json.dump(features_data, f, indent=2, default=str)
        logger.info("Wrote features.json (%d arrays)", len(serializable_arrays))
    except Exception as exc:
        report.add_error(ErrorCode.MANIFEST_WRITE_FAILED, f"features.json: {exc}", recoverable=True)

    # Write features_summary.csv
    csv_path = output_dir / "features_summary.csv"
    try:
        with open(csv_path, "w") as f:
            f.write("feature,statistic,value\n")
            for name, stats in feature_summary.items():
                for stat, value in stats.items():
                    v = value if value is not None else ""
                    f.write(f"{name},{stat},{v}\n")
        logger.info("Wrote features_summary.csv")
    except Exception as exc:
        report.add_error(
            ErrorCode.MANIFEST_WRITE_FAILED, f"features_summary.csv: {exc}", recoverable=True
        )

    # Write segments.jsonl
    segments_path = output_dir / "segments.jsonl"
    try:
        with open(segments_path, "w") as f:
            for seg in segments_data:
                f.write(json.dumps(seg, default=_json_default) + "\n")
        logger.info("Wrote segments.jsonl (%d segments)", len(segments_data))
    except Exception as exc:
        report.add_error(
            ErrorCode.MANIFEST_WRITE_FAILED, f"segments.jsonl: {exc}", recoverable=True
        )

    report.add_phase_report(
        "feature_extraction",
        {
            "total_chunks": len(segments_data),
            "feature_types": list(serializable_arrays.keys()),
            "total_duration": round(total_duration, 2),
            "samples_count": len(y),
        },
    )

    return {
        "duration": total_duration,
        "feature_count": len(serializable_arrays),
        "segment_count": len(segments_data),
    }


def _compute_chunk_features(y: np.ndarray, sr: int, chunk_start: float) -> dict:
    """Compute all feature types for one chunk."""
    result: dict[str, np.ndarray] = {}
    try:
        result["mfcc"] = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N_MFCC)
    except Exception:
        result["mfcc"] = np.zeros((MFCC_N_MFCC, 1))
    try:
        result["chroma"] = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=2048, n_bins=CHROMA_N_CHROMA
        )
    except Exception:
        result["chroma"] = np.zeros((CHROMA_N_CHROMA, 1))
    try:
        result["spectral_centroid"] = librosa.feature.spectral_centroid(y=y, sr=sr)
    except Exception:
        result["spectral_centroid"] = np.zeros((1, 1))
    try:
        result["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    except Exception:
        result["spectral_bandwidth"] = np.zeros((1, 1))
    try:
        result["spectral_rolloff"] = librosa.feature.spectral_rolloff(
            y=y, sr=sr, roll_percentile=SPECTRAL_ROLLOFF_PERCENTILE
        )
    except Exception:
        result["spectral_rolloff"] = np.zeros((1, 1))
    try:
        result["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(y=y)
    except Exception:
        result["zero_crossing_rate"] = np.zeros((1, 1))
    try:
        result["rms_energy"] = librosa.feature.rms(y=y)
    except Exception:
        result["rms_energy"] = np.zeros((1, 1))
    try:
        result["spectral_contrast"] = librosa.feature.spectral_contrast(
            y=y, sr=sr, n_bands=SPECTRAL_CONTRAST_N_BANDS
        )
    except Exception:
        result["spectral_contrast"] = np.zeros((SPECTRAL_CONTRAST_N_BANDS, 1))
    try:
        result["spectral_flatness"] = librosa.feature.spectral_flatness(y=y)
    except Exception:
        result["spectral_flatness"] = np.zeros((1, 1))

    # Sanitize: replace NaN/Inf with 0
    for k, v in result.items():
        result[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "start_s": round(chunk_start, 2),
        "end_s": round(chunk_start + len(y) / sr, 2) if sr > 0 else chunk_start,
        "arrays": result,
        "stats": {
            k: {
                "mean": float(np.nanmean(v.flatten())),
                "std": float(np.nanstd(v.flatten())),
            }
            for k, v in result.items()
        },
    }


def _compute_beat_info(y: np.ndarray, sr: int) -> dict | None:
    """Compute tempo and beat positions on full audio."""
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        return {
            "tempo_bpm": float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo),
            "beat_positions_seconds": [round(float(t), 3) for t in beat_times],
            "beat_count": len(beat_times),
        }
    except Exception as exc:
        logger.warning("Beat tracking failed: %s", str(exc)[:100])
        return None


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
