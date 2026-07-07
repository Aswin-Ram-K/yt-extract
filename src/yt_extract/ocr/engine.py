"""OCR module: text detection and extraction from video frames."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..errors import ErrorCode, ExtractionReport

logger = logging.getLogger(__name__)

try:
    import cv2
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


def check_ocr_dependencies(report: ExtractionReport | None = None) -> bool:
    """Check if OCR dependencies are available."""
    if not TESSERACT_AVAILABLE:
        if report:
            report.add_warning(
                ErrorCode.OCR_ENGINE_MISSING,
                "OCR not available: install with `pip install yt-extract[ocr]` (requires pytesseract + opencv + tesseract-ocr system package).",
                recoverable=True,
            )
        return False
    return True


def compute_text_density(image: cv2.typing.MatLike) -> float:
    """Compute text density: ratio of edge pixels to total pixels.

    Higher values indicate more text-like content.
    Returns value in [0, 1].
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    # Adaptive threshold to find text regions
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # Count non-zero pixels
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_pixels = cv2.countNonZero(thresh)
    return edge_pixels / total_pixels if total_pixels > 0 else 0


def extract_ocr(
    frame_path: Path | str,
    lang: str = "eng",
    density_threshold: float = 0.02,
) -> dict:
    """Run OCR on a single frame.

    Returns dict with: text, confidence, density, preprocessed
    """
    frame_path = Path(frame_path)
    result = {"text": "", "confidence": 0, "density": 0, "success": False}

    if not frame_path.exists():
        result["error"] = "File not found"
        return result

    try:
        image = cv2.imread(str(frame_path))
        if image is None:
            result["error"] = "Failed to read image"
            return result

        # Compute text density
        density = compute_text_density(image)
        result["density"] = round(density, 4)

        if density < density_threshold:
            result["skipped_density"] = True
            return result

        # Preprocess: threshold for readability
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Adaptive threshold - try both directions
        thresh_fwd = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        result_raw = pytesseract.image_to_string(
            thresh_fwd, lang=lang, config=f"--oem {1} --psm {3}"
        ).strip()

        info = pytesseract.image_to_data(
            thresh_fwd,
            lang=lang,
            config=f"--oem {1} --psm {3}",
            output_type=pytesseract.Output.DICT,
        )
        # Average confidence of non-empty words
        confidences = [
            int(c) for c in info["conf"] if c != -1 and info["text"][info["conf"].index(c)].strip()
        ]
        avg_conf = float(np.mean(confidences)) if confidences else 0

        result.update(
            {
                "text": result_raw,
                "confidence": round(avg_conf, 1),
                "success": True,
                "skipped_density": False,
            }
        )
    except Exception as exc:
        result["error"] = str(exc)[:200]

    return result


def _json_default(obj):
    """Handle numpy types in JSON."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")
