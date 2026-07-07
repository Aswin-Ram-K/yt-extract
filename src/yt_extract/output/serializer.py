"""Feature serialization helpers."""

from __future__ import annotations

import json
import numpy as np


def numpy_to_json(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def ndarray_to_serializable(arr):
    """Convert numpy array to JSON-serializable dict."""
    if not isinstance(arr, np.ndarray):
        return arr

    data = arr.flatten().tolist()
    shape = list(arr.shape)
    return {
        "shape": shape,
        "data": data,
        "dtype": str(arr.dtype),
    }


def arrays_to_serializable(feature_dict: dict) -> dict:
    """Convert all numpy arrays in a dict to serializable format."""
    return {
        k: ndarray_to_serializable(v) if isinstance(v, np.ndarray) else v
        for k, v in feature_dict.items()
    }


def safe_json_dumps(obj, indent: int = 2) -> str:
    """Safely serialize to JSON, replacing NaN/Inf with null."""

    def convert(o):
        if isinstance(o, float):
            if np.isnan(o) or np.isinf(o):
                return None
            return o
        if isinstance(o, (np.floating,)):
            val = float(o)
            if np.isnan(val) or np.isinf(val):
                return None
            return val
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(f"Not JSON serializable: {type(o)}")

    return json.dumps(obj, indent=indent, default=convert, ensure_ascii=False)
