"""Structured error and warning system for extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


class ErrorCode(Enum):
    # YouTube / download errors
    VIDEO_NOT_FOUND = "video_not_found"
    VIDEO_PRIVATE = "video_private"
    VIDEO_DELETED = "video_deleted"
    GEO_RESTRICTED = "geo_restricted"
    LIVE_STREAM = "live_stream"
    MEMBERS_ONLY = "members_only"
    RATE_LIMITED = "rate_limited"
    FORMAT_NOT_SUPPORTED = "format_not_supported"
    DOWNLOAD_FAILED = "download_failed"
    CORRUPT_FILE = "corrupt_file"

    # Scene detection errors
    SCDET_VERSION_TOO_OLD = "scdet_version_too_old"
    SCDET_NO_CUTS = "scdet_no_cuts"
    SCDET_PARSE_FAILED = "scdet_parse_failed"

    # Frame extraction errors
    FRAME_EXTRACT_FAILED = "frame_extract_failed"
    FRAME_FPS_INVALID = "frame_fps_invalid"

    # Audio errors
    AUDIO_EXTRACT_FAILED = "audio_extract_failed"
    AUDIO_SILENT = "audio_silent"

    # Librosa errors
    LIBROSA_FEATURE_ERROR = "librosa_feature_error"
    LIBROSA_BEAT_ERROR = "librosa_beat_error"
    LIBROSA_SILENCE_ERROR = "librosa_silence_error"

    # OCR errors
    OCR_ENGINE_MISSING = "ocr_engine_missing"
    OCR_PROCESSING_FAILED = "ocr_processing_failed"
    OCR_LOW_CONFIDENCE = "ocr_low_confidence"

    # Verification errors
    VERIFY_FFPROBE_MISSING = "verify_ffprobe_missing"
    VERIFY_STREAM_MISSING = "verify_stream_missing"

    # Output errors
    MANIFEST_WRITE_FAILED = "manifest_write_failed"

    # Config errors
    INVALID_CONFIG = "invalid_config"


@dataclass
class Warning:
    code: ErrorCode
    detail: str
    recoverable: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "detail": self.detail,
            "recoverable": self.recoverable,
            **self.extra,
        }


@dataclass
class ExtractionError:
    code: ErrorCode
    detail: str
    recoverable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    traceback_str: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = {
            "code": self.code.value,
            "detail": self.detail,
            "recoverable": self.recoverable,
            **self.extra,
        }
        if self.traceback_str:
            result["traceback"] = self.traceback_str
        return result


class ExtractionReport:
    """Collects all errors and warnings across the extraction pipeline."""

    def __init__(self) -> None:
        self.errors: list[ExtractionError] = []
        self.warnings: list[Warning] = []
        self.phases: dict[str, dict[str, Any]] = {}

    def add_error(
        self, code: ErrorCode, detail: str, recoverable: bool = False, tb: str = ""
    ) -> None:
        self.errors.append(
            ExtractionError(code=code, detail=detail, recoverable=recoverable, traceback_str=tb)
        )

    def add_warning(
        self,
        code: ErrorCode,
        detail: str,
        recoverable: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.warnings.append(
            Warning(code=code, detail=detail, recoverable=recoverable, extra=extra or {})
        )

    def add_phase_report(self, name: str, report: dict[str, Any]) -> None:
        self.phases[name] = report

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_nonrecoverable_errors(self) -> bool:
        return any(not e.recoverable for e in self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "phases": self.phases,
        }
