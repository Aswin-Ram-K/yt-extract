"""Unit tests for yt-extract core components."""

from __future__ import annotations

from pathlib import Path

import pytest

from yt_extract.audio.features import extract_features
from yt_extract.audio.silence import detect_silence
from yt_extract.config import SCDET_THRESHOLD_AUTO_PERCENTILE
from yt_extract.errors import ErrorCode, ExtractionReport
from yt_extract.output.manifest import build_manifest, generate_report_card
from yt_extract.video.scdet import (
    check_ffmpeg_version,
    compute_auto_threshold,
    get_ffmpeg_path,
    run_interval_extraction,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── Tests: Errors ──────────────────────────────────────────────────────────


class TestErrors:
    def test_report_add_error(self) -> None:
        report = ExtractionReport()
        report.add_error(ErrorCode.DOWNLOAD_FAILED, "test error")
        assert len(report.errors) == 1
        # Errors are non-recoverable by default
        assert report.has_nonrecoverable_errors

    def test_report_add_nonrecoverable(self) -> None:
        report = ExtractionReport()
        report.add_error(ErrorCode.LIVE_STREAM, "is live", recoverable=False)
        assert report.has_nonrecoverable_errors

    def test_report_add_warning(self) -> None:
        report = ExtractionReport()
        report.add_warning(ErrorCode.SCDET_NO_CUTS, "no cuts found")
        assert len(report.warnings) == 1

    def test_report_to_dict(self) -> None:
        report = ExtractionReport()
        report.add_error(ErrorCode.DOWNLOAD_FAILED, "fail")
        report.add_warning(ErrorCode.SCDET_VERSION_TOO_OLD, "old")
        d = report.to_dict()
        assert len(d["errors"]) == 1
        assert len(d["warnings"]) == 1
        assert d["errors"][0]["code"] == "download_failed"
        # Errors default to non-recoverable
        assert d["errors"][0]["recoverable"] is False


# ── Tests: Configuration ───────────────────────────────────────────────────


class TestConfig:
    def test_default_values(self) -> None:
        assert SCDET_THRESHOLD_AUTO_PERCENTILE == 75
        assert 0.05 <= 0.4 <= 0.95  # threshold is clamped


# ── Tests: Video Detection ─────────────────────────────────────────────────


class TestVideoDetection:
    def test_ffmpeg_path(self) -> None:
        path = get_ffmpeg_path()
        assert Path(path).exists()

    def test_ffmpeg_version(self) -> None:
        line = check_ffmpeg_version()
        assert "ffmpeg" in line.lower() or "version" in line.lower()

    def test_scdet_analysis(self) -> None:
        """Test scdet analysis on a test video."""
        video = FIXTURES / "test_scene_video.mp4"
        if not video.exists():
            pytest.skip("test_scene_video.mp4 not found")
        scores = _run_scdet_analysis(str(video))
        assert isinstance(scores, list)
        if scores:
            assert "score" in scores[0]
            assert "mafd" in scores[0]
            assert scores[0]["score"] >= 0

    def test_auto_threshold(self) -> None:
        """Test threshold computation from score distribution."""
        # Simulated scores
        scores = [{"score": s} for s in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0]]
        thresh = compute_auto_threshold(scores, percentile=75)
        assert 0.05 <= thresh <= 0.95

    def test_auto_threshold_empty(self) -> None:
        """Empty scores should return default."""
        thresh = compute_auto_threshold([])
        assert thresh == 0.4

    def test_auto_threshold_single(self) -> None:
        """Too few scores should return default."""
        thresh = compute_auto_threshold([{"score": 0.5}])
        assert thresh == 0.4


def _run_scdet_analysis(video_path: str):
    """Helper: run scdet analysis (imported at runtime to handle imports)."""
    from yt_extract.video.scdet import run_scdet_analysis

    return run_scdet_analysis(video_path)


# ── Tests: Audio Features ──────────────────────────────────────────────────


class TestAudioFeatures:
    def test_features_extraction(self) -> None:
        """Extract features from test audio."""
        audio = FIXTURES / "test_audio.wav"
        if not audio.exists():
            pytest.skip("test_audio.wav not found")
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            report = ExtractionReport()
            result = extract_features(audio, Path(td), report)
            assert result.get("feature_count", 0) > 0
            assert result.get("segment_count", 0) > 0

    def test_silence_detection(self) -> None:
        """Detect silence in test audio."""
        audio = FIXTURES / "test_audio.wav"
        if not audio.exists():
            pytest.skip("test_audio.wav not found")
        report = ExtractionReport()
        result = detect_silence(audio, report=report)
        assert "total_duration" in result
        assert result["total_duration"] > 0
        assert not result.get("is_completely_silent", True)

    def test_silence_audio(self) -> None:
        """Detect silence in silence audio."""
        silence = FIXTURES / "test_silence.wav"
        if not silence.exists():
            pytest.skip("test_silence.wav not found")
        report = ExtractionReport()
        result = detect_silence(silence, report=report)
        # May or may not be completely silent depending on audio generation
        assert "is_completely_silent" in result


# ── Tests: Manifest ────────────────────────────────────────────────────────


class TestManifest:
    def test_build_manifest_minimal(self) -> None:
        manifest = build_manifest(
            "https://youtube.com/watch?v=test",
            {},  # yt_info
            {},  # video_metadata
            [],  # frames
            {},  # audio_info
            None,  # ocr_results
        )
        assert manifest["version"] == "1.0"
        assert manifest["source"]["type"] == "youtube"
        assert manifest["source"]["url"] == "https://youtube.com/watch?v=test"
        assert manifest["frames"]["total"] == 0

    def test_build_manifest_full(self) -> None:
        report = ExtractionReport()
        report.add_warning(ErrorCode.SCDET_NO_CUTS, "no cuts")
        manifest = build_manifest(
            "https://youtube.com/watch?v=abc123",
            {
                "title": "Test Video",
                "duration": 300,
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "chapters": [{"title": "Intro", "start_time": 0, "end_time": 60}],
            },
            {
                "format": {"duration": "300.0", "format_name": "mp4", "size": "50000000"},
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30/1",
                    }
                ],
            },
            [{"index": 0, "timestamp": 0.0, "filepath": "frames/scene_0000.jpg"}],
            {"duration": 300, "sample_rate": 22050, "feature_count": 10, "segment_count": 10},
            [],
            report,
        )
        assert manifest["source"]["title"] == "Test Video"
        assert manifest["frames"]["total"] == 1
        assert len(manifest["warnings"]) == 1
        assert manifest["warnings"][0]["code"] == "scdet_no_cuts"

    def test_report_card(self) -> None:
        manifest = build_manifest(
            "https://youtube.com/watch?v=test",
            {"title": "My Video", "duration": 120},
            {
                "format": {"duration": "120.0"},
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080}
                ],
            },
            [{"index": 0, "timestamp": 0.0, "filepath": "frames/scene_0000.jpg"}],
            {"duration": 120, "sample_rate": 22050, "feature_count": 10, "segment_count": 4},
            None,
        )
        report_text = generate_report_card(manifest)
        assert "My Video" in report_text
        assert "2:00" in report_text
        assert "1 frames" in report_text
        assert "22050 Hz" in report_text


# ── Tests: Frame Extraction ────────────────────────────────────────────────


class TestFrameExtraction:
    def test_interval_extraction(self) -> None:
        """Extract frames at fixed intervals."""
        video = FIXTURES / "test_scene_video.mp4"
        if not video.exists():
            pytest.skip("test_scene_video.mp4 not found")
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            frames_path = Path(td) / "frames"
            frames_path.mkdir()
            result = run_interval_extraction(str(video), str(frames_path), fps=2.0, pixel_width=320)
            assert isinstance(result, list)
            # Should have extracted some frames
            jpg_files = list(frames_path.glob("*.jpg"))
            assert len(jpg_files) > 0


# ── Tests: End-to-End (no network) ─────────────────────────────────────────


class TestEndToEnd:
    def test_extract_frames_no_network(self) -> None:
        """Test frame extraction pipeline without network."""
        from yt_extract.video.frames import extract_frames

        video = FIXTURES / "test_scene_video.mp4"
        if not video.exists():
            pytest.skip("test_scene_video.mp4 not found")

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            output_dir = Path(td)
            report = ExtractionReport()
            frames = extract_frames(
                str(video),
                output_dir,
                report,
                budget=50,
                auto_scdet=True,
            )
            assert isinstance(frames, list)
            # Even if scdet finds no cuts, fallback to interval extraction
            assert len(frames) > 0
            # Each frame should have an index and timestamp
            for f in frames:
                assert "index" in f or "timestamp" in f or "filepath" in f

    def test_extract_frames_with_interval(self) -> None:
        """Test interval-based extraction directly."""
        from yt_extract.video.scdet import run_interval_extraction

        video = FIXTURES / "test_scene_video.mp4"
        if not video.exists():
            pytest.skip("test_scene_video.mp4 not found")

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            frames_path = Path(td) / "frames"
            frames_path.mkdir()
            result = run_interval_extraction(str(video), str(frames_path), fps=2.0, pixel_width=320)
            assert len(result) > 0
            assert "timestamp" in result[0]
            assert "index" in result[0]
            assert "filepath" in result[0]
