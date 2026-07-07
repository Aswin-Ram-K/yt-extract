# yt-extract

**Fast, non-LLM YouTube video extraction for AI agents.**

Extract scene-cut frames, audio features, and OCR from YouTube videos — entirely deterministic, zero AI inference overhead.

## Quick Start

```bash
pip install yt-extract[ocr]

# Dry run (metadata only, no download)
yt-extract "https://youtube.com/watch?v=VIDEO_ID" --dry-run

# Full extraction (frames + audio features)
yt-extract "https://youtube.com/watch?v=VIDEO_ID" --frame-budget 200

# With OCR
yt-extract "https://youtube.com/watch?v=VIDEO_ID" --ocr --ocr-lang eng

# JSON output (for agents)
yt-extract "https://youtube.com/watch?v=VIDEO_ID" --json-output --compact
```

## How It Works

```
YouTube URL
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 1: yt-dlp download            │
│  • Video stream (best quality)       │
│  • Audio stream (m4a → mp3)          │
│  • Metadata (title, chapters, etc.)  │
│  • Thumbnail                          │
└──────────────────────────────────────┘
    │
    ├──▶  Phase 2: FFmpeg scdet detection
    │     • Auto-tune threshold from score distribution
    │     • Fallback to fps sampling if no cuts
    │
    ├──▶  Phase 3: Frame extraction
    │     • Scene-cut frames (one per cut)
    │     • Interval frames (budget-based)
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 4: Audio features (librosa)   │
│  • MFCCs (13 coefficients)           │
│  • Chroma (12 bins)                  │
│  • Spectral centroid, bandwidth,     │
│    rolloff, contrast, flatness       │
│  • Zero-crossing rate, RMS energy    │
│  • Tempo, beat positions             │
└──────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  Phase 5: Output                     │
│  • manifest.json (master index)      │
│  • frames/*.jpg (extracted frames)   │
│  • audio/features.json (full arrays) │
│  • audio/features_summary.csv        │
│  • audio/segments.jsonl (temporal)   │
│  • report.txt (human summary)        │
└──────────────────────────────────────┘
```

## CLI Reference

```
yt-extract URL [OPTIONS]

Options:
  --output-dir, -o DIR       Output directory (default: yt-extract-<video_id>/)
  --frame-budget, -n N       Max frames to extract (default: 200)
  --scdet-threshold, -t F    Scene detection threshold (default: auto-tune)
  --no-scdet                 Disable auto-threshold, use manual value
  --no-interval              Skip interval-based frame sampling
  --ocr                      Enable OCR on scene-cut frames
  --ocr-lang LANG            OCR language (default: eng)
  --compact                  Compact output: omit full feature arrays
  --dry-run                  Generate manifest only, skip extraction
  --frame-pixel N            Frame width in pixels (default: 720)
  --json-output              Output manifest to stdout as JSON
  --log-level, -v LEVEL      Logging level (default: INFO)
```

## Output Structure

```
yt-extract-<video_id>/
├── manifest.json           # Master index (read this first)
├── report.txt              # Human-readable summary
├── frames/
│   ├── scene_0000.jpg      # Scene-cut frames
│   ├── scene_0001.jpg
│   └── ...
├── audio/
│   ├── features.json       # Full feature arrays (omit with --compact)
│   ├── features_summary.csv
│   └── segments.jsonl      # Per-chunk features
└── text/
    └── ocr.jsonl           # OCR results (if --ocr)
```

## Manifest Schema

The `manifest.json` is the agent's single source of truth:

```json
{
  "version": "1.0",
  "source": {
    "type": "youtube",
    "url": "https://youtube.com/watch?v=...",
    "title": "Video Title",
    "channel": "Channel Name",
    "duration": 300,
    "resolution": "1920x1080",
    "fps": 30,
    "chapters": [
      {"title": "Intro", "start_s": 0, "end_s": 45}
    ]
  },
  "frames": {
    "total": 47,
    "items": [
      {"index": 0, "timestamp": 0.0, "score": 0.82, "filepath": "frames/scene_0000.jpg"}
    ]
  },
  "audio": {
    "duration": 300,
    "sample_rate": 22050,
    "non_silent_duration": 285,
    "feature_count": 10,
    "segment_count": 10
  },
  "errors": [],
  "warnings": [],
  "phases": {
    "scdet_auto_threshold": {"threshold": 0.35, "scores_count": 4500},
    "feature_extraction": {"total_chunks": 10, "feature_types": ["mfcc", ...]},
    "timing": {"total_seconds": 45.2}
  }
}
```

## Agent Integration

For agent pipelines, use `--json-output` to get the manifest as stdout:

```bash
manifest=$(yt-extract "URL" --json-output --compact)
echo "$manifest" | jq '.frames.total'
echo "$manifest" | jq '.audio.duration'
```

Read frame files by path from `manifest.frames.items[].filepath`.

## Installation

### Requirements

- **ffmpeg** ≥ 5.0 (for scene detection and frame extraction)
- **tesseract-ocr** (optional, for OCR: `apt install tesseract-ocr`)
- **Python** ≥ 3.10

### Install

```bash
# Core only (download + frames + audio)
pip install yt-extract

# With OCR support
pip install yt-extract[ocr]

# Development
pip install yt-extract[dev]
pip install yt-extract[dev,ocr]
```

## Limitations

- **No LLM needed** — all processing is deterministic signal analysis
- **Scene cuts only capture hard cuts** — fades/dissolves may be missed by scdet
- **OCR accuracy** depends on frame quality (resolution, contrast, compression)
- **Audio features** assume mono 22050 Hz input (standardized for librosa)

## Architecture

| Module | Role | Dependencies |
|--------|------|-------------|
| `video/downloader.py` | yt-dlp wrapper with error classification | yt-dlp |
| `video/scdet.py` | Scene detection + frame extraction | ffmpeg (subprocess) |
| `video/frames.py` | Orchestration: scene cuts + intervals | scdet, config |
| `video/verify.py` | Post-download ffprobe verification | ffprobe |
| `audio/extractor.py` | Audio extraction + normalization | pydub |
| `audio/features.py` | Full librosa feature set, chunked | librosa, numpy |
| `audio/silence.py` | Silence detection & segment splitting | librosa, numpy |
| `ocr/engine.py` | Tesseract OCR with adaptive preprocessing | pytesseract, opencv |
| `ocr/detector.py` | Text-density screening + batch OCR | pytesseract, opencv |
| `output/manifest.py` | Manifest builder + report card | json |
| `output/serializer.py` | Numpy→JSON serialization | numpy |
