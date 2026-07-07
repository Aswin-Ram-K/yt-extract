"""Default configuration and tunable thresholds."""

# FFmpeg / scdet settings
SCDET_THRESHOLD_AUTO_PERCENTILE = 75  # auto-tune: use this percentile of scores as threshold
SCDET_THRESHOLD_DEFAULT = 0.4  # fallback manual threshold (normalized)
SCDET_PRINT_SCORES = True  # print=1 to get SCDET log lines

# Frame extraction
FRAME_PIXEL_WIDTH = 720  # downscale to this width
FRAME_FORMAT = "mjpeg"  # JPEG encoding
FRAME_QUALITY = 85  # JPEG quality 1-100
FRAME_BUDGET_DEFAULT = 200  # default max frames when using budget mode
INTERVAL_MIN_FPS = 1  # minimum 1 frame per second when budgeted

# Audio
AUDIO_SAMPLE_RATE = 22050  # librosa default
AUDIO_CHUNK_DURATION = 30  # seconds per chunk for feature extraction
AUDIO_TOP_DB = 30  # silence detection threshold in dB

# Audio features
MFCC_N_MFCC = 13
CHROMA_N_CHROMA = 12
SPECTRAL_ROLLOFF_PERCENTILE = 0.85
SPECTRAL_CONTRAST_N_BANDS = 7

# OCR
OCR_TEXT_DENSITY_THRESHOLD = 0.02  # edge pixels / total pixels — skip below this
OCR_LANG_DEFAULT = "eng"
OCR_OEM = 1  # LSTM engine
OCR_PSM = 3  # auto block
OCR_MAX_WIDTH = 800  # upscale limit for OCR speed

# Output
MANIFEST_VERSION = "1.0"

# Retry
DOWNLOAD_RETRY_COUNT = 3
DOWNLOAD_RETRY_BASE_DELAY = 1  # seconds, exponential backoff
