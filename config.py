"""
config.py — Central configuration for the BirdCLEF 2026 project

All configurable parameters live here. Changing a value here
updates behavior across the project without touching other code.
"""

import os

# =============================================================
# DATA PATHS
# =============================================================
# Where the files downloaded from Kaggle are located
# Change these paths according to where you placed the data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TAXONOMY_CSV = os.path.join(DATA_DIR, "taxonomy.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")
TRAIN_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "train_soundscapes")
TRAIN_SOUNDSCAPES_LABELS = os.path.join(DATA_DIR, "train_soundscapes_labels.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

# Where to save agent experiments
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
GENERATED_CODE_DIR = os.path.join(os.path.dirname(__file__), "generated_code")

# =============================================================
# AUDIO / MEL-SPECTROGRAM PARAMETERS
# =============================================================
SAMPLE_RATE = 32000       # Sampling rate in Hz (BirdCLEF standard)
DURATION = 5              # Duration of each audio segment in seconds
N_MELS = 128              # Number of mel bands (image height)
N_FFT = 2048              # FFT window size
HOP_LENGTH = 512          # Hop length between consecutive windows
FMIN = 50                 # Minimum frequency (Hz) - below = noise
FMAX = 14000              # Maximum frequency (Hz) - above = less useful

# Number of audio samples for 5 seconds: 5 * 32000 = 160000
MAX_SAMPLES = SAMPLE_RATE * DURATION

# =============================================================
# MODEL PARAMETERS
# =============================================================
N_CLASSES = 234           # Number of species to classify
INPUT_SHAPE = (N_MELS, 313, 1)  # Input shape: (height, width, channels)
# 313 comes from: ceil(160000 / 512) + 1 = 313 time frames

# =============================================================
# AGENT PARAMETERS
# =============================================================
MAX_ITERATIONS = 10       # Maximum number of experiments
CODE_TIMEOUT = 300        # Timeout for each experiment (seconds)

# =============================================================
# LLM PARAMETERS (Ollama)
# =============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "gemma2:9b"  # LLM model to use (changeable)
LLM_TEMPERATURE = 0.7    # Creativity: 0.0=deterministic, 1.0=very creative
LLM_TIMEOUT = 120         # Timeout for LLM response (seconds)
LLM_CONTEXT_SIZE = 8192  # Context window size in tokens

# =============================================================
# TRAINING PARAMETERS (for agent quick experiments)
# =============================================================
QUICK_TRAIN_SAMPLES = 2000   # Samples for quick experiments
QUICK_TRAIN_EPOCHS = 5       # Epochs for quick experiments
BATCH_SIZE = 32               # Batch size
VALIDATION_SPLIT = 0.2        # Fraction of data used for validation
