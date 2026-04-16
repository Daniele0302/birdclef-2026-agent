"""
config.py — Configurazione centrale del progetto BirdCLEF 2026

Tutti i parametri configurabili sono qui. Cambiando un valore qui,
si aggiorna in tutto il progetto senza toccare altro codice.
"""

import os

# =============================================================
# PERCORSI DEI DATI
# =============================================================
# Dove si trovano i file scaricati da Kaggle
# Cambia questi percorsi in base a dove hai messo i dati
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TAXONOMY_CSV = os.path.join(DATA_DIR, "taxonomy.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")
TRAIN_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "train_soundscapes")
TRAIN_SOUNDSCAPES_LABELS = os.path.join(DATA_DIR, "train_soundscapes_labels.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

# Dove salvare gli esperimenti dell'agente
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
GENERATED_CODE_DIR = os.path.join(os.path.dirname(__file__), "generated_code")

# =============================================================
# PARAMETRI AUDIO / MEL-SPECTROGRAM
# =============================================================
SAMPLE_RATE = 32000       # Frequenza di campionamento in Hz (standard BirdCLEF)
DURATION = 5              # Durata di ogni segmento audio in secondi
N_MELS = 128              # Numero di bande mel (altezza dell'immagine)
N_FFT = 2048              # Dimensione della finestra FFT
HOP_LENGTH = 512          # Spostamento tra finestre consecutive
FMIN = 50                 # Frequenza minima (Hz) - sotto = rumore
FMAX = 14000              # Frequenza massima (Hz) - sopra = poco utile

# Numero di campioni audio per 5 secondi: 5 * 32000 = 160000
MAX_SAMPLES = SAMPLE_RATE * DURATION

# =============================================================
# PARAMETRI DEL MODELLO
# =============================================================
N_CLASSES = 234           # Numero di specie da classificare
INPUT_SHAPE = (N_MELS, 313, 1)  # Forma dell'input: (altezza, larghezza, canali)
# 313 viene da: ceil(160000 / 512) + 1 = 313 frame temporali

# =============================================================
# PARAMETRI DELL'AGENTE
# =============================================================
MAX_ITERATIONS = 10       # Numero massimo di esperimenti
CODE_TIMEOUT = 300        # Timeout per ogni esperimento (secondi)

# =============================================================
# PARAMETRI LLM (Ollama)
# =============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "gemma2:9b"  # Modello LLM da usare (cambiabile)
LLM_TEMPERATURE = 0.7    # Creatività: 0.0=deterministico, 1.0=molto creativo
LLM_TIMEOUT = 120         # Timeout per la risposta dell'LLM (secondi)
LLM_CONTEXT_SIZE = 8192  # Dimensione del contesto in token

# =============================================================
# PARAMETRI DI TRAINING (per esperimenti rapidi dell'agente)
# =============================================================
QUICK_TRAIN_SAMPLES = 2000   # Campioni per esperimenti rapidi
QUICK_TRAIN_EPOCHS = 5       # Epoche per esperimenti rapidi
BATCH_SIZE = 32               # Dimensione del batch
VALIDATION_SPLIT = 0.2        # Percentuale dati per la validazione
