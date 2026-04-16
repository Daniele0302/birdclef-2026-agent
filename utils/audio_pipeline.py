"""
utils/audio_pipeline.py — Pipeline audio per BirdCLEF 2026

Questo modulo gestisce tutto il preprocessing audio:
1. Carica un file .ogg
2. Lo taglia/padda a 5 secondi
3. Lo converte in mel-spectrogram
4. Lo normalizza per la rete neurale

Uso:
    from utils.audio_pipeline import load_and_process_audio
    mel = load_and_process_audio("path/to/file.ogg")
    # mel.shape = (128, 313)  — pronto per la CNN
"""

import numpy as np
import librosa

# Importiamo i parametri dal file di configurazione
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, DURATION, N_MELS, N_FFT, 
    HOP_LENGTH, FMIN, FMAX, MAX_SAMPLES
)


def load_audio(filepath, sr=SAMPLE_RATE):
    """
    Carica un file audio e lo converte al sample rate desiderato.
    
    Args:
        filepath: percorso del file audio (.ogg, .mp3, .wav)
        sr: sample rate desiderato (default: 32000 Hz)
    
    Returns:
        y: array numpy con i campioni audio
        sr: il sample rate
    
    Spiegazione:
        librosa.load() fa due cose:
        1. Legge il file audio (qualunque formato)
        2. Converte automaticamente al sample rate che chiediamo
        Se il file è stereo (2 canali), lo converte in mono (1 canale)
    """
    try:
        # librosa.load restituisce una tupla: (campioni, sample_rate)
        # y è un array numpy 1D, es: [0.01, -0.03, 0.05, ...]
        y, sr = librosa.load(filepath, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Errore nel caricamento di {filepath}: {e}")
        return None, sr


def pad_or_trim(y, max_samples=MAX_SAMPLES):
    """
    Assicura che l'audio abbia esattamente la lunghezza desiderata.
    
    Se è più lungo di 5s: tagliamo (prendiamo i primi 5s)
    Se è più corto di 5s: aggiungiamo zeri alla fine (silenzio)
    
    Args:
        y: array numpy con i campioni audio
        max_samples: numero di campioni desiderato (default: 160000)
    
    Returns:
        y: array numpy con esattamente max_samples elementi
    
    Spiegazione:
        Perché servono tutti della stessa lunghezza?
        Perché la CNN si aspetta input di dimensione fissa.
        Se un audio dura 3s e un altro 10s, non possono entrare
        nello stesso batch. Uniformando a 5s, tutti hanno la stessa dimensione.
    """
    if len(y) > max_samples:
        # L'audio è più lungo di 5 secondi
        # Prendiamo solo i primi max_samples campioni
        y = y[:max_samples]
    elif len(y) < max_samples:
        # L'audio è più corto di 5 secondi
        # np.pad aggiunge zeri alla fine
        # (0, max_samples - len(y)) significa: 0 zeri all'inizio, N zeri alla fine
        y = np.pad(y, (0, max_samples - len(y)), mode='constant', constant_values=0)
    
    return y


def audio_to_melspec(y, sr=SAMPLE_RATE):
    """
    Converte un segnale audio in mel-spectrogram normalizzato.
    
    Questa è la funzione chiave: trasforma i numeri dell'onda sonora
    in un'immagine 2D che la CNN può elaborare.
    
    Args:
        y: array numpy con i campioni audio (es: 160000 numeri)
        sr: sample rate (32000)
    
    Returns:
        mel_norm: matrice numpy (128, 313), valori tra 0 e 1
    
    I passaggi interni:
        1. melspectrogram: divide l'audio in finestre, applica FFT,
           poi proietta sulle bande mel
        2. power_to_db: converte in scala logaritmica (decibel)
           perché l'orecchio percepisce il volume in modo logaritmico
        3. Normalizzazione min-max: porta i valori tra 0 e 1
           (la CNN funziona meglio con valori piccoli e uniformi)
    """
    # --- Step 1: Creare il mel-spectrogram ---
    # n_fft=2048: ogni finestra analizza 2048 campioni (64ms a 32kHz)
    # hop_length=512: le finestre si spostano di 512 campioni (16ms)
    #   → overlap del 75% (molta sovrapposizione = buona risoluzione temporale)
    # n_mels=128: 128 bande di frequenza nella scala mel
    # fmin=50, fmax=14000: range di frequenze utili
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )
    # mel_spec.shape = (128, 313)
    # Valori: potenza (numeri positivi molto variabili)
    
    # --- Step 2: Convertire in decibel ---
    # ref=np.max normalizza rispetto al valore massimo
    # I valori vanno da 0 (massimo) a circa -80 (silenzio)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # --- Step 3: Normalizzare tra 0 e 1 ---
    # Formula min-max: (x - min) / (max - min)
    # Dopo questa operazione, 0 = silenzio, 1 = massima intensità
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()
    
    # Evitiamo divisione per zero nel caso (raro) di audio completamente silenzioso
    if mel_max - mel_min == 0:
        mel_norm = np.zeros_like(mel_spec_db)
    else:
        mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)
    
    return mel_norm


def load_and_process_audio(filepath):
    """
    Funzione principale: da file .ogg a mel-spectrogram normalizzato.
    Combina tutti i passaggi in una sola chiamata.
    
    Args:
        filepath: percorso del file .ogg
    
    Returns:
        mel_norm: array numpy (128, 313) con valori tra 0 e 1
                  oppure None se il caricamento fallisce
    
    Uso:
        mel = load_and_process_audio("data/train_audio/banana/XC12345.ogg")
        if mel is not None:
            print(mel.shape)  # (128, 313)
    """
    # Step 1: Carica l'audio
    y, sr = load_audio(filepath)
    if y is None:
        return None
    
    # Step 2: Taglia o padda a 5 secondi esatti
    y = pad_or_trim(y)
    
    # Step 3: Converti in mel-spectrogram normalizzato
    mel = audio_to_melspec(y, sr)
    
    return mel


def process_batch(filepaths, max_workers=4):
    """
    Processa una lista di file audio in parallelo (più veloce).
    
    Args:
        filepaths: lista di percorsi file
        max_workers: numero di processi paralleli
    
    Returns:
        mels: lista di mel-spectrogrammi (esclude i None)
        valid_indices: indici dei file processati con successo
    
    Spiegazione:
        Processare migliaia di file audio uno alla volta è lento.
        Usando il parallelismo, possiamo processarne 4 alla volta
        (o più, se hai più core nella CPU).
    """
    from concurrent.futures import ThreadPoolExecutor
    
    mels = []
    valid_indices = []
    
    # ThreadPoolExecutor crea un "pool" di thread che lavorano in parallelo
    # max_workers=4 significa che 4 file vengono processati contemporaneamente
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applica load_and_process_audio a ogni filepath
        # restituisce i risultati nello stesso ordine
        results = list(executor.map(load_and_process_audio, filepaths))
    
    # Filtriamo i risultati: teniamo solo quelli che non sono None
    for i, mel in enumerate(results):
        if mel is not None:
            mels.append(mel)
            valid_indices.append(i)
    
    return mels, valid_indices
