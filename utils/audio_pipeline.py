"""
utils/audio_pipeline.py — Audio pipeline for BirdCLEF 2026

This module handles the full audio preprocessing:
1. Load an .ogg file
2. Trim or pad it to 5 seconds
3. Convert it to a mel-spectrogram
4. Normalise it for the neural network

Usage:
    from utils.audio_pipeline import load_and_process_audio
    mel = load_and_process_audio("path/to/file.ogg")
    # mel.shape = (128, 313)  — ready for the CNN
"""

import numpy as np
import librosa

# Pull the audio parameters from the central config module.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    SAMPLE_RATE, DURATION, N_MELS, N_FFT,
    HOP_LENGTH, FMIN, FMAX, MAX_SAMPLES
)


def load_audio(filepath, sr=SAMPLE_RATE):
    """
    Load an audio file and convert it to the requested sample rate.

    Args:
        filepath: path to the audio file (.ogg, .mp3, .wav)
        sr: target sample rate (default: 32000 Hz)

    Returns:
        y: 1D numpy array of audio samples
        sr: the (effective) sample rate

    Notes:
        librosa.load() does two things:
        1. Reads the audio file (any format librosa supports)
        2. Resamples it to the requested sample rate
        Stereo files are converted to mono automatically.
    """
    try:
        # librosa.load returns a tuple: (samples, sample_rate)
        # y is a 1D numpy array, e.g. [0.01, -0.03, 0.05, ...]
        y, sr = librosa.load(filepath, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, sr


def pad_or_trim(y, max_samples=MAX_SAMPLES):
    """
    Force the audio signal to exactly max_samples samples in length.

    Longer than 5 s: keep the first 5 s.
    Shorter than 5 s: pad with zeros (silence) at the end.

    Args:
        y: 1D numpy array of audio samples
        max_samples: target number of samples (default: 160000)

    Returns:
        y: 1D numpy array of exactly max_samples elements

    Notes:
        Why a fixed length? The CNN expects fixed-size inputs.
        If one clip is 3 s and another is 10 s they cannot share a batch.
        Uniforming everything to 5 s gives every sample the same shape.
    """
    if len(y) > max_samples:
        # Clip is longer than 5 seconds: keep the first max_samples samples.
        y = y[:max_samples]
    elif len(y) < max_samples:
        # Clip is shorter than 5 seconds: pad zeros at the end.
        # (0, max_samples - len(y)) means: 0 zeros on the left, N zeros on the right.
        y = np.pad(y, (0, max_samples - len(y)), mode='constant', constant_values=0)

    return y


def audio_to_melspec(y, sr=SAMPLE_RATE):
    """
    Convert an audio signal to a normalised mel-spectrogram.

    This is the key function: it turns the waveform samples into a
    2D image the CNN can consume.

    Args:
        y: 1D numpy array of audio samples (e.g. 160000 numbers)
        sr: sample rate (32000)

    Returns:
        mel_norm: numpy array (128, 313), values in [0, 1]

    Internal steps:
        1. melspectrogram: window the audio, apply FFT, project onto
           the mel bands.
        2. power_to_db: switch to a logarithmic (decibel) scale —
           the ear perceives loudness logarithmically.
        3. min-max normalisation: map the values to [0, 1]
           (CNNs prefer small, uniform inputs).
    """
    # --- Step 1: compute the mel-spectrogram ---
    # n_fft=2048: each window covers 2048 samples (64 ms at 32 kHz)
    # hop_length=512: windows slide by 512 samples (16 ms)
    #   → 75 % overlap (a lot of overlap = good temporal resolution)
    # n_mels=128: 128 mel-frequency bands
    # fmin=50, fmax=14000: useful frequency range
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX
    )
    # mel_spec.shape = (128, 313); values are raw (positive) power.

    # --- Step 2: convert to decibels ---
    # ref=np.max normalises against the maximum value.
    # Values then range from 0 (loudest) to about -80 (silence).
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # --- Step 3: min-max normalise to [0, 1] ---
    # Formula: (x - min) / (max - min)
    # After this, 0 = silence and 1 = maximum intensity.
    mel_min = mel_spec_db.min()
    mel_max = mel_spec_db.max()

    # Guard against division by zero on a (rare) fully-silent clip.
    if mel_max - mel_min == 0:
        mel_norm = np.zeros_like(mel_spec_db)
    else:
        mel_norm = (mel_spec_db - mel_min) / (mel_max - mel_min)

    return mel_norm


def load_and_process_audio(filepath):
    """
    End-to-end: from an .ogg file path to a normalised mel-spectrogram.
    Combines all the steps in a single call.

    Args:
        filepath: path to the .ogg file

    Returns:
        mel_norm: numpy array (128, 313) with values in [0, 1],
                  or None if loading fails.

    Usage:
        mel = load_and_process_audio("data/train_audio/banana/XC12345.ogg")
        if mel is not None:
            print(mel.shape)  # (128, 313)
    """
    # Step 1: load the audio
    y, sr = load_audio(filepath)
    if y is None:
        return None

    # Step 2: trim or pad to exactly 5 seconds
    y = pad_or_trim(y)

    # Step 3: convert to a normalised mel-spectrogram
    mel = audio_to_melspec(y, sr)

    return mel


def process_batch(filepaths, max_workers=4):
    """
    Process a list of audio files in parallel (faster).

    Args:
        filepaths: list of file paths
        max_workers: number of parallel workers

    Returns:
        mels: list of mel-spectrograms (None entries are skipped)
        valid_indices: indices of the files that were processed successfully

    Notes:
        Processing thousands of audio files serially is slow.
        With a thread pool we can process several at once (defaults to
        four; raise it if you have more CPU cores).
    """
    from concurrent.futures import ThreadPoolExecutor

    mels = []
    valid_indices = []

    # ThreadPoolExecutor builds a pool of worker threads.
    # max_workers=4 means up to 4 files are processed concurrently.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # executor.map applies load_and_process_audio to every filepath
        # and returns results in the same order.
        results = list(executor.map(load_and_process_audio, filepaths))

    # Keep only the successful results.
    for i, mel in enumerate(results):
        if mel is not None:
            mels.append(mel)
            valid_indices.append(i)

    return mels, valid_indices

# ─── AUGMENTATION FUNCTIONS ─────────────────────────────────────────────────

def time_shift(y, sr=SAMPLE_RATE, max_shift_seconds=1.0):
    """
    Shifts the audio signal along the time axis.

    Example: a bird call starting at second 1.5 gets shifted
    to start at second 0.3 — the model learns that the temporal
    position of the call is not important.

    Args:
        y: 1D audio array
        max_shift_seconds: maximum shift in seconds (default 1s)

    Returns:
        y_shifted: shifted audio (same length, zeros at the border)
    """
    max_shift = int(max_shift_seconds * sr)
    shift = np.random.randint(-max_shift, max_shift)
    y_shifted = np.roll(y, shift)
    if shift > 0:
        y_shifted[:shift] = 0
    elif shift < 0:
        y_shifted[shift:] = 0
    return y_shifted


def freq_mask(mel_spec, num_masks=1, max_width=20):
    """
    SpecAugment: masks random frequency bands in the spectrogram.

    Example: masks frequencies between 40Hz and 200Hz — the model
    learns to recognise bird calls even with partial information.

    Args:
        mel_spec: array (128, 313) — the spectrogram
        num_masks: number of masks to apply (default 1)
        max_width: maximum mask width in mel bands (default 20)

    Returns:
        mel_spec: spectrogram with masked frequency bands (set to 0)
    """
    mel_spec = mel_spec.copy()
    num_mel_bins = mel_spec.shape[0]  # 128

    for _ in range(num_masks):
        width = np.random.randint(1, max_width)
        start = np.random.randint(0, num_mel_bins - width)
        mel_spec[start:start + width, :] = 0

    return mel_spec


def time_mask(mel_spec, num_masks=1, max_width=40):
    """
    SpecAugment: masks random time windows in the spectrogram.

    Example: masks frames from second 1.2 to second 1.8 — the model
    learns to recognise bird calls even if part of them is missing.

    Args:
        mel_spec: array (128, 313)
        num_masks: number of masks to apply (default 1)
        max_width: maximum mask width in time frames (default 40)

    Returns:
        mel_spec: spectrogram with masked time windows (set to 0)
    """
    mel_spec = mel_spec.copy()
    num_time_frames = mel_spec.shape[1]  # 313

    for _ in range(num_masks):
        width = np.random.randint(1, max_width)
        start = np.random.randint(0, num_time_frames - width)
        mel_spec[:, start:start + width] = 0

    return mel_spec


def apply_specaugment(mel_spec, freq_masks=2, time_masks=2,
                      freq_width=20, time_width=40):
    """
    Full SpecAugment: applies freq_mask + time_mask simultaneously.

    This is the method from Park et al. (2019) which demonstrated
    significant improvements for audio classification tasks.

    Args:
        mel_spec: array (128, 313)
        freq_masks: number of frequency masks (default 2)
        time_masks: number of time masks (default 2)
        freq_width: max frequency mask width in mel bands (default 20)
        time_width: max time mask width in frames (default 40)

    Returns:
        mel_spec: spectrogram with both masks applied
    """
    mel_spec = freq_mask(mel_spec, num_masks=freq_masks, max_width=freq_width)
    mel_spec = time_mask(mel_spec, num_masks=time_masks, max_width=time_width)
    return mel_spec


def add_noise(y, noise_level=0.005):
    """
    Adds Gaussian noise to the audio signal.

    Simulates the noisy conditions of real-world soundscapes
    (wind, rain, other animals in the background).

    Args:
        y: 1D audio array
        noise_level: noise intensity (default 0.005 = very light)

    Returns:
        y_noisy: audio with added noise
    """
    noise = np.random.normal(0, noise_level, len(y))
    return y + noise


def augment_audio(y, mel_spec, augmentation_type='specaugment',
                  noise_level=0.005):
    """
    Main augmentation function. Applies the chosen strategy.

    Args:
        y: 1D audio array (for pre-spectrogram augmentation)
        mel_spec: array (128, 313) (for post-spectrogram augmentation)
        augmentation_type: 'time_shift', 'freq_mask', 'specaugment', 'all'
        noise_level: noise intensity (if used)

    Returns:
        mel_spec_aug: augmented spectrogram (128, 313)
    """
    if augmentation_type == 'time_shift':
        y = time_shift(y)
        mel_spec = audio_to_melspec(y)

    elif augmentation_type == 'freq_mask':
        mel_spec = freq_mask(mel_spec, num_masks=2, max_width=20)

    elif augmentation_type == 'specaugment':
        mel_spec = apply_specaugment(mel_spec)

    elif augmentation_type == 'all':
        y = time_shift(y)
        y = add_noise(y, noise_level=noise_level)
        mel_spec = audio_to_melspec(y)
        mel_spec = apply_specaugment(mel_spec)

    return mel_spec
