"""
utils/data_loader.py — Data loading and preparation for BirdCLEF 2026

This module is responsible for:
1. Reading train.csv and taxonomy.csv
2. Building multi-label vectors (binary vectors of length 234)
3. Preparing a dataset ready for training
4. Performing the train/validation split

Usage:
    from utils.data_loader import prepare_dataset
    X_train, X_val, y_train, y_val, label_names = prepare_dataset(max_samples=2000)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_CSV, TAXONOMY_CSV, TRAIN_AUDIO_DIR,
    N_CLASSES, VALIDATION_SPLIT
)
from utils.audio_pipeline import load_and_process_audio


def load_metadata():
    """
    Load and prepare the training-set metadata.

    Returns:
        train_df: DataFrame with the training data
        taxonomy_df: DataFrame with all 234 species
        label_names: sorted list of the 234 species names (primary_label)

    Step-by-step:
        1. Read train.csv — info about every recording.
        2. Read taxonomy.csv — the 234 target species.
        3. Build the sorted list of species names. This ordering defines
           the column ordering of the 234 outputs and MUST be the same
           at training time and at submission time.
    """
    # pd.read_csv reads a CSV into a DataFrame (an in-memory table).
    train_df = pd.read_csv(TRAIN_CSV)
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)

    # Sort alphabetically so the ordering is reproducible across runs.
    label_names = sorted(taxonomy_df['primary_label'].unique().tolist())

    print(f"Training set: {len(train_df)} recordings")
    print(f"Target species: {len(label_names)}")

    return train_df, taxonomy_df, label_names


def create_label_vector(primary_label, secondary_labels, label_names):
    """
    Build the binary multi-label vector for a single recording.

    Args:
        primary_label: the primary species (e.g. "banana")
        secondary_labels: list of secondary species (e.g. "['rubthr1', 'houspa']")
        label_names: sorted list of the 234 species

    Returns:
        label_vec: numpy array of 234 zeros and ones
                   1 = species present, 0 = not present

    Example:
        Given label_names = ["banana", "houspa", "osprey", ...]
        and primary_label = "banana", secondary_labels = ["osprey"]
        → label_vec = [1, 0, 1, 0, 0, ...]
                       ^banana  ^osprey

    Notes:
        We start with a 234-long vector of zeros, then flip to 1 the
        positions of the species present in this recording. A
        label-to-index dictionary makes the lookup O(1).
    """
    # Map every species name to its position in the sorted list.
    # e.g. {"banana": 0, "houspa": 1, "osprey": 2, ...}
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    # Allocate a vector of zeros (float32 for CNN compatibility).
    label_vec = np.zeros(len(label_names), dtype=np.float32)

    # Mark the primary species.
    # We cast to str defensively, in case the column contains a number.
    primary = str(primary_label)
    if primary in label_to_idx:
        label_vec[label_to_idx[primary]] = 1.0

    # Mark secondary species (if any).
    # secondary_labels is a string like "['rubthr1', 'houspa']" — parse it.
    if isinstance(secondary_labels, str) and secondary_labels != '[]':
        try:
            # ast.literal_eval safely turns the string into a real Python list.
            import ast
            sec_list = ast.literal_eval(secondary_labels)
            for sec in sec_list:
                sec = str(sec)
                if sec in label_to_idx:
                    label_vec[label_to_idx[sec]] = 1.0
        except (ValueError, SyntaxError):
            # If parsing fails, just skip the secondary labels.
            pass

    return label_vec


def prepare_dataset(max_samples=None, random_state=42):
    """
    Build the full dataset: load audio, compute mel-spectrograms, build labels.

    Args:
        max_samples: if set, only use N samples (useful for quick tests).
                     If None, use the whole dataset.
        random_state: seed for reproducibility (same number → same split)

    Returns:
        X_train: array (N_train, 128, 313, 1) — training mel-spectrograms
        X_val:   array (N_val, 128, 313, 1) — validation mel-spectrograms
        y_train: array (N_train, 234) — training multi-labels
        y_val:   array (N_val, 234) — validation multi-labels
        label_names: list of the 234 species

    Flow:
        1. Load metadata (CSV).
        2. Optionally take a random subset (for speed).
        3. For each audio file: compute the mel-spectrogram + label vector.
        4. Split into train and validation.
    """
    # Step 1: load the metadata.
    train_df, taxonomy_df, label_names = load_metadata()

    # Step 2: optionally take a random subset.
    if max_samples is not None and max_samples < len(train_df):
        # .sample(n) draws n random rows.
        # random_state=42 makes the choice reproducible.
        train_df = train_df.sample(n=max_samples, random_state=random_state)
        print(f"Using a subset of {max_samples} samples")

    # Step 3: process every audio file.
    spectrograms = []   # all mel-spectrograms accumulated here
    labels = []         # all label vectors accumulated here
    skipped = 0         # counter for skipped files (missing or corrupt)

    total = len(train_df)
    for idx, (_, row) in enumerate(train_df.iterrows()):
        # Print progress every 100 files.
        if idx % 100 == 0:
            print(f"  Processing: {idx}/{total} ({idx/total*100:.0f}%)")

        # Build the full audio file path.
        # row['filename'] looks like "banana/XC12345.ogg".
        filepath = os.path.join(TRAIN_AUDIO_DIR, row['filename'])

        # Skip if the file doesn't exist.
        if not os.path.exists(filepath):
            skipped += 1
            continue

        # Build the mel-spectrogram.
        mel = load_and_process_audio(filepath)
        if mel is None:
            skipped += 1
            continue

        # Build the label vector.
        label_vec = create_label_vector(
            row['primary_label'],
            row['secondary_labels'],
            label_names
        )

        spectrograms.append(mel)
        labels.append(label_vec)

    print(f"Processed: {len(spectrograms)}, Skipped: {skipped}")

    # Step 4: convert the lists into numpy arrays.
    X = np.array(spectrograms)   # shape: (N, 128, 313)
    y = np.array(labels)         # shape: (N, 234)

    # Add a channel dimension for the CNN.
    # The CNN expects (N, height, width, channels). np.expand_dims appends
    # a new axis at position -1 (the last one): (N, 128, 313) → (N, 128, 313, 1)
    X = np.expand_dims(X, axis=-1)

    # Step 5: split into train and validation.
    # train_test_split shuffles the data and splits it.
    # test_size=0.2 means 80% training, 20% validation.
    # stratification isn't viable in the multi-label case, so we just shuffle.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        random_state=random_state
    )

    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Input shape:  {X_train.shape[1:]}")
    print(f"Output shape: {y_train.shape[1:]}")

    return X_train, X_val, y_train, y_val, label_names
