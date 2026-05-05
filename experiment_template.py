"""
experiment_template.py — Stable template for BirdCLEF 2026 experiments (v3)

What's new v3:
- model_type: "cnn" or "efficientnet" (chosen by the LLM)
- Improved augmentation: noise + time_shift + freq_mask
- 3-channel input for EfficientNet (replicate mel-spec to RGB)

The agent DOES NOT modify this file.
The agent generates a JSON file with parameters, and this script reads them.

Usage:
    python experiment_template.py --config experiments/params_001.json
"""

import os
import sys
import json
import argparse
import numpy as np


def load_params(config_path=None):
    defaults = {
        "experiment_name": "baseline",
        "model_type": "cnn",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
        "n_filters_1": 32,
        "n_filters_2": 64,
        "n_filters_3": 128,
        "dropout_rate": 0.3,
        "dense_units": 256,
        "n_mels": 128,
        "n_fft": 1024,
        "top_db": 80.0,
        "hop_length": 320,
        "fmin": 20,
        "fmax": 16000,
        "max_samples": 2000,
        "mel_norm": None,        # None o "slaney"
        "mel_scale": "htk",      # "htk" o "slaney"
        "use_augmentation": False,
        "augmentation_type": "noise",
        "augmentation_noise": 0.01,
        "unfreeze_layers": 0
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            overrides = json.load(f)
        defaults.update(overrides)
        print(f"Parameters loaded from: {config_path}")
    else:
        print("Using default parameters (baseline)")
    return defaults


# =============================================================
# AUDIO PIPELINE
# =============================================================
def make_melspec(y, sr, params):
    import librosa
    top_db = params.get("top_db", 80.0)
    
    # mel_scale: "htk" usa htk=True, qualsiasi altro valore usa htk=False
    use_htk = params.get("mel_scale", "htk") == "htk"
    
    # Fix: Gemma sometimes sends "null" as string instead of None
    mel_norm_val = params.get("mel_norm", None)
    if mel_norm_val == "null":
        mel_norm_val = None

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=params["n_mels"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        fmin=params["fmin"],
        fmax=params["fmax"],
        norm=mel_norm_val,
        htk=use_htk
    )
    
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=top_db)
    mel_db = np.nan_to_num(mel_db, nan=-top_db, posinf=0.0, neginf=-top_db)
    mel_norm = (mel_db + top_db) / top_db
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    
    return mel_norm

def load_and_process(filepath, sr=32000, duration=5, params=None):
    import librosa
    try:
        y, sr = librosa.load(filepath, sr=sr)
        max_len = sr * duration
        if len(y) > max_len:
            y = y[:max_len]
        elif len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        return make_melspec(y, sr, params)
    except Exception as e:
        print(f"  Error with {filepath}: {e}")
        return None


# =============================================================
# AUGMENTATION
# =============================================================
def augment_batch(X, params):
    """
    Augmentation with 4 options:
    - noise: gaussian noise
    - time_shift: shift the spectrogram in time
    - freq_mask: mask random frequency bands
    - specaugment: simultaneous freq + time masking (Park et al. 2019)
    - all: combines all strategies
    """
    if not params.get("use_augmentation", False):
        return X

    aug_type = params.get("augmentation_type", "noise")
    X_aug = X.copy()

    if aug_type == "noise":
        noise_std = params.get("augmentation_noise", 0.01)
        noise = np.random.normal(0, noise_std, X_aug.shape).astype(np.float32)
        X_aug = X_aug + noise

    elif aug_type == "time_shift":
        for i in range(len(X_aug)):
            shift = np.random.randint(-20, 20)
            X_aug[i] = np.roll(X_aug[i], shift, axis=1)

    elif aug_type == "freq_mask":
        for i in range(len(X_aug)):
            n_mels = X_aug[i].shape[0]
            f_start = np.random.randint(0, n_mels - 10)
            f_width = np.random.randint(5, 20)
            X_aug[i, f_start:f_start + f_width, :] = 0

    elif aug_type == "specaugment":
        # Full SpecAugment: frequency masking + time masking simultaneously
        # Based on Park et al. (2019)
        for i in range(len(X_aug)):
            spec = X_aug[i]
            n_mels = spec.shape[0]       # 128
            n_frames = spec.shape[1]     # 313

            # Apply 2 frequency masks
            for _ in range(2):
                f_width = np.random.randint(1, 20)
                f_start = np.random.randint(0, n_mels - f_width)
                spec[f_start:f_start + f_width, :] = 0

            # Apply 2 time masks
            for _ in range(2):
                t_width = np.random.randint(1, 40)
                t_start = np.random.randint(0, n_frames - t_width)
                spec[:, t_start:t_start + t_width] = 0

            X_aug[i] = spec

    elif aug_type == "all":
        # Everything: time shift + noise + full SpecAugment
        noise_std = params.get("augmentation_noise", 0.005)
        noise = np.random.normal(0, noise_std, X_aug.shape).astype(np.float32)
        X_aug = X_aug + noise

        for i in range(len(X_aug)):
            # Time shift
            shift = np.random.randint(-20, 20)
            X_aug[i] = np.roll(X_aug[i], shift, axis=1)

            # SpecAugment
            spec = X_aug[i]
            n_mels = spec.shape[0]
            n_frames = spec.shape[1]

            for _ in range(2):
                f_width = np.random.randint(1, 20)
                f_start = np.random.randint(0, n_mels - f_width)
                spec[f_start:f_start + f_width, :] = 0

            for _ in range(2):
                t_width = np.random.randint(1, 40)
                t_start = np.random.randint(0, n_frames - t_width)
                spec[:, t_start:t_start + t_width] = 0

            X_aug[i] = spec
    elif aug_type == "background_noise":
        import librosa
        import glob
        soundscapes_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "train_soundscapes"
        )
        noise_files = glob.glob(os.path.join(soundscapes_dir, "*.ogg"))
        if noise_files:
            alpha = params.get("augmentation_noise", 0.1)
            sr = 32000
            samples_per_window = sr * 5
            for i in range(len(X_aug)):
                try:
                    noise_file = noise_files[np.random.randint(len(noise_files))]
                    y_noise, _ = librosa.load(
                        noise_file, sr=sr, mono=True,
                        offset=np.random.uniform(0, 30),
                        duration=5.0
                    )
                    if len(y_noise) < samples_per_window:
                        y_noise = np.pad(y_noise, (0, samples_per_window - len(y_noise)))
                    noise_mel = make_melspec(y_noise, sr, params)
                    noise_rgb = np.stack([noise_mel, noise_mel, noise_mel], axis=-1)
                    X_aug[i] = X_aug[i] + alpha * noise_rgb
                except Exception:
                    continue

    return np.clip(X_aug, 0, 1)
# =============================================================
# MODELS
# =============================================================
def build_cnn(input_shape, n_classes, params):
    """Custom CNN with 3 convolutional blocks (baseline)."""
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(params["n_filters_1"], (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(params["n_filters_2"], (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.1),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(params["n_filters_3"], (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(0.1),
        layers.GlobalAveragePooling2D(),
        layers.Dense(params["dense_units"], activation='relu'),
        layers.Dropout(params["dropout_rate"]),
        layers.Dense(n_classes, activation='sigmoid')
    ])
    return model


def build_efficientnet(input_shape, n_classes, params):
    """
    EfficientNetB0 with two-phase training strategy.

    Phase 1 (always): freeze the entire base, train only the classifier.
    Phase 2 (optional): unfreeze the top N layers with a very low lr.

    Following Chollet (2021) Chapter 8.3.2:
    'It is only possible to fine-tune the top layers once the classifier
    on top has already been trained. If the classifier is not already
    trained, the error signal will be too large and will destroy the
    representations previously learned.'
    """
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    # Load EfficientNetB0 pretrained on ImageNet
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Phase 1: freeze ALL base layers
    base_model.trainable = False

    # Build the full model with frozen base
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    # training=False keeps BatchNorm layers in inference mode
    # even when we later unfreeze layers — this is critical
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(params["dense_units"], activation='relu')(x)
    x = layers.Dropout(params["dropout_rate"])(x)
    outputs = layers.Dense(n_classes, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    return model, base_model


def build_model(input_shape, n_classes, params):
    """
    Builds the model and applies the two-phase fine-tuning strategy.

    Phase 1: train classifier only (base frozen)
    Phase 2: unfreeze top N layers with very low lr (if unfreeze_layers > 0)
    """
    from tensorflow import keras

    model_type = params.get("model_type", "cnn")

    if model_type == "efficientnet":
        print(">>> Model: EfficientNetB0 (transfer learning)")
        model, base_model = build_efficientnet(input_shape, n_classes, params)

        # Phase 1 optimizer: normal learning rate for the classifier
        optimizer_phase1 = keras.optimizers.Adam(
            learning_rate=params["learning_rate"],
            clipnorm=1.0
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer_phase1,
            metrics=[keras.metrics.AUC(name='auc')]
        )
        print(f"Phase 1: base frozen, training classifier only")
        print(f"  lr={params['learning_rate']}, "
              f"unfreeze={params.get('unfreeze_layers', 0)} layers planned for Phase 2")

    else:
        print(f">>> Model: Custom CNN")
        model = build_cnn(input_shape, n_classes, params)
        base_model = None

        optimizer = keras.optimizers.Adam(
            learning_rate=params["learning_rate"],
            clipnorm=1.0
        )
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=[keras.metrics.AUC(name='auc')]
        )

    return model, base_model

# =============================================================
# MAIN
# =============================================================
def run_experiment(params):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow import keras
    import ast
    import time

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {params['experiment_name']}")
    print(f"Model: {params.get('model_type', 'cnn')}")
    print(f"{'='*60}")
    print(f"Parameters: {json.dumps(params, indent=2)}")

    # --- Carica dati ---
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    train_csv = os.path.join(data_dir, "train.csv")
    taxonomy_csv = os.path.join(data_dir, "taxonomy.csv")
    audio_dir = os.path.join(data_dir, "train_audio")

    train_df = pd.read_csv(train_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)
    label_names = sorted(taxonomy_df['primary_label'].unique().tolist())
    n_classes = len(label_names)
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    if params["max_samples"] < len(train_df):
        train_df = train_df.sample(n=params["max_samples"], random_state=42)

    print(f"\nLoading {len(train_df)} audio files...")

    # --- Processa audio ---
    # --- Load train_audio clips ---
    print(f"\nLoading {len(train_df)} train_audio clips...")
    spectrograms = []
    labels = []

    for idx, (_, row) in enumerate(train_df.iterrows()):
        if idx % 200 == 0:
            print(f"  clips: {idx}/{len(train_df)}")

        filepath = os.path.join(audio_dir, row['filename'])
        if not os.path.exists(filepath):
            continue

        mel = load_and_process(filepath, params=params)
        if mel is None:
            continue

        label_vec = np.zeros(n_classes, dtype=np.float32)
        primary = str(row['primary_label'])
        if primary in label_to_idx:
            label_vec[label_to_idx[primary]] = 1.0
        sec = row.get('secondary_labels', '[]')
        if isinstance(sec, str) and sec != '[]':
            try:
                for s in ast.literal_eval(sec):
                    if str(s) in label_to_idx:
                        label_vec[label_to_idx[str(s)]] = 1.0
            except:
                pass

        spectrograms.append(mel)
        labels.append(label_vec)

    print(f"  Loaded {len(spectrograms)} clips from train_audio")

    # --- Load train_soundscapes windows ---
    soundscapes_csv = os.path.join(data_dir, "train_soundscapes_labels.csv")
    soundscapes_dir = os.path.join(data_dir, "train_soundscapes")

    if os.path.exists(soundscapes_csv) and os.path.exists(soundscapes_dir):
        print(f"\nLoading train_soundscapes windows...")
        sc_df = pd.read_csv(soundscapes_csv)

        # Use max_samples/4 soundscape windows to keep balance
        max_sc = min(len(sc_df), params["max_samples"])
        sc_df = sc_df.sample(n=max_sc, random_state=42)

        sc_loaded = 0
        for idx, (_, row) in enumerate(sc_df.iterrows()):
            if idx % 200 == 0:
                print(f"  soundscapes: {idx}/{len(sc_df)}")

            filepath = os.path.join(soundscapes_dir, row['filename'])
            if not os.path.exists(filepath):
                continue

            # Convert start time to seconds
            # Format is HH:MM:SS
            start_str = str(row['start'])
            parts = start_str.split(':')
            start_sec = int(parts[0])*3600 + int(parts[1])*60 + int(parts[2])

            # Load 5-second window
            try:
                import librosa
                y, sr = librosa.load(
                    filepath,
                    sr=32000,
                    offset=start_sec,
                    duration=5.0
                )
                max_len = 32000 * 5
                if len(y) < max_len:
                    y = np.pad(y, (0, max_len - len(y)))
                elif len(y) > max_len:
                    y = y[:max_len]

                mel = make_melspec(y, 32000, params)
                if mel is None:
                    continue

            except Exception as e:
                continue

            # Build label vector from primary_label column
            # Labels are separated by semicolons: "22961;23158;24321"
            label_vec = np.zeros(n_classes, dtype=np.float32)
            labels_str = str(row['primary_label'])
            for lbl in labels_str.split(';'):
                lbl = lbl.strip()
                if lbl in label_to_idx:
                    label_vec[label_to_idx[lbl]] = 1.0

            spectrograms.append(mel)
            labels.append(label_vec)
            sc_loaded += 1

        print(f"  Loaded {sc_loaded} windows from train_soundscapes")
    else:
        print("  train_soundscapes not found, using only train_audio")

    print(f"\nTotal samples: {len(spectrograms)}")
    X = np.array(spectrograms)
    y = np.array(labels)
    # --- Prepara canali in base al modello ---
    model_type = params.get("model_type", "cnn")
    if model_type == "efficientnet":
        # EfficientNet expects 3 channels (RGB)
        # We replicate the mel-spectrogram 3 times
        X = np.stack([X, X, X], axis=-1)
        print(f"Dataset: {X.shape[0]} samples, shape={X.shape[1:]} (3 channels for EfficientNet)")
    else:
        # Custom CNN expects 1 channel
        X = np.expand_dims(X, axis=-1)
        print(f"Dataset: {X.shape[0]} samples, shape={X.shape[1:]}")

    # --- Split FISSO ---
    # Validation sempre uguale per confronto comparabile tra esperimenti
    # Usiamo sempre gli stessi indici indipendentemente da max_samples
    np.random.seed(42)
    n_total = len(spectrograms)
    val_size = max(100, int(n_total * 0.2))
    val_indices = set(np.random.choice(n_total, size=val_size, replace=False))
    train_indices = [i for i in range(n_total) if i not in val_indices]
    val_indices = list(val_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    print(f"Fixed split: {len(X_train)} train, {len(X_val)} val")

    # --- Augmentation ---
    X_train = augment_batch(X_train, params)

    # --- Model ---
    input_shape = X_train.shape[1:]
    model, base_model = build_model(input_shape, n_classes, params)
    model.summary()

    # --- Callbacks ---
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=3,
            restore_best_weights=True, mode='max'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc', factor=0.5,
            patience=2, mode='max'
        )
    ]

    # --- Phase 1 Training: classifier only (base frozen) ---
    unfreeze = params.get("unfreeze_layers", 0)

    # If fine-tuning is planned, use half the epochs in phase 1
    phase1_epochs = max(3, params["epochs"] // 2) if unfreeze > 0 else params["epochs"]

    print(f"\n--- Phase 1: training classifier (base frozen) ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=phase1_epochs,
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    # --- Phase 2 Training: fine-tune top layers (Chollet cap. 8.3.2) ---
    if unfreeze > 0 and base_model is not None:
        print(f"\n--- Phase 2: unfreezing top {unfreeze} layers with lr=1e-5 ---")

        # Unfreeze only the top N layers
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze]:
            layer.trainable = False

        # Recompile with VERY low learning rate
        model.compile(
            loss='binary_crossentropy',
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-5,
                clipnorm=1.0
            ),
            metrics=[keras.metrics.AUC(name='auc')]
        )

        remaining_epochs = params["epochs"] - phase1_epochs

        history2 = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=remaining_epochs,
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=1
        )

        # Merge both histories
        history.history['val_auc'] += history2.history.get('val_auc', [])
        history.history['val_loss'] += history2.history.get('val_loss', [])
        history.history['auc'] += history2.history.get('auc', [])
        history.history['loss'] += history2.history.get('loss', [])

    # --- Results ---
    elapsed = time.time() - start_time
    val_auc = float(max(history.history.get('val_auc', [0])))
    val_loss = float(min(history.history.get('val_loss', [999])))
    train_auc = float(max(history.history.get('auc', [0])))
    epochs_done = len(history.history['loss'])

    model.save('best_model.keras')

    metrics = {
        "experiment_name": params["experiment_name"],
        "model_type": params.get("model_type", "cnn"),
        "val_auc": round(val_auc, 4),
        "val_loss": round(val_loss, 4),
        "train_auc": round(train_auc, 4),
        "epochs_trained": epochs_done,
        "elapsed_seconds": round(elapsed, 1),
        "n_samples": X.shape[0]
    }

    print(f"\n{'='*60}")
    print(f"RESULTS: {params['experiment_name']}")
    print(f"{'='*60}")
    print(json.dumps(metrics))

    return metrics


if __name__ == "__main__":
    # Importa keras qui per evitare errori se non serve
    import tensorflow as tf
    from tensorflow import keras

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    params = load_params(args.config)
    run_experiment(params)
