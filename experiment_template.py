"""
experiment_template.py — Template stabile per esperimenti BirdCLEF 2026 (v3)

Novità v3:
- model_type: "cnn" o "efficientnet" (l'LLM sceglie)
- Augmentation migliorata: noise + time_shift + freq_mask
- Input 3 canali per EfficientNet (replica mel-spec su RGB)

L'agente NON modifica questo file.
L'agente genera un file JSON con i parametri, e questo script li legge.

Uso:
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
        "n_fft": 2048,
        "hop_length": 512,
        "fmin": 50,
        "fmax": 14000,
        "max_samples": 2000,
        "use_augmentation": False,
        "augmentation_type": "noise",
        "augmentation_noise": 0.01,
        "unfreeze_layers": 0
    }
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            overrides = json.load(f)
        defaults.update(overrides)
        print(f"Parametri caricati da: {config_path}")
    else:
        print("Usando parametri di default (baseline)")
    return defaults


# =============================================================
# PIPELINE AUDIO
# =============================================================
def make_melspec(y, sr, params):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=params["n_mels"],
        n_fft=params["n_fft"],
        hop_length=params["hop_length"],
        fmin=params["fmin"],
        fmax=params["fmax"]
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_min, mel_max = mel_db.min(), mel_db.max()
    if mel_max - mel_min == 0:
        return np.zeros_like(mel_db)
    return (mel_db - mel_min) / (mel_max - mel_min)


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
        print(f"  Errore con {filepath}: {e}")
        return None


# =============================================================
# AUGMENTATION
# =============================================================
def augment_batch(X, params):
    """
    Augmentation migliorata con 3 opzioni:
    - noise: rumore gaussiano
    - time_shift: sposta lo spettrogramma nel tempo
    - freq_mask: maschera bande di frequenza casuali
    """
    if not params.get("use_augmentation", False):
        return X

    aug_type = params.get("augmentation_type", "noise")
    X_aug = X.copy()

    if aug_type == "noise" or aug_type == "all":
        noise_std = params.get("augmentation_noise", 0.01)
        noise = np.random.normal(0, noise_std, X_aug.shape).astype(np.float32)
        X_aug = X_aug + noise

    if aug_type == "time_shift" or aug_type == "all":
        for i in range(len(X_aug)):
            shift = np.random.randint(-20, 20)
            X_aug[i] = np.roll(X_aug[i], shift, axis=1)

    if aug_type == "freq_mask" or aug_type == "all":
        for i in range(len(X_aug)):
            n_mels = X_aug[i].shape[0]
            f_start = np.random.randint(0, n_mels - 10)
            f_width = np.random.randint(5, 15)
            X_aug[i, f_start:f_start + f_width, :] = 0

    return np.clip(X_aug, 0, 1)


# =============================================================
# MODELLI
# =============================================================
def build_cnn(input_shape, n_classes, params):
    """CNN custom con 3 blocchi convoluzionali (baseline)."""
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
    EfficientNetB0 pre-addestrato come feature extractor.

    Come funziona:
    1. EfficientNetB0 è stato addestrato su ImageNet (milioni di immagini)
    2. Sa già riconoscere pattern visivi (bordi, texture, forme)
    3. Noi "congeliamo" i suoi pesi e aggiungiamo solo un classificatore
    4. Opzionalmente, "scongeliamo" gli ultimi N layer per fine-tuning

    Input: (H, W, 3) — serve 3 canali, quindi replichiamo il mel-spec
    """
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    # Carichiamo EfficientNetB0 senza il suo classificatore originale
    # weights='imagenet' = pesi pre-addestrati
    # include_top=False = togliamo il layer finale (era per 1000 classi ImageNet)
    base_model = keras.applications.EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Congeliamo tutti i layer del backbone
    base_model.trainable = False

    # Opzionale: scongeliamo gli ultimi N layer per fine-tuning
    # Questo permette al modello di adattarsi meglio ai nostri dati
    unfreeze = params.get("unfreeze_layers", 0)
    if unfreeze > 0:
        for layer in base_model.layers[-unfreeze:]:
            layer.trainable = True

    # Costruiamo il modello completo
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(params["dense_units"], activation='relu'),
        layers.Dropout(params["dropout_rate"]),
        layers.Dense(n_classes, activation='sigmoid')
    ])

    return model


def build_model(input_shape, n_classes, params):
    """Sceglie il modello in base a model_type."""
    model_type = params.get("model_type", "cnn")

    if model_type == "efficientnet":
        print(">>> Modello: EfficientNetB0 (transfer learning)")
        model = build_efficientnet(input_shape, n_classes, params)
    else:
        print(f">>> Modello: CNN custom ({params['n_filters_1']}/{params['n_filters_2']}/{params['n_filters_3']})")
        model = build_cnn(input_shape, n_classes, params)

    # Gradient clipping for stability
    optimizer = keras.optimizers.Adam(
        learning_rate=params["learning_rate"],
        clipnorm=1.0
    )
    
    # Label smoothing for better generalization
    model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=[keras.metrics.AUC(name='auc')]
)
    return model


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
    print(f"ESPERIMENTO: {params['experiment_name']}")
    print(f"Modello: {params.get('model_type', 'cnn')}")
    print(f"{'='*60}")
    print(f"Parametri: {json.dumps(params, indent=2)}")

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

    print(f"\nCaricamento {len(train_df)} file audio...")

    # --- Processa audio ---
    spectrograms = []
    labels = []
    for idx, (_, row) in enumerate(train_df.iterrows()):
        if idx % 200 == 0:
            print(f"  {idx}/{len(train_df)} ({idx/len(train_df)*100:.0f}%)")

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

    X = np.array(spectrograms)
    y = np.array(labels)

    # --- Prepara canali in base al modello ---
    model_type = params.get("model_type", "cnn")
    if model_type == "efficientnet":
        # EfficientNet vuole 3 canali (RGB)
        # Replichiamo il mel-spectrogram 3 volte
        X = np.stack([X, X, X], axis=-1)
        print(f"Dataset: {X.shape[0]} campioni, shape={X.shape[1:]} (3 canali per EfficientNet)")
    else:
        # CNN custom vuole 1 canale
        X = np.expand_dims(X, axis=-1)
        print(f"Dataset: {X.shape[0]} campioni, shape={X.shape[1:]}")

    # --- Split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Augmentation ---
    X_train = augment_batch(X_train, params)

    # --- Modello ---
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, n_classes, params)
    model.summary()

    # --- Training ---
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

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=1
    )

    # --- Risultati ---
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
    print(f"RISULTATI: {params['experiment_name']}")
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
