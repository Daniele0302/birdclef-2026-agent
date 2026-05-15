"""
train_strong_v40.py - Stronger CPU-only retraining for BirdCLEF 2026.

Why this exists: existing exp78 has val_auc=0.85 on focal recordings but
LB 0.592 on soundscapes -> domain gap. This script narrows the gap by:

  1. Mixing train_audio (focal, clean) with ALL train_soundscapes windows
     (passive, real test distribution) so the backbone sees the target domain.
  2. Stratified sampling of train_audio: cap N per class so rare species are
     not lost in the long tail.
  3. Class-balanced sample weights during training.
  4. Phase 1 (head only, base frozen) -> Phase 2 (top-30 backbone layers
     unfrozen at lr=5e-5). Following Chollet 8.3.2.
  5. Validation = soundscape windows held out by FILE (group split). No leakage.
  6. Best checkpoint selected on macro-AUC over the soundscape held-out, the
     metric that matches the Kaggle test distribution.
  7. Mixup on training batches (alpha=0.2) for regularisation.

Run:
    cd /Users/danielemalerba/Downloads/birdclef-agent
    .venv/bin/python train_strong_v40.py

Outputs (always written into the project root):
    cache/specs_focal_v40.npy       (stratified focal mel-specs, float16)
    cache/labels_focal_v40.npy
    cache/specs_soundscape_v40.npy
    cache/labels_soundscape_v40.npy
    cache/soundscape_files_v40.npy  (group ids for held-out split)
    model_v40_strong.keras          (full model, best on soundscape_val_auc)
    train_v40_log.json              (training history)
"""

import os
import sys
import json
import time
import ast
import math
import argparse

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# CONFIG  (must match exp78 spectrogram params for reproducibility)
# ---------------------------------------------------------------
SAMPLE_RATE = 32000
DURATION    = 5.0
N_MELS      = 64
N_FFT       = 2048
HOP_LENGTH  = 256
FMIN        = 20
FMAX        = 16000
TOP_DB      = 40.0
MEL_NORM    = "slaney"
USE_HTK     = True

TARGET_HEIGHT = N_MELS
TARGET_WIDTH  = 626

# Training-data sampling
MAX_PER_CLASS_FOCAL = 50      # cap focal samples per primary_label
MAX_TOTAL_FOCAL     = 8000    # absolute cap on focal samples
SOUNDSCAPE_VAL_FRAC = 0.2     # fraction of soundscape FILES held out
RANDOM_SEED         = 20260507

# Training
BATCH_SIZE        = 16
PHASE1_EPOCHS     = 4
PHASE2_EPOCHS     = 5
PHASE1_LR         = 3e-4
PHASE2_LR         = 5e-5
UNFREEZE_LAYERS   = 30
DENSE_UNITS       = 256
DROPOUT_RATE      = 0.4
MIXUP_ALPHA       = 0.2

# PROJECT_DIR is the repo root: this script lives in scaling_up/, so go up one level
# so that DATA_DIR, CACHE_DIR and model artefacts resolve relative to the root.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
CACHE_DIR   = os.path.join(PROJECT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_FOCAL_X  = os.path.join(CACHE_DIR, "specs_focal_v40.npy")
CACHE_FOCAL_Y  = os.path.join(CACHE_DIR, "labels_focal_v40.npy")
CACHE_SC_X     = os.path.join(CACHE_DIR, "specs_soundscape_v40.npy")
CACHE_SC_Y     = os.path.join(CACHE_DIR, "labels_soundscape_v40.npy")
CACHE_SC_FILES = os.path.join(CACHE_DIR, "soundscape_files_v40.npy")

OUTPUT_MODEL = os.path.join(PROJECT_DIR, "model_v40_strong.keras")
OUTPUT_LOG   = os.path.join(PROJECT_DIR, "train_v40_log.json")


# ---------------------------------------------------------------
# AUDIO PIPELINE  (identical to experiment_template.make_melspec)
# ---------------------------------------------------------------
def make_melspec(y, sr=SAMPLE_RATE):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX, norm=MEL_NORM, htk=USE_HTK,
    )
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    mel_db = np.nan_to_num(mel_db, nan=-TOP_DB, posinf=0.0, neginf=-TOP_DB)
    mel_norm = (mel_db + TOP_DB) / TOP_DB
    mel_norm = np.clip(mel_norm, 0.0, 1.0).astype(np.float32)
    if mel_norm.shape[1] < TARGET_WIDTH:
        mel_norm = np.pad(
            mel_norm, ((0, 0), (0, TARGET_WIDTH - mel_norm.shape[1])),
            mode="constant"
        )
    elif mel_norm.shape[1] > TARGET_WIDTH:
        mel_norm = mel_norm[:, :TARGET_WIDTH]
    return mel_norm  # shape (N_MELS, TARGET_WIDTH)


def load_audio_window(path, offset_sec=0.0, duration=DURATION):
    import librosa
    y, _ = librosa.load(
        path, sr=SAMPLE_RATE, mono=True,
        offset=offset_sec, duration=duration,
    )
    target_len = int(SAMPLE_RATE * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        y = y[:target_len]
    return y


# ---------------------------------------------------------------
# LABEL HELPERS
# ---------------------------------------------------------------
def load_labels_layout():
    tax = pd.read_csv(os.path.join(DATA_DIR, "taxonomy.csv"))
    label_names = sorted(tax["primary_label"].unique().tolist())
    label_to_idx = {lab: i for i, lab in enumerate(label_names)}
    return label_names, label_to_idx


def focal_label_vector(primary_label, secondary_labels_field, label_to_idx, n_classes):
    vec = np.zeros(n_classes, dtype=np.float32)
    p = str(primary_label)
    if p in label_to_idx:
        vec[label_to_idx[p]] = 1.0
    if isinstance(secondary_labels_field, str) and secondary_labels_field != "[]":
        try:
            for s in ast.literal_eval(secondary_labels_field):
                s = str(s)
                if s in label_to_idx:
                    vec[label_to_idx[s]] = 1.0
        except Exception:
            pass
    return vec


def soundscape_label_vector(primary_label_field, label_to_idx, n_classes):
    """soundscape labels are semicolon-separated species codes."""
    vec = np.zeros(n_classes, dtype=np.float32)
    s = str(primary_label_field)
    for tok in s.split(";"):
        tok = tok.strip()
        if tok in label_to_idx:
            vec[label_to_idx[tok]] = 1.0
    return vec


# ---------------------------------------------------------------
# PRECOMPUTE / CACHE MELSPECS
# ---------------------------------------------------------------
def stratified_focal_sample(train_df, max_per_class=MAX_PER_CLASS_FOCAL,
                            max_total=MAX_TOTAL_FOCAL, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    parts = []
    for label, g in train_df.groupby("primary_label"):
        n = min(len(g), max_per_class)
        parts.append(g.sample(n=n, random_state=rng.randint(2**31 - 1)))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_total:
        out = out.sample(n=max_total, random_state=seed).reset_index(drop=True)
    return out


def precompute_focal_cache(label_to_idx, n_classes, force=False):
    if (not force) and os.path.exists(CACHE_FOCAL_X) and os.path.exists(CACHE_FOCAL_Y):
        X = np.load(CACHE_FOCAL_X)
        Y = np.load(CACHE_FOCAL_Y)
        print(f"[cache hit] focal: X={X.shape} Y={Y.shape}")
        return X, Y

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    sample_df = stratified_focal_sample(train_df)
    print(f"[focal] stratified pick: {len(sample_df)} samples "
          f"(from {len(train_df)})")

    X_list = []
    Y_list = []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(sample_df.itertuples(index=False), 1):
        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-3)
            eta = (len(sample_df) - i) / max(rate, 1e-3)
            print(f"  focal {i}/{len(sample_df)}  rate={rate:.1f}/s  eta={eta/60:.1f}min")

        path = os.path.join(DATA_DIR, "train_audio", row.filename)
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            y = load_audio_window(path, offset_sec=0.0)
            mel = make_melspec(y)
            label_vec = focal_label_vector(
                row.primary_label, row.secondary_labels, label_to_idx, n_classes,
            )
            X_list.append(mel)
            Y_list.append(label_vec)
        except Exception as exc:
            skipped += 1
            print(f"  skip {path}: {exc}")

    X = np.stack(X_list, axis=0).astype(np.float16)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    print(f"[focal] kept {len(X)} skipped {skipped} in {time.time()-t0:.1f}s")
    np.save(CACHE_FOCAL_X, X)
    np.save(CACHE_FOCAL_Y, Y)
    return X, Y


def precompute_soundscape_cache(label_to_idx, n_classes, force=False):
    if ((not force) and os.path.exists(CACHE_SC_X)
            and os.path.exists(CACHE_SC_Y)
            and os.path.exists(CACHE_SC_FILES)):
        X = np.load(CACHE_SC_X)
        Y = np.load(CACHE_SC_Y)
        F = np.load(CACHE_SC_FILES, allow_pickle=True)
        print(f"[cache hit] soundscape: X={X.shape} Y={Y.shape} F={F.shape}")
        return X, Y, F

    sc_df = pd.read_csv(os.path.join(DATA_DIR, "train_soundscapes_labels.csv"))
    sc_dir = os.path.join(DATA_DIR, "train_soundscapes")
    print(f"[soundscape] {len(sc_df)} windows over "
          f"{sc_df['filename'].nunique()} files")

    X_list = []
    Y_list = []
    F_list = []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(sc_df.itertuples(index=False), 1):
        if i % 100 == 0:
            print(f"  soundscape {i}/{len(sc_df)}")

        path = os.path.join(sc_dir, row.filename)
        if not os.path.exists(path):
            skipped += 1
            continue
        # 'start' is HH:MM:SS string from competition format
        try:
            parts = str(row.start).split(":")
            start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except Exception:
            skipped += 1
            continue

        try:
            y = load_audio_window(path, offset_sec=start_sec)
            mel = make_melspec(y)
            label_vec = soundscape_label_vector(
                row.primary_label, label_to_idx, n_classes,
            )
            X_list.append(mel)
            Y_list.append(label_vec)
            F_list.append(row.filename)
        except Exception as exc:
            skipped += 1
            print(f"  skip {path}@{start_sec}: {exc}")

    X = np.stack(X_list, axis=0).astype(np.float16)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    F = np.array(F_list, dtype=object)
    print(f"[soundscape] kept {len(X)} skipped {skipped} in {time.time()-t0:.1f}s")
    np.save(CACHE_SC_X, X)
    np.save(CACHE_SC_Y, Y)
    np.save(CACHE_SC_FILES, F, allow_pickle=True)
    return X, Y, F


# ---------------------------------------------------------------
# MIXUP DATA GENERATOR
# ---------------------------------------------------------------
class MixupSequence:
    """Keras Sequence implementing mixup on (spec, multilabel) pairs.

    Each batch is built from sampled indices using `sample_weights`
    so under-represented classes appear more often.
    """

    def __init__(self, X_f16, Y, batch_size, sample_weights, alpha,
                 augment=True, seed=RANDOM_SEED):
        self.X = X_f16  # float16 to save RAM
        self.Y = Y
        self.batch_size = batch_size
        self.alpha = alpha
        self.augment = augment
        self.weights = sample_weights / sample_weights.sum()
        self.rng = np.random.RandomState(seed)
        self.steps_per_epoch = max(1, len(X_f16) // batch_size)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        return self

    def __next__(self):
        # primary batch indices
        idx_a = self.rng.choice(
            len(self.X), size=self.batch_size, replace=True, p=self.weights,
        )
        x_a = self.X[idx_a].astype(np.float32)
        y_a = self.Y[idx_a]

        # 3-channel stack (EfficientNet expects RGB)
        x_a = np.stack([x_a, x_a, x_a], axis=-1)

        if self.alpha > 0:
            idx_b = self.rng.permutation(self.batch_size)
            lam = self.rng.beta(self.alpha, self.alpha, size=self.batch_size).astype(np.float32)
            lam = np.maximum(lam, 1.0 - lam)  # bias toward x_a (label coherence)
            lam_x = lam.reshape(-1, 1, 1, 1)
            lam_y = lam.reshape(-1, 1)
            x_b = x_a[idx_b]
            y_b = y_a[idx_b]
            x = lam_x * x_a + (1.0 - lam_x) * x_b
            y = lam_y * y_a + (1.0 - lam_y) * y_b
        else:
            x = x_a
            y = y_a

        if self.augment:
            # tiny additive noise + random small time shift
            if self.rng.rand() < 0.5:
                shift = self.rng.randint(-30, 30)
                x = np.roll(x, shift, axis=2)
            if self.rng.rand() < 0.5:
                noise = self.rng.normal(0.0, 0.005, x.shape).astype(np.float32)
                x = np.clip(x + noise, 0.0, 1.0)

        return x, y


# ---------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------
def build_model(n_classes):
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    base = keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False,
        input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(n_classes, activation="sigmoid")(x)
    model = keras.Model(inputs, out)
    return model, base


# ---------------------------------------------------------------
# CALLBACK: macro-AUC on held-out soundscape val (matches LB metric)
# ---------------------------------------------------------------
class SoundscapeMacroAuc:
    pass  # placeholder for typing, real class defined inside main


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="rebuild cache even if it exists")
    args = parser.parse_args()

    print("=" * 72)
    print("BirdCLEF 2026 strong retraining v40")
    print("=" * 72)

    label_names, label_to_idx = load_labels_layout()
    n_classes = len(label_names)
    print(f"n_classes = {n_classes}")

    print("\n--- step 1: cache focal melspecs ---")
    Xf, Yf = precompute_focal_cache(label_to_idx, n_classes, force=args.force)

    print("\n--- step 2: cache soundscape melspecs ---")
    Xs, Ys, Fs = precompute_soundscape_cache(label_to_idx, n_classes, force=args.force)

    print("\n--- step 3: group split soundscape by FILE ---")
    rng = np.random.RandomState(RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xs_train, Ys_train = Xs[~val_mask], Ys[~val_mask]
    Xs_val,   Ys_val   = Xs[val_mask],  Ys[val_mask]
    print(f"  train soundscape: {len(Xs_train)} from {len(unique_files)-len(val_files)} files")
    print(f"  val   soundscape: {len(Xs_val)}  from {len(val_files)} files")

    print("\n--- step 4: build training pool ---")
    X_train = np.concatenate([Xf, Xs_train], axis=0)
    Y_train = np.concatenate([Yf, Ys_train], axis=0)
    print(f"  total training samples: {len(X_train)} ({len(Xf)} focal + {len(Xs_train)} soundscape)")

    # sample weights -> rarer classes (and soundscape rows) get sampled more
    class_pos = Y_train.sum(axis=0).clip(min=1)
    inv = 1.0 / class_pos
    inv = inv / inv.max()
    # weight per sample = max class-inv across present classes
    sample_w = (Y_train * inv).max(axis=1)
    # boost soundscape rows: they match test distribution
    boost = np.ones(len(X_train), dtype=np.float32)
    boost[len(Xf):] = 3.0
    sample_w = sample_w * boost
    sample_w = np.where(sample_w > 0, sample_w, sample_w[sample_w > 0].mean())
    print(f"  sample_w stats: min={sample_w.min():.3g} max={sample_w.max():.3g} "
          f"mean={sample_w.mean():.3g}")

    print("\n--- step 5: build model ---")
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score
    model, base = build_model(n_classes)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    model.summary(print_fn=lambda s: print("  " + s))

    # 3-channel val stack (no augmentation)
    Xs_val_3 = np.stack([Xs_val, Xs_val, Xs_val], axis=-1).astype(np.float32)

    history_log = []
    best_macro_auc = -np.inf

    def macro_auc(y_true, y_pred):
        aucs = []
        for c in range(y_true.shape[1]):
            if y_true[:, c].sum() == 0:
                continue
            try:
                aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
            except ValueError:
                continue
        return float(np.mean(aucs)) if aucs else float("nan")

    def evaluate_and_log(phase, epoch):
        nonlocal best_macro_auc
        preds = model.predict(Xs_val_3, batch_size=BATCH_SIZE, verbose=0)
        m_auc = macro_auc(Ys_val, preds)
        print(f"  [{phase} ep {epoch}]  soundscape macro-AUC = {m_auc:.4f}")
        history_log.append({
            "phase": phase, "epoch": epoch,
            "soundscape_macro_auc": m_auc,
        })
        if np.isfinite(m_auc) and m_auc > best_macro_auc:
            best_macro_auc = m_auc
            model.save(OUTPUT_MODEL)
            print(f"  >> new best macro-AUC -> saved {OUTPUT_MODEL}")
        with open(OUTPUT_LOG, "w") as f:
            json.dump({
                "best_macro_auc": best_macro_auc,
                "history": history_log,
                "config": {
                    "max_per_class_focal": MAX_PER_CLASS_FOCAL,
                    "max_total_focal": MAX_TOTAL_FOCAL,
                    "soundscape_val_frac": SOUNDSCAPE_VAL_FRAC,
                    "phase1_epochs": PHASE1_EPOCHS,
                    "phase2_epochs": PHASE2_EPOCHS,
                    "phase1_lr": PHASE1_LR,
                    "phase2_lr": PHASE2_LR,
                    "unfreeze_layers": UNFREEZE_LAYERS,
                    "mixup_alpha": MIXUP_ALPHA,
                    "batch_size": BATCH_SIZE,
                    "n_focal": int(len(Xf)),
                    "n_soundscape_train": int(len(Xs_train)),
                    "n_soundscape_val": int(len(Xs_val)),
                },
            }, f, indent=2)

    print("\n--- step 6: phase 1 (head only, base FROZEN) ---")
    train_seq = MixupSequence(
        X_train, Y_train, BATCH_SIZE, sample_w, MIXUP_ALPHA,
        augment=True, seed=RANDOM_SEED,
    )
    for epoch in range(1, PHASE1_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(train_seq)):
            x_b, y_b = next(train_seq)
            model.train_on_batch(x_b, y_b)
            if step % 50 == 0:
                print(f"  P1 ep{epoch} step {step}/{len(train_seq)}  "
                      f"({time.time()-ep_t0:.0f}s)")
        evaluate_and_log("P1", epoch)
        print(f"  phase1 ep{epoch} took {time.time()-ep_t0:.0f}s")

    print("\n--- step 7: phase 2 (unfreeze top layers) ---")
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    # Keep BatchNorm frozen (Chollet 8.3.2)
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    n_train_layers = sum(1 for l in base.layers if l.trainable)
    print(f"  unfrozen base layers: {n_train_layers}")
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    train_seq2 = MixupSequence(
        X_train, Y_train, BATCH_SIZE, sample_w, MIXUP_ALPHA,
        augment=True, seed=RANDOM_SEED + 1,
    )
    for epoch in range(1, PHASE2_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(train_seq2)):
            x_b, y_b = next(train_seq2)
            model.train_on_batch(x_b, y_b)
            if step % 50 == 0:
                print(f"  P2 ep{epoch} step {step}/{len(train_seq2)}  "
                      f"({time.time()-ep_t0:.0f}s)")
        evaluate_and_log("P2", epoch)
        print(f"  phase2 ep{epoch} took {time.time()-ep_t0:.0f}s")

    print("\n=== DONE ===")
    print(f"best soundscape macro-AUC: {best_macro_auc:.4f}")
    print(f"model saved to: {OUTPUT_MODEL}")
    print(f"log written to: {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
