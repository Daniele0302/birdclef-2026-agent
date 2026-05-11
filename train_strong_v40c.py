"""
train_strong_v40c.py - Plan C, sibling of v40 with maximum independence.

Goal: produce a third model whose errors are uncorrelated with v40 and v40b
so the ensemble keeps growing. Greedy ensemble showed v40b adds value
*because* its individual ranking is shifted from v40, even though it scores
worse alone. We want another such "different lens" model.

Key differences from v40:
  - Two random 5s WINDOWS extracted per focal recording (so the focal cache
    holds 8000 specs from 4000 unique recordings -> different audio content
    seen vs v40 which always used the first 5s).
  - Mixup alpha = 0.3 (between v40's 0.2 and v40b's 0.4)
  - 40 layers unfrozen in P2 (between v40's 30 and v40b's 50)
  - Soundscape boost = 5x (between v40's 3x and v40b's 10x)
  - Different RNG for augmentation seed (just rng numerics, NOT val split,
    which must match v40 / v40b for honest comparison)
  - Same soundscape val split (seed 20260507) -> directly comparable

Run:
    .venv/bin/python train_strong_v40c.py
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_strong_v40 as v40


# Override config (must do this BEFORE any function default eval — function
# defaults are bound at import; we use function args explicitly below)
PROJECT_DIR = v40.PROJECT_DIR
DATA_DIR = v40.DATA_DIR
CACHE_DIR = v40.CACHE_DIR
N_RECORDINGS_FOCAL = 4000
N_WINDOWS_PER_REC  = 2
SOUNDSCAPE_BOOST   = 5.0
MIXUP_ALPHA        = 0.3
PHASE1_EPOCHS      = 3
PHASE2_EPOCHS      = 5
PHASE1_LR          = 2e-4
PHASE2_LR          = 3e-5
UNFREEZE_LAYERS    = 40
BATCH_SIZE         = 16
RANDOM_SEED        = 20260507  # MUST match v40 / v40b for fair val
AUG_SEED           = 4242      # different RNG branch for augmentation diversity

CACHE_FOCAL_X = os.path.join(CACHE_DIR, "specs_focal_v40c.npy")
CACHE_FOCAL_Y = os.path.join(CACHE_DIR, "labels_focal_v40c.npy")

OUTPUT_MODEL = os.path.join(PROJECT_DIR, "model_v40c_strong.keras")
OUTPUT_LOG   = os.path.join(PROJECT_DIR, "train_v40c_log.json")


def precompute_focal_multiwindow(label_to_idx, n_classes, force=False):
    if (not force) and os.path.exists(CACHE_FOCAL_X) and os.path.exists(CACHE_FOCAL_Y):
        X = np.load(CACHE_FOCAL_X)
        Y = np.load(CACHE_FOCAL_Y)
        print(f"[cache hit] focal v40c: X={X.shape} Y={Y.shape}")
        return X, Y

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    rng_pick = np.random.RandomState(RANDOM_SEED + 1)
    rng_offset = np.random.RandomState(AUG_SEED)

    # Stratified pick of 4000 unique recordings
    parts = []
    for label, g in train_df.groupby("primary_label"):
        n = min(len(g), 25)  # max 25 per class
        parts.append(g.sample(n=n, random_state=rng_pick.randint(2**31 - 1)))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > N_RECORDINGS_FOCAL:
        out = out.sample(n=N_RECORDINGS_FOCAL, random_state=RANDOM_SEED + 2).reset_index(drop=True)
    print(f"[focal v40c] {len(out)} unique recordings, {N_WINDOWS_PER_REC} windows each")

    import librosa
    X_list, Y_list = [], []
    skipped = 0
    t0 = time.time()
    for i, row in enumerate(out.itertuples(index=False), 1):
        if i % 200 == 0:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-3)
            print(f"  rec {i}/{len(out)}  rate={rate:.1f}/s  eta={(len(out)-i)/rate/60:.1f}min")
        path = os.path.join(DATA_DIR, "train_audio", row.filename)
        if not os.path.exists(path):
            skipped += 1
            continue
        try:
            # Load full file
            y_full, _ = librosa.load(path, sr=v40.SAMPLE_RATE, mono=True)
            target = int(v40.SAMPLE_RATE * v40.DURATION)
            label_vec = v40.focal_label_vector(
                row.primary_label, row.secondary_labels, label_to_idx, n_classes,
            )
            for k in range(N_WINDOWS_PER_REC):
                if len(y_full) <= target:
                    y = np.pad(y_full, (0, target - len(y_full)), mode="constant")
                else:
                    max_start = len(y_full) - target
                    if k == 0:
                        # First window: deterministic (start at 0)
                        start = 0
                    else:
                        start = rng_offset.randint(0, max_start + 1)
                    y = y_full[start:start + target]
                mel = v40.make_melspec(y)
                X_list.append(mel)
                Y_list.append(label_vec)
        except Exception as exc:
            skipped += 1
            print(f"  skip {path}: {exc}")

    X = np.stack(X_list, axis=0).astype(np.float16)
    Y = np.stack(Y_list, axis=0).astype(np.float32)
    print(f"[focal v40c] kept {len(X)} from {len(out) - skipped} recordings in {time.time()-t0:.1f}s")
    np.save(CACHE_FOCAL_X, X)
    np.save(CACHE_FOCAL_Y, Y)
    return X, Y


def main():
    label_names, label_to_idx = v40.load_labels_layout()
    n_classes = len(label_names)

    print("=" * 72)
    print("BirdCLEF 2026 v40c training (multi-window focal + 5x soundscape boost)")
    print(f"  N_RECORDINGS_FOCAL = {N_RECORDINGS_FOCAL}")
    print(f"  N_WINDOWS_PER_REC  = {N_WINDOWS_PER_REC}")
    print(f"  SOUNDSCAPE_BOOST   = {SOUNDSCAPE_BOOST}")
    print(f"  MIXUP_ALPHA        = {MIXUP_ALPHA}")
    print(f"  UNFREEZE_LAYERS    = {UNFREEZE_LAYERS}")
    print("=" * 72)

    Xf, Yf = precompute_focal_multiwindow(label_to_idx, n_classes, force=False)
    Xs, Ys, Fs = v40.precompute_soundscape_cache(label_to_idx, n_classes, force=False)

    rng = np.random.RandomState(RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * v40.SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xs_train, Ys_train = Xs[~val_mask], Ys[~val_mask]
    Xs_val,   Ys_val   = Xs[val_mask],  Ys[val_mask]
    print(f"train soundscape: {len(Xs_train)} from {len(unique_files)-len(val_files)} files")
    print(f"val   soundscape: {len(Xs_val)}  from {len(val_files)} files")

    X_train = np.concatenate([Xf, Xs_train], axis=0)
    Y_train = np.concatenate([Yf, Ys_train], axis=0)
    print(f"total training: {len(X_train)} ({len(Xf)} focal + {len(Xs_train)} soundscape)")

    class_pos = Y_train.sum(axis=0).clip(min=1)
    inv = 1.0 / class_pos
    inv = inv / inv.max()
    sample_w = (Y_train * inv).max(axis=1)
    boost = np.ones(len(X_train), dtype=np.float32)
    boost[len(Xf):] = SOUNDSCAPE_BOOST
    sample_w = sample_w * boost
    sample_w = np.where(sample_w > 0, sample_w, sample_w[sample_w > 0].mean())

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score
    model, base = v40.build_model(n_classes)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )

    Xs_val_3 = np.stack([Xs_val, Xs_val, Xs_val], axis=-1).astype(np.float32)

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

    history = []
    best = -np.inf
    patience = 0
    PATIENCE_MAX = 3

    def evaluate(phase, epoch):
        nonlocal best, patience
        preds = model.predict(Xs_val_3, batch_size=BATCH_SIZE, verbose=0)
        a = macro_auc(Ys_val, preds)
        print(f"  [{phase} ep {epoch}] soundscape macro-AUC = {a:.4f}")
        history.append({"phase": phase, "epoch": epoch, "soundscape_macro_auc": a})
        if np.isfinite(a) and a > best:
            best = a
            patience = 0
            model.save(OUTPUT_MODEL)
            print(f"  >> new best -> saved {OUTPUT_MODEL}")
        else:
            patience += 1
            print(f"  no improvement (patience {patience}/{PATIENCE_MAX})")
        with open(OUTPUT_LOG, "w") as f:
            json.dump({"best_macro_auc": best, "history": history}, f, indent=2)

    train_seq = v40.MixupSequence(
        X_train, Y_train, BATCH_SIZE, sample_w, MIXUP_ALPHA,
        augment=True, seed=AUG_SEED,
    )
    print("\n--- Phase 1 ---")
    for epoch in range(1, PHASE1_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(train_seq)):
            x_b, y_b = next(train_seq)
            model.train_on_batch(x_b, y_b)
            if step % 100 == 0:
                print(f"  P1 ep{epoch} step {step}/{len(train_seq)}")
        evaluate("P1", epoch)
        print(f"  P1 ep{epoch} took {time.time()-ep_t0:.0f}s")
        if patience >= PATIENCE_MAX:
            print("  early stop")
            break

    print("\n--- Phase 2 ---")
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    n_train_layers = sum(1 for l in base.layers if l.trainable)
    print(f"unfrozen base layers: {n_train_layers}")
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    train_seq2 = v40.MixupSequence(
        X_train, Y_train, BATCH_SIZE, sample_w, MIXUP_ALPHA,
        augment=True, seed=AUG_SEED + 1,
    )
    patience = 0
    for epoch in range(1, PHASE2_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(train_seq2)):
            x_b, y_b = next(train_seq2)
            model.train_on_batch(x_b, y_b)
            if step % 100 == 0:
                print(f"  P2 ep{epoch} step {step}/{len(train_seq2)}")
        evaluate("P2", epoch)
        print(f"  P2 ep{epoch} took {time.time()-ep_t0:.0f}s")
        if patience >= PATIENCE_MAX:
            print("  early stop")
            break

    print("\n=== DONE ===  best macro-AUC", best)
    print("model:", OUTPUT_MODEL)


if __name__ == "__main__":
    main()
