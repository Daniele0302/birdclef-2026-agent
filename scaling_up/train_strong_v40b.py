"""
train_strong_v40b.py - Plan B with much stronger soundscape bias.

Differences vs v40:
  - Focal cap reduced (4000 instead of 8000) to lower domain dominance
  - Soundscape boost weight raised 3x -> 10x
  - Mixup alpha 0.2 -> 0.4
  - Lower lr (1e-4 -> 5e-5) so we don't blow past the optimum in a single
    epoch (v40 converged after one epoch then overfit)
  - Phase 1 reduced (4 -> 2 epochs)
  - Phase 2 unfreezes 50 layers (instead of 30) but at 2e-5
  - Re-uses the same on-disk caches as v40 if present (so no re-loading audio)

Run:
    .venv/bin/python train_strong_v40b.py
"""

import os
import sys
import json
import time
import importlib

# Reuse the v40 module so we don't duplicate the audio + caching code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_strong_v40 as v40

# Override config knobs IN PLACE before main() runs.
v40.MAX_PER_CLASS_FOCAL = 30
v40.MAX_TOTAL_FOCAL     = 4000
v40.PHASE1_EPOCHS       = 2
v40.PHASE2_EPOCHS       = 6
v40.PHASE1_LR           = 1e-4
v40.PHASE2_LR           = 2e-5
v40.UNFREEZE_LAYERS     = 50
v40.MIXUP_ALPHA         = 0.4
v40.BATCH_SIZE          = 16

v40.OUTPUT_MODEL = os.path.join(v40.PROJECT_DIR, "model_v40b_strong.keras")
v40.OUTPUT_LOG   = os.path.join(v40.PROJECT_DIR, "train_v40b_log.json")

# Use a separate focal cache because MAX_TOTAL_FOCAL changed
v40.CACHE_FOCAL_X = os.path.join(v40.CACHE_DIR, "specs_focal_v40b.npy")
v40.CACHE_FOCAL_Y = os.path.join(v40.CACHE_DIR, "labels_focal_v40b.npy")
# Soundscape cache is identical -> reuse v40 file

# Override sample-weight builder via monkey-patch to lift soundscape boost
_old_main = v40.main


def main():
    # patch boost factor by editing the script's local boost logic
    # easiest: edit at runtime via monkey-patching the constants.
    # The boost factor is hardcoded in main(), so we re-implement main() with
    # a different boost.
    import numpy as np
    label_names, label_to_idx = v40.load_labels_layout()
    n_classes = len(label_names)

    print("=" * 72)
    print("BirdCLEF 2026 strong retraining v40b - aggressive plan B")
    print(f"  focal cap   : {v40.MAX_TOTAL_FOCAL} (per class {v40.MAX_PER_CLASS_FOCAL})")
    print(f"  unfreeze    : {v40.UNFREEZE_LAYERS}")
    print(f"  mixup alpha : {v40.MIXUP_ALPHA}")
    print(f"  P1/P2 lr    : {v40.PHASE1_LR} / {v40.PHASE2_LR}")
    print("=" * 72)

    Xf, Yf = v40.precompute_focal_cache(label_to_idx, n_classes, force=False)
    Xs, Ys, Fs = v40.precompute_soundscape_cache(label_to_idx, n_classes, force=False)

    rng = np.random.RandomState(v40.RANDOM_SEED)
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
    boost[len(Xf):] = 10.0  # Plan B: 10x soundscape
    sample_w = sample_w * boost
    sample_w = np.where(sample_w > 0, sample_w, sample_w[sample_w > 0].mean())

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score
    model, base = v40.build_model(n_classes)
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=v40.PHASE1_LR, clipnorm=1.0),
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

    history_log = []
    best = -np.inf
    patience_used = 0
    EARLY_STOP_PATIENCE = 3

    def evaluate(phase, epoch):
        nonlocal best, patience_used
        preds = model.predict(Xs_val_3, batch_size=v40.BATCH_SIZE, verbose=0)
        a = macro_auc(Ys_val, preds)
        print(f"  [{phase} ep {epoch}] soundscape macro-AUC = {a:.4f}")
        history_log.append({"phase": phase, "epoch": epoch, "soundscape_macro_auc": a})
        improved = np.isfinite(a) and a > best
        if improved:
            best = a
            patience_used = 0
            model.save(v40.OUTPUT_MODEL)
            print(f"  >> new best -> saved {v40.OUTPUT_MODEL}")
        else:
            patience_used += 1
            print(f"  no improvement (patience {patience_used}/{EARLY_STOP_PATIENCE})")
        with open(v40.OUTPUT_LOG, "w") as f:
            json.dump({"best_macro_auc": best, "history": history_log}, f, indent=2)
        return improved

    train_seq = v40.MixupSequence(
        X_train, Y_train, v40.BATCH_SIZE, sample_w, v40.MIXUP_ALPHA,
        augment=True, seed=v40.RANDOM_SEED,
    )
    print("\n--- Phase 1 ---")
    for epoch in range(1, v40.PHASE1_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(train_seq)):
            x_b, y_b = next(train_seq)
            model.train_on_batch(x_b, y_b)
            if step % 50 == 0:
                print(f"  P1 ep{epoch} step {step}/{len(train_seq)}")
        evaluate("P1", epoch)
        print(f"  P1 ep{epoch} took {time.time()-ep_t0:.0f}s")
        if patience_used >= EARLY_STOP_PATIENCE:
            print("  early stop")
            break

    print("\n--- Phase 2 ---")
    base.trainable = True
    for layer in base.layers[:-v40.UNFREEZE_LAYERS]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    n_train_layers = sum(1 for l in base.layers if l.trainable)
    print(f"unfrozen base layers: {n_train_layers}")
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=v40.PHASE2_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    train_seq2 = v40.MixupSequence(
        X_train, Y_train, v40.BATCH_SIZE, sample_w, v40.MIXUP_ALPHA,
        augment=True, seed=v40.RANDOM_SEED + 1,
    )
    patience_used = 0
    for epoch in range(1, v40.PHASE2_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(train_seq2)):
            x_b, y_b = next(train_seq2)
            model.train_on_batch(x_b, y_b)
            if step % 50 == 0:
                print(f"  P2 ep{epoch} step {step}/{len(train_seq2)}")
        evaluate("P2", epoch)
        print(f"  P2 ep{epoch} took {time.time()-ep_t0:.0f}s")
        if patience_used >= EARLY_STOP_PATIENCE:
            print("  early stop")
            break

    print("\n=== DONE ===  best macro-AUC", best)
    print("model:", v40.OUTPUT_MODEL)


if __name__ == "__main__":
    main()
