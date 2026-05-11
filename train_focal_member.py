"""
train_focal_member.py - Train a v40-style model with FOCAL LOSS instead of
binary cross-entropy. Focal loss down-weights confident predictions and
focuses gradient on hard examples. The loss landscape is qualitatively
different from BCE so the resulting model often has uncorrelated errors,
which is exactly what a deep ensemble needs.

Same val split as v40 / v40_s2 / etc. Reuses cached arrays.

Run:
    .venv/bin/python train_focal_member.py
"""
import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_strong_v40 as v40

OUT_PATH = os.path.join(v40.PROJECT_DIR, "model_v40_focal.keras")
LOG_PATH = os.path.join(v40.PROJECT_DIR, "train_v40_focal_log.json")

# Focal loss params (Lin et al. 2017): higher gamma -> more emphasis on hard
GAMMA = 2.0
ALPHA = 0.25  # class balance factor
SEED = 7
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 4
UNFREEZE = 30
MIXUP = 0.3
BOOST = 5.0


def make_focal_loss(gamma=2.0, alpha=0.25):
    import tensorflow as tf
    def focal_loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce_pos = -tf.math.log(y_pred)
        ce_neg = -tf.math.log(1.0 - y_pred)
        # focal weights
        w_pos = alpha * tf.pow(1.0 - y_pred, gamma)
        w_neg = (1.0 - alpha) * tf.pow(y_pred, gamma)
        loss = y_true * w_pos * ce_pos + (1.0 - y_true) * w_neg * ce_neg
        return tf.reduce_mean(loss)
    return focal_loss


def main():
    label_names, label_to_idx = v40.load_labels_layout()
    n_classes = len(label_names)

    print("=" * 72)
    print("BirdCLEF 2026 v40_focal training (focal loss, gamma=2)")
    print("=" * 72)

    Xf, Yf = v40.precompute_focal_cache(label_to_idx, n_classes, force=False)
    Xs, Ys, Fs = v40.precompute_soundscape_cache(label_to_idx, n_classes, force=False)

    rng = np.random.RandomState(v40.RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * v40.SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xs_train, Ys_train = Xs[~val_mask], Ys[~val_mask]
    Xs_val, Ys_val = Xs[val_mask], Ys[val_mask]

    X_train = np.concatenate([Xf, Xs_train], axis=0)
    Y_train = np.concatenate([Yf, Ys_train], axis=0)

    class_pos = Y_train.sum(axis=0).clip(min=1)
    inv = 1.0 / class_pos
    inv = inv / inv.max()
    sample_w = (Y_train * inv).max(axis=1)
    boost = np.ones(len(X_train), dtype=np.float32)
    boost[len(Xf):] = BOOST
    sample_w = sample_w * boost
    sample_w = np.where(sample_w > 0, sample_w, sample_w[sample_w > 0].mean())

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score
    tf.keras.utils.set_random_seed(SEED)

    model, base = v40.build_model(n_classes)
    focal = make_focal_loss(GAMMA, ALPHA)
    model.compile(
        loss=focal,
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

    history = []
    best = -np.inf

    def evaluate(phase, epoch):
        nonlocal best
        preds = model.predict(Xs_val_3, batch_size=v40.BATCH_SIZE, verbose=0)
        a = macro_auc(Ys_val, preds)
        print(f"  [{phase} ep {epoch}] macro-AUC = {a:.4f}")
        history.append({"phase": phase, "epoch": epoch, "auc": a})
        if np.isfinite(a) and a > best:
            best = a
            model.save(OUT_PATH)
            print(f"  >> new best -> saved {OUT_PATH}")
        with open(LOG_PATH, "w") as f:
            json.dump({"best_auc": best, "history": history,
                       "loss": "focal", "gamma": GAMMA, "alpha": ALPHA,
                       "seed": SEED}, f, indent=2)

    seq = v40.MixupSequence(
        X_train, Y_train, v40.BATCH_SIZE, sample_w, MIXUP,
        augment=True, seed=SEED * 100 + 1,
    )
    print("\n--- Phase 1 ---")
    for epoch in range(1, PHASE1_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(seq)):
            x_b, y_b = next(seq)
            model.train_on_batch(x_b, y_b)
            if step % 100 == 0:
                print(f"  P1 ep{epoch} step {step}/{len(seq)}")
        evaluate("P1", epoch)
        print(f"  P1 ep{epoch} took {time.time()-ep_t0:.0f}s")

    print("\n--- Phase 2 ---")
    base.trainable = True
    for layer in base.layers[:-UNFREEZE]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        loss=focal,
        optimizer=keras.optimizers.Adam(learning_rate=v40.PHASE2_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    seq2 = v40.MixupSequence(
        X_train, Y_train, v40.BATCH_SIZE, sample_w, MIXUP,
        augment=True, seed=SEED * 100 + 2,
    )
    for epoch in range(1, PHASE2_EPOCHS + 1):
        ep_t0 = time.time()
        for step in range(len(seq2)):
            x_b, y_b = next(seq2)
            model.train_on_batch(x_b, y_b)
            if step % 100 == 0:
                print(f"  P2 ep{epoch} step {step}/{len(seq2)}")
        evaluate("P2", epoch)
        print(f"  P2 ep{epoch} took {time.time()-ep_t0:.0f}s")

    print(f"\n=== focal DONE === best macro-AUC {best:.4f}")


if __name__ == "__main__":
    main()
