"""
train_seed_member.py - Train one v40-style model with a chosen RNG seed.

Usage:
    .venv/bin/python train_seed_member.py --seed 2 --out model_v40_s2.keras
    .venv/bin/python train_seed_member.py --seed 3 --out model_v40_s3.keras

Each model:
  - same v40 hyperparams (mixup 0.3, unfreeze 30 layers, soundscape 5x boost)
  - different sample-order, augmentation, mixup RNG -> different convergence
  - same val split (RANDOM_SEED 20260507) so all models evaluated on the SAME
    held-out soundscape windows
  - reuses cached focal+soundscape arrays from v40 (no re-loading audio)

Used for deep-ensemble multi-seed members. Greedy weighting picks the mix.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_strong_v40 as v40


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True,
                        help="member seed (used for sampling/aug RNG)")
    parser.add_argument("--out", type=str, required=True,
                        help="output .keras filename (relative or absolute)")
    parser.add_argument("--phase1", type=int, default=3)
    parser.add_argument("--phase2", type=int, default=4)
    parser.add_argument("--mixup", type=float, default=0.3)
    parser.add_argument("--unfreeze", type=int, default=30)
    parser.add_argument("--boost", type=float, default=5.0)
    args = parser.parse_args()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(v40.PROJECT_DIR, out_path)
    log_path = out_path.replace(".keras", "_log.json")

    label_names, label_to_idx = v40.load_labels_layout()
    n_classes = len(label_names)

    print(f"\n{'='*72}\nDeep ensemble member: seed={args.seed} -> {out_path}\n{'='*72}")

    # Reuse v40 cached arrays (we don't change focal sampling)
    Xf, Yf = v40.precompute_focal_cache(label_to_idx, n_classes, force=False)
    Xs, Ys, Fs = v40.precompute_soundscape_cache(label_to_idx, n_classes, force=False)

    # Same group split as v40 / v40b -> directly comparable
    rng_split = np.random.RandomState(v40.RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * v40.SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng_split.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xs_train, Ys_train = Xs[~val_mask], Ys[~val_mask]
    Xs_val,   Ys_val   = Xs[val_mask],  Ys[val_mask]

    X_train = np.concatenate([Xf, Xs_train], axis=0)
    Y_train = np.concatenate([Yf, Ys_train], axis=0)

    class_pos = Y_train.sum(axis=0).clip(min=1)
    inv = 1.0 / class_pos
    inv = inv / inv.max()
    sample_w = (Y_train * inv).max(axis=1)
    boost = np.ones(len(X_train), dtype=np.float32)
    boost[len(Xf):] = args.boost
    sample_w = sample_w * boost
    sample_w = np.where(sample_w > 0, sample_w, sample_w[sample_w > 0].mean())

    # MEMBER-SPECIFIC RNGs:
    #   - sampling/mixup uses args.seed
    #   - val split is fixed (v40.RANDOM_SEED) so all members are comparable
    member_seed_a = args.seed * 100 + 1
    member_seed_b = args.seed * 100 + 2

    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import roc_auc_score
    tf.keras.utils.set_random_seed(args.seed)

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

    history = []
    best = -np.inf

    def evaluate(phase, epoch):
        nonlocal best
        preds = model.predict(Xs_val_3, batch_size=v40.BATCH_SIZE, verbose=0)
        a = macro_auc(Ys_val, preds)
        print(f"  [{phase} ep {epoch}] soundscape macro-AUC = {a:.4f}")
        history.append({"phase": phase, "epoch": epoch, "auc": a})
        if np.isfinite(a) and a > best:
            best = a
            model.save(out_path)
            print(f"  >> new best -> saved {out_path}")
        with open(log_path, "w") as f:
            json.dump({"best_auc": best, "history": history, "seed": args.seed}, f, indent=2)

    seq = v40.MixupSequence(
        X_train, Y_train, v40.BATCH_SIZE, sample_w, args.mixup,
        augment=True, seed=member_seed_a,
    )
    print("\n--- Phase 1 (head only) ---")
    for epoch in range(1, args.phase1 + 1):
        ep_t0 = time.time()
        for step in range(len(seq)):
            x_b, y_b = next(seq)
            model.train_on_batch(x_b, y_b)
            if step % 100 == 0:
                print(f"  P1 ep{epoch} step {step}/{len(seq)}")
        evaluate("P1", epoch)
        print(f"  P1 ep{epoch} took {time.time()-ep_t0:.0f}s")

    print("\n--- Phase 2 (unfreeze top layers) ---")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=v40.PHASE2_LR, clipnorm=1.0),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    seq2 = v40.MixupSequence(
        X_train, Y_train, v40.BATCH_SIZE, sample_w, args.mixup,
        augment=True, seed=member_seed_b,
    )
    for epoch in range(1, args.phase2 + 1):
        ep_t0 = time.time()
        for step in range(len(seq2)):
            x_b, y_b = next(seq2)
            model.train_on_batch(x_b, y_b)
            if step % 100 == 0:
                print(f"  P2 ep{epoch} step {step}/{len(seq2)}")
        evaluate("P2", epoch)
        print(f"  P2 ep{epoch} took {time.time()-ep_t0:.0f}s")

    print(f"\n=== member seed={args.seed} DONE ===  best macro-AUC {best:.4f}")


if __name__ == "__main__":
    main()
