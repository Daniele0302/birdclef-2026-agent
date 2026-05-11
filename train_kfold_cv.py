"""
train_kfold_cv.py - 5-fold cross-validation training for true ensemble
diversity. Each fold has DIFFERENT soundscape val files, so each model
is trained on a different subset of the data and learns different
features.

Files saved:
    model_fold0.keras ... model_fold4.keras
    train_kfold_log.json (per-fold val AUCs + OOF macro-AUC)

Run:
    .venv/bin/python train_kfold_cv.py
"""
import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_strong_v40 as v40

K = 5
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 4
UNFREEZE = 30
MIXUP = 0.3
BOOST = 5.0
SHARED_SEED = 999  # deterministic fold partition


def macro_auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    label_names, label_to_idx = v40.load_labels_layout()
    n_classes = len(label_names)

    print("=" * 72)
    print(f"BirdCLEF 2026 {K}-fold CV training")
    print("=" * 72)

    Xf, Yf = v40.precompute_focal_cache(label_to_idx, n_classes, force=False)
    Xs, Ys, Fs = v40.precompute_soundscape_cache(label_to_idx, n_classes, force=False)

    rng = np.random.RandomState(SHARED_SEED)
    unique_files = np.unique(Fs)
    rng.shuffle(unique_files)
    folds = np.array_split(unique_files, K)
    print(f"unique soundscape files: {len(unique_files)}")
    for i, f in enumerate(folds):
        print(f"  fold{i}: {len(f)} files")

    all_log = []
    oof_preds = np.zeros((len(Xs), n_classes), dtype=np.float32)
    oof_filled = np.zeros(len(Xs), dtype=bool)

    for fold_idx in range(K):
        print(f"\n========================== FOLD {fold_idx} ==========================")
        val_files = set(folds[fold_idx])
        val_mask = np.array([f in val_files for f in Fs])
        Xs_train, Ys_train = Xs[~val_mask], Ys[~val_mask]
        Xs_val, Ys_val = Xs[val_mask], Ys[val_mask]
        print(f"  train soundscape: {len(Xs_train)}  val: {len(Xs_val)}")

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
        tf.keras.utils.set_random_seed(fold_idx * 17 + 1)

        model, base = v40.build_model(n_classes)
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=v40.PHASE1_LR, clipnorm=1.0),
            metrics=[keras.metrics.AUC(name="auc")],
        )

        Xs_val_3 = np.stack([Xs_val, Xs_val, Xs_val], axis=-1).astype(np.float32)
        out_path = os.path.join(v40.PROJECT_DIR, f"model_fold{fold_idx}.keras")
        best_auc = -np.inf
        best_pred = None
        history = []

        def evaluate(phase, epoch):
            nonlocal best_auc, best_pred
            preds = model.predict(Xs_val_3, batch_size=v40.BATCH_SIZE, verbose=0)
            a = macro_auc(Ys_val, preds)
            print(f"  [fold{fold_idx} {phase} ep{epoch}] val macro-AUC = {a:.4f}")
            history.append({"phase": phase, "epoch": epoch, "auc": a})
            if np.isfinite(a) and a > best_auc:
                best_auc = a
                best_pred = preds
                model.save(out_path)
                print(f"  >> new best -> saved {out_path}")

        seq = v40.MixupSequence(
            X_train, Y_train, v40.BATCH_SIZE, sample_w, MIXUP,
            augment=True, seed=fold_idx * 100 + 1,
        )
        for epoch in range(1, PHASE1_EPOCHS + 1):
            ep_t0 = time.time()
            for step in range(len(seq)):
                x_b, y_b = next(seq)
                model.train_on_batch(x_b, y_b)
                if step % 100 == 0:
                    print(f"  fold{fold_idx} P1 ep{epoch} step {step}/{len(seq)}")
            evaluate("P1", epoch)

        base.trainable = True
        for layer in base.layers[:-UNFREEZE]:
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
            X_train, Y_train, v40.BATCH_SIZE, sample_w, MIXUP,
            augment=True, seed=fold_idx * 100 + 2,
        )
        for epoch in range(1, PHASE2_EPOCHS + 1):
            ep_t0 = time.time()
            for step in range(len(seq2)):
                x_b, y_b = next(seq2)
                model.train_on_batch(x_b, y_b)
                if step % 100 == 0:
                    print(f"  fold{fold_idx} P2 ep{epoch} step {step}/{len(seq2)}")
            evaluate("P2", epoch)

        # Store OOF predictions for this fold
        val_indices = np.where(val_mask)[0]
        if best_pred is not None:
            oof_preds[val_indices] = best_pred
            oof_filled[val_indices] = True
        all_log.append({
            "fold": fold_idx,
            "best_val_auc": best_auc,
            "history": history,
            "n_val": int(val_mask.sum()),
        })

        # Free memory before next fold
        keras.backend.clear_session()

    # OOF macro-AUC across all soundscape windows
    print("\n" + "=" * 72)
    if oof_filled.all():
        oof_auc = macro_auc(Ys, oof_preds)
        print(f"OOF (out-of-fold) soundscape macro-AUC = {oof_auc:.4f}")
    else:
        print(f"OOF coverage incomplete: {oof_filled.sum()}/{len(oof_filled)}")
        oof_auc = float("nan")

    out = {
        "n_folds": K,
        "oof_macro_auc": oof_auc,
        "folds": all_log,
    }
    out_path = os.path.join(v40.PROJECT_DIR, "train_kfold_log.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
