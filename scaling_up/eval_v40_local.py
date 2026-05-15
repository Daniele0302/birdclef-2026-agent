"""
eval_v40_local.py - Local evaluation on the held-out soundscape windows.

Uses the cached arrays produced by train_strong_v40.py:
    cache/specs_soundscape_v40.npy
    cache/labels_soundscape_v40.npy
    cache/soundscape_files_v40.npy

Reproduces the SAME group-by-FILE split used during training so the
validation set seen here is identical. Reports per-model macro-ROC-AUC and
sweeps ensemble weights to find the best combination.

The macro-AUC reported here is over held-out soundscape windows, which is
the closest local proxy we have for the Kaggle hidden-test metric.

Run:
    .venv/bin/python eval_v40_local.py
"""

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

# Script lives in scaling_up/; PROJECT_DIR points to the repo root.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")

CACHE_SC_X = os.path.join(CACHE_DIR, "specs_soundscape_v40.npy")
CACHE_SC_Y = os.path.join(CACHE_DIR, "labels_soundscape_v40.npy")
CACHE_SC_F = os.path.join(CACHE_DIR, "soundscape_files_v40.npy")

MODEL_EXP78 = os.path.join(PROJECT_DIR, "exp78_full_model.keras")
MODEL_V40   = os.path.join(PROJECT_DIR, "model_v40_strong.keras")

SOUNDSCAPE_VAL_FRAC = 0.2
RANDOM_SEED = 20260507
BATCH_SIZE = 32


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


def load_keras_with_patch(path):
    """Loads a .keras file, stripping BatchNorm 'renorm' keys that some
    Keras versions reject. Returns the loaded model.
    """
    import tensorflow as tf
    from tensorflow import keras
    import zipfile, shutil, json as _json

    BAD = ("renorm", "renorm_clipping", "renorm_momentum", "quantization_config")

    def strip(o):
        if isinstance(o, dict):
            for k in BAD:
                o.pop(k, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for it in o:
                strip(it)

    tag = os.path.basename(path).replace(".keras", "")
    tmp = os.path.join(PROJECT_DIR, f".patch_tmp_{tag}")
    out = os.path.join(PROJECT_DIR, f".patched_{tag}.keras")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    cfg_p = os.path.join(tmp, "config.json")
    with open(cfg_p) as f:
        cfg = _json.load(f)
    strip(cfg)
    with open(cfg_p, "w") as f:
        _json.dump(cfg, f)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(tmp):
            for fn in files:
                full = os.path.join(root, fn)
                z.write(full, os.path.relpath(full, tmp))
    shutil.rmtree(tmp)
    return keras.models.load_model(out, compile=False, safe_mode=False)


def main():
    print("Loading cached soundscape arrays...")
    Xs = np.load(CACHE_SC_X).astype(np.float32)  # (N, 64, 626)
    Ys = np.load(CACHE_SC_Y)                     # (N, 234)
    Fs = np.load(CACHE_SC_F, allow_pickle=True)  # (N,) of filename strings

    rng = np.random.RandomState(RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xv, Yv = Xs[val_mask], Ys[val_mask]
    print(f"Held-out soundscape windows: {len(Xv)} from {len(val_files)} files")

    Xv3 = np.stack([Xv, Xv, Xv], axis=-1).astype(np.float32)

    results = {}

    if os.path.exists(MODEL_EXP78):
        print("\nLoading exp78 ...")
        m = load_keras_with_patch(MODEL_EXP78)
        preds = m.predict(Xv3, batch_size=BATCH_SIZE, verbose=0)
        a = macro_auc(Yv, preds)
        print(f"  exp78  macro-AUC = {a:.4f}")
        results["exp78"] = (a, preds)

    if os.path.exists(MODEL_V40):
        print("\nLoading v40_strong ...")
        m = load_keras_with_patch(MODEL_V40)
        preds = m.predict(Xv3, batch_size=BATCH_SIZE, verbose=0)
        a = macro_auc(Yv, preds)
        print(f"  v40    macro-AUC = {a:.4f}")
        results["v40"] = (a, preds)

    if "exp78" in results and "v40" in results:
        print("\nEnsemble weight sweep (geometric mean):")
        p_exp = results["exp78"][1]
        p_v40 = results["v40"][1]
        eps = 1e-7
        best = (-1.0, None)
        for w_v40 in np.linspace(0.0, 1.0, 11):
            w_exp = 1.0 - w_v40
            log_mix = (
                w_exp * np.log(np.clip(p_exp, eps, 1.0))
                + w_v40 * np.log(np.clip(p_v40, eps, 1.0))
            )
            mix = np.exp(log_mix)
            a = macro_auc(Yv, mix)
            print(f"  exp78 {w_exp:.2f} + v40 {w_v40:.2f}  -> macro-AUC = {a:.4f}")
            if a > best[0]:
                best = (a, w_v40)
        print(f"\nBest ensemble: w_v40 = {best[1]:.2f}  -> macro-AUC = {best[0]:.4f}")

    out = {
        "n_val_windows": int(len(Xv)),
        "n_val_files": int(len(val_files)),
        "macro_auc": {k: float(v[0]) for k, v in results.items()},
    }
    out_path = os.path.join(PROJECT_DIR, "eval_v40_local.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
