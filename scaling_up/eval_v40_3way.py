"""
eval_v40_3way.py - Evaluate exp78 + v40_strong + v40b_strong on cached
soundscape val. Sweeps 3-way ensemble weights on a coarse simplex grid
and reports the best macro-AUC.

Run AFTER train_strong_v40b.py finishes:
    .venv/bin/python eval_v40_3way.py
"""
import os
import json
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score

# Script lives in scaling_up/; PROJECT_DIR points to the repo root.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
CACHE_SC_X = os.path.join(CACHE_DIR, "specs_soundscape_v40.npy")
CACHE_SC_Y = os.path.join(CACHE_DIR, "labels_soundscape_v40.npy")
CACHE_SC_F = os.path.join(CACHE_DIR, "soundscape_files_v40.npy")

MODELS = {
    "exp78": os.path.join(PROJECT_DIR, "exp78_full_model.keras"),
    "v40":   os.path.join(PROJECT_DIR, "model_v40_strong.keras"),
    "v40b":  os.path.join(PROJECT_DIR, "model_v40b_strong.keras"),
}

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
    print("Loading cached soundscape arrays ...")
    Xs = np.load(CACHE_SC_X).astype(np.float32)
    Ys = np.load(CACHE_SC_Y)
    Fs = np.load(CACHE_SC_F, allow_pickle=True)
    rng = np.random.RandomState(RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xv, Yv = Xs[val_mask], Ys[val_mask]
    print(f"Held-out: {len(Xv)} windows from {len(val_files)} files")

    Xv3 = np.stack([Xv, Xv, Xv], axis=-1).astype(np.float32)

    preds_by = {}
    auc_by = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: missing {path}")
            continue
        print(f"\nLoading {name} ...")
        m = load_keras_with_patch(path)
        p = m.predict(Xv3, batch_size=BATCH_SIZE, verbose=0)
        a = macro_auc(Yv, p)
        print(f"  {name} macro-AUC = {a:.4f}")
        preds_by[name] = p
        auc_by[name] = a

    if len(preds_by) >= 2:
        print("\nEnsemble sweep (geometric mean):")
        names = list(preds_by.keys())
        eps = 1e-7
        log_p = {n: np.log(np.clip(preds_by[n], eps, 1.0)) for n in names}
        # Coarse simplex: 0.0, 0.1, 0.2, ..., 1.0 with sum=1
        results = []
        step = 0.1
        if len(names) == 2:
            for w in np.arange(0.0, 1.0 + step / 2, step):
                weights = [1 - w, w]
                mix = np.exp(weights[0] * log_p[names[0]] + weights[1] * log_p[names[1]])
                a = macro_auc(Yv, mix)
                results.append((a, dict(zip(names, weights))))
        elif len(names) == 3:
            for w0 in np.arange(0.0, 1.0 + step / 2, step):
                for w1 in np.arange(0.0, 1.0 - w0 + step / 2, step):
                    w2 = 1 - w0 - w1
                    if w2 < -1e-6:
                        continue
                    weights = [w0, w1, max(0.0, w2)]
                    mix = np.exp(
                        weights[0] * log_p[names[0]]
                        + weights[1] * log_p[names[1]]
                        + weights[2] * log_p[names[2]]
                    )
                    a = macro_auc(Yv, mix)
                    results.append((a, dict(zip(names, weights))))
        results.sort(key=lambda x: -x[0])
        for a, ws in results[:10]:
            wstr = " + ".join(f"{n}={ws[n]:.2f}" for n in names)
            print(f"  AUC={a:.4f}  {wstr}")
        best_a, best_w = results[0]
        print(f"\nBEST ensemble: macro-AUC = {best_a:.4f}")
        print(f"  weights: {best_w}")

    out = {
        "single_model_auc": auc_by,
        "best_ensemble": {
            "macro_auc": float(results[0][0]) if len(preds_by) >= 2 else None,
            "weights": results[0][1] if len(preds_by) >= 2 else None,
        } if len(preds_by) >= 2 else None,
    }
    out_path = os.path.join(PROJECT_DIR, "eval_v40_3way.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
