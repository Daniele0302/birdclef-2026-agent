"""
eval_v40_full_ensemble.py - Try ALL local .keras models on cached soundscape
val, then run greedy forward weight selection to build the best ensemble.

Greedy step: at each iteration, find (model, weight) addition that maximises
macro-AUC. Geometric-mean aggregation. Discrete weight grid {0.1, 0.2, ...,
1.0} per added model. Stops when no addition improves the AUC.

Run after Plan B finishes:
    .venv/bin/python eval_v40_full_ensemble.py
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

# Candidate models. Drop duplicates (GOLD == exp78_full per disk size).
CANDIDATES = {
    "exp78":          os.path.join(PROJECT_DIR, "exp78_full_model.keras"),
    "exp58_sc":       os.path.join(PROJECT_DIR, "model_exp58_soundscape07820.keras"),
    "exp78_chkp_sc":  os.path.join(PROJECT_DIR, "model_exp78_checkpoint_soundscape08076.keras"),
    "exp78_sc":       os.path.join(PROJECT_DIR, "model_exp78_soundscape07759.keras"),
    "exp97_sc":       os.path.join(PROJECT_DIR, "model_exp97_soundscape08074.keras"),
    "v40":            os.path.join(PROJECT_DIR, "model_v40_strong.keras"),
    "v40b":           os.path.join(PROJECT_DIR, "model_v40b_strong.keras"),
}

SOUNDSCAPE_VAL_FRAC = 0.2
RANDOM_SEED = 20260507
BATCH_SIZE = 32
EPS = 1e-7


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


def compute_val_predictions():
    Xs = np.load(CACHE_SC_X).astype(np.float32)
    Ys = np.load(CACHE_SC_Y)
    Fs = np.load(CACHE_SC_F, allow_pickle=True)
    rng = np.random.RandomState(RANDOM_SEED)
    unique_files = np.unique(Fs)
    n_val_files = max(1, int(round(len(unique_files) * SOUNDSCAPE_VAL_FRAC)))
    val_files = set(rng.choice(unique_files, size=n_val_files, replace=False))
    val_mask = np.array([f in val_files for f in Fs])
    Xv, Yv = Xs[val_mask], Ys[val_mask]
    Xv3 = np.stack([Xv, Xv, Xv], axis=-1).astype(np.float32)
    print(f"val: {len(Xv)} windows from {len(val_files)} files")

    preds = {}
    for name, path in CANDIDATES.items():
        if not os.path.exists(path):
            print(f"  [skip] missing {name}: {path}")
            continue
        try:
            print(f"  loading {name}")
            m = load_keras_with_patch(path)
            if m.input_shape[1:] != (64, 626, 3) or m.output_shape[-1] != 234:
                print(f"  [skip] {name}: shape mismatch in/out")
                continue
            p = m.predict(Xv3, batch_size=BATCH_SIZE, verbose=0)
            a = macro_auc(Yv, p)
            print(f"  {name:18s} macro-AUC = {a:.4f}")
            preds[name] = p
            del m
        except Exception as exc:
            print(f"  [err] {name}: {exc}")
    return Yv, preds


def geo_mean_score(weights_dict, log_preds, Yv):
    log_mix = None
    for n, w in weights_dict.items():
        if w <= 0:
            continue
        contrib = w * log_preds[n]
        log_mix = contrib if log_mix is None else log_mix + contrib
    if log_mix is None:
        return -1.0
    mix = np.exp(log_mix)
    return macro_auc(Yv, mix)


def normalize(d):
    s = sum(d.values())
    if s <= 0:
        return d
    return {k: v / s for k, v in d.items()}


def greedy_forward(Yv, preds):
    """At each step, pick the (model, weight in {0.1...1.0}) addition that
    maximises macro-AUC of the geometric ensemble. Stop when no addition helps.
    """
    log_preds = {n: np.log(np.clip(p, EPS, 1.0)) for n, p in preds.items()}
    weights = {n: 0.0 for n in preds.keys()}
    grid = np.arange(0.1, 1.01, 0.1)

    print("\nGreedy forward weight selection:")
    print("  start: AUC = NaN  (no models)")
    best_auc = -1.0
    best_state = None
    history = []

    while True:
        improved = False
        local_best_auc = best_auc
        local_best_change = None
        for name in preds.keys():
            for delta in grid:
                trial = dict(weights)
                trial[name] = trial[name] + delta
                a = geo_mean_score(normalize(trial), log_preds, Yv)
                if a > local_best_auc + 1e-5:
                    local_best_auc = a
                    local_best_change = (name, delta)
                    improved = True
        if not improved:
            break
        name, delta = local_best_change
        weights[name] += delta
        best_auc = local_best_auc
        best_state = (dict(weights), best_auc)
        norm = normalize(weights)
        wstr = ", ".join(f"{n}={norm[n]:.2f}" for n in norm if norm[n] > 0)
        print(f"  + {name} (delta={delta:.1f})  -> AUC={best_auc:.4f}  [{wstr}]")
        history.append({"weights": norm, "auc": best_auc})

    final_w = normalize(weights)
    return best_auc, final_w, history


def main():
    Yv, preds = compute_val_predictions()
    if not preds:
        print("No predictions, abort")
        return

    print("\nIndividual macro-AUCs:")
    aucs = {}
    for n, p in preds.items():
        a = macro_auc(Yv, p)
        aucs[n] = a
        print(f"  {n:18s} {a:.4f}")

    # Models trained by the agent with soundscape data on a DIFFERENT val
    # split overlap with our val set, so their scores are inflated by leakage
    # (confirmed: exp78_sc local 0.62 -> Kaggle LB 0.48 in v37). Exclude.
    LEAKY = {"exp58_sc", "exp78_chkp_sc", "exp78_sc", "exp97_sc"}
    clean_preds = {n: p for n, p in preds.items() if n not in LEAKY}
    print("\nCLEAN ensemble candidates (no soundscape-val leakage):")
    print(f"  {list(clean_preds.keys())}")

    best_auc, final_w, history = greedy_forward(Yv, clean_preds)

    print("\n=== BEST ENSEMBLE ===")
    print(f"  macro-AUC = {best_auc:.4f}")
    for n, w in final_w.items():
        if w > 0:
            print(f"  {n:18s} weight = {w:.3f}")

    # Persist
    out = {
        "individual_auc": aucs,
        "best_auc": best_auc,
        "weights": {k: float(v) for k, v in final_w.items()},
        "history": history,
    }
    out_path = os.path.join(PROJECT_DIR, "eval_v40_full_ensemble.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
