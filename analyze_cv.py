"""
analyze_cv.py — GroupKFold soundscape validation for saved BirdCLEF models.

This script does not train. It evaluates saved Keras models on train_soundscapes
using folds grouped by source soundscape filename.

Examples:
    python analyze_cv.py
    python analyze_cv.py --models exp78_full_model.keras=experiments/params_040.json
    python analyze_cv.py --models new_model.keras=experiments/params_040.json --folds 5
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from experiment_template import make_melspec


DEFAULT_MODELS = {
    "exp78_full_model.keras": "experiments/params_040.json",
    "model_exp97_soundscape08074.keras": "experiments/params_exp97_soundscape.json",
}


def load_params(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_models(items):
    if not items:
        return DEFAULT_MODELS

    models = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid model spec: {item}. Use model.keras=params.json"
            )
        model_path, params_path = item.split("=", 1)
        models[model_path] = params_path
    return models


def time_to_seconds(value):
    parts = str(value).split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def make_label_vector(labels_str, label_to_idx):
    y = np.zeros(len(label_to_idx), dtype=np.float32)
    for label in str(labels_str).split(";"):
        label = label.strip()
        if label in label_to_idx:
            y[label_to_idx[label]] = 1.0
    return y


def load_soundscape_window(filepath, start_sec, params):
    import librosa

    y, _ = librosa.load(
        filepath,
        sr=32000,
        offset=start_sec,
        duration=5.0
    )
    max_len = 32000 * 5
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    elif len(y) > max_len:
        y = y[:max_len]
    return make_melspec(y, 32000, params)


def build_fold_tensors(fold_df, data_dir, params, label_to_idx, model_input_shape):
    X = []
    y = []
    soundscapes_dir = os.path.join(data_dir, "train_soundscapes")

    for _, row in fold_df.iterrows():
        filepath = os.path.join(soundscapes_dir, row["filename"])
        if not os.path.exists(filepath):
            continue

        try:
            mel = load_soundscape_window(
                filepath,
                time_to_seconds(row["start"]),
                params
            )
        except Exception as e:
            print(f"  skipped {row['filename']} {row['start']}: {e}")
            continue

        X.append(mel)
        y.append(make_label_vector(row["primary_label"], label_to_idx))

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if len(X) == 0:
        return X, y

    if len(model_input_shape) == 3 and model_input_shape[-1] == 3:
        X = np.stack([X, X, X], axis=-1)
    elif len(model_input_shape) == 3 and model_input_shape[-1] == 1:
        X = np.expand_dims(X, axis=-1)

    expected = tuple(model_input_shape)
    if tuple(X.shape[1:]) != expected:
        raise ValueError(
            f"Generated shape {X.shape[1:]} does not match model input {expected}"
        )

    return X, y


def evaluate_model(model_path, params_path, folds, sc_df, data_dir, label_to_idx,
                   batch_size):
    import tensorflow as tf
    from tensorflow import keras

    params = load_params(params_path)
    model = keras.models.load_model(model_path, compile=False)
    model_input_shape = tuple(model.input_shape[1:])

    fold_aucs = []
    fold_rows = []

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_path}")
    print(f"PARAMS: {params_path}")
    print(f"INPUT: {model_input_shape}")
    print(f"{'=' * 70}")

    for fold_idx, (_, val_idx) in enumerate(folds, start=1):
        fold_df = sc_df.iloc[val_idx].copy()
        n_groups = fold_df["filename"].nunique()
        print(
            f"\nFold {fold_idx}: {len(fold_df)} windows, "
            f"{n_groups} soundscape files"
        )

        X_val, y_val = build_fold_tensors(
            fold_df,
            data_dir,
            params,
            label_to_idx,
            model_input_shape
        )
        if len(X_val) == 0:
            print("  no valid windows, skipping")
            continue

        y_pred = model.predict(X_val, batch_size=batch_size, verbose=0)
        auc = float(roc_auc_score(y_val.ravel(), y_pred.ravel()))
        fold_aucs.append(auc)
        fold_rows.append({
            "fold": fold_idx,
            "windows": int(len(X_val)),
            "soundscape_files": int(n_groups),
            "auc": auc,
        })
        print(f"  auc={auc:.6f}")

    del model
    tf.keras.backend.clear_session()

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else None
    std_auc = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
    return {
        "model_path": model_path,
        "params_path": params_path,
        "folds": fold_rows,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="*",
        help="Model/config pairs as model.keras=params.json"
    )
    args = parser.parse_args()

    models = parse_models(args.models)
    data_dir = os.path.abspath(args.data_dir)
    soundscapes_csv = os.path.join(data_dir, "train_soundscapes_labels.csv")
    taxonomy_csv = os.path.join(data_dir, "taxonomy.csv")

    sc_df = pd.read_csv(soundscapes_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)
    label_names = sorted(taxonomy_df["primary_label"].unique().tolist())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    groups = sc_df["filename"].values
    splitter = GroupKFold(n_splits=args.folds)
    folds = list(splitter.split(sc_df, groups=groups))

    print(f"Soundscape windows: {len(sc_df)}")
    print(f"Soundscape files: {sc_df['filename'].nunique()}")
    print(f"Classes: {len(label_names)}")
    print(f"Folds: {args.folds}")

    results = []
    for model_path, params_path in models.items():
        if not os.path.exists(model_path):
            print(f"\nWARNING: model not found, skipping: {model_path}")
            continue
        if not os.path.exists(params_path):
            print(f"\nWARNING: params not found, skipping: {params_path}")
            continue
        results.append(evaluate_model(
            model_path,
            params_path,
            folds,
            sc_df,
            data_dir,
            label_to_idx,
            args.batch_size
        ))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for result in results:
        print(
            f"{result['model_path']}: "
            f"mean_auc={result['mean_auc']:.6f}, "
            f"std_auc={result['std_auc']:.6f}"
        )

    print("\nJSON")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
