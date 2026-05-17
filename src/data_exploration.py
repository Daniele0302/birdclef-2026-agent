"""
data_exploration.py — Dataset exploration step for the autonomous agent.

This runs once at the start of every agent.py session. It loads and explores
the competition training data, prints a concise summary to stdout, and saves
the full breakdown to experiments/data_exploration.json so it can be cited
in the report or fed into the LLM prompt.

The exploration covers:
  - taxonomy and class structure (234 target species across multiple taxa)
  - focal training-set size and per-species file counts (class imbalance)
  - which species in the taxonomy have no training audio
  - labelled soundscape windows vs the soundscape pool
  - audio sample-rate, duration and format (probed from a sample of files)

Run standalone:
    python src/data_exploration.py
"""

import json
import os
import random
import statistics
import sys

# This module lives in src/; resolve paths against the repo root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR              = os.path.join(_PROJECT_ROOT, "data")
TRAIN_CSV             = os.path.join(DATA_DIR, "train.csv")
TAXONOMY_CSV          = os.path.join(DATA_DIR, "taxonomy.csv")
TRAIN_AUDIO_DIR       = os.path.join(DATA_DIR, "train_audio")
TRAIN_SOUNDSCAPES_DIR = os.path.join(DATA_DIR, "train_soundscapes")
SOUNDSCAPES_LABELS    = os.path.join(DATA_DIR, "train_soundscapes_labels.csv")
OUTPUT_PATH           = os.path.join(_PROJECT_ROOT, "experiments", "data_exploration.json")


def _read_csv_columns(path):
    """Read a CSV into a list-of-dicts without pandas (keeps the agent lean)."""
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _summarise_numeric(values):
    """Return min / median / mean / max / total for a list of numbers."""
    if not values:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None, "total": 0}
    return {
        "n":      len(values),
        "min":    int(min(values)),
        "median": int(statistics.median(values)),
        "mean":   round(statistics.mean(values), 1),
        "max":    int(max(values)),
        "total":  int(sum(values)),
    }


def _probe_audio_samples(audio_files, n_probe=12):
    """Open a random sample of audio files and report sample-rate / duration stats."""
    try:
        import soundfile as sf
    except ImportError:
        return {"probed": 0, "note": "soundfile not installed — probe skipped"}

    sample = random.sample(audio_files, min(n_probe, len(audio_files)))
    sample_rates = []
    durations = []
    for f in sample:
        try:
            info = sf.info(f)
            sample_rates.append(info.samplerate)
            durations.append(round(info.duration, 2))
        except Exception:
            pass
    return {
        "probed":             len(durations),
        "sample_rate_hz":     sorted(set(sample_rates)),
        "duration_seconds":   {
            "min":    round(min(durations), 2)    if durations else None,
            "median": round(statistics.median(durations), 2) if durations else None,
            "mean":   round(statistics.mean(durations), 2)   if durations else None,
            "max":    round(max(durations), 2)    if durations else None,
        },
    }


def explore_dataset(save_to_disk=True, verbose=True, audio_probe=True):
    """Run the full exploration pass. Returns a dict, optionally saves and prints it."""
    report = {}

    # --- Taxonomy: which species are we asked to predict? ---
    taxonomy = _read_csv_columns(TAXONOMY_CSV)
    by_class = {}
    for row in taxonomy:
        by_class.setdefault(row["class_name"], 0)
        by_class[row["class_name"]] += 1
    report["taxonomy"] = {
        "n_target_species":      len(taxonomy),
        "species_per_taxonomic_class": dict(sorted(by_class.items(),
                                                   key=lambda kv: -kv[1])),
    }

    # --- Focal training set: per-species file counts ---
    train = _read_csv_columns(TRAIN_CSV)
    per_species = {}
    for row in train:
        per_species.setdefault(row["primary_label"], 0)
        per_species[row["primary_label"]] += 1
    counts = list(per_species.values())
    sorted_species = sorted(per_species.items(), key=lambda kv: -kv[1])

    report["focal_training_set"] = {
        "n_recordings":      len(train),
        "n_species_present": len(per_species),
        "files_per_species": _summarise_numeric(counts),
        "top_10_most_recorded":  [{"species": s, "files": n} for s, n in sorted_species[:10]],
        "bottom_10_least_recorded": [{"species": s, "files": n} for s, n in sorted_species[-10:]],
    }

    # --- Species in the taxonomy with NO focal training data ---
    taxonomy_ids = {row["primary_label"] for row in taxonomy}
    species_with_audio = set(per_species)
    missing = sorted(taxonomy_ids - species_with_audio)
    report["species_with_no_focal_audio"] = {
        "count":   len(missing),
        "species": missing,
    }

    # --- Soundscape pool + labelled subset (= near-test-distribution validation) ---
    sc_pool_count = 0
    if os.path.isdir(TRAIN_SOUNDSCAPES_DIR):
        sc_pool_count = sum(1 for f in os.listdir(TRAIN_SOUNDSCAPES_DIR) if f.endswith(".ogg"))

    sc_labels = _read_csv_columns(SOUNDSCAPES_LABELS) if os.path.exists(SOUNDSCAPES_LABELS) else []
    sc_files_labelled = sorted({row["filename"] for row in sc_labels})
    # Each label row may carry multiple species separated by ';' — count them.
    n_multi_species_windows = sum(
        1 for row in sc_labels if ";" in row.get("primary_label", "")
    )
    report["soundscapes"] = {
        "files_in_pool":               sc_pool_count,
        "labelled_5s_windows":         len(sc_labels),
        "labelled_unique_files":       len(sc_files_labelled),
        "windows_with_multiple_species": n_multi_species_windows,
        "note": ("the labelled subset is the basis for the soundscape-aware validation "
                 "split used to verify the agent's architecture choice at scale"),
    }

    # --- Audio probe (sample-rate, duration) ---
    if audio_probe:
        focal_files = []
        for sp_dir in os.listdir(TRAIN_AUDIO_DIR)[:20]:
            sp_path = os.path.join(TRAIN_AUDIO_DIR, sp_dir)
            if os.path.isdir(sp_path):
                for fn in os.listdir(sp_path)[:3]:
                    if fn.endswith(".ogg"):
                        focal_files.append(os.path.join(sp_path, fn))
        report["audio_format_probe"] = _probe_audio_samples(focal_files, n_probe=12)

    # --- Save + print ---
    if save_to_disk:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(report, f, indent=2)

    if verbose:
        _print_summary(report)

    return report


def _print_summary(r):
    """Pretty one-pager so the agent operator sees what it found."""
    print("=" * 70)
    print("DATA EXPLORATION  —  what the agent sees before it starts proposing")
    print("=" * 70)
    print(f"Target species (taxonomy)        : {r['taxonomy']['n_target_species']}")
    for cls, n in r["taxonomy"]["species_per_taxonomic_class"].items():
        print(f"  · {cls:<12s} {n:>4d}")
    print("-" * 70)
    f = r["focal_training_set"]
    print(f"Focal training recordings        : {f['n_recordings']}")
    print(f"Species with at least one clip   : {f['n_species_present']}  "
          f"(of {r['taxonomy']['n_target_species']})")
    fp = f["files_per_species"]
    print(f"Files per species   min / median / max / mean : "
          f"{fp['min']} / {fp['median']} / {fp['max']} / {fp['mean']}")
    print(f"Species with zero focal audio    : "
          f"{r['species_with_no_focal_audio']['count']}  → 234-class problem "
          "is effectively long-tailed")
    print("-" * 70)
    s = r["soundscapes"]
    print(f"Soundscape audio files in pool   : {s['files_in_pool']}")
    print(f"Labelled 5-second windows        : {s['labelled_5s_windows']}  "
          f"(across {s['labelled_unique_files']} files)")
    print(f"  · windows w/ >1 species label  : {s['windows_with_multiple_species']}  "
          "→ confirms the multi-label nature")
    print("-" * 70)
    if "audio_format_probe" in r:
        a = r["audio_format_probe"]
        print(f"Audio probe ({a.get('probed', 0)} files):")
        if a.get("sample_rate_hz"):
            print(f"  · sample rate(s)  : {a['sample_rate_hz']} Hz")
        if a.get("duration_seconds", {}).get("median") is not None:
            d = a["duration_seconds"]
            print(f"  · duration (s)    : "
                  f"min={d['min']}, median={d['median']}, mean={d['mean']}, max={d['max']}")
    print("=" * 70)
    print(f"Full report written to: {os.path.relpath(OUTPUT_PATH, _PROJECT_ROOT)}")
    print("=" * 70)


def to_prompt_block(report):
    """Render a compact, LLM-friendly summary of the exploration report.

    Used by agent.py to inject dataset facts into every prompt, so the LLM
    proposes configurations grounded in what's actually in the data instead
    of generic guesses.
    """
    f = report["focal_training_set"]
    s = report["soundscapes"]
    t = report["taxonomy"]
    tax_breakdown = ", ".join(f"{n} {cls}" for cls, n in
                              t["species_per_taxonomic_class"].items())
    fps = f["files_per_species"]
    audio = report.get("audio_format_probe", {})
    sr = audio.get("sample_rate_hz", [])
    dur = audio.get("duration_seconds", {})
    return (
        "DATASET FACTS (computed once at agent startup):\n"
        f"- {t['n_target_species']} target species ({tax_breakdown}); "
        f"{f['n_species_present']} have focal audio, "
        f"{report['species_with_no_focal_audio']['count']} have none.\n"
        f"- {f['n_recordings']} focal recordings; files/species range "
        f"{fps['min']}-{fps['max']} (median {fps['median']}, mean {fps['mean']}) "
        "→ heavy class imbalance, long-tailed.\n"
        f"- {s['files_in_pool']} soundscape files in pool, only "
        f"{s['labelled_5s_windows']} 5-second windows are labelled "
        f"(across {s['labelled_unique_files']} files); "
        f"{s['windows_with_multiple_species']} of those windows contain >1 species "
        "→ multi-label is the rule, not the exception.\n"
        f"- audio format: sample-rate {sr} Hz, "
        f"clip duration median {dur.get('median')}s, max {dur.get('max')}s.\n"
    )


if __name__ == "__main__":
    explore_dataset()

