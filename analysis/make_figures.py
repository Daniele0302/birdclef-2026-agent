"""
make_figures.py — Regenerate every data-driven figure in the report and video
straight from the raw artefacts, so each chart is reproducible and traceable.

Run from the repo root:
    .venv/bin/python analysis/make_figures.py

Outputs (written to figures/):
    fig_discovery_trajectory.png   — report Fig. 3 / video Block 5
    fig_reliability_bars.png       — report Fig. 2
    fig_distribution_shift.png     — video Block 6 (focal vs soundscape mel-spec)
    fig_submission_progression.png — video Block 6

Sources read:
    experiments/experiment_log.json        (the 108 logged experiments)
    results/agent_reliability_stats.json    (per-stage reliability)
    data/train.csv, data/train_soundscapes_labels.csv, audio files
The submission-progression numbers are the team's Kaggle results, kept in a
single SUBMISSIONS table below so the chart and the report's Table 2 stay in sync.
"""

import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths (this file lives in analysis/, data sits at the repo root) ──────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH   = os.path.join(ROOT, "experiments", "experiment_log.json")
REL_PATH   = os.path.join(ROOT, "results", "agent_reliability_stats.json")
DATA_DIR   = os.path.join(ROOT, "data")
OUT_DIR    = os.path.join(ROOT, "figures")

# Mel-spectrogram parameters the agent converged on (best run, iter 78).
SR, N_MELS, N_FFT, HOP, FMIN, FMAX, TOP_DB = 32000, 64, 2048, 256, 20, 16000, 60.0
DURATION_S = 5

# Team Kaggle submission history (matches Table 2 of the report).
SUBMISSIONS = [
    # (number, local_auc, kaggle_lb)   kaggle_lb=None means "not submitted"
    (1, 0.55,   None),
    (2, 0.6004, 0.592),
    (3, 0.625,  0.598),
    (4, 0.6707, 0.615),
    (5, 0.6520, 0.633),
    (6, 0.8619, 0.725),
]


def _load_snapshot(n=108):
    """First n logged experiments (the snapshot the report analyses)."""
    return json.load(open(LOG_PATH))[:n]


# ── Figure 1: discovery trajectory ───────────────────────────────────────────
def discovery_trajectory():
    snap = _load_snapshot()
    eff_pts, cnn_pts = [], []
    for i, e in enumerate(snap):
        m = e.get("metrics")
        if e.get("success") and m and "val_auc" in m:
            pt = (i + 1, round(m["val_auc"], 4))
            (cnn_pts if m.get("model_type") == "cnn" else eff_pts).append(pt)

    # running best over all successful runs in order
    best, best_line = 0.0, []
    for i, e in enumerate(snap):
        m = e.get("metrics")
        if e.get("success") and m and "val_auc" in m:
            best = max(best, m["val_auc"])
            best_line.append((i + 1, round(best, 4)))

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=130, facecolor="white")
    ax.scatter([x for x, _ in eff_pts], [y for _, y in eff_pts],
               s=26, color="#5b6db5", alpha=0.55, label="successful experiment", zorder=3)
    ax.scatter([x for x, _ in cnn_pts], [y for _, y in cnn_pts],
               s=90, marker="^", color="#b03030", label="custom-CNN trial", zorder=4)
    ax.plot([x for x, _ in best_line], [y for _, y in best_line],
            color="#1f3f8f", lw=2.0, label="running best", zorder=5)
    ax.set_xlabel("Experiment index (order the agent ran them)", fontsize=12)
    ax.set_ylabel("Focal validation macro-AUC", fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.58, 0.88)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#bbb")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    _save(fig, "fig_discovery_trajectory.png")
    print(f"  discovery_trajectory: {len(eff_pts)} EfficientNet + {len(cnn_pts)} CNN points")


# ── Figure 2: per-stage reliability bars ─────────────────────────────────────
def reliability_bars():
    s = json.load(open(REL_PATH))
    cond = s["conditional_rates"]
    stages = ["S1", "S2", "S3", "S4", "S5"]
    rates = [1.0, cond["P_S2_given_S1"], cond["P_S3_given_S2"],
             cond["P_S4_given_S3"], cond["P_S5_given_S4"]]

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=130, facecolor="white")
    bars = ax.bar(stages, rates, width=0.55,
                  color=["#5b6db5"] * 5, edgecolor="#1f3f8f")
    bars[3].set_color("#d39a2d")  # highlight the weak link S4
    for st, r in zip(stages, rates):
        ax.text(st, r + 0.006, f"{r:.2f}", ha="center", fontsize=11)
    ax.set_ylim(0.70, 1.05)
    ax.set_ylabel("Conditional success rate", fontsize=12)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, color="#bbb")
    _save(fig, "fig_reliability_bars.png")
    print(f"  reliability_bars: conditional rates {[round(r,3) for r in rates]}")


# ── Figure 3: distribution shift (focal vs soundscape mel-spectrograms) ───────
def distribution_shift():
    try:
        import numpy as np
        import librosa
        import librosa.display
    except ImportError:
        print("  distribution_shift: skipped (librosa not installed)")
        return

    # The two files used in the report/video, so this regenerates the exact figure.
    # If they are missing, fall back to an automatic pick.
    focal_rel = "train_audio/22983/iNat1243068.ogg"          # single-species frog, clean
    sound_rel = "train_soundscapes/BC2026_Train_0039_S22_20211231_201500.ogg"  # 5-species window
    focal_path = os.path.join(DATA_DIR, focal_rel)
    sound_path = os.path.join(DATA_DIR, sound_rel)
    if os.path.exists(focal_path) and os.path.exists(sound_path):
        focal_off, n_species = 3.0, _count_species(sound_path, "00:00:00")
    else:
        focal_path, focal_off = _pick_focal_window(np, librosa)
        sound_path, n_species = _pick_busiest_soundscape()

    def mel_db(path, offset):
        y, _ = librosa.load(path, sr=SR, offset=offset, duration=DURATION_S)
        if len(y) < SR * DURATION_S:
            y = np.pad(y, (0, SR * DURATION_S - len(y)))
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT,
                                             hop_length=HOP, fmin=FMIN, fmax=FMAX)
        return librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)

    fig = plt.figure(figsize=(16, 6.5), dpi=130, facecolor="white")
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.97, top=0.80, bottom=0.10, wspace=0.18)
    panels = [
        ("Focal recording  (training domain)\nsingle species, clean", focal_path, focal_off, "#1f4f8b"),
        (f"Soundscape window  (Kaggle test domain)\n{n_species} species + background noise", sound_path, 0.0, "#c0392b"),
    ]
    for (title, path, off, color), col in zip(panels, range(2)):
        ax = fig.add_subplot(gs[0, col])
        librosa.display.specshow(mel_db(path, off), sr=SR, hop_length=HOP, fmin=FMIN,
                                 fmax=FMAX, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(title, fontsize=16, fontweight="bold", color=color, pad=12)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Mel frequency (Hz)", fontsize=12)
    fig.text(0.5, 0.92, "Distribution shift: what the agent trained on  vs  what Kaggle scores",
             ha="center", fontsize=22, fontweight="bold")
    _save(fig, "fig_distribution_shift.png")
    print(f"  distribution_shift: focal={os.path.basename(focal_path)}, "
          f"soundscape={os.path.basename(sound_path)} ({n_species} species)")


def _pick_focal_window(np, librosa, scan=30):
    """Scan a few focal recordings, return the file+offset with the strongest call."""
    import glob, random
    best = (None, 0.0, -1.0)
    for sp in sorted(os.listdir(os.path.join(DATA_DIR, "train_audio")))[:scan]:
        files = glob.glob(os.path.join(DATA_DIR, "train_audio", sp, "*.ogg"))
        if not files:
            continue
        try:
            y, _ = librosa.load(files[0], sr=SR, duration=30)
        except Exception:
            continue
        step, win = SR, SR * DURATION_S
        for start in range(0, max(1, len(y) - win), step):
            seg = y[start:start + win]
            rms = librosa.feature.rms(y=seg, frame_length=1024, hop_length=256)[0]
            peakiness = (rms.max() - rms.mean()) * rms.mean()
            if peakiness > best[2]:
                best = (files[0], start / SR, peakiness)
    return best[0], best[1]


def _pick_busiest_soundscape():
    """Return (file_path, n_species) of the labelled window with the most species."""
    import csv
    rows = list(csv.DictReader(open(os.path.join(DATA_DIR, "train_soundscapes_labels.csv"))))
    best = max(rows, key=lambda r: len(r["primary_label"].split(";")))
    path = os.path.join(DATA_DIR, "train_soundscapes", best["filename"])
    return path, len(best["primary_label"].split(";"))


def _count_species(sound_path, start):
    """Number of species labelled in a specific window of a soundscape file."""
    import csv
    fname = os.path.basename(sound_path)
    rows = list(csv.DictReader(open(os.path.join(DATA_DIR, "train_soundscapes_labels.csv"))))
    row = next((r for r in rows if r["filename"] == fname and r["start"] == start), None)
    return len(row["primary_label"].split(";")) if row else 0


# ── Figure 4: submission progression ─────────────────────────────────────────
def submission_progression():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=130, facecolor="white")
    nums = [s[0] for s in SUBMISSIONS]
    local = [s[1] for s in SUBMISSIONS]
    kaggle = [(s[0], s[2]) for s in SUBMISSIONS if s[2] is not None]

    ax.axvspan(1.7, 5.3, color="#fff4d6", zorder=0)
    ax.text(3.5, 0.90, "CPU-only submissions", ha="center", fontsize=13,
            color="#8a6d1a", style="italic")
    ax.plot(nums, local, marker="s", color="#c0392b", lw=2.3, ms=9,
            label="Local soundscape val")
    ax.plot([n for n, _ in kaggle], [v for _, v in kaggle], marker="o",
            color="#1f4f8b", lw=2.3, ms=9, label="Kaggle public LB")
    ax.set_xticks(nums)
    ax.set_xticklabels([f"#{n}" for n in nums])
    ax.set_xlim(0.6, 6.4)
    ax.set_ylim(0.50, 0.92)
    ax.set_xlabel("Submission #", fontsize=12)
    ax.set_ylabel("Macro-AUC", fontsize=12)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#bbb")
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    _save(fig, "fig_submission_progression.png")
    print(f"  submission_progression: {len(SUBMISSIONS)} submissions, best LB "
          f"{max(v for _, v in kaggle)}")


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    print("Regenerating figures from raw data ...")
    discovery_trajectory()
    reliability_bars()
    submission_progression()
    distribution_shift()
    print(f"Done. Figures written to {os.path.relpath(OUT_DIR, ROOT)}/")
