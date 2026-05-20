# Figures

Every chart in the report and video is generated from the raw data by one script:

```bash
.venv/bin/python analysis/make_figures.py
```

| File | Where it's used | Source data |
|---|---|---|
| `fig_discovery_trajectory.png` | Report Fig. 3 / Video Block 5 | `experiments/experiment_log.json` |
| `fig_reliability_bars.png` | Report Fig. 2 | `results/agent_reliability_stats.json` |
| `fig_distribution_shift.png` | Video Block 6 | two real audio files (focal vs soundscape) |
| `fig_submission_progression.png` | Video Block 6 | the `SUBMISSIONS` table in `make_figures.py` (= Table 2 of the report) |

Re-running the script reproduces the figures from scratch, so the numbers in the
report and video can be traced back to the logged experiments.
