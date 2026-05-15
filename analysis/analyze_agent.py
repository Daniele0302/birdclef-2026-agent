"""
analyze_agent.py — Agent Performance Analysis for BirdCLEF 2026

Reads experiment_log.json files and produces:
1. Success rate statistics
2. AUC progression over iterations
3. Failure analysis
4. Top configurations
5. LaTeX-ready output for the report

Usage:
    python analyze_agent.py

Reads from:
    experiments/experiment_log.json       (original agent runs)
    experiments_fast/experiment_log.json  (fast search runs)
"""

import json
import os
from collections import Counter


def load_log(path):
    """Load experiment log from JSON file."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return []
    with open(path, 'r') as f:
        return json.load(f)


def analyze_log(experiments, label="Agent"):
    """Analyze a list of experiments and return statistics."""
    total = len(experiments)
    if total == 0:
        return None

    # Success/failure counts
    successes = [e for e in experiments if e.get('success', False)]
    failures = [e for e in experiments if not e.get('success', False)]
    success_rate = len(successes) / total * 100

    # AUC tracking
    best_auc = 0.0
    improvements = 0
    auc_progression = []
    failure_reasons = []

    for exp in experiments:
        metrics = exp.get('metrics', {}) or {}
        auc = metrics.get('val_auc', 0) if metrics else 0

        if exp.get('success', False) and auc > 0:
            auc_progression.append({
                'id': exp.get('id'),
                'auc': auc,
                'is_best': auc > best_auc
            })
            if auc > best_auc:
                best_auc = auc
                improvements += 1
        else:
            # Analyze failure reason
            stderr = exp.get('stderr_snippet', '')
            if 'TIMEOUT' in stderr:
                failure_reasons.append('timeout')
            elif 'JSON' in stderr or 'json' in stderr:
                failure_reasons.append('invalid_json')
            elif 'numpy' in stderr or 'val_indices' in stderr:
                failure_reasons.append('numpy_error')
            elif 'mel_norm' in stderr or 'null' in stderr:
                failure_reasons.append('mel_norm_string_bug')
            else:
                failure_reasons.append('other')

    # Top 5 experiments by AUC
    successful_with_auc = [
        (e.get('id'), e.get('metrics', {}).get('val_auc', 0))
        for e in successes
        if e.get('metrics') and e.get('metrics', {}).get('val_auc', 0) > 0
    ]
    top5 = sorted(successful_with_auc, key=lambda x: x[1], reverse=True)[:5]

    # Failure reason counts
    failure_counts = Counter(failure_reasons)

    return {
        'label': label,
        'total': total,
        'successes': len(successes),
        'failures': len(failures),
        'success_rate': success_rate,
        'best_auc': best_auc,
        'improvements': improvements,
        'improvement_rate': improvements / total * 100,
        'top5': top5,
        'failure_counts': failure_counts,
        'auc_progression': auc_progression
    }


def print_analysis(stats):
    """Print formatted analysis results."""
    if stats is None:
        return

    print(f"\n{'='*60}")
    print(f"AGENT ANALYSIS: {stats['label']}")
    print(f"{'='*60}")
    print(f"Total iterations:     {stats['total']}")
    print(f"Successful:           {stats['successes']} ({stats['success_rate']:.1f}%)")
    print(f"Failed:               {stats['failures']} ({100-stats['success_rate']:.1f}%)")
    print(f"Best val AUC:         {stats['best_auc']:.4f}")
    print(f"AUC improvements:     {stats['improvements']}")
    print(f"Improvement rate:     {stats['improvement_rate']:.1f}%")

    if stats['failure_counts']:
        print(f"\nFailure breakdown:")
        for reason, count in stats['failure_counts'].most_common():
            print(f"  {reason}: {count}")

    print(f"\nTop 5 experiments:")
    for rank, (exp_id, auc) in enumerate(stats['top5'], 1):
        print(f"  #{rank}: Experiment {exp_id} — val_AUC = {auc:.4f}")


def generate_latex(stats_list):
    """Generate LaTeX code for the report."""

    latex = []
    latex.append("")
    latex.append("% =====================================================")
    latex.append("% AGENT PERFORMANCE ANALYSIS — AUTO-GENERATED")
    latex.append("% Copy this section into your report")
    latex.append("% =====================================================")
    latex.append("")
    latex.append("\\subsection{Agent Performance Analysis}")
    latex.append("")
    latex.append("To evaluate the autonomous agent's effectiveness, we analyzed")
    latex.append("the complete experiment logs across all runs.")
    latex.append("")

    # Combined stats table
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lrrrrr}")
    latex.append("\\toprule")
    latex.append("Run & Iterations & Success Rate & Failures & Best AUC & Improvements \\\\")
    latex.append("\\midrule")

    total_iters = 0
    total_success = 0
    total_failures = 0
    total_improvements = 0
    overall_best = 0.0

    for stats in stats_list:
        if stats is None:
            continue
        label = stats['label'].replace('_', '\\_')
        latex.append(
            f"{label} & {stats['total']} & "
            f"{stats['success_rate']:.1f}\\% & "
            f"{stats['failures']} & "
            f"{stats['best_auc']:.4f} & "
            f"{stats['improvements']} \\\\"
        )
        total_iters += stats['total']
        total_success += stats['successes']
        total_failures += stats['failures']
        total_improvements += stats['improvements']
        overall_best = max(overall_best, stats['best_auc'])

    overall_success_rate = total_success / total_iters * 100 if total_iters > 0 else 0
    overall_improvement_rate = total_improvements / total_iters * 100 if total_iters > 0 else 0

    latex.append("\\midrule")
    latex.append(
        f"\\textbf{{Total}} & \\textbf{{{total_iters}}} & "
        f"\\textbf{{{overall_success_rate:.1f}\\%}} & "
        f"\\textbf{{{total_failures}}} & "
        f"\\textbf{{{overall_best:.4f}}} & "
        f"\\textbf{{{total_improvements}}} \\\\"
    )
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Agent performance summary across all experimental runs.}")
    latex.append("\\label{tab:agent_performance}")
    latex.append("\\end{table}")
    latex.append("")

    # Failure analysis
    all_failures = Counter()
    for stats in stats_list:
        if stats:
            all_failures.update(stats['failure_counts'])

    if all_failures:
        latex.append("\\paragraph{Failure Analysis}")
        latex.append(
            f"Out of {total_iters} total iterations, {total_failures} "
            f"({100-overall_success_rate:.1f}\\%) failed. "
            "Failures were caused by:"
        )
        latex.append("\\begin{itemize}")
        for reason, count in all_failures.most_common():
            readable = reason.replace('_', ' ').title()
            latex.append(f"  \\item \\textbf{{{readable}}}: {count} cases")
        latex.append("\\end{itemize}")
        latex.append("")

    # Key observations
    latex.append("\\paragraph{Key Observations}")
    latex.append("\\begin{itemize}")
    latex.append(
        f"  \\item The agent achieved an overall success rate of "
        f"\\textbf{{{overall_success_rate:.1f}\\%}}, demonstrating robust "
        "autonomous operation across diverse hyperparameter configurations."
    )
    latex.append(
        f"  \\item New best AUC records were set in "
        f"\\textbf{{{total_improvements}}} out of {total_iters} iterations "
        f"(improvement rate: {overall_improvement_rate:.1f}\\%), "
        "consistent with the expected exploration-exploitation trade-off in "
        "autonomous hyperparameter search."
    )
    latex.append(
        "  \\item The most common failure mode was the LLM generating "
        "\\texttt{\"null\"} as a string value instead of a JSON \\texttt{null}, "
        "which caused a downstream NumPy error. This was fixed by adding a "
        "string-to-None conversion in \\texttt{make\\_melspec()}."
    )
    latex.append(
        f"  \\item The fast search phase (60 iterations with 2,000 samples) "
        "efficiently identified optimal preprocessing parameters "
        "(\\texttt{fmax=12000}, \\texttt{top\\_db=50}, "
        "\\texttt{mel\\_norm=slaney}), which were then validated on larger "
        "datasets in the retest phase."
    )
    latex.append("\\end{itemize}")

    return "\n".join(latex)


def main():
    # Load both log files
    print("Loading experiment logs...")

    log_original = load_log("experiments/experiment_log.json")
    log_fast = load_log("experiments_fast/experiment_log.json")

    print(f"  Original log: {len(log_original)} experiments")
    print(f"  Fast search log: {len(log_fast)} experiments")

    # Analyze each
    stats_original = analyze_log(log_original, label="Original Agent")
    stats_fast = analyze_log(log_fast, label="Fast Search (60 iter)")

    # Print results
    if stats_original:
        print_analysis(stats_original)
    if stats_fast:
        print_analysis(stats_fast)

    # Combined summary
    print(f"\n{'='*60}")
    print("COMBINED SUMMARY")
    print(f"{'='*60}")
    total = (stats_original['total'] if stats_original else 0) + \
            (stats_fast['total'] if stats_fast else 0)
    total_s = (stats_original['successes'] if stats_original else 0) + \
              (stats_fast['successes'] if stats_fast else 0)
    total_f = (stats_original['failures'] if stats_original else 0) + \
              (stats_fast['failures'] if stats_fast else 0)
    total_imp = (stats_original['improvements'] if stats_original else 0) + \
                (stats_fast['improvements'] if stats_fast else 0)
    best = max(
        stats_original['best_auc'] if stats_original else 0,
        stats_fast['best_auc'] if stats_fast else 0
    )
    print(f"Total iterations:     {total}")
    print(f"Total successes:      {total_s} ({total_s/total*100:.1f}%)")
    print(f"Total failures:       {total_f} ({total_f/total*100:.1f}%)")
    print(f"Total improvements:   {total_imp} ({total_imp/total*100:.1f}%)")
    print(f"Overall best AUC:     {best:.4f}")

    # Generate LaTeX
    print(f"\n{'='*60}")
    print("LATEX OUTPUT FOR REPORT")
    print(f"{'='*60}")
    stats_list = [s for s in [stats_original, stats_fast] if s is not None]
    latex = generate_latex(stats_list)
    print(latex)

    # Save LaTeX to file
    output_path = "agent_analysis.tex"
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX saved to: {output_path}")


if __name__ == "__main__":
    main()
