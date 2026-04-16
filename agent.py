"""
agent.py — Autonomous BirdCLEF 2026 agent (v3 - multi-architecture)

What's new v3:
- The LLM can choose between a custom CNN and EfficientNet (transfer learning)
- Augmentation types: noise, time_shift, freq_mask, all
- Fine-tuning: unfreeze_layers for EfficientNet
"""

import json
import os
import subprocess
import time
from llm_provider import call_llm
from memory import ExperimentMemory

EXPERIMENTS_DIR = "experiments"
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

SYSTEM_PROMPT = """You are an autonomous ML research agent for BirdCLEF 2026 (Track B).

Your job is to propose experiment configurations for classifying mel-spectrograms
of wildlife audio (234 species, multi-label, sigmoid output, binary crossentropy loss).

You must respond with ONLY a valid JSON object. No text, no explanation, no markdown.

The JSON must have ALL of these keys:
{
    "experiment_name": "short_descriptive_name",
    "model_type": "cnn",
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 5,
    "n_filters_1": 32,
    "n_filters_2": 64,
    "n_filters_3": 128,
    "dropout_rate": 0.3,
    "dense_units": 256,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 50,
    "fmax": 14000,
    "max_samples": 2000,
    "use_augmentation": false,
    "augmentation_type": "noise",
    "augmentation_noise": 0.01,
    "unfreeze_layers": 0
}

ARCHITECTURE CHOICE (most important decision):
- model_type "cnn": custom 3-block CNN. Good for fast exploration. Filters matter.
- model_type "efficientnet": EfficientNetB0 pretrained on ImageNet. MUCH stronger features.
  When using efficientnet: n_filters are ignored, set lower learning_rate (0.0001-0.001),
  unfreeze_layers=0 for frozen backbone, or 10-20 for fine-tuning last layers.

PARAMETER RANGES:
- model_type: "cnn" or "efficientnet"
- learning_rate: 0.0001 to 0.01 (use lower for efficientnet: 0.0001-0.0005)
- batch_size: 16, 32, or 64
- epochs: 3 to 15
- dropout_rate: 0.1 to 0.5
- dense_units: 64 to 512
- max_samples: 1000 to 3000
- use_augmentation: true or false
- augmentation_type: "noise", "time_shift", "freq_mask", or "all"
- unfreeze_layers: 0 (frozen) to 20 (fine-tune last 20 layers). Only for efficientnet.

STRATEGY:
- Do NOT repeat the same configuration family more than twice in a row.
- If the last 3 experiments did not improve best AUC, force exploration of a meaningfully different configuration.
- Vary at least one of these strongly across experiments: batch_size, dropout_rate, max_samples, augmentation_type, unfreeze_layers.
- Do not always use augmentation_type "all". Also test "noise", "time_shift", and "freq_mask".
- If efficientnet has failed repeatedly, try a CNN experiment before returning to efficientnet.
- Explore dropout_rate values such as 0.2, 0.4, and 0.5, not only 0.3.
- Explore batch_size values 16 and 64, not only 32.
- Explore max_samples values 2500 and 3000 when previous runs are stable.
- When using efficientnet with unfreeze_layers > 0, keep learning_rate very low (0.0001 to 0.0002).
- Change 1-3 parameters per experiment, but avoid near-duplicate experiments.

Respond with ONLY the JSON."""


def build_prompt(memory_summary):
    return f"""{SYSTEM_PROMPT}

Previous experiments and results:
{memory_summary}

Based on the results, propose the next experiment. Respond with ONLY a JSON object."""


def parse_json_from_llm(response):
    text = response.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def run_experiment_from_params(params, config_path):
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)

    try:
        result = subprocess.run(
            ['python3', 'experiment_template.py', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=900  # 15 min per EfficientNet
        )

        metrics = None
        for line in reversed(result.stdout.strip().split('\n')):
            try:
                metrics = json.loads(line.strip())
                if 'val_auc' in metrics:
                    break
            except:
                continue

        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "metrics": metrics
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "returncode": -1,
            "stdout": "",
            "stderr": "TIMEOUT: experiment exceeded 900 seconds",
            "metrics": None
        }


def main():
    memory = ExperimentMemory()

    N_ITERATIONS = 8

    print("=" * 60)
    print("AUTONOMOUS AGENT — BirdCLEF 2026 (v3 multi-architecture)")
    print(f"Planned iterations: {N_ITERATIONS}")
    print(f"Available architectures: custom CNN, EfficientNetB0")
    print("=" * 60)

    for iteration in range(1, N_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{N_ITERATIONS}")
        print(f"{'='*60}")

        # Step 1: Request parameters
        print("\n[1/4] Asking the LLM to propose parameters...")
        memory_summary = memory.summarize_recent(n=15)
        prompt = build_prompt(memory_summary)
        raw_response = call_llm(prompt)
        print(f"LLM response: {raw_response[:500]}")

        # Step 2: Parse JSON
        print("\n[2/4] Parsing parameters...")
        params = parse_json_from_llm(raw_response)

        if params is None:
            print("ERROR: Invalid JSON. Skipping iteration.")
            memory.add_experiment(
                prompt=prompt,
                code=raw_response,
                result={"success": False, "stdout": "", "stderr": "Invalid JSON", "metrics": None},
                analysis="JSON parsing failed."
            )
            continue

        print(f"Model: {params.get('model_type', 'cnn')}")
        print(f"Parameters: {json.dumps(params, indent=2)}")

        # Step 3: Run
        config_path = os.path.join(EXPERIMENTS_DIR, f"params_{iteration:03d}.json")
        print(f"\n[3/4] Running '{params.get('experiment_name', 'unknown')}'...")
        start = time.time()
        result = run_experiment_from_params(params, config_path)
        elapsed = time.time() - start

        print(f"Completed in {elapsed:.0f}s | Success: {result['success']}")
        if result['metrics']:
            print(f"Metrics: {json.dumps(result['metrics'])}")
        elif result['stderr']:
            print(f"Error: {result['stderr'][:500]}")

        # Step 4: Analysis
        print("\n[4/4] LLM analysis...")
        metrics_str = json.dumps(result['metrics']) if result['metrics'] else "No metrics"
        analysis_prompt = f"""Analyze this BirdCLEF experiment concisely.

Parameters: {json.dumps(params)}
Success: {result['success']}
Metrics: {metrics_str}
Best AUC so far: {memory.best_auc}

In 3-5 sentences: what worked, what didn't, what to try next.
Focus on whether to use CNN or EfficientNet and why."""

        analysis = call_llm(analysis_prompt)
        print(f"Analysis: {analysis[:500]}")

        memory.add_experiment(
            prompt=prompt,
            code=json.dumps(params),
            result=result,
            analysis=analysis
        )

        print(f"\nBest AUC so far: {memory.best_auc}")

    print(f"\n{'='*60}")
    print(f"AGENT COMPLETED — {N_ITERATIONS} experiments")
    print(f"Best AUC: {memory.best_auc}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
