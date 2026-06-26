"""
Experiment 4: Privacy-Utility Tradeoff

Measures the trade-off between DP privacy budget ε and trust accuracy.
For each ε in {0.1, 0.5, 1.0, 2.0, 5.0}:
  - Apply Gaussian DP noise to aggregate trust outputs
  - Measure MAE vs. ground-truth trust
  - Measure F1 score of malicious detection under noise

Output: results/data/exp4_privacy_utility.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import TrustEngine
from src.attacks.attack_models import assign_attack_roles
from src.crypto.differential_privacy import DifferentialPrivacyLayer
from src.metrics.evaluator import classification_metrics


def run_single(cfg: dict, run_id: int) -> dict:
    seed = cfg["simulation"]["seed"] + run_id + 2000
    rng = np.random.default_rng(seed)

    G, _ = build_ntn_graph(cfg, rng)
    nodes = list(G.nodes())
    N = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    model = TrustEngine(N, cfg, node_types, rng)

    # Run simulation to get converged trust scores
    for rnd in range(cfg["simulation"]["num_rounds"]):
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality = float(quality_fns[j](j, rnd))
            model.update(i, j, quality, delay_ms)

    # Ground truth: trust scores without any noise
    true_scores = np.array([model.get(0, j) for j in nodes])
    true_malicious = np.array([j in malicious_ids for j in nodes])

    # Without DP: baseline F1
    baseline = classification_metrics(true_scores, true_malicious,
                                       cfg["trust"]["threshold_theta"])

    epsilon_results = []
    for eps in cfg["crypto"]["dp_epsilon_values"]:
        dp = DifferentialPrivacyLayer(
            epsilon=eps,
            delta=cfg["crypto"]["dp_delta"],
            sensitivity=1.0
        )
        # Apply DP noise to aggregate query output (not individual scores)
        noisy_scores = dp.apply_batch(true_scores, rng)
        cm = classification_metrics(noisy_scores, true_malicious,
                                     cfg["trust"]["threshold_theta"])
        mae = float(np.mean(np.abs(noisy_scores - true_scores)))
        epsilon_results.append({
            "epsilon": eps,
            "sigma": dp.sigma,
            "mae": mae,
            "f1": cm["f1"],
            "precision": cm["precision"],
            "recall": cm["recall"],
            "fpr": cm["fpr"],
        })

    return {
        "run_id": run_id,
        "baseline_f1": baseline["f1"],
        "baseline_mae": 0.0,
        "epsilon_results": epsilon_results,
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_runs = cfg["simulation"]["num_runs"]
    results = []
    print(f"[Exp4] Running {num_runs} privacy-utility tradeoff runs...")
    for run_id in tqdm(range(num_runs)):
        results.append(run_single(cfg, run_id))

    # Aggregate across runs for each epsilon
    epsilon_values = cfg["crypto"]["dp_epsilon_values"]
    summary = []
    for eps in epsilon_values:
        maes = [r["epsilon_results"][epsilon_values.index(eps)]["mae"]
                for r in results]
        f1s = [r["epsilon_results"][epsilon_values.index(eps)]["f1"]
               for r in results]
        summary.append({
            "epsilon": eps,
            "mae_mean": float(np.mean(maes)),
            "mae_ci95": 1.96 * float(np.std(maes)) / np.sqrt(len(maes)),
            "f1_mean": float(np.mean(f1s)),
            "f1_ci95": 1.96 * float(np.std(f1s)) / np.sqrt(len(f1s)),
            "sigma": results[0]["epsilon_results"][epsilon_values.index(eps)]["sigma"],
        })

    baseline_f1 = float(np.mean([r["baseline_f1"] for r in results]))
    output = {
        "runs": results,
        "summary": summary,
        "baseline_f1_no_dp": baseline_f1,
    }
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp4_privacy_utility.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp4] Done. Results saved to results/data/exp4_privacy_utility.json")
    print(f"[Exp4] Baseline F1 (no DP): {baseline_f1:.3f}")
    for s in summary:
        print(f"  ε={s['epsilon']}: MAE={s['mae_mean']:.4f}, F1={s['f1_mean']:.3f}")


if __name__ == "__main__":
    main()
