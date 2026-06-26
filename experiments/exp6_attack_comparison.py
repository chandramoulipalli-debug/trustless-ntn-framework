"""
Experiment 6: Attack Comparison (ROC + F1 vs. Malicious Fraction)

Sweeps malicious node fraction from 5% to 30%.
For each fraction and each model, computes:
- F1, Precision, Recall, FPR at detection threshold θ
- AUC-ROC
- Full ROC curve data for proposed model at 20% malicious

Also covers: Sybil resilience and replay detection rate.

Output: results/data/exp6_attack_comparison.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import (TrustEngine, CentralizedTrust,
                                     StaticDLT, ZTAuthOnly, UAVBlockchainFL)
from src.attacks.attack_models import assign_attack_roles
from src.metrics.evaluator import classification_metrics, roc_data


MALICIOUS_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MODEL_CLASSES = {
    "proposed":    TrustEngine,
    "centralized": CentralizedTrust,
    "static_dlt":  StaticDLT,
    "zt_auth":     ZTAuthOnly,
    "uav_fl":      UAVBlockchainFL,
}


def run_single(cfg: dict, run_id: int, mal_frac: float) -> dict:
    seed = cfg["simulation"]["seed"] + run_id + int(mal_frac * 10000) + 3000
    rng = np.random.default_rng(seed)

    G, _ = build_ntn_graph(cfg, rng)
    nodes = list(G.nodes())
    N = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    # Override malicious fraction for this sweep point
    cfg_copy = {**cfg, "attacks": {**cfg["attacks"], "malicious_fraction": mal_frac}}
    quality_fns, malicious_ids = assign_attack_roles(nodes, mal_frac, cfg_copy, rng)
    true_malicious = np.array([j in malicious_ids for j in nodes])

    models = {}
    for name, cls in MODEL_CLASSES.items():
        if name == "proposed":
            models[name] = cls(N, cfg_copy, node_types, rng)
        else:
            models[name] = cls(N, cfg_copy, rng)

    for rnd in range(cfg["simulation"]["num_rounds"]):
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality = float(quality_fns[j](j, rnd))
            for model in models.values():
                model.update(i, j, quality, delay_ms)

    results = {}
    for name, model in models.items():
        scores = np.array([model.get(0, j) for j in nodes])
        cm = classification_metrics(scores, true_malicious,
                                     cfg["trust"]["threshold_theta"])
        roc = roc_data(scores, true_malicious)
        results[name] = {**cm, "auc": roc["auc"]}
        if name == "proposed" and abs(mal_frac - 0.20) < 0.001:
            results[name]["roc_curve"] = roc

    return {
        "run_id": run_id,
        "mal_frac": mal_frac,
        "n_malicious": len(malicious_ids),
        "metrics": results,
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_runs = cfg["simulation"]["num_runs"]
    all_results = []

    for mal_frac in MALICIOUS_FRACTIONS:
        print(f"[Exp6] Malicious fraction = {mal_frac:.0%}")
        for run_id in tqdm(range(num_runs), leave=False):
            all_results.append(run_single(cfg, run_id, mal_frac))

    # Aggregate per (model, mal_frac)
    summary = {}
    for mal_frac in MALICIOUS_FRACTIONS:
        frac_runs = [r for r in all_results if abs(r["mal_frac"] - mal_frac) < 1e-6]
        summary[str(mal_frac)] = {}
        for name in MODEL_CLASSES:
            f1s = [r["metrics"][name]["f1"] for r in frac_runs]
            aucs = [r["metrics"][name]["auc"] for r in frac_runs]
            fprs = [r["metrics"][name]["fpr"] for r in frac_runs]
            summary[str(mal_frac)][name] = {
                "f1_mean": float(np.mean(f1s)),
                "f1_ci95": 1.96 * float(np.std(f1s)) / np.sqrt(len(f1s)),
                "auc_mean": float(np.mean(aucs)),
                "fpr_mean": float(np.mean(fprs)),
            }

    output = {"results": all_results, "summary": summary}
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp6_attack_comparison.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp6] Done. Results saved to results/data/exp6_attack_comparison.json")
    print("\n[Exp6] F1 @ 20% malicious:")
    for name in MODEL_CLASSES:
        v = summary["0.2"][name]
        print(f"  {name:15s}: F1={v['f1_mean']:.3f} ± {v['f1_ci95']:.3f}, "
              f"AUC={v['auc_mean']:.3f}")


if __name__ == "__main__":
    main()
