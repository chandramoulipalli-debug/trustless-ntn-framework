"""
Experiment 7: Threshold Sensitivity Analysis

Sweeps detection threshold theta from 0.10 to 0.50, measuring F1,
FPR, Precision, and Recall for the proposed model at 20% malicious.
Empirically confirms the analytically derived optimal theta* = 0.20.

Output: results/data/exp7_threshold_sensitivity.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import TrustEngine
from src.attacks.attack_models import assign_attack_roles
from src.metrics.evaluator import classification_metrics, roc_data


def run_single(cfg, run_id):
    rng = np.random.default_rng(cfg["simulation"]["seed"] + run_id)
    G, _ = build_ntn_graph(cfg, rng)
    nodes = list(G.nodes())
    N = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )
    model = TrustEngine(N, cfg, node_types, rng)

    for rnd in range(cfg["simulation"]["num_rounds"]):
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality  = float(quality_fns[j](j, rnd))
            model.update(i, j, quality, delay_ms)

    scores   = np.array([model.get(0, j) for j in nodes])
    true_mal = np.array([j in malicious_ids for j in nodes])
    return scores, true_mal


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_runs = cfg["simulation"]["num_runs"]
    thetas   = np.round(np.arange(0.10, 0.51, 0.05), 2).tolist()

    print(f"[Exp7] {num_runs} runs x {cfg['simulation']['num_rounds']} rounds")

    all_scores, all_labels = [], []
    for r in tqdm(range(num_runs), desc="Collecting trust scores"):
        s, l = run_single(cfg, r)
        all_scores.append(s)
        all_labels.append(l)

    aucs     = [roc_data(all_scores[r], all_labels[r])["auc"] for r in range(num_runs)]
    auc_mean = float(np.mean(aucs))
    print(f"  AUC (threshold-independent): {auc_mean:.3f}")

    print(f"\n[Exp7] Sweeping theta...")
    sweep = []
    for theta in thetas:
        ms   = [classification_metrics(all_scores[r], all_labels[r], theta)
                for r in range(num_runs)]
        f1s  = [m["f1"]        for m in ms]
        fprs = [m["fpr"]       for m in ms]
        prec = [m["precision"] for m in ms]
        rec  = [m["recall"]    for m in ms]
        sweep.append({
            "theta":          float(theta),
            "f1_mean":        float(np.mean(f1s)),
            "f1_ci95":        float(1.96 * np.std(f1s) / np.sqrt(num_runs)),
            "fpr_mean":       float(np.mean(fprs)),
            "precision_mean": float(np.mean(prec)),
            "recall_mean":    float(np.mean(rec)),
        })
        print(f"  theta={theta:.2f}: F1={np.mean(f1s):.3f}  "
              f"FPR={np.mean(fprs):.3f}  Prec={np.mean(prec):.3f}  "
              f"Rec={np.mean(rec):.3f}")

    # Analytical steady-state
    alpha, T_round = cfg["trust"]["alpha"], cfg["trust"]["round_duration_sec"]
    analytic = {}
    for seg, lam in [("GEO",    cfg["trust"]["lambda_geo"]),
                     ("Ground", cfg["trust"]["lambda_ground"]),
                     ("LEO",    cfg["trust"]["lambda_leo"]),
                     ("UAV",    cfg["trust"]["lambda_uav"])]:
        denom   = 1.0 - alpha * np.exp(-lam * (T_round + 0.06))
        T_legit = (1 - alpha) * 0.85 / denom
        T_mal   = (1 - alpha) * 0.10 / denom
        analytic[seg] = {
            "lambda":       float(lam),
            "T_star_legit": float(T_legit),
            "T_star_mal":   float(T_mal),
            "theta_opt":    float((T_legit + T_mal) / 2),
        }

    output = {
        "auc_mean":              auc_mean,
        "analytic_steady_state": analytic,
        "theta_sweep":           sweep,
    }
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp7_threshold_sensitivity.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp7] Done -> results/data/exp7_threshold_sensitivity.json")


if __name__ == "__main__":
    main()
