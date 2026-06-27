"""
Experiment 10: 3GPP NR-NTN Channel Model Validation

Validates framework robustness under realistic 3GPP TR 38.811 physical-layer
conditions:
  - Free-space path loss at Ka-band (20 GHz)
  - Orbital mechanics for LEO (550 km, T=95.5 min) and MEO (8000 km)
  - GEO fixed at 35,786 km (~270 ms one-way delay)
  - Dynamic link activation/deactivation with MIN_ELEV = 10 deg (LEO handovers)
  - Link quality degraded by SNR (packet-error-rate model)

Compares detection latency and classification metrics against the random-graph
baseline (Exp. 1) using identical protocol parameters.

Output: results/data/exp10_3gpp_channel.json
"""

import sys, os, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.topology.ntn_3gpp_channel import NTNChannel3GPP
from src.trust.trust_engine import TrustEngine
from src.attacks.attack_models import assign_attack_roles
from src.metrics.evaluator import detection_latency_all, classification_metrics

NUM_RUNS  = 20          # 20 independent Monte-Carlo runs (10 min est. wall-clock)
NUM_ROUNDS = 500
THETA     = 0.20


def run_single(cfg: dict, run_id: int, use_3gpp: bool) -> dict:
    seed = cfg["simulation"]["seed"] + run_id + (5000 if use_3gpp else 0)
    rng  = np.random.default_rng(seed)

    G, type_to_ids = build_ntn_graph(cfg, rng)
    nodes      = list(G.nodes())
    N          = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    model = TrustEngine(N, cfg, node_types, rng)

    # 3GPP channel model (constructed once per run)
    channel = NTNChannel3GPP(G, type_to_ids, cfg, rng) if use_3gpp else None

    trust_history = {j: [] for j in nodes}
    handovers      = 0    # count how many link-inactive events occurred

    for rnd in range(NUM_ROUNDS):
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))

            if use_3gpp:
                if not channel.link_active(i, j, rnd):
                    handovers += 1
                    # Simulate handover: node i finds alternate visible satellite
                    alt_nbrs = [nb for nb in neighbours
                                if nb != j and channel.link_active(i, nb, rnd)]
                    if alt_nbrs:
                        j = int(rng.choice(alt_nbrs))
                    else:
                        continue  # all links down — skip this node this round
                delay_ms = channel.delay_ms(i, j, rnd)
                quality  = channel.link_quality(j, rnd, quality_fns[j], i)
            else:
                delay_ms = G[i][j]["delay_ms"]
                quality  = float(quality_fns[j](j, rnd))

            model.update(i, j, quality, delay_ms)

        for jj in nodes:
            trust_history[jj].append(model.get(0, jj))

    mal_start = cfg["attacks"]["build_betray_buildup"]
    lats = detection_latency_all(trust_history, malicious_ids, mal_start, THETA)

    scores    = np.array([model.get(0, jj) for jj in nodes])
    true_mal  = np.array([jj in malicious_ids for jj in nodes])
    cm        = classification_metrics(scores, true_mal, THETA)

    return {
        "run_id":       run_id,
        "mean_latency": float(np.mean(lats)) if lats else -1,
        "handovers":    handovers,
        "f1":           cm["f1"],
        "recall":       cm["recall"],
        "precision":    cm["precision"],
        "fpr":          cm["fpr"],
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["topology"]["ground_nodes"] = 904
    base_cfg["simulation"]["num_rounds"] = NUM_ROUNDS

    results = {}

    for label, use_3gpp in [("random_graph", False), ("3gpp_ntn", True)]:
        print(f"\n[Exp10] Channel model: {label}")
        run_results = []
        for run_id in tqdm(range(NUM_RUNS), desc=label):
            r = run_single(base_cfg, run_id, use_3gpp)
            run_results.append(r)

        lats   = [r["mean_latency"] for r in run_results if r["mean_latency"] > 0]
        f1s    = [r["f1"] for r in run_results]
        fprs   = [r["fpr"] for r in run_results]
        recs   = [r["recall"] for r in run_results]
        h_counts = [r["handovers"] for r in run_results]

        results[label] = {
            "detection_latency_mean": round(float(np.mean(lats)), 2) if lats else -1,
            "detection_latency_ci95": round(1.96 * float(np.std(lats))
                                            / np.sqrt(max(len(lats), 1)), 2),
            "f1_mean":      round(float(np.mean(f1s)), 4),
            "recall_mean":  round(float(np.mean(recs)), 4),
            "fpr_mean":     round(float(np.mean(fprs)), 4),
            "handovers_mean": round(float(np.mean(h_counts)), 1),
            "n_runs": NUM_RUNS,
        }

        lat_str = f"{results[label]['detection_latency_mean']}" \
                  f"+-{results[label]['detection_latency_ci95']}"
        print(f"  Latency: {lat_str} rounds  |  "
              f"F1={results[label]['f1_mean']:.3f}  |  "
              f"FPR={results[label]['fpr_mean']:.4f}  |  "
              f"Handovers/run={results[label]['handovers_mean']:.0f}")

    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp10_3gpp_channel.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[Exp10] Saved -> results/data/exp10_3gpp_channel.json")


if __name__ == "__main__":
    main()
