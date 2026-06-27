"""
Experiment 11: Multi-Aggregator Consensus

Extends the single-observer (node 0) design to K=5 designated aggregator nodes,
one drawn from each NTN segment (GEO, MEO/LEO, HAPS, UAV, Ground).  Each
aggregator runs an independent TrustEngine.  A majority-vote consensus (≥ K/2
aggregators flag node j as malicious) determines the final binary verdict.

Measures:
  (a) Detection latency under multi-aggregator consensus vs single-observer
  (b) Recall improvement: multi-observer covers the full node population
  (c) Communication overhead: messages per round published to DLT
  (d) FPR under consensus (consensus should suppress individual false positives)

Network overhead model:
  - Each aggregator publishes N trust scores per round (float32 = 4 bytes)
  - Overhead = K * N * 4 bytes / round (before DLT encoding)
  - Reported in KB/round

Output: results/data/exp11_multi_aggregator.json
"""

import sys, os, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import TrustEngine
from src.attacks.attack_models import assign_attack_roles
from src.metrics.evaluator import detection_latency_all, classification_metrics

NUM_RUNS   = 20
NUM_ROUNDS = 500
THETA      = 0.20
K_AGG      = 5        # number of aggregator nodes
BYTES_PER_SCORE = 4   # float32 per trust score published to DLT


def select_aggregators(type_to_ids: dict, rng: np.random.Generator) -> list[int]:
    """
    Select K_AGG aggregator nodes, one per NTN segment where available.
    Segment priority: GEO, LEO, HAPS, UAV, GROUND.
    """
    agg_ids = []
    priority = ["GEO", "LEO", "MEO", "HAPS", "UAV", "GROUND"]
    for seg in priority:
        ids = type_to_ids.get(seg, [])
        if ids and len(agg_ids) < K_AGG:
            agg_ids.append(int(rng.choice(ids)))
    # Pad with random ground nodes if needed
    ground = type_to_ids.get("GROUND", [])
    while len(agg_ids) < K_AGG and ground:
        nid = int(rng.choice(ground))
        if nid not in agg_ids:
            agg_ids.append(nid)
    return agg_ids[:K_AGG]


def run_single(cfg: dict, run_id: int, multi_agg: bool) -> dict:
    seed = cfg["simulation"]["seed"] + run_id + (2000 if multi_agg else 0)
    rng  = np.random.default_rng(seed)

    G, type_to_ids = build_ntn_graph(cfg, rng)
    nodes      = list(G.nodes())
    N          = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    mal_start = cfg["attacks"]["build_betray_buildup"]

    # ── Single-observer (node 0) ─────────────────────────────────────────────
    if not multi_agg:
        model = TrustEngine(N, cfg, node_types, rng)
        trust_history = {j: [] for j in nodes}

        for rnd in range(NUM_ROUNDS):
            for i in nodes:
                nbrs = list(G.neighbors(i))
                if not nbrs:
                    continue
                j = int(rng.choice(nbrs))
                model.update(i, j, float(quality_fns[j](j, rnd)),
                             G[i][j]["delay_ms"])
            for jj in nodes:
                trust_history[jj].append(model.get(0, jj))

        lats   = detection_latency_all(trust_history, malicious_ids, mal_start, THETA)
        scores = np.array([model.get(0, jj) for jj in nodes])
        true_m = np.array([jj in malicious_ids for jj in nodes])
        cm     = classification_metrics(scores, true_m, THETA)
        return {
            "mean_latency": float(np.mean(lats)) if lats else -1,
            "f1": cm["f1"], "recall": cm["recall"],
            "precision": cm["precision"], "fpr": cm["fpr"],
            "overhead_kb_per_round": 0.0,
        }

    # ── Multi-aggregator (K=5 independent TrustEngines) ──────────────────────
    agg_ids = select_aggregators(type_to_ids, rng)

    # One TrustEngine per aggregator; each runs same simulation but from its own
    # perspective (it observes its own neighbours)
    engines = {agg: TrustEngine(N, cfg, node_types, rng) for agg in agg_ids}
    # trust_history[agg][j] = time-series of agg's trust in j
    trust_histories = {agg: {j: [] for j in nodes} for agg in agg_ids}

    for rnd in range(NUM_ROUNDS):
        for i in nodes:
            nbrs = list(G.neighbors(i))
            if not nbrs:
                continue
            j = int(rng.choice(nbrs))
            quality  = float(quality_fns[j](j, rnd))
            delay_ms = G[i][j]["delay_ms"]
            # Update the TrustEngine for each aggregator that is 'i'
            for agg in agg_ids:
                if i == agg:
                    engines[agg].update(i, j, quality, delay_ms)

        # Each aggregator also updates via its direct neighbours (observe randomly)
        for agg in agg_ids:
            nbrs_agg = list(G.neighbors(agg))
            if not nbrs_agg:
                continue
            j = int(rng.choice(nbrs_agg))
            engines[agg].update(agg, j,
                                float(quality_fns[j](j, rnd)),
                                G[agg][j]["delay_ms"])

        for agg in agg_ids:
            for jj in nodes:
                trust_histories[agg][jj].append(engines[agg].get(agg, jj))

    # Consensus verdict at final round: majority vote across aggregators
    # Latency: earliest round where MAJORITY of aggregators flag node j
    latencies = []
    final_votes = np.zeros(N, dtype=float)

    for agg in agg_ids:
        for jj in malicious_ids:
            th = trust_histories[agg][jj]
            for rnd in range(mal_start, NUM_ROUNDS):
                if th[rnd] < THETA:
                    latencies.append(rnd - mal_start)
                    break

        # Aggregator's final binary vote for each node
        final_scores = np.array([engines[agg].get(agg, jj) for jj in nodes])
        final_votes += (final_scores < THETA).astype(float)

    # Consensus: node flagged if >= ceil(K/2) aggregators flag it
    consensus_flag = final_votes >= np.ceil(K_AGG / 2)
    true_m = np.array([jj in malicious_ids for jj in nodes])

    # Compute consensus metrics
    TP = int(np.sum(consensus_flag & true_m))
    FP = int(np.sum(consensus_flag & ~true_m))
    TN = int(np.sum(~consensus_flag & ~true_m))
    FN = int(np.sum(~consensus_flag & true_m))

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fpr       = FP / (FP + TN + 1e-9)

    # Communication overhead: each aggregator publishes N float32 trust scores/round
    overhead_bytes_per_round = K_AGG * N * BYTES_PER_SCORE
    overhead_kb_per_round    = overhead_bytes_per_round / 1024.0

    return {
        "mean_latency": float(np.mean(latencies)) if latencies else -1,
        "f1":       round(f1, 4),
        "recall":   round(recall, 4),
        "precision": round(precision, 4),
        "fpr":      round(fpr, 4),
        "overhead_kb_per_round": round(overhead_kb_per_round, 2),
        "n_aggregators": K_AGG,
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["topology"]["ground_nodes"] = 904
    base_cfg["simulation"]["num_rounds"] = NUM_ROUNDS

    results = {}

    for label, multi_agg in [("single_observer", False), ("multi_aggregator", True)]:
        print(f"\n[Exp11] {label} (K={K_AGG if multi_agg else 1})")
        run_results = []
        for run_id in tqdm(range(NUM_RUNS), desc=label):
            r = run_single(base_cfg, run_id, multi_agg)
            run_results.append(r)

        def agg_metric(key):
            vals = [r[key] for r in run_results if r.get(key, -1) > 0]
            if not vals:
                return -1, 0
            return round(float(np.mean(vals)), 3), round(1.96 * float(np.std(vals)) /
                                                         np.sqrt(len(vals)), 3)

        lat_m, lat_ci = agg_metric("mean_latency")
        f1_m,  _      = agg_metric("f1")
        rec_m, _      = agg_metric("recall")
        fpr_m, _      = agg_metric("fpr")
        oh_m,  _      = agg_metric("overhead_kb_per_round")

        results[label] = {
            "detection_latency_mean": lat_m,
            "detection_latency_ci95": lat_ci,
            "f1_mean":          f1_m,
            "recall_mean":      rec_m,
            "fpr_mean":         fpr_m,
            "overhead_kb_per_round": oh_m,
        }

        print(f"  Latency={lat_m}+-{lat_ci}  F1={f1_m}  "
              f"Recall={rec_m}  FPR={fpr_m}  "
              f"Overhead={oh_m} KB/round")

    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp11_multi_aggregator.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[Exp11] Saved -> results/data/exp11_multi_aggregator.json")


if __name__ == "__main__":
    main()
