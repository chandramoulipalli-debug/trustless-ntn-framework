"""
Experiment 12: Transitive (Indirect) Trust Propagation

Evaluates whether indirect trust propagation via DLT-consensus scores improves
Recall without breaking the zero-FPR guarantee at theta* = 0.20.

Problem in direct-only model:
  - Node 0 (single observer) contacts ~500/1000 nodes in 500 rounds.
  - Malicious nodes never observed directly stay near init trust (0.50 > theta*).
  - Recall ≈ 0.14 (only directly-observed malicious nodes detected).

Indirect trust mechanism (Layer 1 DLT integration):
  - All N nodes publish their trust vectors to the DLT after each round.
  - For node k not directly observed by node 0:
    T_indirect[k] = weighted average of T[j,k] across all j that observed k,
                    weighted by T[0,j] (how much node 0 trusts intermediary j).
  - T_combined[k] = blend of direct and indirect:
    w_d = min(n_direct_interactions / 20, 0.60)   (caps at 60% direct)
    T_combined[k] = w_d * T_direct[k] + (1-w_d) * T_indirect[k]

Zero-FPR preservation analysis:
  - For malicious k (never observed by 0):
    Most intermediaries j report T[j,k] ≈ T*_mal ≈ 0.042
    T_indirect[k] ≈ 0.042 << theta*=0.20  → flagged ✓
  - For legitimate k (never observed by 0):
    Most intermediaries j report T[j,k] ≈ T*_legit ≈ 0.356
    T_indirect[k] ≈ 0.356 >> theta*=0.20  → not flagged ✓

Measures:
  - Recall improvement: direct-only vs direct+indirect
  - FPR: must remain ≈ 0 under consensus weighting
  - Detection latency: rounds to first consensus flag via indirect evidence

Output: results/data/exp12_transitive_trust.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import TrustEngine
from src.attacks.attack_models import assign_attack_roles
from src.metrics.evaluator import classification_metrics

NUM_RUNS   = 20
NUM_ROUNDS = 500
THETA      = 0.20
GAMMA      = 0.40      # max weight for indirect evidence when n_direct=0


def run_single(cfg: dict, run_id: int) -> dict:
    seed = cfg["simulation"]["seed"] + run_id + 3000
    rng  = np.random.default_rng(seed)

    G, type_to_ids = build_ntn_graph(cfg, rng)
    nodes      = list(G.nodes())
    N          = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    model = TrustEngine(N, cfg, node_types, rng)

    # Interaction counter: interaction_counts[i,j] = # direct observations by i of j
    interaction_counts = np.zeros((N, N), dtype=np.int32)

    true_mal = np.array([jj in malicious_ids for jj in nodes])

    # Simulate rounds — all nodes update from ALL perspectives
    # (so the DLT has full network trust evidence for indirect trust)
    for rnd in range(NUM_ROUNDS):
        for i in nodes:
            nbrs = list(G.neighbors(i))
            if not nbrs:
                continue
            j = int(rng.choice(nbrs))
            quality  = float(quality_fns[j](j, rnd))
            delay_ms = G[i][j]["delay_ms"]
            model.update(i, j, quality, delay_ms)
            interaction_counts[i, j] += 1

    # ── Classify at round 500: direct-only ───────────────────────────────────
    direct_scores  = np.array([model.get(0, jj) for jj in nodes])
    cm_direct      = classification_metrics(direct_scores, true_mal, THETA)

    # ── Classify at round 500: direct + indirect ──────────────────────────────
    combined_scores = model.combined_trust_scores(0, interaction_counts, gamma=GAMMA)
    cm_combined     = classification_metrics(combined_scores, true_mal, THETA)

    # ── Recall breakdown: how many of node 0's missed malicious nodes
    #    are now captured by indirect trust? ──────────────────────────────────
    direct_flag   = direct_scores   < THETA    # bool (N,)
    combined_flag = combined_scores < THETA

    mal_mask = true_mal
    # Newly detected by indirect: was false-negative in direct, now true-positive
    newly_detected = int(np.sum(combined_flag & mal_mask & ~direct_flag))
    # New false positives introduced: was true-neg in direct, now false-pos
    new_fps        = int(np.sum(combined_flag & ~mal_mask & ~direct_flag))

    # How many malicious nodes has node 0 NEVER directly observed?
    n_never_observed = int(np.sum((interaction_counts[0, :] == 0) & mal_mask))

    return {
        # Direct-only metrics
        "direct_recall":    round(cm_direct["recall"],    4),
        "direct_precision": round(cm_direct["precision"], 4),
        "direct_f1":        round(cm_direct["f1"],        4),
        "direct_fpr":       round(cm_direct["fpr"],       4),
        # Combined metrics
        "combined_recall":    round(cm_combined["recall"],    4),
        "combined_precision": round(cm_combined["precision"], 4),
        "combined_f1":        round(cm_combined["f1"],        4),
        "combined_fpr":       round(cm_combined["fpr"],       4),
        # Incremental breakdown
        "newly_detected":    newly_detected,
        "new_fps":           new_fps,
        "n_never_observed":  n_never_observed,
    }


def agg(run_results, key):
    vals = [r[key] for r in run_results]
    m    = float(np.mean(vals))
    ci   = 1.96 * float(np.std(vals)) / np.sqrt(max(len(vals), 1))
    return round(m, 4), round(ci, 4)


def main():
    with open("configs/simulation_config.yaml") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["topology"]["ground_nodes"] = 904
    base_cfg["simulation"]["num_rounds"] = NUM_ROUNDS

    print(f"[Exp12] Transitive trust: {NUM_RUNS} runs x {NUM_ROUNDS} rounds, "
          f"theta={THETA}, gamma={GAMMA}")

    run_results = []
    for run_id in tqdm(range(NUM_RUNS), desc="exp12"):
        run_results.append(run_single(base_cfg, run_id))

    results = {}
    for label in ("direct", "combined"):
        rec_m,  rec_ci  = agg(run_results, f"{label}_recall")
        prec_m, prec_ci = agg(run_results, f"{label}_precision")
        f1_m,   f1_ci   = agg(run_results, f"{label}_f1")
        fpr_m,  fpr_ci  = agg(run_results, f"{label}_fpr")
        results[label] = {
            "recall_mean":    rec_m,    "recall_ci95":    rec_ci,
            "precision_mean": prec_m,   "precision_ci95": prec_ci,
            "f1_mean":        f1_m,     "f1_ci95":        f1_ci,
            "fpr_mean":       fpr_m,    "fpr_ci95":       fpr_ci,
        }

    nd_m, nd_ci = agg(run_results, "newly_detected")
    fp_m, fp_ci = agg(run_results, "new_fps")
    no_m, no_ci = agg(run_results, "n_never_observed")
    results["incremental"] = {
        "newly_detected_mean": nd_m, "newly_detected_ci95": nd_ci,
        "new_fps_mean":        fp_m, "new_fps_ci95":        fp_ci,
        "n_never_observed_mean": no_m,
    }

    print("\n[Exp12] Results:")
    print(f"  Direct-only: Recall={results['direct']['recall_mean']:.4f}  "
          f"F1={results['direct']['f1_mean']:.4f}  "
          f"FPR={results['direct']['fpr_mean']:.4f}")
    print(f"  Direct+Indirect: Recall={results['combined']['recall_mean']:.4f}  "
          f"F1={results['combined']['f1_mean']:.4f}  "
          f"FPR={results['combined']['fpr_mean']:.4f}")
    print(f"  Newly detected: {nd_m:.1f}  |  New FPs: {fp_m:.1f}  "
          f"|  Never-observed malicious: {no_m:.1f}")

    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp12_transitive_trust.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[Exp12] Saved -> results/data/exp12_transitive_trust.json")


if __name__ == "__main__":
    main()
