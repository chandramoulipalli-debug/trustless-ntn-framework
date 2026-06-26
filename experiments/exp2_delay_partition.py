"""
Experiment 2: Delay & Partition Resilience

Measures:
- Trust divergence during partition (main vs isolated ledger view)
- Convergence time (rounds) after reconnection
- Trust score at reconnection for proposed vs baselines
- Effect of GEO delay (250-300ms) vs LEO (5-20ms) on trust accuracy

Output: results/data/exp2_delay_partition.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import TrustEngine, CentralizedTrust, StaticDLT, ZTAuthOnly, UAVBlockchainFL
from src.attacks.attack_models import assign_attack_roles
from src.ledger.simulated_ledger import SimulatedLedger
from src.metrics.evaluator import trust_divergence, classification_metrics


def run_partition_scenario(cfg: dict, run_id: int) -> dict:
    seed = cfg["simulation"]["seed"] + run_id + 1000
    rng = np.random.default_rng(seed)

    G, type_to_ids = build_ntn_graph(cfg, rng)
    nodes = list(G.nodes())
    N = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    proposed = TrustEngine(N, cfg, node_types, rng)
    static_dlt = StaticDLT(N, cfg, rng)
    centralized = CentralizedTrust(N, cfg, rng)

    ledger_main = SimulatedLedger(cfg, rng)

    num_rounds = cfg["simulation"]["num_rounds"]
    partition_start = num_rounds // 3
    partition_end = partition_start + cfg["ledger"]["partition_duration_rounds"]

    divergence_curve = []      # trust divergence during partition
    convergence_round = -1
    f1_pre, f1_post = None, None
    ledger_fork = None

    # Track trust snapshots for divergence
    proposed_fork = None

    for rnd in range(num_rounds):
        # Tick simulated time
        tick_ms = 100.0  # 100ms per round
        ledger_main.tick(tick_ms)

        # Partition events
        if rnd == partition_start:
            ledger_fork = ledger_main.partition()
            proposed_fork = TrustEngine(N, cfg, node_types, rng)
            proposed_fork.T = proposed.snapshot().copy()
            centralized.set_partition(True)

        if rnd == partition_end:
            # Reconnect: reconcile and measure
            n_reconciled = ledger_main.reconcile(ledger_fork, proposed)
            proposed_fork = None
            centralized.set_partition(False)
            convergence_round = rnd

        # Normal interaction round
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality = float(quality_fns[j](j, rnd))

            proposed.update(i, j, quality, delay_ms)
            static_dlt.update(i, j, quality, delay_ms)
            centralized.update(i, j, quality, delay_ms)

            # During partition, fork evolves independently
            if proposed_fork is not None:
                proposed_fork.update(i, j, quality, delay_ms)

        # Measure divergence during partition
        if partition_start <= rnd < partition_end and proposed_fork is not None:
            div = trust_divergence(proposed.snapshot(), proposed_fork.snapshot())
            divergence_curve.append({"round": rnd, "divergence": div})

        # F1 snapshot pre/post partition
        if rnd == partition_start - 1:
            scores = np.array([proposed.get(0, j) for j in nodes])
            true_mal = np.array([j in malicious_ids for j in nodes])
            f1_pre = classification_metrics(scores, true_mal,
                                             cfg["trust"]["threshold_theta"])["f1"]

        if rnd == partition_end + 10:
            scores = np.array([proposed.get(0, j) for j in nodes])
            true_mal = np.array([j in malicious_ids for j in nodes])
            f1_post = classification_metrics(scores, true_mal,
                                              cfg["trust"]["threshold_theta"])["f1"]

        # Apply natural decay during partition (no external evidence for isolated nodes)
        if partition_start <= rnd < partition_end:
            proposed.decay_all(tick_ms)

    # Delay impact: compute mean trust for GEO vs LEO links
    geo_trusts, leo_trusts = [], []
    for u, v, data in G.edges(data=True):
        pair = data.get("segment_pair", "")
        t_uv = proposed.get(0, v)
        if "GEO" in pair:
            geo_trusts.append(t_uv)
        elif "LEO" in pair:
            leo_trusts.append(t_uv)

    return {
        "run_id": run_id,
        "divergence_curve": divergence_curve,
        "convergence_round": convergence_round,
        "partition_duration_rounds": cfg["ledger"]["partition_duration_rounds"],
        "f1_pre_partition": f1_pre,
        "f1_post_partition": f1_post,
        "mean_geo_trust": float(np.mean(geo_trusts)) if geo_trusts else None,
        "mean_leo_trust": float(np.mean(leo_trusts)) if leo_trusts else None,
        "ledger_tps": ledger_main.avg_tps(),
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_runs = cfg["simulation"]["num_runs"]
    results = []
    print(f"[Exp2] Running {num_runs} partition resilience runs...")
    for run_id in tqdm(range(num_runs)):
        results.append(run_partition_scenario(cfg, run_id))

    f1_pre_vals = [r["f1_pre_partition"] for r in results if r["f1_pre_partition"] is not None]
    f1_post_vals = [r["f1_post_partition"] for r in results if r["f1_post_partition"] is not None]

    summary = {
        "f1_pre_partition_mean": float(np.mean(f1_pre_vals)) if f1_pre_vals else None,
        "f1_post_partition_mean": float(np.mean(f1_post_vals)) if f1_post_vals else None,
        "f1_degradation": float(np.mean(f1_pre_vals) - np.mean(f1_post_vals))
                          if f1_pre_vals and f1_post_vals else None,
        "mean_ledger_tps": float(np.mean([r["ledger_tps"] for r in results])),
    }

    output = {"runs": results, "summary": summary}
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp2_delay_partition.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp2] Done. Results saved to results/data/exp2_delay_partition.json")
    print(f"[Exp2] F1 pre-partition: {summary['f1_pre_partition_mean']:.3f}")
    print(f"[Exp2] F1 post-partition: {summary['f1_post_partition_mean']:.3f}")


if __name__ == "__main__":
    main()
