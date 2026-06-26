"""
Experiment 1: Trust Dynamics & Malicious Behavior Detection

Measures:
- Trust score evolution curves for: on-off attacker, build-then-betray, normal node
- Detection latency (rounds until trust < theta) per model
- F1 score at each round for all 5 models

Output: results/data/exp1_trust_dynamics.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.topology.ntn_topology import build_ntn_graph
from src.trust.trust_engine import TrustEngine, CentralizedTrust, StaticDLT, ZTAuthOnly, UAVBlockchainFL
from src.attacks.attack_models import assign_attack_roles
from src.metrics.evaluator import detection_latency_all, classification_metrics, summarize_runs


def run_single(cfg: dict, run_id: int) -> dict:
    seed = cfg["simulation"]["seed"] + run_id
    rng = np.random.default_rng(seed)

    G, type_to_ids = build_ntn_graph(cfg, rng)
    nodes = list(G.nodes())
    N = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    # Assign attack roles
    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    # Initialise all models
    models = {
        "proposed":    TrustEngine(N, cfg, node_types, rng),
        "centralized": CentralizedTrust(N, cfg, rng),
        "static_dlt":  StaticDLT(N, cfg, rng),
        "zt_auth":     ZTAuthOnly(N, cfg, rng),
        "uav_fl":      UAVBlockchainFL(N, cfg, rng),
    }

    num_rounds = cfg["simulation"]["num_rounds"]
    # Track trust of observer node 0 toward each other node per round
    trust_history = {name: {j: [] for j in nodes} for name in models}
    f1_history = {name: [] for name in models}

    # Find a target malicious node for curve plotting
    target_mal = next(iter(malicious_ids)) if malicious_ids else nodes[1]
    target_legit = next(n for n in nodes if n not in malicious_ids and n != 0)

    for rnd in range(num_rounds):
        # Sample interactions: each node interacts with a random subset of neighbours
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality = float(quality_fns[j](j, rnd))

            for model in models.values():
                model.update(i, j, quality, delay_ms)

        # Record trust scores (observer = node 0)
        for name, model in models.items():
            for j in nodes:
                trust_history[name][j].append(model.get(0, j))

        # Compute F1 at this round
        for name, model in models.items():
            scores = np.array([model.get(0, j) for j in nodes])
            true_mal = np.array([j in malicious_ids for j in nodes])
            cm = classification_metrics(scores, true_mal, cfg["trust"]["threshold_theta"])
            f1_history[name].append(cm["f1"])

    # Detection latencies
    mal_start = cfg["attacks"]["build_betray_buildup"]
    detection = {}
    for name, model in models.items():
        lats = detection_latency_all(
            trust_history[name], malicious_ids, mal_start,
            cfg["trust"]["threshold_theta"]
        )
        detection[name] = lats

    return {
        "run_id": run_id,
        "n_nodes": N,
        "n_malicious": len(malicious_ids),
        "trust_curve_malicious": {name: trust_history[name].get(target_mal, [])
                                   for name in models},
        "trust_curve_legit": {name: trust_history[name].get(target_legit, [])
                               for name in models},
        "f1_history": f1_history,
        "detection_latencies": {name: detection[name] for name in models},
        "mean_detection": {
            name: float(np.mean(detection[name])) if detection[name] else -1
            for name in models
        },
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    num_runs = cfg["simulation"]["num_runs"]
    results = []
    print(f"[Exp1] Running {num_runs} runs...")
    for run_id in tqdm(range(num_runs)):
        results.append(run_single(cfg, run_id))

    # Aggregate mean detection latency across runs
    summary = {}
    for name in ["proposed", "centralized", "static_dlt", "zt_auth", "uav_fl"]:
        all_lats = [lat for r in results for lat in r["detection_latencies"].get(name, [])]
        summary[name] = {
            "mean_detection_rounds": float(np.mean(all_lats)) if all_lats else -1,
            "std": float(np.std(all_lats)) if all_lats else 0,
            "ci95": 1.96 * float(np.std(all_lats)) / np.sqrt(max(len(all_lats), 1)),
        }

    output = {"runs": results, "summary": summary}
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp1_trust_dynamics.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp1] Done. Results saved to results/data/exp1_trust_dynamics.json")
    print("[Exp1] Detection latency summary:")
    for name, s in summary.items():
        print(f"  {name:15s}: {s['mean_detection_rounds']:.1f} ± {s['ci95']:.1f} rounds")


if __name__ == "__main__":
    main()
