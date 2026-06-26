"""
Experiment 5: Scalability Analysis

Varies total node count N from 100 to 5000.
Measures per N:
- Trust computation time per round (ms)
- Ledger size growth (KB)
- Consensus latency (ms)
- Memory usage (MB)

Designed to show that overhead grows sub-quadratically due to sparse NTN topology.

Output: results/data/exp5_scalability.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
import psutil
from tqdm import tqdm

from src.trust.trust_engine import TrustEngine
from src.ledger.simulated_ledger import SimulatedLedger


def build_minimal_topology(n_nodes: int, rng: np.random.Generator,
                            cfg: dict) -> tuple:
    """Build a simple random graph for scalability tests (faster than full NTN builder)."""
    import networkx as nx

    # Erdos-Renyi sparse graph — representative of NTN connectivity
    p = min(0.05, 5.0 / n_nodes)
    G = nx.erdos_renyi_graph(n_nodes, p, seed=int(rng.integers(0, 2**31)))
    if not nx.is_connected(G):
        # Add random spanning edges
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            G.add_edge(u, v)

    for u, v in G.edges():
        G[u][v]["delay_ms"] = float(rng.uniform(5, 300))

    node_types_all = ["LEO", "MEO", "GEO", "HAPS", "UAV", "GROUND"]
    node_types = [node_types_all[i % len(node_types_all)] for i in range(n_nodes)]
    return G, node_types


def measure_at_scale(n_nodes: int, cfg: dict, rng: np.random.Generator,
                     n_rounds: int = 50) -> dict:
    G, node_types = build_minimal_topology(n_nodes, rng, cfg)
    nodes = list(G.nodes())
    model = TrustEngine(n_nodes, cfg, node_types, rng)
    ledger = SimulatedLedger(cfg, rng)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB

    round_times = []
    for rnd in range(n_rounds):
        ledger.tick(100.0)
        t0 = time.perf_counter()
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality = float(rng.uniform(0.5, 1.0))
            model.update(i, j, quality, delay_ms)
            ledger.submit_record(i, j, quality, delay_ms)
        round_times.append((time.perf_counter() - t0) * 1000)

    mem_after = process.memory_info().rss / (1024 * 1024)

    return {
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "mean_round_time_ms": float(np.mean(round_times)),
        "std_round_time_ms": float(np.std(round_times)),
        "ledger_size_kb": ledger.ledger_size_kb(),
        "avg_tps": ledger.avg_tps(),
        "memory_delta_mb": mem_after - mem_before,
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    node_counts = cfg["scalability"]["node_counts"]
    results = []
    print(f"[Exp5] Scalability sweep over {node_counts} node counts...")
    for n in tqdm(node_counts):
        rng = np.random.default_rng(cfg["simulation"]["seed"] + n)
        r = measure_at_scale(n, cfg, rng)
        results.append(r)
        print(f"  N={n:5d}: {r['mean_round_time_ms']:.2f} ms/round, "
              f"ledger={r['ledger_size_kb']:.1f} KB, "
              f"mem Δ={r['memory_delta_mb']:.1f} MB")

    output = {"results": results}
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp5_scalability.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp5] Done. Results saved to results/data/exp5_scalability.json")


if __name__ == "__main__":
    main()
