"""
Experiment 3: Cryptographic Overhead

Measures:
- Paillier HE: encryption time, aggregation time, decryption time
  vs. number of contributors (10, 50, 100, 200, 500)
- Ciphertext size (KB) vs. contributors
- Simulated ledger consensus throughput (TPS) vs. network size
- MPC: simulated round-trip overhead for 2, 3, 5 domains

Output: results/data/exp3_crypto_overhead.json
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import yaml
from tqdm import tqdm

from src.crypto.paillier_agg import PaillierAggregator


def benchmark_paillier(cfg: dict) -> list[dict]:
    key_bits = cfg["crypto"]["paillier_key_bits"]
    agg = PaillierAggregator(key_bits=key_bits)

    contributor_counts = [5, 10, 25, 50, 100, 200]
    results = []
    print(f"  [Paillier] Key size: {key_bits} bits")
    for n in tqdm(contributor_counts, desc="  Paillier benchmark"):
        # Run 5 repetitions for stability
        enc_times, agg_times, dec_times = [], [], []
        for _ in range(5):
            r = agg.benchmark(n)
            enc_times.append(r["enc_time_ms"])
            agg_times.append(r["agg_time_ms"])
            dec_times.append(r["dec_time_ms"])
        results.append({
            "n_contributors": n,
            "enc_time_ms": float(np.mean(enc_times)),
            "agg_time_ms": float(np.mean(agg_times)),
            "dec_time_ms": float(np.mean(dec_times)),
            "total_time_ms": float(np.mean(enc_times) + np.mean(agg_times) + np.mean(dec_times)),
            "total_time_std": float(np.std([e+a+d for e,a,d in zip(enc_times,agg_times,dec_times)])),
            "ciphertext_size_kb": (key_bits // 4 * n) / 1024,
        })
    return results


def benchmark_mpc_simulated(cfg: dict) -> list[dict]:
    """
    Simulate MPC overhead: each domain contributes one encrypted partial evaluation.
    Cost scales with number of domains and secret-sharing rounds.
    """
    domain_counts = [2, 3, 5]
    results = []
    rng = np.random.default_rng(cfg["simulation"]["seed"])

    for n_domains in domain_counts:
        # Simulate secret-sharing: n_domains round-trips at consensus delay
        round_trip_ms = float(rng.uniform(*cfg["ledger"]["consensus_delay_ms"]))
        total_ms = n_domains * round_trip_ms
        results.append({
            "n_domains": n_domains,
            "simulated_mpc_latency_ms": total_ms,
            "round_trip_ms": round_trip_ms,
        })
    return results


def benchmark_consensus_tps(cfg: dict) -> list[dict]:
    """
    Simulate ledger throughput vs. batch size and consensus delay.
    """
    from src.ledger.simulated_ledger import SimulatedLedger
    rng = np.random.default_rng(cfg["simulation"]["seed"])
    results = []
    batch_sizes = [5, 10, 20, 50, 100]

    for batch in batch_sizes:
        cfg_copy = {**cfg, "ledger": {**cfg["ledger"], "batch_size": batch}}
        ledger = SimulatedLedger(cfg_copy, rng)
        for i in range(500):
            ledger.tick(50.0)
            ledger.submit_record(i % 10, (i + 1) % 10,
                                  float(rng.uniform(0.5, 1.0)), 20.0)
        results.append({
            "batch_size": batch,
            "avg_tps": ledger.avg_tps(),
            "total_records": ledger.total_records(),
            "ledger_size_kb": ledger.ledger_size_kb(),
        })
    return results


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("[Exp3] Benchmarking Paillier HE...")
    paillier_results = benchmark_paillier(cfg)

    print("[Exp3] Benchmarking simulated MPC...")
    mpc_results = benchmark_mpc_simulated(cfg)

    print("[Exp3] Benchmarking ledger consensus throughput...")
    consensus_results = benchmark_consensus_tps(cfg)

    output = {
        "paillier": paillier_results,
        "mpc": mpc_results,
        "consensus": consensus_results,
    }
    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp3_crypto_overhead.json", "w") as f:
        json.dump(output, f, indent=2)
    print("[Exp3] Done. Results saved to results/data/exp3_crypto_overhead.json")
    print("[Exp3] Paillier overhead (100 contributors):",
          next(r for r in paillier_results if r["n_contributors"] == 100)["total_time_ms"],
          "ms")


if __name__ == "__main__":
    main()
