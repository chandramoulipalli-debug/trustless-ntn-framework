"""
Experiment 8: Paillier HE overhead — 1024-bit vs 2048-bit key comparison

NIST SP 800-131A Rev.2 recommends minimum 2048-bit moduli for RSA/DH.
IEEE security reviewers will flag 1024-bit as the primary benchmark.
This experiment benchmarks both key sizes for direct comparison.

Output: results/data/exp8_paillier_2048.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from src.crypto.paillier_agg import PaillierAggregator


def main():
    with open("configs/simulation_config.yaml") as f:
        cfg = yaml.safe_load(f)

    contributor_counts = [5, 10, 25, 50, 100, 200]
    output = {}

    for key_bits in [1024, 2048]:
        print(f"\n[Exp8] Benchmarking Paillier at {key_bits}-bit key...")
        agg = PaillierAggregator(key_bits=key_bits)
        results = []
        for n in contributor_counts:
            r = agg.benchmark(n)
            results.append(r)
            print(f"  n={n:3d}: enc={r['enc_time_ms']:.1f}ms  "
                  f"agg={r['agg_time_ms']:.1f}ms  "
                  f"dec={r['dec_time_ms']:.1f}ms  "
                  f"total={r['total_time_ms']:.1f}ms  "
                  f"size={r['ciphertext_size_kb']:.2f}KB")
        output[f"key_bits_{key_bits}"] = results

    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp8_paillier_2048.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n[Exp8] Done -> results/data/exp8_paillier_2048.json")


if __name__ == "__main__":
    main()
