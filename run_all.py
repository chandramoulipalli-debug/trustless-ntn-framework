"""
Master runner: executes all 6 experiments sequentially.
Usage:
    python run_all.py              # run everything
    python run_all.py --exp 1 3    # run only experiments 1 and 3
"""

import sys
import os
import time
import argparse


EXPERIMENTS = [
    ("exp1_trust_dynamics",    "experiments.exp1_trust_dynamics"),
    ("exp2_delay_partition",   "experiments.exp2_delay_partition"),
    ("exp3_crypto_overhead",   "experiments.exp3_crypto_overhead"),
    ("exp4_privacy_utility",   "experiments.exp4_privacy_utility"),
    ("exp5_scalability",       "experiments.exp5_scalability"),
    ("exp6_attack_comparison", "experiments.exp6_attack_comparison"),
]


def run_experiment(name: str, module_path: str):
    import importlib
    print(f"\n{'='*60}")
    print(f"  Running: {name}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    mod = importlib.import_module(module_path)
    mod.main()
    elapsed = time.perf_counter() - t0
    print(f"  Finished {name} in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", type=int,
                        help="Experiment numbers to run (1-6). Default: all.")
    args = parser.parse_args()

    # Ensure src is importable
    sys.path.insert(0, os.path.dirname(__file__))

    to_run = args.exp if args.exp else list(range(1, len(EXPERIMENTS) + 1))

    total_start = time.perf_counter()
    for idx in to_run:
        name, module = EXPERIMENTS[idx - 1]
        run_experiment(name, module)

    total = time.perf_counter() - total_start
    print(f"\n{'='*60}")
    print(f"  All experiments complete. Total time: {total:.1f}s")
    print(f"  Results saved in: results/data/")
    print(f"  Open notebooks/results_visualization.ipynb to generate figures.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
