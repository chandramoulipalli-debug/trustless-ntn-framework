"""
Experiment 9: Round-Period Sensitivity

Varies T_round in {30, 60, 90, 120} s while holding all other parameters
constant (N=1000, f=20%, 500 rounds, 15 runs for 95% CI).

Measures:
- Detection latency for the Proposed model at each T_round
- Empirical T*_legit and T*_mal (steady-state trust, rounds 400-499 mean)
- Confirms analytical predictions from Eq. (9) of the paper

Output: results/data/exp9_round_period.json
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


TROUND_VALUES = [30, 60, 90, 120]
NUM_RUNS = 15   # 15 × 4 = 60 total runs; enough for 95% CI


def analytical_T_star(alpha, lam, delta_t, s_bar):
    """Closed-form steady-state trust from Eq. (9)."""
    L = alpha * np.exp(-lam * delta_t)
    return (1 - alpha) * s_bar / (1 - L)


def run_single(cfg: dict, run_id: int) -> dict:
    seed = cfg["simulation"]["seed"] + run_id
    rng = np.random.default_rng(seed)

    G, _ = build_ntn_graph(cfg, rng)
    nodes = list(G.nodes())
    N = len(nodes)
    node_types = [G.nodes[n]["type"] for n in range(N)]

    quality_fns, malicious_ids = assign_attack_roles(
        nodes, cfg["attacks"]["malicious_fraction"], cfg, rng
    )

    model = TrustEngine(N, cfg, node_types, rng)
    num_rounds = cfg["simulation"]["num_rounds"]
    theta = cfg["trust"]["threshold_theta"]

    trust_history = {j: [] for j in nodes}
    f1_history = []

    for rnd in range(num_rounds):
        for i in nodes:
            neighbours = list(G.neighbors(i))
            if not neighbours:
                continue
            j = int(rng.choice(neighbours))
            delay_ms = G[i][j]["delay_ms"]
            quality = float(quality_fns[j](j, rnd))
            model.update(i, j, quality, delay_ms)

        for j in nodes:
            trust_history[j].append(model.get(0, j))

        scores = np.array([model.get(0, j) for j in nodes])
        true_mal = np.array([j in malicious_ids for j in nodes])
        cm = classification_metrics(scores, true_mal, theta)
        f1_history.append(cm["f1"])

    mal_start = cfg["attacks"]["build_betray_buildup"]
    lats = detection_latency_all(trust_history, malicious_ids, mal_start, theta)

    # Empirical steady-state: mean trust over last 100 rounds
    legit_ids = [j for j in nodes if j not in malicious_ids and j != 0]
    T_star_legit_emp = float(np.mean(
        [np.mean(trust_history[j][400:]) for j in legit_ids if trust_history[j]]
    ))
    T_star_mal_emp = float(np.mean(
        [np.mean(trust_history[j][400:]) for j in malicious_ids if trust_history[j]]
    )) if malicious_ids else 0.0

    return {
        "run_id": run_id,
        "detection_latencies": lats,
        "mean_detection": float(np.mean(lats)) if lats else -1,
        "T_star_legit_empirical": T_star_legit_emp,
        "T_star_mal_empirical": T_star_mal_emp,
        "final_f1": f1_history[-1] if f1_history else 0.0,
    }


def main():
    with open("configs/simulation_config.yaml") as f:
        base_cfg = yaml.safe_load(f)

    base_cfg["simulation"]["num_runs"] = NUM_RUNS
    # Use N=1000 full topology
    base_cfg["topology"]["ground_nodes"] = 904

    alpha = base_cfg["trust"]["alpha"]
    lam_ground = base_cfg["trust"]["lambda_ground"]
    s_legit = 0.85
    s_mal = 0.10

    results = {}

    for t_round in TROUND_VALUES:
        cfg = copy.deepcopy(base_cfg)
        cfg["trust"]["round_duration_sec"] = float(t_round)

        # Analytical prediction (d_bar=0 approximation, Ground segment)
        delta_t = float(t_round)
        T_star_legit_analytic = analytical_T_star(alpha, lam_ground, delta_t, s_legit)
        T_star_mal_analytic   = analytical_T_star(alpha, lam_ground, delta_t, s_mal)
        theta_opt_analytic = (T_star_legit_analytic + T_star_mal_analytic) / 2.0

        print(f"\n[Exp9] T_round={t_round}s  "
              f"analytic T*_legit={T_star_legit_analytic:.3f}  "
              f"T*_mal={T_star_mal_analytic:.3f}  "
              f"theta*={theta_opt_analytic:.3f}")

        run_results = []
        for run_id in tqdm(range(NUM_RUNS), desc=f"T_round={t_round}s"):
            run_results.append(run_single(cfg, run_id))

        all_lats = [lat for r in run_results for lat in r["detection_latencies"]]
        T_legit_vals = [r["T_star_legit_empirical"] for r in run_results]
        T_mal_vals   = [r["T_star_mal_empirical"]   for r in run_results]

        results[str(t_round)] = {
            "t_round_sec": t_round,
            "analytic": {
                "T_star_legit": round(T_star_legit_analytic, 4),
                "T_star_mal":   round(T_star_mal_analytic,   4),
                "theta_opt":    round(theta_opt_analytic,    4),
            },
            "empirical": {
                "T_star_legit_mean": round(float(np.mean(T_legit_vals)), 4),
                "T_star_legit_ci95": round(1.96 * float(np.std(T_legit_vals))
                                           / np.sqrt(NUM_RUNS), 4),
                "T_star_mal_mean":   round(float(np.mean(T_mal_vals)), 4),
                "T_star_mal_ci95":   round(1.96 * float(np.std(T_mal_vals))
                                           / np.sqrt(NUM_RUNS), 4),
                "detection_latency_mean": round(float(np.mean(all_lats)), 2)
                                          if all_lats else -1,
                "detection_latency_ci95": round(1.96 * float(np.std(all_lats))
                                                / np.sqrt(max(len(all_lats), 1)), 2),
            },
        }

    os.makedirs("results/data", exist_ok=True)
    with open("results/data/exp9_round_period.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[Exp9] Summary (Ground segment, λ=0.015 s⁻¹, Proposed model):")
    print(f"{'T_round':>10} {'T*_legit analytic':>20} {'T*_legit empirical':>22} "
          f"{'T*_mal analytic':>18} {'Latency (rounds)':>18}")
    for t in TROUND_VALUES:
        r = results[str(t)]
        print(f"{t:>10}s  {r['analytic']['T_star_legit']:>20.3f}  "
              f"{r['empirical']['T_star_legit_mean']:>18.3f}±"
              f"{r['empirical']['T_star_legit_ci95']:.3f}  "
              f"{r['analytic']['T_star_mal']:>16.3f}  "
              f"{r['empirical']['detection_latency_mean']:>14.1f}±"
              f"{r['empirical']['detection_latency_ci95']:.1f}")

    print("\n[Exp9] Results saved to results/data/exp9_round_period.json")


if __name__ == "__main__":
    main()
