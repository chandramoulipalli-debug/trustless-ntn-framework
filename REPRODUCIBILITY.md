# Reproducibility Guide

Every figure and table in the paper can be reproduced from scratch
using the steps below. Total runtime on a dual-core i5 laptop: ~4–6 hours.

## 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/chandramoulipalli/trustless-ntn-framework.git
cd trustless-ntn-framework

# Create virtual environment (Python 3.11 required)
py -3.11 -m venv venv311          # Windows
python3.11 -m venv venv311        # Linux/Mac

# Activate
venv311\Scripts\activate           # Windows
source venv311/bin/activate        # Linux/Mac

# Install exact pinned dependencies
pip install -r requirements.txt
```

## 2. Configuration

All hyperparameters are in `configs/simulation_config.yaml`.
The defaults reproduce the paper exactly.
Key parameters:

| Parameter | Value | Paper Reference |
|---|---|---|
| `simulation.seed` | 42 | Fixed for reproducibility |
| `simulation.num_runs` | 30 | 30 repetitions for 95% CI |
| `trust.alpha` | 0.7 | Eq. 4 weight |
| `trust.lambda_uav` | 0.05 | Per-segment decay (UAV) |
| `trust.lambda_geo` | 0.005 | Per-segment decay (GEO) |
| `trust.threshold_theta` | 0.4 | Detection threshold θ |
| `attacks.malicious_fraction` | 0.20 | 20% malicious nodes |

## 3. Run All Experiments

```bash
python run_all.py
```

Or run individual experiments:

```bash
python run_all.py --exp 1        # Trust dynamics
python run_all.py --exp 2        # Delay & partition
python run_all.py --exp 3        # Crypto overhead
python run_all.py --exp 4        # Privacy-utility
python run_all.py --exp 5        # Scalability
python run_all.py --exp 6        # Attack comparison
```

Results are written to `results/data/exp*.json`.

## 4. Generate Figures

```bash
jupyter notebook notebooks/results_visualization.ipynb
```

Run all cells. Figures are saved to `results/figures/`.

## 5. Run Unit Tests

```bash
pytest tests/ -v
```

Expected: all tests pass. Tests verify mathematical properties of
Eq. 4 (decay, boundedness, per-segment λ ordering) and Paillier HE
correctness, independently of simulation scale.

## 6. Quick Smoke Test (5 minutes)

To verify the codebase runs without errors (reduced scale):

```bash
python -c "
import yaml
with open('configs/simulation_config.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['simulation']['num_runs'] = 3
cfg['simulation']['num_rounds'] = 50
with open('configs/simulation_config.yaml','w') as f:
    yaml.dump(cfg, f)
"
python run_all.py
```

## 7. Hardware Used in Paper

- CPU: Intel Core i5-7200U @ 2.50 GHz (2 cores, 4 threads)
- RAM: 16 GB
- OS: Windows 10 Home
- Python: 3.11.2
- All experiments single-threaded (no GPU required)

## 8. Archived Version

A frozen snapshot of this repository (including all result JSON files)
is archived at: [Zenodo DOI — add after submission]

This DOI ensures the exact version used in the paper is permanently accessible.
