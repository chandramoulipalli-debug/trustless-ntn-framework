"""
Differential Privacy module implementing Eq. 3:
    Q'(D) = Q(D) + N(0, σ²)

Calibrates σ² from privacy budget ε using the Gaussian mechanism.
Measures the privacy-utility tradeoff for Experiment 4.
"""

import numpy as np


class DifferentialPrivacyLayer:
    """
    Applies Gaussian DP noise to aggregate trust queries.
    Prevents statistical inference attacks on trust outputs.
    """

    def __init__(self, epsilon: float, delta: float, sensitivity: float = 1.0):
        """
        epsilon: privacy budget ε (lower = more private, less accurate)
        delta:   failure probability δ
        sensitivity: L2-sensitivity of the query function (default 1.0 for trust in [0,1])
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        # Calibrate σ using Gaussian mechanism formula
        self.sigma = self._calibrate_sigma(epsilon, delta, sensitivity)

    def _calibrate_sigma(self, eps: float, delta: float, sensitivity: float) -> float:
        """
        Gaussian mechanism: σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        """
        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / eps

    def apply(self, query_result: float, rng: np.random.Generator) -> float:
        """Add calibrated Gaussian noise to a query result (Eq. 3)."""
        noise = rng.normal(0.0, self.sigma)
        return float(np.clip(query_result + noise, 0.0, 1.0))

    def apply_batch(self, results: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
        """Apply DP noise to a batch of aggregate trust values."""
        noise = rng.normal(0.0, self.sigma, size=results.shape)
        return np.clip(results + noise, 0.0, 1.0)

    @property
    def info(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "sigma": self.sigma,
            "sensitivity": self.sensitivity,
        }


def measure_privacy_utility_tradeoff(
    true_trust: np.ndarray,
    epsilon_values: list[float],
    delta: float,
    rng: np.random.Generator,
    n_queries: int = 500,
) -> list[dict]:
    """
    For each ε, apply DP noise to true_trust values and compute
    Mean Absolute Error (MAE) vs. ground truth.
    Returns list of {epsilon, sigma, mae, std} for plotting.
    """
    results = []
    for eps in epsilon_values:
        dp = DifferentialPrivacyLayer(epsilon=eps, delta=delta)
        noisy = dp.apply_batch(
            np.tile(true_trust, (n_queries, 1)), rng
        )
        mae = float(np.mean(np.abs(noisy - true_trust)))
        std = float(np.std(np.abs(noisy - true_trust)))
        results.append({
            "epsilon": eps,
            "sigma": dp.sigma,
            "mae": mae,
            "std": std,
        })
    return results
