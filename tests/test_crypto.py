"""Unit tests for Paillier HE and Differential Privacy modules."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import yaml

from src.crypto.paillier_agg import PaillierAggregator
from src.crypto.differential_privacy import DifferentialPrivacyLayer


@pytest.fixture
def cfg():
    with open("configs/simulation_config.yaml") as f:
        return yaml.safe_load(f)


def test_paillier_encrypt_decrypt_roundtrip():
    """Encrypting then decrypting a single value must recover the original."""
    agg = PaillierAggregator(key_bits=512)  # small key for test speed
    original = 0.75
    ct = agg.encrypt_feedback(original)
    recovered = agg.decrypt_aggregate(ct, n_contributors=1)
    assert abs(recovered - original) < 0.01, \
        f"Roundtrip failed: {original} -> {recovered}"


def test_paillier_homomorphic_sum():
    """Sum of encrypted values must equal encryption of the sum."""
    agg = PaillierAggregator(key_bits=512)
    feedbacks = [0.8, 0.6, 0.7]
    cts = [agg.encrypt_feedback(f) for f in feedbacks]
    agg_ct = agg.aggregate(cts)
    result = agg.decrypt_aggregate(agg_ct, n_contributors=len(feedbacks))
    expected = sum(feedbacks) / len(feedbacks)
    assert abs(result - expected) < 0.02, \
        f"Homomorphic sum failed: expected {expected:.3f}, got {result:.3f}"


def test_dp_noise_magnitude(cfg):
    """Smaller epsilon must produce larger sigma (more noise)."""
    delta = cfg["crypto"]["dp_delta"]
    dp_small = DifferentialPrivacyLayer(epsilon=0.1, delta=delta)
    dp_large = DifferentialPrivacyLayer(epsilon=5.0, delta=delta)
    assert dp_small.sigma > dp_large.sigma, \
        "Smaller ε must yield larger noise σ"


def test_dp_output_bounded(cfg):
    """DP-noised trust scores must remain in [0, 1]."""
    rng = np.random.default_rng(42)
    dp = DifferentialPrivacyLayer(epsilon=1.0, delta=cfg["crypto"]["dp_delta"])
    scores = np.full(100, 0.7)
    noisy = dp.apply_batch(scores, rng)
    assert np.all(noisy >= 0.0) and np.all(noisy <= 1.0), \
        "DP output must be clipped to [0, 1]"


def test_paillier_benchmark_returns_all_keys():
    """Benchmark must return all required metric keys."""
    agg = PaillierAggregator(key_bits=512)
    result = agg.benchmark(n_contributors=5)
    required_keys = {"enc_time_ms", "agg_time_ms", "dec_time_ms",
                     "total_time_ms", "ciphertext_size_kb", "result"}
    assert required_keys.issubset(result.keys())
