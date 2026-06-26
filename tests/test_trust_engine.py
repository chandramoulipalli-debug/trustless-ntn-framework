"""Unit tests for the trust engine (Eq. 4 and baselines)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import yaml

from src.trust.trust_engine import TrustEngine, StaticDLT, CentralizedTrust


@pytest.fixture
def cfg():
    with open("configs/simulation_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_trust_decays_without_interaction(cfg, rng):
    """Trust must decrease when no positive evidence arrives (Eq. 4 with S=0)."""
    node_types = ["GROUND"] * 10
    engine = TrustEngine(10, cfg, node_types, rng)
    initial = engine.get(0, 1)
    engine.update(0, 1, s_ij=0.0, delta_t_ms=5000)
    assert engine.get(0, 1) < initial, "Trust should decay with bad interaction"


def test_trust_bounded(cfg, rng):
    """Trust score must stay in [0, 1] under any update."""
    node_types = ["UAV"] * 5
    engine = TrustEngine(5, cfg, node_types, rng)
    for _ in range(1000):
        engine.update(0, 1, s_ij=float(rng.uniform(0, 1)), delta_t_ms=float(rng.uniform(1, 500)))
    val = engine.get(0, 1)
    assert 0.0 <= val <= 1.0, f"Trust out of bounds: {val}"


def test_uav_decays_faster_than_geo(cfg, rng):
    """UAV links (high λ) must decay faster than GEO links (low λ)."""
    types_uav = ["UAV"] * 5
    types_geo = ["GEO"] * 5
    engine_uav = TrustEngine(5, cfg, types_uav, rng)
    engine_geo = TrustEngine(5, cfg, types_geo, rng)
    delta_t_ms = 10000.0  # large elapsed time to amplify difference
    engine_uav.update(0, 1, s_ij=0.0, delta_t_ms=delta_t_ms)
    engine_geo.update(0, 1, s_ij=0.0, delta_t_ms=delta_t_ms)
    assert engine_uav.get(0, 1) < engine_geo.get(0, 1), \
        "UAV trust should decay faster than GEO trust"


def test_static_dlt_monotone_increase(cfg, rng):
    """Static DLT trust must not decrease under positive interactions."""
    engine = StaticDLT(5, cfg, rng)
    prev = engine.get(0, 1)
    for _ in range(50):
        engine.update(0, 1, s_ij=0.9, delta_t_ms=100)
        assert engine.get(0, 1) >= prev - 1e-9
        prev = engine.get(0, 1)


def test_trust_engine_decay_all(cfg, rng):
    """decay_all must reduce all off-diagonal trust values."""
    node_types = ["LEO"] * 10
    engine = TrustEngine(10, cfg, node_types, rng)
    # Set high trust for all pairs
    engine.T[:] = 0.9
    np.fill_diagonal(engine.T, 1.0)
    engine.decay_all(delta_t_ms=50000)
    for i in range(10):
        for j in range(10):
            if i != j:
                assert engine.get(i, j) < 0.9, \
                    f"T[{i},{j}] should have decayed"


def test_is_trusted_threshold(cfg, rng):
    """is_trusted must return False when trust < theta."""
    node_types = ["GROUND"] * 5
    engine = TrustEngine(5, cfg, node_types, rng)
    engine.T[0, 1] = 0.2  # below theta=0.4
    assert not engine.is_trusted(0, 1)
    engine.T[0, 1] = 0.8  # above theta
    assert engine.is_trusted(0, 1)
