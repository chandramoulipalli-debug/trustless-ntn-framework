"""Unit tests for NTN topology builder."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import yaml
import networkx as nx

from src.topology.ntn_topology import build_ntn_graph, node_count


@pytest.fixture
def cfg():
    with open("configs/simulation_config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_graph_is_connected(cfg, rng):
    """NTN graph must always be connected (no isolated nodes)."""
    G, _ = build_ntn_graph(cfg, rng)
    assert nx.is_connected(G), "NTN graph must be connected"


def test_node_counts_correct(cfg, rng):
    """Total node count must match config."""
    G, _ = build_ntn_graph(cfg, rng)
    tc = cfg["topology"]
    expected = (tc["leo_satellites"] + tc["meo_satellites"] + tc["geo_satellites"]
                + tc["haps"] + tc["uavs"] + tc["ground_nodes"])
    assert G.number_of_nodes() == expected, \
        f"Expected {expected} nodes, got {G.number_of_nodes()}"


def test_all_edges_have_delay(cfg, rng):
    """Every edge must have a delay_ms attribute."""
    G, _ = build_ntn_graph(cfg, rng)
    for u, v, data in G.edges(data=True):
        assert "delay_ms" in data, f"Edge ({u},{v}) missing delay_ms"
        assert data["delay_ms"] > 0, f"Edge ({u},{v}) has non-positive delay"


def test_all_nodes_have_type(cfg, rng):
    """Every node must have a 'type' attribute."""
    G, _ = build_ntn_graph(cfg, rng)
    for nid, data in G.nodes(data=True):
        assert "type" in data, f"Node {nid} missing type"


def test_reproducibility(cfg):
    """Same seed must produce identical graphs."""
    G1, _ = build_ntn_graph(cfg, np.random.default_rng(99))
    G2, _ = build_ntn_graph(cfg, np.random.default_rng(99))
    assert G1.number_of_edges() == G2.number_of_edges(), \
        "Same seed must produce same graph"
