"""
NTN topology builder.
Creates a heterogeneous 6G Non-Terrestrial Network graph with
LEO/MEO/GEO satellites, HAPS, UAVs, and ground nodes.
Assigns per-link delay drawn from segment-specific ranges.
"""

import random
import networkx as nx
import numpy as np


# Node type constants
LEO = "LEO"
MEO = "MEO"
GEO = "GEO"
HAPS = "HAPS"
UAV = "UAV"
GROUND = "GROUND"

SEGMENT_ORDER = {LEO: 0, MEO: 1, GEO: 2, HAPS: 3, UAV: 4, GROUND: 5}


def build_ntn_graph(cfg: dict, rng: np.random.Generator) -> nx.Graph:
    """
    Build the NTN graph from simulation_config topology block.
    Returns a networkx Graph where each node has attribute 'type'
    and each edge has attribute 'delay_ms' and 'segment_pair'.
    """
    tc = cfg["topology"]
    counts = {
        LEO: tc["leo_satellites"],
        MEO: tc["meo_satellites"],
        GEO: tc["geo_satellites"],
        HAPS: tc["haps"],
        UAV: tc["uavs"],
        GROUND: tc["ground_nodes"],
    }
    delays = tc["delay"]

    G = nx.Graph()
    node_id = 0
    type_to_ids: dict[str, list[int]] = {}

    # Add all nodes
    for ntype, count in counts.items():
        ids = list(range(node_id, node_id + count))
        for nid in ids:
            G.add_node(nid, type=ntype, malicious=False, trust_score=cfg["trust"]["init_trust"])
        type_to_ids[ntype] = ids
        node_id += count

    # Helper: add edges between two groups with a delay range
    def connect_groups(group_a, group_b, delay_range, connectivity=0.3):
        lo, hi = delay_range
        for a in group_a:
            for b in group_b:
                if rng.random() < connectivity:
                    d = float(rng.uniform(lo, hi))
                    G.add_edge(a, b, delay_ms=d,
                               segment_pair=f"{G.nodes[a]['type']}-{G.nodes[b]['type']}")

    # LEO ↔ Ground (dense coverage)
    connect_groups(type_to_ids[LEO], type_to_ids[GROUND],
                   delays["leo_ground_ms"], connectivity=0.4)
    # GEO ↔ Ground (sparse but stable)
    connect_groups(type_to_ids[GEO], type_to_ids[GROUND],
                   delays["geo_ground_ms"], connectivity=0.2)
    # MEO ↔ Ground
    connect_groups(type_to_ids[MEO], type_to_ids[GROUND],
                   delays["meo_ground_ms"], connectivity=0.25)
    # HAPS ↔ Ground
    connect_groups(type_to_ids[HAPS], type_to_ids[GROUND],
                   delays["haps_ground_ms"], connectivity=0.5)
    # UAV ↔ Ground
    connect_groups(type_to_ids[UAV], type_to_ids[GROUND],
                   delays["uav_uav_ms"], connectivity=0.3)
    # UAV ↔ UAV (mesh)
    connect_groups(type_to_ids[UAV], type_to_ids[UAV],
                   delays["uav_uav_ms"], connectivity=0.2)
    # LEO ↔ LEO (ISLs)
    connect_groups(type_to_ids[LEO], type_to_ids[LEO],
                   [1, 10], connectivity=0.1)
    # LEO ↔ HAPS
    connect_groups(type_to_ids[LEO], type_to_ids[HAPS],
                   delays["leo_ground_ms"], connectivity=0.3)
    # HAPS ↔ UAV
    connect_groups(type_to_ids[HAPS], type_to_ids[UAV],
                   delays["haps_ground_ms"], connectivity=0.4)

    # Ensure graph is connected (add spanning tree edges if needed)
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            G.add_edge(u, v, delay_ms=float(rng.uniform(10, 50)),
                       segment_pair="bridge")

    return G, type_to_ids


def get_lambda(node_type: str, cfg: dict) -> float:
    """Return per-segment trust decay rate λ."""
    mapping = {
        UAV:    cfg["trust"]["lambda_uav"],
        LEO:    cfg["trust"]["lambda_leo"],
        MEO:    cfg["trust"]["lambda_meo"],
        GEO:    cfg["trust"]["lambda_geo"],
        HAPS:   cfg["trust"]["lambda_haps"],
        GROUND: cfg["trust"]["lambda_ground"],
    }
    return mapping.get(node_type, cfg["trust"]["lambda_leo"])


def node_count(G: nx.Graph) -> dict:
    counts = {}
    for _, data in G.nodes(data=True):
        t = data["type"]
        counts[t] = counts.get(t, 0) + 1
    return counts
