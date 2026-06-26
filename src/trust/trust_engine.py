"""
Trust engine implementing the delay-aware aging model (Eq. 4):

    T_ij(t + Δt) = α · T_ij(t) · exp(−λ·Δt) + (1−α) · S_ij(t)

Also implements the four baseline trust models for comparison.
"""

import numpy as np
from src.topology.ntn_topology import get_lambda


class TrustEngine:
    """
    Manages pairwise trust scores T_ij for all node pairs.
    Trust matrix is stored as a dense NxN numpy array for speed.
    """

    def __init__(self, num_nodes: int, cfg: dict, node_types: list[str],
                 rng: np.random.Generator):
        self.N = num_nodes
        self.cfg = cfg
        self.node_types = node_types
        self.rng = rng
        self.alpha = cfg["trust"]["alpha"]
        self.theta = cfg["trust"]["threshold_theta"]
        init = cfg["trust"]["init_trust"]

        # T[i,j] = trust node i has in node j
        self.T = np.full((num_nodes, num_nodes), init, dtype=np.float64)
        np.fill_diagonal(self.T, 1.0)  # each node fully trusts itself

    def update(self, i: int, j: int, s_ij: float, delta_t_ms: float):
        """
        Apply Eq. 4: delay-aware trust aging update.
        delta_t_ms: link propagation delay (ms) — added on top of round_duration_sec.
        delta_t for Eq.4 = round_duration_sec + propagation_delay_sec, so that
        trust ages with real elapsed NTN time and propagation delay adds staleness penalty.
        s_ij: latest interaction quality in [0,1].
        """
        round_dur = self.cfg["trust"]["round_duration_sec"]
        delta_t_sec = round_dur + delta_t_ms / 1000.0
        lam = get_lambda(self.node_types[j], self.cfg)
        alpha = self.alpha
        self.T[i, j] = (alpha * self.T[i, j] * np.exp(-lam * delta_t_sec)
                        + (1 - alpha) * s_ij)
        self.T[i, j] = float(np.clip(self.T[i, j], 0.0, 1.0))

    def decay_all(self, delta_t_ms: float):
        """
        Apply decay to ALL pairs simultaneously (no new interaction).
        Used when evidence is absent (e.g., during partitions).
        delta_t = round_duration_sec + partition_link_delay (ms).
        """
        round_dur = self.cfg["trust"]["round_duration_sec"]
        for j in range(self.N):
            lam = get_lambda(self.node_types[j], self.cfg)
            delta_t_sec = round_dur + delta_t_ms / 1000.0
            self.T[:, j] *= np.exp(-lam * delta_t_sec)
        np.fill_diagonal(self.T, 1.0)

    def get(self, i: int, j: int) -> float:
        return float(self.T[i, j])

    def is_trusted(self, i: int, j: int) -> bool:
        return self.T[i, j] >= self.theta

    def snapshot(self) -> np.ndarray:
        return self.T.copy()


# ── Baseline models ──────────────────────────────────────────────────────────

class CentralizedTrust:
    """
    Single authority maintains global trust.
    Fails during partitions: returns cached last-known value with no decay.
    """

    def __init__(self, num_nodes: int, cfg: dict, rng: np.random.Generator):
        init = cfg["trust"]["init_trust"]
        self.T = np.full((num_nodes, num_nodes), init, dtype=np.float64)
        np.fill_diagonal(self.T, 1.0)
        self.theta = cfg["trust"]["threshold_theta"]
        self._authority_reachable = True

    def set_partition(self, partitioned: bool):
        self._authority_reachable = not partitioned

    def update(self, i: int, j: int, s_ij: float, delta_t_ms: float):
        if self._authority_reachable:
            self.T[i, j] = 0.9 * self.T[i, j] + 0.1 * s_ij
            self.T[i, j] = float(np.clip(self.T[i, j], 0.0, 1.0))
        # else: stale — no update possible

    def get(self, i: int, j: int) -> float:
        return float(self.T[i, j])

    def is_trusted(self, i: int, j: int) -> bool:
        return self.T[i, j] >= self.theta

    def snapshot(self) -> np.ndarray:
        return self.T.copy()


class StaticDLT:
    """
    NxGenT-like: trust accumulates monotonically from positive interactions.
    No decay — build-then-betray attackers coast on accumulated score.
    """

    def __init__(self, num_nodes: int, cfg: dict, rng: np.random.Generator):
        init = cfg["trust"]["init_trust"]
        self.T = np.full((num_nodes, num_nodes), init, dtype=np.float64)
        np.fill_diagonal(self.T, 1.0)
        self.theta = cfg["trust"]["threshold_theta"]

    def update(self, i: int, j: int, s_ij: float, delta_t_ms: float):
        # Monotonic accumulation — no decay regardless of delay
        self.T[i, j] = 0.8 * self.T[i, j] + 0.2 * s_ij
        self.T[i, j] = float(np.clip(self.T[i, j], 0.0, 1.0))

    def get(self, i: int, j: int) -> float:
        return float(self.T[i, j])

    def is_trusted(self, i: int, j: int) -> bool:
        return self.T[i, j] >= self.theta

    def snapshot(self) -> np.ndarray:
        return self.T.copy()


class ZTAuthOnly:
    """
    ZTA Authentication-only: binary trust — authenticated or not.
    No trust decay, no evolution. Misbehaving authenticated nodes undetected.
    """

    def __init__(self, num_nodes: int, cfg: dict, rng: np.random.Generator):
        self.N = num_nodes
        self.authenticated = np.ones((num_nodes, num_nodes), dtype=bool)
        np.fill_diagonal(self.authenticated, True)
        self.theta = cfg["trust"]["threshold_theta"]

    def update(self, i: int, j: int, s_ij: float, delta_t_ms: float):
        # Re-authenticate based solely on credential check (not behavior)
        self.authenticated[i, j] = s_ij > 0.3  # simple threshold

    def get(self, i: int, j: int) -> float:
        return 1.0 if self.authenticated[i, j] else 0.0

    def is_trusted(self, i: int, j: int) -> bool:
        return bool(self.authenticated[i, j])

    def snapshot(self) -> np.ndarray:
        return self.authenticated.astype(float)


class UAVBlockchainFL:
    """
    UAV Blockchain + Federated Learning baseline.
    Dynamic trust but UAV-focused only; GEO/MEO delays ignored.
    Approximated as proposed model but without delay-aware λ tuning.
    """

    def __init__(self, num_nodes: int, cfg: dict, rng: np.random.Generator):
        init = cfg["trust"]["init_trust"]
        self.T = np.full((num_nodes, num_nodes), init, dtype=np.float64)
        np.fill_diagonal(self.T, 1.0)
        self.alpha = cfg["trust"]["alpha"]
        self.lam = cfg["trust"]["lambda_uav"]  # fixed λ — no per-segment tuning
        self.theta = cfg["trust"]["threshold_theta"]
        self.round_dur = cfg["trust"]["round_duration_sec"]

    def update(self, i: int, j: int, s_ij: float, delta_t_ms: float):
        delta_t_sec = self.round_dur + delta_t_ms / 1000.0
        # Same decay rate for ALL link types — key weakness vs. proposed
        self.T[i, j] = (self.alpha * self.T[i, j] * np.exp(-self.lam * delta_t_sec)
                        + (1 - self.alpha) * s_ij)
        self.T[i, j] = float(np.clip(self.T[i, j], 0.0, 1.0))

    def get(self, i: int, j: int) -> float:
        return float(self.T[i, j])

    def is_trusted(self, i: int, j: int) -> bool:
        return self.T[i, j] >= self.theta

    def snapshot(self) -> np.ndarray:
        return self.T.copy()


BASELINE_MAP = {
    "centralized": CentralizedTrust,
    "static_dlt": StaticDLT,
    "zt_auth_only": ZTAuthOnly,
    "uav_fl": UAVBlockchainFL,
    "proposed": TrustEngine,
}
