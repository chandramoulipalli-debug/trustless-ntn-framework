"""
Attack scenario generators.
Each returns a function: (node_id, round_num) -> interaction_quality in [0,1].
- Normal node:      consistently good quality
- On-off attacker:  alternates good/bad phases
- Build-then-betray: builds high trust then drops to malicious
- Sybil node:       creates fake identities (handled at topology level)
- Replay:           injects stale evidence (age-based detection test)
- Badmouthing:      sends false low ratings about legitimate nodes
"""

import numpy as np


def normal_node(rng: np.random.Generator, noise: float = 0.05):
    """Legitimate node: high quality with small Gaussian noise."""
    def quality(node_id: int, round_num: int) -> float:
        return float(np.clip(rng.normal(0.85, noise), 0.0, 1.0))
    return quality


def on_off_attacker(rng: np.random.Generator,
                    good_rounds: int, bad_rounds: int):
    """
    Alternates between good phases (cooperative) and bad phases (malicious).
    Tests whether delay-aware decay catches the attacker faster than static DLT.
    """
    cycle = good_rounds + bad_rounds

    def quality(node_id: int, round_num: int) -> float:
        phase = round_num % cycle
        if phase < good_rounds:
            return float(np.clip(rng.normal(0.85, 0.05), 0.0, 1.0))
        else:
            return float(np.clip(rng.normal(0.10, 0.05), 0.0, 1.0))
    return quality


def build_then_betray(rng: np.random.Generator, buildup_rounds: int):
    """
    Behaves well for buildup_rounds, then permanently turns malicious.
    Tests resilience of decay mechanism against accumulated reputation.
    """
    def quality(node_id: int, round_num: int) -> float:
        if round_num < buildup_rounds:
            return float(np.clip(rng.normal(0.90, 0.03), 0.0, 1.0))
        else:
            return float(np.clip(rng.normal(0.05, 0.05), 0.0, 1.0))
    return quality


def badmouthing_attacker(rng: np.random.Generator, target_id: int):
    """
    Good self-behavior but sends false low ratings about target node.
    Returns quality function for its OWN interactions (appears trustworthy),
    but inject_feedback flag signals the ledger to add false testimony.
    """
    def quality(node_id: int, round_num: int) -> float:
        return float(np.clip(rng.normal(0.80, 0.05), 0.0, 1.0))
    return quality


def replay_evidence(legitimate_quality: float, staleness_ms: float):
    """
    Returns a fake 'quality observation' with a stale timestamp.
    The ledger's aging model should discount/reject this.
    staleness_ms: how old the replayed evidence is.
    """
    return {
        "quality": legitimate_quality,
        "staleness_ms": staleness_ms,
        "is_replay": True,
    }


def assign_attack_roles(node_ids: list[int],
                        malicious_fraction: float,
                        cfg: dict,
                        rng: np.random.Generator) -> dict:
    """
    Randomly assign attack roles to malicious_fraction of nodes.
    Returns dict: node_id -> quality_function.
    """
    n_malicious = int(len(node_ids) * malicious_fraction)
    malicious_ids = rng.choice(node_ids, size=n_malicious, replace=False)
    malicious_set = set(malicious_ids.tolist())

    attack_cfg = cfg["attacks"]
    roles = {}
    attack_types = ["on_off", "build_betray", "on_off"]  # distribution
    for idx, nid in enumerate(malicious_ids):
        attack_type = attack_types[idx % len(attack_types)]
        if attack_type == "on_off":
            roles[nid] = on_off_attacker(
                rng,
                attack_cfg["on_off_good_rounds"],
                attack_cfg["on_off_bad_rounds"],
            )
        else:
            roles[nid] = build_then_betray(
                rng,
                attack_cfg["build_betray_buildup"],
            )

    for nid in node_ids:
        if nid not in malicious_set:
            roles[nid] = normal_node(rng)

    return roles, malicious_set
