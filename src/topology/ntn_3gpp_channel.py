"""
3GPP TR 38.811 NR-NTN channel model for NTN trust framework simulation.

Replaces static random delays with physically-grounded link parameters:
- Free-space path loss at Ka-band (20 GHz)
- Orbital mechanics for LEO/MEO (elevation angle, slant range, handover)
- One-way propagation delay from slant range / c
- Link quality S_ij derived from received SNR

References:
  3GPP TR 38.811 V15.4.0 (2020-09) — Study on NR to support NTN
  3GPP TR 38.821 V16.0.0 (2019-09) — Solutions for NR to support NTN
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional

# Physical constants
C_KM_S       = 299_792.458          # speed of light (km/s)
R_EARTH_KM   = 6_371.0              # Earth radius (km)
FREQ_GHZ     = 20.0                 # Ka-band uplink frequency
NOISE_DBM    = -174 + 10 + 3        # thermal noise floor (dBm/Hz) + 10 dB NF + 3 dB impl
TX_POWER_DBM = 43.0                 # 20 W EIRP (dBm), typical NTN terminal
BW_HZ        = 500e6                # 500 MHz bandwidth
MIN_ELEV_DEG = 10.0                 # 3GPP minimum elevation angle (deg)


# Orbital altitudes and derived parameters (3GPP TR 38.811 Table 6.1-1)
ALTITUDES_KM = {
    "LEO":  550.0,    # Low Earth Orbit (Sun-sync shell)
    "MEO":  8_000.0,  # Medium Earth Orbit (between GPS/Galileo)
    "GEO":  35_786.0, # Geostationary
    "HAPS": 20.0,     # High Altitude Platform Station (stratosphere)
    "UAV":  0.5,      # UAV (tactical, ~500 m AGL)
}

# Orbital periods (s) from Kepler's 3rd law: T = 2π√((R_E+h)³/(GM))
# GM = 3.986004418e5 km³/s²
_GM = 3.986004418e5

def _orbital_period(h_km: float) -> float:
    """Keplerian orbital period for circular orbit at altitude h_km (seconds)."""
    return 2 * np.pi * np.sqrt((R_EARTH_KM + h_km)**3 / _GM)

ORBITAL_PERIOD_S = {seg: _orbital_period(h) for seg, h in ALTITUDES_KM.items()
                    if seg in ("LEO", "MEO", "GEO")}
# LEO: ~5731 s (95.5 min), MEO: ~18,956 s (316 min), GEO: ~86,164 s (24 h ≈ geostationary)


def fspl_db(dist_km: float, freq_ghz: float = FREQ_GHZ) -> float:
    """
    Free-space path loss (3GPP TR 38.811 Eq. 6.6.1-1).
    FSPL [dB] = 20·log10(4π·d·f/c)
    In practical form: FSPL = 20·log10(d_km) + 20·log10(f_GHz) + 92.44
    """
    return 20 * np.log10(dist_km) + 20 * np.log10(freq_ghz) + 92.44


def slant_range_km(h_km: float, elev_deg: float) -> float:
    """
    Slant range from ground station to satellite at elevation angle elev_deg.
    Derived from law of cosines on Earth-ground-satellite triangle.
    """
    elev_rad = np.radians(elev_deg)
    # Quadratic solution: d² + 2·R·sin(θ)·d - h·(h+2R) = 0
    a_coef = R_EARTH_KM * np.sin(elev_rad)
    disc   = a_coef**2 + h_km * (h_km + 2 * R_EARTH_KM)
    return -a_coef + np.sqrt(disc)


def one_way_delay_ms(dist_km: float) -> float:
    return 1000.0 * dist_km / C_KM_S


def snr_db(dist_km: float, freq_ghz: float = FREQ_GHZ) -> float:
    """
    Received SNR for a link of length dist_km.
    SNR = TX_POWER - FSPL - noise_power
    noise_power = NOISE_DBM + 10·log10(BW_HZ)
    """
    loss = fspl_db(dist_km, freq_ghz)
    noise_power_dbm = NOISE_DBM + 10 * np.log10(BW_HZ)
    return TX_POWER_DBM - loss - noise_power_dbm


def snr_to_quality(snr_db_val: float,
                   snr_min: float = -5.0,
                   snr_max: float = 30.0) -> float:
    """
    Map SNR (dB) to link quality in [0.05, 0.95] via sigmoid.
    snr_min: 5G NR minimum decodable SNR (~QPSK, rate 1/3)
    snr_max: excellent link (64-QAM, rate 5/6)
    """
    snr_norm = (snr_db_val - snr_min) / (snr_max - snr_min)
    snr_norm = float(np.clip(snr_norm, 0.0, 1.0))
    # Sigmoid in [0.05, 0.95]
    return 0.05 + 0.90 / (1 + np.exp(-8 * (snr_norm - 0.5)))


@dataclass
class NodePosition:
    """Geographic position of a ground/aerial node."""
    lat_deg: float    # latitude -90..90
    lon_deg: float    # longitude -180..180
    alt_km:  float = 0.0  # altitude above ground (km)


@dataclass
class SatelliteState:
    """Keplerian orbital state (circular orbit, simplified)."""
    seg_type: str          # "LEO", "MEO", or "GEO"
    sat_id: int
    inc_deg: float         # inclination (deg)
    raan_deg: float        # RAAN (right ascension ascending node, deg)
    phase0_deg: float      # initial orbital phase at t=0


def _sat_ecef(state: SatelliteState, t_sec: float):
    """
    Approximate ECEF position (km) of satellite at time t_sec.
    Simplified circular orbit (J2 perturbations ignored).
    """
    h  = ALTITUDES_KM[state.seg_type]
    r  = R_EARTH_KM + h
    T  = ORBITAL_PERIOD_S.get(state.seg_type, 86164.0)
    omega = 2 * np.pi / T
    phase = np.radians(state.phase0_deg) + omega * t_sec
    inc   = np.radians(state.inc_deg)
    raan  = np.radians(state.raan_deg)

    # Position in orbital plane
    x_orb = r * np.cos(phase)
    y_orb = r * np.sin(phase)

    # Rotate to ECEF via RAAN and inclination
    x = x_orb * np.cos(raan) - y_orb * np.cos(inc) * np.sin(raan)
    y = x_orb * np.sin(raan) + y_orb * np.cos(inc) * np.cos(raan)
    z = y_orb * np.sin(inc)
    return np.array([x, y, z])


def _ground_ecef(pos: NodePosition):
    """Ground node position in ECEF (km)."""
    lat = np.radians(pos.lat_deg)
    lon = np.radians(pos.lon_deg)
    r   = R_EARTH_KM + pos.alt_km
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])


def elevation_angle_deg(ground_ecef: np.ndarray, sat_ecef: np.ndarray) -> float:
    """Elevation angle of satellite as seen from ground node (degrees)."""
    diff = sat_ecef - ground_ecef
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return 90.0
    # Elevation: angle between diff and local horizontal plane
    # Local up direction = normalized ground_ecef
    up = ground_ecef / np.linalg.norm(ground_ecef)
    sin_el = np.dot(diff, up) / dist
    return float(np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0))))


# ── Main channel class ────────────────────────────────────────────────────────

class NTNChannel3GPP:
    """
    Time-varying 3GPP NR-NTN channel model.

    At each simulation round, provides:
      - link_active(i, j, round): bool (False during LEO handover)
      - delay_ms(i, j, round): one-way propagation delay
      - link_quality(i, j, round, is_malicious): interaction quality S_ij
    """

    def __init__(self, G: nx.Graph, type_to_ids: dict,
                 cfg: dict, rng: np.random.Generator):
        self.G   = G
        self.ids = type_to_ids
        self.cfg = cfg
        self.rng = rng
        self.round_dur = cfg["trust"]["round_duration_sec"]

        N = G.number_of_nodes()

        # Assign geographic positions to all nodes
        self._positions: list[Optional[NodePosition]] = [None] * N
        self._init_positions()

        # Create satellite orbital states
        self._sat_states: dict[int, SatelliteState] = {}
        self._init_satellites()

        # Cache for current-round delay matrix (lazy, updated per round)
        self._cached_round = -1
        # Satellite ECEF for the cached round (fast: only ~100 satellites)
        self._sat_ecef: dict[int, np.ndarray] = {}
        # Per-round per-edge delay cache (populated ON-DEMAND, not upfront)
        self._delay_cache:  dict[tuple, float] = {}
        self._active_cache: dict[tuple, bool]  = {}

        # Precompute ground/HAPS/UAV ECEF once (static positions)
        # Must be done AFTER _init_positions()
        self._ground_ecef_static: dict[int, np.ndarray] = {}

    def _init_positions(self):
        """Assign realistic geographic positions to all nodes."""
        rng = self.rng
        G   = self.G

        # GEO nodes: over equator (sub-satellite points)
        geo_lons = np.linspace(-120, 120, max(len(self.ids.get("GEO", [])), 1))
        for k, nid in enumerate(self.ids.get("GEO", [])):
            self._positions[nid] = NodePosition(0.0, float(geo_lons[k % len(geo_lons)]))

        # Ground nodes: random lat/lon skewed toward mid-latitudes
        for nid in self.ids.get("GROUND", []):
            lat = float(rng.uniform(-60, 60))
            lon = float(rng.uniform(-180, 180))
            self._positions[nid] = NodePosition(lat, lon)

        # UAV: near random ground nodes, 0.5 km altitude
        for nid in self.ids.get("UAV", []):
            lat = float(rng.uniform(-60, 60))
            lon = float(rng.uniform(-180, 180))
            self._positions[nid] = NodePosition(lat, lon, alt_km=0.5)

        # HAPS: 20 km altitude, covering a region
        for nid in self.ids.get("HAPS", []):
            lat = float(rng.uniform(-50, 50))
            lon = float(rng.uniform(-180, 180))
            self._positions[nid] = NodePosition(lat, lon, alt_km=20.0)

        # Satellite nodes get placeholder positions (computed from orbital state)
        for seg in ("LEO", "MEO"):
            for nid in self.ids.get(seg, []):
                self._positions[nid] = NodePosition(0.0, 0.0, ALTITUDES_KM[seg])

        # Precompute static ground ECEF (ground/HAPS/UAV positions don't move)
        for seg in ("GROUND", "HAPS", "UAV"):
            for nid in self.ids.get(seg, []):
                pos = self._positions[nid]
                if pos is not None:
                    self._ground_ecef_static[nid] = _ground_ecef(pos)
        # GEO is geostationary — precompute it once too (handled in _update_cache)

    def _init_satellites(self):
        """Create orbital states for LEO and MEO satellites."""
        rng = self.rng

        # LEO: 550 km, inclination 53° (Starlink-like), evenly distributed RAAN and phase
        leo_ids = self.ids.get("LEO", [])
        n_leo = len(leo_ids)
        for k, nid in enumerate(leo_ids):
            raan = 360.0 * k / max(n_leo, 1)
            phase0 = float(rng.uniform(0, 360))
            self._sat_states[nid] = SatelliteState("LEO", nid, 53.0, raan, phase0)

        # MEO: 8000 km, inclination 55° (GPS-like)
        meo_ids = self.ids.get("MEO", [])
        n_meo = len(meo_ids)
        for k, nid in enumerate(meo_ids):
            raan = 360.0 * k / max(n_meo, 1)
            phase0 = float(rng.uniform(0, 360))
            self._sat_states[nid] = SatelliteState("MEO", nid, 55.0, raan, phase0)

        # GEO: stationary (orbital period ~86164 s, effectively zero drift)
        for nid in self.ids.get("GEO", []):
            self._sat_states[nid] = SatelliteState("GEO", nid, 0.0, 0.0,
                                                    float(rng.uniform(0, 360)))

    def _update_round(self, round_num: int):
        """
        Per-round update: recompute satellite ECEF positions (cheap: ~100 sats).
        Edge delays are computed ON-DEMAND in _get_link(), not upfront for all edges.
        This reduces per-round overhead from O(|E|) to O(|satellites|).
        """
        if self._cached_round == round_num:
            return
        self._cached_round = round_num
        # Clear per-edge cache for this round (avoids stale values from last round)
        self._delay_cache  = {}
        self._active_cache = {}

        t_sec = round_num * self.round_dur
        # Compute satellite ECEF positions (only ~100 satellites — fast)
        for nid, state in self._sat_states.items():
            self._sat_ecef[nid] = _sat_ecef(state, t_sec)

    def _get_link(self, i: int, j: int) -> tuple[float, bool]:
        """
        Compute and cache delay_ms + active for edge (i,j) within the current round.
        Called on-demand — only for edges actually used in the simulation loop.
        """
        key = (i, j)
        if key in self._delay_cache:
            return self._delay_cache[key], self._active_cache[key]

        ti = self.G.nodes[i]["type"]
        tj = self.G.nodes[j]["type"]
        delay, active = self._compute_link(i, j, ti, tj)

        self._delay_cache[(i, j)]  = delay
        self._delay_cache[(j, i)]  = delay
        self._active_cache[(i, j)] = active
        self._active_cache[(j, i)] = active
        return delay, active

    def _compute_link(self, i, j, ti, tj) -> tuple[float, bool]:
        """
        Compute (delay_ms, active) for edge (i,j) using pre-cached satellite ECEF
        and static ground ECEF.  Called lazily — only for edges actually used.
        """
        sat_ecef   = self._sat_ecef              # pre-computed for this round
        ground_ecef = self._ground_ecef_static   # static (computed once at init)

        # Satellite–ground links
        for (sat_id, sat_t), (gnd_id, gnd_t) in [
            ((i, ti), (j, tj)), ((j, tj), (i, ti))
        ]:
            if sat_t in ("LEO", "MEO", "GEO") and gnd_t in ("GROUND", "HAPS", "UAV"):
                if sat_id not in sat_ecef or gnd_id not in ground_ecef:
                    return self._fallback_delay(sat_t), True
                se   = sat_ecef[sat_id]
                ge   = ground_ecef[gnd_id]
                elev = elevation_angle_deg(ge, se)
                dist = float(np.linalg.norm(se - ge))
                active = elev >= MIN_ELEV_DEG
                delay  = one_way_delay_ms(dist) if active else 999.0
                return delay, active

        # ISL (satellite–satellite)
        if ti in ("LEO", "MEO", "GEO") and tj in ("LEO", "MEO", "GEO"):
            if i in sat_ecef and j in sat_ecef:
                dist = float(np.linalg.norm(sat_ecef[i] - sat_ecef[j]))
                return one_way_delay_ms(dist), True
            return self._fallback_delay("LEO"), True

        # Aerial–aerial or ground–ground
        if i in ground_ecef and j in ground_ecef:
            dist = float(np.linalg.norm(ground_ecef[i] - ground_ecef[j]))
            dist = max(dist, 0.01)
            return one_way_delay_ms(dist), True

        return self._fallback_delay(ti), True

    def _fallback_delay(self, seg: str) -> float:
        delays = {"LEO": 3.0, "MEO": 50.0, "GEO": 270.0,
                  "HAPS": 0.1, "UAV": 0.01, "GROUND": 2.0}
        return delays.get(seg, 10.0)

    def link_active(self, i: int, j: int, round_num: int) -> bool:
        self._update_round(round_num)
        _, active = self._get_link(i, j)
        return active

    def delay_ms(self, i: int, j: int, round_num: int) -> float:
        self._update_round(round_num)
        delay, _ = self._get_link(i, j)
        return delay

    def link_quality(self, node_j: int, round_num: int,
                     quality_fn, i: int) -> float:
        """
        Return interaction quality modulated by channel SNR.
        SNR < 5 dB: packet-error-rate degrades quality by up to 20%.
        """
        base_quality = float(quality_fn(node_j, round_num))
        self._update_round(round_num)
        delay, active = self._get_link(i, node_j)
        if not active or delay >= 900:
            return 0.0
        # delay in ms → dist in km = delay_ms * c_km_s / 1000
        d_km = delay * C_KM_S / 1000.0
        snr  = snr_db(d_km)
        if snr < 5.0:
            base_quality *= max(0.8, snr / 5.0)
        return float(np.clip(base_quality, 0.0, 1.0))
