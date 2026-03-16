"""Aircraft Heading Predictor System — concrete implementations of the abstract verification API.

Provides:
- ``AircraftHeadingEnvironment``    — samples initial states from recorded ADS-B snapshots,
                                       conditioned on time-of-day, proximity to airport,
                                       and airport ICAO.
- ``AircraftHeadingDisturbanceModel`` — independent Gaussian noise on heading commands
                                        and position observations.
- ``DirectObservationModel``          — pass-through observation (no additional noise
                                        beyond the disturbance model).
- ``PassthroughAgentModel``           — baseline agent that holds the current heading
                                        (useful for testing the rollout loop).
- ``AircraftHeadingPredictorSystem``  — composes all of the above; uses flat-Earth
                                        kinematics to propagate the ego aircraft and
                                        background traffic one timestep at a time.
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from verification.system.samplers import (
    CandidateFeatures,
    ScenarioSampler,
    UniformSampler,
)
from verification.system.state import AircraftState, SystemState
from verification.system.actions import Action
from verification.system.observations import Observation
from aircraft_heading_gnn.utils.adsb_features import get_distance_nm
from verification.system.system import (
    AgentModel,
    DisturbanceModel,
    Environment,
    ObservationModel,
    System,
)

KTS_TO_MPS = 0.514444          # knots → m/s
FT_PER_MIN_TO_MPS = 1.0 / (60.0 * 3.28084)  # ft/min → m/s
DEG_LAT_TO_M = 111_320.0       # metres per degree of latitude (WGS-84 approx)
MAX_TURN_RATE_RAD_S = math.radians(3.0)      # 3 deg/s — standard-rate turn ÷ 10

# Named buckets for convenience
_TIME_OF_DAY_BUCKETS: Dict[str, Tuple[int, int]] = {
    "night":     (0,  6),
    "morning":   (6,  12),
    "afternoon": (12, 18),
    "evening":   (18, 24),
}

_PROXIMITY_BUCKETS: Dict[str, Tuple[float, float]] = {
    "near": (0.0,  40.0),    # terminal environment
    "far":  (40.0, 150.0),   # en-route
}


def latlon_to_local_m(
    lat: float, lon: float, ref_lat: float, ref_lon: float,
    alt_ft: float = 0.0,
) -> np.ndarray:
    """Convert (lat, lon, alt) to local ENU metres centred on (ref_lat, ref_lon).

    Returns a (3,) array: [east_m, north_m, alt_m].
    """
    north = (lat - ref_lat) * DEG_LAT_TO_M
    east  = (lon - ref_lon) * DEG_LAT_TO_M * math.cos(math.radians(ref_lat))
    alt_m = alt_ft * 0.3048
    return np.array([east, north, alt_m], dtype=np.float32)


def heading_to_velocity(
    heading_deg: float, speed_mps: float, vertrate_fpm: float = 0.0
) -> np.ndarray:
    """Return (vx, vy, vz) = (east, north, up) velocity from heading, speed, and vertical rate."""
    h = math.radians(heading_deg)
    vz = vertrate_fpm * FT_PER_MIN_TO_MPS
    return np.array([speed_mps * math.sin(h), speed_mps * math.cos(h), vz], dtype=np.float32)


def angle_diff_rad(a: float, b: float) -> float:
    """Shortest signed difference (a − b) wrapped to (−π, π]."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def gaussian_log_prob(x: float, sigma: float) -> float:
    """Log-probability of *x* under N(0, sigma²)."""
    return -0.5 * (x / sigma) ** 2 - 0.5 * math.log(2.0 * math.pi * sigma ** 2)



def parse_utc_hour(timestamp_iso: str) -> Optional[int]:
    """Return UTC hour from an ISO-8601 string, or ``None`` on parse failure."""
    try:
        return datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00")).hour
    except (ValueError, AttributeError):
        return None


def load_snapshots(data_path: str) -> Dict[float, List[Dict[str, Any]]]:
    """Load an ADS-B JSON observations file and group by simulation timestamp.

    Each entry in the returned dict maps ``timestamp_s`` (float) to a list of
    aircraft dicts with normalised field names.  On-ground aircraft are dropped.

    Expected JSON layout::

        {"observations": [
            {"timestamp_s": 0.0, "data": {
                "icao_address": ..., "aircraft_callsign": ...,
                "latitude_deg": ..., "longitude_deg": ...,
                "heading_deg": ..., "groundspeed_knots": ...,
                "geometric_altitude_ft": ..., "vertical_speed_fpm": ...,
                "timestamp_utc_iso": ..., "onground": false
            }},
            ...
        ]}
    """
    with open(data_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    snapshots: Dict[float, List[Dict[str, Any]]] = {}
    for obs in raw["observations"]:
        d = obs["data"]
        if d.get("onground", False):
            continue
        lat = d.get("latitude_deg")
        lon = d.get("longitude_deg")
        if lat is None or lon is None:
            continue
        ac: Dict[str, Any] = {
            "icao24":        d.get("icao_address", d.get("aircraft_callsign", "UNKNOWN")),
            "callsign":      d.get("aircraft_callsign", ""),
            "lat":           float(lat),
            "lon":           float(lon),
            "heading_deg":   float(d.get("heading_deg") or 0.0),
            "speed_mps":     float(d.get("groundspeed_knots") or 0.0) * KTS_TO_MPS,
            "alt_ft":        float(d.get("geometric_altitude_ft") or 0.0),
            "vertrate_fpm":  float(d.get("vertical_speed_fpm") or 0.0),
            "timestamp_utc": d.get("timestamp_utc_iso", ""),
        }
        snapshots.setdefault(float(obs["timestamp_s"]), []).append(ac)
    return snapshots


# Environment
class AircraftHeadingEnvironment(Environment):
    """Environment that samples initial states from recorded ADS-B snapshots.

    Sampling is conditioned on up to three independent filters:

    * **time_of_day** — UTC hour range.  Pass a ``(start_h, end_h)`` tuple or
      one of the named strings ``"night"`` (0–6), ``"morning"`` (6–12),
      ``"afternoon"`` (12–18), ``"evening"`` (18–24).  ``None`` = no filter.

    * **proximity_nm** — ego aircraft distance from airport in nautical miles.
      Pass a ``(min_nm, max_nm)`` tuple or ``"near"`` (<40 NM) / ``"far"``
      (40–150 NM).  ``None`` = no filter.

    * **airport_icao** — ICAO identifier of the reference airport (informational;
      the coordinate system is defined by *airport_lat* / *airport_lon*).

    All positions in returned ``SystemState`` objects are expressed in a local
    ENU frame (metres) centred on the airport reference point.
    """

    def __init__(
        self,
        data_path: str,
        airport_lat: float,
        airport_lon: float,
        airport_icao: str = "KSEA",
        time_of_day: Optional[Any] = None,
        proximity_nm: Optional[Any] = None,
        max_alt_ft: Optional[float] = None,
        sampler: Optional[ScenarioSampler] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            data_path:      Path to the ADS-B JSON observations file.
            airport_lat:    Airport reference latitude (degrees).
            airport_lon:    Airport reference longitude (degrees).
            airport_icao:   Airport ICAO identifier (informational).
            time_of_day:    UTC hour-range filter — tuple or bucket name.
            proximity_nm:   Ego distance filter — tuple or bucket name.
            max_alt_ft:     Maximum ego altitude (feet) — excludes en-route
                            traffic above this threshold.  ``None`` = no filter.
            sampler:        Scenario sampling strategy.  Defaults to
                            ``UniformSampler(seed=seed)``.
            seed:           Optional RNG seed.  Passed to the default
                            ``UniformSampler`` when no sampler is provided.
        """
        self.airport_lat = airport_lat
        self.airport_lon = airport_lon
        self.airport_icao = airport_icao
        self.rng = np.random.default_rng(seed)

        # Resolve named aliases
        self._hour_range: Optional[Tuple[int, int]]
        if isinstance(time_of_day, str):
            self._hour_range = _TIME_OF_DAY_BUCKETS[time_of_day]
        elif time_of_day is not None:
            self._hour_range = (int(time_of_day[0]), int(time_of_day[1]))
        else:
            self._hour_range = None

        self._proximity_range: Optional[Tuple[float, float]]
        if isinstance(proximity_nm, str):
            self._proximity_range = _PROXIMITY_BUCKETS[proximity_nm]
        elif proximity_nm is not None:
            self._proximity_range = (float(proximity_nm[0]), float(proximity_nm[1]))
        else:
            self._proximity_range = None

        # Build the list of (timestamp, ego_index, aircraft_list) candidates
        all_snapshots = load_snapshots(data_path)
        # Keep the full snapshot dict for reference-path lookups.
        self._snapshot_by_ts: Dict[float, List[Dict[str, Any]]] = all_snapshots
        self._sorted_timestamps: List[float] = sorted(all_snapshots.keys())
        self._candidates: List[Tuple[float, int, List[Dict[str, Any]]]] = []

        for ts, aircraft_list in sorted(all_snapshots.items()):
            if not aircraft_list:
                continue

            # Time-of-day filter (keyed off the first aircraft in the snapshot)
            if self._hour_range is not None:
                utc_hour = parse_utc_hour(aircraft_list[0]["timestamp_utc"])
                if utc_hour is None:
                    continue
                h_lo, h_hi = self._hour_range
                if not (h_lo <= utc_hour < h_hi):
                    continue

            # Proximity and altitude filter: accept each aircraft satisfying constraints
            for i, ac in enumerate(aircraft_list):
                if self._proximity_range is not None:
                    dist = get_distance_nm(
                        self.airport_lat, self.airport_lon, ac["lat"], ac["lon"]
                    )
                    lo, hi = self._proximity_range
                    if not (lo <= dist <= hi):
                        continue
                if max_alt_ft is not None and ac["alt_ft"] > max_alt_ft:
                    continue
                self._candidates.append((ts, i, aircraft_list))

        if not self._candidates:
            raise ValueError(
                "No snapshots match the specified filters "
                f"(time_of_day={time_of_day!r}, proximity_nm={proximity_nm!r}). "
                "Try relaxing the filter criteria."
            )

        # Build CandidateFeatures list and hand it to the sampler
        self._candidate_features: List[CandidateFeatures] = [
            self._extract_features(idx, ts, ego_idx, aircraft_list)
            for idx, (ts, ego_idx, aircraft_list) in enumerate(self._candidates)
        ]
        self._sampler: ScenarioSampler = (
            sampler if sampler is not None else UniformSampler(seed=seed)
        )
        self._sampler.initialize(self._candidate_features)

    def get_reference_path(
        self,
        initial_state: SystemState,
        num_steps: Optional[int] = None,
        dt: float = 10.0,
    ) -> np.ndarray:
        """Return the ground-truth ADS-B trajectory for the ego aircraft.

        Walks forward through subsequent ADS-B snapshots, collecting the ENU
        position of the aircraft whose ``icao24`` matches
        ``initial_state.ego.icao24``, up to ``num_steps * dt`` seconds ahead
        (or until the aircraft disappears from the data).

        The resulting polyline is the *reference path* for
        :func:`cross_track_spec` / :func:`cross_track_robustness`.  Because
        those functions measure nearest-point-on-segment distance, the path
        does not need to be time-aligned with the simulation — it just needs
        to capture the general corridor the aircraft flew.

        Args:
            initial_state: Initial system state.  ``ego.icao24`` identifies
                the aircraft to track.
            num_steps:  Optional cap on forward steps.  When *None* all
                remaining appearances in the dataset are used.
            dt:         Simulation timestep in seconds used to compute the
                time horizon ``t_start + num_steps * dt``.

        Returns:
            ``(M, 3)`` float32 array of ENU (x, y, z) positions in metres,
            starting at the initial ego position.  Always has M >= 1.
        """
        icao24 = initial_state.ego.icao24
        t_start = initial_state.time
        t_end = (t_start + num_steps * dt) if num_steps is not None else float("inf")

        positions: List[np.ndarray] = [initial_state.ego.position.copy()]

        for ts in self._sorted_timestamps:
            if ts <= t_start:
                continue
            if ts > t_end:
                break
            for ac in self._snapshot_by_ts[ts]:
                if ac["icao24"] == icao24:
                    pos = latlon_to_local_m(
                        ac["lat"], ac["lon"], self.airport_lat, self.airport_lon,
                        ac.get("alt_ft", 0.0),
                    )
                    positions.append(pos)
                    break  # found the aircraft in this snapshot

        return np.array(positions, dtype=np.float32)

    @property
    def num_scenarios(self) -> int:
        """Number of (snapshot, ego) candidate pairs available for sampling."""
        return len(self._candidates)

    def reset(self) -> SystemState:
        """Return the first qualifying snapshot (deterministic baseline)."""
        ts, idx, aircraft_list = self._candidates[0]
        return self._build_state(ts, idx, aircraft_list)

    def sample_initial_state(self) -> SystemState:
        """Draw an initial state using the configured ``ScenarioSampler``."""
        choice = self._sampler.sample()
        ts, idx, aircraft_list = self._candidates[choice]
        return self._build_state(ts, idx, aircraft_list)

    def step(
        self, state: SystemState, ego_action: Action, dt: float
    ) -> SystemState:
        """Kinematic propagation of ego and traffic aircraft.

        * Ego heading tracks ``ego_action.heading_command`` via a first-order
          response capped at ``MAX_TURN_RATE_RAD_S``.
        * All aircraft positions are integrated from their current velocities.
        * Altitudes are updated from the ``vertrate`` (ft/min) stored in metadata.
        """
        new_state = state.copy()
        new_state.time += dt

        # --- Ego ---
        diff = angle_diff_rad(ego_action.heading_command, state.ego.heading)
        max_delta = MAX_TURN_RATE_RAD_S * dt
        new_heading = state.ego.heading + float(np.clip(diff, -max_delta, max_delta))
        # Horizontal speed from 3D velocity (ignore vz for turn-rate computation)
        speed = float(np.linalg.norm(state.ego.velocity[:2])) or 1.0
        ego_meta = dict(state.ego.metadata) if state.ego.metadata else {}
        vertrate_fpm = ego_meta.get("vertrate", 0.0)
        new_velocity = heading_to_velocity(math.degrees(new_heading), speed, vertrate_fpm)
        new_position = state.ego.position + state.ego.velocity * dt
        ego_meta["altitude"] = new_position[2] / 0.3048  # keep metadata in sync (ft)
        new_state.ego = AircraftState(
            position=new_position,
            velocity=new_velocity,
            heading=new_heading,
            icao24=state.ego.icao24,
            metadata=ego_meta,
        )

        # --- Traffic (constant heading / speed) ---
        new_traffic = []
        for ac in state.traffic:
            ac_meta = dict(ac.metadata) if ac.metadata else {}
            new_ac_pos = ac.position + ac.velocity * dt
            ac_meta["altitude"] = new_ac_pos[2] / 0.3048  # keep metadata in sync (ft)
            new_traffic.append(AircraftState(
                position=new_ac_pos,
                velocity=ac.velocity.copy(),
                heading=ac.heading,
                icao24=ac.icao24,
                metadata=ac_meta,
            ))
        new_state.traffic = new_traffic
        return new_state

    def _extract_features(
        self,
        index: int,
        ts: float,
        ego_idx: int,
        aircraft_list: List[Dict[str, Any]],
    ) -> CandidateFeatures:
        """Build a ``CandidateFeatures`` descriptor for a candidate entry."""
        ego = aircraft_list[ego_idx]
        proximity = get_distance_nm(
            self.airport_lat, self.airport_lon, ego["lat"], ego["lon"]
        )
        utc_hour = parse_utc_hour(ego["timestamp_utc"])
        return CandidateFeatures(
            index=index,
            timestamp_s=ts,
            utc_hour=utc_hour if utc_hour is not None else -1,
            proximity_nm=proximity,
            altitude_ft=ego["alt_ft"],
            traffic_count=len(aircraft_list) - 1,
            speed_kts=ego["speed_mps"] / KTS_TO_MPS,
        )

    def _build_state(
        self, ts: float, ego_idx: int, aircraft_list: List[Dict[str, Any]]
    ) -> SystemState:
        """Convert a raw snapshot entry to a ``SystemState`` in local ENU metres."""
        def _ac(d: Dict[str, Any]) -> AircraftState:
            pos = latlon_to_local_m(d["lat"], d["lon"], self.airport_lat, self.airport_lon, d["alt_ft"])
            vel = heading_to_velocity(d["heading_deg"], d["speed_mps"], d["vertrate_fpm"])
            return AircraftState(
                position=pos,
                velocity=vel,
                heading=math.radians(d["heading_deg"]),
                icao24=d["icao24"],
                metadata={
                    "altitude":  d["alt_ft"],        # feet
                    "vertrate":  d["vertrate_fpm"],  # ft/min
                    "callsign":  d["callsign"],
                    "lat":       d["lat"],
                    "lon":       d["lon"],
                },
            )

        return SystemState(
            ego=_ac(aircraft_list[ego_idx]),
            traffic=[_ac(ac) for i, ac in enumerate(aircraft_list) if i != ego_idx],
            time=ts,
        )

class AircraftHeadingDisturbanceModel(DisturbanceModel):
    """Independent Gaussian noise on heading commands and position observations.

    Args:
        sigma_heading_rad: Std dev of noise added to heading commands (radians).
                           Set to 0 for a perfect actuator.
        sigma_position_m:  Std dev of noise added to observed aircraft positions
                           (metres).  Set to 0 for perfect sensors.
        seed:              Optional RNG seed.
    """

    def __init__(
        self,
        sigma_heading_rad: float = 0.0,
        sigma_position_m: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.sigma_heading_rad = sigma_heading_rad
        self.sigma_position_m = sigma_position_m
        self.rng = np.random.default_rng(seed)
        self._last_action_log_prob: float = 0.0
        self._last_obs_log_prob: float = 0.0

    def apply_action_disturbance(self, action: Action) -> Action:
        if self.sigma_heading_rad == 0.0:
            self._last_action_log_prob = 0.0
            return action
        noise = float(self.rng.normal(0.0, self.sigma_heading_rad))
        self._last_action_log_prob = gaussian_log_prob(noise, self.sigma_heading_rad)
        return Action(
            heading_command=action.heading_command + noise,
            metadata=action.metadata,
        )

    def apply_observation_disturbance(self, observation: Observation) -> Observation:
        if self.sigma_position_m == 0.0:
            self._last_obs_log_prob = 0.0
            return observation

        total_log_prob = 0.0

        def _perturb(ac: AircraftState) -> AircraftState:
            nonlocal total_log_prob
            # Perturb only horizontal axes; altitude is not a sensor noise source here
            noise = np.zeros(3, dtype=np.float32)
            noise[:2] = self.rng.normal(0.0, self.sigma_position_m, 2).astype(np.float32)
            total_log_prob += gaussian_log_prob(float(noise[0]), self.sigma_position_m)
            total_log_prob += gaussian_log_prob(float(noise[1]), self.sigma_position_m)
            return AircraftState(
                position=ac.position + noise,
                velocity=ac.velocity.copy(),
                heading=ac.heading,
                icao24=ac.icao24,
                metadata=ac.metadata,
            )

        result = Observation(
            ego_state=_perturb(observation.ego_state),
            traffic_states=[_perturb(t) for t in observation.traffic_states],
            graph_data=observation.graph_data,
            features=observation.features,
            metadata=observation.metadata,
        )
        self._last_obs_log_prob = total_log_prob
        return result

    @property
    def last_action_log_prob(self) -> float:
        return self._last_action_log_prob

    @property
    def last_obs_log_prob(self) -> float:
        return self._last_obs_log_prob

    def sample_environment_disturbance(self) -> Dict[str, Any]:
        return {}



class DirectObservationModel(ObservationModel):
    """Pass-through observation: exposes the true system state to the agent."""

    def observe(self, state: SystemState) -> Observation:
        return Observation(
            ego_state=state.ego,
            traffic_states=state.traffic,
        )


class PassthroughAgentModel(AgentModel):
    """Baseline agent that holds the aircraft's current heading.

    Useful for smoke-testing the rollout loop before connecting a real policy.
    """

    def act(self, observation: Observation) -> Action:
        return Action(heading_command=observation.ego_state.heading)


def _path_lookahead(pos: np.ndarray, path: np.ndarray, lookahead_m: float) -> np.ndarray:
    """Return a point *lookahead_m* ahead of *pos* along a polyline *path*.

    Projects *pos* onto the nearest segment of *path*, then walks
    *lookahead_m* further along the remaining polyline.  Returns the
    path endpoint when lookahead exceeds the remaining length.
    """
    if len(path) == 1:
        return path[0].copy()

    # --- find nearest projection onto any segment ---
    best_seg  = 0
    best_frac = 0.0
    best_d2   = float("inf")
    for i in range(len(path) - 1):
        ab      = path[i + 1] - path[i]
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-10:
            continue
        frac = float(np.dot(pos - path[i], ab)) / ab_len2
        frac = max(0.0, min(1.0, frac))
        proj = path[i] + frac * ab
        d2   = float(np.dot(pos - proj, pos - proj))
        if d2 < best_d2:
            best_d2, best_seg, best_frac = d2, i, frac

    # --- walk lookahead_m ahead from that projection ---
    proj      = path[best_seg] + best_frac * (path[best_seg + 1] - path[best_seg])
    remaining = lookahead_m

    # consume rest of the current segment
    to_end = path[best_seg + 1] - proj
    to_end_len = float(np.linalg.norm(to_end))
    if to_end_len >= remaining:
        return proj + (remaining / max(to_end_len, 1e-10)) * to_end
    remaining -= to_end_len

    # walk subsequent segments
    for i in range(best_seg + 1, len(path) - 1):
        seg_len = float(np.linalg.norm(path[i + 1] - path[i]))
        if seg_len >= remaining:
            return path[i] + (remaining / max(seg_len, 1e-10)) * (path[i + 1] - path[i])
        remaining -= seg_len

    return path[-1].copy()


class ReferencePathAgent(AgentModel):
    """Pure-pursuit lateral guidance agent that steers toward a reference path.

    At each step the agent:

    1. Projects the ego position onto the reference polyline.
    2. Finds a lookahead point *lookahead_m* ahead along the path.
    3. Commands a heading directly toward that lookahead point.

    This is significantly more realistic than :class:`PassthroughAgentModel`
    because the aircraft actively corrects cross-track error instead of
    flying a straight line on its initial heading.

    Args:
        reference_path: ``(N, 2)`` float array of ENU (x, y) waypoints in
                        metres, as returned by
                        :meth:`AircraftHeadingEnvironment.get_reference_path`.
        lookahead_m:    Pure-pursuit lookahead distance in metres.  Larger
                        values give smoother but slower corrections;
                        smaller values track tightly but can oscillate.
                        Default: 2 km (roughly 20 s of flight at 100 m/s).
    """

    def __init__(self, reference_path: np.ndarray, lookahead_m: float = 2000.0) -> None:
        self.reference_path = reference_path
        self.lookahead_m    = lookahead_m

    def act(self, observation: Observation) -> Action:
        pos    = observation.ego_state.position[:2]
        target = _path_lookahead(pos, self.reference_path, self.lookahead_m)
        delta  = target - pos
        # ENU convention: x = East, y = North → heading = atan2(East, North)
        heading_cmd = math.atan2(float(delta[0]), float(delta[1]))
        return Action(heading_command=heading_cmd)


class AircraftHeadingPredictorSystem(System):
    """Closed-loop verification system for the aircraft heading predictor.

    Composes an :class:`AircraftHeadingEnvironment`, an :class:`AgentModel`
    (e.g. ``PassthroughAgentModel`` or the ``GNNAgent`` from
    ``scripts/verification/gnn_integration_example.py``), a
    :class:`AircraftHeadingDisturbanceModel`, and a
    :class:`DirectObservationModel`.

    The per-step sequence is:

    1. Extract observation from state (with position noise if configured).
    2. Agent computes a heading command from the observation.
    3. Heading command noise is applied.
    4. Environment propagates kinematics for ego and all traffic.
    5. New ``SystemState`` is returned.

    Args:
        environment:          ``AircraftHeadingEnvironment`` instance.
        agent_model:          Heading predictor policy.
        disturbance_model:    Noise model (default: no noise).
        observation_model:    Observation extractor (default: ``DirectObservationModel``).
        dt:                   Simulation timestep in seconds (default: 10.0).
    """

    def __init__(
        self,
        environment: AircraftHeadingEnvironment,
        agent_model: AgentModel,
        disturbance_model: Optional[AircraftHeadingDisturbanceModel] = None,
        observation_model: Optional[DirectObservationModel] = None,
        dt: float = 10.0,
        heading_deadband_rad: float = 0.0,
    ) -> None:
        if disturbance_model is None:
            disturbance_model = AircraftHeadingDisturbanceModel()
        if observation_model is None:
            observation_model = DirectObservationModel()
        super().__init__(environment, agent_model, observation_model, disturbance_model)
        self.dt = dt
        self.heading_deadband_rad = heading_deadband_rad

    def get_observation(self, state: SystemState) -> Observation:
        obs = self.observation_model.observe(state)
        return self.disturbance_model.apply_observation_disturbance(obs)

    def step(self, state: SystemState, action: Action) -> SystemState:
        """Apply disturbances and propagate one timestep via environment kinematics."""
        noisy_action = self.disturbance_model.apply_action_disturbance(action)
        # Heading deadband: suppress micro-corrections smaller than the threshold.
        # Prevents the agent from issuing tiny continuous corrections that cause
        # slow heading drift (heading-lock failure mode).
        if self.heading_deadband_rad > 0.0:
            diff = angle_diff_rad(noisy_action.heading_command, state.ego.heading)
            if abs(diff) < self.heading_deadband_rad:
                noisy_action = Action(
                    heading_command=state.ego.heading,
                    metadata=noisy_action.metadata,
                )
        return self.environment.step(state, noisy_action, self.dt)

