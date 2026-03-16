"""Library with specification definitions for the aircraft heading guidance generation system.

Specifications are expressed as Boolean predicates over trajectories, following
a discrete-time interpretation of Signal Temporal Logic (STL). Each function
evaluates the always operator (□) over the full rollout horizon [0, T].
"""

import math

import numpy as np
from verification.system.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

NM_TO_METERS = 1852.0          # 1 nautical mile in metres
FT_VERTICAL_SEP_MIN = 1000.0   # vertical separation minimum (feet)

# Anisotropic separation zone defaults.
#
# The exclusion zone around each aircraft is a rectangle in ego-relative
# coordinates, not a circle.  This reflects the real-world asymmetry:
#
#   Longitudinal (fore/aft along ego heading): 3.0 NM
#     — FAA JO 7110.65 standard terminal radar separation.
#     — Reduced to 2.5 NM when ego altitude < 3 000 ft (same-category on final).
#     — This is the dominant constraint: you cannot fly directly behind/ahead
#       of traffic in the same lane.
#
#   Lateral (perpendicular to ego heading): 1.0 NM
#     — Aircraft beside you in a parallel lane separated by ILS/RNAV geometry
#       are routinely < 3 NM laterally and are *not* in conflict.
#     — 1.0 NM is a conservative lateral radar resolution / wake-turbulence
#       buffer for non-parallel-approach situations.
#     — For true parallel ILS approaches the lateral clearance is enforced by
#       approach geometry (~0.12 NM), not a radar minimum, so this value is
#       already much tighter than what ATC maintains.
#
# STL OR-robustness for the rectangle:
#   rho_rect = max(|d_along| - SEP_LON, |d_cross| - SEP_LAT)
#   positive → separated in at least one axis  (no conflict)
#   negative → inside the box                  (conflict)
#
DEFAULT_SEP_LON_NM  = 3.0   # longitudinal minimum (NM)
DEFAULT_SEP_LAT_NM  = 1.0   # lateral minimum (NM) — much narrower than lon
APPROACH_ALT_FT_THRESHOLD = 3000.0   # ft AGL below which approach sep applies
APPROACH_SEP_LON_NM = 2.5           # reduced longitudinal min on final (NM)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Perpendicular distance from point *p* to line segment *a*→*b*.

    If the projection of *p* falls outside the segment, the distance to the
    nearer endpoint is returned instead.
    """
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq == 0.0:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / len_sq, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _cross_track_error(position: np.ndarray, reference_path: np.ndarray) -> float:
    """Minimum perpendicular distance from *position* to the reference polyline.

    Uses only the horizontal (east, north) components; altitude is ignored.
    """
    p2 = position[:2]
    min_dist = np.inf
    for i in range(len(reference_path) - 1):
        a2 = reference_path[i][:2] if reference_path[i].shape[0] > 2 else reference_path[i]
        b2 = reference_path[i+1][:2] if reference_path[i+1].shape[0] > 2 else reference_path[i+1]
        d = _point_to_segment_distance(p2, a2, b2)
        if d < min_dist:
            min_dist = d
    return min_dist


def _angular_diff_rad(a: float, b: float) -> float:
    """Signed difference *a* − *b* wrapped to (−π, π]."""
    return (a - b + np.pi) % (2.0 * np.pi) - np.pi


def _anisotropic_horiz_margin(
    ego_position: np.ndarray,
    ego_heading: float,
    traffic_position: np.ndarray,
    sep_lon_m: float,
    sep_lat_m: float,
) -> float:
    """Horizontal separation margin using a rectangular (anisotropic) exclusion zone.

    Decomposes the relative position vector into the ego aircraft's own frame:

    * **Longitudinal** axis — along the ego heading vector.  This captures
      fore-aft separation: aircraft directly ahead or behind in the same
      corridor.
    * **Lateral** axis — perpendicular to the ego heading.  This captures
      side-by-side separation: aircraft in a parallel lane.

    Returns the STL OR-robustness of the rectangular exclusion zone::

        rho = max(|d_along| − sep_lon_m,  |d_cross| − sep_lat_m)

    Positive → separated in at least one axis (no conflict).
    Negative → inside the rectangle (conflict); magnitude = deepest penetration.

    Args:
        ego_position:    (2,) ENU position of ego aircraft (metres).
        ego_heading:     Ego heading in radians (0 = North, clockwise).
        traffic_position:(2,) ENU position of traffic aircraft (metres).
        sep_lon_m:       Longitudinal separation minimum (metres).
        sep_lat_m:       Lateral separation minimum (metres).
    """
    rel = (traffic_position - ego_position)[:2]          # horizontal only (east, north)
    # Forward unit vector in ENU: heading 0=North, clockwise → sin/cos decomp
    fwd = np.array([math.sin(ego_heading), math.cos(ego_heading)], dtype=np.float64)
    rgt = np.array([math.cos(ego_heading), -math.sin(ego_heading)], dtype=np.float64)
    d_along = abs(float(np.dot(rel, fwd)))   # fore/aft distance
    d_cross = abs(float(np.dot(rel, rgt)))   # side distance
    return max(d_along - sep_lon_m, d_cross - sep_lat_m)


def _horizontal_sep_min_m(
    r_m: float,
    sep_tiers=None,
    ego_alt_ft: float = None,
) -> float:
    """Legacy isotropic helper — kept for cross_track_… callers only.

    Returns the longitudinal separation minimum based on range from airport
    (en-route 5 NM / terminal 3 NM boundary at 40 NM), with an altitude gate
    that reduces the terminal tier to 2.5 NM when on final approach.
    """
    nm = r_m / NM_TO_METERS
    if nm > 40.0:
        base_nm = 5.0
    else:
        base_nm = 3.0
        if ego_alt_ft is not None and ego_alt_ft < APPROACH_ALT_FT_THRESHOLD:
            base_nm = APPROACH_SEP_LON_NM
    return base_nm * NM_TO_METERS


def _range_d_max(
    position: np.ndarray,
    airport_position: np.ndarray,
    d_max_cap: float,
    theta_max_rad: float,
) -> float:
    """Angular-corridor cross-track tolerance at one waypoint.

    Returns ``min(d_max_cap, r × tan(θ_max))``, where *r* is the aircraft's
    ENU distance from the airport.  This tightens the cross-track constraint
    proportionally with range, enforcing a fixed angular corridor around the
    nominal reference track rather than a flat absolute bound.
    """
    r = float(np.linalg.norm((position - airport_position)[:2]))
    return min(d_max_cap, r * float(np.tan(theta_max_rad)))


# ---------------------------------------------------------------------------
# Individual specifications
# ---------------------------------------------------------------------------


def cross_track_spec(
    trajectory: Trajectory,
    reference_path: np.ndarray,
    d_max: float,
    airport_position=None,
    theta_max_rad: float = 0.0,
) -> bool:
    """Evaluate φ_track = □[0,T] (e_xt(t) ≤ d_max(t)).

    When *airport_position* and *theta_max_rad* > 0 are supplied the tolerance
    tightens with range: d_max(t) = min(d_max, r(t) × tan(θ_max)).  This
    enforces a constant angular corridor rather than a flat absolute bound,
    which is the physically meaningful constraint across all approach ranges.

    Args:
        trajectory:       Rollout trajectory.
        reference_path:   (N, 2) array of 2-D reference path waypoints (metres).
        d_max:            Absolute upper-bound cap on cross-track tolerance (m).
        airport_position: (2,) airport ENU position (metres).  Required when
                          *theta_max_rad* > 0.
        theta_max_rad:    Angular corridor half-width (radians).  0 → flat cap.

    Returns:
        ``True`` if the cross-track constraint is satisfied at every timestep.
    """
    use_angular = airport_position is not None and theta_max_rad > 0.0
    for step in trajectory:
        tol = (_range_d_max(step.state.ego.position, airport_position, d_max, theta_max_rad)
               if use_angular else d_max)
        e_xt = _cross_track_error(step.state.ego.position[:2], reference_path)
        if e_xt > tol:
            return False
    return True


def separation_spec(
    trajectory: Trajectory,
    airport_position: np.ndarray,
    sep_lon_nm: float = DEFAULT_SEP_LON_NM,
    sep_lat_nm: float = DEFAULT_SEP_LAT_NM,
) -> bool:
    """Evaluate φ_sep using an anisotropic rectangular exclusion zone.

    Separation is satisfied when, for every traffic pair, at least one of:
      * Longitudinal clearance ≥ sep_lon_nm  (along ego heading)
      * Lateral clearance     ≥ sep_lat_nm  (perpendicular to ego heading)
      * Vertical clearance    ≥ 1 000 ft    (when altitude available)

    The longitudinal minimum is reduced to APPROACH_SEP_LON_NM (2.5 NM) when
    the ego altitude is below APPROACH_ALT_FT_THRESHOLD (3 000 ft).

    Args:
        trajectory:   Rollout trajectory.
        airport_position: unused (kept for API compatibility).
        sep_lon_nm:   Longitudinal separation minimum (NM).
        sep_lat_nm:   Lateral separation minimum (NM).

    Returns:
        ``True`` if separation is satisfied at every timestep.
    """
    for step in trajectory:
        ego     = step.state.ego
        ego_alt = ego.metadata.get("altitude") if ego.metadata else None
        if ego_alt is None and ego.position.shape[0] > 2:
            ego_alt = float(ego.position[2]) / 0.3048
        lon_nm  = (APPROACH_SEP_LON_NM
                   if ego_alt is not None and ego_alt < APPROACH_ALT_FT_THRESHOLD
                   else sep_lon_nm)
        sep_lon_m = lon_nm * NM_TO_METERS
        sep_lat_m = sep_lat_nm * NM_TO_METERS

        for traffic_ac in step.state.traffic:
            h_margin = _anisotropic_horiz_margin(
                ego.position, ego.heading, traffic_ac.position, sep_lon_m, sep_lat_m
            )
            if h_margin >= 0:
                continue  # horizontally separated in at least one axis

            # Inside the rectangle — check vertical separation as escape
            if ego_alt is not None and traffic_ac.metadata is not None:
                traffic_alt = traffic_ac.metadata.get("altitude")
                if traffic_alt is not None and abs(ego_alt - traffic_alt) >= FT_VERTICAL_SEP_MIN:
                    continue

            return False

    return True


def heading_rate_spec(
    trajectory: Trajectory,
    psi_dot_max: float,
    dt: float,
) -> bool:
    """Evaluate φ_rate = □[0,T] (|ψ̇(t)| ≤ ψ̇_max).

    The commanded heading rate at step *t* is computed as the circular
    difference between consecutive heading commands divided by the timestep:

        ψ̇(t) = (ψ_t − ψ_{t-1}) / Δt

    Args:
        trajectory:  Rollout trajectory.
        psi_dot_max: Maximum allowable heading rate (radians/second).
        dt:          Simulation timestep Δt (seconds).

    Returns:
        ``True`` if the heading-rate constraint is satisfied at every timestep.
    """
    for i in range(1, len(trajectory)):
        psi_curr = trajectory[i].action.heading_command
        psi_prev = trajectory[i - 1].action.heading_command
        psi_dot = _angular_diff_rad(psi_curr, psi_prev) / dt
        if abs(psi_dot) > psi_dot_max:
            return False
    return True


# ---------------------------------------------------------------------------
# Combined safety specification
# ---------------------------------------------------------------------------


def safety_spec(
    trajectory: Trajectory,
    airport_position: np.ndarray,
    psi_dot_max: float,
    dt: float,
    sep_lon_nm: float = DEFAULT_SEP_LON_NM,
    sep_lat_nm: float = DEFAULT_SEP_LAT_NM,
) -> bool:
    """Evaluate the combined safety specification φ_sep ∧ φ_rate.

    Cross-track error is intentionally excluded: the GNN is an advisory heading
    predictor for approach/terminal traffic, not a path-following controller.
    At terminal ranges a large cross-track deviation does not constitute a safety
    violation — it just means the pilot chose a different routing.  Separation
    and heading-rate are the physically meaningful safety constraints.
    """
    return (
        separation_spec(trajectory, airport_position, sep_lon_nm, sep_lat_nm)
        and heading_rate_spec(trajectory, psi_dot_max, dt)
    )


# ===========================================================================
# Quantitative STL robustness (ρ semantics)
# ===========================================================================
# Positive ρ → specification satisfied with margin ρ.
# Negative ρ → specification violated (|ρ| measures severity).
#
# Individual robustness functions mirror the Boolean specs above but return
# the real-valued "distance to boundary" at the tightest point in time.
# ===========================================================================


def cross_track_robustness(
    trajectory: Trajectory,
    reference_path: np.ndarray,
    d_max: float,
    airport_position=None,
    theta_max_rad: float = 0.0,
) -> float:
    """Quantitative robustness of φ_track over *trajectory*.

    ρ(φ_track) = min_t (d_max(t) − e_xt(t))

    When *theta_max_rad* > 0, d_max(t) = min(d_max, r(t) × tan(θ_max)).

    Args:
        trajectory:       Rollout trajectory.
        reference_path:   (N, 2) array of 2-D reference path waypoints (metres).
        d_max:            Absolute upper-bound cap on cross-track tolerance (m).
        airport_position: (2,) airport ENU position.  Required for angular mode.
        theta_max_rad:    Angular corridor half-width (radians).  0 → flat cap.

    Returns:
        Minimum margin to the cross-track constraint boundary (metres).
    """
    if not trajectory:
        return float("inf")
    use_angular = airport_position is not None and theta_max_rad > 0.0
    return min(
        (_range_d_max(step.state.ego.position, airport_position, d_max, theta_max_rad)
         if use_angular else d_max)
        - _cross_track_error(step.state.ego.position[:2], reference_path)
        for step in trajectory
    )


def separation_robustness(
    trajectory: Trajectory,
    airport_position: np.ndarray,
    sep_lon_nm: float = DEFAULT_SEP_LON_NM,
    sep_lat_nm: float = DEFAULT_SEP_LAT_NM,
) -> float:
    """Quantitative robustness of φ_sep using the anisotropic rectangular zone.

    STL robustness for the OR-conjunction over all pairs::

        rho_horiz(pair) = max(|d_along| - sep_lon_m,  |d_cross| - sep_lat_m)
        rho_pair        = max(rho_horiz, d_v - 1000)   # OR with vertical
        rho_step        = min_pairs rho_pair            # AND over pairs
        rho(phi_sep)    = min_t rho_step(t)            # always

    Positive → all pairs separated; negative → at least one pair in conflict.

    Args:
        trajectory:       Rollout trajectory.
        airport_position: unused (kept for API compatibility).
        sep_lon_nm:       Longitudinal separation minimum (NM).
        sep_lat_nm:       Lateral separation minimum (NM).
    """
    if not trajectory:
        return float("inf")

    step_robustnesses = []
    for step in trajectory:
        ego     = step.state.ego
        ego_alt = ego.metadata.get("altitude") if ego.metadata else None
        if ego_alt is None and ego.position.shape[0] > 2:
            ego_alt = float(ego.position[2]) / 0.3048
        lon_nm  = (APPROACH_SEP_LON_NM
                   if ego_alt is not None and ego_alt < APPROACH_ALT_FT_THRESHOLD
                   else sep_lon_nm)
        sep_lon_m = lon_nm * NM_TO_METERS
        sep_lat_m = sep_lat_nm * NM_TO_METERS

        if not step.state.traffic:
            step_robustnesses.append(float("inf"))
            continue

        pair_margins = []
        for traffic_ac in step.state.traffic:
            h_margin = _anisotropic_horiz_margin(
                ego.position, ego.heading, traffic_ac.position, sep_lon_m, sep_lat_m
            )
            if ego_alt is not None and traffic_ac.metadata is not None:
                traffic_alt = traffic_ac.metadata.get("altitude")
                if traffic_alt is not None:
                    d_v = abs(ego_alt - traffic_alt)
                    pair_margins.append(max(h_margin, d_v - FT_VERTICAL_SEP_MIN))
                    continue
            pair_margins.append(h_margin)

        step_robustnesses.append(min(pair_margins))

    return min(step_robustnesses)


def heading_rate_robustness(
    trajectory: Trajectory,
    psi_dot_max: float,
    dt: float,
) -> float:
    """Quantitative robustness of φ_rate over *trajectory*.

    ρ(φ_rate) = min_{t≥1} (ψ̇_max − |ψ̇(t)|)

    Args:
        trajectory:  Rollout trajectory.
        psi_dot_max: Maximum allowable heading rate (radians/second).
        dt:          Simulation timestep Δt (seconds).

    Returns:
        Minimum margin to the heading-rate constraint boundary (rad/s).
        Returns +∞ for trajectories shorter than 2 steps.
    """
    if len(trajectory) < 2:
        return float("inf")
    return min(
        psi_dot_max - abs(_angular_diff_rad(trajectory[i].action.heading_command,
                                            trajectory[i - 1].action.heading_command) / dt)
        for i in range(1, len(trajectory))
    )


def safety_robustness(
    trajectory: Trajectory,
    airport_position: np.ndarray,
    psi_dot_max: float,
    dt: float,
    sep_lon_nm: float = DEFAULT_SEP_LON_NM,
    sep_lat_nm: float = DEFAULT_SEP_LAT_NM,
) -> float:
    """Quantitative robustness of the combined safety specification φ_sep ∧ φ_rate.

    ρ(φ_safety) = min(ρ(φ_sep), ρ(φ_rate))

    Cross-track error is excluded — see :func:`safety_spec` for rationale.
    The individual :func:`cross_track_robustness` function remains available
    for diagnostic/exploratory use.
    """
    return min(
        separation_robustness(trajectory, airport_position, sep_lon_nm, sep_lat_nm),
        heading_rate_robustness(trajectory, psi_dot_max, dt),
    )


def safety_robustness_normalized(
    trajectory: Trajectory,
    airport_position: np.ndarray,
    psi_dot_max: float,
    dt: float,
    sep_lon_nm: float = DEFAULT_SEP_LON_NM,
    sep_lat_nm: float = DEFAULT_SEP_LAT_NM,
) -> float:
    """Normalized combined robustness — dimensionless, directly comparable.

    Each component is divided by its own violation budget before the min:

        ρ̃_sep  = ρ_sep  / (sep_lon_nm × NM_TO_METERS)
        ρ̃_rate = ρ_rate / ψ̇_max

    Both map to the same scale:
        +1   → margin equal to the full budget (very safe)
         0   → exactly at the constraint boundary
        −1   → violating by a full budget's worth

    Sign is preserved under normalization, so violation detection is identical
    to the raw version.  Use this when comparing or plotting the two specs
    against each other (e.g. scatter plots, failure histograms).
    """
    rho_sep  = separation_robustness(trajectory, airport_position, sep_lon_nm, sep_lat_nm)
    rho_rate = heading_rate_robustness(trajectory, psi_dot_max, dt)
    sep_budget = sep_lon_nm * NM_TO_METERS
    return min(rho_sep / sep_budget, rho_rate / psi_dot_max)
