#!/usr/bin/env python3
"""Verification rollout runner.

Rolls out the GNN heading predictor (or a passthrough baseline) under the
kinematic simulation and evaluates the three STL safety specifications against
the ground-truth ADS-B reference path.

The reference path is ONLY used post-hoc to evaluate specs — it is never
visible to the agent during the rollout.

Usage (from workspace root):
    # GNN agent (default) — best GAT checkpoint (79.7% @15°, hidden=128)
    python scripts/verification/run_rollouts.py

    # Explicit checkpoint path
    python scripts/verification/run_rollouts.py \\
        --checkpoint checkpoints_parquet/run_20251210_174458/best_model.pt --model-type gat

    # Kinematic baseline (holds initial heading, no model)
    python scripts/verification/run_rollouts.py --agent passthrough

    # More rollouts, longer horizon
    python scripts/verification/run_rollouts.py \\
        --rollouts 10 --steps 30
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from verification.system.aircraft_heading_predictor_system import (
    AircraftHeadingDisturbanceModel,
    AircraftHeadingEnvironment,
    AircraftHeadingPredictorSystem,
    PassthroughAgentModel,
)
from verification.system.gnn_agent import GNNAgent
from verification.system.rollouts import rollout
from verification.system.state import AircraftState, SystemState
from verification.system.trajectory import TrajectoryStep
from verification.specifications.aircraft_heading_spec import (
    cross_track_robustness,
    heading_rate_robustness,
    safety_robustness,
    safety_robustness_normalized,
    separation_robustness,
)

CROSS_TRACK_D_MAX = 5.0 * 1852.0  # 5 NM flat cap for diagnostic cross-track logging

AIRPORT_LAT = 47.4498889
AIRPORT_LON = -122.3117778
AIRPORT_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
PSI_DOT_MAX = math.radians(3.0)  # 3 deg/s — matches MAX_TURN_RATE_RAD_S
DT_DEFAULT  = 10.0             # seconds per step (default; override with --dt)
MAX_ALT_FT_DEFAULT = 10_000.0  # exclude en-route traffic above this altitude


# ---------------------------------------------------------------------------
# Reference-path baseline evaluation
# ---------------------------------------------------------------------------

def _build_ref_trajectory(
    ref: np.ndarray,
    initial_state: SystemState,
    dt: float,
) -> list:
    """Synthesise a trajectory where ego follows the ADS-B reference path.

    Traffic is propagated at constant heading/speed from the initial state —
    exactly as the kinematic simulation does.  The resulting trajectory is
    evaluated with the same specs as the agent rollout, so we can distinguish:

    * **Structural violations**: the nominal ADS-B path *also* fails the spec
      (inherent scenario geometry, not caused by the GNN).
    * **Agent-fault violations**: the ADS-B path passes but the agent rollout
      fails (degradation introduced by the GNN's heading predictions).

    Args:
        ref:           (M, 2) float32 array of ENU positions from
                       ``env.get_reference_path()``.
        initial_state: SystemState at rollout t=0 (provides initial traffic).
        dt:            Simulation timestep in seconds.

    Returns:
        List of ``TrajectoryStep`` with only ``.state`` populated (action,
        observation, and next_state are ``None``); sufficient for
        ``separation_robustness``.
    """
    steps = []
    for i, ego_pos in enumerate(ref):
        t_elapsed = i * dt
        # Propagate traffic from t=0 at constant velocity; update altitude from vertrate
        traffic = []
        for ac in initial_state.traffic:
            ac_meta = dict(ac.metadata) if ac.metadata else {}
            if "altitude" in ac_meta and "vertrate" in ac_meta:
                ac_meta["altitude"] = (
                    ac_meta["altitude"] + ac_meta["vertrate"] * t_elapsed / 60.0
                )
            traffic.append(AircraftState(
                position=ac.position + ac.velocity * t_elapsed,
                velocity=ac.velocity.copy(),
                heading=ac.heading,
                icao24=ac.icao24,
                metadata=ac_meta,
            ))
        ego_meta = dict(initial_state.ego.metadata) if initial_state.ego.metadata else {}
        if "altitude" in ego_meta and "vertrate" in ego_meta:
            ego_meta["altitude"] = (
                ego_meta["altitude"] + ego_meta["vertrate"] * t_elapsed / 60.0
            )
        # Build 3D position: if ref positions are 2D, append altitude from metadata
        ref_pos_2d = np.asarray(ego_pos, dtype=np.float32)
        if ref_pos_2d.shape[0] == 2:
            alt_m = ego_meta.get("altitude", 0.0) * 0.3048
            ref_pos_3d = np.array([ref_pos_2d[0], ref_pos_2d[1], alt_m], dtype=np.float32)
        else:
            ref_pos_3d = ref_pos_2d
        ego = AircraftState(
            position=ref_pos_3d,
            velocity=initial_state.ego.velocity.copy(),
            heading=initial_state.ego.heading,
            icao24=initial_state.ego.icao24,
            metadata=ego_meta,
        )
        state = SystemState(
            time=initial_state.time + t_elapsed,
            ego=ego,
            traffic=traffic,
        )
        steps.append(TrajectoryStep(
            state=state,
            action=None,
            observation=None,
            next_state=None,
            disturbance_log_prob=0.0,
        ))
    return steps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run GNN verification rollouts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--agent",          choices=["gnn", "passthrough"], default="gnn",
                   help="'gnn' uses the trained heading predictor; "
                        "'passthrough' holds the initial heading (kinematic baseline)")
    p.add_argument("--checkpoint",     default="checkpoints_parquet/run_20251210_174458/best_model.pt",
                   help="Path to model checkpoint (.pt state dict). "
                        "Relative paths are resolved from workspace root.")
    p.add_argument("--model-type",     choices=["gat", "gcn"], default="gat",
                   help="Architecture matching the checkpoint")
    p.add_argument("--rollouts",       type=int,   default=3)
    p.add_argument("--steps",          type=int,   default=20)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--min-path-pts",   type=int,   default=3,
                   help="Skip scenarios where the ADS-B reference path has "
                        "fewer than this many points ahead")
    p.add_argument("--proximity-nm",   type=str,   default="near",
                   help="Range filter for initial-state sampling. "
                        "Named: 'near' (0-40 NM), 'far' (40-150 NM). "
                        "Or 'lo,hi' numeric pair, e.g. '5,40'. "
                        "Use 'all' to disable filtering.")
    p.add_argument("--max-alt-ft",     type=float, default=MAX_ALT_FT_DEFAULT,
                   help="Exclude ego aircraft above this altitude (feet). "
                        "Removes en-route traffic that is outside the model's "
                        "intended approach/terminal operating envelope.")
    p.add_argument("--min-dist-nm",    type=float, default=2.0,
                   help="Minimum ego-aircraft distance from airport (NM). "
                        "Excludes aircraft on short final/landing roll where the "
                        "kinematic model is invalid and the angular corridor "
                        "collapses to <100 m. Set to 0 to disable.")
    p.add_argument("--dt",             type=float, default=DT_DEFAULT,
                   help="Simulation timestep in seconds.")
    p.add_argument("--deadband-deg",   type=float, default=0.0,
                   help="Heading deadband: ignore predicted corrections smaller than "
                        "this many degrees. Prevents heading-lock micro-jitter. "
                        "Sensible range: 1.0–3.0°.")
    p.add_argument("--save-json",      type=str,   default=None, metavar="FILE",
                   help="Write a JSON file with per-rollout trajectory data for "
                        "downstream visualisation (see visualize_rollouts.py). "
                        "Paths relative to workspace root.")
    p.add_argument("--filter-structural", action="store_true",
                   help="Skip scenarios where the ADS-B reference path already "
                        "fails the separation or cross-track spec (structural "
                        "violations). All remaining violations are then cleanly "
                        "agent-caused.")
    return p.parse_args()


def _build_agent(args: argparse.Namespace):
    if args.agent == "passthrough":
        return PassthroughAgentModel(), "Passthrough (holds initial heading)"

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Pass --checkpoint with a valid path relative to the workspace root."
        )

    agent = GNNAgent.from_checkpoint(
        checkpoint_path=str(ckpt),
        model_type=args.model_type,
        airport_lat=AIRPORT_LAT,
        airport_lon=AIRPORT_LON,
    )
    return agent, f"GNN ({args.model_type.upper()}) — {ckpt.name}"


def main() -> None:
    args  = parse_args()
    agent, agent_label = _build_agent(args)

    # Parse proximity filter and apply min-dist-nm floor
    _BUCKETS = {"near": (0.0, 40.0), "far": (40.0, 150.0)}
    prox_arg = args.proximity_nm.strip()
    if prox_arg == "all":
        raw_lo, raw_hi = 0.0, float("inf")
    elif "," in prox_arg:
        lo_s, hi_s = prox_arg.split(",", 1)
        raw_lo, raw_hi = float(lo_s), float(hi_s)
    else:
        raw_lo, raw_hi = _BUCKETS[prox_arg]

    effective_lo = max(raw_lo, args.min_dist_nm)
    if effective_lo == 0.0 and raw_hi == float("inf"):
        proximity_nm = None
    else:
        proximity_nm = (effective_lo, raw_hi)

    # Altitude filter: cap to exclude en-route traffic
    max_alt_ft = args.max_alt_ft

    env = AircraftHeadingEnvironment(
        data_path="data/seatac_adsb_states.json",
        airport_lat=AIRPORT_LAT,
        airport_lon=AIRPORT_LON,
        proximity_nm=proximity_nm,
        max_alt_ft=max_alt_ft,
        seed=args.seed,
    )
    # No per-step disturbance: randomness comes from initial-state sampling only.
    # Rollouts are otherwise deterministic so failure results reflect the model
    # itself rather than noise injected during simulation.
    dm = AircraftHeadingDisturbanceModel(
        sigma_heading_rad=0.0,
        sigma_position_m=0.0,
        seed=args.seed,
    )
    system = AircraftHeadingPredictorSystem(
        environment=env,
        agent_model=agent,
        disturbance_model=dm,
        dt=args.dt,
        heading_deadband_rad=math.radians(args.deadband_deg),
    )

    DT = args.dt  # local alias used in per-rollout calculations below
    if isinstance(proximity_nm, tuple):
        prox_label = f"{proximity_nm[0]:.0f}–{proximity_nm[1]:.0f} NM"
    elif proximity_nm is None:
        prox_label = "all ranges"
    else:
        prox_label = str(proximity_nm)
    print(f"Agent      : {agent_label}")
    print(f"Airport    : KSEA (SeaTac)  proximity={prox_label}  max_alt={args.max_alt_ft:.0f} ft")
    print(f"Scenarios  : {env.num_scenarios} candidates")
    print(f"Config     : {args.rollouts} rollouts × {args.steps} steps × {DT:.1f}s"
          f" = {args.steps * DT:.0f}s horizon, "
          f"spec=sep+rate, deterministic rollouts (initial-state sampling only)")
    if args.deadband_deg > 0.0:
        print(f"Deadband   : {args.deadband_deg:.1f}°")
    if args.save_json:
        print(f"Output JSON: {args.save_json}")
    if args.filter_structural:
        print(f"Filter     : structural scenarios excluded (ref-path pre-check)")
    print("=" * 70)

    rollout_count = 0
    attempts      = 0
    violations    = 0
    structural_violations = 0   # violations where the ref path also fails
    structural_skipped    = 0   # scenarios skipped by --filter-structural
    max_attempts  = args.rollouts * 20
    rollout_log   = []  # populated when --save-json is given

    while rollout_count < args.rollouts and attempts < max_attempts:
        attempts += 1
        state = env.sample_initial_state()

        # Reference path: ground-truth ADS-B positions — POST-HOC JUDGE ONLY
        ref = env.get_reference_path(state, num_steps=args.steps, dt=DT)
        if len(ref) < args.min_path_pts:
            continue

        # --filter-structural: skip scenarios where the ADS-B ref path itself fails
        if args.filter_structural:
            ref_traj_pre = _build_ref_trajectory(ref, state, DT)
            if separation_robustness(ref_traj_pre, AIRPORT_POS) < 0:
                structural_skipped += 1
                continue

        traj = rollout(system, initial_state=state, num_steps=args.steps)
        rollout_count += 1

        ego      = state.ego
        prox_nm  = float(np.linalg.norm(ego.position)) / 1852.0
        alt      = ego.metadata.get("altitude", float("nan")) if ego.metadata else float("nan")
        callsign = (ego.metadata.get("callsign") or ego.icao24) if ego.metadata else ego.icao24

        print(f"\nRollout {rollout_count}  |  {callsign} ({ego.icao24})")
        print(f"  t={state.time:.0f}s   {prox_nm:.1f} NM   {alt:.0f} ft   "
              f"hdg={math.degrees(ego.heading):.0f}°   {ego.speed:.1f} m/s   "
              f"ref={len(ref)} pts")

        print(f"\n  {'Step':>4}  {'hdg_cmd°':>9}  {'Δhdg°/s':>8}  {'conf%':>6}")
        for i, step in enumerate(traj):
            hdg_cmd = math.degrees(step.action.heading_command) % 360
            conf    = (step.action.metadata or {}).get("confidence", float("nan"))
            dpsi_ds = 0.0
            if i > 0:
                diff    = step.action.heading_command - traj[i-1].action.heading_command
                diff    = (diff + math.pi) % (2 * math.pi) - math.pi
                dpsi_ds = abs(math.degrees(diff / DT))
            print(f"  {i+1:>4}  {hdg_cmd:>9.1f}  {dpsi_ds:>8.3f}  {conf*100:>5.1f}%")

        rho_sep   = separation_robustness(traj, AIRPORT_POS)
        rho_rate  = heading_rate_robustness(traj, PSI_DOT_MAX, DT)
        rho_track = cross_track_robustness(traj, ref, CROSS_TRACK_D_MAX)
        rho_all   = safety_robustness(traj, AIRPORT_POS, PSI_DOT_MAX, DT)
        rho_norm  = safety_robustness_normalized(traj, AIRPORT_POS, PSI_DOT_MAX, DT)

        # --- Reference baseline: does the ground-truth ADS-B path also fail? ---
        ref_traj     = _build_ref_trajectory(ref, state, DT)
        ref_rho_sep  = separation_robustness(ref_traj, AIRPORT_POS)
        ref_rho_all  = ref_rho_sep

        structural_sep  = ref_rho_sep < 0    # inherent violation — not agent's fault

        if rho_all < 0:
            violations += 1
            if structural_sep:
                structural_violations += 1

        status = "VIOLATION" if rho_all < 0 else "SAFE"
        # Annotate whether the violation is agent-caused or structural
        if rho_all < 0 and structural_sep:
            verdict = "VIOLATION — structural (ref also fails)"
        elif rho_all < 0:
            verdict = "VIOLATION — agent fault"
        else:
            verdict = "SAFE"

        print(f"\n  [{verdict}]")
        print(f"    φ_sep    : {rho_sep:>10.1f} m      {'✓' if rho_sep >= 0 else '✗'}  "
              f"  (ref: {ref_rho_sep:>8.1f} m {'✓' if ref_rho_sep >= 0 else '✗'})")
        print(f"    φ_rate   : {math.degrees(rho_rate):>10.3f} °/s  {'✓' if rho_rate >= 0 else '✗'}")
        print(f"    φ_track  : {rho_track/1852:>10.3f} NM  {'✓' if rho_track >= 0 else '✗'}  (diagnostic)")
        print(f"    φ_safety : {rho_all:>10.3f}        {'✓' if rho_all  >= 0 else '✗'}")
        print(f"    ρ_norm   : {rho_norm:>10.3f}        (dimensionless)")
        print("-" * 70)

        # ---------------------------------------------------------------
        # Trajectory logging for --save-json visualisation
        # ---------------------------------------------------------------
        if args.save_json is not None:
            cos_lat = math.cos(math.radians(AIRPORT_LAT))

            def _enu_to_ll(pos):
                """Convert ENU (east_m, north_m) relative to airport → [lat, lon]."""
                lat = float(pos[1]) / 111_320.0 + AIRPORT_LAT
                lon = float(pos[0]) / (111_320.0 * cos_lat) + AIRPORT_LON
                return [round(lat, 7), round(lon, 7)]

            ego_track = [_enu_to_ll(step.state.ego.position) for step in traj]
            ref_track = [_enu_to_ll(p) for p in ref]

            traffic_tracks: dict = {}
            for step in traj:
                for ac in step.state.traffic:
                    traffic_tracks.setdefault(ac.icao24, []).append(
                        _enu_to_ll(ac.position)
                    )

            step_confs = [
                round((step.action.metadata or {}).get("confidence", 0.0) * 100, 1)
                for step in traj
            ]
            step_hdgs = [
                round(math.degrees(step.action.heading_command) % 360, 1)
                for step in traj
            ]

            rollout_log.append({
                "rollout":          rollout_count,
                "callsign":         callsign,
                "icao24":           ego.icao24,
                "initial_dist_nm":  round(prox_nm, 3),
                "initial_alt_ft":   round(float(alt), 0),
                "initial_hdg_deg":  round(math.degrees(ego.heading) % 360, 1),
                "status":           status,
                # --- agent robustness (4 specs) ---
                "rho_sep":          round(float(rho_sep), 1),
                "rho_rate":         round(float(math.degrees(rho_rate)), 4),
                "rho_track":        round(float(rho_track / 1852.0), 4),  # stored in NM
                "rho_norm":         round(float(rho_norm), 4),
                # --- reference robustness (separation only — only meaningful reference check) ---
                "ref_rho_sep":      round(float(ref_rho_sep), 1),
                # ego_track / ref_track / traffic for visualisation
                "ego_track":        ego_track,
                "ref_track":        ref_track,
                "traffic_tracks":   traffic_tracks,
                "hdg_commands":     step_hdgs,
                "confidences":      step_confs,
            })

    print(f"\nSummary: {violations}/{rollout_count} violations "
          f"(MC failure rate ≈ {violations/max(rollout_count,1):.2%})")
    agent_fault = violations - structural_violations
    print(f"         {structural_violations}/{violations} structural (ref path also fails) — not agent-caused")
    print(f"         {agent_fault}/{violations} agent-fault (ref passes, agent fails)")
    if args.filter_structural and structural_skipped > 0:
        print(f"         {structural_skipped} structural scenarios skipped before rollout (--filter-structural)")
    if rollout_count < args.rollouts:
        print(f"Warning: only {rollout_count}/{args.rollouts} rollouts completed "
              "(increase --rollouts or relax --min-path-pts)")

    if args.save_json is not None and rollout_log:
        out_path = Path(args.save_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "meta": {
                    "agent":          agent_label,
                    "checkpoint":     args.checkpoint,
                    "rollouts":       rollout_count,
                    "steps":          args.steps,
                    "dt_s":           DT,
                    "deadband_deg":   args.deadband_deg,
                    "max_alt_ft":     args.max_alt_ft,
                    "airport_lat":    AIRPORT_LAT,
                    "airport_lon":    AIRPORT_LON,
                    "violations":     violations,
                },
                "rollouts": rollout_log,
            }, f, indent=2)
        print(f"Saved trajectory log → {out_path}")


if __name__ == "__main__":
    main()

