#!/usr/bin/env python3
"""SMC failure search for the GNN aircraft heading predictor.

Runs the Sequential Monte Carlo failure-search algorithm to concentrate a
population of initial scenarios toward safety-specification violations.

Unlike ``run_rollouts.py`` (which draws each scenario independently), SMC
maintains a population over multiple generations: each generation evaluates
all N particles, retains the elite fraction (lowest robustness), and resamples
the next generation from those elites with a Gaussian perturbation on ego
position and heading.

The scoring function is the negated normalised safety robustness:
    score = −ρ_norm = −min(ρ_sep / budget_sep, ρ_rate / ψ̇_max)
so positive score ↔ specification violation.

Usage (from workspace root):
    # Default: 100 particles, 5 generations, GNN agent
    python scripts/verification/run_smc.py

    # Larger search with saved output
    python scripts/verification/run_smc.py \\
        --population 200 --generations 10 \\
        --save-json analysis/smc_results.json

    # Passthrough baseline for comparison
    python scripts/verification/run_smc.py --agent passthrough

    # Tune perturbation kernel
    python scripts/verification/run_smc.py \\
        --sigma-pos-m 500 --sigma-hdg-deg 10 --elite-frac 0.1
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
from verification.system.state import AircraftState, SystemState
from verification.system.trajectory import Trajectory, TrajectoryStep
from verification.specifications.aircraft_heading_spec import (
    cross_track_robustness,
    heading_rate_robustness,
    safety_robustness,
    safety_robustness_normalized,
    separation_robustness,
)
from verification.failures_estimation.sequential_montecarlo import (
    smc_failure_search,
    sample_seed_population,
)

CROSS_TRACK_D_MAX = 5.0 * 1852.0  # 5 NM flat cap for diagnostic logging

AIRPORT_LAT = 47.4498889
AIRPORT_LON = -122.3117778
AIRPORT_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
PSI_DOT_MAX = math.radians(3.0)  # 3 deg/s — matches MAX_TURN_RATE_RAD_S
DT_DEFAULT = 10.0
MAX_ALT_FT_DEFAULT = 10_000.0


# ---------------------------------------------------------------------------
# Reference-path construction (identical to run_rollouts.py)
# ---------------------------------------------------------------------------

def _build_ref_trajectory(
    ref: np.ndarray,
    initial_state: SystemState,
    dt: float,
) -> list:
    """Synthesise a trajectory where ego follows the ADS-B reference path.

    Traffic is propagated at constant heading/speed from the initial state —
    exactly as the kinematic simulation does.  Used post-hoc to distinguish
    structural violations (ref path also fails) from agent-fault violations.
    """
    steps = []
    for i, ego_pos in enumerate(ref):
        t_elapsed = i * dt
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SMC failure search for the GNN heading predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Agent
    p.add_argument("--agent",        choices=["gnn", "passthrough"], default="gnn")
    p.add_argument("--checkpoint",   default="checkpoints_parquet/run_20251210_174458/best_model.pt")
    p.add_argument("--model-type",   choices=["gat", "gcn"], default="gat")
    # SMC hyperparameters
    p.add_argument("--population",   type=int,   default=100,
                   help="Number of particles (rollouts) per generation.")
    p.add_argument("--generations",  type=int,   default=5,
                   help="Number of SMC generations.")
    p.add_argument("--elite-frac",         type=float, default=0.2,
                   help="Fraction of particles retained as elite each generation.")
    p.add_argument("--sigma-pos-m",        type=float, default=200.0,
                   help="Gaussian perturbation std dev on ego ENU position (m).")
    p.add_argument("--sigma-hdg-deg",      type=float, default=math.degrees(0.1),
                   help="Gaussian perturbation std dev on ego heading (degrees).")
    p.add_argument("--max-elite-per-icao", type=int,   default=None,
                   help="Maximum elite slots per unique icao24. Prevents population "
                        "collapse onto a single scenario. Defaults to "
                        "ceil(elite_k / 5) so no icao24 can hold more than ~20%% "
                        "of the elite set.")
    # Simulation
    p.add_argument("--steps",        type=int,   default=20)
    p.add_argument("--dt",           type=float, default=DT_DEFAULT)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--deadband-deg", type=float, default=0.0)
    # Scenario filtering (mirrors run_rollouts.py)
    p.add_argument("--min-path-pts", type=int,   default=3)
    p.add_argument("--proximity-nm", type=str,   default="near")
    p.add_argument("--max-alt-ft",   type=float, default=MAX_ALT_FT_DEFAULT)
    p.add_argument("--min-dist-nm",  type=float, default=2.0)
    p.add_argument("--filter-structural", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Exclude seed scenarios where the ADS-B reference path "
                        "already fails separation (structural violations). "
                        "On by default; use --no-filter-structural to disable.")
    # Output
    p.add_argument("--save-json",    type=str,   default=None, metavar="FILE")
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
            "Pass --checkpoint with a valid path relative to workspace root."
        )
    agent = GNNAgent.from_checkpoint(
        checkpoint_path=str(ckpt),
        model_type=args.model_type,
        airport_lat=AIRPORT_LAT,
        airport_lon=AIRPORT_LON,
    )
    return agent, f"GNN ({args.model_type.upper()}) — {ckpt.name}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    agent, agent_label = _build_agent(args)
    DT = args.dt

    # Proximity filter (same logic as run_rollouts.py)
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
    proximity_nm = None if (effective_lo == 0.0 and raw_hi == float("inf")) \
        else (effective_lo, raw_hi)

    env = AircraftHeadingEnvironment(
        data_path="data/seatac_adsb_states.json",
        airport_lat=AIRPORT_LAT,
        airport_lon=AIRPORT_LON,
        proximity_nm=proximity_nm,
        max_alt_ft=args.max_alt_ft,
        seed=args.seed,
    )
    dm = AircraftHeadingDisturbanceModel(
        sigma_heading_rad=0.0,
        sigma_position_m=0.0,
        seed=args.seed,
    )
    system = AircraftHeadingPredictorSystem(
        environment=env,
        agent_model=agent,
        disturbance_model=dm,
        dt=DT,
        heading_deadband_rad=math.radians(args.deadband_deg),
    )

    prox_label = (
        f"{proximity_nm[0]:.0f}–{proximity_nm[1]:.0f} NM"
        if proximity_nm else "all ranges"
    )
    sigma_hdg_rad = math.radians(args.sigma_hdg_deg)
    total_rollouts_planned = args.population * args.generations

    # Diversity cap: default ceil(elite_k / 5) so no single icao24 holds
    # more than ~20% of the elite set, preventing population collapse.
    elite_k = max(1, int(args.population * args.elite_frac))
    max_elite_per_icao = (
        args.max_elite_per_icao
        if args.max_elite_per_icao is not None
        else max(1, math.ceil(elite_k / 5))
    )

    print(f"Agent      : {agent_label}")
    print(f"Airport    : KSEA (SeaTac)  proximity={prox_label}  max_alt={args.max_alt_ft:.0f} ft")
    print(f"Scenarios  : {env.num_scenarios} candidates")
    print(f"SMC config : {args.population} particles × {args.generations} generations "
          f"= {total_rollouts_planned} rollouts total")
    print(f"SMC params : elite_frac={args.elite_frac:.0%}  "
          f"σ_pos={args.sigma_pos_m:.0f} m  "
          f"σ_hdg={args.sigma_hdg_deg:.1f}°  "
          f"max_elite_per_icao={max_elite_per_icao}")
    print(f"Sim config : {args.steps} steps × {DT:.1f}s = {args.steps * DT:.0f}s horizon")
    if args.deadband_deg > 0.0:
        print(f"Deadband   : {args.deadband_deg:.1f}°")
    if args.save_json:
        print(f"Output JSON: {args.save_json}")
    if args.filter_structural:
        print("Filter     : structural scenarios excluded from seed population (--filter-structural)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Sample seed population
    # ------------------------------------------------------------------
    print(f"Sampling seed population ({args.population} particles)…")
    structural_skipped = 0

    def _accept(state: SystemState) -> bool:
        """Return False to reject a candidate state from the seed population."""
        nonlocal structural_skipped
        ref = env.get_reference_path(state, num_steps=args.steps, dt=DT)
        if len(ref) < args.min_path_pts:
            return False
        if args.filter_structural:
            ref_traj = _build_ref_trajectory(ref, state, DT)
            if separation_robustness(ref_traj, AIRPORT_POS) < 0:
                structural_skipped += 1
                return False
        return True

    # Manual sampling loop so we can apply the accept filter
    seed_states = []
    attempts = 0
    max_attempts = args.population * 40
    while len(seed_states) < args.population and attempts < max_attempts:
        attempts += 1
        s = env.sample_initial_state()
        if _accept(s):
            seed_states.append(s)

    if len(seed_states) < args.population:
        print(f"Warning: only {len(seed_states)}/{args.population} seed states "
              "collected — running with smaller population.")

    if structural_skipped:
        print(f"  {structural_skipped} structural scenarios excluded from seed population")
    print(f"  Seed population ready ({len(seed_states)} particles, {attempts} attempts)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Define scoring function (negated normalised robustness)
    # ------------------------------------------------------------------
    def scoring_fn(traj: Trajectory, state: SystemState) -> float:
        """Higher = worse. Positive ↔ violation."""
        return -float(safety_robustness_normalized(traj, AIRPORT_POS, PSI_DOT_MAX, DT))

    # ------------------------------------------------------------------
    # 3. Run SMC
    # ------------------------------------------------------------------
    print("Running SMC generations…")
    result = smc_failure_search(
        system=system,
        initial_states=seed_states,
        scoring_fn=scoring_fn,
        elite_frac=args.elite_frac,
        num_generations=args.generations,
        num_steps=args.steps,
        sigma_pos_m=args.sigma_pos_m,
        sigma_hdg_rad=sigma_hdg_rad,
        seed=args.seed,
        verbose=True,
        group_key_fn=lambda s: s.ego.icao24,
        max_per_group=max_elite_per_icao,
    )
    print("=" * 70)

    # ------------------------------------------------------------------
    # 4. Post-hoc evaluation of final generation
    #    (compute full robustness breakdown + reference comparison)
    # ------------------------------------------------------------------
    cos_lat = math.cos(math.radians(AIRPORT_LAT))

    def _enu_to_ll(pos):
        lat = float(pos[1]) / 111_320.0 + AIRPORT_LAT
        lon = float(pos[0]) / (111_320.0 * cos_lat) + AIRPORT_LON
        return [round(lat, 7), round(lon, 7)]

    violations = 0
    structural_violations = 0
    rollout_log = []

    print(f"\nFinal generation details ({len(result.trajectories)} trajectories):")
    print("-" * 70)

    for i, (traj, state, score) in enumerate(
        zip(result.trajectories, result.initial_states, result.scores)
    ):
        ego = state.ego
        prox_nm = float(np.linalg.norm(ego.position[:2])) / 1852.0
        alt = ego.metadata.get("altitude", float("nan")) if ego.metadata else float("nan")
        callsign = (
            (ego.metadata.get("callsign") or ego.icao24) if ego.metadata else ego.icao24
        )

        ref = env.get_reference_path(state, num_steps=args.steps, dt=DT)

        rho_sep   = separation_robustness(traj, AIRPORT_POS)
        rho_rate  = heading_rate_robustness(traj, PSI_DOT_MAX, DT)
        rho_track = cross_track_robustness(traj, ref, CROSS_TRACK_D_MAX)
        rho_norm  = -score  # score = -rho_norm by definition

        ref_traj    = _build_ref_trajectory(ref, state, DT)
        ref_rho_sep = separation_robustness(ref_traj, AIRPORT_POS)

        is_violation     = rho_norm < 0.0
        is_structural    = ref_rho_sep < 0.0

        if is_violation:
            violations += 1
            if is_structural:
                structural_violations += 1

        status = "VIOLATION" if is_violation else "SAFE"
        if is_violation and is_structural:
            verdict = "VIOLATION — structural (ref also fails)"
        elif is_violation:
            verdict = "VIOLATION — agent fault"
        else:
            verdict = "SAFE"

        print(f"\nParticle {i+1:>3}  |  {callsign} ({ego.icao24})")
        print(f"  {prox_nm:.1f} NM   {alt:.0f} ft   hdg={math.degrees(ego.heading):.0f}°   "
              f"ref={len(ref)} pts   score={score:+.4f}")
        print(f"  [{verdict}]")
        print(f"    φ_sep    : {rho_sep:>10.1f} m      {'✓' if rho_sep >= 0 else '✗'}  "
              f"  (ref: {ref_rho_sep:>8.1f} m {'✓' if ref_rho_sep >= 0 else '✗'})")
        print(f"    φ_rate   : {math.degrees(rho_rate):>10.3f} °/s  {'✓' if rho_rate >= 0 else '✗'}")
        print(f"    φ_track  : {rho_track/1852:>10.3f} NM  {'✓' if rho_track >= 0 else '✗'}  (diagnostic)")
        print(f"    ρ_norm   : {rho_norm:>10.4f}        (dimensionless)")

        if args.save_json is not None:
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
                "rollout":         i + 1,
                "callsign":        callsign,
                "icao24":          ego.icao24,
                "initial_dist_nm": round(prox_nm, 3),
                "initial_alt_ft":  round(float(alt), 0),
                "initial_hdg_deg": round(math.degrees(ego.heading) % 360, 1),
                "status":          status,
                "rho_sep":         round(float(rho_sep), 1),
                "rho_rate":        round(float(math.degrees(rho_rate)), 4),
                "rho_track":       round(float(rho_track / 1852.0), 4),
                "rho_norm":        round(float(rho_norm), 4),
                "ref_rho_sep":     round(float(ref_rho_sep), 1),
                "ego_track":       ego_track,
                "ref_track":       ref_track,
                "traffic_tracks":  traffic_tracks,
                "hdg_commands":    step_hdgs,
                "confidences":     step_confs,
            })

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    N_final = len(result.trajectories)
    print(f"\n{'=' * 70}")
    print(f"SMC summary:")
    print(f"  Total rollouts   : {result.total_rollouts}  "
          f"({args.generations} gens × {args.population} particles)")
    print(f"  Seed pop size    : {len(seed_states)}")
    print(f"  Final violations : {violations}/{N_final}  "
          f"(final-gen failure rate ≈ {violations/max(N_final, 1):.2%})")

    gen_fail_rates = [g.failure_rate for g in result.generation_stats]
    print(f"  Per-gen P_fail   : "
          + "  ".join(f"G{g.generation}={g.failure_rate:.1%}" for g in result.generation_stats))

    agent_fault = violations - structural_violations
    print(f"  Structural       : {structural_violations}/{violations} "
          "(ref path also fails — not agent-caused)")
    print(f"  Agent-fault      : {agent_fault}/{violations}")
    if args.filter_structural and structural_skipped:
        print(f"  Seed excluded    : {structural_skipped} structural scenarios "
              "removed from seed population")

    # ------------------------------------------------------------------
    # 6. Save JSON
    # ------------------------------------------------------------------
    if args.save_json is not None and rollout_log:
        out_path = Path(args.save_json)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        gen_stats_log = [
            {
                "generation":       g.generation,
                "failure_rate":     round(g.failure_rate, 4),
                "mean_score":       round(g.mean_score, 4),
                "elite_min":        round(g.elite_min_score, 4),
                "elite_max":        round(g.elite_max_score, 4),
                "num_rollouts":     g.num_rollouts,
                "elite_groups":     g.num_elite_groups,
            }
            for g in result.generation_stats
        ]

        with open(out_path, "w") as f:
            json.dump({
                "meta": {
                    "method":         "smc",
                    "agent":          agent_label,
                    "checkpoint":     args.checkpoint,
                    "population":     args.population,
                    "generations":    args.generations,
                    "elite_frac":           args.elite_frac,
                    "max_elite_per_icao":   max_elite_per_icao,
                    "sigma_pos_m":          args.sigma_pos_m,
                    "sigma_hdg_deg":        args.sigma_hdg_deg,
                    "total_rollouts": result.total_rollouts,
                    "steps":          args.steps,
                    "dt_s":           DT,
                    "deadband_deg":   args.deadband_deg,
                    "max_alt_ft":     args.max_alt_ft,
                    "airport_lat":    AIRPORT_LAT,
                    "airport_lon":    AIRPORT_LON,
                    "final_violations": violations,
                    "filter_structural": args.filter_structural,
                },
                "generation_stats": gen_stats_log,
                "rollouts": rollout_log,
            }, f, indent=2)
        print(f"\nSaved SMC trajectory log → {out_path}")


if __name__ == "__main__":
    main()
