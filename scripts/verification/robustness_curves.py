#!/usr/bin/env python3
"""Generate robustness curves: P(violation) vs heading-noise sigma.

Architecture
------------
Robustness evaluation is *independent* of how trajectories are sampled.
A ``RobustnessEvaluator`` (from ``failure_scoring``) is constructed once from
the spec parameters and environment, then passed as a plain callable to
whichever estimation method is selected:

    evaluator : (Trajectory, SystemState) → float   # positive=safe, negative=violation

Two estimation modes
--------------------
direct
    Run ROLLOUTS independent rollouts at *each* sigma value.  Slow but
    unbiased at every sigma point.

is
    Run ROLLOUTS rollouts once at a high proposal sigma, then reweight to
    every target sigma via importance sampling.  Fast: one rollout batch
    covers the full curve.  Accuracy degrades when the target sigma is much
    smaller than the proposal (low ESS).

Examples
--------
    python scripts/verification/robustness_curves.py \\
        --mode direct --rollouts 50 --sigmas 0,2,5,10,15,20

    python scripts/verification/robustness_curves.py \\
        --mode is --rollouts 200 --sigma-proposal 20 \\
        --sigmas 0,2,5,10,15,20,25,30 --out plots/robustness_curves.png
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
    AircraftHeadingEnvironment,
    AircraftHeadingDisturbanceModel,
    AircraftHeadingPredictorSystem,
    PassthroughAgentModel,
)
from verification.system.gnn_agent import GNNAgent
from verification.system.rollouts import rollout
from verification.failures_estimation.failure_scoring import RobustnessEvaluator
from verification.failures_estimation.importance_sampling import (
    trajectory_log_weight,
    is_failure_rate_sweep,
)

# ---------------------------------------------------------------------------
# Constants (same as run_rollouts.py)
# ---------------------------------------------------------------------------

AIRPORT_LAT   = 47.4498889
AIRPORT_LON   = -122.3117778
DATA_PATH     = str(ROOT / "data" / "seatac_adsb_states.json")
CHECKPOINT    = str(ROOT / "checkpoints_parquet" / "run_20251210_174458" / "best_model.pt")
AIRPORT_POS   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
PSI_DOT_MAX   = math.radians(3.0)
MAX_ALT_FT    = 10_000.0


# ---------------------------------------------------------------------------
# Build env + system (sigma only affects the disturbance model)
# ---------------------------------------------------------------------------


def _build_env() -> AircraftHeadingEnvironment:
    return AircraftHeadingEnvironment(
        data_path=DATA_PATH,
        airport_lat=AIRPORT_LAT,
        airport_lon=AIRPORT_LON,
        proximity_nm=(0, 100),
        max_alt_ft=MAX_ALT_FT,
    )


def _build_system(
    env: AircraftHeadingEnvironment,
    sigma_heading_rad: float,
    args,
) -> AircraftHeadingPredictorSystem:
    agent = GNNAgent.from_checkpoint(
        checkpoint_path=CHECKPOINT,
        model_type="gat",
        airport_lat=AIRPORT_LAT,
        airport_lon=AIRPORT_LON,
    )
    dist = AircraftHeadingDisturbanceModel(
        sigma_heading_rad=sigma_heading_rad,
        sigma_position_m=0.0,   # must be zero so IS log-weights are exact
    )
    return AircraftHeadingPredictorSystem(
        environment=env,
        agent_model=agent,
        disturbance_model=dist,
        heading_deadband_rad=math.radians(args.deadband_deg),
    )


def _make_evaluator(args) -> RobustnessEvaluator:
    """Create the robustness evaluator — independent of sigma or estimation method."""
    return RobustnessEvaluator(
        airport_pos=AIRPORT_POS,
        psi_dot_max=PSI_DOT_MAX,
        dt=args.dt,
        negate=False,   # positive = safe, negative = violation
    )


# ---------------------------------------------------------------------------
# Rollout collection — returns (trajectories, initial_states), no spec logic
# ---------------------------------------------------------------------------


def _collect_rollouts(env, system, args):
    """Collect rollouts and return (trajectories, initial_states) pairs.

    Deliberately spec-free: the evaluator is applied separately by the
    estimation method.
    """
    DT = args.dt
    trajs, states = [], []
    attempts = 0
    while len(trajs) < args.rollouts and attempts < args.rollouts * 15:
        attempts += 1
        state = env.sample_initial_state()
        traj = rollout(system, initial_state=state, num_steps=args.steps)
        trajs.append(traj)
        states.append(state)
    return trajs, states


# ---------------------------------------------------------------------------
# Direct MC — estimation method
# ---------------------------------------------------------------------------


def run_direct(evaluator: RobustnessEvaluator, args) -> list:
    """Run independent rollout batches at each sigma.

    The evaluator is passed in; this function knows nothing about *which*
    specification is being evaluated.
    """
    sigma_degs = [float(x) for x in args.sigmas.split(",")]
    results = []

    for sigma_d in sigma_degs:
        sigma_r = math.radians(sigma_d)
        env = _build_env()
        system = _build_system(env, sigma_r, args)
        trajs, states = _collect_rollouts(env, system, args)
        n = len(trajs)

        rho_scores = evaluator.scores(trajs, states)
        violations = int(np.sum(rho_scores < 0.0))
        p = violations / max(n, 1)
        se = math.sqrt(p * (1.0 - p) / max(n, 1))
        results.append((sigma_d, p, se, n))
        print(f"  σ={sigma_d:5.1f}°  P_fail={p:.4f} ± {se:.4f}  ({violations}/{n})")

    return results


# ---------------------------------------------------------------------------
# IS — estimation method
# ---------------------------------------------------------------------------


def run_is(evaluator: RobustnessEvaluator, args) -> list:
    """Collect one batch at sigma_proposal, reweight to every target sigma.

    The evaluator is called once per trajectory to compute violation labels,
    completely independently of the IS weight computation.
    """
    sigma_degs = [float(x) for x in args.sigmas.split(",")]
    sigma_targets_rad = [math.radians(x) for x in sigma_degs]
    sigma_proposal_rad = math.radians(args.sigma_proposal)

    print(
        f"Collecting {args.rollouts} rollouts at proposal "
        f"σ={args.sigma_proposal:.1f}° ..."
    )
    env = _build_env()
    system = _build_system(env, sigma_proposal_rad, args)
    trajs, states = _collect_rollouts(env, system, args)
    n = len(trajs)
    print(f"  Collected {n} rollouts.")

    # Evaluate robustness once — independent of sigma targets
    rho_scores = evaluator.scores(trajs, states)
    n_fail_q = int(np.sum(rho_scores < 0.0))
    print(f"  Failure rate under proposal: {n_fail_q/n:.2%}")

    # IS sweep: violation labels fixed, only weights vary per target sigma
    sweep = is_failure_rate_sweep(trajs, states, sigma_proposal_rad, sigma_targets_rad, evaluator)

    results = []
    for (sigma_t_rad, p_fail, ess), sigma_d in zip(sweep, sigma_degs):
        results.append((sigma_d, p_fail, ess, n))
        print(
            f"  σ={sigma_d:5.1f}°  P_fail≈{p_fail:.4f}  "
            f"ESS={ess:.0f}/{n}  ({ess/n:.0%})"
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Robustness curves: P_fail vs σ_heading",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode",           choices=["direct", "is"], default="direct",
                   help="Estimation mode: direct MC or importance sampling")
    p.add_argument("--rollouts",       type=int,   default=100,
                   help="Rollouts per sigma (direct) or total (is)")
    p.add_argument("--steps",          type=int,   default=20,
                   help="Steps per rollout")
    p.add_argument("--dt",             type=float, default=10.0,
                   help="Simulation timestep (seconds)")
    p.add_argument("--deadband-deg",   type=float, default=0.0,
                   help="Heading deadband (degrees)")
    p.add_argument("--sigmas",         type=str,   default="0,2,5,10,15,20,30",
                   help="Comma-separated σ_heading values to evaluate (degrees)")
    p.add_argument("--sigma-proposal", type=float, default=20.0,
                   help="Proposal sigma for IS mode (degrees)")
    p.add_argument("--out",            type=str,   default=None,
                   help="Output plot file (.png)")
    p.add_argument("--save-json",      type=str,   default=None,
                   help="Save results to JSON file")
    return p.parse_args()


def main():
    args = parse_args()

    sigma_degs_str = [f"{float(x):.1f}°" for x in args.sigmas.split(",")]
    print("=" * 60)
    print(f"Robustness Curves — mode={args.mode}")
    print(f"  rollouts={args.rollouts}  steps={args.steps}  dt={args.dt}s")
    print(f"  sigmas: {', '.join(sigma_degs_str)}")
    if args.mode == "is":
        print(f"  proposal σ={args.sigma_proposal:.1f}°")
    print("=" * 60)

    # Build the evaluator exactly once — completely independent of sigma /
    # estimation method.  Both run_direct and run_is receive the same object.
    evaluator = _make_evaluator(args)

    if args.mode == "direct":
        results = run_direct(evaluator, args)
        sigma_col = [r[0] for r in results]
        pfail_col = [r[1] for r in results]
        err_col   = [r[2] for r in results]
        err_label = "±SE"
    else:
        results = run_is(evaluator, args)
        sigma_col = [r[0] for r in results]
        pfail_col = [r[1] for r in results]
        err_col   = [r[2] / results[0][3] for r in results]  # ESS fraction
        err_label = "ESS%"

    print("\nRobustness Curve Summary:")
    print(f"  {'σ_hdg (°)':>10}  {'P_fail':>8}  {err_label:>8}")
    print("  " + "-" * 32)
    for sd, pf, er in zip(sigma_col, pfail_col, err_col):
        print(f"  {sd:>10.1f}  {pf:>8.4f}  {er:>8.4f}")

    if args.save_json:
        out_data = {
            "mode":          args.mode,
            "rollouts":      args.rollouts,
            "steps":         args.steps,
            "dt":            args.dt,
            "sigma_proposal": args.sigma_proposal if args.mode == "is" else None,
            "results": [
                {"sigma_deg": r[0], "p_fail": r[1],
                 ("se" if args.mode == "direct" else "ess"): r[2]}
                for r in results
            ],
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved JSON → {args.save_json}")

    if args.out:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plot.")
            return

        fig, ax = plt.subplots(figsize=(7, 4))
        label = f"{args.mode} (N={args.rollouts})"
        ax.plot(sigma_col, pfail_col, "o-", lw=2, color="steelblue", label=label)

        if args.mode == "direct":
            lo = [max(0.0, p - e) for p, e in zip(pfail_col, err_col)]
            hi = [min(1.0, p + e) for p, e in zip(pfail_col, err_col)]
            ax.fill_between(sigma_col, lo, hi, alpha=0.2, color="steelblue",
                            label="±1 SE")

        ax.axhline(0.6, ls="--", color="grey", lw=0.8, label="baseline (σ=0°)")
        ax.set_xlabel("Heading noise σ (°)")
        ax.set_ylabel("P(safety violation)")
        ax.set_title("GNN Heading Predictor — Robustness Curve (KSEA)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"Plot saved → {args.out}")


if __name__ == "__main__":
    main()
