"""Sequential Monte Carlo (SMC) failure search for the aircraft heading verification system.

Implements a Cross-Entropy / Adaptive Multi-Level Splitting style search that
concentrates a population of initial states toward specification violations.

Algorithm
---------
At each generation:

1. Roll out all N particles from their (possibly perturbed) initial states.
2. Score each trajectory: score = −safety_robustness (positive = violation).
3. Select the elite fraction (highest score = largest violation).
4. Resample N new particles from the elite set with Gaussian perturbation on
   ego position and heading.
5. Repeat for the configured number of generations.

The population converges toward failure-concentrating initial conditions.
The output is the full final-generation population and per-generation stats.

Signature note
--------------
The *scoring_fn* receives ``(trajectory, initial_state)`` so callers can
compute per-scenario reference paths and spec checks inside:

    def score(traj, s0):
        ref = env.get_reference_path(s0, num_steps=STEPS, dt=DT)
        return -safety_robustness(traj, ref, ...)

Positive return value = violation; negative = safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from verification.system.state import AircraftState, SystemState
from verification.system.trajectory import Trajectory
from verification.system.rollouts import rollout
from verification.system.system import System


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class GenerationStats:
    """Per-generation diagnostics from one SMC iteration."""
    generation:         int
    failure_rate:       float   # fraction with score > 0
    mean_score:         float
    elite_min_score:    float
    elite_max_score:    float
    num_rollouts:       int
    num_elite_groups:   int = 0  # unique group keys in elite set (0 = no grouping)


@dataclass
class SMCResult:
    """Results from an SMC failure-search run.

    Attributes:
        trajectories:    All trajectories from the *final* generation in
                         order matching ``initial_states``.
        initial_states:  Final-generation initial states (perturbed particles).
        scores:          Score (−robustness) for each final trajectory.
        generation_stats: Per-generation diagnostics list.
        total_rollouts:  Total rollouts executed across all generations.
    """
    trajectories:    List[Trajectory]
    initial_states:  List[SystemState]
    scores:          np.ndarray
    generation_stats: List[GenerationStats]
    total_rollouts:  int

    @property
    def failure_rate(self) -> float:
        """Failure rate of the *final* generation."""
        return float(np.mean(self.scores > 0.0))

    @property
    def failure_trajectories(self) -> List[Trajectory]:
        """All final-generation trajectories that received a positive score."""
        return [t for t, s in zip(self.trajectories, self.scores) if s > 0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _perturb_state(
    state: SystemState,
    sigma_pos_m: float,
    sigma_hdg_rad: float,
    rng: np.random.Generator,
) -> SystemState:
    """Add Gaussian perturbation to ego position (x, y) and heading.

    Altitude (position[2]) is not perturbed — it is kept faithful to the
    original ADS-B measurement so altitude-gated specs remain consistent.
    """
    new_state = state.copy()
    if sigma_pos_m > 0.0:
        noise_pos = np.zeros(3, dtype=np.float32)
        noise_pos[:2] = rng.normal(0.0, sigma_pos_m, 2).astype(np.float32)
        new_pos = new_state.ego.position + noise_pos
    else:
        new_pos = new_state.ego.position.copy()

    new_hdg = new_state.ego.heading
    if sigma_hdg_rad > 0.0:
        new_hdg = new_hdg + float(rng.normal(0.0, sigma_hdg_rad))

    meta = dict(new_state.ego.metadata) if new_state.ego.metadata else {}
    new_state.ego = AircraftState(
        position=new_pos,
        velocity=new_state.ego.velocity.copy(),
        heading=new_hdg,
        icao24=new_state.ego.icao24,
        metadata=meta,
    )
    return new_state


# ---------------------------------------------------------------------------
# Main SMC runner
# ---------------------------------------------------------------------------


def smc_failure_search(
    system: System,
    initial_states: List[SystemState],
    scoring_fn: Callable[[Trajectory, SystemState], float],
    elite_frac: float = 0.2,
    num_generations: int = 5,
    num_steps: int = 20,
    sigma_pos_m: float = 200.0,
    sigma_hdg_rad: float = 0.1,
    seed: Optional[int] = None,
    verbose: bool = True,
    group_key_fn: Optional[Callable[[SystemState], Any]] = None,
    max_per_group: Optional[int] = None,
) -> SMCResult:
    """Run SMC failure search over a population of initial states.

    Args:
        system:          Simulation system (env + agent + disturbance).
        initial_states:  Seed population of N initial states from
                         ``env.sample_initial_state()``.
        scoring_fn:      ``(trajectory, initial_state) → float``.  Higher
                         values are worse; positive = violation.
        elite_frac:      Fraction of particles to keep as elite each
                         generation (default: 20 %).
        num_generations: Number of SMC generations to run.
        num_steps:       Rollout length in timesteps.
        sigma_pos_m:     Gaussian perturbation std dev on ego position (m).
        sigma_hdg_rad:   Gaussian perturbation std dev on ego heading (rad).
        seed:            Optional RNG seed for reproducibility.
        verbose:         Print per-generation summary to stdout.
        group_key_fn:    Optional callable ``(SystemState) → hashable`` that
                         assigns each particle to a diversity group (e.g.,
                         ``lambda s: s.ego.icao24``).  When provided together
                         with ``max_per_group``, at most ``max_per_group``
                         particles sharing the same key are admitted to the
                         elite set.  This prevents population collapse onto a
                         single scenario.
        max_per_group:   Maximum number of elite particles allowed per group.
                         Has no effect when ``group_key_fn`` is ``None``.

    Returns:
        :class:`SMCResult` with the final-generation population, per-step
        scores, and generation diagnostics.
    """
    rng = np.random.default_rng(seed)
    N = len(initial_states)
    elite_k = max(1, int(N * elite_frac))

    particles: List[SystemState] = list(initial_states)
    generation_stats: List[GenerationStats] = []
    total_rollouts = 0

    # These are set in the final generation
    final_trajectories: List[Trajectory] = []
    final_scores = np.zeros(N)

    for gen in range(num_generations):
        trajectories: List[Trajectory] = []
        scores = np.zeros(N)

        for i, state in enumerate(particles):
            traj = rollout(system, initial_state=state, num_steps=num_steps)
            score = scoring_fn(traj, state)
            trajectories.append(traj)
            scores[i] = score

        total_rollouts += N

        failure_rate = float(np.mean(scores > 0.0))

        # --- Diversity-aware elite selection ---
        if group_key_fn is not None and max_per_group is not None:
            sorted_desc = np.argsort(scores)[::-1]  # best first
            group_counts: Dict[Any, int] = {}
            elite_list: List[int] = []
            for idx in sorted_desc:
                key = group_key_fn(particles[idx])
                if group_counts.get(key, 0) < max_per_group:
                    elite_list.append(int(idx))
                    group_counts[key] = group_counts.get(key, 0) + 1
                if len(elite_list) == elite_k:
                    break
            elite_idx = np.array(elite_list, dtype=int)
        else:
            elite_idx = np.argsort(scores)[-elite_k:]

        elite_scores = scores[elite_idx]

        stats = GenerationStats(
            generation=gen + 1,
            failure_rate=failure_rate,
            mean_score=float(scores.mean()),
            elite_min_score=float(elite_scores.min()),
            elite_max_score=float(elite_scores.max()),
            num_rollouts=N,
            num_elite_groups=len(set(group_key_fn(particles[int(i)]) for i in elite_idx))
                if group_key_fn is not None else 0,
        )
        generation_stats.append(stats)

        if verbose:
            diversity_str = (
                f"  elite_groups={stats.num_elite_groups}"
                if group_key_fn is not None else ""
            )
            print(
                f"  Gen {gen+1}/{num_generations}: "
                f"P_fail={failure_rate:.2%}  "
                f"mean={scores.mean():.4f}  "
                f"elite=[{elite_scores.min():.4f}, {elite_scores.max():.4f}]"
                f"{diversity_str}"
            )

        final_trajectories = trajectories
        final_scores = scores

        if gen == num_generations - 1:
            break  # no resampling after last generation

        # --- Resample with Gaussian kernel perturbation from elite ---
        new_particles: List[SystemState] = []
        for _ in range(N):
            parent_idx = int(elite_idx[rng.integers(0, elite_k)])
            parent = particles[parent_idx]
            new_particles.append(
                _perturb_state(parent, sigma_pos_m, sigma_hdg_rad, rng)
            )
        particles = new_particles

    return SMCResult(
        trajectories=final_trajectories,
        initial_states=particles,
        scores=final_scores,
        generation_stats=generation_stats,
        total_rollouts=total_rollouts,
    )


# ---------------------------------------------------------------------------
# Convenience: sample seeds from environment
# ---------------------------------------------------------------------------


def sample_seed_population(
    env,
    n: int,
    min_path_pts: int = 2,
    max_attempts: int = None,
    seed: Optional[int] = None,
) -> List[SystemState]:
    """Draw N valid initial states from an environment.

    Args:
        env:           An environment with a ``sample_initial_state()`` method
                       and optional ``get_reference_path()`` for path check.
        n:             Number of particles to return.
        min_path_pts:  Minimum reference-path length; shorter scenarios are
                       skipped.  Set to 1 to disable.
        max_attempts:  Cap on total sampling attempts (default: n × 20).
        seed:          RNG seed passed to the environment if it accepts one
                       (informational; ignored if env has its own RNG).

    Returns:
        List of N ``SystemState`` objects.
    """
    if max_attempts is None:
        max_attempts = n * 20
    states: List[SystemState] = []
    attempts = 0
    while len(states) < n and attempts < max_attempts:
        attempts += 1
        state = env.sample_initial_state()
        if min_path_pts > 1 and hasattr(env, "get_reference_path"):
            ref = env.get_reference_path(state, num_steps=20, dt=10.0)
            if len(ref) < min_path_pts:
                continue
        states.append(state)
    if len(states) < n:
        raise RuntimeError(
            f"Could only collect {len(states)}/{n} valid initial states "
            f"after {max_attempts} attempts. Relax min_path_pts or increase "
            "the scenario pool."
        )
    return states
