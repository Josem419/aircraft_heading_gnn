"""Importance sampling (IS) for rare-event failure probability estimation.

Samples trajectories under a high-noise *proposal* distribution q and
reweights them to estimate the failure probability under a lower-noise
*target* distribution p.  This lets you estimate very small failure
probabilities without running thousands of rollouts at the target noise level.

Scope
-----
Only **heading-command noise** is reweighted.  Position-observation noise is
not handled (set ``sigma_position_m = 0`` in the biased system).

When position noise is zero, each :class:`~verification.system.trajectory.TrajectoryStep`
stores exactly one scalar noise contribution in ``disturbance_log_prob``
(the heading disturbance log-prob), so the reweighting formula is exact.

Math
----
Let ε_t ~ q_t = N(0 , σ_q²) be the heading noise applied at step t.
The log-likelihood stored in the step is::

    log q(ε_t) = −½(ε_t/σ_q)² − log σ_q − ½ log 2π  ← step.disturbance_log_prob

For a target distribution p_t = N(0, σ_p²), the log importance ratio per
step is::

    log(p_t/q_t) = ½ ε_t²(1/σ_q² − 1/σ_p²) + log(σ_q/σ_p)
                 = A_t · (r² − 1) + log r       where r = σ_q/σ_p

    A_t = −½(ε_t/σ_q)²  (recovered as  log q(ε_t) + log σ_q + ½ log 2π)

The total log importance weight of trajectory τ is::

    log w(τ) = Σ_t  A_t · (r² − 1) + log r

The IS failure-probability estimate is the self-normalised estimator::

    P̂_p(fail) = Σ_i w_i · 1[fail(τ_i)] / Σ_i w_i

Effective Sample Size::

    ESS = (Σ w_i)² / Σ w_i²

ESS close to N indicates low variance; ESS ≪ N signals proposal mismatch.
"""

from __future__ import annotations

import math
from typing import Any, Callable, List, Tuple

import numpy as np

from verification.system.trajectory import Trajectory
from verification.system.state import SystemState

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Per-step log weight
# ---------------------------------------------------------------------------


def step_log_weight(
    log_q: float,
    sigma_proposal: float,
    sigma_target: float,
) -> float:
    """Compute log(p(ε)/q(ε)) for a single Gaussian heading-noise step.

    Recovers the whitened noise magnitude A = −½(ε/σ_q)² from the stored
    log-probability and applies the Gaussian ratio formula.

    Args:
        log_q:          log q(ε) stored in ``step.disturbance_log_prob``.
        sigma_proposal: Proposal std dev σ_q (radians).
        sigma_target:   Target std dev σ_p (radians).  Pass 0 or a very
                        small value to approximate a deterministic nominal.

    Returns:
        Scalar log importance weight contribution for this step.
    """
    if sigma_proposal <= 0.0:
        # No disturbance was applied — p(τ) = q(τ), weight is 0
        return 0.0

    # Clamp sigma_target to avoid division by zero / degenerate weights
    sigma_target = max(sigma_target, 1e-9)

    # A = −½(ε/σ_q)²  recovered from  log q(ε)
    # log q(ε) = A − log σ_q − ½ log 2π  →  A = log q(ε) + log σ_q + ½ log 2π
    A = log_q + math.log(sigma_proposal) + _HALF_LOG_2PI   # ≤ 0

    r2 = (sigma_proposal / sigma_target) ** 2              # (σ_q/σ_p)²
    # log(p/q) = A·(r²−1) + ½ log r²  =  A·(r²−1) + log(σ_q/σ_p)
    return A * (r2 - 1.0) + 0.5 * math.log(r2)


# ---------------------------------------------------------------------------
# Trajectory-level log weight
# ---------------------------------------------------------------------------


def trajectory_log_weight(
    trajectory: Trajectory,
    sigma_proposal: float,
    sigma_target: float,
) -> float:
    """Sum log weight contributions across all steps of a trajectory.

    Args:
        trajectory:     Rollout sampled under the proposal distribution.
        sigma_proposal: Heading noise std dev used to generate the trajectory (σ_q).
        sigma_target:   Heading noise std dev of the target distribution (σ_p).

    Returns:
        Total log importance weight  log w(τ) = Σ_t log(p(ε_t)/q(ε_t)).
    """
    if sigma_proposal <= 0.0:
        return 0.0  # no disturbance → p = q → weight 1
    return sum(
        step_log_weight(step.disturbance_log_prob, sigma_proposal, sigma_target)
        for step in trajectory
    )


# ---------------------------------------------------------------------------
# IS estimators
# ---------------------------------------------------------------------------


def is_failure_rate(
    trajectories: List[Trajectory],
    initial_states: List[SystemState],
    sigma_proposal: float,
    sigma_target: float,
    robustness_fn: Callable[[Trajectory, Any], float],
) -> Tuple[float, float]:
    """Estimate P_p(fail) using samples collected under σ_q.

    The *robustness_fn* is the sole point of contact with specification
    knowledge.  It must return a scalar: positive = safe, negative = violation.
    Neither this function nor the importance weights depend on which spec is
    being evaluated.

    Args:
        trajectories:   Rollouts sampled under the proposal (σ_q).
        initial_states: Corresponding initial states (same length as
                        ``trajectories``); passed through to ``robustness_fn``
                        so it can retrieve per-scenario reference paths.
        sigma_proposal: Heading noise std dev used to generate rollouts (σ_q).
        sigma_target:   Target heading noise std dev to evaluate at (σ_p).
        robustness_fn:  ``(Trajectory, SystemState) → float``.  Positive =
                        safe; negative = violation.  Typically a
                        :class:`~verification.failures_estimation.failure_scoring.RobustnessEvaluator`
                        instance.

    Returns:
        ``(p_fail_estimate, ess)`` where ESS is the effective sample size in
        the range [1, N].  Low ESS indicates large variance.
    """
    if len(trajectories) != len(initial_states):
        raise ValueError(
            "trajectories and initial_states must have the same length; "
            f"got {len(trajectories)} and {len(initial_states)}"
        )
    log_weights = np.array(
        [trajectory_log_weight(t, sigma_proposal, sigma_target) for t in trajectories],
        dtype=np.float64,
    )
    # Numerically stable normalisation (subtract max before exp)
    log_w_max = float(log_weights.max())
    w = np.exp(log_weights - log_w_max)
    w_sum = w.sum()
    if w_sum == 0.0:
        return 0.0, 0.0
    w_norm = w / w_sum

    failure_mask = np.array(
        [0.0 if robustness_fn(t, s) >= 0.0 else 1.0
         for t, s in zip(trajectories, initial_states)],
        dtype=np.float64,
    )
    p_fail = float(np.dot(w_norm, failure_mask))
    ess = float(w_sum ** 2 / np.sum(w ** 2))
    return p_fail, ess


def is_failure_rate_sweep(
    trajectories: List[Trajectory],
    initial_states: List[SystemState],
    sigma_proposal: float,
    sigma_targets: List[float],
    robustness_fn: Callable[[Trajectory, Any], float],
) -> List[Tuple[float, float, float]]:
    """Estimate P_fail at multiple target sigma values from one set of rollouts.

    This is the key advantage of IS: a *single* biased sample set can be
    reweighted to estimate failure rates at many different noise levels without
    additional rollouts.  The *robustness_fn* is evaluated once per trajectory
    (not once per sigma) since the violation label is sigma-independent.

    Args:
        trajectories:   Rollouts sampled under sigma_proposal.
        initial_states: Corresponding initial states.
        sigma_proposal: Heading noise std dev used to generate rollouts (σ_q).
        sigma_targets:  List of target sigma values (σ_p) to evaluate.
        robustness_fn:  ``(Trajectory, SystemState) → float``.

    Returns:
        List of ``(sigma_target, p_fail_estimate, ess)`` tuples.
    """
    # Pre-compute violation labels once — they don't depend on sigma
    failure_mask = np.array(
        [0.0 if robustness_fn(t, s) >= 0.0 else 1.0
         for t, s in zip(trajectories, initial_states)],
        dtype=np.float64,
    )

    results = []
    for sigma_t in sigma_targets:
        log_weights = np.array(
            [trajectory_log_weight(t, sigma_proposal, sigma_t) for t in trajectories],
            dtype=np.float64,
        )
        log_w_max = float(log_weights.max())
        w = np.exp(log_weights - log_w_max)
        w_sum = w.sum()
        if w_sum == 0.0:
            results.append((sigma_t, 0.0, 0.0))
            continue
        w_norm = w / w_sum
        p_fail = float(np.dot(w_norm, failure_mask))
        ess = float(w_sum ** 2 / np.sum(w ** 2))
        results.append((sigma_t, p_fail, ess))
    return results


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def effective_sample_size(
    trajectories: List[Trajectory],
    sigma_proposal: float,
    sigma_target: float,
) -> float:
    """Return the effective sample size for a given (σ_q, σ_p) pair.

    Values close to N indicate efficient reweighting; values near 1 mean the
    proposal and target distributions are almost disjoint — increase σ_q or
    collect more rollouts.

    Args:
        trajectories:   Rollouts sampled under sigma_proposal.
        sigma_proposal: σ_q.
        sigma_target:   σ_p (the distribution we're targeting).

    Returns:
        ESS value in [1, N].
    """
    log_weights = np.array(
        [trajectory_log_weight(t, sigma_proposal, sigma_target) for t in trajectories],
        dtype=np.float64,
    )
    log_w_max = float(log_weights.max())
    w = np.exp(log_weights - log_w_max)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def weight_diagnostics(
    trajectories: List[Trajectory],
    sigma_proposal: float,
    sigma_target: float,
) -> dict:
    """Return a diagnostic summary of the importance weights.

    Useful for checking whether the proposal is well-matched to the target.

    Returns:
        Dictionary with keys: ``n``, ``ess``, ``log_w_min``, ``log_w_max``,
        ``log_w_mean``, ``log_w_std``, ``ess_fraction``.
    """
    log_weights = np.array(
        [trajectory_log_weight(t, sigma_proposal, sigma_target) for t in trajectories],
        dtype=np.float64,
    )
    log_w_max = float(log_weights.max())
    w = np.exp(log_weights - log_w_max)
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    n = len(trajectories)
    return {
        "n":            n,
        "ess":          ess,
        "ess_fraction": ess / n,
        "log_w_min":    float(log_weights.min()),
        "log_w_max":    float(log_weights.max()),
        "log_w_mean":   float(log_weights.mean()),
        "log_w_std":    float(log_weights.std()),
    }
