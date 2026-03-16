"""Failure scoring utilities for the aircraft heading verification system.

Functions here operate over *collections* of trajectories rather than a single
state, and are intended for use after a batch of rollouts has been collected.

Key architectural separation
-----------------------------
``RobustnessEvaluator`` is the central building block: a standalone callable
``(Trajectory, SystemState) → float`` that measures how far a rollout is from
violating the safety specification, *independent of how trajectories were
sampled*.  Positive values mean safe-with-margin; negative values mean
violation.

All failure estimation methods (Monte Carlo, importance sampling, SMC) accept a
``RobustnessEvaluator`` (or any compatible callable) as their scoring objective.
They are responsible for *sampling*, not for knowing which specification to
evaluate.
"""

import math
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from verification.system.trajectory import Trajectory
from verification.system.state import SystemState
from verification.failures_estimation.trajectory_likelihood import (
    TrajectoryDensityModel,
    StepFeatureExtractor,
    default_step_features,
    GaussianTrajectoryModel,
    NominalTrajectoryModel,
    TransitionModel,
)


# ---------------------------------------------------------------------------
# RobustnessEvaluator — standalone (trajectory, state) → float
# ---------------------------------------------------------------------------


class RobustnessEvaluator:
    """Standalone robustness oracle: ``(Trajectory, SystemState) → float``.

    Wraps the separation + heading-rate safety specification so that failure
    estimation methods (MC, IS, SMC) can treat it as a pure black-box scoring
    function without knowing which environment or spec parameters are in use.

    Cross-track error is excluded from the combined spec: the GNN is an
    advisory predictor for approach/terminal traffic, not a path-following
    controller.  Separation and heading rate are the physically meaningful
    safety constraints.

    Args:
        airport_pos:   (3,) ENU airport position array (usually all-zeros).
        psi_dot_max:   Maximum heading rate (rad/s).
        dt:            Simulation timestep (seconds).
        negate:        When ``True``, return ``-robustness`` so that positive
                       values indicate violations (SMC convention).
    """

    def __init__(
        self,
        airport_pos: np.ndarray,
        psi_dot_max: float,
        dt: float,
        negate: bool = False,
        normalize: bool = True,
    ) -> None:
        self.airport_pos = airport_pos
        self.psi_dot_max = psi_dot_max
        self.dt          = dt
        self._sign       = -1.0 if negate else 1.0
        self._normalize  = normalize

    def __call__(self, trajectory: Trajectory, initial_state: SystemState) -> float:
        """Return the safety robustness of *trajectory*.

        ``initial_state`` is accepted for interface compatibility with IS and
        SMC (which pass it through), but is not needed for sep/rate evaluation.

        Returns:
            Scalar robustness ρ, positive = safe, negative = violation
            (sign flipped when ``negate=True``).  Dimensionless when
            ``normalize=True`` (default); raw mixed units otherwise.
        """
        if self._normalize:
            from verification.specifications.aircraft_heading_spec import safety_robustness_normalized
            rho = safety_robustness_normalized(
                trajectory, self.airport_pos, self.psi_dot_max, self.dt,
            )
        else:
            from verification.specifications.aircraft_heading_spec import safety_robustness
            rho = safety_robustness(
                trajectory, self.airport_pos, self.psi_dot_max, self.dt,
            )
        return self._sign * float(rho)

    def is_violation(self, trajectory: Trajectory, initial_state: SystemState) -> bool:
        """Return ``True`` when the trajectory violates the specification."""
        rho = self(trajectory, initial_state)
        # If negate=True, sign is flipped so violation is still rho < 0 in
        # the original space, which means the called value > 0 here.
        return rho < 0.0 if not (self._sign < 0) else rho > 0.0

    def violation_mask(
        self,
        trajectories: List[Trajectory],
        initial_states: List[SystemState],
    ) -> np.ndarray:
        """Return a boolean array ``(N,)`` where ``True`` = violation."""
        return np.array(
            [self.is_violation(t, s) for t, s in zip(trajectories, initial_states)],
            dtype=bool,
        )

    def scores(
        self,
        trajectories: List[Trajectory],
        initial_states: List[SystemState],
    ) -> np.ndarray:
        """Return robustness scores ``(N,)`` for a batch of rollouts."""
        return np.array(
            [self(t, s) for t, s in zip(trajectories, initial_states)],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Low-level robustness helpers (trajectory-only, spec passed explicitly)
# ---------------------------------------------------------------------------


def robustness(trajectory: Trajectory, specification) -> bool:
    """Evaluate a Boolean safety specification over a trajectory.

    Args:
        trajectory:    Rollout to evaluate.
        specification: Any callable ``(Trajectory) -> bool``, e.g. a function
                       from ``aircraft_heading_spec`` or a composed
                       ``Specification`` object from ``spec_framework``.

    Returns:
        ``True`` if the trajectory satisfies the specification.
    """
    return bool(specification(trajectory))


def signed_robustness(
    trajectory: Trajectory,
    signed_spec: Callable[[Trajectory], float],
) -> float:
    """Return a signed (quantitative) robustness measure.

    Positive values indicate the specification is satisfied with margin;
    negative values indicate a violation.

    Args:
        trajectory:   Rollout to evaluate.
        signed_spec:  A callable that returns a real-valued robustness score,
                      e.g. the minimum safety margin across all timesteps.

    Returns:
        Scalar robustness value.
    """
    return float(signed_spec(trajectory))


# ---------------------------------------------------------------------------
# Failure rate estimation from a batch of rollouts
# ---------------------------------------------------------------------------


def monte_carlo_failure_rate(
    trajectories: List[Trajectory],
    specification,
) -> Tuple[float, float]:
    """Estimate the failure probability via crude Monte Carlo.

    Args:
        trajectories:  Collection of rollout trajectories sampled i.i.d.
                       from the nominal distribution.
        specification: Boolean specification callable.

    Returns:
        Tuple of (failure_rate_estimate, standard_error).
    """
    n = len(trajectories)
    if n == 0:
        raise ValueError("Need at least one trajectory.")
    failures = sum(1 for t in trajectories if not robustness(t, specification))
    p_hat = failures / n
    se = math.sqrt(p_hat * (1.0 - p_hat) / n)
    return p_hat, se


def importance_weighted_failure_rate(
    trajectories: List[Trajectory],
    log_weights: List[float],
    specification,
) -> float:
    """Estimate the failure probability using importance sampling.

    Each trajectory is weighted by its importance weight w_i = p(τ_i) / q(τ_i),
    supplied in log-space.  The estimator is

        P̂(failure) = Σ_i w_i · 1[τ_i fails] / Σ_i w_i

    Args:
        trajectories: Rollout trajectories sampled from proposal distribution q.
        log_weights:  Log importance weights log(p / q) for each trajectory.
        specification: Boolean specification callable.

    Returns:
        Importance-weighted failure probability estimate.
    """
    if len(trajectories) != len(log_weights):
        raise ValueError("trajectories and log_weights must have the same length.")

    log_w = np.array(log_weights, dtype=np.float64)
    # Normalise in log-space for numerical stability
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum()

    failure_mask = np.array(
        [0.0 if robustness(t, specification) else 1.0 for t in trajectories]
    )
    return float(np.dot(w, failure_mask))


# ---------------------------------------------------------------------------
# Trajectory log-likelihood relative to a corpus or nominal
# ---------------------------------------------------------------------------


def corpus_log_likelihood(
    trajectory: Trajectory,
    reference_trajectories: List[Trajectory],
    model_type: str = "gaussian",
    extractor: StepFeatureExtractor = default_step_features,
) -> Tuple[float, List[float]]:
    """Score a trajectory against an empirical distribution estimated from a corpus.

    Fits a density model to *reference_trajectories* and evaluates the
    log-likelihood of *trajectory* under that model.

    Args:
        trajectory:             Rollout to evaluate.
        reference_trajectories: Reference corpus (same rollout horizon).
        model_type:             ``"gaussian"`` (per-step independent Gaussians)
                                or ``"transition"`` (autoregressive transition).
        extractor:              Feature extraction function.

    Returns:
        Tuple of (total_log_likelihood, per_step_log_likelihoods).
    """
    if model_type == "gaussian":
        model: TrajectoryDensityModel = GaussianTrajectoryModel.fit(
            reference_trajectories, extractor
        )
    elif model_type == "transition":
        model = TransitionModel.fit(reference_trajectories, extractor)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}.")

    per_step = model.step_log_likelihoods(trajectory)
    return sum(per_step), per_step


def nominal_log_likelihood(
    trajectory: Trajectory,
    nominal: Trajectory,
    sigma: float | np.ndarray = 1.0,
    extractor: StepFeatureExtractor = default_step_features,
) -> Tuple[float, np.ndarray]:
    """Score a trajectory relative to a nominal (ground-truth) reference.

    Args:
        trajectory: Rollout to evaluate.
        nominal:    Ground-truth / reference trajectory.
        sigma:      Per-feature (or global) standard deviation.
        extractor:  Feature extraction function.

    Returns:
        Tuple of (total_log_likelihood, deviation_profile) where
        ``deviation_profile`` is a (T,) array of per-step Mahalanobis
        distances.
    """
    model = NominalTrajectoryModel(nominal=nominal, sigma=sigma, extractor=extractor)
    return model.log_likelihood(trajectory), model.deviation_profile(trajectory)


# ---------------------------------------------------------------------------
# Batch scoring helpers
# ---------------------------------------------------------------------------


def score_batch(
    trajectories: List[Trajectory],
    model: TrajectoryDensityModel,
) -> np.ndarray:
    """Score every trajectory in a batch and return log-likelihoods.

    Args:
        trajectories: List of rollout trajectories.
        model:        A fitted ``TrajectoryDensityModel``.

    Returns:
        Array of shape (N,) with log p(τ_i) for each trajectory.
    """
    return np.array([model.log_likelihood(t) for t in trajectories])


def rank_by_likelihood(
    trajectories: List[Trajectory],
    model: TrajectoryDensityModel,
    ascending: bool = True,
) -> List[Tuple[int, float, Trajectory]]:
    """Sort trajectories by log-likelihood under a model.

    Args:
        trajectories: List of rollout trajectories.
        model:        A fitted ``TrajectoryDensityModel``.
        ascending:    If ``True``, most anomalous (lowest log-likelihood)
                      trajectories appear first.

    Returns:
        List of (original_index, log_likelihood, trajectory) tuples.
    """
    scored = [
        (i, model.log_likelihood(t), t)
        for i, t in enumerate(trajectories)
    ]
    return sorted(scored, key=lambda x: x[1], reverse=not ascending)
