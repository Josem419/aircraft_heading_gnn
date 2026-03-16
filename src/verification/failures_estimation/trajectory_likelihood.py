"""Trajectory likelihood scoring for the aircraft heading verification system.

The core idea: treat a trajectory as a sequence of state-action feature
vectors x_0, x_1, …, x_{T-1} and assign it a log-likelihood under a density
model p.  Three density models are provided:

``GaussianTrajectoryModel``
    Fits *independent per-step* multivariate Gaussians from a collection of
    reference trajectories.  At each timestep t the model stores (μ_t, Σ_t)
    estimated across all reference trajectories at that step.  Log-likelihood
    of a new trajectory is Σ_t log N(x_t; μ_t, Σ_t).

    Requires aligned trajectories of equal length (typical for fixed-horizon
    rollouts).

``NominalTrajectoryModel``
    Compares a trajectory step-by-step to a single *nominal* (ground-truth)
    trajectory.  Each step contributes an isotropic Gaussian log-likelihood:
    -½ ‖(x_t − x_t^nom) / σ‖² − const.  Useful for measuring how far a
    rollout deviates from a reference approach path.

``TransitionModel``
    Fits the *transition distribution* p(x_t | x_{t-1}) as a conditional
    Gaussian from (previous, next) state pairs across reference trajectories.
    Log-likelihood of a new trajectory is
    Σ_{t=1}^{T-1} log p(x_t | x_{t-1}).

    Captures dynamics rather than per-step marginals.

All three models expose the same interface so they can be used
interchangeably in the failure estimation pipeline.

Feature extraction
------------------
By default each ``TrajectoryStep`` is mapped to a 7-dimensional feature
vector::

    [pos_x_m, pos_y_m, heading_rad, speed_mps, altitude_ft,
     heading_command_rad, traffic_count]

You can supply a custom ``StepFeatureExtractor`` callable to any model.
"""

from __future__ import annotations

import abc
import math
from typing import Callable, List, Tuple

import numpy as np

from verification.system.trajectory import Trajectory, TrajectoryStep

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

StepFeatureExtractor = Callable[[TrajectoryStep], np.ndarray]

_DEFAULT_FEATURE_NAMES = [
    "pos_x_m",
    "pos_y_m",
    "heading_rad",
    "speed_mps",
    "altitude_ft",
    "heading_command_rad",
    "traffic_count",
]


def default_step_features(step: TrajectoryStep) -> np.ndarray:
    """Extract the default 7-dimensional feature vector from a trajectory step.

    Features
    --------
    pos_x_m, pos_y_m : float
        Ego aircraft position in local ENU metres.
    heading_rad : float
        Ego heading in radians.
    speed_mps : float
        Ego ground speed in m/s.
    altitude_ft : float
        Ego geometric altitude in feet (from metadata; 0 if absent).
    heading_command_rad : float
        Commanded heading from the agent action.
    traffic_count : int
        Number of other aircraft in the snapshot.
    """
    ego = step.state.ego
    alt = ego.metadata.get("altitude", 0.0) if ego.metadata else 0.0
    return np.array([
        ego.position[0],
        ego.position[1],
        ego.heading,
        float(ego.speed),
        float(alt),
        step.action.heading_command,
        float(len(step.state.traffic)),
    ], dtype=np.float64)


def trajectory_to_matrix(
    trajectory: Trajectory,
    extractor: StepFeatureExtractor = default_step_features,
) -> np.ndarray:
    """Convert a trajectory to an (T, d) feature matrix.

    Args:
        trajectory: Rollout trajectory of length T.
        extractor:  Feature extraction function (default: ``default_step_features``).

    Returns:
        Array of shape (T, d).
    """
    return np.stack([extractor(step) for step in trajectory], axis=0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TrajectoryDensityModel(abc.ABC):
    """Abstract density model over trajectories.

    All models expose:

    * :meth:`log_likelihood` — scalar log p(trajectory) under the model.
    * :meth:`step_log_likelihoods` — per-step contributions as a list.
    """

    @abc.abstractmethod
    def log_likelihood(self, trajectory: Trajectory) -> float:
        """Return log p(trajectory) under this model.

        Args:
            trajectory: Evaluated trajectory.

        Returns:
            Scalar log-likelihood (higher = more typical under the model).
        """

    def step_log_likelihoods(self, trajectory: Trajectory) -> List[float]:
        """Return per-step log-likelihood contributions.

        The sum of the returned values equals :meth:`log_likelihood`.

        Args:
            trajectory: Evaluated trajectory.

        Returns:
            List of T floats.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers shared across models
# ---------------------------------------------------------------------------


def _mahal_log_prob(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log probability of *x* under N(mean, cov) (multivariate Gaussian).

    Uses a regularised pseudo-inverse to handle near-singular covariances.
    """
    d = len(mean)
    diff = x - mean
    # Regularise covariance
    cov_reg = cov + 1e-6 * np.eye(d)
    try:
        sign, log_det = np.linalg.slogdet(cov_reg)
        if sign <= 0:
            raise np.linalg.LinAlgError("Non-positive definite")
        cov_inv = np.linalg.inv(cov_reg)
        mahal = float(diff @ cov_inv @ diff)
        return -0.5 * (mahal + log_det + d * math.log(2.0 * math.pi))
    except np.linalg.LinAlgError:
        # Fall back to isotropic approximation using trace/d as variance
        var = max(float(np.trace(cov)) / d, 1e-8)
        mahal = float(np.dot(diff, diff)) / var
        return -0.5 * (mahal + d * math.log(var) + d * math.log(2.0 * math.pi))


def _isotropic_log_prob(x: np.ndarray, mean: np.ndarray, sigma: float) -> float:
    """Log probability under an isotropic Gaussian N(mean, sigma² I)."""
    d = len(mean)
    sq_dist = float(np.dot(x - mean, x - mean))
    return -0.5 * (sq_dist / (sigma ** 2) + d * math.log(2.0 * math.pi * sigma ** 2))


# ---------------------------------------------------------------------------
# Model 1: Per-step Gaussian (fit from reference corpus)
# ---------------------------------------------------------------------------


class GaussianTrajectoryModel(TrajectoryDensityModel):
    """Per-step multivariate Gaussian model fit from reference trajectories.

    At each timestep *t*, the model estimates the mean μ_t and covariance Σ_t
    of the feature vector across all reference trajectories.  Log-likelihood
    of a new trajectory is

        log p(τ) = Σ_t log N(x_t ; μ_t, Σ_t)

    Parameters
    ----------
    means : list of (d,) arrays, length T
    covariances : list of (d, d) arrays, length T
    extractor : feature extraction function
    """

    def __init__(
        self,
        means: List[np.ndarray],
        covariances: List[np.ndarray],
        extractor: StepFeatureExtractor = default_step_features,
    ) -> None:
        self.means = means
        self.covariances = covariances
        self.extractor = extractor
        self.num_steps = len(means)

    @classmethod
    def fit(
        cls,
        trajectories: List[Trajectory],
        extractor: StepFeatureExtractor = default_step_features,
    ) -> "GaussianTrajectoryModel":
        """Fit per-step Gaussians from a collection of reference trajectories.

        All trajectories must have the same length.

        Args:
            trajectories: Reference rollouts (must be equal-length).
            extractor:    Feature extraction function.

        Returns:
            Fitted ``GaussianTrajectoryModel``.
        """
        if not trajectories:
            raise ValueError("Need at least one reference trajectory.")

        T = len(trajectories[0])
        if any(len(t) != T for t in trajectories):
            raise ValueError(
                "All reference trajectories must have the same length. "
                f"Expected {T}; got lengths {[len(t) for t in trajectories]}."
            )

        # Collect per-step feature matrices: each entry is (N, d)
        step_matrices: List[List[np.ndarray]] = [[] for _ in range(T)]
        for traj in trajectories:
            mat = trajectory_to_matrix(traj, extractor)  # (T, d)
            for t in range(T):
                step_matrices[t].append(mat[t])

        means = []
        covariances = []
        for t in range(T):
            X = np.stack(step_matrices[t], axis=0)  # (N, d)
            means.append(X.mean(axis=0))
            if len(X) < 2:
                covariances.append(np.eye(X.shape[1]))
            else:
                covariances.append(np.cov(X, rowvar=False))

        return cls(means=means, covariances=covariances, extractor=extractor)

    def step_log_likelihoods(self, trajectory: Trajectory) -> List[float]:
        mat = trajectory_to_matrix(trajectory, self.extractor)
        T = min(len(trajectory), self.num_steps)
        return [
            _mahal_log_prob(mat[t], self.means[t], self.covariances[t])
            for t in range(T)
        ]

    def log_likelihood(self, trajectory: Trajectory) -> float:
        return sum(self.step_log_likelihoods(trajectory))


# ---------------------------------------------------------------------------
# Model 2: Nominal trajectory comparison
# ---------------------------------------------------------------------------


class NominalTrajectoryModel(TrajectoryDensityModel):
    """Step-by-step isotropic Gaussian centred on a nominal (truth) trajectory.

    Each step contributes

        log p(x_t | x_t^nom) = log N(x_t ; x_t^nom, diag(σ²))

    where *sigma* can differ per feature dimension, allowing you to weight
    position deviation more (or less) than heading deviation.

    Parameters
    ----------
    nominal : Trajectory
        The reference / ground-truth trajectory.
    sigma : float or array-like of shape (d,)
        Standard deviation(s).  A scalar applies the same value to all
        features; a vector applies per-feature standard deviations.
    extractor : feature extraction function
    """

    def __init__(
        self,
        nominal: Trajectory,
        sigma: float | np.ndarray = 1.0,
        extractor: StepFeatureExtractor = default_step_features,
    ) -> None:
        self.nominal_matrix = trajectory_to_matrix(nominal, extractor)  # (T, d)
        self.extractor = extractor

        d = self.nominal_matrix.shape[1]
        if np.isscalar(sigma):
            self._sigma = np.full(d, float(sigma))
        else:
            self._sigma = np.asarray(sigma, dtype=np.float64)
            if self._sigma.shape != (d,):
                raise ValueError(
                    f"sigma must be a scalar or array of length {d}; got {self._sigma.shape}"
                )

    def step_log_likelihoods(self, trajectory: Trajectory) -> List[float]:
        mat = trajectory_to_matrix(trajectory, self.extractor)
        T = min(len(trajectory), len(self.nominal_matrix))
        results = []
        for t in range(T):
            diff = mat[t] - self.nominal_matrix[t]
            sq_dist = float(np.sum((diff / self._sigma) ** 2))
            lp = -0.5 * (sq_dist + np.sum(np.log(2.0 * math.pi * self._sigma ** 2)))
            results.append(float(lp))
        return results

    def log_likelihood(self, trajectory: Trajectory) -> float:
        return sum(self.step_log_likelihoods(trajectory))

    def deviation_profile(self, trajectory: Trajectory) -> np.ndarray:
        """Return per-step Mahalanobis distance to the nominal trajectory.

        Args:
            trajectory: Evaluated trajectory.

        Returns:
            Array of shape (T,) with per-step distances (lower = more nominal).
        """
        mat = trajectory_to_matrix(trajectory, self.extractor)
        T = min(len(trajectory), len(self.nominal_matrix))
        dists = np.zeros(T)
        for t in range(T):
            diff = (mat[t] - self.nominal_matrix[t]) / self._sigma
            dists[t] = float(np.sqrt(np.dot(diff, diff)))
        return dists


# ---------------------------------------------------------------------------
# Model 3: Transition model p(x_t | x_{t-1})
# ---------------------------------------------------------------------------


class TransitionModel(TrajectoryDensityModel):
    """Conditional Gaussian transition model p(x_t | x_{t-1}).

    Fit by stacking all consecutive (x_{t-1}, x_t) pairs from reference
    trajectories and learning the conditional mean and residual covariance via
    OLS regression:

        μ(x_{t-1}) = A @ x_{t-1} + b
        ε ~ N(0, Σ)

    Log-likelihood of a new trajectory is

        log p(τ) = Σ_{t=1}^{T-1} log N(x_t ; A x_{t-1} + b, Σ)

    Parameters
    ----------
    A : (d, d) regression weight matrix
    b : (d,) bias vector
    residual_cov : (d, d) residual covariance
    extractor : feature extraction function
    """

    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        residual_cov: np.ndarray,
        extractor: StepFeatureExtractor = default_step_features,
    ) -> None:
        self.A = A
        self.b = b
        self.residual_cov = residual_cov
        self.extractor = extractor

    @classmethod
    def fit(
        cls,
        trajectories: List[Trajectory],
        extractor: StepFeatureExtractor = default_step_features,
    ) -> "TransitionModel":
        """Fit the transition model from reference trajectories.

        Args:
            trajectories: Reference rollouts (any lengths; at least 2 steps each).
            extractor:    Feature extraction function.

        Returns:
            Fitted ``TransitionModel``.
        """
        X_prev: List[np.ndarray] = []
        X_next: List[np.ndarray] = []

        for traj in trajectories:
            mat = trajectory_to_matrix(traj, extractor)  # (T, d)
            X_prev.append(mat[:-1])  # (T-1, d)
            X_next.append(mat[1:])   # (T-1, d)

        if not X_prev:
            raise ValueError("Need at least one reference trajectory.")

        Xp = np.vstack(X_prev)  # (N, d)
        Xn = np.vstack(X_next)  # (N, d)

        # OLS: Xn = Xp @ A.T + 1 * b  →  augment with bias column
        Xp_aug = np.hstack([Xp, np.ones((len(Xp), 1))])  # (N, d+1)
        # Solve via least squares
        coeffs, _, _, _ = np.linalg.lstsq(Xp_aug, Xn, rcond=None)  # (d+1, d)
        A = coeffs[:-1].T   # (d, d)
        b = coeffs[-1]      # (d,)

        # Residuals and covariance
        Xn_pred = Xp @ A.T + b  # (N, d)
        residuals = Xn - Xn_pred  # (N, d)
        residual_cov = np.cov(residuals, rowvar=False) if len(residuals) > 1 else np.eye(Xn.shape[1])

        return cls(A=A, b=b, residual_cov=residual_cov, extractor=extractor)

    def predict_next(self, x: np.ndarray) -> np.ndarray:
        """Return the predicted mean of x_t given x_{t-1}.

        Args:
            x: Feature vector at step t-1, shape (d,).

        Returns:
            Predicted mean at step t, shape (d,).
        """
        return self.A @ x + self.b

    def step_log_likelihoods(self, trajectory: Trajectory) -> List[float]:
        """Per-step log p(x_t | x_{t-1}) for t = 1, …, T-1."""
        mat = trajectory_to_matrix(trajectory, self.extractor)
        T = len(trajectory)
        results = []
        for t in range(1, T):
            mean_t = self.predict_next(mat[t - 1])
            results.append(_mahal_log_prob(mat[t], mean_t, self.residual_cov))
        return results

    def log_likelihood(self, trajectory: Trajectory) -> float:
        return sum(self.step_log_likelihoods(trajectory))


# ---------------------------------------------------------------------------
# Convenience functions for common workflows
# ---------------------------------------------------------------------------


def trajectory_log_likelihood(
    trajectory: Trajectory,
    model: TrajectoryDensityModel,
) -> float:
    """Return log p(trajectory) under *model*.

    Args:
        trajectory: Evaluated rollout.
        model:      Any fitted ``TrajectoryDensityModel``.

    Returns:
        Scalar log-likelihood.
    """
    return model.log_likelihood(trajectory)


def compare_to_nominal(
    trajectory: Trajectory,
    nominal: Trajectory,
    sigma: float | np.ndarray = 1.0,
    extractor: StepFeatureExtractor = default_step_features,
) -> Tuple[float, np.ndarray]:
    """Score a trajectory against a nominal/truth trajectory.

    Convenience wrapper around ``NominalTrajectoryModel``.

    Args:
        trajectory: Rollout to evaluate.
        nominal:    Reference / ground-truth trajectory.
        sigma:      Per-feature (or global) standard deviation for the
                    isotropic Gaussian.
        extractor:  Feature extraction function.

    Returns:
        Tuple of (log_likelihood, deviation_profile) where
        ``deviation_profile`` is a (T,) array of per-step Mahalanobis
        distances to the nominal trajectory.
    """
    model = NominalTrajectoryModel(nominal=nominal, sigma=sigma, extractor=extractor)
    return model.log_likelihood(trajectory), model.deviation_profile(trajectory)


def fit_and_score(
    trajectory: Trajectory,
    reference_trajectories: List[Trajectory],
    model_type: str = "gaussian",
    extractor: StepFeatureExtractor = default_step_features,
) -> Tuple[float, List[float]]:
    """Fit a density model from reference trajectories and score a new trajectory.

    Args:
        trajectory:            Rollout to evaluate.
        reference_trajectories: Reference corpus used to fit the model.
        model_type:            ``"gaussian"`` (per-step marginal) or
                               ``"transition"`` (autoregressive).
        extractor:             Feature extraction function.

    Returns:
        Tuple of (total_log_likelihood, per_step_log_likelihoods).
    """
    if model_type == "gaussian":
        model = GaussianTrajectoryModel.fit(reference_trajectories, extractor)
    elif model_type == "transition":
        model = TransitionModel.fit(reference_trajectories, extractor)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}; use 'gaussian' or 'transition'.")

    per_step = model.step_log_likelihoods(trajectory)
    return sum(per_step), per_step
