"""Scenario sampling strategies for the verification environment.

Sampling strategies govern how initial states are selected from the pool of
candidate scenarios built by ``AircraftHeadingEnvironment``.  All strategies
operate on a *flat index* into that candidate pool, so they are decoupled from
the internal candidate representation.

Available strategies
--------------------
``UniformSampler``
    i.i.d. uniform — identical to the original behaviour.

``WeightedSampler``
    Categorical distribution with weights proportional to a user-supplied
    score function.  Good for biasing toward high-risk scenario conditions
    without the burn-in cost of MCMC.

``MetropolisHastingsSampler``
    Markov chain Monte Carlo using the Metropolis–Hastings algorithm.  The
    chain walks through scenario space guided by a target distribution
    π(i) ∝ exp(score(i)).  Supports two proposal kernels:

    * **"independence"** — every step proposes a uniformly-drawn candidate
      (independent of the current state).  The acceptance ratio simplifies
      to π(x') / π(x).  Mixes quickly when the score landscape is smooth.

    * **"random_walk"** — proposes from the k-nearest neighbours of the
      current candidate in a normalised feature space (utc_hour,
      proximity_nm, altitude_ft, traffic_count).  Provides locality-aware
      exploration.

Usage
-----
>>> from verification.system.samplers import MetropolisHastingsSampler, CandidateFeatures
>>>
>>> def my_score(f: CandidateFeatures) -> float:
...     # Prefer crowded, low-altitude, close-in scenarios
...     return -f.altitude_ft / 5000 + f.traffic_count / 50 - f.proximity_nm / 40
>>>
>>> sampler = MetropolisHastingsSampler(score_fn=my_score, kernel="random_walk")
>>> env = AircraftHeadingEnvironment(..., sampler=sampler)
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Candidate feature container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateFeatures:
    """Human-readable features extracted from a candidate scenario.

    Passed to the user-supplied score function, avoiding exposure of the raw
    internal tuple format.

    Attributes:
        index:         Position in the candidate list.
        timestamp_s:   Simulation timestamp (seconds from file epoch).
        utc_hour:      UTC hour of the snapshot (0–23; −1 if unavailable).
        proximity_nm:  Ego aircraft distance from airport (nautical miles).
        altitude_ft:   Ego aircraft geometric altitude (feet).
        traffic_count: Number of other aircraft in the snapshot.
        speed_kts:     Ego aircraft ground speed (knots).
    """

    index: int
    timestamp_s: float
    utc_hour: int
    proximity_nm: float
    altitude_ft: float
    traffic_count: int
    speed_kts: float


ScoreFn = Callable[[CandidateFeatures], float]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ScenarioSampler(abc.ABC):
    """Abstract scenario sampler.

    Concrete subclasses must implement :method:`sample`, which returns an index
    into the candidate list.  :method:`initialize` is called by the environment
    after the candidate list is built, providing the sampler with the feature
    metadata it needs.
    """

    def __init__(self) -> None:
        self._features: List[CandidateFeatures] = []

    def initialize(self, features: List[CandidateFeatures]) -> None:
        """Called once by the environment after the candidate pool is ready.

        Args:
            features: Feature descriptors, one per candidate, in index order.
        """
        self._features = features

    @abc.abstractmethod
    def sample(self) -> int:
        """Return the index of the next candidate to use as an initial state."""


# ---------------------------------------------------------------------------
# Uniform sampler (baseline)
# ---------------------------------------------------------------------------


class UniformSampler(ScenarioSampler):
    """Uniform i.i.d. sampling from the candidate pool.

    Equivalent to the original ``env.sample()`` behaviour.

    Args:
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self.rng.integers(len(self._features)))


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------


class WeightedSampler(ScenarioSampler):
    """Categorical sampling with weights proportional to ``exp(score_fn(f))``.

    This is a one-shot biased sampler (not a Markov chain).  Each call draws
    independently from the same categorical distribution, so there is no
    burn-in period and no autocorrelation between draws.  It is the right
    choice when you have a reliable score function and want unbiased
    importance-weighted estimates.

    Args:
        score_fn: Maps ``CandidateFeatures`` to a real-valued score.
                  Higher score → higher sampling probability.
        seed:     Optional RNG seed.
    """

    def __init__(self, score_fn: ScoreFn, seed: Optional[int] = None) -> None:
        super().__init__()
        self.score_fn = score_fn
        self.rng = np.random.default_rng(seed)
        self._probs: Optional[np.ndarray] = None

    def initialize(self, features: List[CandidateFeatures]) -> None:
        super().initialize(features)
        # Compute log-scores, then softmax → normalised probabilities
        log_scores = np.array([self.score_fn(f) for f in features], dtype=np.float64)
        log_scores -= log_scores.max()  # numerical stability
        weights = np.exp(log_scores)
        self._probs = weights / weights.sum()

    def sample(self) -> int:
        return int(self.rng.choice(len(self._features), p=self._probs))


# ---------------------------------------------------------------------------
# Metropolis–Hastings sampler
# ---------------------------------------------------------------------------


class MetropolisHastingsSampler(ScenarioSampler):
    """Metropolis–Hastings MCMC sampler over the candidate scenario pool.

    The stationary distribution of the chain is π(i) ∝ exp(score_fn(f_i)).

    Two proposal kernels are supported:

    * ``"independence"`` — proposes a candidate drawn uniformly at random.
      The acceptance ratio is simply exp(score(x') − score(x)).  Best when
      the score landscape is relatively flat or when k-NN indexing is too slow.

    * ``"random_walk"`` — proposes from the *k* nearest neighbours of the
      current candidate in a normalised 4-D feature space (utc_hour,
      proximity_nm, altitude_ft, traffic_count).  The symmetric proposal
      means the acceptance ratio is again exp(score(x') − score(x)).  Best
      when you want smooth, localised exploration (e.g. stress-testing a
      specific flight phase).

    The chain must be advanced with :method:`sample`.  Each call performs one
    MH step and returns the accepted (or retained) candidate index.  Call
    :method:`burn_in` after construction to discard early transients.

    Args:
        score_fn:    Maps ``CandidateFeatures`` → float (log-unnormalised score).
        kernel:      ``"independence"`` or ``"random_walk"``.
        k_neighbors: Number of neighbours for the random-walk kernel.
        seed:        Optional RNG seed.
    """

    def __init__(
        self,
        score_fn: ScoreFn,
        kernel: str = "independence",
        k_neighbors: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if kernel not in ("independence", "random_walk"):
            raise ValueError(f"Unknown kernel {kernel!r}; choose 'independence' or 'random_walk'")
        self.score_fn = score_fn
        self.kernel = kernel
        self.k_neighbors = k_neighbors
        self.rng = np.random.default_rng(seed)

        self._current_idx: int = 0
        self._current_score: float = -math.inf
        self._log_scores: Optional[np.ndarray] = None
        self._neighbors: Optional[List[np.ndarray]] = None  # random-walk only

        # Diagnostics
        self.n_proposed: int = 0
        self.n_accepted: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, features: List[CandidateFeatures]) -> None:
        super().initialize(features)
        n = len(features)

        # Pre-compute log-scores for all candidates
        self._log_scores = np.array([self.score_fn(f) for f in features], dtype=np.float64)

        # Initialise chain at the highest-scoring candidate
        self._current_idx = int(np.argmax(self._log_scores))
        self._current_score = float(self._log_scores[self._current_idx])

        if self.kernel == "random_walk":
            self._neighbors = self._build_knn(features, n)

    def _build_knn(
        self, features: List[CandidateFeatures], n: int
    ) -> List[np.ndarray]:
        """Pre-compute k-nearest neighbours in normalised feature space."""
        # Feature matrix: [utc_hour/23, proximity_nm/150, altitude_ft/18000, traffic_count/60]
        X = np.array([
            [
                (f.utc_hour if f.utc_hour >= 0 else 12) / 23.0,
                f.proximity_nm / 150.0,
                f.altitude_ft / 18_000.0,
                f.traffic_count / 60.0,
            ]
            for f in features
        ], dtype=np.float64)

        k = min(self.k_neighbors, n - 1)
        neighbors: List[np.ndarray] = []

        # Brute-force k-NN (fast enough for ~16k candidates at construction time)
        for i in range(n):
            dists = np.linalg.norm(X - X[i], axis=1)
            dists[i] = np.inf  # exclude self
            nn_idx = np.argpartition(dists, k)[:k]
            neighbors.append(nn_idx)

        return neighbors

    # ------------------------------------------------------------------
    # Burn-in
    # ------------------------------------------------------------------

    def burn_in(self, n_steps: int) -> "MetropolisHastingsSampler":
        """Advance the chain *n_steps* steps and reset acceptance counters.

        Returns ``self`` for chaining::

            sampler = MetropolisHastingsSampler(...).burn_in(500)

        Args:
            n_steps: Number of MH steps to discard.
        """
        for _ in range(n_steps):
            self._mh_step()
        # Reset counters so statistics reflect post-burn-in behaviour only
        self.n_proposed = 0
        self.n_accepted = 0
        return self

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals accepted since last reset (or burn-in)."""
        if self.n_proposed == 0:
            return float("nan")
        return self.n_accepted / self.n_proposed

    # ------------------------------------------------------------------
    # Core MH step
    # ------------------------------------------------------------------

    def _propose(self) -> int:
        """Draw a proposal index."""
        if self.kernel == "independence":
            return int(self.rng.integers(len(self._features)))
        # random_walk: pick uniformly from the pre-computed neighbourhood
        nbrs = self._neighbors[self._current_idx]  # type: ignore[index]
        return int(self.rng.choice(nbrs))

    def _mh_step(self) -> int:
        """Perform one MH step; return the (possibly unchanged) current index."""
        proposal = self._propose()
        self.n_proposed += 1

        proposal_score = float(self._log_scores[proposal])  # type: ignore[index]

        # Log acceptance ratio (both kernels are symmetric, so ratio = exp(Δscore))
        log_alpha = proposal_score - self._current_score
        if math.log(self.rng.random() + 1e-300) < log_alpha:
            self._current_idx = proposal
            self._current_score = proposal_score
            self.n_accepted += 1

        return self._current_idx

    def sample(self) -> int:
        return self._mh_step()
