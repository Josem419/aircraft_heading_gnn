"""Extensible discrete-time Signal Temporal Logic (STL) specification framework.

Specifications are objects that can be composed with Boolean and temporal operators.
All evaluate over a ``Trajectory``, returning a ``bool``.

Typical usage
-------------
    always_small_error = Always(Atomic(lambda step: cross_track(step) <= d_max))
    spec = always_small_error & always_separated & always_feasible_rate
    spec(trajectory)   # True  →  no failure
"""

from __future__ import annotations

import abc
from typing import Callable, Optional

from verification.system.system import Trajectory, TrajectoryStep


class Specification(abc.ABC):
    """Abstract base class for all specifications.

    Subclasses must implement :method:`evaluate`.  Operator overloads allow
    Boolean composition:
    """

    @abc.abstractmethod
    def evaluate(self, trajectory: Trajectory) -> bool:
        """Evaluate the specification on a trajectory.

        Args:
            trajectory: Rollout to evaluate.

        Returns:
            ``True`` if the trajectory satisfies the specification.
        """
        raise NotImplementedError

    # Convenience alias so specs are callable directly
    def __call__(self, trajectory: Trajectory) -> bool:
        return self.evaluate(trajectory)

    def __and__(self, other: Specification) -> "And":
        return And(self, other)

    def __or__(self, other: Specification) -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    def __repr__(self) -> str:  # pragma: no cover
        return self.__class__.__name__

class And(Specification):
    """Conjunction: ``left ∧ right``."""

    def __init__(self, left: Specification, right: Specification) -> None:
        self.left = left
        self.right = right

    def evaluate(self, trajectory: Trajectory) -> bool:
        return self.left(trajectory) and self.right(trajectory)

    def __repr__(self) -> str:
        return f"({self.left!r} ∧ {self.right!r})"


class Or(Specification):
    """Disjunction: ``left ∨ right``."""

    def __init__(self, left: Specification, right: Specification) -> None:
        self.left = left
        self.right = right

    def evaluate(self, trajectory: Trajectory) -> bool:
        return self.left(trajectory) or self.right(trajectory)

    def __repr__(self) -> str:
        return f"({self.left!r} ∨ {self.right!r})"


class Not(Specification):
    """Negation: ``¬ spec``."""

    def __init__(self, inner: Specification) -> None:
        self.inner = inner

    def evaluate(self, trajectory: Trajectory) -> bool:
        return not self.inner(trajectory)

    def __repr__(self) -> str:
        return f"¬{self.inner!r}"

class Atomic(Specification):
    """Atomic predicate evaluated at each step independently."""

    def __init__(
        self,
        predicate: Callable[[TrajectoryStep], bool],
        name: Optional[str] = None,
    ) -> None:
        self.predicate = predicate
        self.name = name

    def evaluate(self, trajectory: Trajectory) -> bool:
        return all(self.predicate(step) for step in trajectory)

    def at_step(self, step: TrajectoryStep) -> bool:
        """Evaluate the predicate at a single step (no temporal semantics)."""
        return self.predicate(step)

    def __repr__(self) -> str:
        return self.name if self.name else "Atomic(?)"


class Always(Specification):
    """
    Universal / box operator: □[a,b] inner.
    """

    def __init__(
        self,
        inner: Specification,
        a: int = 0,
        b: Optional[int] = None,
    ) -> None:
        self.inner = inner
        self.a = a
        self.b = b

    def evaluate(self, trajectory: Trajectory) -> bool:
        b = self.b if self.b is not None else len(trajectory) - 1
        sub = trajectory[self.a : b + 1]
        return self.inner(sub)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        interval = f"[{self.a},{self.b if self.b is not None else 'T'}]"
        return f"Always({interval}({self.inner!r}))"


class Eventually(Specification):
    """
    Existential / diamond operator: ◇[a,b] inner.
    """

    def __init__(
        self,
        inner: Specification,
        a: int = 0,
        b: Optional[int] = None,
    ) -> None:
        self.inner = inner
        self.a = a
        self.b = b

    def evaluate(self, trajectory: Trajectory) -> bool:
        b = self.b if self.b is not None else len(trajectory) - 1
        for step in trajectory[self.a : b + 1]:
            if self.inner([step]):  # type: ignore[arg-type]
                return True
        return False

    def __repr__(self) -> str:
        interval = f"[{self.a},{self.b if self.b is not None else 'T'}]"
        return f"Eventually({interval}({self.inner!r}))"


# ---------------------------------------------------------------------------
# Step-level wrapper (for use inside Always / Eventually)
# ---------------------------------------------------------------------------


class StepSpec(Specification):
    """Specification that evaluates a predicate at each step independently.

    Unlike :class:`Atomic`, ``StepSpec`` does *not* implicitly iterate — it
    evaluates the predicate on a single-element trajectory list (one step).
    Intended to be used as the ``inner`` argument of :class:`Always` or
    :class:`Eventually`.

    Parameters
    ----------
    predicate:
        Callable ``(TrajectoryStep) -> bool``.
    name:
        Optional human-readable label.
    """

    def __init__(
        self,
        predicate: Callable[[TrajectoryStep], bool],
        name: Optional[str] = None,
    ) -> None:
        self.predicate = predicate
        self.name = name

    def evaluate(self, trajectory: Trajectory) -> bool:
        return all(self.predicate(step) for step in trajectory)

    def __repr__(self) -> str:
        return self.name if self.name else "StepSpec(?)"
