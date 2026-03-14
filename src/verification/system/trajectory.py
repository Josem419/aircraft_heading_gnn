"""Trajectory representation for system rollouts."""

from dataclasses import dataclass
from typing import List
from verification.system.state import SystemState
from verification.system.actions import Action
from verification.system.observations import Observation


@dataclass
class TrajectoryStep:
    """Single step in a trajectory.

    Attributes:
        state: system state at this step
        action: action taken at this step
        observation: observation received at this step
        next_state: resulting state after action
        disturbance_log_prob: sum of log-probabilities of observation and
            action disturbances applied at this step (0.0 if the disturbance
            model does not provide density information, or when sigma == 0).
    """

    state: SystemState
    action: Action
    observation: Observation
    next_state: SystemState
    disturbance_log_prob: float = 0.0


class Trajectory(List[TrajectoryStep]):
    """Trajectory represented as a sequence of (state, action, observation, next_state) steps.

    This is a list subclass for convenience, but adds semantic meaning.
    """

    pass
