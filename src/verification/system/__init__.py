"""Verification system module - core abstractions for verification and validation."""

from verification.system.system import (
    # State representations
    AircraftState,
    SystemState,
    # Action and observation
    Action,
    Observation,
    # Trajectory
    TrajectoryStep,
    Trajectory,
    # Abstract base classes
    Environment,
    DisturbanceModel,
    AgentModel,
    System,
)

from verification.system.rollouts import (
    rollout,
    batch_rollout,
)

__all__ = [
    # State
    'AircraftState',
    'SystemState',
    # Action/Observation
    'Action',
    'Observation',
    # Trajectory
    'TrajectoryStep',
    'Trajectory',
    # Base classes
    'Environment',
    'DisturbanceModel',
    'AgentModel',
    'System',
    # Rollout functions
    'rollout',
    'batch_rollout',
]