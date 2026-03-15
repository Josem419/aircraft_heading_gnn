"""Verification system module - core abstractions for verification and validation."""

from verification.system.state import AircraftState, SystemState
from verification.system.actions import Action
from verification.system.observations import Observation
from verification.system.trajectory import TrajectoryStep, Trajectory
from verification.system.system import Environment, DisturbanceModel, AgentModel, System

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