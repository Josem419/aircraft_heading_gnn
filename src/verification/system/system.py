"""Library System models, including environment, agent, and disturbance models.

This module provides the core abstractions for verification and validation:
- State representations for systems
- Action and observation types
- Environment, disturbance, and agent abstractions
- System composition with standardized API in the abstract base System class

The API is designed to support rollout-based verification where:
1. Environment samples initial states
2. Agent produces actions from observations
3. Disturbances perturb actions and observations
4. System propagates state forward
"""

import abc
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np


# ============================================================================
# State Representations
# ============================================================================


@dataclass
class AircraftState:
    """State of a single aircraft.

    Attributes:
        position: (x, y) position in meters (local coordinates)
        velocity: (vx, vy) velocity in m/s
        heading: heading angle in radians (0 = North, clockwise)
        icao24: unique aircraft identifier (optional)
        metadata: additional aircraft data (e.g., callsign, altitude)
    """

    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)
    heading: float
    icao24: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure arrays are numpy arrays with correct shape."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        assert self.position.shape == (
            2,
        ), f"Position must be 2D, got {self.position.shape}"
        assert self.velocity.shape == (
            2,
        ), f"Velocity must be 2D, got {self.velocity.shape}"

    @property
    def speed(self) -> float:
        """Ground speed in m/s."""
        return np.linalg.norm(self.velocity)

    def copy(self) -> "AircraftState":
        """Create a deep copy of this aircraft state."""
        return AircraftState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=self.heading,
            icao24=self.icao24,
            metadata=self.metadata.copy() if self.metadata else None,
        )


@dataclass
class SystemState:
    """Full system state including ego aircraft and surrounding traffic.

    Attributes:
        ego: state of the ego aircraft (under our control)
        traffic: list of other aircraft states
        time: current simulation time in seconds
        metadata: additional system-level data
    """

    ego: AircraftState
    traffic: List[AircraftState]
    time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def copy(self) -> "SystemState":
        """Create a deep copy of this system state."""
        return SystemState(
            ego=self.ego.copy(),
            traffic=[ac.copy() for ac in self.traffic],
            time=self.time,
            metadata=self.metadata.copy() if self.metadata else None,
        )


# ============================================================================
# Action and Observation Types
# ============================================================================


@dataclass
class Action:
    """Action taken by the agent.

    For the aircraft heading advisory system, this is a commanded heading.

    Attributes:
        heading_command: target heading in radians
        metadata: additional action data
    """

    heading_command: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Observation:
    """Observation received by the agent.

    This can contain raw or processed features used by the agent.

    Attributes:
        ego_state: observed ego aircraft state
        traffic_states: observed traffic aircraft states
        graph_data: optional graph representation (e.g., PyG Data object)
        features: optional processed feature vector
        metadata: additional observation data
    """

    ego_state: AircraftState
    traffic_states: List[AircraftState]
    graph_data: Optional[Any] = None  # Could be torch_geometric.data.Data
    features: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrajectoryStep:
    """Single step in a trajectory.

    Attributes:
        state: system state at this step
        action: action taken at this step
        observation: observation received at this step
        next_state: resulting state after action
    """

    state: SystemState
    action: Action
    observation: Observation
    next_state: SystemState


class Trajectory(List[TrajectoryStep]):
    """Trajectory represented as a sequence of (state, action, observation, next_state) steps.

    This is a list subclass for convenience, but adds semantic meaning.
    """

    pass


# ============================================================================
# Abstract Base Classes
# ============================================================================


class Environment(abc.ABC):
    """Abstract base class for environments.

    The environment is responsible for:
    - Sampling initial states (scenarios)
    - Propagating system dynamics (physics)
    - Managing background traffic
    """

    @abc.abstractmethod
    def reset(self) -> SystemState:
        """Reset the environment to a fixed initial state.

        Returns:
            Initial system state
        """
        pass

    @abc.abstractmethod
    def sample(self) -> SystemState:
        """Sample a random initial state from the environment.

        Returns:
            Random initial system state
        """
        pass

    @abc.abstractmethod
    def step(self, state: SystemState, ego_action: Action, dt: float) -> SystemState:
        """Propagate the system forward by dt seconds.

        This applies the ego action and propagates all aircraft forward
        according to their dynamics. Background traffic follows their
        recorded trajectories.

        Args:
            state: current system state
            ego_action: action for ego aircraft
            dt: time step in seconds

        Returns:
            Next system state
        """
        pass


class AgentModel(abc.ABC):
    """Abstract base class for agent models.

    The agent maps observations to actions (e.g., GNN heading predictor).
    """

    @abc.abstractmethod
    def act(self, observation: Observation) -> Action:
        """Compute action from observation.

        Args:
            observation: current observation

        Returns:
            Action to execute
        """
        pass

class ObservationModel(abc.ABC):
    """Abstract base class for observation models.

    The observation model defines how the true system state is converted
    into the observation that the agent receives, potentially with noise.
    """

    @abc.abstractmethod
    def observe(self, state: SystemState) -> Observation:
        """Convert system state to observation.

        Args:
            state: current system state

        Returns:
            Observation for the agent
        """
        pass


class DisturbanceModel(abc.ABC):
    """Abstract base class for disturbance models.

    Disturbances model uncertainty in:
    - Action execution (heading command errors)
    - Observations (sensor noise, tracking errors)
    """

    @abc.abstractmethod
    def apply_action_disturbance(self, action: Action) -> Action:
        """Apply disturbance to an action.

        Args:
            action: commanded action

        Returns:
            Perturbed action (what actually gets executed)
        """
        pass

    @abc.abstractmethod
    def apply_observation_disturbance(self, observation: Observation) -> Observation:
        """Apply disturbance to an observation.

        Args:
            observation: true observation

        Returns:
            Perturbed observation (what the agent sees)
        """
        pass
    
    @abc.abstractmethod
    def sample_environment_disturbance(self) -> Dict[str, Any]:
        """Sample a random disturbance for the environment.

        This can include things like wind gusts, or other random factors
        that affect the system state but are not directly tied to actions
        or observations.

        Returns:
            Dictionary of disturbance parameters to apply in the environment step
        """
        pass

class System(abc.ABC):
    """Abstract base class for the overall system.

    The system composes environment, agent, and disturbances into a
    closed-loop that can be rolled out for verification.

    API Contract (must be implemented):
    - step: propagate system forward one time step
    - get_observation: extract observation from state
    """

    def __init__(
        self,
        environment: Environment,
        agent_model: AgentModel,
        observation_model: ObservationModel,
        disturbance_model: DisturbanceModel,
    ):
        """Initialize system.

        Args:
            environment: environment model
            agent_model: agent/policy model
            observation_model: observation model
            disturbance_model: disturbance model
        """
        self.environment: Environment = environment
        self.agent_model: AgentModel = agent_model
        self.observation_model: ObservationModel = observation_model
        self.disturbance_model: DisturbanceModel = disturbance_model

    @abc.abstractmethod
    def step(self, state: SystemState, action: Action) -> SystemState:
        """Propagate the system forward one time step with disturbances.

        This is the core simulation step:
        1. Apply action disturbance
        2. Propagate state via environment
        3. Return next state

        Args:
            state: current system state
            action: commanded action (will be perturbed)

        Returns:
            Next system state
        """
        pass

    @abc.abstractmethod
    def get_observation(self, state: SystemState) -> Observation:
        """Extract observation from system state.

        This converts raw state into what the agent sees, potentially
        with observation disturbances applied.

        Args:
            state: current system state

        Returns:
            Observation for the agent
        """
        pass

    def reset(self) -> SystemState:
        """Reset to a fixed initial state.

        Returns:
            Initial system state
        """
        return self.environment.reset()

    def sample(self) -> SystemState:
        """Sample a random initial state.

        Returns:
            Random initial system state
        """
        return self.environment.sample()
