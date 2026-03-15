"""Library System models, including environment, agent, and disturbance models.
    Defines the API contract for systems so that we can apply verifications tools
    in a modular way.
"""

import abc
from typing import Dict, Any
from verification.system.state import SystemState
from verification.system.observations import Observation
from verification.system.actions import Action


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
    def sample_initial_state(self) -> SystemState:
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

    @property
    def last_action_log_prob(self) -> float:
        """Log-probability of the disturbance applied in the most recent
        ``apply_action_disturbance`` call.  Subclasses should override this
        when they can compute the density.  Defaults to 0.0."""
        return 0.0

    @property
    def last_obs_log_prob(self) -> float:
        """Log-probability of the disturbance applied in the most recent
        ``apply_observation_disturbance`` call.  Subclasses should override
        this when they can compute the density.  Defaults to 0.0."""
        return 0.0

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

    def sample_initial_state(self) -> SystemState:
        """Sample a random initial state.

        Returns:
            Random initial system state
        """
        return self.environment.sample_initial_state()
