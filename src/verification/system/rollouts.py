"""Library of rollout definitions and utilities for the verification system.

This module provides functions for executing system rollouts given a policy.
Rollouts form the basis for verification and validation experiments.
"""

from typing import Optional
from verification.system.state import SystemState
from verification.system.trajectory import Trajectory, TrajectoryStep
from verification.system.system import System, AgentModel


def rollout(
    system: System,
    initial_state: Optional[SystemState] = None,
    num_steps: int = 100,
    agent: Optional[AgentModel] = None
) -> Trajectory:
    """Execute a rollout of the system for a given number of steps.
    
    The rollout follows this sequence at each step:
    1. Get observation from current state (with disturbances)
    2. Agent computes action from observation
    3. System steps forward with action (with disturbances)
    4. Record (state, action, observation, next_state)
    
    Args:
        system: system to roll out
        initial_state: initial state (if None, samples from environment)
        num_steps: number of steps to simulate
        agent: agent model to use (if None, uses system.agent_model)
    
    Returns:
        Trajectory containing all steps
    """
    # Use provided agent or fall back to system's agent
    if agent is None:
        agent = system.agent_model
    
    # Initialize state
    if initial_state is None:
        state = system.sample_initial_state()
    else:
        state = initial_state.copy()  # Copy to avoid mutating input
    
    trajectory = Trajectory()
    
    for _ in range(num_steps):
        # Get observation (may include disturbances)
        observation = system.get_observation(state)
        obs_lp = system.disturbance_model.last_obs_log_prob

        # Agent decides action
        action = agent.act(observation)

        # Step system forward (applies disturbances and dynamics)
        next_state = system.step(state, action)
        act_lp = system.disturbance_model.last_action_log_prob

        # Record step
        step = TrajectoryStep(
            state=state,
            action=action,
            observation=observation,
            next_state=next_state,
            disturbance_log_prob=obs_lp + act_lp,
        )
        trajectory.append(step)
        
        # Advance
        state = next_state
    
    return trajectory


def batch_rollout(
    system: System,
    num_rollouts: int,
    num_steps: int = 100,
    initial_states: Optional[list[SystemState]] = None,
    agent: Optional[AgentModel] = None
) -> list[Trajectory]:
    """Execute multiple rollouts in batch.
    
    Args:
        system: system to roll out
        num_rollouts: number of rollouts to execute
        num_steps: number of steps per rollout
        initial_states: list of initial states (if None, samples from environment)
        agent: agent model to use (if None, uses system.agent_model)
    
    Returns:
        List of trajectories
    """
    trajectories = []
    
    for i in range(num_rollouts):
        # Use provided initial state if available
        if initial_states is not None and i < len(initial_states):
            initial_state = initial_states[i]
        else:
            initial_state = None  # Will sample
        
        traj = rollout(
            system=system,
            initial_state=initial_state,
            num_steps=num_steps,
            agent=agent
        )
        trajectories.append(traj)
    
    return trajectories