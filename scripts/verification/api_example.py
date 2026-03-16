#!/usr/bin/env python3
"""
Minimal example demonstrating the verification system API.

This script shows how to:
1. Define concrete implementations of abstract base classes
2. Compose them into a System
3. Execute rollouts
4. Access trajectory data

This serves as both documentation and a test of the API.
"""

import sys
from pathlib import Path

import numpy as np
from verification.system import (
    AircraftState, SystemState, Action, Observation,
    Environment, DisturbanceModel, AgentModel, System,
    rollout, batch_rollout, Trajectory
)


# ============================================================================
# Minimal Implementations
# ============================================================================

class SimpleEnvironment(Environment):
    """Simple environment with circular motion."""
    
    def reset(self) -> SystemState:
        """Start at origin, heading north."""
        ego = AircraftState(
            position=np.array([0., 0.]),
            velocity=np.array([0., 100.]),  # 100 m/s north
            heading=0.0,  # North
            icao24="EGO001"
        )
        return SystemState(ego=ego, traffic=[], time=0.0)
    
    def sample_initial_state(self) -> SystemState:
        """For now, just use reset. Could randomize later."""
        return self.reset()
    
    def step(self, state: SystemState, ego_action: Action, dt: float) -> SystemState:
        """Simple kinematic propagation."""
        # Create new state (don't mutate input)
        new_state = state.copy()
        new_state.time += dt
        
        # Update ego heading (simple first-order response)
        target_heading = ego_action.heading_command
        current_heading = new_state.ego.heading
        
        # Apply simple heading change (could model turn rate limits)
        max_turn_rate = 0.1  # radians per second
        heading_diff = self._angle_diff(target_heading, current_heading)
        heading_change = np.clip(heading_diff, -max_turn_rate * dt, max_turn_rate * dt)
        new_state.ego.heading = current_heading + heading_change
        
        # Update velocity based on heading
        speed = new_state.ego.speed
        new_state.ego.velocity = np.array([
            speed * np.sin(new_state.ego.heading),
            speed * np.cos(new_state.ego.heading)
        ])
        
        # Update position
        new_state.ego.position = new_state.ego.position + new_state.ego.velocity * dt
        
        return new_state
    
    @staticmethod
    def _angle_diff(target, current):
        """Compute shortest angle difference."""
        diff = target - current
        # Wrap to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff


class NoDisturbance(DisturbanceModel):
    """No disturbances (perfect execution and observation)."""
    
    def apply_action_disturbance(self, action: Action) -> Action:
        """No disturbance."""
        return action
    
    def apply_observation_disturbance(self, observation: Observation) -> Observation:
        """No disturbance."""
        return observation


class GaussianDisturbance(DisturbanceModel):
    """Gaussian noise on actions and observations."""
    
    def __init__(self, sigma_act: float = 0.0, sigma_obs: float = 0.0):
        """
        Args:
            sigma_act: standard deviation for action noise (radians)
            sigma_obs: standard deviation for observation noise (meters)
        """
        self.sigma_act = sigma_act
        self.sigma_obs = sigma_obs
    
    def apply_action_disturbance(self, action: Action) -> Action:
        """Add Gaussian noise to heading command."""
        if self.sigma_act == 0:
            return action
        
        noise = np.random.normal(0, self.sigma_act)
        return Action(
            heading_command=action.heading_command + noise,
            metadata=action.metadata
        )
    
    def apply_observation_disturbance(self, observation: Observation) -> Observation:
        """Add Gaussian noise to aircraft positions."""
        if self.sigma_obs == 0:
            return observation
        
        # Perturb ego position
        ego_perturbed = AircraftState(
            position=observation.ego_state.position + np.random.normal(0, self.sigma_obs, 2),
            velocity=observation.ego_state.velocity.copy(),
            heading=observation.ego_state.heading,
            icao24=observation.ego_state.icao24,
            metadata=observation.ego_state.metadata
        )
        
        # Perturb traffic positions
        traffic_perturbed = []
        for ac in observation.traffic_states:
            ac_perturbed = AircraftState(
                position=ac.position + np.random.normal(0, self.sigma_obs, 2),
                velocity=ac.velocity.copy(),
                heading=ac.heading,
                icao24=ac.icao24,
                metadata=ac.metadata
            )
            traffic_perturbed.append(ac_perturbed)
        
        return Observation(
            ego_state=ego_perturbed,
            traffic_states=traffic_perturbed,
            graph_data=observation.graph_data,
            features=observation.features,
            metadata=observation.metadata
        )


class ConstantHeadingAgent(AgentModel):
    """Simple agent that always commands the same heading."""
    
    def __init__(self, heading: float = np.pi / 2):  # Default: East
        """
        Args:
            heading: constant heading to command (radians)
        """
        self.heading = heading
    
    def act(self, observation: Observation) -> Action:
        """Always return the same heading."""
        return Action(heading_command=self.heading)


class SimpleSystem(System):
    """Simple system implementation."""
    
    def step(self, state: SystemState, action: Action) -> SystemState:
        """One simulation step with disturbances."""
        # Apply action disturbance
        perturbed_action = self.disturbance_model.apply_action_disturbance(action)
        
        # Propagate environment
        next_state = self.environment.step(state, perturbed_action, self.dt)
        
        return next_state
    
    def get_observation(self, state: SystemState) -> Observation:
        """Extract observation with disturbances."""
        # Create clean observation
        obs = Observation(
            ego_state=state.ego.copy(),
            traffic_states=[ac.copy() for ac in state.traffic]
        )
        
        # Apply observation disturbance
        return self.disturbance_model.apply_observation_disturbance(obs)


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Run example rollouts."""
    print("=" * 70)
    print("Verification System API Example")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Example 1: Perfect system (no disturbances)
    # ========================================================================
    print("Example 1: Perfect system (no disturbances)")
    print("-" * 70)
    
    env = SimpleEnvironment()
    disturbance = NoDisturbance()
    agent = ConstantHeadingAgent(heading=np.pi / 2)  # Command East
    system = SimpleSystem(env, agent, disturbance, dt=1.0)
    
    trajectory = rollout(system, num_steps=10)
    
    print(f"Rollout completed: {len(trajectory)} steps")
    print()
    print("First 5 steps:")
    for i, step in enumerate(trajectory[:5]):
        print(f"  Step {i}:")
        print(f"    Time: {step.state.time:.1f}s")
        print(f"    Position: ({step.state.ego.position[0]:.1f}, {step.state.ego.position[1]:.1f})")
        print(f"    Heading: {step.state.ego.heading:.3f} rad ({np.degrees(step.state.ego.heading):.1f}°)")
        print(f"    Commanded: {step.action.heading_command:.3f} rad ({np.degrees(step.action.heading_command):.1f}°)")
    print()
    
    # ========================================================================
    # Example 2: System with disturbances
    # ========================================================================
    print("Example 2: System with disturbances")
    print("-" * 70)
    
    disturbance_noisy = GaussianDisturbance(sigma_act=0.05, sigma_obs=1.0)
    system_noisy = SimpleSystem(env, agent, disturbance_noisy, dt=1.0)
    
    trajectory_noisy = rollout(system_noisy, num_steps=10)
    
    print(f"Rollout completed: {len(trajectory_noisy)} steps")
    print()
    print("First 5 steps (with noise):")
    for i, step in enumerate(trajectory_noisy[:5]):
        print(f"  Step {i}:")
        print(f"    Time: {step.state.time:.1f}s")
        print(f"    Position: ({step.state.ego.position[0]:.1f}, {step.state.ego.position[1]:.1f})")
        print(f"    Heading: {step.state.ego.heading:.3f} rad")
        print(f"    Commanded: {step.action.heading_command:.3f} rad (includes noise)")
    print()
    
    # ========================================================================
    # Example 3: Batch rollouts
    # ========================================================================
    print("Example 3: Batch rollouts")
    print("-" * 70)
    
    trajectories = batch_rollout(system_noisy, num_rollouts=5, num_steps=20)
    
    print(f"Completed {len(trajectories)} rollouts")
    print()
    print("Final positions:")
    for i, traj in enumerate(trajectories):
        final_pos = traj[-1].next_state.ego.position
        print(f"  Rollout {i}: ({final_pos[0]:.1f}, {final_pos[1]:.1f})")
    print()
    
    # ========================================================================
    # Example 4: Accessing trajectory data
    # ========================================================================
    print("Example 4: Extracting data from trajectories")
    print("-" * 70)
    
    traj = trajectories[0]
    
    # Extract time series
    times = np.array([step.state.time for step in traj])
    positions = np.array([step.state.ego.position for step in traj])
    headings = np.array([step.state.ego.heading for step in traj])
    commands = np.array([step.action.heading_command for step in traj])
    
    print(f"Extracted {len(times)} timesteps")
    print(f"Time range: {times[0]:.1f}s to {times[-1]:.1f}s")
    print(f"Position range: ({positions[:, 0].min():.1f}, {positions[:, 1].min():.1f}) to "
          f"({positions[:, 0].max():.1f}, {positions[:, 1].max():.1f})")
    print(f"Heading range: {headings.min():.3f} to {headings.max():.3f} rad")
    print()
    
    print("=" * 70)
    print("API example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
