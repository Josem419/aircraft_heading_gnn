""" Library of system state representations"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, List
import numpy as np

@dataclass
class AircraftState:
    """State of a single aircraft.

    Attributes:
        position: (x, y, z) position in meters (local coordinates)
        velocity: (vx, vy, vz) velocity in m/s
        heading: heading angle in radians (0 = North, clockwise)
        icao24: unique aircraft identifier (optional)
        metadata: additional aircraft data (e.g., callsign, altitude)
    """

    position: np.ndarray  # shape (3,)
    velocity: np.ndarray  # shape (3,)
    heading: float
    icao24: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure arrays are numpy arrays with correct shape."""
        self.position = np.asarray(self.position, dtype=np.float32)
        self.velocity = np.asarray(self.velocity, dtype=np.float32)
        assert self.position.shape == (
            3,
        ), f"Position must be 3D, got {self.position.shape}"
        assert self.velocity.shape == (
            3,
        ), f"Velocity must be 3D, got {self.velocity.shape}"

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

