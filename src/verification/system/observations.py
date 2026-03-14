""" Observation representations for system rollouts """

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from verification.system.state import AircraftState

@dataclass
class Observation:
    """Observation received by the agent.

    An observations include the observable ego state, which we will take
    as the dead-reckoned state of the ego aircraft based on our agent model. We assume
    we follow our agent model perfectly, so the ego state is not perturbed by noise.


    And observable traffic states. In this case that is any information we
    can extract from ADS-B messages, which includes position, velocity, heading, 

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
