#!/usr/bin/env python3
"""
Example demonstrating GNN integration with the verification system.

This shows how to:
1. Load a trained GNN model
2. Wrap it as an AgentModel
3. Convert observations to graph format
4. Use it in rollouts for verification
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from verification.system import (
    AircraftState, SystemState, Action, Observation,
    AgentModel, rollout
)

# Import your GNN models
from aircraft_heading_gnn.models.base_gnn import BaseGNN, GATHeadingPredictor
from aircraft_heading_gnn.utils.adsb_features import get_distance_nm, get_azimuth_to_point


class GNNAgent(AgentModel):
    """Agent that wraps a trained GNN heading predictor.
    
    This converts observations to graph format and runs GNN inference.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        airport_lat: float,
        airport_lon: float,
        device: str = "cpu",
        num_classes: int = 72,
        heading_bin_size: float = 5.0,
    ):
        """
        Args:
            model: Trained GNN model (BaseGNN or GATHeadingPredictor)
            airport_lat: Airport latitude for reference features
            airport_lon: Airport longitude for reference features  
            device: Device for inference ("cpu" or "cuda")
            num_classes: Number of heading bins (72 for 5-degree bins)
            heading_bin_size: Size of each heading bin in degrees
        """
        self.model = model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        self.airport_lat = airport_lat
        self.airport_lon = airport_lon
        self.device = device
        self.num_classes = num_classes
        self.heading_bin_size = heading_bin_size
    
    def act(self, observation: Observation) -> Action:
        """Compute heading command from observation using GNN.
        
        Args:
            observation: Current observation with ego and traffic states
        
        Returns:
            Action with predicted heading command
        """
        # Convert observation to graph
        graph_data = self._observation_to_graph(observation)
        
        # Run GNN inference
        with torch.no_grad():
            graph_data = graph_data.to(self.device)
            logits = self.model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                None  # No batch for single graph
            )
            
            # Get ego node prediction (assumes ego is first node)
            ego_logits = logits[0]  # Shape: [num_classes]
            predicted_class = torch.argmax(ego_logits).item()
        
        # Convert class to heading
        predicted_heading_deg = predicted_class * self.heading_bin_size
        predicted_heading_rad = np.radians(predicted_heading_deg)
        
        return Action(
            heading_command=predicted_heading_rad,
            metadata={
                'predicted_class': predicted_class,
                'confidence': torch.softmax(ego_logits, dim=0)[predicted_class].item(),
                'logits': ego_logits.cpu().numpy()
            }
        )
    
    def _observation_to_graph(self, observation: Observation) -> Data:
        """Convert observation to PyTorch Geometric graph.
        
        This constructs a graph with:
        - Nodes: ego + traffic aircraft
        - Edges: proximity-based connections
        - Node features: position, velocity, heading relative to airport
        - Edge features: distance, bearing
        
        Args:
            observation: Current observation
        
        Returns:
            PyG Data object
        """
        # Collect all aircraft (ego first, then traffic)
        all_aircraft = [observation.ego_state] + observation.traffic_states
        num_nodes = len(all_aircraft)
        
        # Build node features
        node_features = []
        for ac in all_aircraft:
            # Convert position to lat/lon (assuming position is in local x,y meters)
            # For proper implementation, you'd convert back to lat/lon
            # Here we use a simple approximation
            lat = ac.position[1] / 111000 + self.airport_lat  # ~111km per degree
            lon = ac.position[0] / (111000 * np.cos(np.radians(self.airport_lat))) + self.airport_lon
            
            # Node features: [lat, lon, heading, velocity, geoaltitude, vertrate]
            features = [
                lat,
                lon,
                np.degrees(ac.heading),  # Convert to degrees
                ac.speed,  # Use speed property
                ac.metadata.get('altitude', 0.0) if ac.metadata else 0.0,
                ac.metadata.get('vertrate', 0.0) if ac.metadata else 0.0,
            ]
            node_features.append(features)
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edges based on proximity
        edge_index = []
        edge_features = []
        max_distance_nm = 35.0  # Match training default
        
        for i, ac_i in enumerate(all_aircraft):
            for j, ac_j in enumerate(all_aircraft):
                if i == j:
                    continue  # No self-loops
                
                # Compute distance
                # For proper implementation, convert positions to lat/lon
                dist_m = np.linalg.norm(ac_i.position - ac_j.position)
                dist_nm = dist_m / 1852.0  # meters to nautical miles
                
                if dist_nm <= max_distance_nm:
                    edge_index.append([i, j])
                    
                    # Edge features: [distance, relative_bearing, speed_diff, ...]
                    # Compute bearing from i to j
                    delta = ac_j.position - ac_i.position
                    bearing = np.arctan2(delta[0], delta[1])  # Bearing in radians
                    relative_bearing = bearing - ac_i.heading
                    
                    # Normalize angle to [-pi, pi]
                    while relative_bearing > np.pi:
                        relative_bearing -= 2 * np.pi
                    while relative_bearing < -np.pi:
                        relative_bearing += 2 * np.pi
                    
                    edge_feat = [
                        dist_nm,
                        np.degrees(relative_bearing),
                        ac_j.speed - ac_i.speed,
                    ]
                    edge_features.append(edge_feat)
        
        if len(edge_index) == 0:
            # No edges - create empty tensor
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        
        return data
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_type: str,
        airport_lat: float,
        airport_lon: float,
        num_node_features: int = 6,
        num_edge_features: int = 3,
        device: str = "cpu",
        **kwargs
    ) -> 'GNNAgent':
        """Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt model file
            model_type: "gcn" or "gat"
            airport_lat: Airport latitude
            airport_lon: Airport longitude
            num_node_features: Number of node features (default: 6)
            num_edge_features: Number of edge features (default: 3)
            device: Device for inference
            **kwargs: Additional model hyperparameters (hidden_dim, num_layers, etc.)
        
        Returns:
            GNNAgent instance
        """
        # Set default hyperparameters
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_layers = kwargs.get('num_layers', 3)
        num_classes = kwargs.get('num_classes', 72)
        dropout = kwargs.get('dropout', 0.2)
        
        # Create model based on type
        if model_type == "gcn":
            model = BaseGNN(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            )
        elif model_type == "gat":
            num_heads = kwargs.get('num_heads', 4)
            model = GATHeadingPredictor(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        
        print(f"Loaded {model_type.upper()} model from {checkpoint_path}")
        
        return cls(
            model=model,
            airport_lat=airport_lat,
            airport_lon=airport_lon,
            device=device,
            num_classes=num_classes,
        )


def main():
    """Demonstrate GNN integration."""
    print("=" * 70)
    print("GNN Agent Integration Example")
    print("=" * 70)
    print()
    
    # ========================================================================
    # Option 1: Create a simple mock GNN for demonstration
    # ========================================================================
    print("Creating mock GNN agent...")
    
    # Create a simple GNN model (untrained, just for demonstration)
    mock_model = BaseGNN(
        num_node_features=6,
        num_edge_features=3,
        hidden_dim=64,
        num_layers=2,
        num_classes=72,
        dropout=0.0,
    )
    
    # Create agent (Seattle coordinates)
    gnn_agent = GNNAgent(
        model=mock_model,
        airport_lat=47.4502,
        airport_lon=-122.3088,
        device="cpu"
    )
    
    print(f"✓ GNN agent created")
    print(f"  Model: {type(mock_model).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in mock_model.parameters()):,}")
    print()
    
    # ========================================================================
    # Test observation to graph conversion
    # ========================================================================
    print("Testing observation → graph conversion...")
    
    # Create test observation
    ego = AircraftState(
        position=np.array([0., 0.]),
        velocity=np.array([100., 0.]),
        heading=np.pi/4,  # 45 degrees
        icao24="EGO001",
        metadata={'altitude': 1000.0, 'vertrate': 0.0}
    )
    
    traffic = [
        AircraftState(
            position=np.array([5000., 2000.]),
            velocity=np.array([80., 50.]),
            heading=np.pi/2,
            icao24="TFC001",
            metadata={'altitude': 1500.0, 'vertrate': 5.0}
        ),
        AircraftState(
            position=np.array([-3000., 4000.]),
            velocity=np.array([90., -30.]),
            heading=0.0,
            icao24="TFC002",
            metadata={'altitude': 2000.0, 'vertrate': -2.0}
        ),
    ]
    
    observation = Observation(
        ego_state=ego,
        traffic_states=traffic
    )
    
    graph = gnn_agent._observation_to_graph(observation)
    print(f"✓ Graph created:")
    print(f"  Nodes: {graph.x.shape[0]}")
    print(f"  Node features: {graph.x.shape[1]}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Edge features: {graph.edge_attr.shape if graph.edge_attr is not None else 'None'}")
    print()
    
    # ========================================================================
    # Test GNN inference
    # ========================================================================
    print("Testing GNN inference...")
    
    action = gnn_agent.act(observation)
    print(f"✓ Action computed:")
    print(f"  Heading command: {action.heading_command:.4f} rad ({np.degrees(action.heading_command):.1f}°)")
    print(f"  Predicted class: {action.metadata['predicted_class']}")
    print(f"  Confidence: {action.metadata['confidence']:.4f}")
    print()
    
    # ========================================================================
    # Option 2: Load from checkpoint (commented - uncomment to use real model)
    # ========================================================================
    print("To use a trained model:")
    print("-" * 70)
    print("""
    gnn_agent = GNNAgent.from_checkpoint(
        checkpoint_path="best_model_gcn.pt",
        model_type="gcn",
        airport_lat=47.4502,
        airport_lon=-122.3088,
        hidden_dim=128,
        num_layers=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Then use in rollouts with your Environment and System implementations
    from my_system import MyEnvironment, MyDisturbance, MySystem
    
    env = MyEnvironment(...)
    disturbance = MyDisturbance(...)
    system = MySystem(env, gnn_agent, disturbance)
    
    trajectory = rollout(system, num_steps=100)
    """)
    print()
    
    print("=" * 70)
    print("GNN integration example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
