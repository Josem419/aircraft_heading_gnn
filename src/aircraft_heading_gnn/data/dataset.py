"""
PyTorch Geometric Dataset for aircraft trajectory graphs.
Constructs spatial-temporal graphs from ADS-B data for heading prediction.
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from typing import Optional, List, Tuple, Dict
import os
import json
from pathlib import Path
import pickle

from aircraft_heading_gnn.utils.adsb_features import get_distance_nm, get_azimuth_to_point
from aircraft_heading_gnn.utils.angles import normalize_angle, ang_diff_deg
from aircraft_heading_gnn.data.preprocessing import (
            filter_terminal_area, clean_trajectories, compute_derived_features
        )
        

# Terminal area radius in degrees (40 nautical miles ≈ 0.67 degrees)
TERMINAL_RADIUS_DEG = 40.0 / 60.0
TERMINAL_RADIUS_NM = 180.0
MAX_DISTANCE_BETWEEN_AIRCRAFT_NM = 35.0  # for edge creation
PREDICTION_HORIZON_SEC = 15  # seconds ahead to predict heading
TIME_BETWEEN_SNAPSHOTS_SEC = 10  # seconds between graph snapshots


class AircraftGraphDataset(Dataset):
    """
    Dataset that creates graph snapshots from aircraft trajectories.

    Each graph represents aircraft in a terminal area at a specific time:
    - Nodes: Individual aircraft with their state features
    - Edges: Spatial relationships (proximity, bearing)
    - Target: Future heading for each aircraft
    """

    @property
    def raw_file_names(self):
        """Raw parquet file."""
        return ['out.parquet'] if self.use_parquet else []

    @property
    def processed_file_names(self):
        """Processed graph snapshots."""
        return ['snapshots.pt'] if self.use_parquet else []

    def download(self):
        """Download not needed - parquet file already exists."""
        pass

    def process(self):
        """Process parquet file into graph snapshots."""
        if not self.use_parquet:
            return
            
        print("Processing parquet file into graph snapshots...")
        
        # Load parquet file
        if self.parquet_path:
            parquet_path = self.parquet_path
        else:
            parquet_path = os.path.join(self.raw_dir, 'out.parquet')
        df = pd.read_parquet(parquet_path)
        
        print(f"Loaded {len(df):,} records from parquet")
        
        # Filter to terminal area
        df = filter_terminal_area(
            df, self.airport_lat, self.airport_lon,
            radius_nm=TERMINAL_RADIUS_NM,
            max_altitude_ft=18000.0
        )
        print(f"After terminal area filter: {len(df):,} records")
        
        # Clean trajectories
        df = clean_trajectories(
            df,
            min_points=30,
            max_gap_seconds=60.0,
            min_speed_kts=50.0,
            max_speed_kts=600.0
        )
        print(f"After cleaning: {len(df):,} records")
        
        # Compute derived features
        df = compute_derived_features(df)
        
        # Store processed dataframe
        self.data_df = df
        
        # Create snapshots
        self.snapshots = self._create_snapshots()
        print(f"Created {len(self.snapshots)} graph snapshots")
        
        # Save processed snapshots
        processed_path = os.path.join(self.processed_dir, 'snapshots.pt')
        torch.save({
            'snapshots': self.snapshots,
            'data_df': self.data_df,
            'airport_lat': self.airport_lat,
            'airport_lon': self.airport_lon,
            'params': {
                'time_step': self.time_step,
                'prediction_horizon': self.prediction_horizon,
                'max_distance_nm': self.max_distance_nm
            }
        }, processed_path)
        print(f"Saved processed data to {processed_path}")

    def __init__(
        self,
        data_df: pd.DataFrame = None,
        airport_lat: float = None,
        airport_lon: float = None,
        airports_json_path: str = None,
        airport_icao: str = "KSEA",
        time_step_s: int = TIME_BETWEEN_SNAPSHOTS_SEC,
        prediction_horizon_s: int = PREDICTION_HORIZON_SEC,
        max_distance_nm: float = MAX_DISTANCE_BETWEEN_AIRCRAFT_NM,
        transform=None,
        pre_transform=None,
        use_parquet: bool = False,
        parquet_path: str = None,
    ):
        """
        Args:
            data_df: Preprocessed DataFrame with trajectories (if use_parquet=False)
            airport_lat: Airport latitude for reference features (overrides JSON lookup)
            airport_lon: Airport longitude for reference features (overrides JSON lookup)
            airports_json_path: Path to airports.json file for coordinate lookup
            airport_icao: ICAO code of airport to use (default: KSEA for SeaTac)
            time_step: Temporal resolution for graph snapshots (seconds)
            prediction_horizon: Time ahead to predict heading (seconds)
            max_distance_nm: Maximum distance for edge creation (nautical miles)
            transform: Optional transform to apply to each graph
            pre_transform: Optional pre-transform
            use_parquet: Whether to load data from parquet file instead of data_df
            parquet_path: Path to parquet file (if use_parquet=True)
        """

        # Load airport coordinates from JSON if provided
        if airports_json_path and (airport_lat is None or airport_lon is None):
            with open(airports_json_path, 'r') as f:
                airports_data = json.load(f)
            
            # Find airport by ICAO code
            airport_info = None
            for airport in airports_data['airports']:
                if airport['icao'] == airport_icao:
                    airport_info = airport
                    break
            
            if airport_info is None:
                raise ValueError(f"Airport with ICAO code '{airport_icao}' not found in {airports_json_path}")
            
            self.airport_lat = airport_info['latitude_deg']
            self.airport_lon = airport_info['longitude_deg']
            print(f"Loaded airport {airport_info['name']} ({airport_icao}): {self.airport_lat:.6f}, {self.airport_lon:.6f}")
        else:
            self.airport_lat = airport_lat
            self.airport_lon = airport_lon
        
        # Store other parameters
        self.time_step = time_step_s
        self.prediction_horizon = prediction_horizon_s
        self.max_distance_nm = max_distance_nm
        self.use_parquet = use_parquet
        self.parquet_path = parquet_path

        # Set root directory for PyTorch Geometric Dataset
        root_dir = None
        if use_parquet and parquet_path:
            root_dir = os.path.dirname(parquet_path)
        
        super().__init__(root_dir, transform, pre_transform)

        self.node_features = [
            "lat",
            "lon",
            "heading",
            "velocity",
            "geoaltitude",
            "vertrate",
        ]

        if use_parquet:
            # Try to load processed snapshots first
            processed_path = os.path.join(os.path.dirname(parquet_path), 'snapshots.pt')
            if os.path.exists(processed_path):
                print(f"Loading processed snapshots from {processed_path}")
                data = torch.load(processed_path)
                self.snapshots = data['snapshots']
                self.data_df = data['data_df']
                
                # Verify parameters match
                saved_params = data['params']
                if (saved_params['time_step'] != self.time_step or
                    saved_params['prediction_horizon'] != self.prediction_horizon or
                    saved_params['max_distance_nm'] != self.max_distance_nm):
                    print("Parameters changed, reprocessing...")
                    self.process()
            else:
                print("No cached snapshots found, processing from parquet...")
                self.process()
        else:
            # Traditional in-memory mode
            if data_df is None:
                raise ValueError("data_df must be provided when use_parquet=False")
            self.data_df = data_df.copy()
            
            # Create time-based snapshots - each time snapshot is a graph to be trained on 
            self.snapshots = self._create_snapshots()

    def _create_snapshots(self) -> List[Dict]:
        """
        Create discrete time snapshots of the airspace.

        Returns:
            List of snapshot dictionaries with timestamp and aircraft states
        """
        # Round times to time_step intervals
        self.data_df["snapshot_time"] = (
            (self.data_df["time"] // self.time_step) * self.time_step
        ).astype(int)

        snapshots = []
        for snap_time, group in self.data_df.groupby("snapshot_time"):
            # Only include snapshots with multiple aircraft
            if len(group) >= 2:
                snapshots.append(
                    {"time": snap_time, "data": group.reset_index(drop=True)}
                )

        return snapshots

    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.snapshots)

    def get(self, idx: int) -> Data:
        """
        Get a graph at a specific snapshot.

        Args:
            idx: Snapshot index

        Returns:
            PyTorch Geometric Data object representing the graph
        """
        snapshot = self.snapshots[idx]
        snap_df = snapshot["data"]
        snap_time = snapshot["time"]

        # Extract node features
        node_features = self._extract_node_features(snap_df)

        # Build edges based on spatial proximity
        edge_index, edge_attr = self._build_edges(snap_df)

        # Get labels (future headings)
        labels = self._get_labels(snap_df, snap_time)

        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels["heading_bins"],
            pos=torch.tensor(snap_df[["lat", "lon"]].values, dtype=torch.float),
            # Additional metadata
            time=torch.tensor([snap_time], dtype=torch.long),
            has_label=labels["has_label"],
            delta_heading=labels["delta_heading"],
        )

        return graph

    def _extract_node_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Extract and normalize node features from aircraft states.

        Args:
            df: DataFrame with aircraft states at a snapshot

        Returns:
            Tensor of shape [num_nodes, num_features]
        """
        features_list = []

        for _, row in df.iterrows():
            features = []

            # Position features (normalized lat/lon)
            lat_norm = (row["lat"] - self.airport_lat) / TERMINAL_RADIUS_DEG  
            lon_norm = (row["lon"] - self.airport_lon) / TERMINAL_RADIUS_DEG
            features.extend([lat_norm, lon_norm])

            # Heading as sin/cos to handle circularity
            heading_rad = np.radians(row["heading"])
            features.extend([np.sin(heading_rad), np.cos(heading_rad)])

            # Velocity (normalize by typical approach speed)
            velocity_norm = row.get("velocity", 150.0) / 200.0
            features.append(velocity_norm)

            # Altitude (normalize by typical approach altitude)
            altitude_norm = row.get("geoaltitude", 1000.0) / 3000.0
            features.append(altitude_norm)

            # Vertical rate (normalize)
            vertrate_norm = row.get("vertrate", 0.0) / 10.0  # m/s
            features.append(vertrate_norm)

            # Distance and bearing to airport
            dist_to_airport = get_distance_nm(
                self.airport_lat, self.airport_lon, row["lat"], row["lon"]
            )
            bearing_to_airport = get_azimuth_to_point(
                row["lat"], row["lon"], self.airport_lat, self.airport_lon
            )

            # Normalize distance by terminal area size
            dist_norm = dist_to_airport / TERMINAL_RADIUS_NM
            bearing_rad = np.radians(bearing_to_airport)
            features.extend([dist_norm, np.sin(bearing_rad), np.cos(bearing_rad)])

            features_list.append(features)

        return torch.tensor(features_list, dtype=torch.float)

    def _build_edges(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on spatial proximity between aircraft.

        Args:
            df: DataFrame with aircraft states

        Returns:
            edge_index: Tensor of shape [2, num_edges]
            edge_attr: Tensor of edge features [num_edges, num_edge_features]
        """
        num_nodes = len(df)
        edge_list = []
        edge_features = []

        # Build bidirectional directed graph - create both directions with proper features
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # j > i to avoid duplicate computation
                # Calculate distance between aircraft
                dist_nm = get_distance_nm(
                    df.iloc[i]["lat"],
                    df.iloc[i]["lon"],
                    df.iloc[j]["lat"],
                    df.iloc[j]["lon"],
                )

                # Only connect nearby aircraft, once found grab some other edge features
                if dist_nm <= self.max_distance_nm:
                    # adding directed edges going both ways
                    edge_list.extend([[i, j], [j, i]])

                    # Normalize distance (same for both directions)
                    dist_norm = dist_nm / self.max_distance_nm

                    # Edge features for i->j direction
                    bearing_ij = get_azimuth_to_point(
                        df.iloc[i]["lat"],
                        df.iloc[i]["lon"],
                        df.iloc[j]["lat"],
                        df.iloc[j]["lon"],
                    )
                    rel_heading_ij = ang_diff_deg(
                        df.iloc[j]["heading"], df.iloc[i]["heading"]
                    )
                    bearing_ij_rad = np.radians(bearing_ij)
                    rel_heading_ij_rad = np.radians(rel_heading_ij)

                    edge_feature_ij = [
                        dist_norm,
                        np.sin(bearing_ij_rad),
                        np.cos(bearing_ij_rad),
                        np.sin(rel_heading_ij_rad),
                        np.cos(rel_heading_ij_rad),
                    ]

                    # Edge features for j->i direction (inverted bearing and relative heading)
                    bearing_ji = get_azimuth_to_point(
                        df.iloc[j]["lat"],
                        df.iloc[j]["lon"],
                        df.iloc[i]["lat"],
                        df.iloc[i]["lon"],
                    )
                    rel_heading_ji = ang_diff_deg(
                        df.iloc[i]["heading"], df.iloc[j]["heading"]
                    )
                    bearing_ji_rad = np.radians(bearing_ji)
                    rel_heading_ji_rad = np.radians(rel_heading_ji)

                    edge_feature_ji = [
                        dist_norm,
                        np.sin(bearing_ji_rad),
                        np.cos(bearing_ji_rad),
                        np.sin(rel_heading_ji_rad),
                        np.cos(rel_heading_ji_rad),
                    ]
                    
                    # Add features for each direction
                    edge_features.append(edge_feature_ij)
                    edge_features.append(edge_feature_ji)

        if len(edge_list) == 0:
            # No edges - create empty tensors
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return edge_index, edge_attr

    # TODO: consider moving this logic to labeling.py
    def _get_labels(
        self, df: pd.DataFrame, current_time: int
    ) -> Dict[str, torch.Tensor]:
        """
        Get future heading labels for each aircraft.

        Args:
            df: Current aircraft states
            current_time: Current snapshot time

        Returns:
            Dictionary with label tensors
        """
        future_time = current_time + self.prediction_horizon
        labels = []
        has_label = []
        delta_headings = []

        for _, row in df.iterrows():
            # Look for future state of this aircraft
            future_state = self.data_df[
                (self.data_df["icao24"] == row["icao24"])
                & (self.data_df["time"] >= future_time - 2)  # Allow small tolerance
                & (self.data_df["time"] <= future_time + 2)
            ]

            if len(future_state) > 0:
                future_heading = future_state.iloc[0]["heading"]
                delta_heading = ang_diff_deg(future_heading, row["heading"])

                # Bin the heading (360 degrees / 5 degree bins = 72 classes)
                heading_bin = int(future_heading // 5) % 72

                labels.append(heading_bin)
                has_label.append(True)
                delta_headings.append(delta_heading)
            else:
                # make sure label masking is used otherwise
                # the -1 label will be treated as a valid class
                # has_label must be checked during training
                labels.append(-1)  # Invalid label
                has_label.append(False)
                delta_headings.append(0.0)

        return {
            "heading_bins": torch.tensor(labels, dtype=torch.long),
            "has_label": torch.tensor(has_label, dtype=torch.bool),
            "delta_heading": torch.tensor(delta_headings, dtype=torch.float),
        }


def create_train_val_test_split(
    dataset: AircraftGraphDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    temporal_split: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset into train/val/test sets.

    Args:
        dataset: AircraftGraphDataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        temporal_split: If True, split by time (earlier=train, later=test)

    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    n = len(dataset)
    indices = np.arange(n)

    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, "Train/val/test ratios must sum to 1."

    if temporal_split:
        # Sort by time for temporal split
        times = [dataset.snapshots[i]["time"] for i in indices]
        sorted_idx = np.argsort(times)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = sorted_idx[:train_end].tolist()
        val_idx = sorted_idx[train_end:val_end].tolist()
        test_idx = sorted_idx[val_end:].tolist()

        
    else:
        # Random split
        np.random.shuffle(indices)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end].tolist()
        val_idx = indices[train_end:val_end].tolist()
        test_idx = indices[val_end:].tolist()


    return train_idx, val_idx, test_idx
