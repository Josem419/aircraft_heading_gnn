"""
PyTorch Geometric Dataset for aircraft trajectory graphs.
Constructs spatial-temporal graphs from ADS-B data for heading prediction.
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset

from aircraft_heading_gnn.utils.adsb_features import (
    get_distance_nm,
    get_azimuth_to_point,
)
from aircraft_heading_gnn.utils.angles import ang_diff_deg
from aircraft_heading_gnn.data.preprocessing import (
    filter_terminal_area,
    clean_trajectories,
    compute_derived_features,
)

# Terminal area radius based on default terminal radius in nautical miles
TERMINAL_RADIUS_NM = 180.0
TERMINAL_RADIUS_DEG = TERMINAL_RADIUS_NM / 60.0
MAX_DISTANCE_BETWEEN_AIRCRAFT_NM = 35.0  # for edge creation
PREDICTION_HORIZON_SEC = 15  # seconds ahead to predict heading
TIME_BETWEEN_SNAPSHOTS_SEC = 10  # seconds between graph snapshots
INCLUDE_AIRPORT_NODE = True  # Add airport as virtual hub node


class AircraftGraphDataset(Dataset):
    """
    Dataset that creates graph snapshots from aircraft trajectories.

    Each graph represents aircraft in a terminal area at a specific time:
    - Nodes: Individual aircraft with their state features
    - Edges: Spatial relationships (proximity, bearing)
    - Target: Future heading for each aircraft
    """

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
        batch_mode: bool = False,
        max_files: Optional[int] = None,
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
            pre_transform: Optional pre-transform (not used currently)
            use_parquet: Whether to load data from parquet file instead of data_df
            parquet_path: Path to parquet file or directory (if use_parquet=True)
            batch_mode: If True and parquet_path is directory, load multiple files
            max_files: Maximum number of parquet files to process (None = all)
        """
        super().__init__()
        self.transform = transform
        self.pre_transform = pre_transform

        # Load airport coordinates from JSON if provided
        if airports_json_path and (airport_lat is None or airport_lon is None):
            with open(airports_json_path, "r") as f:
                airports_data = json.load(f)

            airport_info = None
            for airport in airports_data["airports"]:
                if airport["icao"] == airport_icao:
                    airport_info = airport
                    break

            if airport_info is None:
                raise ValueError(
                    f"Airport with ICAO code '{airport_icao}' not found in {airports_json_path}"
                )

            self.airport_lat = airport_info["latitude_deg"]
            self.airport_lon = airport_info["longitude_deg"]
            print(
                f"Loaded airport {airport_info['name']} ({airport_icao}): "
                f"{self.airport_lat:.6f}, {self.airport_lon:.6f}"
            )
        else:
            self.airport_lat = airport_lat
            self.airport_lon = airport_lon

        # Ensure we have valid airport coordinates
        if self.airport_lat is None or self.airport_lon is None:
            raise ValueError(
                "Airport coordinates must be provided or airports_json_path must be set."
            )

        # Store other parameters
        self.time_step = time_step_s
        self.prediction_horizon = prediction_horizon_s
        self.max_distance_nm = max_distance_nm
        self.use_parquet = use_parquet
        self.parquet_path = parquet_path
        self.batch_mode = batch_mode
        self.max_files = max_files

        # Batch-mode bookkeeping
        if use_parquet and parquet_path and batch_mode:
            if not os.path.isdir(parquet_path):
                raise ValueError(
                    f"batch_mode=True but parquet_path '{parquet_path}' is not a directory"
                )
            self._batch_datasets: Optional[List[AircraftGraphDataset]] = []
            self._cumulative_lengths: Optional[List[int]] = [0]
        else:
            self._batch_datasets = None
            self._cumulative_lengths = None

        self.node_features = [
            "lat",
            "lon",
            "heading",
            "velocity",
            "geoaltitude",
            "vertrate",
        ]

        # Main data setup
        if use_parquet:
            if batch_mode:
                # Batch mode: collection of mini-datasets
                self._process_batch_mode()
            else:
                # Single parquet file -> one dataset
                if parquet_path is None:
                    raise ValueError(
                        "parquet_path must be provided when use_parquet=True"
                    )
                self._process_single_mode()
        else:
            # In-memory DataFrame mode
            if data_df is None:
                raise ValueError("data_df must be provided when use_parquet=False")
            self.data_df = data_df.copy()
            self.snapshots = self._create_snapshots()

        # Cache graphs only for non-batch mode
        if not self.batch_mode:
            self._graphs: Optional[List[Optional[Data]]] = [None] * len(self.snapshots)
        else:
            self._graphs = None

    # --------- parquet processing helpers ---------

    def _process_single_mode(self):
        """Process a single parquet file into snapshots."""
        parquet_file = self.parquet_path
        print(f"Loading parquet file: {parquet_file}")
        df = pd.read_parquet(parquet_file)

        # Remove rows with missing critical data
        initial_len = len(df)
        df = df.dropna(subset=["time", "icao24", "lat", "lon", "heading"])
        print(
            f"    Loaded {initial_len:,} records, {len(df):,} after removing NaN values"
        )

        if len(df) == 0:
            raise ValueError(
                "No valid records after NaN removal in single parquet mode"
            )

        # Filter to terminal area
        df = filter_terminal_area(
            df,
            self.airport_lat,
            self.airport_lon,
            radius_nm=TERMINAL_RADIUS_NM,
            max_altitude_ft=18000.0,
        )
        print(f"    After terminal area filter: {len(df):,} records")

        if len(df) == 0:
            raise ValueError(
                "No records left after terminal area filter in single mode"
            )

        # Create trajectory IDs
        df = df.sort_values(["icao24", "time"]).reset_index(drop=True)
        df["time_diff"] = df.groupby("icao24")["time"].diff()
        df["trajectory_id"] = (
            (df["time_diff"] > 120.0) | (df["icao24"] != df["icao24"].shift(1))
        ).cumsum()
        df = df.drop(columns=["time_diff"])

        # Compute derived features
        try:
            df = compute_derived_features(df)
            print(f"    After computing features: {len(df):,} records")
        except Exception as e:
            print(f"    Warning: Feature computation failed in single mode ({e})")

        self.data_df = df
        self.snapshots = self._create_snapshots()
        print(f"    Created {len(self.snapshots)} snapshots from single parquet file")

    def _process_batch_mode(self):
        """Process multiple parquet files in batch mode."""
        pattern = os.path.join(self.parquet_path, "*.parquet")
        parquet_files = sorted(glob.glob(pattern))

        if self.max_files:
            parquet_files = parquet_files[: self.max_files]

        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.parquet_path}")

        print(f"Found {len(parquet_files)} parquet files")

        for i, parquet_file in enumerate(parquet_files):
            print(
                f"\n[{i+1}/{len(parquet_files)}] Processing {os.path.basename(parquet_file)}..."
            )
            try:
                mini_dataset = self._create_dataset_from_parquet(parquet_file)
                if mini_dataset is not None and len(mini_dataset.snapshots) > 0:
                    self._batch_datasets.append(mini_dataset)
                    self._cumulative_lengths.append(
                        self._cumulative_lengths[-1] + len(mini_dataset.snapshots)
                    )
                    print(
                        f"  Created dataset with {len(mini_dataset.snapshots)} snapshots"
                    )
                else:
                    print(
                        f"  No valid snapshots created from {os.path.basename(parquet_file)}"
                    )
            except Exception as e:
                print(f"  Error processing {os.path.basename(parquet_file)}: {e}")

        if not self._batch_datasets:
            raise ValueError("No valid datasets created from parquet files")

        print(f"\nSuccessfully created {len(self._batch_datasets)} datasets")
        print(f"Total snapshots: {self._cumulative_lengths[-1]}")

    def _create_dataset_from_parquet(self, parquet_file: str):
        """Create a mini-dataset from a single parquet file."""
        df = pd.read_parquet(parquet_file)

        # Remove rows with missing critical data
        initial_len = len(df)
        df = df.dropna(subset=["time", "icao24", "lat", "lon", "heading"])
        print(
            f"    Loaded {initial_len:,} records, {len(df):,} after removing NaN values"
        )

        if len(df) == 0:
            print("    No valid records after NaN removal")
            return None

        # Filter to terminal area
        df = filter_terminal_area(
            df,
            self.airport_lat,
            self.airport_lon,
            radius_nm=TERMINAL_RADIUS_NM,
            max_altitude_ft=18000.0,
        )
        print(f"    After terminal area filter: {len(df):,} records")

        if len(df) == 0:
            print("    No records left after terminal area filter")
            return None

        # Create trajectory IDs manually
        df = df.sort_values(["icao24", "time"]).reset_index(drop=True)
        df["time_diff"] = df.groupby("icao24")["time"].diff()
        df["trajectory_id"] = (
            (df["time_diff"] > 120.0) | (df["icao24"] != df["icao24"].shift(1))
        ).cumsum()
        df = df.drop(columns=["time_diff"])

        # Compute derived features
        try:
            df = compute_derived_features(df)
            print(f"    After computing features: {len(df):,} records")
        except Exception as e:
            print(f"    Warning: Feature computation failed ({e})")

        mini_dataset = AircraftGraphDataset(
            data_df=df,
            airport_lat=self.airport_lat,
            airport_lon=self.airport_lon,
            time_step_s=self.time_step,
            prediction_horizon_s=self.prediction_horizon,
            max_distance_nm=self.max_distance_nm,
            use_parquet=False,  # Use DataFrame mode
        )

        return mini_dataset

    # --------- snapshot creation & indexing ---------

    def _create_snapshots(self) -> List[Dict]:
        """
        Create discrete time snapshots of the airspace.

        Returns:
            List of snapshot dictionaries with timestamp and aircraft states
        """
        self.data_df["snapshot_time"] = (
            (self.data_df["time"] // self.time_step) * self.time_step
        ).astype(int)

        snapshots: List[Dict] = []
        for snap_time, group in self.data_df.groupby("snapshot_time"):
            # Only include snapshots with multiple aircraft
            if len(group) >= 2:
                snapshots.append(
                    {"time": snap_time, "data": group.reset_index(drop=True)}
                )

        return snapshots

    def __len__(self) -> int:
        """Return the number of graphs in the dataset."""
        if self.batch_mode and self._batch_datasets is not None:
            return self._cumulative_lengths[-1] if self._cumulative_lengths else 0
        else:
            return len(self.snapshots)

    def __getitem__(self, idx: int) -> Data:
        """
        Get a graph at a specific snapshot.

        Args:
            idx: Snapshot index

        Returns:
            PyTorch Geometric Data object representing the graph
        """
        if self.batch_mode and self._batch_datasets is not None:
            # Find which mini-dataset contains this index
            dataset_idx = 0
            for i, cum_len in enumerate(self._cumulative_lengths[1:]):
                if idx < cum_len:
                    dataset_idx = i
                    break

            local_idx = idx - self._cumulative_lengths[dataset_idx]
            return self._batch_datasets[dataset_idx][local_idx]

        # Single dataset mode with caching
        if hasattr(self, "_graphs") and self._graphs[idx] is not None:
            return self._graphs[idx]

        snapshot = self.snapshots[idx]
        snap_df = snapshot["data"]
        snap_time = snapshot["time"]

        node_features = self._extract_node_features(snap_df, snap_time)
        edge_index, edge_attr = self._build_edges(snap_df)
        labels = self._get_labels(snap_df, snap_time)

        positions = torch.tensor(snap_df[["lat", "lon"]].values, dtype=torch.float)
        if INCLUDE_AIRPORT_NODE:
            airport_pos = torch.tensor(
                [[self.airport_lat, self.airport_lon]], dtype=torch.float
            )
            positions = torch.cat([positions, airport_pos], dim=0)

        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels["heading_bins"],
            pos=positions,
            time=torch.tensor([snap_time], dtype=torch.long),
            has_label=labels["has_label"],
            delta_heading=labels["delta_heading"],
        )

        if hasattr(self, "_graphs"):
            self._graphs[idx] = graph

        return graph

    # --------- feature / edge / label construction ---------

    def _extract_node_features(self, df: pd.DataFrame, snap_time: int) -> torch.Tensor:
        """
        Extract and normalize node features from aircraft states.

        Args:
            df: DataFrame with aircraft states at a snapshot
            snap_time: Snapshot center time for temporal feature calculation

        Returns:
            Tensor of shape [num_nodes, num_features] (11D including temporal features)
        """
        features_list = []

        # Temporal normalization based on actual time range in this snapshot
        if len(df) > 1:
            time_min = df["time"].min()
            time_max = df["time"].max()
            time_range = max(time_max - time_min, 1.0)  # Avoid division by zero
        else:
            time_range = 1.0  # Single aircraft, no temporal variation

        for _, row in df.iterrows():
            features = []

            # Position features (normalized lat/lon)
            lat_norm = (row["lat"] - self.airport_lat) / TERMINAL_RADIUS_DEG
            lon_norm = (row["lon"] - self.airport_lon) / TERMINAL_RADIUS_DEG
            features.extend([lat_norm, lon_norm])

            # Temporal feature - time offset from snapshot center
            time_offset_s = row["time"] - snap_time
            if time_range > 1.0:
                time_offset_norm = (2.0 * time_offset_s) / time_range  # [-1, +1]
            else:
                time_offset_norm = 0.0
            features.append(time_offset_norm)

            # Heading as sin/cos
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

            dist_norm = dist_to_airport / TERMINAL_RADIUS_NM
            bearing_rad = np.radians(bearing_to_airport)
            features.extend([dist_norm, np.sin(bearing_rad), np.cos(bearing_rad)])

            features_list.append(features)

        # Add airport node features if enabled
        if INCLUDE_AIRPORT_NODE:
            airport_features = [
                0.0,
                0.0,  # normalized position (0,0)
                0.0,  # temporal offset
                0.0,
                0.0,  # heading (sin, cos)
                0.0,  # velocity
                0.0,  # altitude
                0.0,  # vertical rate
                0.0,  # distance
                0.0,
                1.0,  # bearing (north)
            ]
            features_list.append(airport_features)

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
        edge_list: List[List[int]] = []
        edge_features: List[List[float]] = []

        # Aircraft-aircraft edges
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist_nm = get_distance_nm(
                    df.iloc[i]["lat"],
                    df.iloc[i]["lon"],
                    df.iloc[j]["lat"],
                    df.iloc[j]["lon"],
                )

                if dist_nm <= self.max_distance_nm:
                    edge_list.extend([[i, j], [j, i]])

                    dist_norm = dist_nm / self.max_distance_nm

                    # i -> j
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

                    # j -> i
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

                    edge_features.append(edge_feature_ij)
                    edge_features.append(edge_feature_ji)

        # Airport connections
        if INCLUDE_AIRPORT_NODE:
            airport_node_idx = num_nodes
            for i in range(num_nodes):
                dist_to_airport = get_distance_nm(
                    self.airport_lat,
                    self.airport_lon,
                    df.iloc[i]["lat"],
                    df.iloc[i]["lon"],
                )
                bearing_to_airport = get_azimuth_to_point(
                    df.iloc[i]["lat"],
                    df.iloc[i]["lon"],
                    self.airport_lat,
                    self.airport_lon,
                )

                dist_norm = dist_to_airport / TERMINAL_RADIUS_NM
                bearing_rad = np.radians(bearing_to_airport)

                # Aircraft -> Airport
                edge_list.append([i, airport_node_idx])
                edge_features.append(
                    [dist_norm, np.sin(bearing_rad), np.cos(bearing_rad), 0.0, 0.0]
                )

                # Airport -> Aircraft
                bearing_from_airport = (bearing_to_airport + 180) % 360
                bearing_from_rad = np.radians(bearing_from_airport)

                edge_list.append([airport_node_idx, i])
                edge_features.append(
                    [
                        dist_norm,
                        np.sin(bearing_from_rad),
                        np.cos(bearing_from_rad),
                        0.0,
                        0.0,
                    ]
                )

        if len(edge_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)

        return edge_index, edge_attr

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
        labels: List[int] = []
        has_label: List[bool] = []
        delta_headings: List[float] = []

        for _, row in df.iterrows():
            time_tolerance = 5  # seconds
            future_state = self.data_df[
                (self.data_df["icao24"] == row["icao24"])
                & (self.data_df["time"] >= future_time - time_tolerance)
                & (self.data_df["time"] <= future_time + time_tolerance)
            ]

            if len(future_state) > 0:
                future_heading = future_state.iloc[0]["heading"]
                delta_heading = ang_diff_deg(future_heading, row["heading"])

                heading_bin = int(future_heading // 5) % 72

                labels.append(heading_bin)
                has_label.append(True)
                delta_headings.append(delta_heading)
            else:
                labels.append(-1)
                has_label.append(False)
                delta_headings.append(0.0)

        if INCLUDE_AIRPORT_NODE:
            labels.append(-1)
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

    Supports both single-dataset and batch-mode datasets.
    In temporal_split mode, snapshots are ordered by their 'time' field.
    """
    n = len(dataset)
    indices = np.arange(n)

    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, "Train/val/test ratios must sum to 1."

    if temporal_split:
        if (
            getattr(dataset, "batch_mode", False)
            and dataset.batch_mode
            and dataset._batch_datasets is not None
        ):
            # Flatten times from each mini-dataset using cumulative offsets
            times = np.empty(n, dtype=np.int64)
            for ds_idx, mini_ds in enumerate(dataset._batch_datasets):
                offset = dataset._cumulative_lengths[ds_idx]
                for local_idx, snap in enumerate(mini_ds.snapshots):
                    global_idx = offset + local_idx
                    times[global_idx] = snap["time"]
        else:
            times = np.array([dataset.snapshots[i]["time"] for i in indices])

        sorted_idx = np.argsort(times)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = sorted_idx[:train_end].tolist()
        val_idx = sorted_idx[train_end:val_end].tolist()
        test_idx = sorted_idx[val_end:].tolist()
    else:
        np.random.shuffle(indices)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_idx = indices[:train_end].tolist()
        val_idx = indices[train_end:val_end].tolist()
        test_idx = indices[val_end:].tolist()

    return train_idx, val_idx, test_idx
