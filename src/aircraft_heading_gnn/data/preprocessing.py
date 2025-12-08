"""
Data preprocessing utilities for ADS-B trajectories.
Handles loading, cleaning, and filtering of OpenSky ADS-B data.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import json
from pathlib import Path

from aircraft_heading_gnn.data.labeling import compute_future_heading_bins
from aircraft_heading_gnn.utils.adsb_features import get_distance_nm
from aircraft_heading_gnn.utils.angles import ang_diff_deg

# Constants
METERS_TO_FEET = 3.28084
KTS_PER_MPS = 1.94384 
DEFAULT_TERMINAL_RADIUS_NM = 40.0
DEFAULT_MAX_ALTITUDE_FT = 18000.0
DEFAULT_MIN_TRAJECTORY_POINTS = 30
DEFAULT_MAX_GAP_SECONDS = 60.0
DEFAULT_MIN_SPEED_KTS = 50.0
DEFAULT_MAX_SPEED_KTS = 600.0
DEFAULT_PREDICTION_HORIZON_S = 15
DEFAULT_HEADING_BIN_SIZE = 5
DEFAULT_WINDOW_SIZE = 10
DEFAULT_WINDOW_STRIDE = 1

# OpenSky state vector columns (from README.txt)
# time,icao24,lat,lon,velocity,heading,vertrate,callsign,onground,alert,spi,squawk,baroaltitude,geoaltitude,lastposupdate,lastcontact
ADSB_COLUMNS = [
    "time",
    "icao24",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "callsign",
    "onground",
    "alert",
    "spi",
    "squawk",
    "baroaltitude",
    "geoaltitude",
    "lastposupdate",
    "lastcontact",
]


def load_adsb_csv(filepath: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load ADS-B CSV data from OpenSky format.

    Args:
        filepath: Path to CSV file
        columns: Specific columns to load (None = all)

    Returns:
        DataFrame with ADS-B data
    """
    if columns is None:
        columns = ADSB_COLUMNS

    # opensky data has no header so dont use header row
    df = pd.read_csv(filepath, names=ADSB_COLUMNS, usecols=columns, header=None)
    return df


def filter_terminal_area(
    df: pd.DataFrame,
    airport_lat: float,
    airport_lon: float,
    radius_nm: float = DEFAULT_TERMINAL_RADIUS_NM,
    max_altitude_ft: float = DEFAULT_MAX_ALTITUDE_FT,
) -> pd.DataFrame:
    """
    Filter data to terminal area around an airport.

    Args:
        df: ADS-B DataFrame
        airport_lat: Airport latitude in degrees
        airport_lon: Airport longitude in degrees
        radius_nm: Radius around airport in nautical miles
        max_altitude_ft: Maximum altitude in feet

    Returns:
        Filtered DataFrame
    """

    # remove rows with missing lat/lon
    valid = df["lat"].notna() & df["lon"].notna()
    distances = pd.Series(np.nan, index=df.index)
    distances[valid] = df.loc[valid].apply(
        lambda row: get_distance_nm(airport_lat, airport_lon, row["lat"], row["lon"]),
        axis=1,
    )

    # Filter by distance and altitude
    mask = valid & (distances <= radius_nm)

    if "geoaltitude" in df.columns:
        # Convert meters to feet (geoaltitude is in meters)
        altitude_ft = df["geoaltitude"] * METERS_TO_FEET
        mask &= altitude_ft <= max_altitude_ft

    return df[mask].copy()


def clean_trajectories(
    df: pd.DataFrame,
    min_points: int = DEFAULT_MIN_TRAJECTORY_POINTS,
    max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
    speed_filter: bool = True,
    min_speed_kts: float = DEFAULT_MIN_SPEED_KTS,
    max_speed_kts: float = DEFAULT_MAX_SPEED_KTS,
) -> pd.DataFrame:
    """
    Clean and filter ADS-B trajectories.

    Args:
        df: ADS-B DataFrame
        min_points: Minimum points per trajectory
        max_gap_seconds: Maximum time gap within trajectory
        speed_filter: Whether to filter by speed
        min_speed_kts: Minimum speed in knots
        max_speed_kts: Maximum speed in knots

    Returns:
        Cleaned DataFrame
    """
    df = df.copy()

    # Remove ground positions
    if "onground" in df.columns:
        df = df[df["onground"] == False].copy()

    # Remove missing critical fields
    df = df.dropna(subset=["time", "icao24", "lat", "lon", "heading"])

    # Speed filtering  
    if speed_filter and 'velocity' in df.columns:
        # open sky velocity is in m/s, need to convert to knots
        # to compare with input kts thresholds
        # using knots because it's more common in aviation
        speed_kts = df['velocity'] * KTS_PER_MPS
        df = df[(speed_kts >= min_speed_kts) & (speed_kts <= max_speed_kts)]


    # Sort by aircraft and time
    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)

    # Split trajectories at large time gaps
    df["time_diff"] = df.groupby("icao24")["time"].diff()
    df["trajectory_id"] = (
        (df["time_diff"] > max_gap_seconds) | (df["icao24"] != df["icao24"].shift(1))
    ).cumsum()

    # Filter trajectories by minimum length
    traj_counts = df.groupby("trajectory_id").size()
    valid_trajs = traj_counts[traj_counts >= min_points].index
    df = df[df["trajectory_id"].isin(valid_trajs)].copy()

    # Drop temporary column
    df = df.drop(columns=["time_diff"])

    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw ADS-B data.

    Args:
        df: Cleaned ADS-B DataFrame with trajectory_id

    Returns:
        DataFrame with additional computed features
    """
    df = df.copy()

    # Group by trajectory for temporal features
    for trajectory_id, group in df.groupby("trajectory_id"):
        idx = group.index

        # Time since start of trajectory
        df.loc[idx, "time_in_trajectory"] = group["time"] - group["time"].iloc[0]

        # Compute heading change rate (degrees per second)
        if len(group) > 1:
            time_diff = group["time"].diff()
            heading_diff = group["heading"].diff()

            heading_diff_circ = heading_diff.apply(
                lambda x: ang_diff_deg(x, 0) if pd.notna(x) else np.nan
            )

            # protect for division by zero
            valid = time_diff > 0
            turn_rate = pd.Series(np.nan, index=group.index)
            turn_rate[valid] = heading_diff_circ[valid] / time_diff[valid]
            df.loc[idx, 'turn_rate'] = turn_rate

        # Altitude rate (if available)
        if "geoaltitude" in df.columns and len(group) > 1:
            time_diff = group["time"].diff()
            alt_diff = group["geoaltitude"].diff()

            # protect for division by zero
            valid = time_diff > 0
            altitude_rate = pd.Series(np.nan, index=group.index)
            altitude_rate[valid] = alt_diff[valid] / time_diff[valid]
            df.loc[idx, "altitude_rate"] = altitude_rate

    return df


def load_airport_data(filepath: str = None) -> pd.DataFrame:
    """
    Load airport reference data.

    Args:
        filepath: Path to airports.json (defaults to repo location)

    Returns:
        DataFrame with airport information
    """
    if filepath is None:
        # Default to repo data folder
        repo_root = Path(__file__).parent.parent.parent.parent
        filepath = repo_root / "data" / "airports.json"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return pd.DataFrame(data["airports"])


def prepare_training_data(
    df: pd.DataFrame,
    delta_s: int = DEFAULT_PREDICTION_HORIZON_S,
    bin_size: int = DEFAULT_HEADING_BIN_SIZE,
    min_turn_deg: Optional[float] = None,
) -> pd.DataFrame:
    """
    Prepare data for training with future heading labels.

    Args:
        df: Preprocessed ADS-B DataFrame
        delta_s: Time delta for prediction (seconds)
        bin_size: Heading bin size in degrees
        min_turn_deg: Minimum turn angle to consider as maneuver

    Returns:
        DataFrame with labels ready for training
    """

    labeled_df = compute_future_heading_bins(
        df, delta_s=delta_s, bin_size=bin_size, min_turn_deg=min_turn_deg
    )

    # Only keep samples with valid labels
    training_df = labeled_df[labeled_df["has_label"]].copy()

    return training_df


def create_time_windows(
    df: pd.DataFrame,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> List[pd.DataFrame]:
    """
    Create sliding time windows from trajectories for sequence modeling.

    Args:
        df: Preprocessed trajectory DataFrame with trajectory_id
        window_size: Number of timesteps per window
        stride: Stride between windows

    Returns:
        List of DataFrames, one per window
    """
    windows = []

    for trajectory_id, group in df.groupby("trajectory_id"):
        group = group.sort_values("time").reset_index(drop=True)

        for i in range(0, len(group) - window_size + 1, stride):
            window = group.iloc[i : i + window_size].copy()
            window["window_id"] = len(windows)
            windows.append(window)

    return windows
