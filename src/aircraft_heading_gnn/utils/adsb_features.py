"""Utilities for computing features from ADS-B data."""

import numpy as np
import pandas as pd
from pyproj import Geod

from typing import Optional, Dict, Any, Tuple

from utils.angles import wrap_deg, ang_diff_deg, circ_distance_deg, normalize_angle

# WGS-84 ellipsoid
GEOD = Geod(ellps="WGS84")
METERS_PER_NM = 1852.0
NM_PER_METERS = 1.0 / METERS_PER_NM


def get_distance_nm(
    ref_lat_deg: float, ref_lon_deg: float, target_lat_deg: float, targer_lon_deg: float
) -> float:
    """ Returns great circle distance in nautical miles """
    _, _, distance_m = GEOD.inv(ref_lon_deg, ref_lat_deg, targer_lon_deg, target_lat_deg)  # meters
    return distance_m * NM_PER_METERS


def get_azimuth_to_point(
    ref_lat_deg: float, ref_lon_deg: float, target_lat_deg: float, targer_lon_deg: float
) -> float:
    """ Returns forward azimuth in degrees from ref to target. Normalized to [-180, 180)."""
    fwd_azimuth_deg, _, _ = GEOD.inv(
        ref_lon_deg, ref_lat_deg, targer_lon_deg, target_lat_deg
    )  # degrees
    return normalize_angle(fwd_azimuth_deg)

def get_relative_heading_deg(
    heading_deg: float, target_bearing_deg: float
) -> float:
    """ Returns relative heading in degrees from current heading to target bearing. Normalized to [-180, 180)."""
    return ang_diff_deg(heading_deg, target_bearing_deg)


def get_distance_and_bearing_to_point(
    ref_lat_deg: float, ref_lon_deg: float, target_lat_deg: float, target_lon_deg: float
) -> Tuple[float, float]:
    """ Returns distance in nautical miles and bearing in degrees from ref to target. Bearing normalized to [-180, 180)."""
    fwd_azimuth_deg, _, distance_m = GEOD.inv(
        ref_lon_deg, ref_lat_deg, target_lon_deg, target_lat_deg
    )  # degrees and meters
    distance_nm = distance_m * NM_PER_METERS
    bearing_deg = normalize_angle(fwd_azimuth_deg)
    return distance_nm, bearing_deg

def get_closure_rate(
    ref_lat_deg: float, ref_lon_deg: float, target_lat_deg: float, target_lon_deg: float, velocity_kts: float
) -> float:
    """ Returns closure rate in knots towards the target point. Positive if closing, negative if opening."""
    _, bearing_deg = get_distance_and_bearing_to_point(
        ref_lat_deg, ref_lon_deg, target_lat_deg, target_lon_deg
    )
    relative_heading_deg = get_relative_heading_deg(bearing_deg, 0.0)  # Assuming heading is 0 for closure rate
    closure_rate_kts = velocity_kts * np.cos(np.radians(relative_heading_deg))
    return closure_rate_kts
