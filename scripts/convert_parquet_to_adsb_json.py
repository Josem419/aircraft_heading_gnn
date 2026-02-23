#!/usr/bin/env python3
"""
Convert parquet files to ADS-B JSON format for playback.
Reads all parquet files and creates a JSON file with a list of ADS-B observations.
Filters data by airport location and distance.
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import glob
import numpy as np
from pyproj import Geod
from datetime import datetime, timezone

# WGS-84 ellipsoid for distance calculations
GEOD = Geod(ellps="WGS84")
METERS_PER_NM = 1852.0
NM_PER_METERS = 1.0 / METERS_PER_NM

def get_distance_nm(
    ref_lat_deg: float, ref_lon_deg: float, target_lat_deg: float, target_lon_deg: float
) -> float:
    """Returns great circle distance in nautical miles"""
    _, _, distance_m = GEOD.inv(
        ref_lon_deg, ref_lat_deg, target_lon_deg, target_lat_deg
    )
    return distance_m * NM_PER_METERS

def convert_row_to_adsb(row: pd.Series, relative_timestamp: float) -> Dict[str, Any]:
    """
    Convert a parquet row to ADS-B JSON format.
    
    Args:
        row: A pandas Series representing one aircraft state
        relative_timestamp: Relative timestamp in seconds from first observation
        
    Returns:
        Dictionary in ADS-B JSON format
    """
    # Convert heading to int if it's not NaN
    heading = int(row['heading']) if pd.notna(row['heading']) else None
    
    # Convert velocity (groundspeed) to knots - it's likely already in knots from OpenSky
    groundspeed = float(row['velocity']) if pd.notna(row['velocity']) else None
    
    # Convert vertical rate from m/s to feet per minute (1 m/s = 196.85 fpm)
    vertrate = row['vertrate']
    vertical_speed_fpm = (vertrate * 196.85) if pd.notna(vertrate) else None
    
    # Convert altitude from meters to feet (1 m = 3.28084 ft)
    geo_alt_ft = (row['geoaltitude'] * 3.28084) if pd.notna(row['geoaltitude']) else None
    baro_alt_ft = (row['baroaltitude'] * 3.28084) if pd.notna(row['baroaltitude']) else None
    
    # Convert UTC timestamp to ISO 8601 string
    utc_timestamp_s = int(row['time']) if pd.notna(row['time']) else 0
    try:
        utc_datetime = datetime.fromtimestamp(utc_timestamp_s, tz=timezone.utc)
        timestamp_utc_iso = utc_datetime.isoformat()
    except (ValueError, OSError):
        timestamp_utc_iso = None
    
    adsb_data = {
        "timestamp_utc_iso": timestamp_utc_iso,
        "aircraft_callsign": row['callsign'].strip() if pd.notna(row['callsign']) else None,
        "icao_address": row['icao24'] if pd.notna(row['icao24']) else None,
        "latitude_deg": float(row['lat']) if pd.notna(row['lat']) else None,
        "longitude_deg": float(row['lon']) if pd.notna(row['lon']) else None,
        "geometric_altitude_ft": float(geo_alt_ft) if geo_alt_ft is not None else None,
        "barometric_altitude_ft": float(baro_alt_ft) if baro_alt_ft is not None else None,
        "groundspeed_knots": float(groundspeed) if groundspeed is not None else None,
        "heading_deg": heading,
        "vertical_speed_fpm": float(vertical_speed_fpm) if vertical_speed_fpm is not None else None,
        "squawk_code": str(row['squawk']).strip() if pd.notna(row['squawk']) else None,
        "onground": bool(row['onground']) if pd.notna(row['onground']) else False,
        "last_position_update_utc_s": int(row['lastposupdate']) if pd.notna(row['lastposupdate']) else 0,
        "last_contact_utc_s": int(row['lastcontact']) if pd.notna(row['lastcontact']) else 0,
    }
    
    return {
        "timestamp_s": relative_timestamp,
        "data_type": "adsb",
        "data": adsb_data
    }


def convert_parquet_files_to_json(
    parquet_dir: str,
    output_file: str,
    airport_lat: float = None,
    airport_lon: float = None,
    radius_nm: float = 100.0,
    pattern: str = "*.parquet",
    limit: int = None,
    duration_hours: float = None
) -> None:
    """
    Convert all parquet files in a directory to a single JSON file.
    Optionally filters by airport location and distance.
    
    Args:
        parquet_dir: Directory containing parquet files
        output_file: Output JSON file path
        airport_lat: Airport latitude (optional, defaults to SeaTac)
        airport_lon: Airport longitude (optional, defaults to SeaTac)
        radius_nm: Radius around airport in nautical miles (default 100 nm)
        pattern: Glob pattern for parquet files
        limit: Optional limit on number of observations to process
        duration_hours: Optional duration in hours to limit data collection
    """
    # Default to SeaTac if not specified
    if airport_lat is None:
        airport_lat = 47.4498889
    if airport_lon is None:
        airport_lon = -122.3117778
    
    parquet_dir = Path(parquet_dir)
    parquet_files = sorted(parquet_dir.glob(pattern))
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"Filtering to {radius_nm} nm radius around airport ({airport_lat:.4f}, {airport_lon:.4f})")
    if duration_hours:
        print(f"Limiting to {duration_hours} hour(s) of data")
    
    observations = []
    start_timestamp = None
    max_timestamp = None
    
    if duration_hours:
        max_timestamp_delta = duration_hours * 3600.0  # Convert hours to seconds
    
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}...")
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Skip header row if it exists (where icao24 == 'icao24')
            df_clean = df[df['icao24'] != 'icao24'].copy()
            
            # Convert time column to numeric if needed
            df_clean['time'] = pd.to_numeric(df_clean['time'], errors='coerce')
            
            print(f"  - {len(df_clean)} rows")
            
            # Filter to airport area
            if airport_lat is not None and airport_lon is not None:
                # Calculate distances to airport
                valid = df_clean["lat"].notna() & df_clean["lon"].notna()
                distances = pd.Series(np.nan, index=df_clean.index)
                
                distances[valid] = df_clean.loc[valid].apply(
                    lambda row: get_distance_nm(airport_lat, airport_lon, row["lat"], row["lon"]),
                    axis=1,
                )
                
                # Filter by distance
                mask = valid & (distances <= radius_nm)
                df_clean = df_clean[mask].copy()
                print(f"  - After airport filter ({radius_nm} nm): {len(df_clean):,} records")
            
            for _, row in df_clean.iterrows():
                # Set start timestamp from first valid observation
                if start_timestamp is None and pd.notna(row.get('time')):
                    start_timestamp = row['time']
                    if duration_hours:
                        max_timestamp = start_timestamp + max_timestamp_delta
                        print(f"  - Start timestamp: {start_timestamp}, Max: {max_timestamp}")
                
                # Check if we should stop based on time duration
                if duration_hours and max_timestamp:
                    current_time = row.get('time', 0)
                    if pd.notna(current_time) and current_time > max_timestamp:
                        print(f"Reached duration limit of {duration_hours} hour(s)")
                        break
                
                # Check if we should stop based on observation limit
                if limit and len(observations) >= limit:
                    break
                
                try:
                    # Calculate relative timestamp from start
                    current_time = row.get('time', start_timestamp)
                    if pd.notna(current_time) and start_timestamp is not None:
                        relative_ts = current_time - start_timestamp
                    else:
                        relative_ts = 0.0
                    
                    obs = convert_row_to_adsb(row, relative_ts)
                    observations.append(obs)
                except Exception as e:
                    pass  # Skip problematic rows
            
            # Check if we should stop after processing this file
            if (duration_hours and max_timestamp and len(observations) > 0) or \
               (limit and len(observations) >= limit):
                if duration_hours and max_timestamp:
                    break
                
        except Exception as e:
            print(f"  - Error reading {parquet_file.name}: {e}")
            continue
    
    print(f"\nTotal observations created: {len(observations)}")
    
    # Write to JSON file with observations wrapped in a structure
    print(f"Writing to {output_file}...")
    output_data = {
        "observations": observations
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Conversion complete! Wrote {len(observations)} observations to {output_file}")


if __name__ == "__main__":
    import sys
    
    # Use raw directory by default, but can be customized
    parquet_dir = "/home/jose/workspaces/aircraft_heading_gnn/data/processed/raw"
    output_file = "/home/jose/workspaces/aircraft_heading_gnn/data/seatac_adsb_states.json"
    
    # Optional parameters
    limit = None
    duration_hours = 1.0  # Default to 1 hour
    
    # SeaTac coordinates
    airport_lat = 47.4498889
    airport_lon = -122.3117778
    radius_nm = 100.0  # 100 nm radius
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    if len(sys.argv) > 2:
        duration_hours = float(sys.argv[2])
    if len(sys.argv) > 3:
        radius_nm = float(sys.argv[3])
    
    convert_parquet_files_to_json(
        parquet_dir, 
        output_file, 
        airport_lat=airport_lat,
        airport_lon=airport_lon,
        radius_nm=radius_nm,
        limit=limit,
        duration_hours=duration_hours
    )
