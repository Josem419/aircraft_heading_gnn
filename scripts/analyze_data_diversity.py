#!/usr/bin/env python3
"""
Analyze data diversity - check unique aircraft, time coverage, etc.
Usage: python scripts/analyze_data_diversity.py [--parquet_path PATH]
"""

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def main():
    parser = argparse.ArgumentParser(description="Analyze data diversity")
    parser.add_argument(
        "--parquet_path",
        default="data/processed/out.parquet",
        help="Path to parquet file",
    )

    args = parser.parse_args()

    print(f"Analyzing data diversity in {args.parquet_path}")
    print("=" * 60)

    # Load data
    df = pd.read_parquet(args.parquet_path)

    # Basic stats
    print(f"Total records: {len(df):,}")
    print(f"Unique aircraft (ICAO24): {df['icao24'].nunique():,}")
    print(f"Records per aircraft (avg): {len(df) / df['icao24'].nunique():.1f}")

    # Time coverage
    if "time" in df.columns:
        min_time = df["time"].min()
        max_time = df["time"].max()
        duration = max_time - min_time

        print(f"\nTime Coverage:")
        print(
            f"Start: {datetime.fromtimestamp(min_time).strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"End: {datetime.fromtimestamp(max_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration / 3600:.1f} hours")

        # Records per hour
        records_per_hour = len(df) / (duration / 3600)
        print(f"Records per hour: {records_per_hour:.0f}")

    # Aircraft distribution
    print(f"\nAircraft Distribution:")
    aircraft_counts = df["icao24"].value_counts()
    print(
        f"Most active aircraft: {aircraft_counts.iloc[0]:,} records ({aircraft_counts.index[0]})"
    )
    print(f"Median records per aircraft: {aircraft_counts.median():.0f}")
    print(f"Aircraft with >100 records: {(aircraft_counts > 100).sum():,}")
    print(f"Aircraft with >500 records: {(aircraft_counts > 500).sum():,}")

    # Geographic spread
    if "lat" in df.columns and "lon" in df.columns:
        print(f"\nGeographic Coverage:")
        print(f"Latitude range: {df['lat'].min():.4f} to {df['lat'].max():.4f}")
        print(f"Longitude range: {df['lon'].min():.4f} to {df['lon'].max():.4f}")

        # Calculate approximate area coverage
        lat_span = df["lat"].max() - df["lat"].min()
        lon_span = df["lon"].max() - df["lon"].min()
        print(f"Approximate area: {lat_span:.2f}° × {lon_span:.2f}°")

    # Velocity/altitude distribution
    if "velocity" in df.columns:
        print(f"\nVelocity Distribution:")
        print(f"Range: {df['velocity'].min():.0f} to {df['velocity'].max():.0f} kts")
        print(f"Mean: {df['velocity'].mean():.0f} kts")

    if "geoaltitude" in df.columns:
        # Convert to feet
        alt_ft = df["geoaltitude"] * 3.28084
        print(f"\nAltitude Distribution:")
        print(f"Range: {alt_ft.min():.0f} to {alt_ft.max():.0f} ft")
        print(f"Mean: {alt_ft.mean():.0f} ft")

    # Recommendations
    print(f"\nRecommendations:")
    unique_aircraft = df["icao24"].nunique()

    print(f"Based on number of unique aircraft:")
    if unique_aircraft < 50:
        print("Very low aircraft diversity - consider:")
        print("   • Downloading data from busier time periods")
        print("   • Using data from larger geographic areas")
        print("   • Combining multiple time periods")
    elif unique_aircraft < 200:
        print("Low aircraft diversity - consider:")
        print("   • Expanding time window")
        print("   • Including approach/departure corridors")
    else:
        print("Good aircraft diversity!")

    # Calculate how much data you'd need for better diversity
    aircraft_per_hour = unique_aircraft / (duration / 3600) if duration > 0 else 0
    
    print(f"\\nData Quality Assessment:")
    print(f"Aircraft diversity rate: {aircraft_per_hour:.1f} unique aircraft/hour")
    
    if aircraft_per_hour < 20:
        traffic_density = "Very Low"
        multiplier = 6  # Need much more time
    elif aircraft_per_hour < 50:
        traffic_density = "Low" 
        multiplier = 4
    elif aircraft_per_hour < 100:
        traffic_density = "Moderate"
        multiplier = 2.5
    else:
        traffic_density = "High"
        multiplier = 1.5
    
    print(f"Traffic density: {traffic_density}")
    
    # Calculate time needed to reach target diversity
    target_aircraft = 300  # Good target for training
    current_hours = duration / 3600
    
    if aircraft_per_hour > 0:
        total_hours_needed = target_aircraft / aircraft_per_hour
        additional_hours_needed = max(0, total_hours_needed - current_hours)
        # Apply multiplier for realistic expectations (data quality may vary)
        additional_hours_with_buffer = additional_hours_needed * multiplier
        suggested_total_hours = current_hours + additional_hours_with_buffer
    else:
        suggested_total_hours = 8  # Fallback if no aircraft detected
    
    # Cap at reasonable limits
    suggested_total_hours = max(2, min(suggested_total_hours, 24))
    
    print(f"\\nTime Recommendations:")
    print(f"   • Current coverage: {current_hours:.1f} hours")
    print(f"   • Additional hours needed: {suggested_total_hours - current_hours:.1f} hours")
    print(f"   • Suggested total coverage: {suggested_total_hours:.0f} hours") 
    print(f"   • Expected aircraft with suggestion: ~{aircraft_per_hour * suggested_total_hours:.0f}")
    
    # More specific recommendations based on current state
    if unique_aircraft < 50:
        print(f"\\nPriority Actions (Very Low Diversity):")
        if aircraft_per_hour < 10:
            print("   • Try peak traffic hours (7-9 AM, 5-7 PM local time)")
            print("   • Consider busier airports (LAX, ATL, DFW instead of regional)")
        print("   • Download multiple consecutive days")
        print("   • Combine approach and departure corridors")
    elif unique_aircraft < 200:
        print(f"\\nPriority Actions (Low Diversity):")
        print("   • Extend time window to cover rush hours")
        print("   • Add adjacent time periods (before/after current window)")
    else:
        print(f"\\nOptimization Suggestions:")
        print("   • Current diversity is adequate for initial training")
        print("   • Consider longer time series for temporal modeling")    # Suggest OpenSky download URLs based on current data timeframe
    if "time" in df.columns:
        start_dt = datetime.fromtimestamp(min_time)
        print(f"\nOpenSky Data Download Suggestions:")
        print(f"Current data from: {start_dt.strftime('%Y-%m-%d')}")
        print(f"Download more data from:")

        # Suggest nearby dates
        for days_offset in [-1, 1, -7, 7]:
            suggested_date = start_dt + timedelta(days=days_offset)
            print(
                f"   • {suggested_date.strftime('%Y-%m-%d')} (states_{suggested_date.strftime('%Y_%m_%d')}.csv)"
            )


if __name__ == "__main__":
    main()
