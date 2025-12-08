#!/usr/bin/env python3
"""
Visualize aircraft trajectories from parquet data.
Usage: python scripts/visualize_trajectories.py [--parquet_path PATH] [--airport ICAO] [--max_trajs N]
"""


import pandas as pd
import argparse
import sys
import os
import json

from aircraft_heading_gnn.data.preprocessing import (
    filter_terminal_area,
    clean_trajectories,
)
from aircraft_heading_gnn.utils.visualization import (
    plot_trajectories,
    plot_heading_distribution,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

def main():
    parser = argparse.ArgumentParser(description="Visualize aircraft trajectories")
    parser.add_argument(
        "--parquet_path",
        default="data/processed/out.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--airports_json",
        default="data/airports.json",
        help="Path to airports JSON file",
    )
    parser.add_argument("--airport_icao", default="KSEA", help="Airport ICAO code")
    parser.add_argument(
        "--max_trajs", type=int, default=20, help="Maximum trajectories to plot"
    )
    parser.add_argument(
        "--output_dir", default="plots", help="Output directory for plots"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    print(f"Loaded {len(df):,} records")

    # Load airport coordinates
    with open(args.airports_json, "r", encoding="utf-8") as f:
        airports_data = json.load(f)

    airport_info = None
    for airport in airports_data["airports"]:
        if airport["icao"] == args.airport_icao:
            airport_info = airport
            break

    if airport_info is None:
        print(f"Airport {args.airport_icao} not found in {args.airports_json}")
        return

    airport_lat = airport_info["latitude_deg"]
    airport_lon = airport_info["longitude_deg"]
    print(f"Using airport: {airport_info['name']} ({args.airport_icao})")

    # Filter to terminal area
    df_filtered = filter_terminal_area(df, airport_lat, airport_lon)
    print(f"After terminal area filter: {len(df_filtered):,} records")

    # Clean trajectories
    df_clean = clean_trajectories(df_filtered, min_points=10)
    print(f"After cleaning: {len(df_clean):,} records")

    # get number of unique trajectories
    unique_trajectories = df_clean["icao24"].nunique()
    print(f"Number of unique trajectories: {unique_trajectories}")

    # Plot raw trajectories
    print("Plotting trajectories...")
    plot_trajectories(
        df_clean,
        airport_lat=airport_lat,
        airport_lon=airport_lon,
        airport_icao=args.airport_icao,
        max_trajectories=args.max_trajs,
        save_path=os.path.join(args.output_dir, "trajectories.png"),
    )

    # Plot heading distribution
    print("Plotting heading distribution...")
    plot_heading_distribution(
        df_clean, save_path=os.path.join(args.output_dir, "heading_distribution.png")
    )

    print(f"Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
