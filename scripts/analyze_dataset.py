#!/usr/bin/env python3
"""
Analyze dataset and create comprehensive visualizations.
Usage: python scripts/analyze_dataset.py [--parquet_path PATH] [--airport ICAO]
"""

import random
import argparse
import sys
import os

from aircraft_heading_gnn.data.dataset import AircraftGraphDataset
from aircraft_heading_gnn.utils.visualization import (
    plot_dataset_statistics,
    plot_temporal_features,
    plot_graph_snapshot,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset and create visualizations"
    )
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
        "--output_dir", default="analysis", help="Output directory for analysis plots"
    )
    parser.add_argument(
        "--num_snapshots",
        type=int,
        default=10,
        help="Number of graph snapshots to visualize",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Creating dataset...")
    dataset = AircraftGraphDataset(
        airports_json_path=args.airports_json,
        airport_icao=args.airport_icao,
        use_parquet=True,
        parquet_path=args.parquet_path,
    )

    print(f"Dataset created with {len(dataset)} snapshots")

    # Plot dataset statistics
    print("Creating dataset statistics plot...")
    plot_dataset_statistics(
        dataset.data_df,
        save_path=os.path.join(args.output_dir, "dataset_statistics.png"),
    )

    # Plot temporal features
    print("Creating temporal features plot...")
    plot_temporal_features(
        dataset.data_df,
        save_path=os.path.join(args.output_dir, "temporal_features.png"),
    )

    # Create sorted indices by number of nodes (highest first)
    print("Sorting snapshots by node count...")
    snapshot_info = []
    for idx, graph in enumerate(dataset):
        node_count = graph.x.shape[0]  # Number of nodes
        edge_count = graph.edge_index.shape[1]  # Number of edges
        snapshot_info.append((idx, node_count, edge_count))
        
    # Sort by node count (descending)
    snapshot_info.sort(key=lambda x: x[1], reverse=True)

    print(f"Node count range: {snapshot_info[-1][1]} to {snapshot_info[0][1]} nodes")

    # Plot sample graph snapshots (starting with highest node count)
    print(f"Creating {args.num_snapshots} graph snapshot plots...")

    # plot these at random to get variety
    for _ in range(min(args.num_snapshots, len(dataset))):
        plot_idx = random.randint(0, len(snapshot_info) - 1)
        dataset_idx, node_count, edge_count = snapshot_info[plot_idx]
        graph = dataset[dataset_idx]

        print(f"Snapshot {plot_idx+1}: {node_count} nodes, {edge_count} edges")

        plot_graph_snapshot(
            graph,
            airport_lat=dataset.airport_lat,
            airport_lon=dataset.airport_lon,
            airport_icao=args.airport_icao,
            save_path=os.path.join(
                args.output_dir, f"graph_snapshot_{plot_idx:03d}_nodes_{node_count}.png"
            ),
        )

    print(f"Analysis plots saved to {args.output_dir}/")
    print(
        f"Created {len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])} plots"
    )


if __name__ == "__main__":
    main()
