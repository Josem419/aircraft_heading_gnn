#!/usr/bin/env python3
"""
Analyze dataset and create comprehensive visualizations.
Usage: python scripts/analyze_dataset.py [--parquet_path PATH] [--airport ICAO]
"""

import random
import argparse
import os
import pandas as pd

from aircraft_heading_gnn.data.dataset import AircraftGraphDataset
from aircraft_heading_gnn.utils.visualization import (
    plot_dataset_statistics,
    plot_temporal_features,
    plot_graph_snapshot,
    plot_trajectories,
)

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
    
    # Auto-detect if parquet_path is a directory (batch mode) or file (single mode)
    is_directory = os.path.isdir(args.parquet_path)
    
    dataset = AircraftGraphDataset(
        airports_json_path=args.airports_json,
        airport_icao=args.airport_icao,
        use_parquet=True,
        parquet_path=args.parquet_path,
        batch_mode=is_directory,
        max_files=3 if is_directory else None  # Limit files in batch mode for analysis
    )
    
    print(f"Dataset created with {len(dataset)} snapshots ({'batch' if is_directory else 'single file'} mode)")

    # Plot dataset statistics
    print("Creating dataset statistics plot...")
    try:
        if is_directory:
            # For batch mode, create a sample from multiple files
            sample_dfs = []
            for i, mini_dataset in enumerate(dataset._batch_datasets[:3]):  # Sample first 3
                if hasattr(mini_dataset, 'data_df') and len(mini_dataset.data_df) > 0:
                    sample_size = min(1000, len(mini_dataset.data_df))
                    sample_dfs.append(mini_dataset.data_df.sample(n=sample_size, random_state=42))
            
            if sample_dfs:
                combined_sample = pd.concat(sample_dfs, ignore_index=True)
                plot_dataset_statistics(
                    combined_sample,
                    save_path=os.path.join(args.output_dir, "dataset_statistics.png"),
                )
                print(f"Created statistics plot from {len(combined_sample):,} sample records")
            else:
                print("No data available for statistics plot")
        else:
            # Single file mode
            if hasattr(dataset, 'data_df'):
                plot_dataset_statistics(
                    dataset.data_df,
                    save_path=os.path.join(args.output_dir, "dataset_statistics.png"),
                )
            else:
                print("No data available for statistics plot")
    except Exception as e:
        print(f"Warning: Could not create dataset statistics plot: {e}")

    # Plot temporal features
    print("Creating temporal features plot...")
    try:
        if is_directory:
            # For batch mode, use first available dataset
            plot_dataset = next((ds for ds in dataset._batch_datasets if hasattr(ds, 'data_df') and len(ds.data_df) > 0), None)
            if plot_dataset:
                plot_temporal_features(
                    plot_dataset.data_df,
                    save_path=os.path.join(args.output_dir, "temporal_features.png"),
                )
            else:
                print("No data available for temporal features plot")
        else:
            # Single file mode
            if hasattr(dataset, 'data_df'):
                plot_temporal_features(
                    dataset.data_df,
                    save_path=os.path.join(args.output_dir, "temporal_features.png"),
                )
            else:
                print("No data available for temporal features plot")
    except Exception as e:
        print(f"Warning: Could not create temporal features plot: {e}")

    # Plot trajectories
    print("Creating trajectory plot...")
    try:
        if is_directory:
            # For batch mode, use first available dataset
            plot_dataset = next((ds for ds in dataset._batch_datasets if hasattr(ds, 'data_df') and len(ds.data_df) > 0), None)
            
            if plot_dataset:
                plot_trajectories(
                    plot_dataset.data_df,
                    airport_lat=dataset.airport_lat,
                    airport_lon=dataset.airport_lon,
                    airport_icao=args.airport_icao,
                    save_path=os.path.join(args.output_dir, "sample_trajectories.png"),
                )
                print(f"Created trajectory plot with up to 10 trajectories")
            else:
                print("No data available for trajectory plot")
        else:
            # Single file mode
            if hasattr(dataset, 'data_df') and len(dataset.data_df) > 0:
                plot_trajectories(
                    dataset.data_df,
                    airport_lat=dataset.airport_lat,
                    airport_lon=dataset.airport_lon,
                    airport_icao=args.airport_icao,
                    save_path=os.path.join(args.output_dir, "sample_trajectories.png"),
                )
                print(f"Created trajectory plot with up to 10 trajectories")
            else:
                print("No data available for trajectory plot")
    except Exception as e:
        print(f"Warning: Could not create trajectory plot: {e}")

    # Check if dataset has any snapshots
    if len(dataset) == 0:
        print("Warning: Dataset has 0 snapshots. Analysis cannot proceed.")
        print(f"Analysis plots saved to {args.output_dir}/")
        print("Created 0 plots (no data available)")
        return

    # Create sorted indices by number of nodes (highest first)
    print("Sorting snapshots by node count...")
    print(f"Sampling {min(args.num_snapshots * 3, len(dataset), 20)} snapshots to analyze node distribution...")
    
    snapshot_info = []
    max_samples = min(args.num_snapshots * 3, len(dataset), 20)  # Sample at most 20 for analysis
    
    # Sample snapshots distributed across the dataset
    sample_indices = []
    if len(dataset) > max_samples:
        step = len(dataset) // max_samples
        sample_indices = list(range(0, len(dataset), step))[:max_samples]
    else:
        sample_indices = list(range(len(dataset)))
    
    for i, idx in enumerate(sample_indices):
        print(f"Analyzing snapshot {i+1}/{len(sample_indices)}...")
        graph = dataset[idx]
        node_count = graph.x.shape[0]  # Number of nodes
        edge_count = graph.edge_index.shape[1]  # Number of edges
        snapshot_info.append((idx, node_count, edge_count))
        
    # Sort by node count (descending)
    snapshot_info.sort(key=lambda x: x[1], reverse=True)

    print(f"Sampled node count range: {snapshot_info[-1][1]} to {snapshot_info[0][1]} nodes")

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
