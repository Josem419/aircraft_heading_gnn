#!/usr/bin/env python3
"""
Analyze dataset and create comprehensive visualizations.
Usage: python scripts/analyze_dataset.py [--parquet_path PATH] [--airport ICAO]
"""

import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from aircraft_heading_gnn.data.dataset import AircraftGraphDataset
from aircraft_heading_gnn.utils.visualization import (
    plot_dataset_statistics, 
    plot_temporal_features,
    plot_graph_snapshot
)


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset and create visualizations')
    parser.add_argument('--parquet_path', 
                       default='data/processed/out.parquet',
                       help='Path to parquet file')
    parser.add_argument('--airports_json', 
                       default='data/airports.json',
                       help='Path to airports JSON file')
    parser.add_argument('--airport_icao', 
                       default='KSEA',
                       help='Airport ICAO code')
    parser.add_argument('--output_dir', 
                       default='analysis',
                       help='Output directory for analysis plots')
    parser.add_argument('--num_snapshots', 
                       type=int, 
                       default=5,
                       help='Number of graph snapshots to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Creating dataset...")
    dataset = AircraftGraphDataset(
        airports_json_path=args.airports_json,
        airport_icao=args.airport_icao,
        use_parquet=True,
        parquet_path=args.parquet_path
    )
    
    print(f"Dataset created with {len(dataset)} snapshots")
    
    # Plot dataset statistics
    print("Creating dataset statistics plot...")
    plot_dataset_statistics(
        dataset.data_df,
        save_path=os.path.join(args.output_dir, 'dataset_statistics.png')
    )
    
    # Plot temporal features
    print("Creating temporal features plot...")
    plot_temporal_features(
        dataset.data_df,
        save_path=os.path.join(args.output_dir, 'temporal_features.png')
    )
    
    # Plot sample graph snapshots
    print(f"Creating {args.num_snapshots} graph snapshot plots...")
    for i in range(min(args.num_snapshots, len(dataset))):
        graph = dataset[i]
        snapshot_data = dataset.snapshots[i]
        
        plot_graph_snapshot(
            graph,
            snapshot_data['data'],
            airport_lat=dataset.airport_lat,
            airport_lon=dataset.airport_lon,
            save_path=os.path.join(args.output_dir, f'graph_snapshot_{i:03d}.png')
        )
    
    print(f"Analysis plots saved to {args.output_dir}/")
    print(f"Created {len([f for f in os.listdir(args.output_dir) if f.endswith('.png')])} plots")


if __name__ == '__main__':
    main()