#!/usr/bin/env python3
"""
Test script for parquet-based dataset loading.
Run after sourcing setup_env.sh from repo root.
"""

from aircraft_heading_gnn.data.dataset import AircraftGraphDataset

# Path to the parquet file
parquet_path = "/home/jose/workspaces/aircraft_heading_gnn/data/processed/out.parquet"

# Path to airports JSON file
airports_json_path = "/home/jose/workspaces/aircraft_heading_gnn/data/airports.json"

print("Testing parquet-based dataset loading...")
print(f"Parquet file: {parquet_path}")
print(f"Airports JSON: {airports_json_path}")

try:
    # Create dataset using parquet file and JSON airport coordinates
    dataset = AircraftGraphDataset(
        airports_json_path=airports_json_path,
        airport_icao="KSEA",  # Seattle-Tacoma International Airport
        use_parquet=True,
        parquet_path=parquet_path
    )
    
    print(f"Dataset created successfully!")
    print(f"Number of snapshots: {len(dataset.snapshots)}")
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get first graph
        first_graph = dataset[0]
        print(f"First graph - nodes: {first_graph.x.shape[0]}, edges: {first_graph.edge_index.shape[1]}")
        print(f"Node features shape: {first_graph.x.shape}")
        print(f"Edge features shape: {first_graph.edge_attr.shape}")
        print(f"Labels shape: {first_graph.y.shape}")
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()