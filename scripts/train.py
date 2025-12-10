"""
Example training script for aircraft heading prediction.
Demonstrates how to use the data pipeline, models, and training utilities.
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import argparse

# Import project modules
from aircraft_heading_gnn.data.preprocessing import (
    load_adsb_csv, filter_terminal_area, clean_trajectories,
    compute_derived_features, load_airport_data
)
from aircraft_heading_gnn.data.dataset import (
    AircraftGraphDataset, create_train_val_test_split
)
from aircraft_heading_gnn.models.base_gnn import BaseGNN, GATHeadingPredictor
from aircraft_heading_gnn.models.temporal_gnn import TemporalGNN
from aircraft_heading_gnn.training import Trainer, HeadingLoss
from aircraft_heading_gnn.evaluation import evaluate_model, compute_all_metrics
from aircraft_heading_gnn.visualization import (
    plot_trajectories, plot_heading_distribution, plot_dataset_statistics
)


def main(args):
    print("=" * 70)
    print("Aircraft Heading Prediction with GNNs")
    print("=" * 70)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ===========================
    # 1. Load and Preprocess Data
    # ===========================
    print("\n[Step 1] Loading and preprocessing data...")
    
    # Load airport information
    airports = load_airport_data()
    airport = airports[airports['icao'] == args.airport_icao].iloc[0]
    airport_lat = airport['latitude_deg']
    airport_lon = airport['longitude_deg']
    print(f"Airport: {airport['name']} ({airport['icao']})")
    print(f"Location: {airport_lat:.4f}°, {airport_lon:.4f}°")
    
    # Load ADS-B data
    print(f"\nLoading ADS-B data from: {args.data_path}")
    df = load_adsb_csv(
        args.data_path,
        columns=['time', 'icao24', 'lat', 'lon', 'velocity', 'heading',
                 'vertrate', 'onground', 'geoaltitude']
    )
    print(f"Loaded {len(df):,} records")
    
    # Filter to terminal area
    df = filter_terminal_area(
        df, airport_lat, airport_lon,
        radius_nm=args.radius_nm,
        max_altitude_ft=args.max_altitude_ft
    )
    print(f"After terminal area filter: {len(df):,} records")
    
    # Clean trajectories
    df = clean_trajectories(
        df,
        min_points=args.min_traj_points,
        max_gap_seconds=args.max_gap_seconds,
        min_speed_kts=args.min_speed_kts,
        max_speed_kts=args.max_speed_kts
    )
    print(f"After cleaning: {len(df):,} records")
    print(f"Number of trajectories: {df['trajectory_id'].nunique()}")
    
    # Compute derived features
    df = compute_derived_features(df)
    
    # Optional: Visualize data
    if args.visualize:
        print("\nGenerating visualizations...")
        plot_trajectories(df, airport_lat, airport_lon, max_trajectories=30)
        plot_heading_distribution(df)
        plot_dataset_statistics(df)
    
    # ===========================
    # 2. Create Graph Dataset
    # ===========================
    print("\n[Step 2] Creating graph dataset...")
    
    dataset = AircraftGraphDataset(
        data_df=df,
        airport_lat=airport_lat,
        airport_lon=airport_lon,
        time_step_s=args.time_step,
        prediction_horizon_s=args.prediction_horizon,
        max_distance_nm=args.max_edge_distance,
        node_features=['lat', 'lon', 'heading', 'velocity', 'geoaltitude', 'vertrate']
    )
    
    print(f"Created {len(dataset)} graph snapshots")
    
    # Split dataset
    train_idx, val_idx, test_idx = create_train_val_test_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        temporal_split=True
    )
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Create data loaders
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # ===========================
    # 3. Create Model
    # ===========================
    print(f"\n[Step 3] Creating model: {args.model_type}")
    
    # Get feature dimensions from first graph
    sample_graph = dataset[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0
    
    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")
    
    if args.model_type == 'gcn':
        model = BaseGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=72,
            dropout=args.dropout
        )
    elif args.model_type == 'gat':
        model = GATHeadingPredictor(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=72,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # ===========================
    # 4. Setup Training
    # ===========================
    print("\n[Step 4] Setting up training...")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    criterion = HeadingLoss(
        num_classes=72,
        bin_size=5.0,
        alpha=args.loss_alpha,
        label_smoothing=args.label_smoothing
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        save_dir=args.save_dir,
        use_aux_loss=(args.model_type == 'gat')
    )
    
    # ===========================
    # 5. Train
    # ===========================
    print(f"\n[Step 5] Training for {args.num_epochs} epochs...")
    
    trainer.fit(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # ===========================
    # 6. Evaluate on Test Set
    # ===========================
    print("\n[Step 6] Evaluating on test set...")
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate
    test_metrics, predictions, targets = evaluate_model(
        model,
        test_loader,
        device=device,
        use_aux_output=(args.model_type == 'gat')
    )
    
    print("\n" + "=" * 70)
    print("Test Set Results:")
    print("=" * 70)
    for metric_name, value in test_metrics.items():
        if 'accuracy' in metric_name:
            print(f"{metric_name}: {value:.4f} ({value*100:.2f}%)")
        elif 'deg' in metric_name:
            print(f"{metric_name}: {value:.2f}°")
        else:
            print(f"{metric_name}: {value:.4f}")
    
    print("\n✓ Training complete!")
    print(f"Models saved to: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train aircraft heading prediction model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ADS-B CSV file')
    parser.add_argument('--airport_icao', type=str, default='KSEA',
                        help='Airport ICAO code')
    parser.add_argument('--radius_nm', type=float, default=40.0,
                        help='Terminal area radius in nautical miles')
    parser.add_argument('--max_altitude_ft', type=float, default=18000.0,
                        help='Maximum altitude in feet')
    parser.add_argument('--min_traj_points', type=int, default=30,
                        help='Minimum points per trajectory')
    parser.add_argument('--max_gap_seconds', type=float, default=60.0,
                        help='Maximum gap in trajectory (seconds)')
    parser.add_argument('--min_speed_kts', type=float, default=50.0,
                        help='Minimum speed (knots)')
    parser.add_argument('--max_speed_kts', type=float, default=600.0,
                        help='Maximum speed (knots)')
    
    # Graph parameters
    parser.add_argument('--time_step', type=int, default=10,
                        help='Time step for graph snapshots (seconds)')
    parser.add_argument('--prediction_horizon', type=int, default=15,
                        help='Prediction horizon (seconds)')
    parser.add_argument('--max_edge_distance', type=float, default=5.0,
                        help='Maximum distance for edges (nautical miles)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='gat',
                        choices=['gcn', 'gat'],
                        help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (GAT only)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--loss_alpha', type=float, default=0.5,
                        help='Weight for circular loss (0=CE, 1=circular)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)
