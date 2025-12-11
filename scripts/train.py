#!/usr/bin/env python3
"""
Train aircraft heading prediction model from preprocessed parquet data.

This script assumes you've already converted raw OpenSky tar/CSV data
into parquet files (e.g., via process_opensky_to_parquet.py), and uses
AircraftGraphDataset to build graph snapshots directly from parquet.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from aircraft_heading_gnn.data.preprocessing import load_airport_data
from aircraft_heading_gnn.data.dataset import (
    AircraftGraphDataset,
    create_train_val_test_split,
)
from aircraft_heading_gnn.models.base_gnn import BaseGNN, GATHeadingPredictor
from aircraft_heading_gnn.models.training import Trainer, HeadingLoss
from aircraft_heading_gnn.models.evaluation import evaluate_model


def main(args):
    print("=" * 70)
    print("Aircraft Heading Prediction with GNNs (Parquet)")
    print("=" * 70)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # ===========================
    # 1. Airport / Metadata
    # ===========================
    print("\n[Step 1] Loading airport metadata...")

    airports = load_airport_data(args.airports_json)
    airport_row = airports[airports["icao"] == args.airport_icao]
    if airport_row.empty:
        raise ValueError(
            f"Airport ICAO '{args.airport_icao}' not found in airports data "
            f"({args.airports_json})"
        )

    airport = airport_row.iloc[0]
    airport_lat = airport["latitude_deg"]
    airport_lon = airport["longitude_deg"]
    print(f"Airport: {airport['name']} ({airport['icao']})")
    print(f"Location: {airport_lat:.4f}°, {airport_lon:.4f}°")

    # ===========================
    # 2. Create Graph Dataset from Parquet
    # ===========================
    print("\n[Step 2] Creating graph dataset from parquet...")

    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet path does not exist: {parquet_path}")

    # Dataset handles:
    #  - loading parquet(s)
    #  - filtering to terminal area
    #  - computing derived features
    #  - creating snapshots & graphs
    dataset = AircraftGraphDataset(
        airports_json_path=str(args.airports_json),
        airport_icao=args.airport_icao,
        time_step_s=args.time_step,
        prediction_horizon_s=args.prediction_horizon,
        max_distance_nm=args.max_edge_distance,
        use_parquet=True,
        parquet_path=str(parquet_path),
        batch_mode=args.batch_mode,
        max_files=args.max_files,
    )

    num_snapshots = len(dataset)
    print(f"\nDataset created with {num_snapshots} graph snapshots")

    if num_snapshots == 0:
        print("No graph snapshots were created. Exiting.")
        return

    # ===========================
    # 3. Train/Val/Test Split
    # ===========================
    print("\n[Step 3] Creating train/val/test split...")

    train_idx, val_idx, test_idx = create_train_val_test_split(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        temporal_split=True,
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    num_workers = args.num_workers
    pin_memory = device == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ===========================
    # 4. Create Model
    # ===========================
    print(f"\n[Step 4] Creating model: {args.model_type}")

    # Sample one graph to infer feature dimensions
    sample_graph = dataset[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = (
        sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0
    )

    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")

    if args.model_type == "gcn":
        model = BaseGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=72,  # 360 / 5
            dropout=args.dropout,
        )
        use_aux_loss = False
    elif args.model_type == "gat":
        model = GATHeadingPredictor(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=72,
            num_heads=args.num_heads,
            dropout=args.dropout,
        )
        # Set True only if GATHeadingPredictor returns dict with "heading"
        use_aux_loss = False
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # ===========================
    # 5. Setup Training
    # ===========================
    print("\n[Step 5] Setting up training...")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    criterion = HeadingLoss(
        num_classes=72,
        bin_size=5.0,
        alpha=args.loss_alpha,
        label_smoothing=args.label_smoothing,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_type=args.model_type,
        scheduler=scheduler,
        save_dir=str(save_dir),
        use_aux_loss=use_aux_loss,
    )

    # ===========================
    # 6. Train
    # ===========================
    print(f"\n[Step 6] Training for {args.num_epochs} epochs...")

    trainer.fit(
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    # ===========================
    # 7. Evaluate on Test Set
    # ===========================
    print("\n[Step 7] Evaluating on test set...")

    trainer.load_checkpoint("best_model.pt")

    test_metrics, predictions, targets = evaluate_model(
        model,
        test_loader,
        device=device,
        use_aux_output=use_aux_loss,
    )

    print("\n" + "=" * 70)
    print("Test Set Results:")
    print("=" * 70)
    for metric_name, value in test_metrics.items():
        if "accuracy" in metric_name:
            print(f"{metric_name}: {value:.4f} ({value * 100:.2f}%)")
        elif "deg" in metric_name:
            print(f"{metric_name}: {value:.2f}°")
        else:
            print(f"{metric_name}: {value:.4f}")

    print("\n✓ Training complete!")
    print(f"Models & logs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train aircraft heading prediction model from parquet data"
    )

    # Parquet / data source parameters
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Path to a parquet file or directory of parquet files",
    )
    parser.add_argument(
        "--batch_mode",
        action="store_true",
        help="If set and parquet_path is a directory, load multiple parquet files in batch mode",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of parquet files to process (for testing / debugging)",
    )

    # Airport / metadata
    parser.add_argument(
        "--airports_json",
        type=str,
        default="data/airports.json",
        help="Path to airports.json file",
    )
    parser.add_argument(
        "--airport_icao",
        type=str,
        default="KSEA",
        help="Airport ICAO code",
    )

    # Graph parameters
    parser.add_argument(
        "--time_step",
        type=int,
        default=10,
        help="Time step for graph snapshots (seconds)",
    )
    parser.add_argument(
        "--prediction_horizon",
        type=int,
        default=15,
        help="Prediction horizon (seconds)",
    )
    parser.add_argument(
        "--max_edge_distance",
        type=float,
        default=35.0,
        help="Maximum distance for edges (nautical miles between aircraft)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="gat",
        choices=["gcn", "gat"],
        help="Model architecture",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GNN layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads (GAT only)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )

    # Training parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.5,
        help="Weight for circular loss (0=CE, 1=circular)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Loader / misc
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints_parquet",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    main(args)
