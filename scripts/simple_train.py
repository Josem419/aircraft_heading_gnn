#!/usr/bin/env python3
"""
Simple training script to validate model architecture and loss reduction.
Focuses on proving the model can learn from existing data before expanding dataset.
"""

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import numpy as np
from tqdm import tqdm
import argparse

from aircraft_heading_gnn.data.dataset import (
    AircraftGraphDataset,
    create_train_val_test_split,
)
from aircraft_heading_gnn.models.base_gnn import BaseGNN, GATHeadingPredictor
from aircraft_heading_gnn.models.training import HeadingLoss
from aircraft_heading_gnn.utils.angles import ang_diff_deg


def compute_heading_accuracy(predictions, targets, tolerance_deg=15):
    """Compute accuracy within angular tolerance."""
    pred_headings = predictions * 5.0  # Convert from bin to degrees
    target_headings = targets * 5.0

    errors = np.abs(ang_diff_deg(pred_headings, target_headings))
    return (errors <= tolerance_deg).mean()


def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    total_labeled_nodes = 0  # track how many labeled nodes we actually use

    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        # Only compute loss on nodes with labels
        mask = batch.has_label
        labeled_in_batch = mask.sum().item()
        if labeled_in_batch == 0:  # Skip batch if no labels
            continue

        loss = criterion(out[mask], batch.y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labeled_in_batch
        total_samples += labeled_in_batch
        total_labeled_nodes += labeled_in_batch

    return total_loss / max(total_samples, 1), total_labeled_nodes


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    total_labeled_nodes = 0  # track labeled nodes seen

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            mask = batch.has_label
            labeled_in_batch = mask.sum().item()
            if labeled_in_batch == 0:
                continue

            loss = criterion(out[mask], batch.y[mask])
            total_loss += loss.item() * labeled_in_batch
            total_samples += labeled_in_batch
            total_labeled_nodes += labeled_in_batch

            # Get predictions
            preds = torch.argmax(out[mask], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch.y[mask].cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)

    # Compute heading accuracy
    if all_preds:
        acc_15deg = compute_heading_accuracy(
            np.array(all_preds), np.array(all_targets), 15
        )
        acc_30deg = compute_heading_accuracy(
            np.array(all_preds), np.array(all_targets), 30
        )
    else:
        acc_15deg = acc_30deg = 0.0

    return avg_loss, acc_15deg, acc_30deg, len(all_targets), total_labeled_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_files", type=int, default=2, help="Max parquet files to use")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = AircraftGraphDataset(
        airports_json_path="data/airports.json",
        airport_icao="KSEA",
        use_parquet=True,
        parquet_path="data/processed/batches",
        batch_mode=True,
        max_files=args.max_files,
        prediction_horizon_s=10,  # Shorter prediction horizon for better labels
    )

    print(f"Dataset: {len(dataset)} snapshots")

    if len(dataset) == 0:
        print("No data available!")
        return

    # Check data quality on a single sample
    sample = dataset[0]
    print(
        f"Sample graph: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]//2} edges"
    )
    print(f"Node features: {sample.x.shape[1]}D")

    # Split dataset
    train_idx, val_idx, test_idx = create_train_val_test_split(
        dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )

    # Create data loaders
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    num_workers = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,       # or 8, depending on your CPU
        pin_memory=True      # good when using CUDA
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    num_node_features = sample.x.shape[1]
    num_edge_features = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    if args.model_type == "gcn":
        model = BaseGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=72,  # 360/5 degree bins
            dropout=args.dropout,
        )
    elif args.model_type == "gat":
        model = GATHeadingPredictor(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=72,
            dropout=args.dropout,
        )

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = HeadingLoss(alpha=0.3, label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_loss, train_labeled = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_loss, val_acc_15, val_acc_30, val_samples, val_labeled = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"best_model_{args.model_type}.pt")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch+1:3d}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc@15°: {val_acc_15:.3f} | "
            f"Val Acc@30°: {val_acc_30:.3f} | "
            f"Val Samples: {val_samples} | "
            f"Train labeled: {train_labeled} | "
            f"Val labeled: {val_labeled}"
        )

        if patience_counter >= 15:
            print("Early stopping!")
            break

    # Final evaluation
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    # Load best model for final test
    model.load_state_dict(torch.load(f"best_model_{args.model_type}.pt"))
    test_dataset = Subset(dataset, test_idx)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loss, test_acc_15, test_acc_30, test_samples, test_labeled = evaluate(
        model, test_loader, criterion, device
    )
    print(
        f"Test Results: Loss: {test_loss:.4f} | "
        f"Acc@15°: {test_acc_15:.3f} | "
        f"Acc@30°: {test_acc_30:.3f} | "
        f"Samples: {test_samples} | "
        f"Labeled: {test_labeled}"
    )

    # Quick sanity check
    if train_loss < 3.5:  # Cross-entropy should be < 3.5 for 72 classes if learning
        print(
            "\n Model appears to be learning! Loss reduction suggests architecture is sound."
        )
        if test_acc_15 > 0.1:  # Better than random for angular prediction
            print(" Model shows predictive capability above random chance.")
        else:
            print(" Model may need more data or tuning for good accuracy.")
    else:
        print("\n Model may not be learning effectively. Consider:")
        print("   - Learning rate adjustment")
        print("   - Model architecture changes")
        print("   - Data quality issues")


if __name__ == "__main__":
    main()
