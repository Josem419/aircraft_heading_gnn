#!/usr/bin/env python3
"""
Visualize model predictions and errors.
Usage: python scripts/visualize_predictions.py --model_path PATH [--test_data PATH]
"""

import argparse
import sys
import os

import torch
import torch.utils.data
import numpy as np
from torch_geometric.loader import DataLoader  # CHANGED: use loader.DataLoader

from aircraft_heading_gnn.data.dataset import AircraftGraphDataset
from aircraft_heading_gnn.models.base_gnn import BaseGNN, GATHeadingPredictor
from aircraft_heading_gnn.utils.visualization import plot_heading_prediction_errors
from aircraft_heading_gnn.utils.angles import ang_diff_deg
from tqdm import tqdm


def detect_model_architecture(checkpoint):
    """Detect model architecture and parameters from checkpoint."""
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Check if it's GAT (has attention weights) or GCN
    has_attention = any("att_" in key for key in state_dict.keys())
    has_turn_detector = any("turn_detector" in key for key in state_dict.keys())

    if has_attention:
        model_type = "gat"
    else:
        model_type = "gcn"

    # Infer dimensions from checkpoint
    hidden_dim = None
    num_layers = 0
    num_heads = 8  # default

    # Count conv layers
    conv_layers = [k for k in state_dict.keys() if k.startswith("convs.")]
    if conv_layers:
        layer_nums = [
            int(k.split(".")[1]) for k in conv_layers if k.split(".")[1].isdigit()
        ]
        num_layers = max(layer_nums) + 1 if layer_nums else 3

    # Get hidden dim from node encoder if available
    if "node_encoder.weight" in state_dict:
        hidden_dim = state_dict["node_encoder.weight"].shape[0]

    # For GAT, try to infer num_heads from attention weights
    if model_type == "gat" and "convs.0.att_src" in state_dict:
        att_tensor = state_dict["convs.0.att_src"]
        if len(att_tensor.shape) >= 2:
            # Shape is typically [1, num_heads, head_dim]
            num_heads = att_tensor.shape[1]
            head_dim = (
                att_tensor.shape[2]
                if len(att_tensor.shape) > 2
                else att_tensor.shape[1] // max(num_heads, 1)
            )

            # Verify hidden_dim consistency
            if num_heads > 0 and head_dim > 0:
                inferred_hidden_dim = num_heads * head_dim
                if hidden_dim and hidden_dim != inferred_hidden_dim:
                    print(
                        f"Warning: Hidden dim mismatch. Using inferred: {inferred_hidden_dim}"
                    )
                    hidden_dim = inferred_hidden_dim

            print(
                f"Inferred from attention: num_heads={num_heads}, head_dim={head_dim}, hidden_dim={hidden_dim}"
            )
    elif model_type == "gat":
        # If we can't infer heads, use a reasonable default
        num_heads = 8

    return {
        "model_type": model_type,
        "hidden_dim": hidden_dim or 64,  # fallback
        "num_layers": num_layers or 3,  # fallback
        "num_heads": max(1, num_heads),
        "has_turn_detector": has_turn_detector,
    }


def create_model_from_checkpoint(
    checkpoint, num_node_features, num_edge_features, model_type_override=None
):
    """Create model instance matching the checkpoint architecture."""
    arch_info = detect_model_architecture(checkpoint)

    # Override model type if specified
    if model_type_override:
        arch_info["model_type"] = model_type_override

    print(f"Detected architecture: {arch_info}")

    if arch_info["model_type"] == "gcn":
        model = BaseGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=arch_info["hidden_dim"],
            num_layers=arch_info["num_layers"],
            num_classes=72,
            dropout=0.2,
        )
    elif arch_info["model_type"] == "gat":
        model = GATHeadingPredictor(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=arch_info["hidden_dim"],
            num_layers=arch_info["num_layers"],
            num_classes=72,
            num_heads=arch_info["num_heads"],
            dropout=0.2,
        )
    else:
        raise ValueError(f"Unknown model type: {arch_info['model_type']}")

    return model, arch_info


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions and errors"
    )
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--model_type",
        default=None,
        choices=["gcn", "gat"],
        help="Type of model architecture (auto-detect if not specified)",
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed/batches",
        help="Directory containing processed batch data",
    )
    parser.add_argument(
        "--airports_json",
        default="data/airports.json",
        help="Path to airports JSON file",
    )
    parser.add_argument(
        "--airport_icao", default="KSEA", help="Airport ICAO code"
    )
    parser.add_argument(
        "--output_dir",
        default="prediction_analysis",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of samples to analyze"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for prediction"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading dataset...")
    try:
        dataset = AircraftGraphDataset(
            airports_json_path=args.airports_json,
            airport_icao=args.airport_icao,
            use_parquet=True,
            parquet_path=args.data_dir,
            batch_mode=True,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure the data directory and airport data are correct.")
        return

    print(f"Dataset loaded with {len(dataset)} samples")

    if len(dataset) == 0:
        print("Dataset is empty. Check data directory and airport ICAO code.")
        return

    # Get sample graph to determine dimensions
    sample_graph = dataset[0]
    num_node_features = sample_graph.x.shape[1]
    num_edge_features = (
        sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0
    )

    print(f"Node features: {num_node_features}, Edge features: {num_edge_features}")

    # Load checkpoint first to detect architecture
    print(f"Loading model from {args.model_path}...")
    checkpoint = None

    # Try different loading strategies
    loading_strategies = [
        lambda: torch.load(args.model_path, map_location=device),
        lambda: torch.load(args.model_path, map_location=device, weights_only=False),
    ]

    for i, strategy in enumerate(loading_strategies):
        try:
            checkpoint = strategy()
            print(f"Successfully loaded checkpoint using strategy {i+1}")
            break
        except Exception as e:
            print(f"Loading strategy {i+1} failed: {e}")
            if i == len(loading_strategies) - 1:
                print("All loading strategies failed!")
                return

    # Create model with architecture detection
    print("Detecting model architecture from checkpoint...")
    model, arch_info = create_model_from_checkpoint(
        checkpoint,
        num_node_features,
        num_edge_features,
        model_type_override=args.model_type,
    )

    # Load state dict
    try:
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "metrics" in checkpoint:
                print(f"Loaded model with validation metrics: {checkpoint['metrics']}")
        else:
            # Direct state dict (older format)
            model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print(
            "Checkpoint keys:",
            list(checkpoint.get("model_state_dict", checkpoint).keys())[:10],
        )
        print("Model expecting keys:", list(model.state_dict().keys())[:10])
        return

    model.to(device)
    model.eval()

    print(
        f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test model with a sample to ensure compatibility
    print("Testing model compatibility...")
    try:
        test_loader = DataLoader([sample_graph], batch_size=1, shuffle=False)
        test_batch = next(iter(test_loader)).to(device)
        with torch.no_grad():
            test_output = model(
                test_batch.x,
                test_batch.edge_index,
                test_batch.edge_attr,
                test_batch.batch,
            )
        print(f"Model output shape: {test_output.shape}")
    except Exception as e:
        print(f"Model compatibility test failed: {e}")
        return

    # Generate predictions on test data
    print("Generating predictions...")
    true_headings = []
    predicted_headings = []
    prediction_errors = []

    # Sample some graphs from the dataset
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Create data loader for efficient batch processing
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    data_loader = DataLoader(
        subset_dataset, batch_size=args.batch_size, shuffle=False
    )

    with torch.no_grad():
        for batch_idx, batch_graphs in enumerate(
            tqdm(data_loader, desc="Processing batches")
        ):
            try:
                batch_graphs = batch_graphs.to(device)

                # Get predictions from model
                logits = model(
                    batch_graphs.x,
                    batch_graphs.edge_index,
                    batch_graphs.edge_attr,
                    batch_graphs.batch,
                )
                predicted_bins = torch.argmax(logits, dim=-1)

                # Convert bins to degrees (center of each bin)
                predicted_degrees = (predicted_bins.float() * 5.0 + 2.5) % 360
                true_degrees = (batch_graphs.y.float() * 5.0 + 2.5) % 360

                # Only consider nodes with labels
                if hasattr(batch_graphs, "has_label"):
                    valid_mask = batch_graphs.has_label
                    predicted_degrees = predicted_degrees[valid_mask]
                    true_degrees = true_degrees[valid_mask]

                # Move to CPU for processing
                predicted_degrees = predicted_degrees.cpu().numpy()
                true_degrees = true_degrees.cpu().numpy()

                # Store predictions
                true_headings.extend(true_degrees)
                predicted_headings.extend(predicted_degrees)

                # Calculate angular errors
                batch_errors = [
                    abs(ang_diff_deg(p, t))
                    for p, t in zip(predicted_degrees, true_degrees)
                ]
                prediction_errors.extend(batch_errors)

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

    print(f"Generated {len(true_headings)} predictions")

    if len(true_headings) == 0:
        print("No valid predictions found. Check if the model and data are compatible.")
        return

    # Create prediction error visualization
    print("Creating prediction error plots...")
    filename = f"heading_prediction_errors_{args.airport_icao}.png"
    save_path = os.path.join(args.output_dir, filename)
    plot_heading_prediction_errors(
        np.array(true_headings),
        np.array(predicted_headings),
        save_path=save_path,
    )

    # Print comprehensive statistics
    print(f"\n{'='*60}")
    print("PREDICTION ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total predictions: {len(true_headings)}")
    print(f"Mean absolute error: {np.mean(prediction_errors):.2f}°")
    print(f"Median absolute error: {np.median(prediction_errors):.2f}°")
    print(f"Standard deviation: {np.std(prediction_errors):.2f}°")
    print(f"90th percentile error: {np.percentile(prediction_errors, 90):.2f}°")
    print(f"95th percentile error: {np.percentile(prediction_errors, 95):.2f}°")
    print(f"Maximum error: {np.max(prediction_errors):.2f}°")

    # Accuracy within certain thresholds
    thresholds = [5, 10, 15, 30, 45]
    print("\nAccuracy within thresholds:")
    errors_arr = np.array(prediction_errors)
    for thresh in thresholds:
        accuracy = np.mean(errors_arr <= thresh) * 100
        print(f"  Within {thresh:2d}°: {accuracy:.1f}%")

    print(f"\nPrediction analysis plots saved to {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
