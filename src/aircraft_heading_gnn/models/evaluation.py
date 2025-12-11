"""
Evaluation metrics and utilities for heading prediction.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from aircraft_heading_gnn.utils.angles import ang_diff_deg


def circular_mae(
    predictions: np.ndarray,
    targets: np.ndarray,
    bin_size: float = 5.0
) -> float:
    """
    Mean Absolute Error for circular heading predictions (bins -> degrees).

    Args:
        predictions: Predicted heading bins
        targets: Target heading bins
        bin_size: Bin size in degrees

    Returns:
        MAE in degrees
    """
    if predictions.size == 0:
        return 0.0

    pred_angles = (predictions * bin_size + bin_size / 2) % 360
    target_angles = (targets * bin_size + bin_size / 2) % 360

    errors = np.array(
        [abs(ang_diff_deg(p, t)) for p, t in zip(pred_angles, target_angles)]
    )
    return errors.mean()


def circular_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    bin_size: float = 5.0
) -> float:
    """
    Root Mean Squared Error for circular heading predictions (bins -> degrees).
    """
    if predictions.size == 0:
        return 0.0

    pred_angles = (predictions * bin_size + bin_size / 2) % 360
    target_angles = (targets * bin_size + bin_size / 2) % 360

    errors = np.array(
        [ang_diff_deg(p, t) for p, t in zip(pred_angles, target_angles)]
    )
    return np.sqrt((errors ** 2).mean())


def heading_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    tolerance_deg: float = 10.0,
    bin_size: float = 5.0
) -> float:
    """
    Accuracy within a tolerance (degrees).

    Args:
        predictions: Predicted heading bins
        targets: Target heading bins
        tolerance_deg: Tolerance in degrees
        bin_size: Bin size in degrees

    Returns:
        Accuracy as fraction in [0, 1]
    """
    if predictions.size == 0:
        return 0.0

    pred_angles = (predictions * bin_size + bin_size / 2) % 360
    target_angles = (targets * bin_size + bin_size / 2) % 360

    errors = np.array(
        [abs(ang_diff_deg(p, t)) for p, t in zip(pred_angles, target_angles)]
    )

    return (errors <= tolerance_deg).mean()


def turn_direction_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    current_headings: np.ndarray,
    bin_size: float = 5.0
) -> float:
    """
    Accuracy of predicting turn direction (left/straight/right).

    Args:
        predictions: Predicted heading bins
        targets: Target heading bins
        current_headings: Current heading angles (in degrees)
        bin_size: Bin size in degrees

    Returns:
        Turn direction accuracy
    """
    if predictions.size == 0:
        return 0.0

    pred_angles = (predictions * bin_size + bin_size / 2) % 360
    target_angles = (targets * bin_size + bin_size / 2) % 360

    # Compute turn directions
    pred_turns = np.array(
        [ang_diff_deg(p, c) for p, c in zip(pred_angles, current_headings)]
    )
    target_turns = np.array(
        [ang_diff_deg(t, c) for t, c in zip(target_angles, current_headings)]
    )

    # Classify into left (-1), straight (0), right (1)
    threshold = 5.0

    def classify_turn(delta):
        if delta < -threshold:
            return -1  # left
        elif delta > threshold:
            return 1  # right
        else:
            return 0  # straight

    pred_classes = np.array([classify_turn(t) for t in pred_turns])
    target_classes = np.array([classify_turn(t) for t in target_turns])

    return (pred_classes == target_classes).mean()


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    current_headings: Optional[np.ndarray] = None,
    bin_size: float = 5.0
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: Predicted heading bins
        targets: Target heading bins
        current_headings: Current headings (for turn direction)
        bin_size: Bin size in degrees

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'exact_accuracy': (predictions == targets).mean() if predictions.size > 0 else 0.0,
        'mae_degrees': circular_mae(predictions, targets, bin_size),
        'rmse_degrees': circular_rmse(predictions, targets, bin_size),
        'accuracy_5deg': heading_accuracy(predictions, targets, 5.0, bin_size),
        'accuracy_10deg': heading_accuracy(predictions, targets, 10.0, bin_size),
        'accuracy_15deg': heading_accuracy(predictions, targets, 15.0, bin_size),
    }

    if current_headings is not None and current_headings.size > 0:
        metrics['turn_direction_accuracy'] = turn_direction_accuracy(
            predictions, targets, current_headings, bin_size
        )

    return metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_aux_output: bool = False
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a model on a dataset.

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run on
        use_aux_output: Whether model returns dict with 'heading' key

    Returns:
        Tuple of (metrics_dict, predictions, targets)
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_current_headings = []

    for batch in dataloader:
        batch = batch.to(device)

        # Forward pass
        if use_aux_output:
            outputs = model(batch.x, batch.edge_index, batch.edge_attr)
            logits = outputs['heading']
        else:
            logits = model(batch.x, batch.edge_index, batch.edge_attr)

        # Get predictions
        preds = torch.argmax(logits, dim=-1)

        # Filter by valid labels
        if hasattr(batch, 'has_label'):
            valid_mask = batch.has_label
            preds = preds[valid_mask]
            targets = batch.y[valid_mask]

            # Extract current headings from node features if available
            # Assuming heading is encoded as sin/cos in features[3:5]
            if batch.x.shape[1] >= 5:
                heading_sin = batch.x[valid_mask, 3]
                heading_cos = batch.x[valid_mask, 4]
                current_h = torch.atan2(heading_sin, heading_cos) * 180 / np.pi
                current_h = (current_h + 360) % 360
                all_current_headings.extend(current_h.cpu().numpy())
        else:
            targets = batch.y

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    predictions = np.array(all_preds)
    targets = np.array(all_targets)

    # Compute metrics
    current_headings = (
        np.array(all_current_headings) if len(all_current_headings) > 0 else None
    )
    metrics = compute_all_metrics(predictions, targets, current_headings)

    return metrics, predictions, targets


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_bins: int = 72,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix for heading predictions.

    Args:
        predictions: Predicted bins
        targets: Target bins
        num_bins: Number of heading bins
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(num_bins))

    # Normalize by row (true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_normalized, cmap='viridis', aspect='auto')

    # Labels
    ax.set_xlabel('Predicted Heading Bin')
    ax.set_ylabel('True Heading Bin')
    ax.set_title('Heading Prediction Confusion Matrix (Normalized)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Frequency')

    # Ticks
    tick_marks = np.arange(0, num_bins, max(num_bins // 8, 1))
    tick_labels = [f"{int(i * 5)}°" for i in tick_marks]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(tick_labels)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def plot_error_by_heading(
    predictions: np.ndarray,
    targets: np.ndarray,
    bin_size: float = 5.0,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Plot prediction error as a function of true heading.

    Args:
        predictions: Predicted bins
        targets: Target bins
        bin_size: Bin size in degrees
        figsize: Figure size
        save_path: Optional path to save
    """
    pred_angles = (predictions * bin_size + bin_size / 2) % 360
    target_angles = (targets * bin_size + bin_size / 2) % 360

    errors = np.array(
        [ang_diff_deg(p, t) for p, t in zip(pred_angles, target_angles)]
    )

    # Bin by true heading
    heading_bins = np.arange(0, 360, 10)
    binned_errors = []
    bin_centers = []

    for i in range(len(heading_bins) - 1):
        mask = (target_angles >= heading_bins[i]) & (target_angles < heading_bins[i + 1])
        if mask.sum() > 0:
            binned_errors.append(np.abs(errors[mask]).mean())
            bin_centers.append((heading_bins[i] + heading_bins[i + 1]) / 2)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(bin_centers, binned_errors, width=8, alpha=0.7, edgecolor='black')
    ax.set_xlabel('True Heading (degrees)')
    ax.set_ylabel('Mean Absolute Error (degrees)')
    ax.set_title('Prediction Error by Heading Direction')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def analyze_failure_cases(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold_deg: float = 30.0,
    bin_size: float = 5.0
) -> Dict[str, Any]:
    """
    Analyze cases where prediction error is large.

    Args:
        predictions: Predicted bins
        targets: Target bins
        threshold_deg: Error threshold for failure
        bin_size: Bin size in degrees

    Returns:
        Dictionary with failure analysis
    """
    if predictions.size == 0:
        return {
            'failure_rate': 0.0,
            'num_failures': 0,
            'mean_error_failures': 0.0,
            'median_error_failures': 0.0,
            'failure_indices': []
        }

    pred_angles = (predictions * bin_size + bin_size / 2) % 360
    target_angles = (targets * bin_size + bin_size / 2) % 360

    errors = np.array(
        [abs(ang_diff_deg(p, t)) for p, t in zip(pred_angles, target_angles)]
    )

    failures = errors > threshold_deg

    analysis = {
        'failure_rate': failures.mean(),
        'num_failures': failures.sum(),
        'mean_error_failures': errors[failures].mean() if failures.sum() > 0 else 0.0,
        'median_error_failures': float(np.median(errors[failures])) if failures.sum() > 0 else 0.0,
        'failure_indices': np.where(failures)[0].tolist()
    }

    return analysis
