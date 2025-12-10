"""
Training utilities and trainer class for heading prediction models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm

from aircraft_heading_gnn.utils.angles import ang_diff_deg


class HeadingLoss(nn.Module):
    """
    Custom loss for circular heading prediction.
    Combines cross-entropy with circular distance penalty.
    """

    def __init__(
        self,
        num_classes: int = 72,
        bin_size: float = 5.0,
        alpha: float = 0.5,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            num_classes: Number of heading classes
            bin_size: Size of each bin in degrees
            alpha: Weight for circular distance loss (0=pure CE, 1=pure circular)
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.num_classes = num_classes
        self.bin_size = bin_size
        self.alpha = alpha

        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, logits, targets, has_label=None):
        """
        Compute loss.

        Args:
            logits: Predicted logits [batch_size, num_classes]
            targets: Target class indices [batch_size]
            has_label: Boolean mask for valid labels [batch_size]

        Returns:
            Scalar loss
        """
        # Filter out invalid labels
        if has_label is not None:
            valid_mask = has_label
            if valid_mask.sum() == 0:
                # return a dummy 0-loss that still has a gradient path
                return torch.tensor(0.0, device=logits.device, requires_grad=True)

            logits = logits[valid_mask]
            targets = targets[valid_mask]

        # Cross-entropy loss
        ce = self.ce_loss(logits, targets)

        if self.alpha == 1.0:
            return ce.mean()

        # Circular distance loss
        # Convert bins to angles (center of each bin)
        pred_probs = torch.softmax(logits, dim=-1)
        pred_bins = torch.argmax(pred_probs, dim=-1)

        pred_angles = (pred_bins.float() * self.bin_size + self.bin_size / 2) % 360
        target_angles = (targets.float() * self.bin_size + self.bin_size / 2) % 360

        # Circular distance in degrees mapped to [-180, 180]
        angle_diff = (pred_angles - target_angles + 180) % 360 - 180
        circular_loss = torch.abs(angle_diff) / 180.0  # Normalize to [0, 1]

        # Combine losses
        total_loss = (1 - self.alpha) * ce + self.alpha * circular_loss

        return total_loss.mean()


class Trainer:
    """
    Training pipeline for heading prediction models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        save_dir: str = "./checkpoints",
        use_aux_loss: bool = False,
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function (e.g., HeadingLoss)
            device: Device to train on
            scheduler: Learning rate scheduler
            save_dir: Directory to save checkpoints
            use_aux_loss: Whether model has auxiliary tasks (expects dict output)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_aux_loss = use_aux_loss

        # Use bin size from the loss if available (default 5°)
        self.bin_size = getattr(self.criterion, "bin_size", 5.0)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_mae": [],
            "val_mae": [],
        }

        self.best_val_loss = float("inf")

    def _forward_model(self, batch):
        """
        Helper to forward the model with correct signature.
        Assumes graphs are batched with batch.batch (PyG style).
        """
        if self.use_aux_loss:
            outputs = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            logits = outputs["heading"]
        else:
            logits = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        return logits

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_batches = 0
        correct = 0
        total = 0
        total_mae = 0.0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self._forward_model(batch)

            # has_label mask for valid nodes (e.g. ignore airport node / unlabeled nodes)
            has_label = batch.has_label if hasattr(batch, "has_label") else None
            loss = self.criterion(logits, batch.y, has_label)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track batch loss (average per batch)
            total_loss += loss.item()
            total_batches += 1

            # Metrics
            if has_label is not None:
                valid_mask = has_label
                logits_valid = logits[valid_mask]
                targets_valid = batch.y[valid_mask]
            else:
                logits_valid = logits
                targets_valid = batch.y

            if logits_valid.shape[0] > 0:
                preds = torch.argmax(logits_valid, dim=-1)
                correct += (preds == targets_valid).sum().item()
                total += targets_valid.shape[0]

                # MAE in degrees with proper circular difference
                pred_angles = (preds.cpu().numpy() * self.bin_size + self.bin_size / 2) % 360
                target_angles = (
                    targets_valid.cpu().numpy() * self.bin_size + self.bin_size / 2
                ) % 360

                mae = np.mean(
                    [abs(ang_diff_deg(float(p), float(t)))
                     for p, t in zip(pred_angles, target_angles)]
                )
                total_mae += mae * targets_valid.shape[0]

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / max(total_batches, 1)
        accuracy = correct / total if total > 0 else 0.0
        avg_mae = total_mae / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy, "mae": avg_mae}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0.0
        total_batches = 0
        correct = 0
        total = 0
        total_mae = 0.0

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = batch.to(self.device)

            # Forward pass
            logits = self._forward_model(batch)

            # Compute loss
            has_label = batch.has_label if hasattr(batch, "has_label") else None
            loss = self.criterion(logits, batch.y, has_label)

            total_loss += loss.item()
            total_batches += 1

            # Metrics
            if has_label is not None:
                valid_mask = has_label
                logits_valid = logits[valid_mask]
                targets_valid = batch.y[valid_mask]
            else:
                logits_valid = logits
                targets_valid = batch.y

            if logits_valid.shape[0] > 0:
                preds = torch.argmax(logits_valid, dim=-1)
                correct += (preds == targets_valid).sum().item()
                total += targets_valid.shape[0]

                pred_angles = (preds.cpu().numpy() * self.bin_size + self.bin_size / 2) % 360
                target_angles = (
                    targets_valid.cpu().numpy() * self.bin_size + self.bin_size / 2
                ) % 360

                mae = np.mean(
                    [abs(ang_diff_deg(float(p), float(t)))
                     for p, t in zip(pred_angles, target_angles)]
                )
                total_mae += mae * targets_valid.shape[0]

        avg_loss = total_loss / max(total_batches, 1)
        accuracy = correct / total if total > 0 else 0.0
        avg_mae = total_mae / total if total > 0 else 0.0

        return {"loss": avg_loss, "accuracy": accuracy, "mae": avg_mae}

    def fit(self, num_epochs: int, early_stopping_patience: int = 10):
        """
        Train the model.

        Args:
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs without improvement before stopping
        """
        patience_counter = 0
        last_val_metrics = {"loss": float("inf"), "accuracy": 0.0, "mae": 0.0}

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()
            last_val_metrics = val_metrics

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Log metrics
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_mae"].append(val_metrics["mae"])

            print(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Train MAE: {train_metrics['mae']:.2f}°"
            )
            print(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.2f}°"
            )

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                patience_counter = 0
                print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

        # Save final model
        self.save_checkpoint("final_model.pt", epoch + 1, last_val_metrics)

        # Save training history
        self.save_history()

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.save_dir / filename)

    def save_history(self):
        """Save training history as JSON."""
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self, filename: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        return checkpoint["epoch"], checkpoint.get("metrics", {})
