""" Utility functions for plotting training history. """
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional


def plot_training_history(
    history: Dict[str, list],
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Plot loss, accuracy, and MAE curves over epochs.

    Args:
        history: Trainer.history dict
        save_dir: Optional directory to save PNGs
        show: Whether to call plt.show()
    """
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ---------- Loss ----------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    # ---------- Accuracy ----------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "accuracy_curve.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    # ---------- MAE (degrees) ----------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_mae"], label="Train MAE (deg)")
    plt.plot(epochs, history["val_mae"], label="Val MAE (deg)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (degrees)")
    plt.title("MAE vs Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    if save_dir is not None:
        plt.savefig(save_dir / "mae_curve.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_dashboard(history: Dict[str, list], save_path: Optional[str] = None):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # MAE
    axes[2].plot(epochs, history["train_mae"], label="Train MAE")
    axes[2].plot(epochs, history["val_mae"], label="Val MAE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MAE (deg)")
    axes[2].set_title("Training & Validation MAE")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
