"""script to plot a training history json file."""

import json
from pathlib import Path
import argparse

from aircraft_heading_gnn.utils.plot_training import (
    plot_training_history,
    plot_training_dashboard,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training history from JSON file")
    parser.add_argument(
        "--history_path",
        type=str,
        required=True,
        help="Path to training history JSON file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="plots",
        help="Directory to save the plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to display the plots interactively",
    )
    args = parser.parse_args()

    history_path = Path(args.history_path)  # or wherever you dumped that dict
    with open(history_path, "r") as f:
        history = json.load(f)

    plot_training_dashboard(history, save_path=args.save_dir)
