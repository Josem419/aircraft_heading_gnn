"""
Visualization utilities for aircraft trajectories and graph structures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple
from pathlib import Path

from aircraft_heading_gnn.utils.angles import wrap_deg, ang_diff_deg


def plot_trajectories(
    df: pd.DataFrame,
    airport_lat: Optional[float] = None,
    airport_lon: Optional[float] = None,
    airport_icao: Optional[str] = None,
    color_by: str = 'icao24',
    max_trajectories: int = 50,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Plot aircraft trajectories on a map.
    
    Args:
        df: DataFrame with trajectory data (must have trajectory_id)
        airport_lat: Airport latitude (for reference point)
        airport_lon: Airport longitude (for reference point)
        color_by: Column to use for coloring ('icao24', 'trajectory_id', 'velocity', etc.)
        max_trajectories: Maximum number of trajectories to plot
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique trajectories
    if 'trajectory_id' in df.columns:
        unique_trajs = df['trajectory_id'].unique()
    else:
        unique_trajs = df['icao24'].unique()
    
    # Limit number of trajectories for clarity
    if len(unique_trajs) > max_trajectories:
        unique_trajs = np.random.choice(unique_trajs, max_trajectories, replace=False)
    
    # Plot each trajectory
    for traj_id in unique_trajs:
        if 'trajectory_id' in df.columns:
            traj_data = df[df['trajectory_id'] == traj_id].sort_values('time')
        else:
            traj_data = df[df['icao24'] == traj_id].sort_values('time')
        
        if color_by == 'velocity' and 'velocity' in df.columns:
            scatter = ax.scatter(
                traj_data['lon'], traj_data['lat'],
                c=traj_data['velocity'], cmap='viridis',
                s=10, alpha=0.6
            )
        else:
            ax.plot(
                traj_data['lon'], traj_data['lat'],
                alpha=0.5, linewidth=1
            )
    
    # Plot airport if provided
    if airport_lat is not None and airport_lon is not None:
        ax.plot(airport_lon, airport_lat, 'r*', markersize=20, label=f"Airport: {airport_icao or ''}")
        
        # Add circles for terminal area boundaries
        for radius_nm in [10, 20, 30, 40]:
            # Approximate conversion: 1 nm ≈ 1/60 degree
            radius_deg = radius_nm / 60.0
            circle = plt.Circle(
                (airport_lon, airport_lat), radius_deg,
                fill=False, color='gray', linestyle='--', alpha=0.3
            )
            ax.add_patch(circle)
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(f'Aircraft Trajectories (n={len(unique_trajs)})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if color_by == 'velocity':
        plt.colorbar(scatter, ax=ax, label='Velocity (knots)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_heading_distribution(
    df: pd.DataFrame,
    bins: int = 72,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot heading distribution as a polar histogram.
    
    Args:
        df: DataFrame with heading column
        bins: Number of bins (default 72 = 5 degree bins)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Convert headings to radians
    headings_rad = np.radians(df['heading'].values)
    
    # Create histogram
    counts, bin_edges = np.histogram(headings_rad, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot as bar chart
    width = 2 * np.pi / bins
    bars = ax.bar(bin_centers, counts, width=width, alpha=0.7)
    
    # Color bars by direction
    colors = plt.cm.hsv(bin_centers / (2 * np.pi))
    for bar, color in zip(bars, colors):
        bar.set_facecolor(color)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Heading Distribution', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_temporal_features(
    df: pd.DataFrame,
    trajectory_id: Optional[int] = None,
    features: List[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
):
    """
    Plot temporal evolution of features for a trajectory.
    
    Args:
        df: DataFrame with trajectory data
        trajectory_id: Specific trajectory to plot (None = random)
        features: List of features to plot
        figsize: Figure size
        save_path: Optional path to save figure
    """
    if features is None:
        features = ['heading', 'velocity', 'geoaltitude', 'vertrate']
    
    # Select trajectory
    if trajectory_id is None and 'trajectory_id' in df.columns:
        trajectory_id = df['trajectory_id'].iloc[0]
    
    if 'trajectory_id' in df.columns:
        traj_data = df[df['trajectory_id'] == trajectory_id].sort_values('time')
    else:
        # Use first icao24
        icao = df['icao24'].iloc[0]
        traj_data = df[df['icao24'] == icao].sort_values('time')
    
    # Create subplots
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)
    
    if n_features == 1:
        axes = [axes]
    
    # Normalize time to start at 0
    time_normalized = traj_data['time'] - traj_data['time'].iloc[0]
    
    for ax, feature in zip(axes, features):
        if feature in traj_data.columns:
            ax.plot(time_normalized, traj_data[feature], linewidth=2)
            ax.set_ylabel(feature)
            ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    axes[0].set_title(f'Trajectory Features (ID: {trajectory_id})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_graph_snapshot(
    graph_data,
    airport_lat: Optional[float] = None,
    airport_lon: Optional[float] = None,
    show_edges: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
):
    """
    Visualize a single graph snapshot with nodes and edges.
    
    Args:
        graph_data: PyTorch Geometric Data object
        airport_lat: Airport latitude
        airport_lon: Airport longitude
        show_edges: Whether to show edge connections
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract positions
    positions = graph_data.pos.numpy()
    lons = positions[:, 1]
    lats = positions[:, 0]
    
    # Plot edges if requested
    if show_edges and graph_data.edge_index.shape[1] > 0:
        edge_index = graph_data.edge_index.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            ax.plot(
                [lons[src], lons[dst]],
                [lats[src], lats[dst]],
                'gray', alpha=0.3, linewidth=0.5
            )
    
    # Plot nodes
    # Color by whether they have labels
    if hasattr(graph_data, 'has_label'):
        colors = ['green' if has_lbl else 'red' 
                  for has_lbl in graph_data.has_label.numpy()]
    else:
        colors = 'blue'
    
    ax.scatter(lons, lats, c=colors, s=100, alpha=0.7, edgecolors='black')
    
    # Plot airport
    if airport_lat is not None and airport_lon is not None:
        ax.plot(airport_lon, airport_lat, 'r*', markersize=20, label='Airport')
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(f'Graph Snapshot (Nodes: {len(lats)}, Edges: {graph_data.edge_index.shape[1]})')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_heading_prediction_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    Visualize heading prediction errors.
    
    Args:
        predictions: Predicted heading bins or angles
        targets: True heading bins or angles
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Convert bins to angles if needed (assuming 5-degree bins)
    if predictions.max() < 360:
        pred_angles = (predictions * 5 + 2.5) % 360
        target_angles = (targets * 5 + 2.5) % 360
    else:
        pred_angles = predictions
        target_angles = targets
    
    # Calculate circular errors
    errors = np.array([ang_diff_deg(p, t) for p, t in zip(pred_angles, target_angles)])
    
    # Histogram of errors
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', label='Zero Error')
    axes[0].set_xlabel('Heading Error (degrees)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Heading Prediction Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot: predicted vs actual
    axes[1].scatter(target_angles, pred_angles, alpha=0.5, s=10)
    axes[1].plot([0, 360], [0, 360], 'r--', label='Perfect Prediction')
    axes[1].set_xlabel('True Heading (degrees)')
    axes[1].set_ylabel('Predicted Heading (degrees)')
    axes[1].set_title('Predicted vs True Heading')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 360)
    axes[1].set_ylim(0, 360)
    
    # Add statistics
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    fig.suptitle(f'MAE: {mae:.2f}°, RMSE: {rmse:.2f}°', fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_dataset_statistics(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
):
    """
    Create a comprehensive overview of dataset statistics.
    
    Args:
        df: Preprocessed DataFrame
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # 1. Trajectory length distribution
    if 'trajectory_id' in df.columns:
        traj_lengths = df.groupby('trajectory_id').size()
        axes[0].hist(traj_lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Trajectory Length (points)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Trajectory Length Distribution')
        axes[0].grid(True, alpha=0.3)
    
    # 2. Velocity distribution
    if 'velocity' in df.columns:
        axes[1].hist(df['velocity'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Velocity (knots)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Velocity Distribution')
        axes[1].grid(True, alpha=0.3)
    
    # 3. Altitude distribution
    if 'geoaltitude' in df.columns:
        altitude_ft = df['geoaltitude'].dropna() * 3.28084  # Convert m to ft
        axes[2].hist(altitude_ft, bins=50, alpha=0.7, edgecolor='black')
        axes[2].set_xlabel('Altitude (feet)')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Altitude Distribution')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Vertical rate distribution
    if 'vertrate' in df.columns:
        axes[3].hist(df['vertrate'].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[3].set_xlabel('Vertical Rate (m/s)')
        axes[3].set_ylabel('Count')
        axes[3].set_title('Vertical Rate Distribution')
        axes[3].grid(True, alpha=0.3)
    
    # 5. Turn rate distribution (if available)
    if 'turn_rate' in df.columns:
        turn_rates = df['turn_rate'].dropna()
        # Filter outliers for visualization
        turn_rates = turn_rates[(turn_rates > -10) & (turn_rates < 10)]
        axes[4].hist(turn_rates, bins=50, alpha=0.7, edgecolor='black')
        axes[4].set_xlabel('Turn Rate (deg/s)')
        axes[4].set_ylabel('Count')
        axes[4].set_title('Turn Rate Distribution')
        axes[4].grid(True, alpha=0.3)
    
    # 6. Number of aircraft over time
    time_counts = df.groupby('time').size()
    axes[5].plot(time_counts.index - time_counts.index.min(), time_counts.values)
    axes[5].set_xlabel('Time (seconds from start)')
    axes[5].set_ylabel('Number of Aircraft')
    axes[5].set_title('Aircraft Count Over Time')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
