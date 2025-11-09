import numpy as np

def wrap_deg(angle_deg: float) -> float:
    """Wrap to [0, 360).
    Args: angle_deg: angle in degrees
    Returns: wrapped angle in [0, 360)
    """
    return angle_deg % 360.0

def normalize_angle(angle_deg: float) -> float:
    """ Normalize ang angle between -180 and 180 degrees. """
    return (angle_deg + 180.0) % 360.0 - 180.0

def ang_diff_deg(angle_1_deg: float, angle_2_deg: float) -> float:
    """Smallest signed difference  in degrees in (-180, 180]."""
    difference = (angle_1_deg - angle_2_deg)
    return normalize_angle(difference)

def circ_distance_deg(a, b):
    """Unsigned circular distance in degrees in [0, 180]."""
    return np.abs(ang_diff_deg(a, b))

def bin_heading_deg(h, bin_size=5):
    """Bin heading to integer class index, 0..(360/bin_size - 1)."""
    h = wrap_deg(h)
    n_bins = int(360 // bin_size)
    return np.floor(h / bin_size).astype(int).clip(0, n_bins-1)

def bin_center_deg(idx, bin_size=5):
    n_bins = int(360 // bin_size)
    idx = np.clip(idx, 0, n_bins-1)
    return (idx + 0.5) * bin_size
