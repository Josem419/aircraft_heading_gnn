import numpy as np
import pandas as pd

from aircraft_heading_gnn.utils.angles import bin_heading_deg, ang_diff_deg

def compute_future_heading_bins(df, delta_s=15, bin_size=5, min_turn_deg=None):
    """
    df: rows have 'time', 'icao24', 'heading' (deg). 
    Returns a new DataFrame with columns:
      - 'label_bin': int in [0, 360/bin_size)
      - 'delta_heading': signed deg (future - now, circular)
      - 'is_maneuver': bool (if min_turn_deg specified)
    """
    # For each aircraft, shift heading by +delta_s
    df_sorted = df.sort_values(['icao24', 'time']).copy()
    # Future heading via merge-asof-like logic: assume integer seconds
    future_time = df_sorted['time'] + int(delta_s)
    df_sorted['future_time'] = future_time

    # Build a lookup for (icao24, time)-> heading
    # Index maps
    df_sorted['_idx'] = np.arange(len(df_sorted))
    lut = df_sorted.set_index(['icao24','time'])['heading']

    future_heading = []
    ok = []
    for i in range(len(df_sorted)):
        k = (df_sorted.iloc[i]['icao24'], int(df_sorted.iloc[i]['future_time']))
        if k in lut.index:
            future_heading.append(lut.loc[k])
            ok.append(True)
        else:
            future_heading.append(np.nan)
            ok.append(False)
    df_sorted['future_heading'] = np.array(future_heading, dtype=float)
    df_sorted['has_label'] = np.array(ok, dtype=bool)

    # Label bin
    nmask = df_sorted['has_label']
    labels = np.full(len(df_sorted), -1, dtype=int)
    labels[nmask.to_numpy()] = (df_sorted.loc[nmask, 'future_heading'] // bin_size).astype(int)
    df_sorted['label_bin'] = labels

    # delta heading
    def circ_diff(a, b):
        d = (a - b + 180.0) % 360.0 - 180.0
        return d
    df_sorted['delta_heading'] = circ_diff(df_sorted['future_heading'], df_sorted['heading'])

    # optional maneuver mask
    if min_turn_deg is not None:
        df_sorted['is_maneuver'] = df_sorted['has_label'] & (df_sorted['delta_heading'].abs() >= float(min_turn_deg))
    else:
        df_sorted['is_maneuver'] = False

    return df_sorted

