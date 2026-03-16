#!/usr/bin/env python3
"""Visualise rollout trajectories exported by run_rollouts.py --save-json.

Two plot modes:

    spatial   — Overview scatter:  initial positions of all rollouts coloured
                by outcome (SAFE green / VIOLATION red), range rings at 5/10/20/40 NM,
                airport at origin.  One file per JSON input.

    tracks    — Per-rollout track:  ego predicted track (solid blue), ADS-B
                reference (dashed green), nearby traffic (thin grey).
                Saves one PNG per rollout, or a single figure grid if --grid is set.

Usage (from workspace root):
    # Spatial overview of a saved JSON
    python scripts/verification/visualize_rollouts.py \\
        --json analysis/rollouts.json --mode spatial

    # Individual track plots for all rollouts
    python scripts/verification/visualize_rollouts.py \\
        --json analysis/rollouts.json --mode tracks --out-dir plots/rollouts/

    # Only violation tracks, 4-up grid
    python scripts/verification/visualize_rollouts.py \\
        --json analysis/rollouts.json --mode tracks --filter violations --grid

    # Both plots in one call
    python scripts/verification/visualize_rollouts.py \\
        --json analysis/rollouts.json --mode all --out-dir plots/rollouts/
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise rollout trajectories from run_rollouts.py --save-json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json",    required=True,
                   help="Path to JSON file written by run_rollouts.py --save-json")
    p.add_argument("--mode",    choices=["spatial", "tracks", "all", "report"], default="all",
                   help="Which plot(s) to produce. 'report' creates a \n"
                        "publication-style grid of near-median failures + random safe examples.")
    p.add_argument("--filter",  choices=["all", "violations", "safe"], default="all",
                   help="Subset of rollouts to show in track plots (ignored in report mode)")
    p.add_argument("--out-dir", default="plots/rollouts",
                   help="Directory to write output PNGs")
    p.add_argument("--grid",    action="store_true",
                   help="Pack track plots into a grid figure instead of one file each")
    p.add_argument("--grid-cols",   type=int, default=4,
                   help="Number of columns in --grid mode")
    p.add_argument("--max-per-grid", type=int, default=20,
                   help="Maximum panels per grid page; extra rollouts spill to "
                        "additional numbered files (e.g. tracks_grid_violations_p01.png)")
    p.add_argument("--n-failures", type=int, default=6,
                   help="Number of near-median agent-fault failures to show in report mode")
    p.add_argument("--n-safe",     type=int, default=4,
                   help="Number of random safe rollouts to show in report mode")
    p.add_argument("--report-cols", type=int, default=5,
                   help="Columns in the report grid")
    p.add_argument("--seed",       type=int, default=42,
                   help="RNG seed for random safe sample in report mode")
    p.add_argument("--dpi",     type=int, default=150)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

NM_TO_M = 1_852.0
DEG_LAT_M = 111_320.0


def _ll_to_enu(lat: float, lon: float, ref_lat: float, ref_lon: float):
    """Convert lat/lon to ENU metres relative to reference point."""
    north = (lat - ref_lat) * DEG_LAT_M
    east  = (lon - ref_lon) * DEG_LAT_M * math.cos(math.radians(ref_lat))
    return east, north


def _track_enu(track: List[List[float]], ref_lat: float, ref_lon: float):
    """Convert list of [lat,lon] to two numpy arrays (east_m, north_m)."""
    pts = [_ll_to_enu(p[0], p[1], ref_lat, ref_lon) for p in track]
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    return xs, ys


# ---------------------------------------------------------------------------
# Spatial overview
# ---------------------------------------------------------------------------
# Spatial overview helpers
# ---------------------------------------------------------------------------

# Colour / marker palette shared by all spatial plots
_SPEC_COLORS = {
    "safe":       "#2ca02c",
    "sep":        "#d62728",
    "rate":       "#1300E8",
    "track":      "#17becf",
}

_LEGEND_ENTRIES = [
    ("safe",  "o",  "Safe"),
    ("sep",   "X",  "Separation violation"),
    ("sep",   "^",  "Separation violation (structural)"),
    ("rate",  "X",  "Turn-rate violation"),
    ("rate",  "^",  "Turn-rate violation (structural)"),
    ("track", "X",  "Cross-track violation"),
    ("track", "^",  "Cross-track violation (structural)"),
]


def _classify(r: dict):
    """Return (spec_key, is_structural) for a rollout record."""
    sep_fail   = r.get("rho_sep",   1.0) < 0
    rate_fail  = r.get("rho_rate",  1.0) < 0
    track_fail = r.get("rho_track", 1.0) < 0
    structural = r.get("ref_rho_sep", 1.0) < 0

    if sep_fail:
        spec = "sep"
    elif rate_fail:
        spec = "rate"
    elif track_fail:
        spec = "track"
    else:
        spec = "safe"

    return spec, (structural and spec != "safe")


def _draw_range_rings(ax):
    for r_nm in [5, 10, 20, 40]:
        circ = plt.Circle((0, 0), r_nm * NM_TO_M, fill=False,
                           linestyle="--", linewidth=0.7, color="grey", zorder=1)
        ax.add_patch(circ)
        ax.text(0, r_nm * NM_TO_M + 200, f"{r_nm} NM", ha="center",
                va="bottom", fontsize=7, color="grey")
    ax.plot(0, 0, "k^", markersize=10, zorder=5)


def _scatter_rollouts(ax, rolls, ref_lat, ref_lon,
                      active_specs=("safe", "sep", "rate", "track")):
    """Plot initial positions; return set of legend labels actually used."""
    seen: set = set()
    for r in rolls:
        if not r.get("ego_track"):
            continue
        ex, ey = _ll_to_enu(r["ego_track"][0][0], r["ego_track"][0][1],
                             ref_lat, ref_lon)
        spec, structural = _classify(r)
        if spec not in active_specs and not (spec == "safe" and "safe" in active_specs):
            # For per-spec plots: show safe rollouts + the target spec only
            if spec != "safe":
                continue
        color  = _SPEC_COLORS[spec]
        marker = ("^" if structural else "X") if spec != "safe" else "o"
        label  = (
            ("^" if structural else "X") and
            (("Separation" if spec == "sep" else
              "Turn-rate"  if spec == "rate" else
              "Cross-track") + " violation" +
             (" (structural)" if structural else ""))
        ) if spec != "safe" else "Safe"
        ax.scatter(ex, ey, c=color, marker=marker, s=55, zorder=4, alpha=0.85)
        seen.add(label)
    return seen


def _add_spatial_legend(ax, seen_labels):
    handles = [
        plt.scatter([], [], c=_SPEC_COLORS[key], marker=marker, s=55, label=label)
        for key, marker, label in _LEGEND_ENTRIES
        if label in seen_labels
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def _finalize_spatial_ax(ax, title):
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_title(title, fontsize=10)


def plot_spatial(data: dict, out_dir: Path, dpi: int) -> None:
    """Four spatial overview plots: one combined + one per spec."""
    meta  = data["meta"]
    rolls = data["rollouts"]
    ref_lat = meta["airport_lat"]
    ref_lon = meta["airport_lon"]

    n_viol  = sum(1 for r in rolls if r["status"] == "VIOLATION")
    n_agent = sum(1 for r in rolls
                  if r["status"] == "VIOLATION" and r.get("ref_rho_sep", 1.0) >= 0)
    agent_label = meta.get("agent", "?")[:40]

    # ------------------------------------------------------------------
    # 1. Combined overview (all specs at once)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_range_rings(ax)
    seen = set()
    for r in rolls:
        if not r.get("ego_track"):
            continue
        ex, ey = _ll_to_enu(r["ego_track"][0][0], r["ego_track"][0][1], ref_lat, ref_lon)
        spec, structural = _classify(r)
        color  = _SPEC_COLORS[spec]
        marker = ("^" if structural else "X") if spec != "safe" else "o"
        ax.scatter(ex, ey, c=color, marker=marker, s=55, zorder=4, alpha=0.85)
        lbl = {
            "safe":  "Safe",
            "sep":   "Separation violation" + (" (structural)" if structural else ""),
            "rate":  "Turn-rate violation"  + (" (structural)" if structural else ""),
            "track": "Cross-track violation"+ (" (structural)" if structural else ""),
        }[spec]
        seen.add(lbl)
    _add_spatial_legend(ax, seen)
    _finalize_spatial_ax(ax,
        f"Rollout initial positions — KSEA (all specs)\n"
        f"{n_viol}/{len(rolls)} violations  "
        f"({n_agent} agent-fault, {n_viol - n_agent} structural)  |  {agent_label}")
    fig.tight_layout()
    out = out_dir / "spatial_overview.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ------------------------------------------------------------------
    # 2–4. Per-spec plots (violations for that spec + safe backdrop)
    # ------------------------------------------------------------------
    per_spec = [
        ("sep",   "Separation",  "spatial_separation.png"),
        ("rate",  "Turn rate",   "spatial_turn_rate.png"),
        ("track", "Cross-track", "spatial_cross_track.png"),
    ]
    for spec_key, spec_name, fname in per_spec:
        n_spec = sum(1 for r in rolls if r.get(f"rho_{spec_key.replace('sep','sep').replace('rate','rate').replace('track','track')}", 1.0) < 0)
        # map spec_key to json rho field
        rho_field = {"sep": "rho_sep", "rate": "rho_rate", "track": "rho_track"}[spec_key]
        n_spec = sum(1 for r in rolls if r.get(rho_field, 1.0) < 0)
        n_struct = sum(1 for r in rolls
                       if r.get(rho_field, 1.0) < 0 and r.get("ref_rho_sep", 1.0) < 0)

        fig, ax = plt.subplots(figsize=(9, 9))
        _draw_range_rings(ax)
        seen = set()
        for r in rolls:
            if not r.get("ego_track"):
                continue
            ex, ey = _ll_to_enu(r["ego_track"][0][0], r["ego_track"][0][1],
                                 ref_lat, ref_lon)
            this_fail  = r.get(rho_field, 1.0) < 0
            structural = r.get("ref_rho_sep", 1.0) < 0
            if this_fail:
                color  = _SPEC_COLORS[spec_key]
                marker = "^" if structural else "X"
                lbl    = (f"{spec_name} violation" +
                          (" (structural)" if structural else ""))
            else:
                color, marker, lbl = _SPEC_COLORS["safe"], "o", "Safe"
            ax.scatter(ex, ey, c=color, marker=marker, s=55, zorder=4, alpha=0.85)
            seen.add(lbl)
        _add_spatial_legend(ax, seen)
        _finalize_spatial_ax(ax,
            f"{spec_name} spec — initial positions  |  KSEA\n"
            f"{n_spec}/{len(rolls)} violations  "
            f"({n_spec - n_struct} agent-fault, {n_struct} structural)  |  {agent_label}")
        fig.tight_layout()
        out = out_dir / fname
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Per-rollout track plot (standalone or grid cell)
# ---------------------------------------------------------------------------

def _draw_track(ax: plt.Axes, r: dict, ref_lat: float, ref_lon: float) -> None:
    """Draw a single rollout onto *ax*."""
    ego_xs, ego_ys = _track_enu(r["ego_track"], ref_lat, ref_lon)
    ref_xs, ref_ys = _track_enu(r["ref_track"], ref_lat, ref_lon)

    # Traffic
    for icao, track in r.get("traffic_tracks", {}).items():
        if track:
            tx, ty = _track_enu(track, ref_lat, ref_lon)
            ax.plot(tx / 1000, ty / 1000, color="0.7", linewidth=0.8,
                    alpha=0.7, zorder=2)
            ax.plot(tx[0] / 1000, ty[0] / 1000, "s", color="0.5",
                    markersize=3, zorder=3)

    # Reference (ADS-B ground truth)
    ax.plot(ref_xs / 1000, ref_ys / 1000, "--", color="#2ca02c",
            linewidth=1.4, label="ADS-B ref", zorder=4)
    ax.plot(ref_xs[0] / 1000, ref_ys[0] / 1000, "^", color="#2ca02c",
            markersize=6, zorder=5)

    # Agent track
    color = "#d62728" if r["status"] == "VIOLATION" else "#1f77b4"
    ax.plot(ego_xs / 1000, ego_ys / 1000, "-", color=color,
            linewidth=1.8, label="Agent", zorder=6)
    ax.plot(ego_xs[0] / 1000, ego_ys[0] / 1000, "o", color=color,
            markersize=6, zorder=7)

    # Airport
    ax.plot(0, 0, "k^", markersize=7, zorder=8)

    # Confidence ribbon (background colour proportional to mean conf)
    mean_conf = (sum(r.get("confidences", [0])) /
                 max(len(r.get("confidences", [1])), 1))

    status_txt = r["status"]
    ax.set_title(
        f"{r['callsign']} #{r['rollout']}  [{status_txt}]\n"
        f"{r['initial_dist_nm']:.1f} NM  "
        f"{r['initial_alt_ft']:.0f} ft  "
        f"hdg={r['initial_hdg_deg']:.0f}°  "
        f"conf={mean_conf:.0f}%",
        fontsize=7, pad=3,
    )
    ax.set_xlabel("East (km)", fontsize=7)
    ax.set_ylabel("North (km)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Robustness annotation (rho_track stored in NM in JSON)
    rho_sep   = r.get("rho_sep",   float("nan"))
    rho_rate  = r.get("rho_rate",  float("nan"))
    rho_track = r.get("rho_track", float("nan"))  # NM
    rho_norm  = r.get("rho_norm",  float("nan"))
    ax.annotate(
        f"ρ_norm={rho_norm:+.3f}\nρ_sep={rho_sep:.0f} m\nρ_rate={rho_rate:.3f} °/s",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=6, va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6),
    )


def _save_one_grid(chunk: list, meta: dict, out_path: Path,
                   ncols: int, dpi: int, page: int, total_pages: int) -> None:
    """Render a single page of the track grid and save it."""
    ref_lat = meta["airport_lat"]
    ref_lon = meta["airport_lon"]
    nrows = math.ceil(len(chunk) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows),
                             squeeze=False)
    for idx, r in enumerate(chunk):
        row, col = divmod(idx, ncols)
        _draw_track(axes[row][col], r, ref_lat, ref_lon)
    for idx in range(len(chunk), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)
    n_viol = sum(1 for r in chunk if r["status"] == "VIOLATION")
    page_str = f"  (page {page}/{total_pages})" if total_pages > 1 else ""
    fig.suptitle(
        f"Rollout track overview — {n_viol}/{len(chunk)} violations{page_str}",
        fontsize=11, y=1.002,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {out_path}")


def plot_tracks_grid(rolls: list, meta: dict, out_path: Path,
                     ncols: int, dpi: int, max_per_page: int = 20) -> None:
    """Save one or more grid PNG files, capped at max_per_page panels each."""
    chunks = [rolls[i:i + max_per_page]
              for i in range(0, max(len(rolls), 1), max_per_page)]
    total = len(chunks)
    stem   = out_path.stem
    suffix = out_path.suffix
    parent = out_path.parent
    for page, chunk in enumerate(chunks, start=1):
        if total == 1:
            path = out_path
        else:
            path = parent / f"{stem}_p{page:02d}{suffix}"
        _save_one_grid(chunk, meta, path, ncols, dpi, page, total)


def plot_tracks_individual(rolls: list, meta: dict,
                           out_dir: Path, dpi: int) -> None:
    ref_lat = meta["airport_lat"]
    ref_lon = meta["airport_lon"]

    for r in rolls:
        fig, ax = plt.subplots(figsize=(6, 6))
        _draw_track(ax, r, ref_lat, ref_lon)
        fig.tight_layout()
        fname = f"track_{r['rollout']:04d}_{r['callsign']}_{r['status']}.png"
        fig.savefig(out_dir / fname, dpi=dpi)
        plt.close(fig)

    print(f"  Saved {len(rolls)} track PNGs → {out_dir}/")


# ---------------------------------------------------------------------------
# Report grid: near-median agent-fault failures + random safe examples
# ---------------------------------------------------------------------------

def _select_report_rollouts(
    rollouts: list,
    n_failures: int,
    n_safe: int,
    rng: np.random.Generator,
):
    """Return (failure_picks, safe_picks).

    Failures are the agent-fault violations (ref passes, agent fails) whose
    rho_norm is closest to the median agent-fault rho_norm — i.e. typical
    failures, not the absolute worst.  Safe picks are sampled uniformly at
    random.
    """
    agent_faults = [
        r for r in rollouts
        if r.get("status") == "VIOLATION" and r.get("ref_rho_sep", 1.0) >= 0
    ]
    safe_all = [r for r in rollouts if r.get("status") != "VIOLATION"]

    # Near-median failures: sort by distance from median rho_norm
    if agent_faults:
        rho_vals = np.array([r.get("rho_norm", 0.0) for r in agent_faults])
        med = float(np.median(rho_vals))
        order = np.argsort(np.abs(rho_vals - med))
        fail_picks = [agent_faults[i] for i in order[:n_failures]]
        # Re-sort picks from worst to least-bad for left-to-right presentation
        fail_picks.sort(key=lambda r: r.get("rho_norm", 0.0))
    else:
        fail_picks = []

    # Random safe sample
    if safe_all:
        n_pick = min(n_safe, len(safe_all))
        idx = rng.choice(len(safe_all), size=n_pick, replace=False)
        safe_picks = [safe_all[i] for i in sorted(idx)]
    else:
        safe_picks = []

    return fail_picks, safe_picks


def plot_report_grid(
    rollouts: list,
    meta: dict,
    out_path: Path,
    n_failures: int,
    n_safe: int,
    ncols: int,
    dpi: int,
    seed: int = 42,
) -> None:
    """Publication-style grid: near-median failures (red) + safe examples (blue).

    Failures appear in the first row(s), safe examples in the last row(s).
    Each cell has a coloured header band and a robustness annotation.
    """
    rng = np.random.default_rng(seed)
    fail_picks, safe_picks = _select_report_rollouts(rollouts, n_failures, n_safe, rng)

    all_picks = fail_picks + safe_picks
    if not all_picks:
        print("  No rollouts to plot in report mode.")
        return

    ref_lat = meta["airport_lat"]
    ref_lon = meta["airport_lon"]

    nrows = math.ceil(len(all_picks) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3.8 * nrows),
        squeeze=False,
    )

    for idx, r in enumerate(all_picks):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        _draw_track(ax, r, ref_lat, ref_lon)

        # Colour header band to distinguish failure vs safe
        is_fail = r.get("status") == "VIOLATION"
        band_color = "#ffcccc" if is_fail else "#ccffcc"
        ax.set_facecolor(band_color)
        ax.patch.set_alpha(0.15)

        # Panel label
        kind_str = "Near-median failure" if is_fail else "Safe example"
        ax.set_title(
            f"[{kind_str}] {r['callsign']}\n"
            f"{r['initial_dist_nm']:.1f} NM  "
            f"{r['initial_alt_ft']:.0f} ft  "
            f"hdg={r['initial_hdg_deg']:.0f}°  "
            f"ρ_norm={r.get('rho_norm', float('nan')):+.3f}",
            fontsize=7, pad=3,
        )

    # Hide unused subplots
    for idx in range(len(all_picks), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    n_viol_total = sum(1 for r in rollouts if r.get("status") == "VIOLATION")
    n_agent = sum(
        1 for r in rollouts
        if r.get("status") == "VIOLATION" and r.get("ref_rho_sep", 1.0) >= 0
    )
    fig.suptitle(
        f"Failure and safe trajectory examples — KSEA  |  "
        f"{n_viol_total}/{len(rollouts)} violations  "
        f"({n_agent} agent-fault)\n"
        f"Showing {len(fail_picks)} near-median agent-fault failures  +  "
        f"{len(safe_picks)} random safe examples",
        fontsize=10, y=1.002,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved report grid: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        sys.exit(f"ERROR: JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    meta  = data["meta"]
    rolls = data["rollouts"]
    print(f"Loaded {len(rolls)} rollouts from {json_path}")
    print(f"  Agent    : {meta.get('agent','?')}")
    print(f"  Horizon  : {meta.get('steps','?')} steps × {meta.get('dt_s','?')} s")
    n_viol_meta = meta.get('violations') or meta.get('final_violations', '?')
    print(f"  Violations: {n_viol_meta}/{len(rolls)}")

    # Apply filter
    if args.filter == "violations":
        filtered = [r for r in rolls if r["status"] == "VIOLATION"]
    elif args.filter == "safe":
        filtered = [r for r in rolls if r["status"] == "SAFE"]
    else:
        filtered = rolls

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        workspace_root = Path(__file__).resolve().parents[2]
        out_dir = workspace_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("spatial", "all"):
        print("\n[Spatial overview]")
        plot_spatial(data, out_dir, args.dpi)

    if args.mode in ("tracks", "all"):
        if not filtered:
            print("No rollouts match the --filter; skipping track plots.")
        elif args.grid:
            print(f"\n[Track grid — {len(filtered)} rollouts, max {args.max_per_grid} per page]")
            grid_path = out_dir / f"tracks_grid_{args.filter}.png"
            plot_tracks_grid(filtered, meta, grid_path, args.grid_cols, args.dpi,
                             max_per_page=args.max_per_grid)
        else:
            print(f"\n[Individual track plots — {len(filtered)} rollouts]")
            plot_tracks_individual(filtered, meta, out_dir, args.dpi)

    if args.mode == "report":
        print(f"\n[Report grid — {args.n_failures} near-median failures + "
              f"{args.n_safe} random safe]")
        report_path = out_dir / "report_grid.png"
        plot_report_grid(
            rolls, meta, report_path,
            n_failures=args.n_failures,
            n_safe=args.n_safe,
            ncols=args.report_cols,
            dpi=args.dpi,
            seed=args.seed,
        )

    print("Done.")


if __name__ == "__main__":
    main()
