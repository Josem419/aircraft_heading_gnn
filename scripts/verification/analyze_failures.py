#!/usr/bin/env python3
"""Failure mode analysis: cluster and explain GNN heading violations.

Loads the rollout JSON produced by ``run_rollouts.py --save-json`` and:

1. Prints summary statistics split by outcome (safe / agent-fault / structural).
2. Trains a decision tree on initial-state features to predict violations
   (separately for all violations and agent-fault-only violations).
3. Clusters agent-fault violations with k-means and characterises each cluster.

Usage
-----
    python scripts/verification/analyze_failures.py --json analysis/rollouts_100.json
    python scripts/verification/analyze_failures.py \\
        --json analysis/rollouts_100.json --clusters 4 --max-depth 5
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ---------------------------------------------------------------------------
# Feature schema  (must match keys present in the rollout JSON)
# ---------------------------------------------------------------------------

# Features available as initial-state predictors (known *before* the rollout)
INITIAL_FEATURES = [
    "initial_dist_nm",
    "initial_alt_ft",
    "initial_hdg_deg",
]

# Post-rollout diagnostic scores (useful for clustering failure severity)
ROBUSTNESS_FEATURES = [
    "rho_norm",
    "rho_sep",
    "rho_rate",
    "rho_track",
    "ref_rho_sep",
]

ALL_FEATURES = INITIAL_FEATURES + ROBUSTNESS_FEATURES

LABELS = {
    "initial_dist_nm":  "Distance from KSEA (NM)",
    "initial_alt_ft":   "Initial altitude (ft)",
    "initial_hdg_deg":  "Initial heading (°)",
    "rho_norm":         "Normalised combined robustness",
    "rho_sep":          "Separation robustness (m)",
    "rho_rate":         "Heading-rate robustness (°/s)",
    "rho_track":        "Cross-track robustness (NM)",
    "ref_rho_sep":      "Ref separation robustness (m)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_features(rollouts: list, feature_names: list) -> np.ndarray:
    rows = []
    for r in rollouts:
        rows.append([float(r.get(k, 0.0)) for k in feature_names])
    return np.array(rows, dtype=np.float64) if rows else np.empty((0, len(feature_names)))


def bar(value: float, width: int = 40) -> str:
    return "█" * max(0, int(value * width))


def print_tree(clf, feature_names: list, label_names: list = None):
    try:
        from sklearn.tree import export_text
        print(export_text(
            clf,
            feature_names=[LABELS[n] for n in feature_names],
            max_depth=clf.max_depth or 5,
        ))
    except Exception:
        pass  # fallback: no tree printing


def stats_block(rollouts: list, label: str):
    if not rollouts:
        return
    dists = [r.get("initial_dist_nm", 0.0) for r in rollouts]
    alts  = [r.get("initial_alt_ft",  0.0) for r in rollouts]
    hdgs  = [r.get("initial_hdg_deg", 0.0) for r in rollouts]
    rho_n = [r.get("rho_norm", 0.0) for r in rollouts]
    rho_s = [r.get("rho_sep",  0.0) for r in rollouts]
    rho_r = [r.get("rho_rate", 0.0) for r in rollouts]
    rho_t = [r.get("rho_track",0.0) for r in rollouts]
    n = len(rollouts)
    print(f"\n  {label}  (n={n})")
    print(f"    Distance (NM)   : {np.mean(dists):6.1f} ± {np.std(dists):.1f}  "
          f"[{min(dists):.1f}, {max(dists):.1f}]")
    print(f"    Altitude (ft)   : {np.mean(alts):6.0f} ± {np.std(alts):.0f}  "
          f"[{min(alts):.0f}, {max(alts):.0f}]")
    print(f"    Heading (°)     : {np.mean(hdgs):6.0f} ± {np.std(hdgs):.0f}")
    print(f"    ρ_norm          : {np.mean(rho_n):+8.3f} ± {np.std(rho_n):.3f}")
    print(f"    ρ_sep  (m)      : {np.mean(rho_s):+8.0f} ± {np.std(rho_s):.0f}")
    print(f"    ρ_rate (°/s)    : {np.mean(rho_r):+8.3f} ± {np.std(rho_r):.3f}")
    print(f"    ρ_track (NM)    : {np.mean(rho_t):+8.3f} ± {np.std(rho_t):.3f}")


# ---------------------------------------------------------------------------
# Per-spec breakdown table (for LaTeX table in paper)
# ---------------------------------------------------------------------------

def breakdown_table(rollouts: list, latex: bool = False) -> None:
    """Print per-spec failure counts, median robustness, and agent failure probability.

    Produces exactly the columns needed for the Failure Breakdown table:
      Spec | Structural count | Agent count | Median robustness | P_agent_fail

    Notes on 'structural' vs 'agent' per spec:
      - Separation / Combined: structural = ref_rho_sep < 0 (inherent scenario geometry).
      - Cross-track / Heading rate: no reference baseline exists → all failures are
        attributed to the agent.
    """
    n = len(rollouts)
    is_struct = [r.get("ref_rho_sep", 1.0) < 0 for r in rollouts]
    # Denominator for agent failure probability: exclude structurally-violating
    # scenarios (those where the ref path already fails separation) since they
    # cannot be attributed to the agent regardless of what it predicts.
    n_non_structural = sum(1 for s in is_struct if not s)

    def _counts(viol_mask, has_structural):
        n_viol   = sum(viol_mask)
        n_struct = sum(v and s for v, s in zip(viol_mask, is_struct)) if has_structural else None
        n_agent  = (n_viol - n_struct) if n_struct is not None else n_viol
        return n_viol, n_struct, n_agent

    # (display name, violation mask, rho key, rho unit label, has_structural)
    specs = [
        ("Cross-track",              [r.get("rho_track", 1.0) < 0 for r in rollouts],
         "rho_track", "NM",  False),
        ("Separation",               [r.get("rho_sep",   1.0) < 0 for r in rollouts],
         "rho_sep",   "m",   True),
        ("Heading rate",             [r.get("rho_rate",  1.0) < 0 for r in rollouts],
         "rho_rate",  "°/s", False),
        ("Combined (Sep + Rate)",    [r.get("status") == "VIOLATION" for r in rollouts],
         "rho_norm",  "",    True),
    ]

    rows = []
    for name, viol_mask, rho_key, unit, has_struct in specs:
        _, n_struct, n_agent = _counts(viol_mask, has_struct)
        # Median computed over failing samples only (where that spec is violated)
        rho_vals = [r.get(rho_key) for r, v in zip(rollouts, viol_mask)
                    if v and r.get(rho_key) is not None]
        med = float(np.median(rho_vals)) if rho_vals else float("nan")
        # P_agent_fail = agent-fault violations / non-structural scenarios
        denom = n_non_structural if n_non_structural > 0 else n
        p_agent = n_agent / denom
        rows.append((name, n_struct, n_agent, med, unit, p_agent))

    if not latex:
        # Plain text
        print("\n" + "─" * 85)
        print("FAILURE BREAKDOWN TABLE")
        print(f"  (n={n} rollouts, {n_non_structural} non-structural;  "
              f"Median ρ over failures only;  P_agent_fail = agent faults / non-structural)")
        print("─" * 95)
        hdr = f"  {'Specification':<28}  {'Structural':>11}  {'Agent':>7}  {'Median ρ (failures)':>20}  {'P_agent_fail':>14}"
        print(hdr)
        print("  " + "-" * 81)
        for name, n_struct, n_agent, med, unit, p_agent in rows:
            struct_str = str(n_struct) if n_struct is not None else "N/A"
            med_str    = f"{med:+.3f} {unit}".strip()
            print(
                f"  {name:<28}  {struct_str:>11}  {n_agent:>7}  "
                f"{med_str:>20}  {p_agent:>13.2%}"
            )
        print("─" * 85)
    else:
        # LaTeX tabular rows
        print("\n% --- Failure Breakdown Table (paste into \\begin{tabular}) ---")
        print(f"% n={n} rollouts, {n_non_structural} non-structural; "
              f"Median rho over failures only; P_agent_fail = agent faults / non-structural")
        print("\\hline")
        for name, n_struct, n_agent, med, unit, p_agent in rows:
            struct_str = str(n_struct) if n_struct is not None else "---"
            med_str    = f"\\num{{{med:+.3f}}} {unit}".strip() if not math.isnan(med) else "---"
            print(
                f"{name} & {struct_str} & {n_agent} & {med_str} & "
                f"\\num{{{p_agent:.4f}}} \\\\"
            )
        print("\\hline")
        print("% ---")


# ---------------------------------------------------------------------------
# Decision tree block
# ---------------------------------------------------------------------------


def run_decision_tree(all_rollouts: list, label: str, max_depth: int):
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        print("  scikit-learn not available — skipping decision tree.")
        return

    X = extract_features(all_rollouts, INITIAL_FEATURES)
    y = np.array([1 if r.get("status") == "VIOLATION" else 0 for r in all_rollouts])

    if X.shape[0] == 0 or y.sum() == 0:
        print("  No violations to train on.")
        return

    clf = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=3, random_state=0
    )
    clf.fit(X, y)

    acc = clf.score(X, y)
    pred = clf.predict(X)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    print(f"\n  {label}")
    print(f"    Train acc={acc:.2%}  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print()
    try:
        import io
        tree_str = export_text(
            clf,
            feature_names=[LABELS[n] for n in INITIAL_FEATURES],
            max_depth=max_depth,
        )
        # Indent
        for line in tree_str.splitlines():
            print(f"    {line}")
    except Exception as exc:
        print(f"    [Could not print tree: {exc}]")

    print(f"\n  Feature importances:")
    for name, imp in sorted(
        zip(INITIAL_FEATURES, clf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"    {LABELS[name]:<40}: {imp:.3f}  {bar(imp)}")


# ---------------------------------------------------------------------------
# Clustering block
# ---------------------------------------------------------------------------


def run_clustering(violations: list, n_clusters: int, label: str):
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  scikit-learn not available — skipping clustering.")
        return

    if len(violations) < n_clusters:
        print(f"  Only {len(violations)} samples — need ≥{n_clusters} to cluster.")
        return

    # Use initial state + robustness scores for clustering
    cluster_features = INITIAL_FEATURES + ["rho_sep", "rho_rate"]
    X = extract_features(violations, cluster_features)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, n_init=15, random_state=0)
    km.fit(X_scaled)
    labels = km.labels_

    print(f"\n  {label} — {n_clusters} clusters  (n={len(violations)})")
    for cid in range(n_clusters):
        mask = labels == cid
        c_data = X[mask]
        c_rollouts = [violations[i] for i, m in enumerate(mask) if m]

        # Characterise dominant failure mode
        rho_sep_mean  = float(c_data[:, cluster_features.index("rho_sep")].mean())
        rho_rate_mean = float(c_data[:, cluster_features.index("rho_rate")].mean())
        alt_mean      = float(c_data[:, cluster_features.index("initial_alt_ft")].mean())
        dist_mean     = float(c_data[:, cluster_features.index("initial_dist_nm")].mean())

        dominant = "sep" if rho_sep_mean < rho_rate_mean else "rate"
        phase    = "approach" if alt_mean < 5000 else "terminal"

        print(f"\n    Cluster {cid+1}  ({mask.sum()} violations, "
              f"{dominant}-driven, {phase}):")

        for j, fname in enumerate(cluster_features):
            col = c_data[:, j]
            print(f"      {LABELS[fname]:<42}: "
                  f"{col.mean():+8.1f} ± {col.std():.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Failure mode analysis from rollout JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--json",        required=True,
                   help="Rollout JSON produced by run_rollouts.py --save-json")
    p.add_argument("--clusters",    type=int, default=3,
                   help="Number of k-means clusters")
    p.add_argument("--max-depth",   type=int, default=4,
                   help="Maximum decision tree depth")
    p.add_argument("--table",       action="store_true",
                   help="Print per-spec failure breakdown table (plain text)")
    p.add_argument("--latex-table", action="store_true",
                   help="Print per-spec failure breakdown as LaTeX tabular rows")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.json) as f:
        data = json.load(f)

    rollouts = data["rollouts"]
    meta     = data.get("meta", {})

    all_r      = rollouts
    violations = [r for r in all_r if r.get("status") == "VIOLATION"]
    # structural: ref separation also fails (derivable from ref_rho_sep)
    structural  = [r for r in violations if r.get("ref_rho_sep", 1.0) < 0]
    agent_fault = [r for r in violations if r.get("ref_rho_sep", 1.0) >= 0]
    safe        = [r for r in all_r if r.get("status") != "VIOLATION"]

    n_total = len(all_r)
    n_viol  = len(violations)

    print("=" * 65)
    print("FAILURE MODE ANALYSIS")
    print(f"  Source : {args.json}")
    if meta:
        print(f"  Agent  : {meta.get('agent', '?')}")
        print(f"  Horizon: {meta.get('steps', '?')} steps × {meta.get('dt_s', '?')} s")
    print(f"  Total  : {n_total} rollouts")
    print(f"  Fail   : {n_viol}/{n_total} ({n_viol/max(n_total,1):.1%})")
    # Per-spec failure rates (any rollout where that spec is violated)
    n_sep   = sum(1 for r in all_r if r.get("rho_sep",   1.0) < 0)
    n_rate  = sum(1 for r in all_r if r.get("rho_rate",  1.0) < 0)
    n_track = sum(1 for r in all_r if r.get("rho_track", 1.0) < 0)
    print(f"    φ_sep   violations : {n_sep}/{n_total} ({n_sep/max(n_total,1):.1%})")
    print(f"    φ_rate  violations : {n_rate}/{n_total} ({n_rate/max(n_total,1):.1%})")
    print(f"    φ_track violations : {n_track}/{n_total} ({n_track/max(n_total,1):.1%})  [diagnostic]")
    print(f"    Structural  (ref_rho_sep < 0): {len(structural)}")
    print(f"    Agent-fault (ref passes)     : {len(agent_fault)}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 0. Per-spec breakdown table (always printed; also as LaTeX on request)
    # ------------------------------------------------------------------
    breakdown_table(all_r, latex=False)
    if args.latex_table or args.table:
        breakdown_table(all_r, latex=True)

    # ------------------------------------------------------------------
    # 1. Summary statistics by outcome
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("SUMMARY STATISTICS — by outcome (initial state context)")
    print("─" * 65)
    stats_block(safe,        "SAFE")
    stats_block(agent_fault, "AGENT-FAULT violations")
    stats_block(structural,  "STRUCTURAL violations")

    # ------------------------------------------------------------------
    # 2. Decision trees
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("DECISION TREE — predict violation from initial state")
    print("─" * 65)

    run_decision_tree(
        all_r,
        label="All violations (structural + agent-fault)  vs  safe",
        max_depth=args.max_depth,
    )

    if len(agent_fault) >= 5:
        non_structural = [r for r in all_r if not r.get("structural", False)]
        run_decision_tree(
            non_structural,
            label="Agent-fault violations vs safe  (structural excluded)",
            max_depth=args.max_depth,
        )

    # ------------------------------------------------------------------
    # 3. Clustering
    # ------------------------------------------------------------------
    print("\n" + "─" * 65)
    print("K-MEANS CLUSTERING — characterise failure groups")
    print("─" * 65)

    target_for_clustering = agent_fault if len(agent_fault) >= args.clusters else violations
    cluster_label = (
        "Agent-fault violations" if target_for_clustering is agent_fault
        else "All violations"
    )
    run_clustering(target_for_clustering, args.clusters, cluster_label)

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
