# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare a prior-sensitivity variant fit against its baseline (issue #89 §7).

The robustness criterion is: does each headline quantity of the variant stay
within the *baseline's* 89% interval (the engines report an 89% interval by
default; see :mod:`vocab_growth.intervals`)? The loader is
spec-driven and tolerant of absent files, so it handles both CSV dialects with no
special-casing — the bivariate/univariate engines write ``Ey``/``q``/``gap``
series, the joint engine writes ``q``/``r``/``p_any``/``psi`` + the four-cell
composition. A variant whose fit did not converge (``r_hat > 1.01`` or ESS below
threshold) is never reported as "robust": a shifted estimate from a bad fit is
sampler noise, not prior sensitivity.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

RHAT_MAX = 1.01
ESS_THRESHOLD = 400

# (quantity, filename, median_col, ci_lo_col | None, ci_hi_col | None).
# A series is loaded only if its file exists and carries the median column, so
# each engine contributes exactly the series it emits.
_SERIES: tuple[tuple[str, str, str, str | None, str | None], ...] = (
    ("Ey_understood", "posterior_summary_u.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("Ey_spoken", "posterior_summary_s.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("Ey", "posterior_summary.csv", "Ey_median", "Ey_ci_lo", "Ey_ci_hi"),
    ("q", "posterior_summary_q.csv", "q_median", "q_ci_lo", "q_ci_hi"),
    ("r", "posterior_summary_r.csv", "r_median", "r_ci_lo", "r_ci_hi"),
    ("p_any", "posterior_summary_p_any.csv", "p_any_median", "p_any_ci_lo", "p_any_ci_hi"),
    ("Ey_any", "posterior_summary_p_any.csv", "Ey_any_median", "Ey_any_ci_lo", "Ey_any_ci_hi"),
    ("gap", "comprehension_production_gap.csv", "gap_median", "ci_lo", "ci_hi"),
)

# All loaded series are equal-tailed; only the psi scalar (loaded separately) is
# reported with a highest-density interval, so the comparison table carries an
# ``interval_kind`` column to keep the mixed table honest.
_SERIES_INTERVAL_KIND = "eti"


def _read(dirpath: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(dirpath, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def load_headlines(dirpath: str) -> dict[str, pd.DataFrame]:
    """Return ``{quantity: DataFrame[age_months, median, ci_lo, ci_hi]}`` for
    every headline series present in ``dirpath`` (missing series are skipped)."""
    out: dict[str, pd.DataFrame] = {}
    for qty, fname, mcol, lo, hi in _SERIES:
        df = _read(dirpath, fname)
        if df is None or mcol not in df.columns or "age_months" not in df.columns:
            continue
        frame = pd.DataFrame({"age_months": df["age_months"], "median": df[mcol]})
        frame["ci_lo"] = df[lo] if (lo and lo in df.columns) else np.nan
        frame["ci_hi"] = df[hi] if (hi and hi in df.columns) else np.nan
        out[qty] = frame
    return out


def load_psi(dirpath: str) -> dict[str, float] | None:
    """The VG15 association scalar summary, or ``None`` if not present."""
    df = _read(dirpath, "posterior_summary_psi.csv")
    if df is None or "psi_median" not in df.columns:
        return None
    r = df.iloc[0]
    return {
        "psi_median": float(r["psi_median"]),
        "psi_ci_lo": float(r["psi_ci_lo"]),
        "psi_ci_hi": float(r["psi_ci_hi"]),
        "P_psi_gt_1": float(r["P_psi_gt_1"]),
    }


def diagnostics_gate(dirpath: str) -> tuple[bool | None, float | None, float | None]:
    """``(converged, max_rhat, min_ess)`` from ``diagnostics.csv`` (index = params)."""
    df = _read(dirpath, "diagnostics.csv")
    if df is None or "r_hat" not in df.columns:
        return (None, None, None)
    max_rhat = float(np.nanmax(df["r_hat"].values))
    ess_cols = [c for c in ("ess_bulk", "ess_tail") if c in df.columns]
    min_ess = float(np.nanmin(df[ess_cols].min(axis=1).values)) if ess_cols else None
    converged = bool(
        max_rhat <= RHAT_MAX and min_ess is not None and min_ess >= ESS_THRESHOLD
    )
    return (converged, max_rhat, min_ess)


def compare_dirs(baseline_dir: str, variant_dir: str) -> pd.DataFrame:
    """Per-quantity, per-age comparison of a variant against its baseline.

    Columns: ``quantity, age_months, base_median, var_median, delta,
    base_ci_lo, base_ci_hi, within_baseline_ci, interval_kind`` (``age_months =
    -1`` for the ψ / P(ψ>1) scalars). ``within_baseline_ci`` is ``None`` where the
    series carries no interval (``P_psi_gt_1``, four-cell). ``interval_kind`` is
    ``"eti"`` for the equal-tailed series and ``"hdi"`` for the ψ scalar.
    """
    base, var = load_headlines(baseline_dir), load_headlines(variant_dir)
    rows: list[dict] = []
    for qty in sorted(set(base) & set(var)):
        b = base[qty].set_index("age_months")
        v = var[qty].set_index("age_months")
        for age in b.index.intersection(v.index):
            bm, vm = float(b.loc[age, "median"]), float(v.loc[age, "median"])
            lo, hi = b.loc[age, "ci_lo"], b.loc[age, "ci_hi"]
            within = bool(lo <= vm <= hi) if pd.notna(lo) and pd.notna(hi) else None
            rows.append({
                "quantity": qty, "age_months": int(age),
                "base_median": bm, "var_median": vm, "delta": vm - bm,
                "base_ci_lo": lo, "base_ci_hi": hi, "within_baseline_ci": within,
                "interval_kind": _SERIES_INTERVAL_KIND,
            })
    pb, pv = load_psi(baseline_dir), load_psi(variant_dir)
    if pb and pv:
        rows.append({
            "quantity": "psi", "age_months": -1,
            "base_median": pb["psi_median"], "var_median": pv["psi_median"],
            "delta": pv["psi_median"] - pb["psi_median"],
            "base_ci_lo": pb["psi_ci_lo"], "base_ci_hi": pb["psi_ci_hi"],
            "within_baseline_ci": bool(
                pb["psi_ci_lo"] <= pv["psi_median"] <= pb["psi_ci_hi"]
            ),
            "interval_kind": "hdi",
        })
        rows.append({
            "quantity": "P_psi_gt_1", "age_months": -1,
            "base_median": pb["P_psi_gt_1"], "var_median": pv["P_psi_gt_1"],
            "delta": pv["P_psi_gt_1"] - pb["P_psi_gt_1"],
            "base_ci_lo": np.nan, "base_ci_hi": np.nan, "within_baseline_ci": None,
            "interval_kind": None,
        })
    return pd.DataFrame(rows)


def summarise(comparison: pd.DataFrame, variant_dir: str, label: str) -> dict:
    """One-row robustness verdict for a variant (feeds the §7 matrix)."""
    converged, max_rhat, min_ess = diagnostics_gate(variant_dir)
    checked = comparison.dropna(subset=["within_baseline_ci"])
    # The column mixes Python bools with None (P_psi_gt_1 / four-cell rows), so it
    # is object dtype even after dropna; ~ on object bools yields -2/-1, not a
    # mask. Coerce before inverting.
    within = checked["within_baseline_ci"].astype(bool)
    n_within = int(within.sum())
    n_checked = int(len(checked))
    outside = sorted(checked.loc[~within, "quantity"].unique().tolist())
    max_abs_delta = float(comparison["delta"].abs().max()) if len(comparison) else 0.0
    if converged is False:
        verdict = "NON-CONVERGED (not assessed)"
    elif not outside:
        verdict = "robust (all within baseline 89% interval)"
    else:
        verdict = "sensitive: " + ", ".join(outside)
    return {
        "variant": label,
        "converged": converged,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_within_ci": n_within,
        "n_checked": n_checked,
        "quantities_outside_ci": ", ".join(outside),
        "max_abs_delta": max_abs_delta,
        "verdict": verdict,
    }
