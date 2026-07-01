# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare a prior-sensitivity variant fit against its baseline (issue #89 §7).

The robustness criterion is: does each headline quantity of the variant stay
within the *baseline's* 90% HDI (the engines report 90%, not 94%)? The loader is
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

# (quantity, filename, median_col, hdi_lo_col | None, hdi_hi_col | None).
# A series is loaded only if its file exists and carries the median column, so
# each engine contributes exactly the series it emits.
_SERIES: tuple[tuple[str, str, str, str | None, str | None], ...] = (
    ("Ey_understood", "posterior_summary_u.csv", "Ey_median", "Ey_hdi_lo", "Ey_hdi_hi"),
    ("Ey_spoken", "posterior_summary_s.csv", "Ey_median", "Ey_hdi_lo", "Ey_hdi_hi"),
    ("Ey", "posterior_summary.csv", "Ey_median", "Ey_hdi_lo", "Ey_hdi_hi"),
    ("q", "posterior_summary_q.csv", "q_median", "q_hdi_lo", "q_hdi_hi"),
    ("r", "posterior_summary_r.csv", "r_median", "r_hdi_lo", "r_hdi_hi"),
    ("p_any", "posterior_summary_p_any.csv", "p_any_median", "p_any_hdi_lo", "p_any_hdi_hi"),
    ("Ey_any", "posterior_summary_p_any.csv", "Ey_any_median", None, None),
    ("gap", "comprehension_production_gap.csv", "gap_median", "hdi_lo", "hdi_hi"),
)


def _read(dirpath: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(dirpath, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def load_headlines(dirpath: str) -> dict[str, pd.DataFrame]:
    """Return ``{quantity: DataFrame[age_months, median, hdi_lo, hdi_hi]}`` for
    every headline series present in ``dirpath`` (missing series are skipped)."""
    out: dict[str, pd.DataFrame] = {}
    for qty, fname, mcol, lo, hi in _SERIES:
        df = _read(dirpath, fname)
        if df is None or mcol not in df.columns or "age_months" not in df.columns:
            continue
        frame = pd.DataFrame({"age_months": df["age_months"], "median": df[mcol]})
        frame["hdi_lo"] = df[lo] if (lo and lo in df.columns) else np.nan
        frame["hdi_hi"] = df[hi] if (hi and hi in df.columns) else np.nan
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
        "psi_hdi_lo": float(r["psi_hdi_lo"]),
        "psi_hdi_hi": float(r["psi_hdi_hi"]),
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
    base_hdi_lo, base_hdi_hi, within_baseline_hdi`` (``age_months = -1`` for the
    ψ / P(ψ>1) scalars). ``within_baseline_hdi`` is ``None`` where the series
    carries no HDI (``Ey_any``, ``P_psi_gt_1``, four-cell).
    """
    base, var = load_headlines(baseline_dir), load_headlines(variant_dir)
    rows: list[dict] = []
    for qty in sorted(set(base) & set(var)):
        b = base[qty].set_index("age_months")
        v = var[qty].set_index("age_months")
        for age in b.index.intersection(v.index):
            bm, vm = float(b.loc[age, "median"]), float(v.loc[age, "median"])
            lo, hi = b.loc[age, "hdi_lo"], b.loc[age, "hdi_hi"]
            within = bool(lo <= vm <= hi) if pd.notna(lo) and pd.notna(hi) else None
            rows.append({
                "quantity": qty, "age_months": int(age),
                "base_median": bm, "var_median": vm, "delta": vm - bm,
                "base_hdi_lo": lo, "base_hdi_hi": hi, "within_baseline_hdi": within,
            })
    pb, pv = load_psi(baseline_dir), load_psi(variant_dir)
    if pb and pv:
        rows.append({
            "quantity": "psi", "age_months": -1,
            "base_median": pb["psi_median"], "var_median": pv["psi_median"],
            "delta": pv["psi_median"] - pb["psi_median"],
            "base_hdi_lo": pb["psi_hdi_lo"], "base_hdi_hi": pb["psi_hdi_hi"],
            "within_baseline_hdi": bool(
                pb["psi_hdi_lo"] <= pv["psi_median"] <= pb["psi_hdi_hi"]
            ),
        })
        rows.append({
            "quantity": "P_psi_gt_1", "age_months": -1,
            "base_median": pb["P_psi_gt_1"], "var_median": pv["P_psi_gt_1"],
            "delta": pv["P_psi_gt_1"] - pb["P_psi_gt_1"],
            "base_hdi_lo": np.nan, "base_hdi_hi": np.nan, "within_baseline_hdi": None,
        })
    return pd.DataFrame(rows)


def summarise(comparison: pd.DataFrame, variant_dir: str, label: str) -> dict:
    """One-row robustness verdict for a variant (feeds the §7 matrix)."""
    converged, max_rhat, min_ess = diagnostics_gate(variant_dir)
    checked = comparison.dropna(subset=["within_baseline_hdi"])
    n_within = int(checked["within_baseline_hdi"].sum())
    n_checked = int(len(checked))
    outside = sorted(
        checked.loc[~checked["within_baseline_hdi"], "quantity"].unique().tolist()
    )
    max_abs_delta = float(comparison["delta"].abs().max()) if len(comparison) else 0.0
    if converged is False:
        verdict = "NON-CONVERGED (not assessed)"
    elif not outside:
        verdict = "robust (all within baseline 90% HDI)"
    else:
        verdict = "sensitive: " + ", ".join(outside)
    return {
        "variant": label,
        "converged": converged,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_within_hdi": n_within,
        "n_checked": n_checked,
        "quantities_outside_hdi": ", ".join(outside),
        "max_abs_delta": max_abs_delta,
        "verdict": verdict,
    }
