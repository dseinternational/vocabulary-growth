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

Three ways a comparison can be worthless while still producing numbers, each
detected here rather than left for a reader to notice:

* **A stale pairing.** The baseline is a *moving* target — the model of record
  gets refitted. Comparing a variant fitted before that against the new baseline
  silently reports a mixture of the variant's effect and whatever else changed.
  :func:`definition_mismatch` diffs the two recorded definitions and expects them
  to differ only in the variant's own override keys (plus the naming fields), so
  an intervening definition change is caught by name.
* **Collapsed coverage.** The series are merged on ``age_months``, so a variant
  whose grid differs contributes only the intersection. That reads as a normal
  comparison over fewer points. :func:`coverage_report` measures the shortfall
  against the baseline's own rows and names the series that went missing.
* **A variant that never reported at all** — not fitted, or fitted and stopped by
  the convergence gate. Those used to be skipped with a console note, leaving the
  matrix silent about them; they are now rows with a status.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from dse_research_utils.statistics.diagnostics import ESS_THRESHOLD, RHAT_MAX

#: Definition fields that always differ between a baseline and its variant, and
#: whose difference carries no information: the variant's directory name and the
#: banner printed at the top of its fit log.
NAMING_FIELDS = frozenset({"config_name", "banner", "model_id"})

#: Coverage below this fraction of the baseline's own comparable rows means the
#: age grids disagree badly enough that the verdict is not about priors.
MIN_COVERAGE = 0.9

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


def load_beta_lag(dirpath: str) -> dict[str, float] | None:
    """The VG16 cross-lag coefficient, or ``None`` if the fit carries no lag.

    Read from ``diagnostics.csv`` rather than from a ``posterior_summary_*``
    file, because no engine persists ``beta_lag`` as a summary series -- it is a
    scalar on the model, not a curve over ages. Every fit writes
    ``diagnostics.csv``, so this works on fits made before this function
    existed.

    Without this, a VG16 sensitivity compares only the eight trajectory series
    in ``_SERIES`` and returns a verdict that says nothing about the one
    quantity VG16 exists to estimate: a prior or scope change could halve the
    cross-lag and still be scored **robust** because the trajectories did not
    move. The interval is the equal-tailed 89% ``diagnostics.csv`` reports.
    """
    df = _read(dirpath, "diagnostics.csv")
    if df is None or not len(df.columns):
        return None
    # ``_read`` does a plain ``read_csv``, so the parameter names arrive as the
    # first (unnamed) column rather than as the index.
    df = df.set_index(df.columns[0])
    for column in ("mean", "eti89_lb", "eti89_ub"):
        if column not in df.columns:
            return None
    if "beta_lag" not in df.index:
        return None
    r = df.loc["beta_lag"]
    return {
        "beta_lag_median": float(r["mean"]),
        "beta_lag_ci_lo": float(r["eti89_lb"]),
        "beta_lag_ci_hi": float(r["eti89_ub"]),
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


def _manifest(dirpath: str) -> dict | None:
    """The fit manifest in ``dirpath``, or ``None`` if it has none."""
    path = os.path.join(dirpath, "fit_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fit_created_at(dirpath: str) -> str | None:
    """When the fit in ``dirpath`` was made, from its manifest."""
    manifest = _manifest(dirpath)
    return None if manifest is None else manifest.get("created_at_utc")


def definition_mismatch(
    baseline_dir: str, variant_dir: str, override_keys: set[str]
) -> list[str]:
    """Definition fields that differ but should not — empty when the pair is sound.

    A variant fit is only a controlled comparison if it differs from the baseline
    in the fields the registry overrides and nothing else. ``override_keys`` is
    that expected set; anything else in the diff means the two fits were made
    under different model definitions, so the delta is not attributable to the
    variant. This is the check that catches a **stale pairing** — a variant
    fitted before the model of record was refitted under a changed definition,
    which produces a perfectly well-formed and completely misleading matrix row.

    Returns ``[]`` when either manifest is unreadable: an unverifiable pairing is
    reported through the status column rather than as a false mismatch.
    """
    base_m, var_m = _manifest(baseline_dir), _manifest(variant_dir)
    if base_m is None or var_m is None:
        return []
    base = (base_m.get("model") or {}).get("definition") or {}
    var = (var_m.get("model") or {}).get("definition") or {}
    if not base or not var:
        return []
    allowed = NAMING_FIELDS | set(override_keys)
    return sorted(
        k
        for k in set(base) | set(var)
        if k not in allowed and base.get(k) != var.get(k)
    )


def coverage_report(baseline_dir: str, variant_dir: str) -> tuple[int, int, list[str]]:
    """``(baseline_rows, shared_rows, missing_series)`` for a variant pairing.

    ``baseline_rows`` counts every age the baseline reports across its headline
    series; ``shared_rows`` counts those :func:`compare_dirs` will actually be
    able to pair up. ``missing_series`` names quantities the baseline reports and
    the variant does not at all.

    **This must use exactly the matching rule** :func:`compare_dirs` **uses**, or
    it reports coverage the comparison does not have. The rule is an exact match
    on ``age_months``, which is safe for the query-grid series (integer months
    from ``ages_query``) and consequential for the plot-grid ones: ``gap`` is
    emitted on ``np.linspace(min_age, max_age, n_plot)``, so two fits share its
    ages only if they share an age range. A variant that restricts the pool —
    ``dse-native-only`` is the live example — gets a different linspace, and the
    intersection collapses to the handful of points that coincide by arithmetic
    accident: 3 of 355 in the 2026-08-16 run, which had been reported as a normal
    "sensitive: gap" verdict. Measuring coverage the same way turns that into the
    partial-coverage status it is.
    """
    base, var = load_headlines(baseline_dir), load_headlines(variant_dir)
    baseline_rows = sum(len(frame) for frame in base.values())
    shared_rows = 0
    for qty, frame in base.items():
        if qty not in var:
            continue
        b_ages = pd.Index(frame["age_months"])
        v_ages = pd.Index(var[qty]["age_months"])
        shared_rows += len(b_ages.intersection(v_ages))
    missing = sorted(set(base) - set(var))
    return baseline_rows, shared_rows, missing


def failed_fit_dir(failed_root: str, model_id: str, config_name: str) -> str | None:
    """The most recent retained failed fit for ``<model_id>-<config_name>``.

    A fit stopped by the convergence gate is moved to ``output/failed/`` with a
    UTC timestamp appended, so it is invisible to a ``models/`` lookup. Finding it
    is what lets a non-converged variant appear in the matrix **with its reason**
    rather than as a blank — the requirement recorded in
    ``notes/202608142000-refit-run-record-and-disk-failure.md`` §5b.
    """
    if not os.path.isdir(failed_root):
        return None
    matches = sorted(glob.glob(os.path.join(failed_root, f"{model_id}-{config_name}-*")))
    return matches[-1] if matches else None


def summarise_absent(label: str, status: str, reason: str, variant_dir: str | None = None) -> dict:
    """A matrix row for a variant that produced no comparable summaries.

    ``status`` is ``"not-fitted"`` or ``"failed"``. A failed fit still has its
    diagnostics, so its R-hat, ESS and failing parameters are reported: that is
    the difference between "this variant does not sample" (an informative
    negative) and "nobody ran it".
    """
    row = {
        "variant": label,
        "status": status,
        "converged": None,
        "max_rhat": None,
        "min_ess": None,
        "n_within_ci": 0,
        "n_checked": 0,
        "coverage": None,
        "quantities_outside_ci": "",
        "max_abs_delta": None,
        "verdict": reason,
    }
    if variant_dir:
        converged, max_rhat, min_ess = diagnostics_gate(variant_dir)
        row["converged"] = converged
        row["max_rhat"] = max_rhat
        row["min_ess"] = min_ess
    return row


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
    bb, bv = load_beta_lag(baseline_dir), load_beta_lag(variant_dir)
    if bb and bv:
        rows.append({
            "quantity": "beta_lag", "age_months": -1,
            "base_median": bb["beta_lag_median"], "var_median": bv["beta_lag_median"],
            "delta": bv["beta_lag_median"] - bb["beta_lag_median"],
            "base_ci_lo": bb["beta_lag_ci_lo"], "base_ci_hi": bb["beta_lag_ci_hi"],
            "within_baseline_ci": bool(
                bb["beta_lag_ci_lo"]
                <= bv["beta_lag_median"]
                <= bb["beta_lag_ci_hi"]
            ),
            "interval_kind": _SERIES_INTERVAL_KIND,
        })
    return pd.DataFrame(rows)


def summarise(
    comparison: pd.DataFrame,
    variant_dir: str,
    label: str,
    *,
    mismatch: list[str] | None = None,
    coverage: tuple[int, int, list[str]] | None = None,
) -> dict:
    """One-row robustness verdict for a variant (feeds the §7 matrix).

    The verdict is decided in a fixed order of severity, because a lower-severity
    reading of a higher-severity problem is exactly the failure this function is
    for: a stale pairing or collapsed coverage would otherwise be reported as
    "robust", which is the most confident thing the matrix can say and the least
    warranted. ``mismatch`` and ``coverage`` come from
    :func:`definition_mismatch` and :func:`coverage_report`; both default to "not
    checked" so an older caller keeps its behaviour.
    """
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

    coverage_frac = None
    if coverage is not None:
        baseline_rows, shared_rows, missing = coverage
        coverage_frac = (shared_rows / baseline_rows) if baseline_rows else None

    status = "compared"
    if mismatch:
        status = "stale-pairing"
        verdict = (
            "STALE PAIRING (not assessed): baseline and variant differ in "
            + ", ".join(mismatch)
            + " — refit the variant against the current model of record"
        )
    elif converged is False:
        status = "non-converged"
        verdict = "NON-CONVERGED (not assessed)"
    elif coverage_frac is not None and coverage_frac < MIN_COVERAGE:
        status = "partial-coverage"
        missing_note = (
            f"; missing series: {', '.join(coverage[2])}" if coverage[2] else ""
        )
        verdict = (
            f"PARTIAL COVERAGE (not assessed): only {coverage[1]} of "
            f"{coverage[0]} baseline rows are shared{missing_note}"
        )
    elif not outside:
        verdict = "robust (all within baseline 89% interval)"
    else:
        verdict = "sensitive: " + ", ".join(outside)
    return {
        "variant": label,
        "status": status,
        "converged": converged,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_within_ci": n_within,
        "n_checked": n_checked,
        "coverage": coverage_frac,
        "quantities_outside_ci": ", ".join(outside),
        "max_abs_delta": max_abs_delta,
        "verdict": verdict,
    }
