# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Score a recovery refit against the truth that generated its data (issue #163).

For every target quantity the table records the truth, the recovered posterior,
the signed error in posterior-standard-deviation units, whether the truth falls
inside the reported credible intervals, and the truth's posterior quantile.
Intervals follow the project convention (:mod:`vocab_growth.intervals`): a 50%
inner and an 89% outer interval, equal-tailed except for the named skewed
estimands, which use highest-density intervals.

What this does and does not establish
-------------------------------------
A recovery check answers a specific question: *if this model were true, would this
sampler at this sampling configuration find the parameters back from a dataset of
this size and shape?* A quantity whose truth sits far outside its posterior points
to a non-identified parameter, a mis-specified constraint, or a sampler that has
not converged — all of which matter for the reported intervals.

It is not a calibration proof. Simulation-based calibration needs on the order of
a hundred replicates before rank uniformity means anything, and the interval
coverage computed *within* one replicate is over correlated quantities from a
single truth, so it is descriptive rather than a coverage estimate. The
``coverage_ci89`` column is reported as what it is — the fraction of target
quantities whose interval contained the truth — and the pooled row is marked
indicative. The harness accumulates replicates so a full calibration run remains
possible later without new code.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from vocab_growth import intervals
from vocab_growth.fit_artifacts import require_full_trace
from vocab_growth.sensitivity.compare import diagnostics_gate

# Dimensions whose elements are reported individually. Observation-level
# quantities are excluded: they are per-row latents, not estimands, and there are
# tens of thousands of them.
ELEMENTWISE_DIMS: tuple[str, ...] = ("query_id", "study_id")

# Dimensions summarised in aggregate rather than element by element. Per-child
# effects number in the thousands; their individual recovery is uninformative but
# their aggregate behaviour is not.
AGGREGATE_DIMS: tuple[str, ...] = ("subject_id",)

# A target quantity is "recovered" when the truth lies inside the outer (89%)
# interval. Reported alongside the 50% interval so a systematically biased
# posterior shows up even when the wide interval still covers.
OUTER_PROB = intervals.DEFAULT_CI_PROB
INNER_PROB = intervals.INNER_CI_PROB

# Variables excluded from the target set, with the reason each is not an estimand:
#
#   z_*            standardised age grids — fixed functions of the design, so
#                  "recovering" them is vacuous (zero posterior spread).
#   f_*, h_*       logit-scale latent trajectories. Each is a monotone transform
#                  of a probability-scale quantity that is already a target
#                  (f_u_query of p_u_query, h_query of q_query), so counting both
#                  would double-weight the same information in the coverage
#                  summary. The study reports the probability scale.
#   *_raw          non-centred reparameterisation offsets. delta_raw carries no
#                  interpretation of its own; the scaled delta it produces is a
#                  target.
EXCLUDED_PREFIXES: tuple[str, ...] = ("z_", "f_", "h_")
EXCLUDED_SUFFIXES: tuple[str, ...] = ("_raw",)

# Grid suffixes stripped before resolving an estimand's interval convention, so a
# query-grid dispersion resolves to the same convention as the scalar the policy
# in vocab_growth.intervals names.
_GRID_SUFFIXES: tuple[str, ...] = ("_query", "_plot")


def _as_dataset(node) -> xr.Dataset:
    return node.to_dataset() if isinstance(node, xr.DataTree) else node


def _dims_of(dataset: xr.Dataset, name: str) -> tuple[str, ...]:
    return tuple(d for d in dataset[name].dims if d not in ("chain", "draw"))


def is_excluded_target(name: str) -> bool:
    """Whether a model variable is excluded from the target set (see above)."""
    return name.startswith(EXCLUDED_PREFIXES) or name.endswith(EXCLUDED_SUFFIXES)


def _estimand_key(name: str) -> str:
    """Map a model variable name onto the estimand the interval policy names.

    ``kappa_u_query`` is the age-varying dispersion of the same estimand the
    policy lists as ``kappa``; without this the grid-valued dispersions would be
    summarised with equal-tailed intervals while the scalar uses a
    highest-density one.
    """
    base = name
    for suffix in _GRID_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.startswith("kappa"):
        return "kappa"
    if base.startswith("conc"):
        return "conc"
    return base


def interval_kind_for_target(name: str) -> intervals.IntervalKind:
    """Interval convention for a target quantity, via the project-wide policy."""
    return intervals.interval_kind_for(_estimand_key(name))


def target_variables(
    posterior: xr.Dataset, truth: xr.Dataset
) -> tuple[list[str], list[str]]:
    """Split the variables present in both truth and posterior into target sets.

    Returns ``(elementwise, aggregate)``. Selection is by dimension rather than by
    a hand-maintained list of names, so a model that gains a reported quantity
    gains its recovery check automatically.
    """
    shared = sorted(set(posterior.data_vars) & set(truth.data_vars))
    elementwise: list[str] = []
    aggregate: list[str] = []
    for name in shared:
        if is_excluded_target(name):
            continue
        dims = _dims_of(posterior, name)
        if dims == ():
            elementwise.append(name)
        elif len(dims) == 1 and dims[0] in ELEMENTWISE_DIMS:
            elementwise.append(name)
        elif len(dims) == 1 and dims[0] in AGGREGATE_DIMS:
            aggregate.append(name)
    return elementwise, aggregate


def _score(draws: np.ndarray, truth_value: float, name: str) -> dict[str, Any]:
    """Recovery metrics for one scalar quantity."""
    draws = np.asarray(draws, dtype=float).ravel()
    draws = draws[np.isfinite(draws)]
    kind = interval_kind_for_target(name)
    if draws.size == 0:
        return {
            "truth": truth_value,
            "posterior_median": float("nan"),
            "posterior_mean": float("nan"),
            "posterior_sd": float("nan"),
            "z": float("nan"),
            "ci50_lo": float("nan"),
            "ci50_hi": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "within_ci50": None,
            "within_ci89": None,
            "truth_quantile": float("nan"),
            "interval_kind": kind,
        }
    inner_lo, inner_hi = intervals.interval_1d(draws, INNER_PROB, kind)
    outer_lo, outer_hi = intervals.interval_1d(draws, OUTER_PROB, kind)
    sd = float(np.std(draws, ddof=1)) if draws.size > 1 else float("nan")
    mean = float(np.mean(draws))
    # Rank of the truth within the posterior draws: the statistic simulation-based
    # calibration accumulates. Mid-rank so ties in a discrete posterior do not
    # bias the quantile up.
    quantile = float(
        np.mean(draws < truth_value) + 0.5 * np.mean(draws == truth_value)
    )
    return {
        "truth": float(truth_value),
        "posterior_median": float(np.median(draws)),
        "posterior_mean": mean,
        "posterior_sd": sd,
        "z": float((mean - truth_value) / sd) if sd and np.isfinite(sd) else float("nan"),
        "ci50_lo": inner_lo,
        "ci50_hi": inner_hi,
        "ci_lo": outer_lo,
        "ci_hi": outer_hi,
        "within_ci50": bool(inner_lo <= truth_value <= inner_hi),
        "within_ci89": bool(outer_lo <= truth_value <= outer_hi),
        "truth_quantile": quantile,
        "interval_kind": kind,
    }


def recovery_table(
    truth: xr.Dataset,
    posterior: xr.Dataset,
    *,
    query_ages: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-quantity recovery table for one replicate.

    Columns: ``quantity``, ``index`` (query age, study code, or empty for a
    scalar), the truth, the recovered posterior summary, ``z``, both intervals and
    their containment flags, and the truth's posterior quantile.
    """
    elementwise, _aggregate = target_variables(posterior, truth)
    rows: list[dict[str, Any]] = []
    for name in elementwise:
        dims = _dims_of(posterior, name)
        post = posterior[name]
        true = truth[name]
        if dims == ():
            row = {"quantity": name, "index": "", "dimension": ""}
            row.update(_score(post.values, float(np.asarray(true.values).ravel()[0]), name))
            rows.append(row)
            continue
        dim = dims[0]
        size = post.sizes[dim]
        true_values = np.asarray(true.values).reshape(-1, size)[0]
        # Label query-grid rows with their age only when the supplied ages line up
        # with the grid exactly. A partial match would silently mislabel rows, which
        # matters here because these labels are how a reader locates a recovery
        # failure on the trajectory.
        age_labels = (
            query_ages
            if dim == "query_id" and query_ages is not None and len(query_ages) == size
            else None
        )
        for position in range(size):
            if age_labels is not None:
                label = f"{age_labels[position]:g}"
            else:
                label = str(post.coords[dim].values[position]) if dim in post.coords else str(position)
            row = {"quantity": name, "index": label, "dimension": dim}
            row.update(
                _score(post.isel({dim: position}).values, float(true_values[position]), name)
            )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_table(truth: xr.Dataset, posterior: xr.Dataset) -> pd.DataFrame:
    """Aggregate recovery of the high-dimensional random effects.

    Individual child effects are barely identified by design — each child has one
    or a few administrations — so the informative question is whether the *set* of
    them is recovered: does the posterior track the true effects across children,
    and do their intervals cover at about the nominal rate?
    """
    _elementwise, aggregate = target_variables(posterior, truth)
    rows: list[dict[str, Any]] = []
    for name in aggregate:
        dims = _dims_of(posterior, name)
        dim = dims[0]
        post = posterior[name]
        stacked = post.stack(sample=("chain", "draw")).transpose(dim, "sample").values
        true_values = np.asarray(truth[name].values).reshape(-1, post.sizes[dim])[0]
        kind = interval_kind_for_target(name)
        lo = np.empty(stacked.shape[0])
        hi = np.empty(stacked.shape[0])
        for i in range(stacked.shape[0]):
            lo[i], hi[i] = intervals.interval_1d(stacked[i], OUTER_PROB, kind)
        posterior_mean = stacked.mean(axis=1)
        covered = (true_values >= lo) & (true_values <= hi)
        finite = np.isfinite(true_values) & np.isfinite(posterior_mean)
        correlation = (
            float(np.corrcoef(true_values[finite], posterior_mean[finite])[0, 1])
            if finite.sum() > 2 and np.std(true_values[finite]) > 0
            else float("nan")
        )
        rows.append(
            {
                "quantity": name,
                "dimension": dim,
                "n_elements": int(stacked.shape[0]),
                "coverage_ci89": float(covered.mean()),
                "truth_vs_posterior_mean_correlation": correlation,
                "mean_abs_error": float(
                    np.mean(np.abs(posterior_mean[finite] - true_values[finite]))
                ),
                "truth_sd": float(np.std(true_values[finite])),
                "posterior_mean_sd": float(np.std(posterior_mean[finite])),
                "interval_kind": kind,
            }
        )
    return pd.DataFrame(rows)


def summarise(
    table: pd.DataFrame,
    fit_dir: str,
    *,
    label: str,
    truth_source: str,
    z_threshold: float = 4.0,
) -> dict[str, Any]:
    """One-row verdict for a replicate.

    A non-converged fit is never reported as recovered: a truth outside the
    posterior of a fit that did not converge says nothing about identifiability.
    """
    converged, max_rhat, min_ess = diagnostics_gate(fit_dir)
    checked = table.dropna(subset=["within_ci89"])
    within = checked["within_ci89"].astype(bool) if len(checked) else pd.Series(dtype=bool)
    outside = checked.loc[~within] if len(checked) else checked
    abs_z = table["z"].abs()
    max_abs_z = float(abs_z.max()) if len(table) else float("nan")
    worst = (
        table.loc[abs_z.idxmax(), ["quantity", "index"]].tolist()
        if len(table) and np.isfinite(max_abs_z)
        else ["", ""]
    )
    outside_quantities = sorted(outside["quantity"].unique().tolist()) if len(outside) else []

    # Only a fit whose convergence is positively confirmed can support a recovery
    # claim. A missing diagnostics file is not evidence of convergence, so it is
    # reported as unverified rather than quietly assessed: the sampling
    # configurations used for recovery work (dev, test) are below the reporting
    # tier, and the pipeline's hard convergence gate applies only at that tier.
    if converged is None:
        verdict = "UNVERIFIED (no recorded diagnostics; not assessed)"
    elif converged is False:
        verdict = "NON-CONVERGED (not assessed)"
    elif not outside_quantities and max_abs_z <= z_threshold:
        verdict = "recovered (every target within its 89% interval)"
    elif not outside_quantities:
        verdict = f"recovered, but |z| up to {max_abs_z:.1f}"
    else:
        verdict = "not recovered: " + ", ".join(outside_quantities)

    return {
        "replicate": label,
        "truth_source": truth_source,
        "converged": converged,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_targets": int(len(checked)),
        "n_within_ci89": int(within.sum()) if len(checked) else 0,
        "coverage_ci89": float(within.mean()) if len(checked) else float("nan"),
        "coverage_ci50": (
            float(table["within_ci50"].dropna().astype(bool).mean())
            if table["within_ci50"].notna().any()
            else float("nan")
        ),
        "max_abs_z": max_abs_z,
        "worst_quantity": f"{worst[0]}[{worst[1]}]" if worst[0] else "",
        "quantities_outside_ci89": ", ".join(outside_quantities),
        "verdict": verdict,
    }


def pooled_row(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled indicative row across replicates.

    Marked indicative deliberately: with a handful of replicates over correlated
    quantities this is a descriptive summary, not a coverage estimate.
    """
    assessed = [s for s in summaries if s.get("converged") is True]
    n_targets = sum(s["n_targets"] for s in assessed)
    n_within = sum(s["n_within_ci89"] for s in assessed)
    return {
        "replicate": f"POOLED ({len(assessed)} of {len(summaries)} replicates assessed)",
        "truth_source": ", ".join(sorted({s["truth_source"] for s in summaries})),
        "converged": all(s.get("converged") for s in assessed) if assessed else None,
        "max_rhat": max((s["max_rhat"] for s in assessed if s["max_rhat"]), default=None),
        "min_ess": min((s["min_ess"] for s in assessed if s["min_ess"]), default=None),
        "n_targets": n_targets,
        "n_within_ci89": n_within,
        "coverage_ci89": (n_within / n_targets) if n_targets else float("nan"),
        "coverage_ci50": float(
            np.mean([s["coverage_ci50"] for s in assessed])
        ) if assessed else float("nan"),
        "max_abs_z": max((s["max_abs_z"] for s in assessed), default=float("nan")),
        "worst_quantity": "",
        "quantities_outside_ci89": ", ".join(
            sorted({q for s in assessed for q in s["quantities_outside_ci89"].split(", ") if q})
        ),
        "verdict": "indicative only — coverage over correlated quantities, not SBC",
    }


def compare_replicate(
    truth_tree: xr.DataTree | xr.Dataset,
    trace_path: str,
    fit_dir: str,
    *,
    label: str,
    truth_source: str,
    query_ages: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score one replicate: ``(table, aggregate_table, summary)``."""
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(f"No recovery trace at {trace_path}.")
    truth = _as_dataset(
        truth_tree["posterior"] if "posterior" in getattr(truth_tree, "children", {}) else truth_tree
    )
    # Targets are the intersection of truth and posterior, so anything the fit
    # did not persist drops out of the score silently rather than failing. The
    # scaled random effects are exactly the targets a compacted trace omits (the
    # `_raw` offsets are deliberately excluded from scoring), so a compacted
    # recovery fit would report a quietly smaller target set as if it were the
    # whole one.
    require_full_trace(
        os.path.dirname(trace_path), purpose="Parameter-recovery scoring"
    )
    with xr.open_datatree(trace_path) as tree:
        posterior_full = _as_dataset(tree["posterior"])
        elementwise, aggregate = target_variables(posterior_full, truth)
        wanted = elementwise + aggregate
        posterior = posterior_full[wanted].load().compute() if wanted else posterior_full[[]]
    table = recovery_table(truth, posterior, query_ages=query_ages)
    aggregates = aggregate_table(truth, posterior)
    summary = summarise(table, fit_dir, label=label, truth_source=truth_source)
    return table, aggregates, summary
