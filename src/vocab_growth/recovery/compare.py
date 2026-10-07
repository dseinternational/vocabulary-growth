# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Score recovery posteriors against the truths used to simulate their data.

Tables report signed errors, posterior intervals, interval containment and the
truth's posterior quantile. Intervals follow the project's quantity-specific
policy. Poor recovery can reflect weak information, prior influence, constraints,
simulation error or sampling failure; these summaries do not distinguish them.

Containment fractions within a replicate describe correlated target quantities,
not repeated-sample coverage of each quantity. Pooled rows remain descriptive.
Simulation calibration needs an appropriate generating design and enough
replicates for its intended precision; no universal replicate count suffices.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from dse_research_utils.statistics.loo import as_dataset

from vocab_growth import intervals
from vocab_growth.comparison import total_spread_from_values, total_spread_plan
from vocab_growth.fit_artifacts import (
    FIT_MANIFEST_FILENAME,
    read_json,
    require_full_trace,
)
from vocab_growth.models.definitions import ModelType
from vocab_growth.sensitivity.compare import CAVEATS_SEPARATOR, diagnostics_gate

# Report scalar, query-age and study quantities individually; omit observation
# arrays from this target selection.
ELEMENTWISE_DIMS: tuple[str, ...] = ("query_id", "study_id")

# Summarise the many child effects in aggregate to keep their recovery readable.
AGGREGATE_DIMS: tuple[str, ...] = ("subject_id",)

# Report containment in both intervals. A hit in one interval is not proof of
# good recovery; interpret errors, interval widths and diagnostics alongside it.
OUTER_PROB = intervals.DEFAULT_CI_PROB
INNER_PROB = intervals.INNER_CI_PROB

# Variables excluded from the target set, with the reason each is not an estimand:
#
#   z_*            standardised design grids or non-centred deviates, omitted
#                  in favour of their reported scaled quantities.
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
    return as_dataset(node)


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


# Derive new-child count SD from the same function used in population contrasts.
# Query-age targets are correlated with their inputs. Adding them changes target
# counts and descriptive containment fractions, so compare matrices with matching
# target sets. See notes/202609141600-total-spread-estimand.md.


def total_spread_targets(definition) -> tuple[tuple[str, str], ...]:
    """``(outcome, variable)`` pairs of the total spread scored for ``definition``.

    Empty where it is not derived -- the joint and trivariate engines, and a
    low-rank factor child block (VG22) -- so those models score what they did.
    """
    model_type = getattr(definition, "model_type", None)
    if model_type is ModelType.UNIVARIATE:
        pairs: tuple[tuple[str, str], ...] = (
            (definition.outcome.value, "total_spread_words_query"),
        )
    elif model_type is ModelType.BIVARIATE:
        pairs = (
            ("understood", "total_spread_words_u_query"),
            ("spoken", "total_spread_words_s_query"),
        )
    else:
        return ()
    try:
        for outcome, _ in pairs:
            total_spread_plan(definition, outcome, "query")
    except NotImplementedError:
        return ()
    return pairs


def total_spread_inputs(definition) -> tuple[str, ...]:
    """The variables :func:`with_total_spread` reads for ``definition``."""
    names: list[str] = []
    for outcome, _ in total_spread_targets(definition):
        plan = total_spread_plan(definition, outcome, "query")
        for name in plan.grid_variables + plan.scalar_variables:
            if name not in names:
                names.append(name)
    return tuple(names)


def with_total_spread(
    dataset: xr.Dataset, definition, *, query_ages: np.ndarray | None = None
) -> xr.Dataset:
    """``dataset`` (``chain``, ``draw``, ...) with the total spread added per draw.

    A variable the computation needs and ``dataset`` lacks is an error rather than
    a silently smaller target set: both engines emit every input on the query
    grid, and the truth draw carries them as reported deterministics.
    """
    pairs = total_spread_targets(definition)
    if not pairs:
        return dataset
    ages = np.asarray(
        definition.ages_query if query_ages is None else query_ages, dtype=float
    )
    n_chain, n_draw = dataset.sizes["chain"], dataset.sizes["draw"]
    added = {}
    for outcome, name in pairs:
        plan = total_spread_plan(definition, outcome, "query")
        values: dict[str, np.ndarray] = {}
        for variable in plan.grid_variables + plan.scalar_variables:
            if variable not in dataset:
                raise KeyError(
                    f"{name} needs {variable!r}, which this dataset does not carry."
                )
            array = dataset[variable].transpose("chain", "draw", ...).values
            if variable in plan.grid_variables:
                if array.shape[-1] != ages.size:
                    raise ValueError(
                        f"{variable} has {array.shape[-1]} query ages but "
                        f"{ages.size} were supplied."
                    )
                values[variable] = np.asarray(array, dtype=float).reshape(n_chain * n_draw, -1)
            else:
                values[variable] = np.asarray(array, dtype=float).reshape(n_chain * n_draw)
        spread = total_spread_from_values(plan, values, ages, definition.n_trials)
        added[name] = (
            ("chain", "draw", "query_id"),
            spread.sd_words.reshape(n_chain, n_draw, ages.size),
        )
    return dataset.assign(added)


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
    """Aggregate interval containment and error across high-dimensional effects.

    Report correlation of truth with posterior means, absolute error and spread
    across children. These summaries can reveal poor recovery of the fitted
    effects but do not establish nominal repeated-sample coverage.
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


def sampling_tier(fit_dir: str) -> str | None:
    """Recorded sampling configuration, or ``None`` when absent or unreadable.
    """
    path = os.path.join(fit_dir, FIT_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        manifest = read_json(path)
    except Exception:  # noqa: BLE001 - an unreadable manifest is an unknown tier
        return None
    name = (manifest.get("sampling") or {}).get("configuration_name")
    return None if name is None else str(name)


def summarise(
    table: pd.DataFrame,
    fit_dir: str,
    *,
    label: str,
    truth_source: str,
    z_threshold: float = 4.0,
) -> dict[str, Any]:
    """Summarise interval containment and error, with a convergence-qualified label.

    Missing or failed diagnostics prevent assessment. Confirmed fits with caveats
    retain scores but are not labelled clean recovery. Even the clean recovered
    label only describes containment at these selected truths, not general
    identification. Record the sampling tier for pooled-row checks.
    """
    gate = diagnostics_gate(fit_dir)
    converged, max_rhat, min_ess = gate
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
    clean_pass = gate.clean is True
    if converged is None:
        verdict = "UNVERIFIED (no recorded diagnostics; not assessed)"
    elif converged is False:
        verdict = "NON-CONVERGED (not assessed)"
    elif outside_quantities:
        verdict = "not recovered: " + ", ".join(outside_quantities)
        if not clean_pass:
            verdict += " (converged with caveats)"
    elif clean_pass and max_abs_z <= z_threshold:
        verdict = "recovered (every target within its 89% interval)"
    elif clean_pass:
        verdict = f"recovered, but |z| up to {max_abs_z:.1f}"
    elif max_abs_z <= z_threshold:
        verdict = (
            "converged with caveats: every target within its 89% interval, "
            "but not scored recovered"
        )
    else:
        verdict = (
            "converged with caveats: targets within interval but |z| up to "
            f"{max_abs_z:.1f}; not scored recovered"
        )

    return {
        "replicate": label,
        "truth_source": truth_source,
        "sampling_configuration": sampling_tier(fit_dir),
        "converged": converged,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "caveats": gate.caveats_text,
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


def _tier_groups(summaries: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Replicate labels grouped by recorded sampling tier, unrecorded ones apart."""
    groups: dict[str, list[str]] = {}
    for summary in summaries:
        tier = summary.get("sampling_configuration")
        groups.setdefault("unrecorded" if tier is None else str(tier), []).append(
            str(summary.get("replicate"))
        )
    return groups


def pooled_row(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool confirmed replicates descriptively unless recorded sampling tiers differ.

    Unknown tiers do not trigger the mixed-tier refusal and may contribute if
    diagnostics confirm convergence. Preserve the POOLED prefix for downstream
    readers and disclose how many replicates were assessed. Correlated target
    rows are not independent calibration experiments.
    """
    recorded_tiers = sorted(
        tier for tier in _tier_groups(summaries) if tier != "unrecorded"
    )
    if len(recorded_tiers) > 1:
        by_tier = "; ".join(
            f"{tier}: {', '.join(labels)}"
            for tier, labels in sorted(_tier_groups(summaries).items())
        )
        return {
            "replicate": (
                f"POOLED (refused: {len(summaries)} replicates at mixed sampling tiers)"
            ),
            "truth_source": ", ".join(sorted({s["truth_source"] for s in summaries})),
            "sampling_configuration": by_tier,
            "converged": None,
            "max_rhat": None,
            "min_ess": None,
            "caveats": "",
            "n_targets": 0,
            "n_within_ci89": 0,
            "coverage_ci89": float("nan"),
            "coverage_ci50": float("nan"),
            "max_abs_z": float("nan"),
            "worst_quantity": "",
            "quantities_outside_ci89": "",
            "verdict": (
                "NOT POOLED (not assessed): replicates were sampled at different "
                f"tiers ({by_tier}); score each tier on its own, or refit the odd "
                "one out at the tier the run claims"
            ),
        }

    assessed = [s for s in summaries if s.get("converged") is True]
    n_targets = sum(s["n_targets"] for s in assessed)
    n_within = sum(s["n_within_ci89"] for s in assessed)
    pooled_caveats = sorted(
        {
            caveat
            for s in assessed
            for caveat in (s.get("caveats") or "").split(CAVEATS_SEPARATOR)
            if caveat
        }
    )
    return {
        "replicate": f"POOLED ({len(assessed)} of {len(summaries)} replicates assessed)",
        "truth_source": ", ".join(sorted({s["truth_source"] for s in summaries})),
        "sampling_configuration": ", ".join(
            tier for tier in sorted(_tier_groups(summaries)) if tier != "unrecorded"
        ) or None,
        "converged": all(s.get("converged") for s in assessed) if assessed else None,
        "max_rhat": max((s["max_rhat"] for s in assessed if s["max_rhat"]), default=None),
        "min_ess": min((s["min_ess"] for s in assessed if s["min_ess"]), default=None),
        "caveats": CAVEATS_SEPARATOR.join(pooled_caveats),
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
    definition=None,
    truth_definition=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return elementwise scores, aggregate scores and one summary for a replicate.

    Require a full trace so reduced persistence cannot silently remove scaled
    effects. Select shared target variables by dimensions and exclusion rules.

    When definitions are supplied, derive total spread separately for truth and
    fit only if their query-age grids match exactly. ``query_ages`` overrides the
    fitted grid and, without a separate truth definition, the truth grid too.
    Mismatched grids omit this derived target rather than compare different ages.
    """
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(f"No recovery trace at {trace_path}.")
    truth = _as_dataset(
        truth_tree["posterior"] if "posterior" in getattr(truth_tree, "children", {}) else truth_tree
    )
    derive_spread = False
    if definition is not None:
        fit_ages = np.asarray(
            definition.ages_query if query_ages is None else query_ages, dtype=float
        )
        truth_ages = (
            fit_ages
            if truth_definition is None
            else np.asarray(truth_definition.ages_query, dtype=float)
        )
        derive_spread = np.array_equal(fit_ages, truth_ages)
    if derive_spread:
        truth = with_total_spread(
            truth,
            definition if truth_definition is None else truth_definition,
            query_ages=truth_ages,
        )
    # Reduced persistence omits scaled effects. Since target selection uses
    # the intersection, require full persistence before a missing variable can
    # silently reduce the score set.
    require_full_trace(
        os.path.dirname(trace_path), purpose="Parameter-recovery scoring"
    )
    with xr.open_datatree(trace_path) as tree:
        posterior_full = _as_dataset(tree["posterior"])
        if derive_spread:
            inputs = list(total_spread_inputs(definition))
            derived_names = [name for _, name in total_spread_targets(definition)]
            if derived_names:
                derived = with_total_spread(
                    posterior_full[inputs].load(), definition, query_ages=fit_ages
                )[derived_names]
                posterior_full = posterior_full.assign(derived.data_vars)
        elementwise, aggregate = target_variables(posterior_full, truth)
        wanted = elementwise + aggregate
        posterior = posterior_full[wanted].load().compute() if wanted else posterior_full[[]]
    table = recovery_table(truth, posterior, query_ages=query_ages)
    aggregates = aggregate_table(truth, posterior)
    summary = summarise(table, fit_dir, label=label, truth_source=truth_source)
    return table, aggregates, summary
