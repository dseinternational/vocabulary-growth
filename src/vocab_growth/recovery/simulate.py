# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Forward-simulate a dataset from a model at a known parameter draw (issue #163).

The simulation is done *by the model*, not by a re-implementation of it: the
engine's own build code constructs the likelihood nodes, and
:func:`pymc.sample_posterior_predictive` draws from those nodes at a single fixed
parameter draw. Nothing here restates a mean function, a dispersion function, or
a row denominator, so a change to a model cannot silently invalidate its
recovery check.

The one thing the simulator must sequence itself is the nesting. Words spoken
and words signed are modelled conditionally on the child's comprehension total,
whose denominator is a ``pm.Data`` array fixed at build time. Comprehension is
therefore drawn first, written into the analysis frame, and the model is
**rebuilt** so the engine re-derives every denominator from the *simulated*
parent before the dependent outcomes are drawn. After the final round the
denominators and nested/marginal flags of a model rebuilt from the finished
synthetic frame are compared against the ones the simulation actually used; a
mismatch aborts, because it would mean the data were generated under a different
decomposition from the one that will be fitted.

Missingness is preserved exactly. A row contributes a simulated value only where
the real row contributed an observed one, so the synthetic dataset carries the
real study/age/child design and the real observation pattern — only the counts
are the model's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as model_reporting
import dse_research_utils.statistics.models.sampling as model_sampling
import duckdb
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    normalise_for_json,
    source_data_hash,
    validate_fit_output,
    write_json_atomic,
)
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery import compare
from vocab_growth.recovery.spec import (
    CompositionOutcome,
    CountOutcome,
    EngineRecoverySpec,
    RecoveryTarget,
    outcome_column,
    recovery_target,
)
from vocab_growth.reporting import console, key_value_table

BUILD_STAGE_NAME = "Model definition and initialisation"
PRIORS_STAGE_NAME = "Priors and hyperparameters"
PREPARE_STAGE_NAME = "Prepare data"

SYNTHETIC_FRAME_FILENAME = "synthetic_analysis_frame.parquet"
TRUTH_FILENAME = "truth.nc"
SIMULATION_FILENAME = "simulation.json"


# ==========================================================================
# Output layout
# ==========================================================================


def recovery_label(definition, replicate: int, *, truth_definition=None) -> str:
    """Output label for one recovery replicate, e.g. ``VG10-...-recovery-r01``."""
    return (
        f"{definition.model_id}-"
        f"{recovery_config_name(definition, replicate, truth_definition=truth_definition)}"
    )


def truth_source_tag(truth_definition, fit_definition) -> str:
    """A short name for whose data a cross-definition refit is chasing.

    Config names are hyphen-token paths sharing a base, so the tokens the truth
    carries beyond the shared base name it: a truth that is the fit definition's
    own base leaves nothing over and is the ``record``.
    """
    truth_tokens = truth_definition.config_name.split("-")
    fit_tokens = fit_definition.config_name.split("-")
    shared = 0
    for left, right in zip(truth_tokens, fit_tokens, strict=False):
        if left != right:
            break
        shared += 1
    remainder = truth_tokens[shared:]
    return "-".join(remainder) if remainder else "record"


def recovery_config_name(definition, replicate: int, *, truth_definition=None) -> str:
    """Config name carried by the recovery variant of a definition.

    ``truth_definition`` names the definition the *data* came from, when that is
    not the definition being fitted (issue #226). Such a run answers a different
    question from self-recovery -- it asks whether one model's estimator is
    biased under another's truth, rather than whether a model recovers itself --
    and it must not be able to land in, or be scored as, a self-recovery
    directory. The ``-under-<tag>`` marker in the name is what prevents that;
    passing the same definition twice is not a cross-definition run and adds no
    marker, so existing output keeps its existing names.
    """
    stem = definition.config_name
    if (
        truth_definition is not None
        and truth_definition.config_name != definition.config_name
    ):
        stem = f"{stem}-under-{truth_source_tag(truth_definition, definition)}"
    return f"{stem}-recovery-r{replicate:02d}"


def simulation_dir(definition, replicate: int, output_root: str | None = None) -> str:
    """Directory holding one replicate's synthetic data and truth.

    Deliberately *not* under ``models/``: a completed fit atomically replaces its
    own output directory, which would delete the inputs that produced it.
    """
    root = output_root if output_root is not None else env.output_root()
    return os.path.join(root, "recovery", recovery_label(definition, replicate))


# ==========================================================================
# Truth draws
# ==========================================================================


@dataclass
class TruthDraw:
    """One fixed parameter draw, as a tree ``sample_posterior_predictive`` accepts."""

    tree: xr.DataTree
    source: str
    chain: int
    draw: int
    provenance: dict[str, Any] = field(default_factory=dict)


def _as_dataset(node) -> xr.Dataset:
    """Return an xarray Dataset for a DataTree node or Dataset."""
    return node.to_dataset() if isinstance(node, xr.DataTree) else node


def _single_draw_tree(posterior: xr.Dataset) -> xr.DataTree:
    """Wrap a one-draw posterior Dataset as a tree with a ``posterior`` group.

    The chain and draw labels are reset to zero. They must be: the selected draw
    keeps its original label after ``isel``, so anything later merged into this
    dataset (the computed deterministics, which come back labelled from zero)
    would align on a *different* label and silently outer-join into a second,
    all-missing draw. Sampling from that gives non-finite likelihood parameters.
    The originating chain and draw are recorded in the provenance instead.
    """
    tree = xr.DataTree()
    tree["posterior"] = xr.DataTree(
        posterior.assign_coords(
            chain=np.zeros(posterior.sizes["chain"], dtype=int),
            draw=np.zeros(posterior.sizes["draw"], dtype=int),
        )
    )
    return tree


def reportable_deterministics(model: pm.Model) -> list[str]:
    """Deterministics worth carrying in the truth record.

    The reported estimands (population trajectories at the query ages, the
    association scalar, the study and child effects) are model *deterministics*,
    not free parameters, so a truth made only of free parameters could not be
    scored against the quantities the study publishes. Selection uses the same
    dimension and exclusion rules as the scoring step, so what is stored is
    exactly what will be compared.
    """
    dims_of = model.named_vars_to_dims
    keep_dims = set(compare.ELEMENTWISE_DIMS) | set(compare.AGGREGATE_DIMS)
    names: list[str] = []
    for variable in model.deterministics:
        name = variable.name
        if compare.is_excluded_target(name):
            continue
        dims = tuple(dims_of.get(name) or ())
        if dims == () or (len(dims) == 1 and dims[0] in keep_dims):
            names.append(name)
    return names


def _with_deterministics(truth: TruthDraw, model: pm.Model) -> TruthDraw:
    """Add the reported deterministics to a truth draw, computed from the graph.

    Computed from the model as it stands rather than read from a stored trace, so
    a truth taken from an older trace is still expressed in terms of the current
    reporting quantities.
    """
    names = reportable_deterministics(model)
    posterior = _as_dataset(truth.tree["posterior"])
    if not names:
        return truth
    computed = pm.compute_deterministics(
        posterior, model=model, var_names=names, progressbar=False,
        compile_kwargs={"mode": "FAST_COMPILE"},
    )
    computed_ds = _as_dataset(computed)
    # join="exact" rather than the default outer join: both operands are already
    # on the same single (chain, draw) label, and if that ever stops being true
    # the merge must fail loudly instead of fabricating an all-missing draw.
    merged = xr.merge([posterior, computed_ds], join="exact")
    if merged.sizes["draw"] != posterior.sizes["draw"]:
        raise RuntimeError(
            "Merging computed deterministics changed the number of truth draws "
            f"({posterior.sizes['draw']} -> {merged.sizes['draw']})."
        )
    truth.tree = _single_draw_tree(merged)
    truth.provenance = {
        **truth.provenance,
        "deterministics_computed": sorted(names),
    }
    return truth


#: Fractional part of the golden ratio, the generator of the low-discrepancy
#: sequence used to place truth draws within a chain.
_GOLDEN_RATIO_FRACTION = 0.6180339887498949


def _spread_index(replicate: int, n_available: int) -> int:
    """Pick a well-separated position for ``replicate`` among ``n_available``.

    Replicates must not sit next to each other in a Markov chain: adjacent draws
    are autocorrelated, so they would be near-duplicate truths and the replicates
    would not be distinct parameter settings.

    The positions come from the golden-ratio (Kronecker) low-discrepancy
    sequence, which spreads any number of replicates evenly over the chain
    without needing to know the total in advance — replicate *r* can therefore be
    added later without moving the draws already used. A plain arithmetic rule
    keyed on ``replicate`` alone cannot do this: it either clusters or drifts
    steadily towards one end of the chain.
    """
    if n_available <= 0:
        raise ValueError("No draws available to select a truth from.")
    if n_available == 1:
        return 0
    position = (replicate * _GOLDEN_RATIO_FRACTION) % 1.0
    return min(int(position * n_available), n_available - 1)


def _current_source_data_hash() -> str:
    """The raw-data fingerprint, as every other artefact consumer computes it."""
    return source_data_hash(env.DATA_DIR)


def truth_from_trace(
    trace_path: str,
    free_rv_names: list[str],
    *,
    replicate: int,
    definition: Any | None = None,
    source_data_hash: str | None = None,
) -> TruthDraw:
    """Take replicate ``r``'s truth from a fitted model-of-record trace.

    Only the free random variables are read, so a multi-gigabyte reporting-quality
    trace costs one small slice rather than a full load. Deterministics are
    recomputed from the model graph later, which keeps the truth consistent with
    the code as it stands rather than as it stood when the trace was written.

    **Provenance (issue #233).** That last property is a convenience and a hazard
    at once: recomputing today's deterministics from an older trace's free
    parameters produces a truth that no fit ever held, and nothing in the free
    variable *names* would notice. Matching names were the only check here, and
    names survive most definition changes -- every anchor recalibration, every
    prior widening, the whole reporting-cap family -- so a stale or wrong-model
    trace passed. ``definition`` and ``source_data_hash``, when given, put the
    source fit through :func:`fit_artifacts.validate_fit_output`, which compares
    the normalised definition, the raw-data fingerprint and the fit's lifecycle
    state. They are optional so the unit tests can exercise the reader on a bare
    trace, but ``simulate_replicate`` always passes them.
    """
    if not os.path.isfile(trace_path):
        raise FileNotFoundError(
            f"No model-of-record trace at {trace_path}. Fit the model of record "
            "first, or use --truth prior."
        )
    if definition is not None:
        errors = validate_fit_output(
            os.path.dirname(trace_path),
            expected_definition=definition,
            expected_source_data_hash=source_data_hash,
        )
        if errors:
            raise ValueError(
                "The model-of-record fit at "
                f"{os.path.dirname(trace_path)} cannot supply a truth draw:\n  - "
                + "\n  - ".join(errors)
                + "\nRefit the model of record, or use --truth prior."
            )
    with xr.open_datatree(trace_path) as tree:
        posterior = _as_dataset(tree["posterior"])
        missing = sorted(set(free_rv_names) - set(posterior.data_vars))
        if missing:
            raise ValueError(
                f"Trace posterior is missing free parameter(s) {missing}; it was "
                "probably written by a different model definition."
            )
        n_chains = posterior.sizes["chain"]
        n_draws = posterior.sizes["draw"]
        chain = (replicate - 1) % n_chains
        draw = _spread_index(replicate, n_draws)
        selected = (
            posterior[free_rv_names].isel(chain=[chain], draw=[draw]).load().compute()
        )
    return TruthDraw(
        tree=_single_draw_tree(selected),
        source="posterior",
        chain=int(chain),
        draw=int(draw),
        provenance={
            "trace_path": trace_path,
            "trace_chains": int(n_chains),
            "trace_draws": int(n_draws),
        },
    )


def truth_from_prior(
    model: pm.Model,
    free_rv_names: list[str],
    *,
    replicate: int,
    n_prior_draws: int,
    random_seed: int,
) -> TruthDraw:
    """Draw replicate ``r``'s truth from the model's own prior.

    A prior truth needs no fitted trace, which makes it the option for a model
    that has not been fitted yet and for the harness's own smoke tests. It is
    *not* equivalent to a posterior truth: the prior covers parameter settings
    far from anything the data support, so a prior-truth check tests the sampler
    over the whole prior mass rather than in the regime the study reports.
    """
    prior = pm.sample_prior_predictive(
        draws=n_prior_draws,
        model=model,
        var_names=free_rv_names,
        random_seed=random_seed,
        compile_kwargs={"mode": "FAST_COMPILE"},
    )
    prior_ds = _as_dataset(prior["prior"])
    draw = _spread_index(replicate, prior_ds.sizes["draw"])
    selected = prior_ds[free_rv_names].isel(chain=[0], draw=[draw]).load()
    return TruthDraw(
        tree=_single_draw_tree(selected),
        source="prior",
        chain=0,
        draw=int(draw),
        provenance={"n_prior_draws": int(n_prior_draws), "random_seed": int(random_seed)},
    )


# ==========================================================================
# Simulation
# ==========================================================================


@dataclass
class SimulationResult:
    """What one replicate's simulation produced and where it was written."""

    model_key: str
    replicate: int
    directory: str
    frame_path: str
    truth_path: str
    truth: TruthDraw
    frame: pd.DataFrame
    simulated_columns: tuple[str, ...]
    skipped_nodes: tuple[str, ...]
    row_counts: dict[str, int]


def _data_value(model: pm.Model, name: str) -> np.ndarray:
    """Read a ``pm.Data`` container's current value off a built model."""
    if name not in model.named_vars:
        raise KeyError(f"Model has no data container named {name!r}.")
    return np.asarray(model[name].get_value())


def _mask_rows(model: pm.Model, node, n_rows: int) -> np.ndarray:
    """Row positions this node's likelihood covers."""
    if getattr(node, "row_mask_data", None) is None:
        return np.arange(n_rows)
    return np.flatnonzero(_data_value(model, node.row_mask_data).astype(bool))


def _stage_for(stages, name: str):
    """The named stage callable from an engine's stage list."""
    for stage_name, fn in stages:
        if stage_name == name:
            return fn
    raise KeyError(f"Engine stage {name!r} not found (found: {[s for s, _ in stages]}).")


def _write_column(frame: pd.DataFrame, column: str, rows: np.ndarray, values: np.ndarray):
    """Write ``values`` into ``column`` at row positions ``rows``."""
    if column not in frame.columns:
        raise KeyError(f"Synthetic frame has no column {column!r}.")
    if rows.size != values.size:
        raise ValueError(
            f"{column}: {values.size} simulated value(s) for {rows.size} likelihood row(s)."
        )
    frame.iloc[rows, frame.columns.get_loc(column)] = values.astype(float)


def _neutralise_child_columns(
    frame: pd.DataFrame, spec: EngineRecoverySpec, pending_columns: set[str]
) -> None:
    """Make nested/marginal classification depend only on the simulated parent.

    ``nested_outcome_spec`` classifies a row as nested when the parent count is a
    valid total *and* the child count does not exceed it. At this point the child
    column still holds the real study value, which may exceed the freshly
    simulated parent and would then be classified marginal — while the value the
    simulator is about to draw for it cannot exceed the parent, so the refit would
    classify the same row as nested. Setting the pending child counts to zero
    where they are observed makes the classification a function of the parent
    alone, which is the rule that holds after simulation too. The zeros are
    placeholders: every one of them is overwritten by its simulated draw in this
    same round.
    """
    for link in spec.nested_links:
        if link.child_column not in pending_columns:
            continue
        if link.child_column not in frame.columns:
            continue
        observed = frame[link.child_column].notna().to_numpy()
        frame.iloc[
            np.flatnonzero(observed), frame.columns.get_loc(link.child_column)
        ] = 0.0


def _neutralise_composition_cells(
    frame: pd.DataFrame, spec: EngineRecoverySpec, pending_nodes: set[str]
) -> None:
    """Make pending cross-tab cells consistent with their (possibly new) total.

    The engine validates that a cross-tab's cells sum to its total before it will
    build. Once the total has been repointed at the simulated comprehension count,
    the real cells no longer sum to it, so the build would reject the frame. The
    whole total is parked in the first cell — an arbitrary but valid partition —
    and every cell is overwritten by its Dirichlet-Multinomial draw in this same
    round.
    """
    for stage in spec.stages:
        for node in stage:
            if not isinstance(node, CompositionOutcome):
                continue
            if node.rv_name not in pending_nodes:
                continue
            if not all(column in frame.columns for column in node.columns):
                continue
            rows = np.flatnonzero(frame[node.total_column].notna().to_numpy())
            if not rows.size:
                continue
            totals = frame[node.total_column].to_numpy(dtype=float)[rows]
            for position, column in enumerate(node.columns):
                frame.iloc[rows, frame.columns.get_loc(column)] = (
                    totals if position == 0 else 0.0
                )


def _apply_parent_totals(frame: pd.DataFrame, spec: EngineRecoverySpec) -> None:
    """Point cross-tab totals at the simulated parent count they partition."""
    for total_column, parent_column in spec.totals_tracking_parent:
        if total_column not in frame.columns or parent_column not in frame.columns:
            continue
        rows = np.flatnonzero(frame[total_column].notna().to_numpy())
        if rows.size:
            frame.iloc[rows, frame.columns.get_loc(total_column)] = (
                frame[parent_column].to_numpy(dtype=float)[rows]
            )


def _verify_coherence(
    model: pm.Model,
    spec: EngineRecoverySpec,
    frame: pd.DataFrame,
    recorded: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Assert a model rebuilt from the synthetic frame matches what was simulated.

    This is the guard that makes the check meaningful: the generating
    decomposition and the fitted decomposition must be the same one.
    """
    report: dict[str, Any] = {}
    for link in spec.nested_links:
        for data_name in (link.trials_data, link.is_conditional_data):
            if data_name not in recorded:
                continue
            rebuilt = _data_value(model, data_name)
            used = recorded[data_name]
            if rebuilt.shape != used.shape or not np.array_equal(rebuilt, used):
                n_diff = (
                    int(np.sum(rebuilt != used))
                    if rebuilt.shape == used.shape
                    else "shape mismatch"
                )
                raise RuntimeError(
                    f"Nested-likelihood incoherence for {link.child_column}: the model "
                    f"rebuilt from the synthetic frame derives a different {data_name} "
                    f"from the one used to simulate it ({n_diff} differing row(s)). The "
                    "synthetic data would be fitted under a different decomposition "
                    "from the one that generated it."
                )
            report[data_name] = "matches"
        trials = recorded.get(link.trials_data)
        if trials is not None:
            child = frame[link.child_column].to_numpy(dtype=float)
            rows = np.flatnonzero(frame[link.child_column].notna().to_numpy())
            drawn = child[rows]
            if drawn.size != trials.size:
                raise RuntimeError(
                    f"{link.child_column}: {drawn.size} observed value(s) but "
                    f"{trials.size} likelihood denominator(s)."
                )
            if np.any(drawn < 0) or np.any(drawn > trials):
                raise RuntimeError(
                    f"{link.child_column}: simulated count outside [0, denominator]."
                )
            report[f"{link.child_column}_within_denominator"] = "ok"

    for stage in spec.stages:
        for node in stage:
            if not isinstance(node, CompositionOutcome):
                continue
            if not all(column in frame.columns for column in node.columns):
                continue
            rows = np.flatnonzero(frame[node.total_column].notna().to_numpy())
            if not rows.size:
                continue
            cells = frame.iloc[rows][list(node.columns)].to_numpy(dtype=float)
            totals = frame.iloc[rows][node.total_column].to_numpy(dtype=float)
            if not np.allclose(cells.sum(axis=1), totals):
                raise RuntimeError(
                    f"{node.rv_name}: simulated cells do not sum to {node.total_column}."
                )
            report[f"{node.rv_name}_cells_sum_to_total"] = "ok"

    # A cross-tab total that partitions a simulated parent must equal it. If it
    # did not, the comprehension likelihood and the cross-tab likelihood would be
    # conditioning on two different comprehension totals for the same child —
    # which is exactly the disagreement the engine avoids by treating the
    # four-cell sum as the authoritative total.
    for total_column, parent_column in spec.totals_tracking_parent:
        if total_column not in frame.columns or parent_column not in frame.columns:
            continue
        rows = np.flatnonzero(frame[total_column].notna().to_numpy())
        if not rows.size:
            continue
        totals = frame[total_column].to_numpy(dtype=float)[rows]
        parents = frame[parent_column].to_numpy(dtype=float)[rows]
        if not np.allclose(totals, parents, equal_nan=False):
            raise RuntimeError(
                f"{total_column} disagrees with the simulated {parent_column} on "
                f"{int(np.sum(~np.isclose(totals, parents)))} row(s); the cross-tab "
                "and the parent likelihood would condition on different totals."
            )
        report[f"{total_column}_equals_{parent_column}"] = "ok"
    return report


def _prepared_context(
    definition,
    config: str,
    target: RecoveryTarget,
    output_dir: str,
) -> tuple[ModelFitContext, Any]:
    """Prepare the real design and build the model, returning the build stage too.

    Runs the engine's own preparation, prior-configuration and build stages, so
    the design (ages, studies, children, missingness) and the model graph are
    exactly the model of record's.
    """
    os.makedirs(output_dir, exist_ok=True)
    context = ModelFitContext(
        reporting=model_reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=output_dir,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=model_sampling.get_sampling_configuration(config),
        sampling_config_name=config,
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    stages = target.resolve_stages(definition)
    for name in (PREPARE_STAGE_NAME, PRIORS_STAGE_NAME, BUILD_STAGE_NAME):
        _stage_for(stages, name)(context)
    return context, _stage_for(stages, BUILD_STAGE_NAME)


def available_replicates(definition, output_root: str | None = None) -> list[int]:
    """Replicate numbers that have a written simulation, ascending.

    The recovery matrix summarises every replicate that exists, not only the ones
    a particular invocation asked for. Without that, re-scoring one replicate of a
    staged run would overwrite the matrix with a single row and silently drop the
    others.
    """
    root = output_root if output_root is not None else env.output_root()
    directory = os.path.join(root, "recovery")
    if not os.path.isdir(directory):
        return []
    prefix = f"{definition.model_id}-{definition.config_name}-recovery-r"
    found: list[int] = []
    for name in os.listdir(directory):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit() and os.path.isfile(
            os.path.join(directory, name, SIMULATION_FILENAME)
        ):
            found.append(int(suffix))
    return sorted(found)


def simulate_replicate(
    model_key: str,
    config: str,
    *,
    replicate: int = 1,
    truth_source: str = "posterior",
    n_prior_draws: int = 64,
    random_seed: int = 20260725,
    output_root: str | None = None,
    definition=None,
) -> SimulationResult:
    """Simulate one synthetic dataset for ``model_key`` at a known truth.

    Returns the simulated frame and truth draw, and writes both plus a
    provenance record to :func:`simulation_dir`.

    ``definition`` overrides the model of record so a registered sensitivity
    variant can be simulated from its own structure; see
    :func:`vocab_growth.recovery.refit.fit_recovery_replicate`. With
    ``truth_source="posterior"`` the truth is then read from the *variant's* own
    trace, which is the only coherent choice: a variant carrying parameters the
    record does not have has nowhere else to get them.
    """
    if truth_source not in {"posterior", "prior"}:
        raise ValueError("truth_source must be 'posterior' or 'prior'.")
    if replicate < 1:
        raise ValueError("replicate is 1-based.")

    target = recovery_target(model_key)
    definition = MODEL_REGISTRY[model_key] if definition is None else definition
    spec = target.spec
    root = output_root if output_root is not None else env.output_root()
    directory = simulation_dir(definition, replicate, root)
    os.makedirs(directory, exist_ok=True)

    key_value_table(
        "Recovery simulation",
        [
            ("Model", f"{definition.model_id} ({model_key})"),
            ("Engine", spec.engine),
            ("Replicate", replicate),
            ("Truth source", truth_source),
            ("Sampling config", config),
            ("Simulation directory", directory),
        ],
    )

    context, build_stage = _prepared_context(
        definition, config, target, os.path.join(directory, "build")
    )
    model = context.model
    free_rv_names = [rv.name for rv in model.free_RVs]
    n_rows = len(context.analysis_df)

    if truth_source == "posterior":
        record_dir = model_reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=root,
            ci_prob=0.89,
            interval_kind="eti",
        ).output_dir
        truth = truth_from_trace(
            os.path.join(record_dir, "trace.nc"),
            free_rv_names,
            replicate=replicate,
            definition=definition,
            source_data_hash=_current_source_data_hash(),
        )
    else:
        truth = truth_from_prior(
            model,
            free_rv_names,
            replicate=replicate,
            n_prior_draws=n_prior_draws,
            random_seed=random_seed + replicate,
        )
    truth = _with_deterministics(truth, model)
    console.print(
        f"[dim]Truth draw: {truth.source} chain {truth.chain}, draw {truth.draw} "
        f"({len(_as_dataset(truth.tree['posterior']).data_vars)} recorded quantities)[/dim]"
    )

    frame = context.analysis_df.copy()
    simulated_columns: list[str] = []
    skipped_nodes: list[str] = []
    recorded_data: dict[str, np.ndarray] = {}
    row_counts: dict[str, int] = {}

    # Columns still to be drawn, so classification neutralisation only touches
    # outcomes that this simulation is going to overwrite.
    pending_columns = {
        outcome_column(definition, node.column)
        for stage in spec.stages
        for node in stage
        if isinstance(node, CountOutcome)
    }

    pending_nodes = {node.rv_name for stage in spec.stages for node in stage}

    for stage_index, stage in enumerate(spec.stages):
        if stage_index > 0:
            # Rebuild on the frame as simulated so far: the engine re-derives every
            # row denominator and nested/marginal flag from the *simulated* parent.
            _apply_parent_totals(frame, spec)
            _neutralise_child_columns(frame, spec, pending_columns)
            _neutralise_composition_cells(frame, spec, pending_nodes)
            context.set_model_data(context.model_data, frame)
            build_stage(context)
            model = context.model

        present = [node for node in stage if node.rv_name in model.named_vars]
        skipped_nodes.extend(
            node.rv_name for node in stage if node.rv_name not in model.named_vars
        )
        if not present:
            continue

        for link in spec.nested_links:
            for data_name in (link.trials_data, link.is_conditional_data):
                if data_name in model.named_vars:
                    recorded_data[data_name] = _data_value(model, data_name).copy()

        simulated = pm.sample_posterior_predictive(
            truth.tree,
            model=model,
            var_names=[node.rv_name for node in present],
            progressbar=False,
            random_seed=random_seed + 1000 * replicate + stage_index,
            compile_kwargs={"mode": "FAST_COMPILE"},
        )
        drawn = _as_dataset(simulated["posterior_predictive"])

        for node in present:
            values = np.asarray(drawn[node.rv_name].values)
            # One chain, one draw: drop the sample dimensions.
            values = values.reshape(values.shape[2:])
            rows = _mask_rows(model, node, n_rows)
            if isinstance(node, CountOutcome):
                column = outcome_column(definition, node.column)
                _write_column(frame, column, rows, values)
                simulated_columns.append(column)
                pending_columns.discard(column)
                pending_nodes.discard(node.rv_name)
                row_counts[column] = int(rows.size)
            else:
                if values.shape != (rows.size, len(node.columns)):
                    raise ValueError(
                        f"{node.rv_name}: expected {(rows.size, len(node.columns))} "
                        f"cell draws, got {values.shape}."
                    )
                for cell_index, column in enumerate(node.columns):
                    _write_column(frame, column, rows, values[:, cell_index])
                    simulated_columns.append(column)
                _write_column(frame, node.total_column, rows, values.sum(axis=1))
                pending_nodes.discard(node.rv_name)
                row_counts[node.rv_name] = int(rows.size)

    # Final coherence check against a model rebuilt from the finished frame.
    context.set_model_data(context.model_data, frame)
    build_stage(context)
    coherence = _verify_coherence(context.model, spec, frame, recorded_data)

    frame_path = os.path.join(directory, SYNTHETIC_FRAME_FILENAME)
    frame_schema = _write_frame(frame, frame_path)
    truth_path = os.path.join(directory, TRUTH_FILENAME)
    truth.tree.to_netcdf(truth_path)

    write_json_atomic(
        os.path.join(directory, SIMULATION_FILENAME),
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model": {
                "model_key": model_key,
                "model_id": definition.model_id,
                "config_name": definition.config_name,
                "recovery_config_name": recovery_config_name(definition, replicate),
                "definition": normalise_for_json(definition),
            },
            "simulation": {
                "engine": spec.engine,
                "replicate": replicate,
                "sampling_config_name": config,
                "truth_source": truth.source,
                "truth_chain": truth.chain,
                "truth_draw": truth.draw,
                "truth_provenance": truth.provenance,
                "random_seed": random_seed,
                "stage_order": [
                    [node.rv_name for node in stage] for stage in spec.stages
                ],
                "simulated_columns": sorted(set(simulated_columns)),
                "skipped_nodes": sorted(set(skipped_nodes)),
                "likelihood_row_counts": row_counts,
                "conditioned_totals": list(spec.conditioned_totals),
                "coherence_checks": coherence,
                "frame_schema": frame_schema,
                "rows": int(n_rows),
                "frame_file": SYNTHETIC_FRAME_FILENAME,
                "truth_file": TRUTH_FILENAME,
            },
        },
    )

    key_value_table(
        "Simulated outcomes",
        [
            *[(name, f"{count} likelihood rows") for name, count in row_counts.items()],
            ("Coherence checks", f"{len(coherence)} passed"),
            *([("Skipped nodes", ", ".join(sorted(set(skipped_nodes))))] if skipped_nodes else []),
        ],
    )

    return SimulationResult(
        model_key=model_key,
        replicate=replicate,
        directory=directory,
        frame_path=frame_path,
        truth_path=truth_path,
        truth=truth,
        frame=frame,
        simulated_columns=tuple(sorted(set(simulated_columns))),
        skipped_nodes=tuple(sorted(set(skipped_nodes))),
        row_counts=row_counts,
    )


def _sql_literal(path: str) -> str:
    """Single-quoted SQL string literal for a filesystem path.

    DuckDB takes the target of ``COPY … TO`` as a literal rather than a bound
    parameter, so the path is escaped here rather than interpolated raw.
    """
    escaped = path.replace("'", "''")
    return f"'{escaped}'"


def _dtype_class(dtype) -> str:
    """The dtype identity the round-trip guard should hold fixed.

    Compared instead of the raw dtype string because pandas 3 hands back a
    ``str`` dtype for a column of Python strings that went in as ``object``.
    The values are unchanged -- including the case the guard exists for, where a
    numeric-looking id like ``"001"`` must not return as the integer 1 -- so
    failing on that pair rejects a faithful round trip. Every other dtype is
    still compared exactly, and the value check below is unchanged, so an
    id silently becoming numeric is still caught here: it changes the class.
    """
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return "string"
    return str(dtype)


def _write_frame(frame: pd.DataFrame, path: str) -> dict[str, str]:
    """Write the synthetic frame as Parquet via DuckDB, verifying the round trip.

    DuckDB carries its own Parquet reader and writer, so this needs no
    ``pyarrow`` Parquet support — which matters because the pinned environment
    installs ``pyarrow-core`` (Arrow buffers only, pulled in by nutpie) and no
    ``libparquet`` on either locked platform. DuckDB is already a declared
    dependency and already the project's storage layer for the prepared data.

    Parquet keeps dtypes, integer widths and missingness exactly, so unlike a
    text round trip nothing has to be reconstructed from a recorded schema. The
    schema is still recorded as provenance, and the round trip is still verified
    here — including **dtype identity**, which a CSV round trip could not
    guarantee — so a lossy write fails during simulation rather than surfacing as
    an unexplained difference in the refit hours later.
    """
    schema = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    with duckdb.connect() as connection:
        connection.register("synthetic_frame", frame)
        connection.execute(
            f"COPY (SELECT * FROM synthetic_frame) TO {_sql_literal(path)} (FORMAT PARQUET)"
        )
    reloaded = _read_frame(path)
    if list(reloaded.columns) != list(frame.columns):
        raise RuntimeError("Synthetic frame columns changed on the Parquet round trip.")
    for column in frame.columns:
        original, restored = frame[column], reloaded[column]
        if _dtype_class(original.dtype) != _dtype_class(restored.dtype):
            raise RuntimeError(
                f"Column {column!r} changed dtype on the Parquet round trip "
                f"({original.dtype} -> {restored.dtype})."
            )
        if pd.api.types.is_numeric_dtype(original):
            if not np.allclose(
                original.to_numpy(dtype=float),
                restored.to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise RuntimeError(f"Column {column!r} changed on the Parquet round trip.")
        elif not original.astype(str).equals(restored.astype(str)):
            raise RuntimeError(f"Column {column!r} changed on the Parquet round trip.")
    return schema


def _read_frame(path: str) -> pd.DataFrame:
    """Read a synthetic frame back from Parquet, dtypes intact."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No synthetic frame at {path}.")
    with duckdb.connect() as connection:
        return connection.execute(
            f"SELECT * FROM read_parquet({_sql_literal(path)})"
        ).df()


def load_simulation(
    directory: str,
    *,
    expected_definition: Any | None = None,
) -> tuple[pd.DataFrame, xr.DataTree, dict[str, Any]]:
    """Load a written simulation's frame, truth draw and provenance record.

    ``expected_definition`` compares the definition the simulation recorded
    against the one about to consume it, and refuses a mismatch (issue #233).
    The staged workflow is what makes this necessary: ``--simulate-only`` can be
    hours or days ahead of ``--fit-only``, and a definition edited in between
    leaves synthetic data generated by one model being fitted and scored by
    another. Nothing else would catch it -- the frame carries only counts, and
    the truth carries only parameter values under names that mostly survive a
    definition change. Optional so a caller that genuinely wants the raw record
    can ask for it, but every stage of ``fit_recovery.py`` passes it.
    """
    from vocab_growth.fit_artifacts import read_json

    record = read_json(os.path.join(directory, SIMULATION_FILENAME))
    if expected_definition is not None:
        recorded = (record.get("simulation") or {}).get("definition")
        current = normalise_for_json(expected_definition)
        if recorded is not None and recorded != current:
            differing = sorted(
                key
                for key in set(recorded) | set(current)
                if recorded.get(key) != current.get(key)
            )
            raise ValueError(
                f"The simulation at {directory} was generated from a different "
                f"definition than the one now being fitted. Differing field(s): "
                f"{', '.join(differing) or '(structure)'}. Re-run the simulate "
                "step, or check out the revision the simulation was made under."
            )
    frame_file = record.get("simulation", {}).get(
        "frame_file", SYNTHETIC_FRAME_FILENAME
    )
    frame = _read_frame(os.path.join(directory, frame_file))
    truth = xr.open_datatree(os.path.join(directory, TRUTH_FILENAME)).load()
    return frame, truth, record


def build_model_data(frame: pd.DataFrame, definition) -> model_data.BinomialModelData:
    """Reconstruct the engine's ``BinomialModelData`` for a synthetic frame."""
    outcome = getattr(definition, "outcome", None)
    y_column = outcome.value if outcome is not None else "understood"
    y_obs = np.asarray(
        pd.to_numeric(frame[y_column], errors="coerce").fillna(0), dtype=int
    )
    return model_data.BinomialModelData(
        X_obs=np.asarray(frame["age"], dtype=float).reshape(-1, 1),
        y_obs=y_obs,
        n_trials=definition.n_trials,
    )
