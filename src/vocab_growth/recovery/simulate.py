# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Draw synthetic outcomes from engine likelihoods at one fixed parameter draw.

Rebuild the graph between dependent stages so nested outcomes use simulated
comprehension totals. If a lag predictor reads an outcome drawn in the same
stage as its consumer, repeat the stage sequence in wave order. Verify the
recorded predictors, denominators and row classification against a graph
rebuilt from the finished frame.

Preserve the prepared ages, studies, child codes and missingness pattern while
replacing observed counts. Model-generated nested counts do not reproduce source
violations such as spoken counts exceeding comprehension. Recovery therefore
does not test every data problem handled by the real-data loader.
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
from dse_research_utils.statistics.loo import as_dataset

from vocab_growth import environment as env
from vocab_growth.analysis_frames import expected_analysis_frame_hash
from vocab_growth.fit_artifacts import (
    normalise_for_json,
    source_data_hash,
    validate_fit_output,
    write_json_atomic,
)
from vocab_growth.models.common import (
    BUILD_STAGE_NAME,
    PREPARE_STAGE_NAME,
    PRIORS_STAGE_NAME,
    ModelFitContext,
)
from vocab_growth.models.cross_lag import (
    prev_wave_lag_for_frame,
    prev_wave_sign_share_lag_for_frame,
    wave_index,
)
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery import compare
from vocab_growth.recovery.spec import (
    CompositionOutcome,
    CountOutcome,
    EngineRecoverySpec,
    RecoveryTarget,
    outcome_column,
    outcome_dependent_predictor,
    recovery_target,
    single_pass_is_sound,
)
from vocab_growth.recovery.truth_overrides import (
    apply_truth_overrides,
    check_all_finite,
    override_tag,
)
from vocab_growth.reporting import console, key_value_table

SYNTHETIC_FRAME_FILENAME = "synthetic_analysis_frame.parquet"
TRUTH_FILENAME = "truth.nc"
SIMULATION_FILENAME = "simulation.json"


# ==========================================================================
# Output layout
# ==========================================================================


def recovery_label(
    definition, replicate: int, *, truth_definition=None, truth_overrides=()
) -> str:
    """Output label for one recovery replicate, e.g. ``VG10-...-recovery-r01``."""
    return (
        f"{definition.model_id}-"
        f"{recovery_config_name(definition, replicate, truth_definition=truth_definition, truth_overrides=truth_overrides)}"
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


def recovery_config_stem(definition, *, truth_definition=None, truth_overrides=()) -> str:
    """Everything a recovery config name carries before its replicate number.

    Factored out because :func:`available_replicates` finds a run's replicates by
    matching this as a directory prefix: a marker that reached the name but not
    the prefix would make a run's own output invisible to its scoring step.
    """
    stem = definition.config_name
    if (
        truth_definition is not None
        and truth_definition.config_name != definition.config_name
    ):
        stem = f"{stem}-under-{truth_source_tag(truth_definition, definition)}"
    tag = override_tag(truth_overrides)
    return f"{stem}-{tag}" if tag else stem


def recovery_config_name(
    definition, replicate: int, *, truth_definition=None, truth_overrides=()
) -> str:
    """Recovery config name with generating-config and truth-setting markers.

    A generating config distinct from the fitted config adds ``-under-<tag>``.
    Truth overrides add their own token. These distinguish controlled checks
    from self-recovery and from other settings before the replicate suffix.
    """
    stem = recovery_config_stem(
        definition, truth_definition=truth_definition, truth_overrides=truth_overrides
    )
    return f"{stem}-recovery-r{replicate:02d}"


def simulation_dir(
    definition,
    replicate: int,
    output_root: str | None = None,
    *,
    truth_overrides=(),
) -> str:
    """Directory holding one replicate's synthetic data and truth.

    Deliberately *not* under ``models/``: a completed fit atomically replaces its
    own output directory, which would delete the inputs that produced it.
    """
    root = output_root if output_root is not None else env.output_root()
    return os.path.join(
        root,
        "recovery",
        recovery_label(definition, replicate, truth_overrides=truth_overrides),
    )


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
    return as_dataset(node)


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
    """Reported deterministics selected by recovery dimensions and exclusions.

    Store scalars and one-dimensional query, study or child quantities. Variables
    on other dimensions, duplicate logits and raw offsets are omitted. Derived
    total-spread inputs follow the same selection rules.
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
    """Compute selected deterministics from the current graph for a truth draw.

    The caller must first validate any source trace against the intended fit.
    Recomputing a deterministic alone does not establish likelihood compatibility.
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
    """Select a chain position from a golden-ratio low-discrepancy sequence.

    The replicate number fixes the position without needing the final replicate
    count. Spacing reduces clustering relative to adjacent selection but does
    not establish independence, distinct values or freedom from integer-index
    collisions on a finite chain.
    """
    if n_available <= 0:
        raise ValueError("No draws available to select a truth from.")
    if n_available == 1:
        return 0
    position = (replicate * _GOLDEN_RATIO_FRACTION) % 1.0
    return min(int(position * n_available), n_available - 1)


def _current_source_data_hash() -> str:
    """Raw-data fingerprint using the shared artefact-consumer helper.
    """
    return source_data_hash(env.DATA_DIR)


def truth_from_trace(
    trace_path: str,
    free_rv_names: list[str],
    *,
    replicate: int,
    definition: Any | None = None,
    source_data_hash: str | None = None,
) -> TruthDraw:
    """Select one free-parameter draw from a stored fit, then load only that slice.

    When a definition is supplied, validate recorded definition, raw and prepared
    data fingerprints and lifecycle before selection. simulate_replicate supplies
    these checks. They do not request an executable-signature check here.
    Derived quantities are recomputed later from the current graph.
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
            # A truth draw is only a truth draw for the data the fit saw. The
            # raw-CSV fingerprint cannot see loader-rule changes, because the
            # rules run after the CSVs are read (issue #266 finding 1).
            expected_analysis_frame_hash=expected_analysis_frame_hash(
                definition.model_id.lower(), definition
            ),
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
    """Select a free-parameter draw from a seeded prior sample.

    Needs no fitted trace. Selected prior truths can lie far from the posterior
    region used in reports; a finite set does not cover the whole prior.
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


def _restrict(rows: np.ndarray, mutable: np.ndarray | None) -> np.ndarray:
    """Restrict rows to those still mutable in this simulation round.

    ``None`` allows every row. In a wave loop, mutable rows are the current wave
    and later waves; earlier outcomes must remain fixed for lag predictors.
    """
    if mutable is None:
        return rows
    return np.intersect1d(rows, mutable, assume_unique=False)


def _neutralise_child_columns(
    frame: pd.DataFrame,
    spec: EngineRecoverySpec,
    pending_columns: set[str],
    mutable: np.ndarray | None = None,
) -> None:
    """Zero pending observed child counts before rebuilding nested classification.

    Real child counts can exceed a newly simulated parent and wrongly select a
    marginal branch. Zero placeholders let the simulated parent determine the
    branch. Pending outcomes are then overwritten by their likelihood draws;
    earlier waves and missing entries remain unchanged.
    """
    for link in spec.nested_links:
        if link.child_column not in pending_columns:
            continue
        if link.child_column not in frame.columns:
            continue
        observed = frame[link.child_column].notna().to_numpy()
        rows = _restrict(np.flatnonzero(observed), mutable)
        frame.iloc[rows, frame.columns.get_loc(link.child_column)] = 0.0


def _neutralise_composition_cells(
    frame: pd.DataFrame,
    spec: EngineRecoverySpec,
    pending_nodes: set[str],
    mutable: np.ndarray | None = None,
) -> None:
    """Replace pending cross-tabs with a valid placeholder partition of their total.

    Put the total in the first cell and zero the others so the engine can rebuild
    after parent totals change. The pending likelihood draws replace these cells.
    Earlier waves remain fixed.
    """
    for stage in spec.stages:
        for node in stage:
            if not isinstance(node, CompositionOutcome):
                continue
            if node.rv_name not in pending_nodes:
                continue
            if not all(column in frame.columns for column in node.columns):
                continue
            rows = _restrict(
                np.flatnonzero(frame[node.total_column].notna().to_numpy()), mutable
            )
            if not rows.size:
                continue
            totals = frame[node.total_column].to_numpy(dtype=float)[rows]
            for position, column in enumerate(node.columns):
                frame.iloc[rows, frame.columns.get_loc(column)] = (
                    totals if position == 0 else 0.0
                )


def _apply_parent_totals(
    frame: pd.DataFrame, spec: EngineRecoverySpec, mutable: np.ndarray | None = None
) -> None:
    """Point cross-tab totals at the simulated parent count they partition."""
    for total_column, parent_column in spec.totals_tracking_parent:
        if total_column not in frame.columns or parent_column not in frame.columns:
            continue
        rows = _restrict(
            np.flatnonzero(frame[total_column].notna().to_numpy()), mutable
        )
        if rows.size:
            frame.iloc[rows, frame.columns.get_loc(total_column)] = (
                frame[parent_column].to_numpy(dtype=float)[rows]
            )


def _cross_lag_state(frame: pd.DataFrame, definition, n_trials: int) -> np.ndarray:
    """Recompute comprehension-lag inputs with the engine's frame helper.

    Lag inputs are graph constants, not readable pm.Data. Stack previous-wave
    index, presence flag and previous logit for comparison with the finished
    frame. Call before this round mutates the frame used to build the graph.
    """
    prev_idx, has_lag, y_prev_logit = prev_wave_lag_for_frame(
        frame, n_trials, definition
    )
    return np.vstack(
        [
            np.asarray(prev_idx, dtype=float),
            np.asarray(has_lag, dtype=float),
            np.asarray(y_prev_logit, dtype=float),
        ]
    )


def _sign_cross_lag_state(frame: pd.DataFrame, definition, n_trials: int) -> np.ndarray:
    """Recompute signing-lag inputs for the same coherence check as comprehension.

    ``n_trials`` is unused because the predictor is a within-administration ratio.
    Keep it to match the common state-reader signature.
    """
    prev_idx, has_lag, r_prev_logit = prev_wave_sign_share_lag_for_frame(
        frame, definition
    )
    return np.vstack(
        [
            np.asarray(prev_idx, dtype=float),
            np.asarray(has_lag, dtype=float),
            np.asarray(r_prev_logit, dtype=float),
        ]
    )


#: How to read each declared outcome-dependent predictor's design matrix off a
#: frame. A predictor added to `spec.outcome_dependent_predictor` adds an entry
#: here; `_predictor_state` raises rather than skipping the guard if one is
#: missing, so a half-declared predictor stops the run instead of going unchecked.
_PREDICTOR_STATE = {
    "cross_lag": _cross_lag_state,
    "sign_cross_lag": _sign_cross_lag_state,
}


def _predictor_state(predictor, frame: pd.DataFrame, definition, n_trials: int):
    reader = _PREDICTOR_STATE.get(predictor.name)
    if reader is None:
        raise KeyError(
            f"Outcome-dependent predictor {predictor.name!r} is declared in "
            "recovery.spec but has no state reader in simulate._PREDICTOR_STATE, "
            "so its simulation could not be checked. Add one."
        )
    return reader(frame, definition, n_trials)


def _verify_predictor_coherence(
    predictor,
    final: np.ndarray,
    recorded: list[tuple[int, np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Check recorded consuming rows against their finished-frame lag inputs.

    Recorded entries contain round index, written rows and state at graph build.
    Compare indices, presence flags and logits with absolute tolerance 1e-12.
    This runs for both staged and wave-sequential simulation.
    """
    report: dict[str, Any] = {}
    for round_index, rows, used in recorded:
        if not rows.size:
            continue
        agrees = np.all(np.isclose(used[:, rows], final[:, rows], rtol=0, atol=1e-12), axis=0)
        if not agrees.all():
            raise RuntimeError(
                f"Round {round_index} drew {int((~agrees).sum())} of {rows.size} "
                f"row(s) under a {predictor.name!r} predictor the finished "
                "synthetic frame does not reproduce. The data would be generated "
                "under one design matrix and fitted under another. If this is a "
                "new predictor, it reads an outcome drawn no earlier than the node "
                "it enters, and `spec.single_pass_is_sound` must say so."
            )
        report[f"{predictor.name}_round_{round_index}"] = {
            "rows": int(rows.size),
            "rows_with_predictor": int(final[1, rows].sum()),
        }
    return report


def _verify_coherence(
    model: pm.Model,
    spec: EngineRecoverySpec,
    frame: pd.DataFrame,
    recorded: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Check recorded nesting arrays, count bounds and cross-tab totals after rebuild.
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
    """Prepare the supplied definition's real-data design and build its graph.

    Run the engine's preparation, prior and build stages and return the build
    callable for subsequent simulation stages.
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


def available_replicates(
    definition, output_root: str | None = None, *, truth_overrides=()
) -> list[int]:
    """Find replicate numbers with a simulation record under this run's prefix.

    Includes truth-setting markers, so settings do not mix. The scoring driver
    can use every stored replicate when rescoring a subset.
    """
    root = output_root if output_root is not None else env.output_root()
    directory = os.path.join(root, "recovery")
    if not os.path.isdir(directory):
        return []
    stem = recovery_config_stem(definition, truth_overrides=truth_overrides)
    prefix = f"{definition.model_id}-{stem}-recovery-r"
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
    truth_overrides=(),
    n_prior_draws: int = 64,
    random_seed: int = 20260725,
    output_root: str | None = None,
    definition=None,
) -> SimulationResult:
    """Simulate counts and write the frame, truth draw and provenance record.

    ``definition`` selects the generating model or sensitivity variant. Posterior
    truth comes from that definition's fit; prior truth needs no stored fit.
    Apply free-variable overrides before computing the reported deterministics,
    then draw outcomes in the required stage and wave order.
    """
    if truth_source not in {"posterior", "prior"}:
        raise ValueError("truth_source must be 'posterior' or 'prior'.")
    truth_overrides = tuple(truth_overrides)
    if replicate < 1:
        raise ValueError("replicate is 1-based.")

    target = recovery_target(model_key)
    definition = MODEL_REGISTRY[model_key] if definition is None else definition
    spec = target.spec
    root = output_root if output_root is not None else env.output_root()
    directory = simulation_dir(
        definition, replicate, root, truth_overrides=truth_overrides
    )
    os.makedirs(directory, exist_ok=True)

    key_value_table(
        "Recovery simulation",
        [
            ("Model", f"{definition.model_id} ({model_key})"),
            ("Engine", spec.engine),
            ("Replicate", replicate),
            ("Truth source", truth_source),
            *(
                [("Truth settings", ", ".join(str(o) for o in truth_overrides))]
                if truth_overrides
                else []
            ),
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
    if truth_overrides:
        overridden, applied = apply_truth_overrides(
            _as_dataset(truth.tree["posterior"]), model, truth_overrides
        )
        truth.tree = _single_draw_tree(overridden)
        truth.provenance = {**truth.provenance, "truth_overrides": applied}
        for record in applied:
            console.print(
                f"[dim]Truth setting: {record['name']} -> {record['applied']}[/dim]"
            )
    truth = _with_deterministics(truth, model)
    if truth_overrides:
        # After the recomputation, not before: a setting on a boundary reaches
        # the report through a deterministic long before it reaches the sampler.
        check_all_finite(
            _as_dataset(truth.tree["posterior"]), context="The truth settings"
        )
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
    all_columns, all_nodes = set(pending_columns), set(pending_nodes)

    # The rounds to draw in. One pass over the stage list unless a predictor is
    # built from an outcome that a single pass would leave un-simulated when the
    # node consuming it is drawn -- see `spec.single_pass_is_sound`, which derives
    # that from the stage order rather than being told it.
    predictor = outcome_dependent_predictor(definition)
    wave_sequential = predictor is not None and not single_pass_is_sound(
        spec, definition, predictor
    )
    if not wave_sequential:
        waves: np.ndarray | None = None
        rounds = [(0, stage) for stage in spec.stages]
    else:
        waves = wave_index(
            frame["subject_code"].to_numpy(), frame["age"].to_numpy(dtype=float)
        )
        rounds = [
            (wave, stage)
            for wave in range(int(waves.max()) + 1)
            for stage in spec.stages
        ]
    if predictor is not None:
        console.print(
            f"[dim]Outcome-dependent predictor {predictor.name!r} "
            f"({predictor.description}): "
            + (
                f"simulated wave by wave, {int(waves.max()) + 1} waves."
                if wave_sequential
                else "one pass is sound (its source is drawn in an earlier stage)."
            )
            + "[/dim]"
        )
    # One entry per round that draws a consuming node: (round, rows, state).
    predictor_recorded: list[tuple[int, np.ndarray, np.ndarray]] = []
    consumer_rvs = set() if predictor is None else set(predictor.consumer_rv_names)
    current_wave: int | None = None

    for round_index, (wave, stage) in enumerate(rounds):
        if wave != current_wave:
            # A new wave draws every outcome again, for its own rows.
            pending_columns, pending_nodes = set(all_columns), set(all_nodes)
            current_wave = wave
        # Earlier waves are final: neutralisation must not touch them, or it
        # would destroy the values this wave's predictor was built from.
        mutable = None if waves is None else np.flatnonzero(waves >= wave)

        if round_index > 0:
            # Rebuild on the frame as simulated so far: the engine re-derives every
            # row denominator, nested/marginal flag and lag predictor from the
            # *simulated* parent.
            _apply_parent_totals(frame, spec, mutable)
            _neutralise_child_columns(frame, spec, pending_columns, mutable)
            _neutralise_composition_cells(frame, spec, pending_nodes, mutable)
            context.set_model_data(context.model_data, frame)
            build_stage(context)
            model = context.model

        round_state = None
        if predictor is not None and any(
            node.rv_name in consumer_rvs for node in stage
        ):
            # Read from the frame the build just read, before anything in this
            # round changes it, so it is what the build used. Checked against the
            # finished frame after the loop.
            round_state = _predictor_state(
                predictor, frame, definition, context.model_data.n_trials
            )

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
            random_seed=random_seed + 1000 * replicate + round_index,
            compile_kwargs={"mode": "FAST_COMPILE"},
        )
        drawn = _as_dataset(simulated["posterior_predictive"])

        for node in present:
            values = np.asarray(drawn[node.rv_name].values)
            # One chain, one draw: drop the sample dimensions.
            values = values.reshape(values.shape[2:])
            rows = _mask_rows(model, node, n_rows)
            if isinstance(node, CompositionOutcome) and values.shape != (
                rows.size,
                len(node.columns),
            ):
                raise ValueError(
                    f"{node.rv_name}: expected {(rows.size, len(node.columns))} "
                    f"cell draws, got {values.shape}."
                )
            # Under wave-sequential simulation the node is drawn for every row it
            # covers and only this wave's are kept; the rest are redrawn in their
            # own round, against the frame as it will then stand.
            if waves is not None:
                keep = waves[rows] == wave
                rows, values = rows[keep], values[keep]
                if not rows.size:
                    # This node has no rows in the current wave.
                    continue
            if round_state is not None and node.rv_name in consumer_rvs:
                predictor_recorded.append((round_index, rows.copy(), round_state))
            if isinstance(node, CountOutcome):
                column = outcome_column(definition, node.column)
                _write_column(frame, column, rows, values)
                simulated_columns.append(column)
                pending_columns.discard(column)
                pending_nodes.discard(node.rv_name)
                row_counts[column] = row_counts.get(column, 0) + int(rows.size)
            else:
                for cell_index, column in enumerate(node.columns):
                    _write_column(frame, column, rows, values[:, cell_index])
                    simulated_columns.append(column)
                _write_column(frame, node.total_column, rows, values.sum(axis=1))
                pending_nodes.discard(node.rv_name)
                row_counts[node.rv_name] = row_counts.get(node.rv_name, 0) + int(
                    rows.size
                )

    # Rebuild the final frame for coherence checks. recorded_data holds the
    # last round's denominator arrays, not a per-round history. Earlier waves
    # remain final and counts/totals are checked for every row. An engine whose
    # denominators depend on later waves would need per-round denominator checks,
    # as already used for predictors.
    context.set_model_data(context.model_data, frame)
    build_stage(context)
    coherence = _verify_coherence(context.model, spec, frame, recorded_data)
    if predictor is not None:
        coherence.update(
            _verify_predictor_coherence(
                predictor,
                _predictor_state(
                    predictor, frame, definition, context.model_data.n_trials
                ),
                predictor_recorded,
            )
        )

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
                "recovery_config_name": recovery_config_name(
                    definition, replicate, truth_overrides=truth_overrides
                ),
                "definition": normalise_for_json(definition),
            },
            "simulation": {
                "engine": spec.engine,
                "replicate": replicate,
                "sampling_config_name": config,
                "truth_source": truth.source,
                "truth_overrides": [str(o) for o in truth_overrides],
                "truth_chain": truth.chain,
                "truth_draw": truth.draw,
                "truth_provenance": truth.provenance,
                "random_seed": random_seed,
                "stage_order": [
                    [node.rv_name for node in stage] for stage in spec.stages
                ],
                "outcome_dependent_predictor": (
                    None
                    if predictor is None
                    else {
                        "name": predictor.name,
                        "source_column": predictor.source_column,
                        "consumer_rv_names": list(predictor.consumer_rv_names),
                        "single_pass_sound": not wave_sequential,
                    }
                ),
                "waves": None if waves is None else int(waves.max()) + 1,
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
            *(
                []
                if waves is None
                else [("Waves simulated", f"{int(waves.max()) + 1}, in order")]
            ),
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
    """Normalise string/object dtypes for round-trip comparison.

    Pandas may restore Python-string object columns as string dtype. Treat that
    pair as equivalent while preserving numeric dtype identity, so text IDs such
    as 001 cannot silently become integers.
    """
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        return "string"
    return str(dtype)


def _write_frame(frame: pd.DataFrame, path: str) -> dict[str, str]:
    """Write Parquet with DuckDB and check columns, dtypes and restored values.

    Normalise string/object dtype equivalence. Numeric values use allclose with
    its default tolerances and equal_nan; other values compare as strings. This
    is not a bitwise numeric comparison. Record original dtypes as provenance.
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
    """Read a synthetic Parquet frame through DuckDB.
    """
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
    """Load a synthetic frame, truth tree and simulation record.

    If ``expected_definition`` is supplied, compare the recorded generating
    definition before consuming a staged simulation. A match does not itself
    verify executable compatibility or output hashes; resume checkpoints make
    those additional checks.
    """
    from vocab_growth.fit_artifacts import read_json
    from vocab_growth.models.fit_identity import definition_differences

    record = read_json(os.path.join(directory, SIMULATION_FILENAME))
    if expected_definition is not None:
        recorded = (record.get("model") or {}).get("definition")
        if not isinstance(recorded, dict):
            raise ValueError(
                f"The simulation at {directory} records no model definition; "
                "re-run the simulate step before fitting it."
            )
        differences = definition_differences(recorded, expected_definition)
        if differences:
            differing = [difference.describe() for difference in differences]
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
