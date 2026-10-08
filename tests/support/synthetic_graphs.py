# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build any registered model's real graph on one small fixed synthetic frame.

The fixed frame keeps graph checks independent of changes to the prepared
DuckDB. Fit manifests check prepared data separately with
``data.analysis_frame_hash``.

The default frame has 48 rows, 24 children with two administrations each and
four studies. Ages span the definition's GP domain. The frame supports each
registered builder, but does not represent a plausible study or cover every
missing-data branch. Nothing is fitted to it.
"""

from __future__ import annotations

import os

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd

from vocab_growth.models.catalogue import EngineAdapter, get
from vocab_growth.models.common import ModelFitContext

#: Default row count; each child has two administrations.
N_ROWS = 48
STUDIES = ("uk_01", "uk_02", "us_01", "nz_01")


class _NoopDigraph:
    """Stands in for a Graphviz render; the binary is optional and unused here."""

    def render(self, *args, **kwargs):
        return None


def synthetic_frame(definition, n_rows: int = N_ROWS) -> pd.DataFrame:
    """A deterministic analysis frame every engine's build stage accepts.

    Ages span the definition's own ``gp_domain_months``, because
    ``construct_age_grids`` rejects observations outside it and the
    typically-developing models declare a much narrower domain than the Down
    syndrome ones. Counts are a smooth function of age rather than a random
    draw: nothing is fitted, and a fixed frame keeps the recorded
    log-probability reproducible.
    """
    low, high = getattr(definition, "gp_domain_months", (8, 115))
    ages = np.linspace(float(low) + 0.5, float(high) - 0.5, n_rows)

    understood = np.clip(np.round(ages * 6.0), 1.0, float(definition.n_trials) - 1.0)
    spoken = np.round(understood * 0.45)
    signed = np.round(understood * 0.15)

    frame = pd.DataFrame(
        {
            "age": ages,
            "understood": understood,
            "spoken": spoken,
            "signed": signed,
            "study": [STUDIES[i % len(STUDIES)] for i in range(n_rows)],
            "study_code": [i % len(STUDIES) for i in range(n_rows)],
            "subject_code": np.repeat(np.arange(n_rows // 2), 2),
            "subject_id": np.repeat(np.arange(n_rows // 2), 2),
            "language": ["English (American)"] * n_rows,
        }
    )

    # Reconcile the four cells with the signed, spoken and understood margins.
    frame["signed_spoken"] = np.round(frame["signed"] * 0.4)
    frame["signed_only"] = frame["signed"] - frame["signed_spoken"]
    frame["spoken_only"] = frame["spoken"] - frame["signed_spoken"]
    frame["understood_only"] = (
        frame["understood"]
        - frame["signed_only"]
        - frame["spoken_only"]
        - frame["signed_spoken"]
    )
    frame["produced"] = (
        frame["signed_only"] + frame["spoken_only"] + frame["signed_spoken"]
    )
    frame["prod_signed_only"] = frame["signed_only"]
    frame["prod_spoken_only"] = frame["spoken_only"]
    frame["prod_signed_spoken"] = frame["signed_spoken"]
    frame["prod_total"] = frame["produced"]
    frame["cell_total"] = (
        frame["understood_only"]
        + frame["signed_only"]
        + frame["spoken_only"]
        + frame["signed_spoken"]
    )
    # Each child's two rows have different study codes. A constant form ceiling
    # prevents `lag_same_form_only` from discarding every lag and removing
    # `beta_lag` from the graph.
    frame["survey_vocab_max"] = float(definition.n_trials)

    # Exercise female, male and unrecorded sex codes, constant within each child.
    frame["sex"] = [("F", "M", None)[child % 3] for child in frame["subject_code"]]

    frame["holdout"] = False
    return frame


def build_synthetic_model(
    definition,
    engine: EngineAdapter,
    *,
    output_dir: str,
    monkeypatch=None,
    n_rows: int = N_ROWS,
):
    """Run ``priors`` then ``build`` for ``definition`` and return the context.

    Supply the frame and ``BinomialModelData`` directly to avoid reading the
    prepared DuckDB or producing descriptive reports.
    """
    if monkeypatch is not None:
        monkeypatch.setattr(
            pymc_utils, "model_to_graphviz", lambda model: _NoopDigraph(), raising=False
        )

    frame = synthetic_frame(definition, n_rows=n_rows)
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name="synthetic",
            output_root_dir=output_dir,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        # Never sampled; present because the build stage reads the seed.
        sampling=sampling.SamplingConfiguration(
            draws=4, tune=4, chains=1, cores=1, target_accept=0.8, random_seed=11
        ),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=frame["age"].to_numpy().reshape(-1, 1),
            # The context needs a complete placeholder array; multi-outcome
            # likelihoods read missingness from the unchanged frame below.
            y_obs=frame["understood"].fillna(0).to_numpy().astype(int),
            n_trials=definition.n_trials,
        ),
        frame,
    )
    engine.resolve("priors")(context, definition)
    engine.resolve("build")(context, definition)
    return context


def build_registered_model(model_key: str, *, output_dir: str, monkeypatch=None):
    """As :func:`build_synthetic_model`, for a catalogue key."""
    model = get(model_key)
    return build_synthetic_model(
        model.definition,
        model.engine,
        output_dir=output_dir,
        monkeypatch=monkeypatch,
    )


def graph_fingerprint(model) -> dict:
    """Record variable names, dimensions and coordinate sizes for regression checks.

    Preserve creation order because variable order can affect random sampling.
    Dimensions also form part of the trace interface used by downstream code.
    """
    dims_of = getattr(model, "named_vars_to_dims", {})

    def _dims(name):
        return [str(dim) for dim in (dims_of.get(name) or ())]

    return {
        "free_RVs": [rv.name for rv in model.free_RVs],
        "deterministics": [d.name for d in model.deterministics],
        "observed_RVs": [rv.name for rv in model.observed_RVs],
        "dims": {
            name: _dims(name)
            for name in (
                [rv.name for rv in model.free_RVs]
                + [d.name for d in model.deterministics]
                + [rv.name for rv in model.observed_RVs]
            )
        },
        "coords": {
            str(name): (len(values) if values is not None else None)
            for name, values in sorted(model.coords.items())
        },
    }


def fixed_point(model) -> dict:
    """Choose a valid parameter point to record or use for a local probe.

    This starts from the model's initial values, so changing a prior can change
    the point. Regression comparisons must reuse saved values from the old
    model rather than call this function independently for both models.
    Offsets keep the point away from symmetric prior centres. Transformed
    coordinates allow offsets without directly crossing constrained bounds.
    """
    point = {}
    for index, (name, value) in enumerate(sorted(model.initial_point().items())):
        array = np.asarray(value, dtype=float)
        offset = 0.31 + 0.017 * index
        if array.ndim == 0:
            point[name] = array + offset
        else:
            ramp = 0.011 * np.arange(array.size, dtype=float).reshape(array.shape)
            point[name] = array + offset + ramp
    return point


def fixed_point_logp(model, point: dict | None = None) -> float:
    """Evaluate a saved parameter point, or create one for a local probe.

    Always supply the same ``point`` when comparing compatible models. The
    committed graph regression tests read theirs from graph_reference_points.json.
    """
    if point is None:
        point = fixed_point(model)
    return float(model.compile_logp()(point))
