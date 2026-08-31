# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build any registered model's real graph on one small fixed synthetic frame.

Refactoring the builders needs a check that the graph did not move, and that
check has to be a function of the **code alone**. Building against the prepared
DuckDB would tie every recorded fingerprint to the data as well, so a legitimate
data change would present as a refactor failure and a real refactor failure
could hide inside one. Data changes are already guarded, exactly, by
``data.analysis_frame_hash``.

So the frame here is synthetic, deterministic and deliberately small: 48 rows,
24 children with two administrations each, four studies, ages spread across
whichever GP domain the definition declares. It is not a plausible dataset and
is not meant to be -- nothing is fitted to it. It exists so that
``build(context, definition)`` runs every branch the real definition selects and
produces the same graph structure and the same log-probability expression it
would on real data.

Every registered model builds on it, all six engines included, at roughly a
second each.
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

#: Rows in the synthetic frame, and administrations per child. Small enough to
#: build quickly, large enough that every study and both administrations of a
#: child are present -- the subject and study random-effect blocks need both.
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

    # The joint engine's four-cell partition of `understood`, reconciled with
    # the recorded margins exactly as its loaders require: signed ==
    # signed_only + signed_spoken, spoken == spoken_only + signed_spoken, and
    # the four cells summing to the comprehension total.
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

    The data-preparation stage is deliberately **not** run: it reads the
    prepared DuckDB, prints tables and writes descriptive CSVs into a fit's
    output directory. Its output for these purposes is a frame and a
    ``BinomialModelData``, which :func:`synthetic_frame` supplies directly.
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
            y_obs=frame["understood"].to_numpy().astype(int),
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
    """What a refactor must not change about a built graph.

    Names **in creation order**, not as sets: the order fixes the sampler's RNG
    stream, so a reordering changes the draws of an otherwise identical model.
    Dims and coords travel with them because a variable that keeps its name and
    loses its dims silently changes what every consumer reads back.
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
    """A deterministic, arbitrary point in the model's transformed space.

    **Not** the model's own initial point, and that is the whole design. PyMC
    initialises a positive parameter at its moment, which for the ``HalfNormal``
    scales this family is built from *is* the scale; on the log transform the
    Jacobian then contributes ``+log(sigma)`` while the density contributes
    ``-log(sigma)``, and the two cancel exactly. A log-probability read at the
    initial point is therefore **invariant to every prior scale in the model** --
    a 1% change to ``eta_u_sigma`` moves it by exactly zero. Measured, not
    reasoned about: that is how this function came to exist.

    Offsetting each coordinate by a fixed amount breaks the cancellation while
    keeping the point reproducible. The offsets vary along the vector so a
    permutation within one array is visible too, and every coordinate is on the
    unconstrained scale, so no offset can leave the support.
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


def fixed_point_logp(model) -> float:
    """The joint log-probability at :func:`fixed_point`.

    One float that depends on every prior, every likelihood term and every
    constant in the graph, evaluated without sampling. It moves if any
    expression moves, which is what makes it a refactor guard rather than a
    structural one -- the fingerprint above would not notice a changed scale or
    a swapped operand.
    """
    return float(model.compile_logp()(fixed_point(model)))
