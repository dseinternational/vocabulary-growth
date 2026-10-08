# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check prior-table coverage against graph-reported parameter names.

Build registered models and sensitivity variants on the fixed synthetic frame.
This checks the branches represented by that fixture; it does not exhaust every
data-dependent branch. Graph tests are slow, while the exemption-only guard
runs in the fast set. No test here samples.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from support.synthetic_graphs import build_synthetic_model

from vocab_growth import report_cells
from vocab_growth.fit_artifacts import normalise_for_json
from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.common import diagnostics_var_names


def _variant_keys():
    """Every registered sensitivity variant, as ``(model_key, variant_name)``."""
    from vocab_growth.sensitivity.registry import VARIANTS

    return sorted(VARIANTS)


def _reported_parameters(engine, definition, tmp_path, monkeypatch):
    """Build the synthetic graph and return its diagnostic parameter names.

    The fixture contains several studies and both joint-cell sources. It also
    provides form ceilings for lag variants. This keeps graph-name checks separate
    from real-data preparation.
    """
    context = build_synthetic_model(
        definition,
        engine,
        output_dir=str(tmp_path / "out"),
        monkeypatch=monkeypatch,
    )
    reported, _ = diagnostics_var_names(context.model)
    assert reported, f"{definition.model_id} reports no parameters at all"
    return reported


def _fit_directory(tmp_path, definition, parameters):
    """A manifest and diagnostics table shaped exactly like a real fit's."""
    (tmp_path / "fit_manifest.json").write_text(
        json.dumps({"model": {"definition": normalise_for_json(definition)}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        index=pd.Index(parameters), data={"r_hat": [1.0] * len(parameters)}
    ).to_csv(tmp_path / "diagnostics.csv")
    return str(tmp_path)


@pytest.mark.slow
@pytest.mark.parametrize("model_key", sorted(CATALOGUE))
def test_the_priors_table_covers_every_reported_parameter(
    model_key, tmp_path, monkeypatch
):
    model = CATALOGUE[model_key]
    definition = model.definition
    reported = _reported_parameters(model.engine, definition, tmp_path, monkeypatch)
    directory = _fit_directory(tmp_path, definition, reported)
    coverage = report_cells.prior_coverage(directory)
    assert coverage["uncovered"] == [], (
        f"{model_key}'s priors table has no row for {coverage['uncovered']}. "
        "Add a row to _PRIOR_SPECS (or a block renderer, for a family whose "
        "size depends on the definition), or add an exemption to "
        "report_cells.PRIOR_EXEMPTIONS with the reason it needs none. A "
        "parameter that is neither is a prior the reader cannot find."
    )
    # And the table itself must not then announce a gap.
    report_cells.render_priors_table(directory)


def test_no_parameter_is_both_rendered_and_exempt():
    """Reject exemptions that could conceal a missing rendered prior row."""
    for parameter, _, _, _ in report_cells._PRIOR_SPECS:
        assert report_cells._is_exempt(parameter) is None, (
            f"{parameter} has a priors-table row and is also exempt; the "
            "exemption would absorb the row's loss"
        )


# --- and the registered sensitivity variants ------------------------------------

_VARIANTS = sorted({(model_key, name) for model_key, name in _variant_keys()})


@pytest.mark.slow
@pytest.mark.parametrize("model_key,variant_name", _VARIANTS)
def test_the_priors_table_covers_every_variant_parameter(
    model_key, variant_name, tmp_path, monkeypatch
):
    """Check variant-specific parameters against the table rendered by their model."""
    from vocab_growth.sensitivity.registry import build_variant

    (definition,) = build_variant(model_key, variant_name)
    engine = CATALOGUE[model_key].engine
    reported = _reported_parameters(engine, definition, tmp_path, monkeypatch)
    coverage = report_cells.prior_coverage(
        _fit_directory(tmp_path, definition, reported)
    )
    assert coverage["uncovered"] == [], (
        f"{model_key}/{variant_name}'s priors table has no row for "
        f"{coverage['uncovered']}."
    )
