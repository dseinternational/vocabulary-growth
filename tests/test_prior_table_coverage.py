# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Every registered model's priors table must describe every parameter it fits.

The graph-to-report contract asked for by issue #273. `_PRIOR_SPECS` maps one
fitted parameter to one definition field, and a parameter family that fits
neither half of that shape simply produces no row -- silently, because the table
is built from the specs rather than checked against the fit. VG22 lost its whole
low-rank factor block that way, and the only record was a sentence hand-written
into that model's template.

The check is made against each model's **real graph**: the same variable set
`common.diagnostics_var_names` writes into `diagnostics.csv`, which is what the
priors table gates on. Building twenty graphs needs the prepared DuckDB and
takes minutes, so this is `slow`; nothing here samples.

Writing it found three more omissions of the same class straight away: VG15's
Dirichlet-Multinomial concentration, which has its own prior figure and is named
on that model's page as a prior-sensitivity target; and, among the registered
sensitivity variants, the separate-dispersion fallback offset and Proposal A1's
age-varying scale ratio. Variants are covered here because a variant fit renders
the model of record's own template, so a prior with no row shows up on a real
page.
"""

from __future__ import annotations

import importlib
import json
import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import pandas as pd
import pytest

from vocab_growth import report_cells
from vocab_growth.fit_artifacts import normalise_for_json
from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.common import ModelFitContext, diagnostics_var_names

pytestmark = pytest.mark.slow


def _variant_keys():
    """Every registered sensitivity variant, as ``(model_key, variant_name)``."""
    from vocab_growth.sensitivity.registry import VARIANTS

    return sorted(VARIANTS)


def _reported_parameters(engine, definition, tmp_path, monkeypatch):
    """Build ``definition``'s graph and return what ``diagnostics.csv`` would hold."""
    # The build stage renders a Graphviz diagram; the binary is optional and the
    # picture is not what is under test.
    engine_module = importlib.import_module(engine.module)
    if hasattr(engine_module, "render_model_graph"):
        monkeypatch.setattr(engine_module, "render_model_graph", lambda *a, **k: None)

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=str(tmp_path / "out"),
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    engine.resolve("prepare")(context, definition)
    engine.resolve("priors")(context, definition)
    engine.resolve("build")(context, definition)

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


@pytest.mark.parametrize("model_key", sorted(CATALOGUE))
def test_the_priors_table_covers_every_reported_parameter(
    model_key, tmp_path, monkeypatch, require_prepared_data
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


@pytest.mark.parametrize("model_key", sorted(CATALOGUE))
def test_no_parameter_is_both_rendered_and_exempt(model_key):
    """An exemption that shadows a rendered row would hide a lost row.

    Checked on the exemption predicate alone, so it needs no graph: any
    parameter `_PRIOR_SPECS` names a row for must not also be exempt, or
    dropping that row would leave the coverage check silent.
    """
    for parameter, _, _, _ in report_cells._PRIOR_SPECS:
        assert report_cells._is_exempt(parameter) is None, (
            f"{parameter} has a priors-table row and is also exempt; the "
            "exemption would absorb the row's loss"
        )


# --- and the registered sensitivity variants ------------------------------------

_VARIANTS = sorted({(model_key, name) for model_key, name in _variant_keys()})


@pytest.mark.parametrize("model_key,variant_name", _VARIANTS)
def test_the_priors_table_covers_every_variant_parameter(
    model_key, variant_name, tmp_path, monkeypatch, require_prepared_data
):
    """A variant renders the model of record's template, gaps and all.

    Two variants had one: `fallback-dispersion` samples an offset the table had
    no row for, and `a1-tau-age-varying` names its ratio parameter
    `log_tau_subj_u_ratio`, which the block row's coverage rule did not
    recognise.
    """
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
