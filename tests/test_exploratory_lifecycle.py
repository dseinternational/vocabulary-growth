# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG17 and VG18 are explicitly exploratory, and their output says so.

Issue #273 finding 4, resolved on 2026-08-31 by the study owner's decision:
these two modules are exploratory and non-validatable rather than candidates for
a supported lifecycle, and their query grid is clipped to the observation window
rather than the GP domain widened to meet it.

Two claims are worth a standing check.

**The default build works.** It did not. VG17 copied VG01's query grid, which
runs to 90 months, while observing 12-66 and taking that as its GP domain, so
`construct_age_grids` refused and `fit()` raised on its default path -- the model
could not be built at all. The one test that touched the graph rewrote the
configuration to get past it, which is why nothing noticed.

**The output declares itself.** These modules write a trace and a contrast table
into a directory shaped exactly like a registered fit's, carrying none of the
provenance one has. `sync_report_figures.py` already skips them as unregistered,
but that protects the report rather than the person who finds the directory.
"""

from __future__ import annotations

import importlib
import json

import pytest

from vocab_growth.models.catalogue import CATALOGUE
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.exploratory import (
    EXPLORATORY_MARKER_FILENAME,
    MISSING_ARTEFACTS,
    is_exploratory_output,
    vg17,
    vg18,
    write_exploratory_marker,
)

_EXPLORATORY_MODULES = (vg17, vg18)


# --- the default build works ----------------------------------------------------


def test_the_query_grid_lies_inside_the_observation_window():
    """The defect, stated as the property that was violated.

    Checked on the configuration rather than by building, so it holds without
    the prepared DuckDB and names the cause rather than the symptom.
    """
    grid = vg17._config().ages_query
    assert grid, "the query grid is empty"
    assert min(grid) >= vg17.AGE_LO
    assert max(grid) <= vg17.AGE_HI, (
        f"query ages run to {max(grid)} but the model observes "
        f"{vg17.AGE_LO}-{vg17.AGE_HI} and takes that as its GP domain, so "
        "`construct_age_grids` will refuse to build."
    )


def test_the_grid_was_clipped_rather_than_the_domain_widened():
    """The two are different statistical choices; this pins which was taken.

    Widening the domain to 12-90 would extrapolate the HSGP two years past any
    observation and change the HSGP basis. Clipping changes no fitted number.
    """
    from vocab_growth.models.definitions import VG01

    expected = tuple(
        age for age in VG01.ages_query if vg17.AGE_LO <= age <= vg17.AGE_HI
    )
    assert vg17._config().ages_query == expected
    # And the domain is still the observed range: no explicit widening.
    import inspect

    build_source = inspect.getsource(vg17._build)
    assert "gp_domain_months" not in build_source


@pytest.mark.slow
def test_the_default_build_succeeds(require_prepared_data):
    """`fit()` raised here on its default path until issue #273."""
    frame, studies, subjects = vg17._prepare()
    model, plot_ages = vg17._build(
        frame, studies, vg17._config(), subjects=subjects
    )
    names = {rv.name for rv in model.free_RVs}
    # The three structural additions the module exists for.
    assert "beta_sign_raw" in names or "beta_sign" in {
        d.name for d in model.deterministics
    }
    assert "tau_subj" in names
    assert len(plot_ages) == vg17._config().n_plot


# --- the output declares itself -------------------------------------------------


def test_the_marker_records_what_the_output_does_not_carry(tmp_path):
    path = write_exploratory_marker(str(tmp_path), model_label="VG17")
    payload = json.loads(open(path, encoding="utf-8").read())

    assert payload["exploratory"] is True
    assert payload["validatable"] is False
    assert payload["publishable"] is False
    assert payload["model"] == "VG17"
    # The gaps are listed rather than left for the reader to know.
    assert payload["missing_artefacts"] == list(MISSING_ARTEFACTS)
    assert any("fit_manifest.json" in item for item in payload["missing_artefacts"])
    assert any("convergence gate" in item for item in payload["missing_artefacts"])
    assert is_exploratory_output(str(tmp_path))


def test_a_registered_fit_directory_is_not_mistaken_for_an_exploratory_one(tmp_path):
    (tmp_path / "fit_manifest.json").write_text("{}")
    assert not is_exploratory_output(str(tmp_path))


def test_the_marker_lands_before_anything_else_in_the_fit_path():
    """An interrupted run must still leave the directory labelled."""
    import inspect

    source = inspect.getsource(vg17.fit)
    marker_at = source.index("write_exploratory_marker(")
    trace_at = source.index("save_trace(")
    assert marker_at < trace_at, (
        "the marker is written after the trace, so an interrupted run leaves an "
        "unlabelled directory shaped like a publishable fit"
    )


def test_the_sync_skips_unregistered_output():
    """What actually keeps this output out of the report, pinned.

    The marker is for a human reading the directory; this is the mechanism.
    """
    from pathlib import Path

    source = (
        Path(__file__).parents[1] / "scripts" / "sync_report_figures.py"
    ).read_text(encoding="utf-8")
    assert "unregistered model output" in source
    # Keyed on `{model_id}-{config_name}` from MODEL_REGISTRY, which these
    # modules are deliberately absent from, so their directory names cannot match.
    assert "definitions_by_label.get(name)" in source


# --- and they stay unreachable from the registered path -------------------------


def test_the_exploratory_modules_are_not_registered():
    for module in _EXPLORATORY_MODULES:
        assert "exploratory" in module.__name__
    assert "vg17" not in MODEL_REGISTRY
    assert "vg18" not in MODEL_REGISTRY
    assert "vg17" not in CATALOGUE
    assert "vg18" not in CATALOGUE


def test_no_model_wrapper_module_exists_for_them():
    """`fit_model.py` resolves `model_vgNN` by name; nothing here may match.

    Living outside that naming convention is what makes them unreachable by
    construction rather than by remembering.
    """
    for key in ("vg17", "vg18"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"vocab_growth.models.model_{key}")


def test_fit_model_cannot_reach_them():
    from vocab_growth.models import catalogue

    for key in ("vg17", "vg18"):
        with pytest.raises(KeyError, match="not in the model catalogue"):
            catalogue.get(key)


@pytest.mark.parametrize("module", _EXPLORATORY_MODULES)
def test_each_module_says_its_output_must_not_be_published(module):
    """The status is in the docstring a reader actually opens, not only here."""
    assert module.__doc__
    lowered = module.__doc__.lower()
    assert "exploratory" in lowered
    assert "must not" in lowered and "publish" in lowered


@pytest.mark.parametrize("module", _EXPLORATORY_MODULES)
def test_the_publication_hazard_is_in_the_first_lines_not_buried(module):
    """Position, not merely presence.

    VG18 had it as the final clause of a 200-character line at the end of a 50-line
    docstring whose *first* line is a different hazard, and VG17 had it as the
    fourth paragraph. A reader who opens the file and reads the top has to meet it,
    so this pins where it is rather than that it exists somewhere.
    """
    assert module.__doc__
    head = module.__doc__.strip().splitlines()[:5]
    joined = " ".join(head).lower()
    assert "exploratory" in joined, head
    assert "must not be published" in joined, head


def test_the_package_records_why_productionising_is_a_statistical_decision():
    """Routing VG17 through `common_univariate_re` would change the model.

    That engine constrains its study effects to sum to zero; VG17 uses
    unconstrained offsets. Recorded so the next reader does not treat the move
    as packaging.
    """
    import vocab_growth.models.exploratory as package

    assert "sum to zero" in package.__doc__
    assert "266" in package.__doc__


def test_the_marker_filename_sorts_beside_the_file_a_reader_looks_for():
    assert EXPLORATORY_MARKER_FILENAME == "exploratory_output.json"
    assert EXPLORATORY_MARKER_FILENAME.endswith(".json")
