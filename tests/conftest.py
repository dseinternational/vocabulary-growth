# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Suite-wide fixtures.

Two things happen here, both of which used to be done ad hoc — or not at all —
in individual modules.

**The matplotlib backend is fixed before anything imports pyplot.** Five modules
used to call ``matplotlib.use("Agg")`` at import time, and the suite only worked
because alphabetical collection happened to reach one of them before any module
that draws. Run a subset that excludes them and matplotlib falls back to TkAgg,
tries to build real GUI windows, and takes the interpreter down with a fatal Tk
exception. A ``conftest.py`` is imported before any test module, so setting it
once here is both correct and order-independent.

**The fit pipeline's reporting output is silenced.** Every engine's
``configure_*_priors`` stage renders roughly ten prior distributions to PNG and
SVG, and every ``prepare_*_data`` stage runs :func:`describe_all` over the whole
analysis frame. Both exist to populate a model's report; nothing in a test run
reads either, and together they were the largest single cost in the suite --
4.1--4.4 s per model build for the plots, and 18 s for the descriptive pass over
the typically-developing pool. Both are looked up as module attributes at call
time, so patching them at the source reaches all six ``common_*`` engines at
once and lets the per-module stubs go.

Neither is patched blind: ``tests/test_pipeline_reporting_artefacts.py`` carries
the ``emits_reporting_artefacts`` marker, which opts out of the fixture and
asserts that a real build still writes its prior plots and still computes its
descriptive statistics. That turns two expensive implicit side effects into one
cheap explicit test.

See ``notes/202608241530-test-suite-performance.md``.
"""

import matplotlib

# Before `import matplotlib.pyplot` anywhere: selecting a backend after pyplot
# has been imported is a different, weaker operation.
matplotlib.use("Agg")

import dataclasses  # noqa: E402
import os  # noqa: E402

import dse_research_utils.plot.distributions as plot_dist  # noqa: E402
import dse_research_utils.statistics.descriptive as descriptive_stats  # noqa: E402
import dse_research_utils.statistics.models.reporting as reporting  # noqa: E402
import dse_research_utils.statistics.models.sampling as sampling  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import vocab_growth.data_utils as vocab_data_utils  # noqa: E402
from vocab_growth.models import common_univariate_re as cur  # noqa: E402
from vocab_growth.models.common import ModelFitContext  # noqa: E402
from vocab_growth.models.definitions import (  # noqa: E402
    VG12,
    SingletonMarginalisationParams,
    UnivariateMarginalisedREModelDefinition,
    _as_definition_subclass,
)

# Captured before anything is patched, so the opt-out below can put the real
# implementations back.
_REAL_PLOT_DISTRIBUTION = plot_dist.plot_distribution
_REAL_DESCRIBE_ALL = descriptive_stats.describe_all


@pytest.fixture(scope="session", autouse=True)
def quiet_pipeline_reporting():
    """Silence the fit pipeline's figure and console output for the session.

    Session-scoped, not function-scoped, because pytest sets a higher-scoped
    fixture up first: a module-scoped fixture that builds a model would
    otherwise be constructed before any function-scoped silencing applied, and
    would pay the full cost this exists to avoid.

    Opt out with ``@pytest.mark.emits_reporting_artefacts``.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(plot_dist, "plot_distribution", lambda *a, **k: None)
        # `describe_all`'s return value is handed straight to `dataframe_table`,
        # so the stand-in has to be a frame rather than None.
        monkeypatch.setattr(
            descriptive_stats, "describe_all", lambda *a, **k: pd.DataFrame()
        )
        yield


@pytest.fixture(autouse=True)
def restore_pipeline_reporting(request, monkeypatch):
    """Put the real implementations back for the tests that assert on them."""
    if "emits_reporting_artefacts" not in request.keywords:
        return

    monkeypatch.setattr(plot_dist, "plot_distribution", _REAL_PLOT_DISTRIBUTION)
    monkeypatch.setattr(descriptive_stats, "describe_all", _REAL_DESCRIBE_ALL)


# ---------------------------------------------------------------------------
# The singleton-marginalisation engine, shared by test_subject_marginal.py and
# test_subject_marginal_sampling.py. The two live in separate modules so that
# `--dist loadfile` can put the sampler run — by some way the longest single
# test in the suite — on a worker of its own; the build itself costs about two
# seconds, so each module paying for its own copy is immaterial.
# ---------------------------------------------------------------------------


# A cheap stand-in for VG12: same engine, a twentieth of the children.
SMALL = dataclasses.replace(VG12, sample_fraction=0.05, min_study_observations=20)
SMALL_MARGINAL = _as_definition_subclass(
    SMALL,
    UnivariateMarginalisedREModelDefinition,
    singleton_marginalisation=SingletonMarginalisationParams(n_nodes=12),
    config_name="marg-test",
)


@pytest.fixture(scope="module")
def require_prepared_data():
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")


def build_univariate_re_context(definition, tmp_path_factory, monkeypatch):
    """Prepare, configure and build one univariate random-effect model."""
    monkeypatch.setattr(cur, "render_model_graph", lambda *a, **k: None)
    root = str(tmp_path_factory.mktemp(definition.config_name))
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=root,
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    cur.prepare_univariate_re_data(context, definition)
    cur.configure_univariate_priors(context, definition)
    cur.build_univariate_re_model(context, definition)
    return context


@pytest.fixture(scope="module")
def subject_explicit_context(require_prepared_data, tmp_path_factory):
    """VG12 at a twentieth of the children, with every child effect explicit."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield build_univariate_re_context(SMALL, tmp_path_factory, monkeypatch)


@pytest.fixture(scope="module")
def subject_marginal_context(require_prepared_data, tmp_path_factory):
    """The same model with the singleton child effects integrated out."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield build_univariate_re_context(
            SMALL_MARGINAL, tmp_path_factory, monkeypatch
        )
