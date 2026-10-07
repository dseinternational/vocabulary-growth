# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Set a non-interactive plotting backend and suppress routine report output.

Selecting Agg before test collection avoids GUI dependencies and import-order
effects. Most tests do not need prior plots or descriptive report tables, so
the fixtures below replace those outputs at their shared source modules.
Tests marked ``emits_reporting_artefacts`` restore the real functions and check
those outputs explicitly in ``test_pipeline_reporting_artefacts.py``.
"""

import matplotlib

# Select the backend before any test imports pyplot.
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

    Session scope applies the patch before module-scoped model builds.

    Opt out with ``@pytest.mark.emits_reporting_artefacts``.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(plot_dist, "plot_distribution", lambda *a, **k: None)
        # `dataframe_table` requires a DataFrame even when reporting is disabled.
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


# Shared by the explicit and marginalised child-effect tests. Separate modules
# let `--dist loadfile` schedule the sampling test on a separate worker.


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


def build_univariate_re_context(definition, tmp_path_factory):
    """Prepare, configure and build one univariate random-effect model."""
    root = str(tmp_path_factory.mktemp(definition.config_name))
    context = ModelFitContext(
        report_build=False,
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
    return build_univariate_re_context(SMALL, tmp_path_factory)


@pytest.fixture(scope="module")
def subject_marginal_context(require_prepared_data, tmp_path_factory):
    """The same model with the singleton child effects integrated out."""
    return build_univariate_re_context(SMALL_MARGINAL, tmp_path_factory)
