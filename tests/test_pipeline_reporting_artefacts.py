# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The reporting output the rest of the suite silences is still produced.

``tests/conftest.py`` patches out :func:`plot_distribution` and
:func:`describe_all` for every other test, because between them they were the
largest single cost in the suite and nothing asserted on either. Silencing an
untested side effect would leave it untested *and* unexercised, so these two
tests opt back in via ``@pytest.mark.emits_reporting_artefacts`` and check the
wiring on a real engine: that configuring a model's priors writes the prior
figures a report embeds, and that preparing its data computes the descriptive
table a report prints.

Deliberately the cheapest engine and a synthetic frame -- what is under test is
that the calls still happen and still land on disk, not what they contain.
"""

import os

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import pytest

import vocab_growth.models.common_bivariate as common_bivariate
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import VG07


def _context(tmp_path):
    n = 12
    ages = np.linspace(12.0, 60.0, n)
    analysis_df = pd.DataFrame(
        {
            "age": ages,
            "understood": np.round(ages * 5.0),
            "spoken": np.round(ages * 3.0),
            "study": ["study_a"] * n,
            "study_code": [0] * n,
            "subject_code": np.repeat(np.arange(n // 2), 2),
        }
    )
    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name="TEST_VG07_ARTEFACTS",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    context.set_model_data(
        model_data.BinomialModelData(
            X_obs=ages.reshape(-1, 1),
            y_obs=np.zeros(n, dtype=int),
            n_trials=VG07.n_trials,
        ),
        analysis_df,
    )
    return context


@pytest.mark.emits_reporting_artefacts
def test_configuring_priors_writes_the_prior_figures(tmp_path):
    context = _context(tmp_path)

    common_bivariate.configure_bivariate_priors(context, VG07)

    written = os.listdir(context.reporting.output_dir)
    png = {name for name in written if name.endswith("_dist.png")}
    svg = {name for name in written if name.endswith("_dist.svg")}
    assert png, f"no prior figures written to {context.reporting.output_dir}"
    # Both formats, and one of each per prior: the report embeds the PNG and the
    # book's PDF build uses the SVG.
    assert {name[:-4] for name in png} == {name[:-4] for name in svg}
    # The context is what the report cells read the figures back through.
    assert {f"{name}_dist" for name in ("ell_unit_u", "eta_u")} <= set(context.plots)


@pytest.mark.emits_reporting_artefacts
def test_describe_all_still_runs_over_the_analysis_frame(tmp_path):
    """The descriptive pass is console-only, so catch it at the call instead."""
    import dse_research_utils.statistics.descriptive as descriptive_stats

    import vocab_growth.data_utils as vocab_data_utils

    # The preparation stage loads the real pool rather than the frame above.
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    seen = []
    real = descriptive_stats.describe_all

    def _record(frame, *args, **kwargs):
        seen.append(frame)
        return real(frame, *args, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(descriptive_stats, "describe_all", _record)
        common_bivariate.prepare_bivariate_data(_context(tmp_path), VG07)

    assert len(seen) == 1, "the preparation stage no longer describes its frame"
    assert not seen[0].empty
