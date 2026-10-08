# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check the anchored VG11 graph without posterior sampling.

Prior draws must pass through zero at the reference anchor and have near-zero
centred linear slope over observed rows. The test requires prepared data;
data-free helper checks are in test_gp_anchor_orthogonalisation.
"""

import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pymc as pm
import pytest

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import common_univariate_re as cur
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import VG11


@pytest.fixture
def vg11_model(tmp_path):
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")
    context = ModelFitContext(
        report_build=False,
        reporting=reporting.ReportingConfiguration(
            model_name=VG11.model_id,
            config_name=VG11.config_name,
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    cur.prepare_univariate_re_data(context, VG11)
    cur.configure_univariate_priors(context, VG11)
    cur.build_univariate_re_model(context, VG11)
    return context.model


def test_anchored_gp_is_orthogonal_to_linear_trend(vg11_model):
    assert VG11.anchor_g_at_ref, "VG11 is expected to anchor its GP"
    z = vg11_model["X_all_z"].get_value()[:, 0]
    n_obs = len(vg11_model.coords["obs_id"])
    with vg11_model:
        g = pm.draw(vg11_model["g"], draws=48, random_seed=0)
    # Point anchor: some grid row is pinned to zero on every draw (the reference age).
    assert np.abs(g).max(axis=0).min() < 1e-6
    # Centred slopes stay near zero after the point anchor adds a constant shift.
    g_obs = g[:, :n_obs]
    zc = z[:n_obs] - z[:n_obs].mean()
    gc = g_obs - g_obs.mean(axis=1, keepdims=True)
    slopes = (gc * zc).sum(axis=1) / (zc * zc).sum()
    assert np.abs(slopes).max() < 1e-6
