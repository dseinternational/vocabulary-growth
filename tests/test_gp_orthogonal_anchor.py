# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard the orthogonal-GP anchor.

When a model anchors its HSGP deviation (``anchor_g*_at_ref``), the deviation is
orthogonalised against the linear trend: its constant AND linear-in-age
components are projected out over the grid, so ``g`` carries only nonlinear
curvature and cannot alias with ``slope``. (The previous single-point anchor
``g_unit - g_unit[idx]`` removed only the level trade-off, leaving a trend-vs-GP
R-hat ridge that heavier tuning did not fix.)

This builds the real anchored VG11 model (no sampling) and checks that prior
draws of ``g`` are orthogonal to ``[1, z]`` over the grid; it skips cleanly when
the prepared DuckDB is not present.
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
def vg11_model(tmp_path, monkeypatch):
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")
    monkeypatch.setattr(cur, "render_model_graph", lambda *a, **k: None)
    context = ModelFitContext(
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
    zc = z - z.mean()
    with vg11_model:
        g = pm.draw(vg11_model["g"], draws=48, random_seed=0)
    # Each draw: zero mean and zero linear slope over the grid (orthogonal to [1, z]).
    assert np.abs(g.mean(axis=1)).max() < 1e-6
    slopes = (g * zc).sum(axis=1) / (zc * zc).sum()
    assert np.abs(slopes).max() < 1e-6
