# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check non-centred, zero-sum study random intercepts in the VG07 graph.

Raw offsets and scales remain free variables. Scaled offsets remain named
deterministics for trace consumers. The graph build requires prepared data.
"""

import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import pytest

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import common_bivariate_re as cbr
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import VG07


@pytest.fixture
def vg07_model(tmp_path):
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    # The model-graph render shells out to graphviz `dot`; not needed here.

    context = ModelFitContext(
        report_build=False,
        reporting=reporting.ReportingConfiguration(
            model_name=VG07.model_id,
            config_name=VG07.config_name,
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)

    cbr.prepare_bivariate_re_data(context, VG07)
    cbr.configure_bivariate_priors(context, VG07)
    cbr.build_model_re(context, VG07)
    return context.model


def test_study_intercepts_are_non_centred(vg07_model):
    named = set(vg07_model.named_vars)
    # Non-centred raw offsets for each study trajectory.
    assert {"delta_u_raw", "delta_q_raw"}.issubset(named)
    # Public names preserved for downstream extraction.
    assert {"delta_u", "delta_q", "tau_u", "tau_q"}.issubset(named)


def test_study_deltas_are_deterministic_not_free(vg07_model):
    deterministics = {d.name for d in vg07_model.deterministics}
    free = {v.name for v in vg07_model.free_RVs}
    # delta_u/delta_q are now derived (tau * raw), so deterministic, not sampled.
    assert {"delta_u", "delta_q"}.issubset(deterministics)
    assert {"delta_u", "delta_q"}.isdisjoint(free)
    # The raw offsets and scales are the sampled quantities.
    assert {"delta_u_raw", "delta_q_raw", "tau_u", "tau_q"}.issubset(free)


def test_study_raw_offsets_are_sum_to_zero(vg07_model):
    """Check zero-sum raw offsets and their underlying sampled variables."""
    import numpy as np
    import pymc as pm

    with vg07_model:
        draws = pm.draw(
            [vg07_model["delta_u_raw"], vg07_model["delta_q_raw"]], draws=64, random_seed=0
        )
    for arr in draws:
        # last axis is study_id; each draw sums to zero up to float tolerance
        assert np.allclose(arr.sum(axis=-1), 0.0, atol=1e-6)

    # The sampled free variables are the zero-sum reparameterisation, not plain Normal.
    free_names = {v.name for v in vg07_model.free_RVs}
    assert {"delta_u_raw", "delta_q_raw"}.issubset(free_names)
    for name in ("delta_u_raw", "delta_q_raw"):
        rv_op = type(vg07_model[name].owner.op).__name__
        assert "ZeroSum" in rv_op, f"{name} is {rv_op}, expected a ZeroSumNormal"
