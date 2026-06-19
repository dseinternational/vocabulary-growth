# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Guard the study random-intercept parameterisation in ``build_model_re``.

Issue #65 switched the study-level random intercepts (``delta_u`` / ``delta_q``)
from a centred form (``delta ~ Normal(0, tau)``) to the non-centred form used
everywhere else in the codebase (``delta = tau * delta_raw``, ``delta_raw ~
Normal(0, 1)``). This test pins two things that downstream code depends on:

1. the non-centred raw variables ``delta_u_raw`` / ``delta_q_raw`` exist, and
   the named ``delta_u`` / ``delta_q`` are *deterministic* (not free RVs) — i.e.
   we have not regressed to the centred form; and
2. the public names ``delta_u`` / ``delta_q`` / ``tau_u`` / ``tau_q`` are still
   exposed, because ``scripts/vg07_study_effects.py`` and ``scripts/loso_compare*``
   extract them from the trace by name.

It builds the real VG07 model (no sampling), so it needs the prepared DuckDB; it
skips cleanly when that isn't present (the CI fit job runs ``prepare_data`` first,
but bare ``pytest`` may not).
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
def vg07_model(tmp_path, monkeypatch):
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    # The model-graph render shells out to graphviz `dot`; not needed here.
    monkeypatch.setattr(cbr, "render_model_graph", lambda *a, **k: None)

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=VG07.model_id,
            config_name=VG07.config_name,
            output_root_dir=str(tmp_path),
            hdi=0.90,
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
