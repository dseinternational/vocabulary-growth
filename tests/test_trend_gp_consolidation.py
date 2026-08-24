# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Graph-shape guard for the shared trend + HSGP builder (issue #86).

The inlined trend + HSGP construction was lifted into
``vocab_growth.models.gp_utils`` (``trend_and_gp`` / ``intercept_and_gp``) and
adopted across all six engines. Because ``hsgp.prior`` is the sole RNG-bearing
call, the consolidation must not change any model's PyMC graph. These tests build
the real model for one representative of each engine (no sampling) and pin the
graph shape the helpers must reproduce:

- the trend / GP free RVs each engine creates;
- the named ``Deterministic``\\ s the *store-deterministic* engines keep
  (``g`` / ``f_all`` and the suffixed ``g_u`` / ``f_u_all`` / ``g_q`` / ``h_all``),
  and that the trace-memory engines (trivariate / joint) do **not** store them;
- that the q-side latent is ``h_all`` and never ``f_q_all`` (the latent name is
  passed explicitly, not derived from the suffix); and
- that the signed trajectory uses a three-anchor tent mean: three free anchor RVs
  (``p_slope_low_sign`` / ``p_slope_mid_sign`` / ``p_slope_hi_sign``) and two
  segment-slope ``Deterministic``\\ s (``slope_up_sign`` / ``slope_dn_sign``), with
  no intercept-only ``intercept_sign`` and no single ``slope_sign``.

Builds the real models, so they need the prepared DuckDB; they skip cleanly when
it is absent (the CI fit job runs ``prepare_data`` first, but bare ``pytest`` may
not).
"""

import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import pytest

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth.models import common
from vocab_growth.models import common_bivariate as cb
from vocab_growth.models import common_bivariate_re as cbr
from vocab_growth.models import common_joint_modality as cj
from vocab_growth.models import common_trivariate as ct
from vocab_growth.models import common_univariate_re as cur
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
)

_DEFS = {definition.model_id: definition for definition in MODEL_REGISTRY.values()}


def _build_uncached(model_id, tmp_path, monkeypatch):
    """Build a representative model for one engine (no sampling)."""
    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")
    # The model-graph render shells out to graphviz `dot`; not needed here.
    for mod in (common, cur, cb, cbr, ct, cj):
        monkeypatch.setattr(mod, "render_model_graph", lambda *a, **k: None, raising=False)
    d = _DEFS[model_id]
    ctx = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=d.model_id,
            config_name=d.config_name,
            output_root_dir=str(tmp_path),
            ci_prob=0.90,
            interval_kind="hdi",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(ctx.reporting.output_dir, exist_ok=True)
    if model_id in {"VG01", "VG02", "VG03", "VG04"}:
        common.prepare_univariate_data(ctx, d)
        common.configure_univariate_priors(ctx, d)
        common.build_model(ctx, d)
    elif model_id in {"VG11", "VG12"}:
        cur.prepare_univariate_re_data(ctx, d)
        common.configure_univariate_priors(ctx, d)
        cur.build_univariate_re_model(ctx, d)
    elif model_id == "VG05":
        cb.prepare_bivariate_data(ctx, d)
        cb.configure_bivariate_priors(ctx, d)
        cb.build_model(ctx, d)
    elif model_id in {
        "VG07",
        "VG08",
        "VG09",
        "VG10",
        "VG13",
        "VG16",
        "VG19",
        "VG20",
        "VG21",
        "VG22",
    }:
        cbr.prepare_bivariate_re_data(ctx, d)
        cb.configure_bivariate_priors(ctx, d)
        cbr.build_model_re(ctx, d)
    elif model_id == "VG14":
        ct.prepare_trivariate_data(ctx, d)
        ct.configure_trivariate_priors(ctx, d)
        ct.build_model(ctx, d)
    elif model_id == "VG15":
        cj.prepare_joint_data(ctx, d)
        cj.configure_joint_priors(ctx, d)
        cj.build_model(ctx, d)
    else:
        # Without this, a newly registered model falls through with no model
        # built and fails on "Model has not been set in the context" — which
        # reads as a defect in the model rather than an omission here. VG20 hit
        # exactly that.
        raise AssertionError(
            f"{model_id} is registered but this test does not know how to build "
            "it; add it to the matching engine branch above."
        )
    return ctx.model


@pytest.fixture(scope="session")
def _built_models():
    """One built model per registered id, for the whole session.

    Six of the nineteen models are asked for twice -- once by the parametrised
    build check and again by the dedicated graph test below -- and a build costs
    seconds. Nothing here mutates a model, only reads its variable names, so the
    same object serves both. Under ``--dist loadfile`` this file is one worker's
    work, so the cache is never split across processes.
    """
    return {}


@pytest.fixture
def build(_built_models, tmp_path_factory, monkeypatch):
    def _build_or_reuse(model_id):
        if model_id not in _built_models:
            _built_models[model_id] = _build_uncached(
                model_id, tmp_path_factory.mktemp(model_id), monkeypatch
            )
        return _built_models[model_id]

    return _build_or_reuse


@pytest.mark.parametrize("model_id", _DEFS)
def test_every_registered_model_builds(model_id, build):
    """Catch registry combinations that definition-only validation cannot see."""
    model = build(model_id)

    assert model.free_RVs
    assert model.observed_RVs


def _names(m):
    return (
        {v.name for v in m.free_RVs},
        {d.name for d in m.deterministics},
        set(m.named_vars),
    )


def test_common_vg01_graph(build):
    free, det, named = _names(build("VG01"))
    assert {"p_slope_low", "p_slope_hi", "ell_unit", "eta"} <= free
    assert {"slope", "intercept", "ell", "g", "f_all"} <= det
    assert "f_q_all" not in named


def test_univariate_re_vg11_graph(build):
    free, det, _ = _names(build("VG11"))
    assert {"p_slope_low", "p_slope_hi", "ell_unit", "eta"} <= free
    assert {"delta_subject_raw"} <= free
    assert {"slope", "intercept", "ell", "g", "f_all"} <= det
    assert "delta_subject" in det
    # VG11 carries the variance partition, so the subject scale and the young
    # dispersion anchor are derived from one shared budget rather than sampled as
    # competing free scales. Both keep their names for downstream consumers.
    assert {"v_total", "subject_variance_share"} <= free
    assert {"tau_subject", "kappa_excess_young"} <= det
    # ...and the centred study block, so `delta` is sampled directly.
    assert "delta" in free
    assert "delta_raw" not in free | det


def test_bivariate_vg05_graph(build):
    free, det, named = _names(build("VG05"))
    assert {"p_slope_low_u", "p_slope_hi_u", "p_slope_low_q", "p_slope_hi_q"} <= free
    # The q-side latent is `h_all`, never `f_q_all`.
    assert {"g_u", "f_u_all", "g_q", "h_all"} <= det
    assert "f_q_all" not in named


def test_bivariate_re_vg10_graph(build):
    _, det, named = _names(build("VG10"))
    assert {"g_u", "f_u_all", "g_q", "h_all"} <= det
    assert "f_q_all" not in named


def test_trivariate_vg14_graph(build):
    free, det, _ = _names(build("VG14"))
    # Trace-memory engine: full-grid GP latents are plain tensors, not stored.
    assert {"g_u", "f_u_all", "g_q", "h_all"}.isdisjoint(det)
    # The slope/intercept/ell scalars are still stored.
    assert {"slope_u", "slope_q", "ell_u", "ell_q", "ell_sign"} <= det
    # Signed mean is a three-anchor tent -> three free anchor RVs + two segment
    # slopes as Deterministics; no intercept-only intercept_sign, no single slope.
    assert {"p_slope_low_sign", "p_slope_mid_sign", "p_slope_hi_sign"} <= free
    assert "intercept_sign" not in free
    assert {"slope_up_sign", "slope_dn_sign"} <= det
    assert "slope_sign" not in det


def test_joint_vg15_graph(build):
    free, det, _ = _names(build("VG15"))
    assert {"g_u", "f_u_all", "g_q", "h_all"}.isdisjoint(det)
    assert {"slope_u", "slope_q", "ell_sign"} <= det
    assert {"p_slope_low_sign", "p_slope_mid_sign", "p_slope_hi_sign"} <= free
    assert "intercept_sign" not in free
    assert {"slope_up_sign", "slope_dn_sign"} <= det
    assert "slope_sign" not in det
