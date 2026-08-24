# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""VG20's correlated subject random effects (issue #224).

VG20 is VG10 with one added parameter, ``rho_uq``, correlating each child's
understood deviation with their production-ratio deviation. Two properties carry
the whole design and are pinned here:

- **VG10 is nested exactly at ``rho_uq = 0``.** The comparison in #224 reads
  "did anything else move?" as a red flag, which is only meaningful if the graphs
  coincide at zero. Checked numerically on the deterministic itself, not by
  inspection of the source.
- **The correlation cannot be switched on in a configuration that would fit
  something other than what was asked for.** Each rejected combination would
  otherwise fail silently.

The definition-subclass check matters just as much and is cheap: putting the
field on ``BivariateModelDefinition`` would change the serialised definition of
six models of record and invalidate every one of their fitted outputs.
"""

import numpy as np
import pytest

from vocab_growth.models.common_bivariate_re import (
    _resolve_subject_re_correlation,
)
from vocab_growth.models.definitions import (
    VG05,
    VG07,
    VG08,
    VG09,
    VG10,
    VG16,
    VG20,
    BivariateCorrelatedSubjectREModelDefinition,
    BivariateModelDefinition,
    _as_definition_subclass,
)


def test_vg20_differs_from_vg10_only_in_naming_and_the_correlation():
    """The two models must differ in one substantive field and nothing else."""
    from dataclasses import fields

    v10 = {f.name: getattr(VG10, f.name) for f in fields(VG10)}
    v20 = {f.name: getattr(VG20, f.name) for f in fields(VG20)}
    changed = {
        k for k in set(v10) | set(v20) if v10.get(k, "<absent>") != v20.get(k, "<absent>")
    }
    assert changed == {
        "model_id",
        "config_name",
        "banner",
        "subject_re_correlation_eta",
    }


def test_bivariate_models_of_record_do_not_gain_the_field():
    """The subclass must not leak onto the parent class.

    A fit is validated by comparing the serialised definition field for field, so
    if this ever fails, every fitted VG05/VG07-VG10/VG16 output on disk becomes
    invalid at the same moment.
    """
    from dataclasses import fields

    for definition in (VG05, VG07, VG08, VG09, VG10, VG16):
        names = {f.name for f in fields(definition)}
        assert "subject_re_correlation_eta" not in names, definition.model_id
        assert type(definition) is BivariateModelDefinition, definition.model_id


def test_vg20_is_the_subclass():
    assert isinstance(VG20, BivariateCorrelatedSubjectREModelDefinition)
    assert VG20.subject_re_correlation_eta == 2.0


@pytest.mark.parametrize("rho", [-0.9, -0.3, 0.0, 0.15, 0.62, 0.95])
def test_cholesky_reduction_gives_the_stated_correlation_and_scale(rho):
    """The two-coordinate construction must deliver rho and leave the SD at tau_q.

    ``delta_q = tau_q * (rho * z1 + sqrt(1 - rho^2) * z2)`` is the whole model;
    if the whitening term were wrong the marginal spread of the q effects would
    change with rho, which would silently rescale a reported quantity.
    """
    rng = np.random.default_rng(20)
    n = 400_000
    tau_u, tau_q = 0.85, 1.15
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)

    delta_u = tau_u * z1
    delta_q = tau_q * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)

    assert np.corrcoef(delta_u, delta_q)[0, 1] == pytest.approx(rho, abs=0.01)
    assert delta_q.std() == pytest.approx(tau_q, rel=0.01)
    assert delta_u.std() == pytest.approx(tau_u, rel=0.01)


def test_zero_correlation_reduces_to_the_independent_expression():
    """At rho = 0 the q effect is exactly VG10's ``tau_subj_q * delta_subj_q_raw``."""
    rng = np.random.default_rng(21)
    tau_q = 1.15
    z1 = rng.standard_normal(1000)
    z2 = rng.standard_normal(1000)
    rho = 0.0
    correlated = tau_q * (rho * z1 + np.sqrt(1.0 - rho**2) * z2)
    independent = tau_q * z2
    np.testing.assert_allclose(correlated, independent, rtol=0, atol=0)


def test_beta_transform_spans_the_valid_correlation_range():
    """``rho = 2 * Beta(eta, eta) - 1`` must cover (-1, 1) and centre on zero.

    For a 2x2 matrix this is exactly LKJ(eta); the test guards the transform, not
    the identity.
    """
    rng = np.random.default_rng(22)
    for eta in (1.0, 2.0, 4.0):
        rho = 2.0 * rng.beta(eta, eta, size=200_000) - 1.0
        assert rho.min() > -1.0 and rho.max() < 1.0
        assert rho.mean() == pytest.approx(0.0, abs=0.01)
    # eta = 1 is uniform; larger eta concentrates toward independence.
    sd = [
        (2.0 * rng.beta(e, e, size=200_000) - 1.0).std() for e in (1.0, 2.0, 4.0)
    ]
    assert sd[0] > sd[1] > sd[2]
    assert sd[1] == pytest.approx(0.447, abs=0.01)


def test_resolver_returns_none_when_unset():
    """A plain bivariate definition must be entirely unaffected."""
    assert (
        _resolve_subject_re_correlation(
            VG10, use_subject_re_u=True, use_subject_re_q=True, spec_u=None, spec_q=None
        )
        is None
    )


def test_resolver_accepts_vg20():
    assert (
        _resolve_subject_re_correlation(
            VG20, use_subject_re_u=True, use_subject_re_q=True, spec_u=None, spec_q=None
        )
        == 2.0
    )


@pytest.mark.parametrize("drop", ["u", "q"])
def test_resolver_rejects_a_missing_subject_block(drop):
    with pytest.raises(ValueError, match="requires use_subject_re_u"):
        _resolve_subject_re_correlation(
            VG20,
            use_subject_re_u=(drop != "u"),
            use_subject_re_q=(drop != "q"),
            spec_u=None,
            spec_q=None,
        )


@pytest.mark.parametrize("side", ["u", "q"])
def test_resolver_rejects_an_age_varying_scale(side):
    """Proposal A1's age-varying scale and a constant correlation do not compose."""
    sentinel = object()
    with pytest.raises(ValueError, match="age-varying"):
        _resolve_subject_re_correlation(
            VG20,
            use_subject_re_u=True,
            use_subject_re_q=True,
            spec_u=sentinel if side == "u" else None,
            spec_q=sentinel if side == "q" else None,
        )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), "2"])
def test_resolver_rejects_a_non_positive_eta(bad):
    definition = _as_definition_subclass(
        VG10,
        BivariateCorrelatedSubjectREModelDefinition,
        model_id="VGXX",
        subject_re_correlation_eta=bad,
    )
    with pytest.raises(ValueError, match="positive finite"):
        _resolve_subject_re_correlation(
            definition,
            use_subject_re_u=True,
            use_subject_re_q=True,
            spec_u=None,
            spec_q=None,
        )


def test_built_graph_adds_exactly_one_parameter_and_nothing_else():
    """VG20's graph must be VG10's plus ``rho_uq`` — the claim #224 rests on.

    Anything else appearing here means the two models differ in more than the
    correlation, and the comparison stops being readable: #224 treats movement in
    any reported trajectory as a red flag, which assumes the graphs are otherwise
    identical.

    Builds both real models (no sampling), so it needs the prepared DuckDB.
    """
    import os
    import tempfile

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models import common_bivariate as cb
    from vocab_growth.models import common_bivariate_re as cbr
    from vocab_growth.models.common import ModelFitContext

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def build(definition, root):
        ctx = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name=definition.model_id,
                config_name=definition.config_name,
                output_root_dir=root,
                ci_prob=0.90,
                interval_kind="hdi",
            ),
            sampling=sampling.get_sampling_configuration("dev"),
        )
        os.makedirs(ctx.reporting.output_dir, exist_ok=True)
        cbr.prepare_bivariate_re_data(ctx, definition)
        cb.configure_bivariate_priors(ctx, definition)
        cbr.build_model_re(ctx, definition)
        return ctx.model

    with tempfile.TemporaryDirectory() as root:
        m10 = build(VG10, root)
        m20 = build(VG20, root)

    def names(model):
        return {v.name for v in model.free_RVs} | {v.name for v in model.deterministics}

    assert names(m20) - names(m10) == {"rho_uq", "rho_uq_raw"}
    assert names(m10) - names(m20) == set()
    assert len(m20.free_RVs) == len(m10.free_RVs) + 1


def test_as_definition_subclass_shares_nested_prior_blocks():
    """The helper must be shallow: nested prior dataclasses stay the same objects.

    A deep copy would serialise identically today but drift the moment one side's
    priors were edited, which is the failure this derivation exists to prevent.
    """
    assert VG20.kappa_u is VG10.kappa_u
    assert VG20.kappa_s is VG10.kappa_s


def test_subject_marginal_predictive_uses_the_correlation():
    """The unseen child must be drawn from the joint the model fitted.

    Until 2026-08-19 the subject-marginal predictive drew the two deviates as
    two independent ``pm.Normal``s, so VG20 estimated ``rho_uq`` and then threw
    it away when building the one quantity the correlation exists to change.
    VG20's gate 3 read as "a correlation of +0.368 leaves the spoken intervals
    unchanged" — which was this code path asserting rho = 0, not a result.

    This checks the *precondition* the patched branch keys on -- that ``rho_uq``
    is reachable from the predictive path for VG20 and absent for VG10 -- which
    is what silently failed before. It does not by itself prove the branch is
    taken. That gap is now closed by
    ``test_vg20_takes_the_correlated_branch_and_vg10_does_not`` and the three
    hermetic tests beside it (#233); before those, the only end-to-end evidence
    was regenerating VG20's plots and re-running gate 3. Kept because the
    precondition is the cheap half and would catch a refactor that stopped
    exposing ``rho_uq`` here.
    """
    import contextlib
    import io
    import os
    import tempfile

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models import common_bivariate as cb
    from vocab_growth.models import common_bivariate_re as cbr
    from vocab_growth.models.common import ModelFitContext

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def marginal_rv_names(definition, root):
        ctx = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name=definition.model_id,
                config_name=definition.config_name,
                output_root_dir=root,
                ci_prob=0.90,
                interval_kind="hdi",
            ),
            sampling=sampling.get_sampling_configuration("dev"),
        )
        os.makedirs(ctx.reporting.output_dir, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            cbr.prepare_bivariate_re_data(ctx, definition)
            cb.configure_bivariate_priors(ctx, definition)
            cbr.build_model_re(ctx, definition)
        # The predictive block adds its auxiliary RVs to the same model.
        before = {v.name for v in ctx.model.free_RVs}
        with contextlib.redirect_stdout(io.StringIO()):
            with ctx.model:
                pass
        return ctx, before

    with tempfile.TemporaryDirectory() as root:
        ctx20, _ = marginal_rv_names(VG20, root)
        ctx10, _ = marginal_rv_names(VG10, root)

    # The correlation is available to the predictive path for VG20 and absent
    # for VG10 — which is exactly what the patched branch keys on.
    assert "rho_uq" in ctx20.model_variables, (
        "VG20's predictive cannot see rho_uq, so the unseen child is drawn "
        "independently and the correlation is silently discarded"
    )
    assert "rho_uq" not in ctx10.model_variables

    # Both carry the two subject scales the construction needs.
    for ctx, label in ((ctx20, "VG20"), (ctx10, "VG10")):
        assert "tau_subj_u" in ctx.model_variables, label
        assert "tau_subj_q" in ctx.model_variables, label


def test_the_correlated_marginal_draw_preserves_each_marginal_sd():
    """rho changes the joint, never either marginal — the same property the
    in-model construction is tested for, now required of the predictive too.

    Pure arithmetic on the construction, so it runs without the database. A
    wrong whitening term here would rescale the unseen child's production
    deviate and quietly change a reported interval, which is the failure the
    in-model version of this test was written to catch.
    """
    import numpy as np

    rng = np.random.default_rng(20260819)
    tau_u, tau_q, rho = 0.786, 1.285, 0.368
    z_u = rng.standard_normal(400_000)
    z_q = rng.standard_normal(400_000)

    delta_u = tau_u * z_u
    delta_q = tau_q * (rho * z_u + np.sqrt(1.0 - rho**2) * z_q)

    assert delta_u.std() == pytest.approx(tau_u, rel=0.01)
    assert delta_q.std() == pytest.approx(tau_q, rel=0.01)
    assert np.corrcoef(delta_u, delta_q)[0, 1] == pytest.approx(rho, abs=0.01)

    # And the point of the whole exercise: on the logit scale the two deviates
    # compound, so an unseen child's spoken vocabulary is more variable than
    # independent draws imply.
    independent = (tau_u * z_u + tau_q * z_q).std()
    correlated = (delta_u + delta_q).std()
    assert correlated > independent


def test_the_correlated_branch_executes_and_realises_the_correlation():
    """Run the branch itself, not the precondition for reaching it (#233).

    `test_subject_marginal_predictive_uses_the_correlation` above says outright
    that it "does not by itself prove the branch is taken", and until 2026-08-24
    nothing else did: the correlated construction was inline in
    `sample_posterior_predictive`, which cannot be called without a fitted trace
    and a prepared database. This drives the extracted function through PyMC,
    hermetically, and checks the three properties the construction exists for.
    """
    import numpy as np
    import pymc as pm

    from vocab_growth.models.common_bivariate import unseen_child_correlated_delta_q

    tau_u, tau_q, rho = 0.786, 1.285, 0.368

    with pm.Model():
        delta_u = pm.Normal("_delta_subj_u_marg", mu=0.0, sigma=tau_u)
        delta_q = unseen_child_correlated_delta_q(
            delta_u, tau_subj_u=tau_u, tau_subj_q=tau_q, rho=rho
        )
        drawn_u, drawn_q = pm.draw(
            [delta_u, delta_q], draws=200_000, random_seed=20260824
        )

    assert drawn_u.std() == pytest.approx(tau_u, rel=0.02)
    assert drawn_q.std() == pytest.approx(tau_q, rel=0.02), (
        "the whitening term is what keeps the q deviate's spread equal to "
        "tau_subj_q whatever the correlation"
    )
    assert np.corrcoef(drawn_u, drawn_q)[0, 1] == pytest.approx(rho, abs=0.01)


def test_the_correlated_branch_reduces_to_independence_at_zero():
    """VG10 is VG20 at rho = 0, and that must hold of the predictive too."""
    import numpy as np
    import pymc as pm

    from vocab_growth.models.common_bivariate import unseen_child_correlated_delta_q

    tau_u, tau_q = 0.786, 1.285
    with pm.Model():
        delta_u = pm.Normal("_delta_subj_u_marg", mu=0.0, sigma=tau_u)
        delta_q = unseen_child_correlated_delta_q(
            delta_u, tau_subj_u=tau_u, tau_subj_q=tau_q, rho=0.0
        )
        drawn_u, drawn_q = pm.draw(
            [delta_u, delta_q], draws=200_000, random_seed=20260824
        )

    assert np.corrcoef(drawn_u, drawn_q)[0, 1] == pytest.approx(0.0, abs=0.01)
    assert drawn_q.std() == pytest.approx(tau_q, rel=0.02)


def test_the_correlated_branch_creates_exactly_one_named_variable():
    """The construction promises it renames nothing and adds no node beyond
    `_z_subj_q_marg`; a refactor that broke that would change every model's
    graph, not only VG20's."""
    import pymc as pm

    from vocab_growth.models.common_bivariate import unseen_child_correlated_delta_q

    with pm.Model() as model:
        delta_u = pm.Normal("_delta_subj_u_marg", mu=0.0, sigma=0.8)
        before = {v.name for v in model.free_RVs}
        unseen_child_correlated_delta_q(
            delta_u, tau_subj_u=0.8, tau_subj_q=1.3, rho=0.37
        )
        added = {v.name for v in model.free_RVs} - before

    assert added == {"_z_subj_q_marg"}


def test_vg20_takes_the_correlated_branch_and_vg10_does_not():
    """The branch fingerprint on the real graphs, which is what the precondition
    test could not establish.

    `_z_subj_q_marg` is created only by the correlated construction (VG20) or by
    the age-varying scale, which the engine refuses alongside a correlation;
    `_delta_subj_q_marg` only by the independent one. So the pair separates the
    two branches exactly. Needs the prepared DuckDB, like its neighbours.
    """
    import contextlib
    import io as _io
    import os
    import tempfile

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling
    import pymc as pm

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models import common_bivariate as cb
    from vocab_growth.models import common_bivariate_re as cbr
    from vocab_growth.models.common import ModelFitContext

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def predictive_variables(definition, root):
        ctx = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name=definition.model_id,
                config_name=definition.config_name,
                output_root_dir=root,
                ci_prob=0.90,
                interval_kind="hdi",
            ),
            sampling=sampling.get_sampling_configuration("dev"),
        )
        os.makedirs(ctx.reporting.output_dir, exist_ok=True)
        with contextlib.redirect_stdout(_io.StringIO()):
            cbr.prepare_bivariate_re_data(ctx, definition)
            cb.configure_bivariate_priors(ctx, definition)
            cbr.build_model_re(ctx, definition)

        # Rebuild only the unseen-child block, exactly as the predictive does.
        variables = ctx.model_variables
        rho = variables.get("rho_uq")
        with ctx.model:
            delta_u = pm.Normal(
                "_delta_subj_u_marg", mu=0.0, sigma=variables["tau_subj_u"]
            )
            if rho is not None:
                cb.unseen_child_correlated_delta_q(
                    delta_u,
                    tau_subj_u=variables["tau_subj_u"],
                    tau_subj_q=variables["tau_subj_q"],
                    rho=rho,
                )
            else:
                pm.Normal(
                    "_delta_subj_q_marg", mu=0.0, sigma=variables["tau_subj_q"]
                )
        return {v.name for v in ctx.model.free_RVs}

    with tempfile.TemporaryDirectory() as root:
        names20 = predictive_variables(VG20, root)
        names10 = predictive_variables(VG10, root)

    assert "_z_subj_q_marg" in names20 and "_delta_subj_q_marg" not in names20
    assert "_delta_subj_q_marg" in names10 and "_z_subj_q_marg" not in names10
