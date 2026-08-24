# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The spoken-fallback treatments (issues #233 and #236).

455 of the current frame's 1,428 spoken observations cannot condition on an
observed understood count, and have always been given ``S ~ BB(810, p_U*q,
kappa_S)`` instead of the paired model's second line. These tests pin the four
treatments of that branch: that the default is unchanged, that dropping the rows
drops exactly those rows, that the separate-dispersion form nests the default,
and -- the substantive one -- that the moment-matched concentration really is
the true marginal's, checked against a brute-force sum over the latent parent
count rather than against a restatement of the formula.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.stats import betabinom

from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_MOMENT_MATCHED,
    SPOKEN_FALLBACK_PAIRED_ONLY,
    SPOKEN_FALLBACK_PRODUCT,
    SPOKEN_FALLBACK_SEPARATE_DISPERSION,
    SPOKEN_FALLBACK_TREATMENTS,
    nested_outcome_alpha_beta,
    nested_outcome_spec,
    product_marginal_concentration,
    resolve_fallback_treatment,
)


def _spec(**overrides):
    df = pd.DataFrame(
        {
            "understood": [100, np.nan, 40, np.nan, 810],
            "spoken": [25, 10, 50, 3, 810],
        }
    )
    return nested_outcome_spec(
        df, parent_col="understood", outcome_col="spoken", n_trials=810, **overrides
    )


# --------------------------------------------------------------------------
# The moment-matched concentration, against the true marginal
# --------------------------------------------------------------------------


def _exact_marginal_pmf(n_trials, p_u, kappa_u, q, kappa_s):
    """P(S = s) under the paired model, by summing over the latent parent count.

    ``U ~ BB(n, p_U, kappa_U)`` then ``S | U ~ BB(U, q, kappa_S)``, summed over
    every U that could have produced each S. Feasible only for a small
    inventory, which is exactly why the graph cannot do this.
    """
    u = np.arange(n_trials + 1)
    p_parent = betabinom.pmf(u, n_trials, p_u * kappa_u, (1 - p_u) * kappa_u)
    s = np.arange(n_trials + 1)
    # rows: latent U, columns: observed S.
    conditional = betabinom.pmf(
        s[None, :], u[:, None], q * kappa_s, (1 - q) * kappa_s
    )
    conditional = np.nan_to_num(conditional, nan=0.0)
    conditional[0, 0] = 1.0  # U = 0 forces S = 0; BB(0, .) is degenerate.
    return p_parent @ conditional


def _mean_and_variance(pmf):
    s = np.arange(pmf.size)
    mean = float(s @ pmf)
    return mean, float((s**2) @ pmf) - mean**2


@pytest.mark.parametrize(
    ("p_u", "kappa_u", "q", "kappa_s"),
    [
        (0.6, 8.0, 0.5, 12.0),
        (0.3, 25.0, 0.2, 5.0),
        (0.85, 3.0, 0.7, 40.0),
    ],
)
def test_the_moment_matched_concentration_is_the_true_marginals(p_u, kappa_u, q, kappa_s):
    """The claim the treatment rests on, checked without reusing its algebra.

    The exact marginal is computed by summing the paired model over the latent
    parent count; the moment-matched Beta-Binomial must reproduce its mean and
    its variance, not just its mean.
    """
    n_trials = 40
    pmf = _exact_marginal_pmf(n_trials, p_u, kappa_u, q, kappa_s)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    exact_mean, exact_variance = _mean_and_variance(pmf)

    kappa_eff = float(
        product_marginal_concentration(
            np.array(p_u), np.array(kappa_u), np.array(q), np.array(kappa_s),
            epsilon=1e-9,
        ).eval()
    )
    m = p_u * q
    matched_mean = n_trials * m
    matched_variance = (
        n_trials * m * (1 - m) * (n_trials + kappa_eff) / (1 + kappa_eff)
    )

    assert matched_mean == pytest.approx(exact_mean, rel=1e-10)
    assert matched_variance == pytest.approx(exact_variance, rel=1e-10)


@pytest.mark.parametrize(
    ("p_u", "kappa_u", "q", "kappa_s"),
    [
        # q*kappa_S < kappa_U: the default over-disperses the branch.
        (0.6, 8.0, 0.5, 12.0),
        (0.3, 25.0, 0.2, 5.0),
        # q*kappa_S > kappa_U: it under-disperses it.
        (0.85, 3.0, 0.7, 40.0),
        (0.4, 5.0, 0.5, 30.0),
        # And exactly at the crossover the two agree, p_U notwithstanding.
        (0.4, 15.0, 0.5, 30.0),
        (0.9, 15.0, 0.5, 30.0),
    ],
)
def test_the_default_fallbacks_variance_error_flips_sign_at_q_kappa_s_equals_kappa_u(
    p_u, kappa_u, q, kappa_s
):
    """The defect, stated exactly rather than as a direction.

    Working ``Var(theta_U theta_S) = ab + a q^2 + b p_U^2`` against the default's
    ``p_U q (1 - p_U q)/(1 + kappa_S)`` cancels to the single condition
    ``q kappa_S > kappa_U``, with ``p_U`` dropping out. That is why the branch
    needed a sensitivity family rather than a one-line correction in a known
    direction: which way the current models err is a question about their fitted
    concentrations.
    """
    n_trials = 40
    _, exact_variance = _mean_and_variance(
        _exact_marginal_pmf(n_trials, p_u, kappa_u, q, kappa_s)
    )
    m = p_u * q
    default_variance = (
        n_trials * m * (1 - m) * (n_trials + kappa_s) / (1 + kappa_s)
    )

    if q * kappa_s > kappa_u:
        assert exact_variance > default_variance
    elif q * kappa_s < kappa_u:
        assert exact_variance < default_variance
    else:
        assert exact_variance == pytest.approx(default_variance, rel=1e-10)


def test_the_moment_match_reduces_to_the_default_in_the_limit_it_assumes():
    """At ``p_U = 1`` with a deterministic comprehension process the two agree.

    Which is the sense in which the default is not wrong but incomplete: it is
    the moment-matched form under an assumption the data contradict.
    """
    kappa_eff = float(
        product_marginal_concentration(
            np.array(1.0 - 1e-12), np.array(1e12), np.array(0.4), np.array(9.0),
            epsilon=1e-15,
        ).eval()
    )
    assert kappa_eff == pytest.approx(9.0, rel=1e-4)


# --------------------------------------------------------------------------
# Dropping the rows
# --------------------------------------------------------------------------


def test_conditional_only_drops_exactly_the_fallback_rows():
    spec = _spec()
    assert (spec.n_conditional, spec.n_marginal) == (2, 3)

    paired = spec.conditional_only()

    np.testing.assert_array_equal(paired.indices, spec.indices[spec.is_conditional])
    np.testing.assert_array_equal(paired.trials, [100, 810])
    assert paired.n_observed == 2
    assert paired.n_marginal == 0
    assert bool(np.all(paired.is_conditional))


def test_conditional_only_keeps_reporting_the_source_data_violations():
    """Dropping the rows must not drop the reason they exist.

    ``spoken > understood`` is a property of the source data, and a build report
    that stopped mentioning it under one variant would hide the finding rather
    than bound it.
    """
    df = pd.DataFrame({"understood": [100, 10], "spoken": [25, 50]})
    spec = nested_outcome_spec(
        df, parent_col="understood", outcome_col="spoken", n_trials=810
    )
    assert spec.n_parent_violations == 1
    assert spec.conditional_only().n_parent_violations == 1


# --------------------------------------------------------------------------
# Treatment resolution
# --------------------------------------------------------------------------


def test_resolve_defaults_for_a_definition_without_the_field():
    class _Old:
        pass

    assert resolve_fallback_treatment(_Old()) == SPOKEN_FALLBACK_PRODUCT


def test_resolve_rejects_an_unknown_treatment():
    class _Bad:
        spoken_fallback = "logsumexp"

    with pytest.raises(ValueError, match="Unknown spoken_fallback"):
        resolve_fallback_treatment(_Bad())


def test_every_registered_model_carries_a_known_treatment():
    from vocab_growth.models.definitions import (
        MODEL_REGISTRY,
        BivariateModelDefinition,
    )

    bivariate = [
        d for d in MODEL_REGISTRY.values() if isinstance(d, BivariateModelDefinition)
    ]
    assert len(bivariate) == 11
    for definition in bivariate:
        assert definition.spoken_fallback == SPOKEN_FALLBACK_PRODUCT, (
            f"{definition.model_id} is a model of record; the variants live in "
            "the sensitivity registry, not on the definitions"
        )


# --------------------------------------------------------------------------
# The graph the treatments emit
# --------------------------------------------------------------------------


def _alpha_beta(treatment, **overrides):
    import pymc as pm

    parameters = {
        "is_conditional": np.array([1, 0, 1, 0]),
        "conditional_p": np.array([0.4, 0.5, 0.6, 0.7]),
        "marginal_p": np.array([0.2, 0.25, 0.3, 0.35]),
        "parent_p": np.array([0.5, 0.5, 0.5, 0.5]),
        "parent_kappa": np.array([9.0, 9.0, 9.0, 9.0]),
        "kappa": np.array([12.0, 12.0, 12.0, 12.0]),
    }
    parameters.update(overrides)
    with pm.Model() as model:
        alpha, beta = nested_outcome_alpha_beta(
            treatment=treatment,
            epsilon=1e-9,
            outcome="s",
            fallback_kappa_sigma=0.5,
            **parameters,
        )
        evaluated = (np.asarray(alpha.eval()), np.asarray(beta.eval()))
    return evaluated, model


def test_the_default_treatment_emits_the_historical_expression():
    """The eleven models' graphs must be unchanged by the refactor.

    Hand-built here rather than compared to a pinned number, so the check keeps
    meaning if the data change.
    """
    (alpha, beta), _ = _alpha_beta(SPOKEN_FALLBACK_PRODUCT)
    p = np.where([1, 0, 1, 0], [0.4, 0.5, 0.6, 0.7], [0.2, 0.25, 0.3, 0.35])
    np.testing.assert_allclose(alpha, p * 12.0, rtol=0, atol=0)
    np.testing.assert_allclose(beta, (1 - p) * 12.0, rtol=0, atol=0)


def test_the_separate_dispersion_treatment_nests_the_default_at_zero():
    """Its whole value as a readout depends on this.

    The offset is replaced by a constant zero rather than left to whatever test
    value the RV happens to carry, so this is arithmetic and not sampling.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.graph.replace import graph_replace

    (default_alpha, default_beta), _ = _alpha_beta(SPOKEN_FALLBACK_PRODUCT)

    parameters = {
        "is_conditional": np.array([1, 0, 1, 0]),
        "conditional_p": np.array([0.4, 0.5, 0.6, 0.7]),
        "marginal_p": np.array([0.2, 0.25, 0.3, 0.35]),
        "parent_p": np.array([0.5, 0.5, 0.5, 0.5]),
        "parent_kappa": np.array([9.0, 9.0, 9.0, 9.0]),
        "kappa": np.array([12.0, 12.0, 12.0, 12.0]),
    }
    with pm.Model() as model:
        alpha, beta = nested_outcome_alpha_beta(
            treatment=SPOKEN_FALLBACK_SEPARATE_DISPERSION,
            epsilon=1e-9,
            outcome="s",
            fallback_kappa_sigma=0.5,
            **parameters,
        )
    offset = model["log_kappa_s_fallback"]
    at_zero = {offset: pt.constant(0.0, dtype=offset.dtype)}
    np.testing.assert_allclose(
        graph_replace(alpha, at_zero).eval(), default_alpha, rtol=1e-12
    )
    np.testing.assert_allclose(
        graph_replace(beta, at_zero).eval(), default_beta, rtol=1e-12
    )
    assert "log_kappa_s_fallback" in {rv.name for rv in model.free_RVs}


def test_the_moment_matched_treatment_touches_only_the_fallback_rows():
    (default_alpha, default_beta), _ = _alpha_beta(SPOKEN_FALLBACK_PRODUCT)
    (alpha, beta), _ = _alpha_beta(SPOKEN_FALLBACK_MOMENT_MATCHED)

    conditional = np.array([True, False, True, False])
    np.testing.assert_allclose(alpha[conditional], default_alpha[conditional])
    np.testing.assert_allclose(beta[conditional], default_beta[conditional])

    # And on the fallback rows it moves the concentration, in the direction the
    # crossover condition gives: here q = 0.5, kappa_S = 12 and kappa_U = 9, so
    # q*kappa_S = 6 < 9 and the default is the *over*-dispersed one.
    kappa_default = default_alpha[~conditional] + default_beta[~conditional]
    kappa_matched = alpha[~conditional] + beta[~conditional]
    assert np.all(kappa_matched > kappa_default)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_the_registered_variants_cover_the_three_alternatives():
    from vocab_growth.sensitivity.registry import build_variant

    expected = {
        "paired-only": SPOKEN_FALLBACK_PAIRED_ONLY,
        "fallback-dispersion": SPOKEN_FALLBACK_SEPARATE_DISPERSION,
        "marginal-moments": SPOKEN_FALLBACK_MOMENT_MATCHED,
    }
    for model_key in ("vg10", "vg20"):
        for name, treatment in expected.items():
            definition = build_variant(model_key, name)[0]
            assert definition.spoken_fallback == treatment
            assert definition.config_name.endswith(name)
    assert set(expected.values()) | {SPOKEN_FALLBACK_PRODUCT} == set(
        SPOKEN_FALLBACK_TREATMENTS
    )


def test_the_field_is_part_of_the_recorded_definition():
    """It is a graph field, so a fit made under one value must not validate
    against another. The manifest comparison is what enforces that."""
    from vocab_growth.fit_artifacts import normalise_for_json
    from vocab_growth.models.definitions import MODEL_REGISTRY

    base = MODEL_REGISTRY["vg20"]
    changed = dataclasses.replace(
        base, spoken_fallback=SPOKEN_FALLBACK_PAIRED_ONLY
    )
    assert normalise_for_json(base) != normalise_for_json(changed)
