# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import arviz as az
import numpy as np
import xarray as xr

from vocab_growth.models.diagnostics_utils import (
    capped_plot_var_names,
    plot_required_subplots,
    plot_variable_count,
)


def test_plot_required_subplots_counts_non_sample_dimensions():
    trace = _trace_with_scalar_and_vector_parameters()

    assert plot_required_subplots(trace, ["alpha", "beta"], squared=True) == 9


def test_capped_plot_var_names_limits_pair_plot_grid():
    trace = _trace_with_scalar_and_vector_parameters()

    with az.rc_context({"plot.max_subplots": 8}):
        var_names = capped_plot_var_names(
            trace,
            ["alpha", "beta"],
            squared=True,
        )

    assert var_names == ["alpha"]
    assert plot_required_subplots(trace, var_names, squared=True) <= 8


def test_capped_plot_var_names_keeps_pair_plot_vars_when_limit_is_sufficient():
    trace = _trace_with_scalar_and_vector_parameters()

    with az.rc_context({"plot.max_subplots": 9}):
        assert capped_plot_var_names(
            trace,
            ["alpha", "beta"],
            squared=True,
        ) == ["alpha", "beta"]


def test_capped_plot_var_names_skips_large_observed_diagnostic():
    trace = _trace_with_large_observed_diagnostic()

    with az.rc_context({"plot.max_subplots": 40}):
        var_names = capped_plot_var_names(trace, ["alpha", "kappa_obs"])

    assert var_names == ["alpha"]
    assert plot_variable_count(trace, "kappa_obs") == 100


def _trace_with_scalar_and_vector_parameters():
    posterior = xr.Dataset(
        data_vars={
            "alpha": (("chain", "draw"), np.ones((1, 2))),
            "beta": (("chain", "draw", "coef"), np.ones((1, 2, 2))),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "coef": ["intercept", "slope"],
        },
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


def _trace_with_large_observed_diagnostic():
    posterior = xr.Dataset(
        data_vars={
            "alpha": (("chain", "draw"), np.ones((1, 2))),
            "kappa_obs": (("chain", "draw", "obs"), np.ones((1, 2, 100))),
        },
        coords={
            "chain": [0],
            "draw": [0, 1],
            "obs": range(100),
        },
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})


# --------------------------------------------------------------------------
# Model-specific pair plots (#233)
# --------------------------------------------------------------------------


def test_the_parameter_a_model_was_added_for_survives_the_cap():
    """VG20's `rho_uq` fell off the end of a grid built in model order.

    The cap keeps `floor(sqrt(max_subplots))` variables from the front of the
    list, and model order is build order — mean function, then GP, then the
    scales. So the pair plot omitted the one parameter VG20's own caption sends
    the reader there to inspect.
    """
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    build_order = [
        "p_slope_low_u", "p_slope_hi_u", "eta_u", "ell_unit_u",
        "p_slope_low_q", "p_slope_hi_q", "eta_q", "ell_unit_q",
        "tau_u", "tau_q", "tau_subj_u", "tau_subj_q", "rho_uq",
    ]
    trace = _trace_with_named_scalars(build_order)

    with az.rc_context({"plot.max_subplots": 36}):  # floor(sqrt(36)) = 6 slots
        assert "rho_uq" not in capped_plot_var_names(trace, build_order, squared=True)

        priority = pair_plot_priority(MODEL_REGISTRY["vg20"])
        ordered = _apply(priority, build_order)
        assert "rho_uq" in capped_plot_var_names(trace, ordered, squared=True)


def test_the_child_slope_block_survives_the_cap():
    """VG19's rates and their offset-rate correlations, same defect."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    build_order = [
        "p_slope_low_u", "p_slope_hi_u", "eta_u", "ell_unit_u",
        "p_slope_low_q", "p_slope_hi_q", "eta_q", "ell_unit_q",
        "tau_u", "tau_q",
        "tau_subj_u_0", "tau_subj_u_1", "tau_subj_u_rho",
        "tau_subj_q_0", "tau_subj_q_1", "tau_subj_q_rho",
    ]
    trace = _trace_with_named_scalars(build_order)

    priority = pair_plot_priority(MODEL_REGISTRY["vg19"])
    with az.rc_context({"plot.max_subplots": 36}):
        kept = capped_plot_var_names(trace, _apply(priority, build_order), squared=True)

    assert {"tau_subj_u_1", "tau_subj_q_1"} <= set(kept)
    assert "tau_subj_u_rho" in kept


def test_ordering_never_drops_a_variable():
    """This reorders; it must not filter, or a reader loses a marginal."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    build_order = ["eta_u", "tau_u", "tau_subj_u", "tau_subj_q", "rho_uq"]
    ordered = _apply(pair_plot_priority(MODEL_REGISTRY["vg20"]), build_order)

    assert sorted(ordered) == sorted(build_order)
    assert len(ordered) == len(set(ordered))


def test_the_sign_lag_coefficient_survives_the_cap_on_the_joint_engine():
    """VG25's `beta_sign_lag`, the same defect on the other engine (#339 review).

    The joint engine led with `psi` and `conc` and then took model order, which
    on VG25 puts the coefficient eighteenth -- so the six-slot grid showed the
    two headline associations and four understood-GP hyperparameters, while the
    report's diagnostics callout sends the reader there to inspect the
    coefficient against the signing child block. The build order below is VG25's
    own, read off the built graph.
    """
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    build_order = [
        "p_slope_low_u", "p_slope_hi_u", "ell_unit_u", "eta_u",
        "p_slope_low_q", "p_slope_hi_q", "ell_unit_q", "eta_q",
        "p_slope_low_sign", "p_slope_mid_sign", "p_slope_hi_sign",
        "peak_unit_sign", "ell_unit_sign", "eta_sign",
        "tau_u", "tau_q", "tau_sign",
        "beta_sign_lag",
        "tau_subj_u", "tau_subj_q", "tau_subj_sign",
        "rho_uq", "rho_u_sign", "rho_sign_q", "psi", "conc",
    ]
    trace = _trace_with_named_scalars(build_order)

    with az.rc_context({"plot.max_subplots": 36}):  # floor(sqrt(36)) = 6 slots
        # What the engine did before: psi and conc, then build order.
        was = capped_plot_var_names(trace, _apply(("psi", "conc"), build_order), squared=True)
        assert "beta_sign_lag" not in was

        priority = pair_plot_priority(MODEL_REGISTRY["vg25"])
        kept = capped_plot_var_names(trace, _apply(priority, build_order), squared=True)

    # The geometry the report's callout actually names.
    assert {"beta_sign_lag", "rho_sign_q", "tau_subj_sign"} <= set(kept)
    assert kept[:2] == ["psi", "conc"]


def test_the_plain_joint_model_keeps_the_order_it_had():
    """VG15 has no distinguishing child structure, so its plot must not move.

    The consolidation's trap: on the joint engine `psi` and `conc` lead
    unconditionally, so a priority list built by appending to them is never
    empty, and VG15 would have picked up the scale block that is supposed to
    mark a model with a structure worth prioritising. The head is kept separate
    from the definition-driven part for exactly this case.
    """
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    assert pair_plot_priority(MODEL_REGISTRY["vg15"]) == ("psi", "conc")


def test_vg24s_correlations_survive_the_cap():
    """The same #233 defect, live in a fitted model of record.

    VG24's report sends the reader to the pair plot for the child block's
    correlations — "a correlation near ±1 ... is the failure mode to watch for
    in the energy and pair plots" — and the grid showed `psi`, `conc` and four
    understood-GP hyperparameters, because build order puts the mean functions
    first and the cap keeps six.

    `rho_sign_q` leads the three: the naive consolidation would have applied the
    bivariate rule and prioritised `rho_uq`, which is the understood–spoken
    correlation and not what VG24 was registered to estimate.
    """
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    build_order = [
        "p_slope_low_u", "p_slope_hi_u", "ell_unit_u", "eta_u",
        "p_slope_low_q", "p_slope_hi_q", "ell_unit_q", "eta_q",
        "tau_u", "tau_q", "tau_sign",
        "tau_subj_u", "tau_subj_q", "tau_subj_sign",
        "rho_uq", "rho_u_sign", "rho_sign_q", "psi", "conc",
    ]
    trace = _trace_with_named_scalars(build_order)

    with az.rc_context({"plot.max_subplots": 36}):  # floor(sqrt(36)) = 6 slots
        was = capped_plot_var_names(
            trace, _apply(("psi", "conc"), build_order), squared=True
        )
        kept = capped_plot_var_names(
            trace,
            _apply(pair_plot_priority(MODEL_REGISTRY["vg24"]), build_order),
            squared=True,
        )

    assert not {"rho_sign_q", "rho_u_sign", "rho_uq"} & set(was)
    assert {"rho_sign_q", "rho_u_sign", "rho_uq"} <= set(kept)
    assert kept.index("rho_sign_q") < kept.index("rho_uq")


def test_engines_that_do_not_order_have_nothing_to_order():
    """The consolidation's fail-closed half.

    Three engines install no reordering — univariate, univariate with random
    effects, and trivariate. That is only safe while their models have an empty
    priority; otherwise the function would declare an intent the engine silently
    drops. Registering a model with a distinguishing structure on one of them
    fails here, naming the engine that needs wiring.
    """
    from vocab_growth.models.catalogue import get as catalogue_get
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    unordered = {"univariate", "univariate_re", "trivariate"}
    for key, definition in MODEL_REGISTRY.items():
        engine = catalogue_get(key).engine.name
        if engine in unordered:
            assert pair_plot_priority(definition) == (), (
                f"{key} runs on the {engine} engine, which installs no pair-plot "
                f"reordering, but declares a priority — wire that engine through "
                f"pair_plot_var_names_fn or the intent is silently dropped."
            )


def test_the_joint_ordering_never_drops_a_variable():
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    build_order = [
        "eta_u", "tau_u", "beta_sign_lag", "tau_subj_sign", "rho_sign_q",
        "psi", "conc",
    ]
    ordered = _apply(pair_plot_priority(MODEL_REGISTRY["vg25"]), build_order)

    assert sorted(ordered) == sorted(build_order)
    assert len(ordered) == len(set(ordered))


def test_models_without_a_child_structure_keep_model_order_exactly():
    """VG05, VG07-VG10 and every univariate model must render as before."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    for key in ("vg05", "vg07", "vg08", "vg09", "vg10"):
        assert pair_plot_priority(MODEL_REGISTRY[key]) == ()


def test_the_cross_lag_priority_is_unchanged():
    """VG16 already had this treatment; generalising it must not move VG16."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    assert pair_plot_priority(MODEL_REGISTRY["vg16"]) == (
        "beta_lag", "tau_subj_u", "tau_subj_q", "tau_u", "tau_q",
    )


def test_the_factor_priority_omits_the_correlation_matrix():
    """`subject_factor_corr` is 4x4 — 16 plot items would eat the whole grid."""
    from vocab_growth.models.definitions import MODEL_REGISTRY
    from vocab_growth.models.diagnostics_utils import pair_plot_priority

    priority = pair_plot_priority(MODEL_REGISTRY["vg22"])
    assert "rho_uq" in priority
    assert "subject_factor_corr" not in priority


def _apply(priority, names):
    """The reordering `_shared_diagnostics` installs, in test form."""
    seen, ordered = set(), []
    for name in (*priority, *names):
        if name in names and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _trace_with_named_scalars(names):
    """A posterior of scalar parameters, in the shape `capped_plot_var_names` reads."""
    posterior = xr.Dataset(
        data_vars={name: (("chain", "draw"), np.ones((1, 2))) for name in names},
        coords={"chain": [0], "draw": [0, 1]},
        attrs={"sample_dims": ["chain", "draw"]},
    )
    return xr.DataTree.from_dict({"posterior": posterior})
