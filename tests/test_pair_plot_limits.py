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
    from vocab_growth.models.common_bivariate import pair_plot_priority
    from vocab_growth.models.definitions import MODEL_REGISTRY

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
    from vocab_growth.models.common_bivariate import pair_plot_priority
    from vocab_growth.models.definitions import MODEL_REGISTRY

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
    from vocab_growth.models.common_bivariate import pair_plot_priority
    from vocab_growth.models.definitions import MODEL_REGISTRY

    build_order = ["eta_u", "tau_u", "tau_subj_u", "tau_subj_q", "rho_uq"]
    ordered = _apply(pair_plot_priority(MODEL_REGISTRY["vg20"]), build_order)

    assert sorted(ordered) == sorted(build_order)
    assert len(ordered) == len(set(ordered))


def test_models_without_a_child_structure_keep_model_order_exactly():
    """VG05, VG07-VG10 and every univariate model must render as before."""
    from vocab_growth.models.common_bivariate import pair_plot_priority
    from vocab_growth.models.definitions import MODEL_REGISTRY

    for key in ("vg05", "vg07", "vg08", "vg09", "vg10"):
        assert pair_plot_priority(MODEL_REGISTRY[key]) == ()


def test_the_cross_lag_priority_is_unchanged():
    """VG16 already had this treatment; generalising it must not move VG16."""
    from vocab_growth.models.common_bivariate import pair_plot_priority
    from vocab_growth.models.definitions import MODEL_REGISTRY

    assert pair_plot_priority(MODEL_REGISTRY["vg16"]) == (
        "beta_lag", "tau_subj_u", "tau_subj_q", "tau_u", "tau_q",
    )


def test_the_factor_priority_omits_the_correlation_matrix():
    """`subject_factor_corr` is 4x4 — 16 plot items would eat the whole grid."""
    from vocab_growth.models.common_bivariate import pair_plot_priority
    from vocab_growth.models.definitions import MODEL_REGISTRY

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
