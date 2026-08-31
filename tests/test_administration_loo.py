# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leave-one-administration-out, as the reports have always claimed.

Issue #266 finding 4. The multi-outcome engines computed a PSIS-LOO per
*outcome* while the reports described the predictive unit as a complete
administration, and the two differ in a way that flatters the model: the spoken
likelihood's trial count is the same row's observed comprehension, so holding
out a spoken term scores prediction conditional on that comprehension, and
holding out a comprehension term leaves its own value in the spoken term's
denominator. A paired administration also became two held-out cases with two
importance weights.

The combined score sums every factor belonging to one row of the frame. The
arithmetic is small; what it depends on is the ``obs_*_mask`` constant data
mapping each factor's likelihood rows back to administration rows, which is why
finding 3's mask defect had to be fixed first -- a mask marking recorded rows
rather than likelihood rows would sum the wrong factors onto the wrong
administrations, silently and plausibly.

These tests are on synthetic traces, so the arithmetic is checked against values
computed by hand rather than against a fit nobody can reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from vocab_growth.administration_loo import (
    ADMINISTRATION_DIM,
    ADMINISTRATION_VAR,
    LikelihoodFactor,
    administration_log_likelihood,
    attach_administration_log_likelihood,
)

_BIVARIATE = (
    LikelihoodFactor("y_u_obs", "obs_u_mask"),
    LikelihoodFactor("y_s_obs", "obs_s_mask"),
)


class _Trace:
    """The two groups the combination reads, and nothing else."""

    def __init__(self, log_likelihood=None, constant_data=None):
        self.log_likelihood = log_likelihood
        self.constant_data = constant_data


def _factor(values, dim):
    """A pointwise log-likelihood shaped (chain, draw, row[, cell])."""
    array = np.asarray(values, dtype=float)
    dims = ("chain", "draw", dim) + tuple(
        f"{dim}_cell{i}" for i in range(array.ndim - 3)
    )
    return xr.DataArray(
        array,
        dims=dims,
        coords={"chain": np.arange(array.shape[0]), "draw": np.arange(array.shape[1])},
    )


def _trace(masks: dict[str, np.ndarray], factors: dict[str, xr.DataArray]) -> _Trace:
    return _Trace(
        log_likelihood=xr.Dataset(factors),
        constant_data=xr.Dataset(
            {name: ("obs_id", np.asarray(mask)) for name, mask in masks.items()}
        ),
    )


# --- the arithmetic -------------------------------------------------------------


def test_a_paired_administration_becomes_one_case_not_two():
    """The defect, stated as the property that was violated."""
    # Four administrations: rows 0 and 2 paired, row 1 understood-only, row 3
    # spoken-only.
    u_mask = np.array([True, True, True, False])
    s_mask = np.array([True, False, True, True])
    # One chain, one draw, values chosen so every sum is distinguishable.
    u = _factor([[[-1.0, -2.0, -4.0]]], "obs_u_id")   # rows 0, 1, 2
    s = _factor([[[-0.5, -8.0, -16.0]]], "obs_s_id")  # rows 0, 2, 3

    combined = administration_log_likelihood(
        _trace({"obs_u_mask": u_mask, "obs_s_mask": s_mask},
               {"y_u_obs": u, "y_s_obs": s}),
        _BIVARIATE,
    )
    assert combined.dims == ("chain", "draw", ADMINISTRATION_DIM)
    # One entry per administration, not one per likelihood term.
    assert combined.sizes[ADMINISTRATION_DIM] == 4
    assert u.sizes["obs_u_id"] + s.sizes["obs_s_id"] == 6
    np.testing.assert_allclose(
        combined.values[0, 0],
        [
            -1.0 + -0.5,   # paired: both factors summed
            -2.0,          # understood only
            -4.0 + -8.0,   # paired
            -16.0,         # spoken only
        ],
    )


def test_the_kept_rows_are_labelled_by_their_frame_position():
    """A reader must be able to map a held-out case back to its administration."""
    u_mask = np.array([False, True, False, True, False])
    s_mask = np.array([False, False, True, True, False])
    combined = administration_log_likelihood(
        _trace(
            {"obs_u_mask": u_mask, "obs_s_mask": s_mask},
            {
                "y_u_obs": _factor([[[-1.0, -2.0]]], "obs_u_id"),
                "y_s_obs": _factor([[[-3.0, -4.0]]], "obs_s_id"),
            },
        ),
        _BIVARIATE,
    )
    # Row 0 and row 4 carry no factor at all and are absent, not zero.
    np.testing.assert_array_equal(combined[ADMINISTRATION_DIM].values, [1, 2, 3])
    np.testing.assert_allclose(combined.values[0, 0], [-1.0, -3.0, -2.0 + -4.0])


def test_a_composition_factor_contributes_its_row_not_its_cells():
    """VG15's cross-tabulation is stored per row AND per cell.

    The held-out unit is the administration, so a row's cells sum into one
    contribution. Treating each cell as a case would multiply that row's weight
    by four.
    """
    u_mask = np.array([True, True])
    cells_mask = np.array([True, False])
    cells = _factor([[[[-1.0, -2.0, -3.0, -4.0]]]], "obs_cells_id")
    combined = administration_log_likelihood(
        _trace(
            {"obs_u_mask": u_mask, "obs_cells_mask": cells_mask},
            {"y_u_obs": _factor([[[-10.0, -20.0]]], "obs_u_id"), "cells_obs": cells},
        ),
        (
            LikelihoodFactor("y_u_obs", "obs_u_mask"),
            LikelihoodFactor("cells_obs", "obs_cells_mask"),
        ),
    )
    np.testing.assert_allclose(
        combined.values[0, 0], [-10.0 + (-1.0 - 2.0 - 3.0 - 4.0), -20.0]
    )


def test_every_chain_and_draw_is_combined_independently():
    u_mask = np.array([True])
    s_mask = np.array([True])
    u = _factor([[[-1.0], [-2.0]], [[-3.0], [-4.0]]], "obs_u_id")
    s = _factor([[[-0.1], [-0.2]], [[-0.3], [-0.4]]], "obs_s_id")
    combined = administration_log_likelihood(
        _trace({"obs_u_mask": u_mask, "obs_s_mask": s_mask},
               {"y_u_obs": u, "y_s_obs": s}),
        _BIVARIATE,
    )
    np.testing.assert_allclose(
        combined.values[..., 0], [[-1.1, -2.2], [-3.3, -4.4]]
    )


# --- what it refuses ------------------------------------------------------------


def test_a_mask_that_does_not_match_its_factor_raises():
    """Finding 3's defect, in the place it would do the most damage.

    A mask marking recorded rows rather than likelihood rows would map factors
    onto the wrong administrations. That must fail loudly, not produce a
    plausible number.
    """
    u_mask = np.array([True, True, True])
    s_mask = np.array([True, True, True])  # claims 3 rows
    with pytest.raises(ValueError, match="finding 3"):
        administration_log_likelihood(
            _trace(
                {"obs_u_mask": u_mask, "obs_s_mask": s_mask},
                {
                    "y_u_obs": _factor([[[-1.0, -2.0, -3.0]]], "obs_u_id"),
                    "y_s_obs": _factor([[[-1.0, -2.0]]], "obs_s_id"),  # stores 2
                },
            ),
            _BIVARIATE,
        )


@pytest.mark.parametrize("missing", ["y_s_obs", "obs_s_mask"])
def test_a_missing_factor_or_mask_yields_nothing_rather_than_a_partial_score(missing):
    """A partial sum would be a different estimand wearing the same label."""
    masks = {"obs_u_mask": np.array([True]), "obs_s_mask": np.array([True])}
    factors = {
        "y_u_obs": _factor([[[-1.0]]], "obs_u_id"),
        "y_s_obs": _factor([[[-2.0]]], "obs_s_id"),
    }
    masks.pop(missing, None)
    factors.pop(missing, None)
    assert administration_log_likelihood(_trace(masks, factors), _BIVARIATE) is None


def test_a_trace_without_the_groups_yields_nothing():
    assert administration_log_likelihood(_Trace(), _BIVARIATE) is None


def test_no_observed_rows_yields_nothing():
    combined = administration_log_likelihood(
        _trace(
            {"obs_u_mask": np.array([False, False]), "obs_s_mask": np.array([False, False])},
            {
                "y_u_obs": _factor(np.zeros((1, 1, 0)), "obs_u_id"),
                "y_s_obs": _factor(np.zeros((1, 1, 0)), "obs_s_id"),
            },
        ),
        _BIVARIATE,
    )
    assert combined is None


# --- attaching it ---------------------------------------------------------------


def test_attaching_adds_the_variable_and_is_idempotent():
    trace = _trace(
        {"obs_u_mask": np.array([True, True]), "obs_s_mask": np.array([True, False])},
        {
            "y_u_obs": _factor([[[-1.0, -2.0]]], "obs_u_id"),
            "y_s_obs": _factor([[[-3.0]]], "obs_s_id"),
        },
    )
    assert attach_administration_log_likelihood(trace, _BIVARIATE)
    assert ADMINISTRATION_VAR in trace.log_likelihood.data_vars
    stored = trace.log_likelihood[ADMINISTRATION_VAR].values.copy()

    # Re-running diagnostics must not recompute or double it.
    assert attach_administration_log_likelihood(trace, _BIVARIATE)
    np.testing.assert_array_equal(
        trace.log_likelihood[ADMINISTRATION_VAR].values, stored
    )
    np.testing.assert_allclose(stored[0, 0], [-1.0 + -3.0, -2.0])


def test_attaching_reports_failure_rather_than_raising():
    """A trace written before this existed must not break a re-run."""
    assert not attach_administration_log_likelihood(_Trace(), _BIVARIATE)


# --- what the engines declare ---------------------------------------------------


def test_every_multi_outcome_engine_declares_its_factors():
    """A factor left out is a term silently excluded from the score.

    VG15's two composition terms are the case that matters: they identify
    `psi`, and the per-outcome scores omit them entirely.
    """
    import inspect

    from vocab_growth.models import (
        common_bivariate,
        common_joint_modality,
        common_trivariate,
    )

    expected = {
        common_bivariate: {"y_u_obs", "y_s_obs"},
        common_trivariate: {"y_u_obs", "y_s_obs", "y_sign_obs"},
        common_joint_modality: {
            "y_u_obs",
            "y_s_obs",
            "y_sign_obs",
            "cells_obs",
            "nz_prod_cells_obs",
        },
    }
    for module, names in expected.items():
        source = inspect.getsource(module)
        assert "administration_factors=" in source, module.__name__
        for name in names:
            assert f'LikelihoodFactor("{name}"' in source, (
                f"{module.__name__} does not declare {name} as an "
                "administration likelihood factor"
            )


def test_the_single_outcome_engine_needs_no_combination():
    """One likelihood term over administration rows already IS the unit."""
    import inspect

    from vocab_growth.models import common

    source = inspect.getsource(common.fit_single_outcome_model)
    assert "administration_factors" not in source


def test_the_total_log_likelihood_is_conserved():
    """The sharpest invariant: regrouping terms must not change their sum.

    Verified on a real 2-chain VG10 fit while this was written -- 96 per-outcome
    terms collapsing to 48 administration cases with a maximum absolute
    difference of 0.0 -- and pinned here on synthetic values so it runs
    everywhere. A combination that dropped, duplicated or misplaced a factor
    would break it.
    """
    rng = np.random.default_rng(3)
    n = 12
    u_mask = rng.random(n) < 0.8
    s_mask = rng.random(n) < 0.6
    u_mask[0] = s_mask[0] = True  # at least one paired row
    u = _factor(rng.normal(size=(2, 5, int(u_mask.sum()))), "obs_u_id")
    s = _factor(rng.normal(size=(2, 5, int(s_mask.sum()))), "obs_s_id")

    combined = administration_log_likelihood(
        _trace({"obs_u_mask": u_mask, "obs_s_mask": s_mask},
               {"y_u_obs": u, "y_s_obs": s}),
        _BIVARIATE,
    )
    per_term = u.sum(dim="obs_u_id") + s.sum(dim="obs_s_id")
    per_case = combined.sum(dim=ADMINISTRATION_DIM)
    np.testing.assert_allclose(per_case.values, per_term.values, rtol=0, atol=1e-12)
