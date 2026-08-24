# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PSIS-LOO's relative efficiency is pinned to the sampled parameters.

ArviZ's default ``reff`` averages ESS over every variable in the posterior group,
so it moved with the storage policy (0.561 over everything VG10 stored at
``dev``, 0.554 without the observation-sized variables, 0.905 over the sampled
parameters alone). ``vocab_growth.loo_reff`` computes the same quantity over the
model's free random variables only — named by the caller, or read from the
record the ``sample`` stage writes into the trace — and falls back to ArviZ's
convention, audibly, only when neither is available.
"""

import json

import arviz_stats  # noqa: F401  (registers the ``azstats`` accessor)
import numpy as np
import pytest
import xarray as xr

from vocab_growth.fit_artifacts import SAMPLED_PARAMETERS_ATTR
from vocab_growth.loo_reff import (
    reff_or_default,
    sampled_parameter_names,
    sampled_parameter_reff,
)


def _trace(n_chains=4, n_draws=60, *, attr=None, seed=3):
    rng = np.random.default_rng(seed)
    # Mildly autocorrelated chains so ESS is informative, not trivially n.
    def ar1(shape):
        x = rng.normal(size=shape)
        for t in range(1, shape[1]):
            x[:, t] = 0.5 * x[:, t - 1] + x[:, t]
        return x

    posterior = xr.Dataset(
        {
            "mu": (("chain", "draw"), ar1((n_chains, n_draws))),
            "delta_raw": (("chain", "draw", "study"), ar1((n_chains, n_draws, 3)).reshape(n_chains, n_draws, 3)),
            "f_plot": (("chain", "draw", "plot_id"), np.repeat(ar1((n_chains, n_draws))[:, :, None], 50, axis=2)),
        },
        coords={"chain": range(n_chains), "draw": range(n_draws), "study": list("abc"), "plot_id": range(50)},
    )
    if attr is not None:
        posterior.attrs[SAMPLED_PARAMETERS_ATTR] = json.dumps(attr)
    return xr.DataTree.from_dict({"posterior": posterior})


def _arviz_style_reff(posterior, names):
    ess = posterior[names].azstats.ess(method="mean")
    values = np.hstack([ess[v].values.ravel() for v in ess.data_vars])
    return values.mean() / (posterior.sizes["chain"] * posterior.sizes["draw"])


def test_reff_is_the_arviz_formula_restricted_to_the_named_parameters():
    trace = _trace()
    post = trace["posterior"].to_dataset()
    got = sampled_parameter_reff(trace, names=["mu", "delta_raw"])
    assert got == pytest.approx(_arviz_style_reff(post, ["mu", "delta_raw"]))
    # And it is not the posterior-wide number: f_plot's 50 identical copies of a
    # single chain would dominate ArviZ's average.
    assert got != pytest.approx(_arviz_style_reff(post, ["mu", "delta_raw", "f_plot"]))


def test_reff_reads_the_sampled_parameters_record_when_names_are_not_given():
    trace = _trace(attr=["mu", "delta_raw"])
    assert sampled_parameter_names(trace) == ["mu", "delta_raw"]
    assert sampled_parameter_reff(trace) == pytest.approx(
        sampled_parameter_reff(trace, names=["mu", "delta_raw"])
    )


def test_single_chain_is_one_as_in_arviz():
    assert sampled_parameter_reff(_trace(n_chains=1), names=["mu"]) == 1.0


def test_unpinnable_trace_raises_and_the_default_path_says_so():
    trace = _trace()  # no record, no names
    with pytest.raises(LookupError, match="sampled parameters"):
        sampled_parameter_reff(trace)
    notices = []
    assert reff_or_default(trace, label="VG99", warn=notices.append) is None
    assert notices and "VG99" in notices[0] and "default" in notices[0]
    # With names it pins, and says nothing.
    notices.clear()
    assert reff_or_default(trace, names=["mu"], warn=notices.append) is not None
    assert not notices


def test_names_missing_from_the_posterior_are_an_error():
    with pytest.raises(KeyError, match="absent"):
        sampled_parameter_reff(_trace(), names=["mu", "not_there"])
