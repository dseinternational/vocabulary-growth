# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The marginalised engine survives an end-to-end sampler run.

Split out of ``test_subject_marginal.py``, which holds every other claim about
the singleton marginalisation. This one test is by some way the longest in the
suite -- eight draws, but the cost is nutpie compiling the quadrature graph and
a posterior-predictive pass that numba runs in object mode -- so it lives in a
module of its own, where ``--dist loadfile`` can give it a worker to itself
instead of gluing it to the twenty fast tests that share its fixture.

The fixture it needs (``subject_marginal_context``) is in ``tests/conftest.py``
for exactly that reason; rebuilding it here costs about two seconds.
"""

import numpy as np
import pymc as pm
import pytest

from vocab_growth.models.subject_marginal import partition_subject_rows

pytestmark = pytest.mark.slow


def test_the_marginalised_model_samples_and_predicts(subject_marginal_context):
    """The whole path a fit needs: NUTS, log-likelihood, posterior predictive."""
    model = subject_marginal_context.model
    with model:
        trace = pm.sample(
            draws=8,
            tune=8,
            chains=1,
            cores=1,
            progressbar=False,
            random_seed=17,
            compute_convergence_checks=False,
        )
        pm.compute_log_likelihood(trace, progressbar=False)
        pm.sample_posterior_predictive(
            trace, var_names=["y_obs"], progressbar=False, random_seed=17,
            extend_inferencedata=True,
        )

    n_rows = len(subject_marginal_context.analysis_df)
    assert trace.log_likelihood["y_obs"].shape == (1, 8, n_rows)
    assert trace.posterior_predictive["y_obs"].shape == (1, 8, n_rows)
    assert np.isfinite(trace.log_likelihood["y_obs"].values).all()
    codes = np.asarray(subject_marginal_context.analysis_df["subject_code"], dtype=int)
    partition = partition_subject_rows(codes)
    assert trace.posterior["delta_subject_raw"].shape[-1] == partition.n_repeat_subjects
