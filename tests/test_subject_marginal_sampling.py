# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Run singleton marginalisation through sampling and posterior prediction.

Keep this compilation-heavy test separate so loadfile scheduling can give it
a worker. The shared graph fixture lives in conftest.py.
"""

import numpy as np
import pymc as pm
import pytest

from vocab_growth.models.subject_marginal import partition_subject_rows

#: The `xdist_group` half keeps this module on one worker under `--dist
#: loadgroup`: its context comes from a module-scoped fixture in `conftest.py`
#: that builds a real marginalised model, so tests here must not be spread
#: across workers. It holds one test today, which makes the mark a no-op and a
#: statement of the invariant for the second one.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("subject-marginal")]


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
