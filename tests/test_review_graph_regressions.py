# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Statistical regressions from the September 2026 model review."""

from dataclasses import replace

import numpy as np
import pymc as pm
import pytest
from scipy.special import logsumexp
from scipy.stats import dirichlet_multinomial
from support.synthetic_graphs import (
    build_synthetic_model,
    fixed_point,
    fixed_point_logp,
)

from vocab_growth.models import common_joint_modality as joint
from vocab_growth.models.catalogue import CATALOGUE


@pytest.mark.slow
@pytest.mark.parametrize("key", ["vg01", "vg05", "vg10", "vg11", "vg14", "vg15"])
def test_reporting_grid_does_not_change_the_observed_model(key, tmp_path, monkeypatch):
    """Exercise every engine, with observations strictly inside the GP domain."""
    record = CATALOGUE[key]
    low, high = record.definition.gp_domain_months
    base = replace(record.definition, ages_query=(low + 1, high - 1))
    extended = replace(base, ages_query=(low, low + 1, high - 1), n_plot=base.n_plot + 7)
    logps, covariances = [], []
    for definition in (base, extended):
        context = build_synthetic_model(
            definition, record.engine, output_dir=str(tmp_path), monkeypatch=monkeypatch)
        model = context.model
        logps.append(fixed_point_logp(model))
        if key != "vg10":
            continue
        # Compare the induced covariance, not just a changed basis convention.
        latent = model.replace_rvs_by_values([model["f_u_obs"]])[0]
        fn = model.compile_fn(latent, inputs=model.value_vars, on_unused_input="ignore")
        point = fixed_point(model)
        coeff = "g_unit_u_hsgp_coeffs"
        point[coeff] = np.zeros_like(point[coeff])
        origin = fn(point)
        columns = []
        for index in range(point[coeff].size):
            point[coeff].fill(0)
            point[coeff][index] = 1.0
            columns.append(fn(point) - origin)
        design = np.stack(columns, axis=1)
        covariances.append(design @ design.T)
    np.testing.assert_allclose(logps[0], logps[1], atol=1e-9, rtol=1e-12)
    if covariances:
        np.testing.assert_allclose(covariances[0], covariances[1], atol=1e-12, rtol=1e-12)


def test_produced_likelihood_is_the_conditional_four_cell_distribution(tmp_path, monkeypatch):
    """Enumerate the conditional distribution and check the engine's actual alpha."""
    pi = np.array([0.6, 0.1, 0.2, 0.1])
    monkeypatch.setattr(joint, "_composition_probabilities",
                        lambda r, q, psi: 0 * r[:, None] + pi)
    captured = {}
    original = pm.DirichletMultinomial

    def capture(name, **kwargs):
        captured[name] = kwargs["a"]
        return original(name, **kwargs)

    monkeypatch.setattr(pm, "DirichletMultinomial", capture)
    record = CATALOGUE["vg15"]
    model = build_synthetic_model(record.definition, record.engine,
                                  output_dir=str(tmp_path), monkeypatch=monkeypatch).model
    expressions = model.replace_rvs_by_values([captured["nz_prod_cells_obs"], model["conc"]])
    evaluate = model.compile_fn(expressions, inputs=model.value_vars, on_unused_input="ignore")
    alpha, conc = evaluate(fixed_point(model))
    np.testing.assert_allclose(alpha, np.broadcast_to(conc * pi[1:], alpha.shape))
    produced, understood = 4, 10
    counts = np.array([[i, j, produced - i - j]
                       for i in range(produced + 1) for j in range(produced - i + 1)])
    full = np.column_stack((np.full(len(counts), understood - produced), counts))
    conditional = dirichlet_multinomial.logpmf(full, conc * pi, understood)
    conditional -= logsumexp(conditional)
    actual = dirichlet_multinomial.logpmf(counts, alpha[0], produced)
    np.testing.assert_allclose(actual, conditional, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("r,q,psi", [(0.3, 0.4, 1.0), (0.01, 0.99, 3.0), (0.6, 0.8, 10.0)])
def test_composition_probabilities_preserve_marginals(r, q, psi):
    pi = joint._composition_probabilities(np.array([r]), np.array([q]), np.array([psi])).eval()[0]
    np.testing.assert_allclose(pi.sum(), 1.0)
    np.testing.assert_allclose(pi[1] + pi[3], r)
    np.testing.assert_allclose(pi[2] + pi[3], q)
    np.testing.assert_allclose(pi[0] * pi[3] / (pi[1] * pi[2]), psi)
