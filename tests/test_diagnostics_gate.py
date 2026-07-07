# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the convergence-gate variable selection in ``models.common``.

The readable diagnostics table shows only scalar (size <= 2) unobserved RVs,
but the pass/fail convergence gate must screen every sampled parameter —
including vector-valued free RVs such as the HSGP basis coefficients and the
study/subject random intercepts — element-wise. These tests pin that split,
and the gate-payload-driven console banner.
"""

import numpy as np
import preliz as pz
import pymc as pm

from vocab_growth.models.common import (
    _report_diagnostic_warnings,
    diagnostics_var_names,
)
from vocab_growth.models.gp_utils import GPGrid, trend_and_gp


def test_gate_includes_vector_free_rvs_summary_stays_scalar():
    with pm.Model() as m:
        a = pm.Normal("a")
        pm.Normal("pair", shape=2)
        coeffs = pm.Normal("coeffs", shape=18)
        pm.HalfNormal("sigma")
        pm.Deterministic("a_doubled", 2.0 * a)
        pm.Deterministic("f_grid", coeffs.cumsum())
        pm.Normal("y_obs", mu=a, sigma=1.0, observed=np.zeros(4))

    summary_names, gate_names = diagnostics_var_names(m)

    # The readable table: scalar/2-element unobserved RVs (incl. deterministics).
    assert {"a", "pair", "sigma", "a_doubled"} <= set(summary_names)
    assert "coeffs" not in summary_names
    assert "f_grid" not in summary_names

    # The gate: everything in the table plus every vector-valued free RV.
    assert "coeffs" in gate_names
    assert set(summary_names) <= set(gate_names)
    # Vector deterministics are derived, not sampled — still excluded.
    assert "f_grid" not in gate_names
    # Observed RVs never enter either set.
    assert "y_obs" not in gate_names
    # No duplicates (scalar free RVs are already in the summary set).
    assert len(gate_names) == len(set(gate_names))


def test_gate_covers_hsgp_basis_coefficients():
    grid = GPGrid(sa_z=-1.0, sb_z=1.0, ell_low_z=0.2, ell_high_z=2.0, M=[10], L=[2.0])
    x = np.linspace(-2.0, 2.0, 8).reshape(-1, 1)
    with pm.Model(coords={"all_id": range(8), "x_dim": range(1)}) as m:
        X = pm.Data("X_all_z", x, dims=("all_id", "x_dim"))
        trend_and_gp(
            cfg_low=pz.Beta(alpha=1, beta=15),
            cfg_hi=pz.Beta(alpha=1.1, beta=1.1),
            cfg_ell=pz.Beta(alpha=2, beta=2),
            cfg_eta=pz.HalfNormal(sigma=1.0),
            suffix="_u",
            X_all_z_data=X,
            grid=grid,
            store_deterministic=True,
            latent_name="f_u_all",
        )

    summary_names, gate_names = diagnostics_var_names(m)

    coeff_names = [n for n in (v.name for v in m.free_RVs) if "hsgp_coeffs" in n]
    assert coeff_names, "expected the HSGP prior to create a basis-coefficient RV"
    for name in coeff_names:
        assert name not in summary_names
        assert name in gate_names


def _payload(rhat_failing=(), ess_failing=(), max_rhat=1.005, min_ess=1200.0):
    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "rhat_failing": list(rhat_failing),
        "ess_failing": list(ess_failing),
        "thresholds": {"rhat_max": 1.01, "ess_threshold": 400},
    }


def test_banner_passes_on_clean_payload(capsys):
    _report_diagnostic_warnings(_payload())
    out = capsys.readouterr().out
    assert "✓" in out
    assert "free parameters" in out


def test_banner_warns_on_failing_vector_elements(capsys):
    _report_diagnostic_warnings(
        _payload(
            rhat_failing=["g_unit_hsgp_coeffs_[3]"],
            ess_failing=["study_intercept[1]", "study_intercept[4]"],
            max_rhat=1.021,
            min_ess=145.0,
        )
    )
    out = capsys.readouterr().out
    assert "⚠" in out
    assert "1 parameter(s) with r_hat > 1.01" in out
    assert "2 parameter(s) with bulk or tail ESS < 400" in out
    assert "✓" not in out


def test_banner_claims_nothing_when_gate_scan_failed(capsys):
    _report_diagnostic_warnings(_payload(max_rhat=None, min_ess=None))
    out = capsys.readouterr().out
    assert "✓" not in out
    assert "⚠" not in out
