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
import pytest

from vocab_growth.models.common import (
    ConvergenceGateError,
    _report_diagnostic_warnings,
    diagnostics_var_names,
    enforce_convergence_gate,
    is_reporting_quality_config,
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


def test_reporting_fit_fails_closed_and_retains_marker(tmp_path):
    payload = _payload(rhat_failing=["theta[0]"], max_rhat=1.03)

    with pytest.raises(ConvergenceGateError):
        enforce_convergence_gate(
            payload,
            sampling_config_name="rep",
            output_dir=str(tmp_path),
        )

    marker = tmp_path / "CONVERGENCE_FAILED.txt"
    assert marker.exists()
    assert "must not proceed" in marker.read_text()


def test_development_fit_reports_but_does_not_raise(tmp_path):
    enforce_convergence_gate(
        _payload(ess_failing=["theta[0]"], min_ess=100),
        sampling_config_name="dev",
        output_dir=str(tmp_path),
    )

    assert not (tmp_path / "CONVERGENCE_FAILED.txt").exists()


def test_sampling_config_classification_is_explicit():
    assert is_reporting_quality_config("rep")
    assert is_reporting_quality_config("reporting-lite")
    assert not is_reporting_quality_config("dev")
    assert not is_reporting_quality_config("testing")

    with pytest.raises(ValueError, match="no convergence-gate classification"):
        is_reporting_quality_config("new-upstream-tier")


def test_unknown_sampling_tier_cannot_bypass_gate(tmp_path):
    with pytest.raises(ValueError, match="no convergence-gate classification"):
        enforce_convergence_gate(
            _payload(),
            sampling_config_name="new-upstream-tier",
            output_dir=str(tmp_path),
        )


# ---- soft tier: divergences and energy BFMI ----
#
# The gate previously read only the hard keys (max_rhat/min_ess and their failing
# lists), so a reporting fit with divergences or a low BFMI left no durable trace
# and nothing downstream could see it. These pin the two-tier split: the soft tier
# is recorded, not fail-closed, and it is recomputed from the diagnostics payload
# on disk so fits made before the marker existed are judged on their diagnostics.

def _soft(divergences=0, bfmi=(0.9, 0.9), **kwargs):
    payload = _payload(**kwargs)
    payload["divergences"] = divergences
    payload["bfmi_per_chain"] = list(bfmi)
    payload["checks"] = {
        "rhat": not payload["rhat_failing"],
        "ess": not payload["ess_failing"],
        "divergences": divergences == 0,
        "bfmi": all(b is not None and b >= 0.3 for b in bfmi),
    }
    payload["thresholds"]["bfmi_threshold"] = 0.3
    return payload


def test_soft_tier_records_caveats_without_failing_closed(tmp_path):
    caveats = enforce_convergence_gate(
        _soft(divergences=3, bfmi=(0.28, 0.35)),
        sampling_config_name="rep",
        output_dir=str(tmp_path),
    )

    # Reportable: the hard tier passed, so no failure marker and no exception.
    assert not (tmp_path / "CONVERGENCE_FAILED.txt").exists()
    assert len(caveats) == 2
    marker = tmp_path / "CONVERGENCE_CAVEATS.txt"
    assert marker.exists()
    text = marker.read_text()
    assert "3 divergent transition(s)" in text
    assert "energy BFMI below 0.3 (min 0.280)" in text


def test_clean_reporting_fit_records_no_caveats_and_clears_stale_marker(tmp_path):
    stale = tmp_path / "CONVERGENCE_CAVEATS.txt"
    stale.write_text("from an earlier run\n")

    assert enforce_convergence_gate(
        _soft(), sampling_config_name="rep", output_dir=str(tmp_path)
    ) == []

    # Absence of the marker must mean "checked and clean", never "never looked".
    assert not stale.exists()


def test_hard_tier_failure_still_takes_precedence(tmp_path):
    with pytest.raises(ConvergenceGateError):
        enforce_convergence_gate(
            _soft(divergences=5, rhat_failing=["theta[0]"], max_rhat=1.04),
            sampling_config_name="rep",
            output_dir=str(tmp_path),
        )
    assert (tmp_path / "CONVERGENCE_FAILED.txt").exists()
    assert not (tmp_path / "CONVERGENCE_CAVEATS.txt").exists()


def test_caveats_are_read_from_the_diagnostics_payload(tmp_path):
    import json

    from vocab_growth.fit_artifacts import read_convergence_caveats

    assert read_convergence_caveats(str(tmp_path)) == []   # no payload yet
    (tmp_path / "diagnostics_summary.json").write_text(
        json.dumps(_soft(divergences=0, bfmi=(0.277, 0.41)))
    )
    caveats = read_convergence_caveats(str(tmp_path))
    assert len(caveats) == 1
    assert "energy BFMI below 0.3 (min 0.277)" in caveats[0]


def test_publication_validation_rejects_a_caveated_fit(tmp_path):
    import json

    from vocab_growth.fit_artifacts import validate_fit_output

    (tmp_path / "diagnostics_summary.json").write_text(
        json.dumps(_soft(divergences=2))
    )
    caveat_errors = [
        error
        for error in validate_fit_output(str(tmp_path), require_clean_convergence=True)
        if "Convergence caveat" in error
    ]
    assert len(caveat_errors) == 1
    assert "2 divergent transition(s)" in caveat_errors[0]

    # Without the publication flag the same fit raises no convergence complaint,
    # so development and review keep working from it.
    assert not [
        error
        for error in validate_fit_output(str(tmp_path))
        if "Convergence caveat" in error
    ]


def test_publication_policy_requires_clean_convergence_provisional_does_not():
    from vocab_growth.fit_artifacts import fit_validation_kwargs

    shared = dict(
        expected_definition={},
        expected_sampling_config_name="rep",
        expected_sampling_parameters={},
    )
    published = fit_validation_kwargs(
        "sync", current_source_data_hash="sha256:x", **shared
    )
    assert published["require_clean_convergence"] is True
    assert "require_clean_convergence" not in fit_validation_kwargs(
        "provisional-sync", **shared
    )
