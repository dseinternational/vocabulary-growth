# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The registered hard-tier convergence exception, and what must still close it.

The R-hat/ESS gate is the project's only fail-closed convergence check, so an
escape hatch in it is exactly the kind of mechanism that rots into a blanket
override. These tests pin the four ways it must refuse to widen, and the one
way it is allowed to apply.
"""

from __future__ import annotations

import json
import os

import pytest

from vocab_growth.fit_artifacts import (
    ACCEPTED_EXCEPTION_KEY,
    CONVERGENCE_EXCEPTIONS,
    accepted_rhat_exception,
    convergence_caveats,
)
from vocab_growth.models.common import ConvergenceGateError, enforce_convergence_gate

VG11_SUMMARY = {
    "max_rhat": 1.0125,
    "min_ess": 1139.0,
    "rhat_failing": ["g_unit_hsgp_coeffs[4]"],
    "ess_failing": [],
    "checks": {},
}


def test_only_vg11_has_an_exception():
    """One entry. A second is a decision, not a refactor."""
    assert set(CONVERGENCE_EXCEPTIONS) == {"VG11"}


def test_the_exception_applies_to_the_failure_it_was_written_for():
    assert accepted_rhat_exception("VG11", VG11_SUMMARY) is not None


@pytest.mark.parametrize(
    ("label", "override"),
    [
        ("a different parameter", {"rhat_failing": ["g_unit_hsgp_coeffs[6]"]}),
        ("an additional parameter", {"rhat_failing": ["g_unit_hsgp_coeffs[4]", "eta"]}),
        ("a worse r_hat", {"max_rhat": 1.02}),
        ("an ESS failure alongside", {"ess_failing": ["tau_subject"]}),
        ("a scan that did not complete", {"max_rhat": None}),
    ],
)
def test_the_exception_refuses_to_widen(label, override):
    """Each of these must close the gate again, not be absorbed by the exception."""
    summary = {**VG11_SUMMARY, **override}
    assert accepted_rhat_exception("VG11", summary) is None, label


def test_it_does_not_leak_to_other_models():
    """The identical failure on any other model still fails closed."""
    for model_id in ("VG10", "VG12", "VG13", "VG15"):
        assert accepted_rhat_exception(model_id, VG11_SUMMARY) is None


def test_an_accepted_exception_surfaces_as_a_caveat():
    """It must reach Appendix B by the same route as a divergence.

    `convergence_caveats` recomputes from the payload on disk rather than
    trusting a marker file, so the acceptance has to live in the payload — this
    is what makes the exception impossible to apply silently.
    """
    caveats = convergence_caveats(
        {
            "checks": {},
            ACCEPTED_EXCEPTION_KEY: {
                "parameters": ["g_unit_hsgp_coeffs[4]"],
                "observed_max_rhat": 1.0125,
                "decided": "2026-08-15",
            },
        }
    )
    assert len(caveats) == 1
    assert "did not clear the hard convergence gate" in caveats[0]
    assert "g_unit_hsgp_coeffs[4]" in caveats[0]


def test_the_gate_accepts_vg11_and_records_it(tmp_path):
    summary_path = tmp_path / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(VG11_SUMMARY), encoding="utf-8")

    caveats = enforce_convergence_gate(
        dict(VG11_SUMMARY),
        sampling_config_name="rep",
        output_dir=str(tmp_path),
        model_id="VG11",
    )
    assert any("accepted R-hat exception" in c for c in caveats)

    # Recorded into the payload, so every downstream reader recomputes it.
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert ACCEPTED_EXCEPTION_KEY in payload
    assert payload[ACCEPTED_EXCEPTION_KEY]["observed_max_rhat"] == 1.0125

    # And the fit is NOT marked as a hard failure.
    assert not os.path.isfile(tmp_path / "CONVERGENCE_FAILED.txt")


def test_the_gate_still_closes_for_the_same_failure_on_another_model(tmp_path):
    with pytest.raises(ConvergenceGateError):
        enforce_convergence_gate(
            dict(VG11_SUMMARY),
            sampling_config_name="rep",
            output_dir=str(tmp_path),
            model_id="VG12",
        )
    assert os.path.isfile(tmp_path / "CONVERGENCE_FAILED.txt")


def test_no_model_id_means_no_exception(tmp_path):
    """A caller that does not identify the model gets the strict gate."""
    with pytest.raises(ConvergenceGateError):
        enforce_convergence_gate(
            dict(VG11_SUMMARY),
            sampling_config_name="rep",
            output_dir=str(tmp_path),
        )
