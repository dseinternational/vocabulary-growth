# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shared cross-validation fold fit, and the bug that made it shared.

``kfold_loso.py`` and ``wave_forward_score.py`` hold different things out and
score different units, but the fit between those two decisions is identical.
The second script had it copied by hand, and the copy read the energy verdict
from ``gate["bfmi_ok"]`` where the payload carries it under
``gate["checks"]["bfmi"]`` -- so every fold reported a passing BFMI whatever the
sampler found, and nothing in the output would have shown it.

So the two claims pinned here are the reading itself, and that there is only one
of it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vocab_growth.fold_fits import fold_gate_fields

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _gate(**overrides) -> dict:
    payload = {
        "passed": True,
        "max_rhat": 1.004,
        "min_ess": 1500.0,
        "divergences": 0,
        "checks": {
            "rhat": True,
            "ess": True,
            "divergences": True,
            "bfmi": True,
            "diagnostics_assessable": True,
        },
    }
    payload.update(overrides)
    return payload


def test_the_energy_verdict_is_read_from_the_nested_checks():
    """The exact mistake the copy made.

    A gate that passed everything but the energy check must come back with
    ``bfmi_ok`` False. Reading a top-level ``bfmi_ok`` returns the default and
    silently reports every fold as fine.
    """
    checks = _gate()["checks"] | {"bfmi": False}
    fields = fold_gate_fields(_gate(passed=False, checks=checks))
    assert fields["bfmi_ok"] is False
    assert fields["passed"] is False


def test_an_absent_energy_check_is_unknown_rather_than_passing():
    """``None``, not ``True``.

    A gate payload with no energy check has not told us the fit was fine; a
    default of True would put that claim in the fits table.
    """
    fields = fold_gate_fields(_gate(checks={}))
    assert fields["bfmi_ok"] is None


@pytest.mark.parametrize(
    "field,value",
    [("max_rhat", 1.03), ("min_ess", 55.0), ("divergences", 7)],
)
def test_the_scalar_verdicts_pass_through(field, value):
    assert fold_gate_fields(_gate(**{field: value}))[field] == value


def test_no_script_defines_its_own_copy_of_the_gate_reading():
    """One reading of the payload, or the next copy repeats the bug.

    Checked on the source rather than by import, so a script that is not
    importable in a test environment still counts.
    """
    offenders = []
    for path in sorted(SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in {
                "fold_gate_fields",
                "fit_holdout_fold",
            }:
                offenders.append(f"{path.name}:{node.name}")
    assert not offenders, (
        "These scripts define their own copy of a fold-fit helper instead of "
        f"importing vocab_growth.fold_fits: {', '.join(offenders)}."
    )
