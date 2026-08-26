# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for nested held-out scoring in ``kfold_loso``."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.stats import betabinom

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "kfold_loso.py"
_SPEC = importlib.util.spec_from_file_location("kfold_loso_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

N_TRIALS = _MODULE.N_TRIALS
holdout_subject_elpds = _MODULE.holdout_subject_elpds


def test_non_integer_parent_uses_marginal_spoken_likelihood():
    analysis_df = pd.DataFrame(
        {
            "subject_code": [0],
            "understood": [100.5],
            "spoken": [25],
        },
        index=[42],
    )
    posterior = xr.Dataset(
        {
            "p_u_obs": (("chain", "draw", "obs_id"), [[[0.4]]]),
            "p_s_obs": (("chain", "draw", "obs_id"), [[[0.2]]]),
            "q_obs": (("chain", "draw", "obs_id"), [[[0.9]]]),
            "kappa_u_obs": (("chain", "draw", "obs_id"), [[[20.0]]]),
            "kappa_s_obs": (("chain", "draw", "obs_id"), [[[15.0]]]),
        }
    )
    trace = xr.DataTree.from_dict({"posterior": posterior})

    actual = holdout_subject_elpds(analysis_df, trace, np.array([0]))[0]
    expected = betabinom.logpmf(100, N_TRIALS, 0.4 * 20.0, 0.6 * 20.0)
    expected += betabinom.logpmf(25, N_TRIALS, 0.2 * 15.0, 0.8 * 15.0)

    assert actual == pytest.approx(expected)


# --- Per-fold convergence diagnostics (issue #266, finding 7c) -----------------


def _fold_gate_payload(**overrides):
    """A payload in the shape ``write_diagnostics_summary`` returns."""
    payload = {
        "passed": True,
        "checks": {
            "rhat": True,
            "ess": True,
            "divergences": True,
            "bfmi": True,
            "diagnostics_assessable": True,
        },
        "divergences": 0,
        "max_rhat": 1.004,
        "min_ess": 1500.0,
        "bfmi_per_chain": [0.9, 0.85],
        "rhat_failing": [],
        "ess_failing": [],
        "unassessable_parameters": [],
        "thresholds": {"rhat_max": 1.01, "ess_threshold": 400, "bfmi_threshold": 0.3},
    }
    payload.update(overrides)
    return payload


def _record(model_short, fold, passed):
    return _MODULE.FoldFitRecord(
        model_short=model_short,
        fold=fold,
        n_holdout_subjects=1,
        n_holdout_obs_u=1,
        n_holdout_obs_s=1,
        wall_seconds=0.1,
        **_MODULE.fold_gate_fields(_fold_gate_payload(passed=passed)),
    )


def test_fit_fold_runs_the_canonical_diagnostics_scan(monkeypatch, tmp_path):
    canned = _fold_gate_payload()
    calls = {}

    monkeypatch.setattr(_MODULE, "KFOLD_TMP_DIR", str(tmp_path))
    monkeypatch.setattr(
        _MODULE, "configure_bivariate_priors", lambda context, definition: None
    )

    def fake_build(context, definition):
        context.set_model(object(), {})

    monkeypatch.setattr(_MODULE, "build_model_re", fake_build)

    sentinel_trace = object()

    def fake_sample(context, **kwargs):
        context.set_trace(sentinel_trace)

    monkeypatch.setattr(_MODULE, "sample", fake_sample)
    monkeypatch.setattr(
        _MODULE, "diagnostics_var_names", lambda model: (["eta"], ["eta", "delta"])
    )

    def fake_write(trace, output_dir, var_names=None):
        calls["trace"] = trace
        calls["output_dir"] = output_dir
        calls["var_names"] = var_names
        return canned

    monkeypatch.setattr(
        _MODULE.shared_diagnostics, "write_diagnostics_summary", fake_write
    )

    analysis_df = pd.DataFrame(
        {"age": [24.0, 30.0], "understood": [100.0, 200.0], "spoken": [10.0, 20.0]}
    )
    sampling_cfg = _MODULE.sampling.get_sampling_configuration("test")
    trace, n, gate = _MODULE.fit_fold(
        _MODULE.AVAILABLE["VG07"], analysis_df, sampling_cfg, "VG07_fold0"
    )

    assert trace is sentinel_trace
    assert n == 2
    # The payload is returned to the caller: the fold directory is deleted when
    # the next fold reuses it, so the JSON written into it is transient.
    assert gate is canned
    assert calls["trace"] is sentinel_trace
    # The scan covers the gate set (every free RV element-wise), not merely the
    # scalar summary set.
    assert calls["var_names"] == ["eta", "delta"]
    assert str(calls["output_dir"]).startswith(str(tmp_path))


def test_fold_fit_record_carries_the_gate_verdict_into_the_fits_csv(tmp_path):
    gate = _fold_gate_payload(
        passed=False,
        checks={
            "rhat": True,
            "ess": True,
            "divergences": False,
            "bfmi": False,
            "diagnostics_assessable": True,
        },
        divergences=7,
        max_rhat=1.006,
        min_ess=900.0,
    )
    record = _MODULE.FoldFitRecord(
        model_short="VG07",
        fold=0,
        n_holdout_subjects=10,
        n_holdout_obs_u=20,
        n_holdout_obs_s=18,
        wall_seconds=1.5,
        **_MODULE.fold_gate_fields(gate),
    )
    assert record.passed is False
    assert record.max_rhat == 1.006
    assert record.min_ess == 900.0
    assert record.divergences == 7
    assert record.bfmi_ok is False

    fit_df = pd.DataFrame([record.__dict__])
    out = tmp_path / "kfold_loso_fits.csv"
    fit_df.to_csv(out, index=False)
    written = pd.read_csv(out)
    for column in ("passed", "max_rhat", "min_ess", "divergences", "bfmi_ok"):
        assert column in written.columns
    assert bool(written.loc[0, "passed"]) is False


def test_unconverged_folds_flag_rows_rather_than_dropping_them():
    records = [
        _record("VG07", 0, True),
        _record("VG07", 1, True),
        _record("VG08", 0, True),
        _record("VG08", 1, False),
    ]
    flags = _MODULE.model_convergence_flags(records)
    assert flags == {"VG07": True, "VG08": False}

    elpd_df = pd.DataFrame(
        {"VG07": [-10.0, -12.0, -9.0], "VG08": [-11.0, -12.5, -9.5]},
        index=pd.Index([0, 1, 2], name="subject_code"),
    )
    summary = _MODULE.summarise_models(elpd_df, ("VG07", "VG08"), flags)
    assert "all_folds_converged" in summary.columns
    by_model = summary.set_index("model")["all_folds_converged"]
    assert bool(by_model["VG07"]) is True
    assert bool(by_model["VG08"]) is False
    # The flagged model's elpd row is still present, not dropped.
    assert set(summary["model"]) == {"VG07", "VG08"}

    pair = _MODULE.pairwise_compare(elpd_df, ("VG07", "VG08"), flags)
    assert len(pair) == 1
    assert pair.loc[0, "model_a"] == "VG07"
    assert pair.loc[0, "model_b"] == "VG08"
    # A comparison involving the unconverged model is flagged, not dropped.
    assert bool(pair.loc[0, "all_folds_converged"]) is False
