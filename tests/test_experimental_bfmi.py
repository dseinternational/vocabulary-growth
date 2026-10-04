# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""BFMI compatibility in the archived geometry and repeater experiments."""

import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from vocab_growth.fit_artifacts import SAMPLED_PARAMETERS_ATTR

SCRIPTS = (
    "compare_arms",
    "verify_geometry",
    "vg08_replication_thinning",
    "vg12_repeaters_prior",
    "vg13_repeaters_cut",
)


def _trace(energy):
    rng = np.random.default_rng(42)
    names = (
        "tau",
        "tau_subject",
        "tau_subj_u",
        "tau_subj_q",
        "kappa_young",
        "kappa_young_u",
        "kappa_young_s",
        "kappa_min_u",
        "v_total",
        "subject_variance_share",
        "ell",
        "eta",
        "eta_u",
        "eta_q",
    )
    posterior = xr.Dataset(
        {name: (("chain", "draw"), rng.uniform(0.1, 1, (3, 50))) for name in names},
        coords={"chain": [5, 2, 9]},
        attrs={SAMPLED_PARAMETERS_ATTR: json.dumps(["tau", "eta"])},
    )
    stats = xr.Dataset(
        {
            "energy": energy,
            "diverging": (("chain", "draw"), np.zeros((3, 50), dtype=bool)),
        },
        coords={"chain": [5, 2, 9]},
    )
    return xr.DataTree.from_dict({"posterior": posterior, "sample_stats": stats})


@pytest.fixture(scope="module", params=SCRIPTS)
def experiment(request, tmp_path_factory):
    script = Path(__file__).parents[1] / "scripts/experiments" / f"{request.param}.py"
    spec = importlib.util.spec_from_file_location(f"{request.param}_under_test", script)
    module = importlib.util.module_from_spec(spec)
    root = tmp_path_factory.mktemp(request.param)
    (root / "geom").mkdir()
    values = np.random.default_rng(17).normal(size=(3, 50))
    trace = _trace((("chain", "draw"), values))
    output = io.StringIO()
    isfile = os.path.isfile
    read_csv = pd.read_csv

    # Two archived scripts run on import. Supply synthetic archive reads and
    # put compare_arms' summary beside a temporary __file__, away from the repo.
    with pytest.MonkeyPatch.context() as patch, contextlib.redirect_stdout(output):
        patch.setitem(sys.modules, spec.name, module)
        patch.setattr(module, "__file__", str(root / script.name))
        patch.setattr(az, "from_netcdf", lambda _path: trace)
        patch.setattr(
            os.path, "isfile", lambda path: str(path).startswith("/scratch/") or isfile(path)
        )
        patch.setattr(
            pd,
            "read_csv",
            lambda path, *args, **kwargs: (
                pd.DataFrame({"subject_code": [1, 1, 2]})
                if str(path).startswith("/scratch/")
                else read_csv(path, *args, **kwargs)
            ),
        )
        spec.loader.exec_module(module)
    return module, trace, output.getvalue(), root


def _previous_bfmi(trace):
    values = trace.sample_stats["energy"].values
    return np.array(
        [np.sum(np.diff(row) ** 2) / np.sum((row - row.mean()) ** 2) for row in values]
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_bfmi_keeps_array_contract_and_values(experiment, dtype):
    module, _, _, _ = experiment
    values = np.random.default_rng(23).normal(size=(3, 50)).astype(dtype)
    trace = SimpleNamespace(
        sample_stats=xr.Dataset(
            {"energy": (("chain", "draw"), values)}, coords={"chain": [5, 2, 9]}
        )
    )
    result = module.bfmi_per_chain(trace)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)
    assert result.dtype == float
    np.testing.assert_allclose(result, _previous_bfmi(trace), rtol=1e-6)
    trace.sample_stats = trace.sample_stats.transpose("draw", "chain")
    np.testing.assert_allclose(module.bfmi_per_chain(trace), result)


def test_bfmi_does_not_drop_constant_or_nonfinite_chains(experiment):
    module, _, _, _ = experiment
    values = np.random.default_rng(23).normal(size=(3, 50))
    values[1] = 0
    values[2, 0] = np.nan
    trace = SimpleNamespace(
        sample_stats=xr.Dataset({"energy": (("chain", "draw"), values)})
    )
    result = module.bfmi_per_chain(trace)
    assert np.isfinite(result[0])
    assert np.isnan(result[1:]).all()
    assert np.isnan(result.min())
    assert np.isnan(result.mean())


def test_missing_energy_still_stops_scoring(experiment):
    module, _, _, _ = experiment
    with pytest.raises(KeyError, match="energy"):
        module.bfmi_per_chain(SimpleNamespace(sample_stats=xr.Dataset()))


def test_energy_with_extra_dimensions_is_refused(experiment):
    module, _, _, _ = experiment
    trace = SimpleNamespace(
        sample_stats=xr.Dataset(
            {"energy": (("chain", "draw", "event"), np.zeros((3, 50, 2)))}
        )
    )
    with pytest.raises(ValueError, match="chain and draw"):
        module.bfmi_per_chain(trace)


def test_recorded_scores_and_gate_decisions_are_preserved(experiment, monkeypatch):
    module, trace, output, root = experiment
    expected_bfmi = _previous_bfmi(trace)
    if module.__name__ == "compare_arms_under_test":
        records = json.loads((root / "geom/summary.json").read_text())
        assert [row["arm"] for row in records] == module.ARMS
        for row in records:
            assert row["min_bfmi"] == pytest.approx(expected_bfmi.min())
            assert row["mean_bfmi"] == pytest.approx(expected_bfmi.mean())
        return
    if module.__name__ == "verify_geometry_under_test":
        assert output.count(f"min BFMI = {expected_bfmi.min():.3f}") == 2
        return

    gate = {"passed": False, "checks": {"bfmi": False}, "max_rhat": 1.02, "min_ess": 120}
    args = (trace, gate)
    if module.__name__ == "vg08_replication_thinning_under_test":
        args = (trace, SimpleNamespace(slope_anchors=(8, 24), ages_query=(8, 24)), gate)
    with monkeypatch.context() as patch:
        patch.setattr(module, "bfmi_per_chain", _previous_bfmi)
        expected = module.score_arm(*args)
    result = module.score_arm(*args)
    assert result == expected
    assert result["bfmi_per_chain"] == pytest.approx(expected_bfmi.tolist())
    assert result["n_free_parameters"] == 2
    assert result["gate_passed"] is False
    assert result["bfmi_ok"] is False
