# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Permission and experimental energy contracts for shared helpers 0.17.0."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def marginal_comparison(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/experiments/compare_marginal_arms.py"
    )
    spec = importlib.util.spec_from_file_location(
        "marginal_comparison_under_test", script
    )
    module = importlib.util.module_from_spec(spec)
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(sys, "argv", [str(script), "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="No VG12 marginalisation arms"):
        spec.loader.exec_module(module)
    return module


def test_experimental_energy_uses_named_chain_and_draw_order(marginal_comparison):
    values = np.random.default_rng(42).normal(size=(3, 50))
    energy = xr.Dataset(
        {"energy": (("chain", "draw"), values)}, coords={"chain": [5, 2, 9]}
    )
    expected = np.sum(np.diff(values, axis=1) ** 2, axis=1) / np.sum(
        (values - values.mean(axis=1, keepdims=True)) ** 2, axis=1
    )

    np.testing.assert_allclose(marginal_comparison._bfmi(energy), expected)
    np.testing.assert_allclose(
        marginal_comparison._bfmi(energy.transpose("draw", "chain")), expected
    )


def test_experimental_constant_energy_remains_undefined(marginal_comparison):
    energy = xr.Dataset({"energy": (("chain", "draw"), np.zeros((2, 10)))})
    assert np.isnan(marginal_comparison._bfmi(energy)).all()


def test_experimental_missing_energy_still_stops_comparison(marginal_comparison):
    with pytest.raises(KeyError, match="energy"):
        marginal_comparison._bfmi(xr.Dataset())


def test_experimental_energy_with_an_event_dimension_is_refused(marginal_comparison):
    energy = xr.Dataset({"energy": (("chain", "draw", "event"), np.zeros((2, 10, 2)))})
    with pytest.raises(ValueError, match="chain and draw"):
        marginal_comparison._bfmi(energy)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("mask", [0o022, 0o027, 0o077])
def test_new_and_replaced_files_keep_ordinary_creation_modes(tmp_path, mask):
    script = """
import json
import os
import stat
import sys
from pathlib import Path
from vocab_growth.fit_artifacts import write_atomic
os.umask(int(sys.argv[2]))
root = Path(sys.argv[1])
for name in ('new.json', 'replaced.json'):
    target = root / name
    if name == 'replaced.json':
        target.write_text('old')
        target.chmod(0o400)
    def write(temporary):
        temporary.write_text('new')
        temporary.chmod(0o400)
    write_atomic(target, write)
print(json.dumps({p.name: stat.S_IMODE(p.stat().st_mode) for p in root.iterdir()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), str(mask)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {
        "new.json": 0o666 & ~mask,
        "replaced.json": 0o666 & ~mask,
    }
