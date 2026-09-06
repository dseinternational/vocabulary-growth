# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for two reporting guards in the comparison scripts.

Both were written after a real run misreported rather than failed loudly, which
is the failure mode these pin:

* ``aggregate_summary._group_names`` -- ArviZ 1.x reads a trace into a
  ``DataTree`` whose ``groups`` is a *property* holding *paths*
  (``"/sample_stats"``). The old ``idata.groups()`` raised ``TypeError``, but
  the obvious repair -- dropping the parentheses -- silently reports every model
  as having no divergences, because ``"sample_stats"`` is not among the paths.
* ``loo_compare._warn_if_unusable`` -- the high Pareto-k count was printed but
  never judged. VG11's first LOO put 48% of observations above k = 0.7 with
  p_loo = 10,720, and the row still entered the table looking like any other.
"""

import importlib.util
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest


def _load(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_AGGREGATE = _load("aggregate_summary")
_LOO_COMPARE = _load("loo_compare")


# ---------------------------------------------------------------------------
# aggregate_summary._group_names
# ---------------------------------------------------------------------------
def _trace_with_sample_stats():
    rng = np.random.default_rng(0)
    return az.from_dict(
        {
            "posterior": {"theta": rng.normal(size=(2, 50))},
            "sample_stats": {"diverging": np.zeros((2, 50), dtype=bool)},
        }
    )


def test_group_names_strips_the_arviz_path_prefix():
    """The real reader's paths must reduce to bare names."""
    names = _AGGREGATE._group_names(_trace_with_sample_stats())
    assert "sample_stats" in names
    assert "posterior" in names
    assert not any(name.startswith("/") for name in names)


def test_group_names_accepts_a_method_style_groups():
    """Pre-1.x readers exposed ``groups()``; both shapes must work."""

    class OldStyle:
        def groups(self):
            return ["posterior", "sample_stats"]

    assert _AGGREGATE._group_names(OldStyle()) == {"posterior", "sample_stats"}


def test_group_names_finds_nothing_that_is_absent():
    """A trace without sample_stats must not be reported as having it."""
    rng = np.random.default_rng(0)
    trace = az.from_dict({"posterior": {"theta": rng.normal(size=(2, 50))}})
    assert "sample_stats" not in _AGGREGATE._group_names(trace)


def test_trace_divergences_reads_a_real_count(tmp_path):
    """The end-to-end path the TypeError broke: a stored trace's divergences.

    Pins the number rather than merely that it returns, because the tempting
    repair returns ``None`` here -- indistinguishable, in the summary table,
    from a model that genuinely never diverged.
    """
    rng = np.random.default_rng(1)
    diverging = np.zeros((2, 50), dtype=bool)
    diverging[0, :7] = True
    diverging[1, :5] = True
    trace = az.from_dict(
        {
            "posterior": {"theta": rng.normal(size=(2, 50))},
            "sample_stats": {"diverging": diverging},
        }
    )
    path = tmp_path / "trace.nc"
    trace.to_netcdf(str(path))
    assert _AGGREGATE.trace_divergences(str(path)) == 12


def test_trace_divergences_is_none_for_a_missing_trace(tmp_path):
    assert _AGGREGATE.trace_divergences(str(tmp_path / "absent.nc")) is None


# ---------------------------------------------------------------------------
# loo_compare._warn_if_unusable
# ---------------------------------------------------------------------------
def _row(n_high: int, n_obs: int, p_loo: float = 1.0) -> dict:
    return {"n_observations": n_obs, "pareto_k_gt_0.7": n_high, "pareto_k_unusable": n_high, "p_loo": p_loo}


@pytest.mark.parametrize(
    ("label", "n_high", "n_obs"),
    [
        ("VG11", 8815, 18522),  # 48% -- the fit that prompted the guard
        ("VG13", 3143, 6358),  # 49%
        ("VG12", 2705, 7052),  # 38%
        ("VG08", 309, 987),  # 31% -- the smallest degenerate share observed
    ],
)
def test_degenerate_models_are_called_out(capsys, label, n_high, n_obs):
    _LOO_COMPARE._warn_if_unusable(label, _row(n_high, n_obs, p_loo=10720.4))
    printed = capsys.readouterr().out
    assert "[unusable]" in printed
    assert label in printed
    # It must point the reader somewhere, not merely refuse the number.
    assert "loso" in printed.lower()


@pytest.mark.parametrize(
    ("label", "n_obs"),
    [("VG01", 1428), ("VG07", 987), ("VG03", 4075)],
)
def test_clean_models_stay_silent(capsys, label, n_obs):
    """Every model measured without subject random effects had zero high-k."""
    _LOO_COMPARE._warn_if_unusable(label, _row(0, n_obs, p_loo=21.9))
    assert capsys.readouterr().out == ""


def test_the_threshold_separates_the_two_families_with_room():
    """0% for every clean model, 30.3% for the lowest degenerate one.

    The guard is only useful if it sits strictly inside that gap; pinning it
    stops a later tweak from quietly moving it outside.
    """
    assert 0.0 < _LOO_COMPARE.HIGH_PARETO_K_UNUSABLE_SHARE < 299 / 987


def test_an_empty_frame_does_not_divide_by_zero(capsys):
    _LOO_COMPARE._warn_if_unusable("EMPTY", _row(0, 0))
    assert capsys.readouterr().out == ""
