# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Compare registered graphs with the recorded structure and two saved points.

Each model uses a deterministic synthetic frame. The checks preserve variable
names and order, likelihood factors, dimensions, complete coordinate values,
and log probabilities evaluated at the same transformed parameter values.
They do not prove equivalence everywhere in parameter space. Separate tests
exercise non-default settings and the quantities returned by extracted helpers.

The baseline must stay fixed during refactoring. An intentional statistical
change needs its own explanation and review of any baseline update. Only then
regenerate it with::

    uv run python tests/support/regenerate_graph_baseline.py

The tests are slow because they build graphs and compile log probabilities.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from support.synthetic_graphs import (
    build_registered_model,
    fixed_point_logp,
    graph_fingerprint,
)

from vocab_growth.models.catalogue import CATALOGUE

# Keep this module on one worker: its fixture caches each graph across tests.
# Other slow modules can distribute independent builds across workers.
pytestmark = [pytest.mark.slow, pytest.mark.xdist_group("graph-equivalence")]

BASELINE_PATH = Path(__file__).parent / "support" / "graph_baseline.json"
REFERENCE_PATH = BASELINE_PATH.with_name("graph_reference_points.json")

#: Tolerance on the recorded log probability. PyTensor's rewrites can reassociate
#: a sum without changing the model -- the same effect measured at 4.4e-16
#: absolute on the sampled draws in ``test_observation_deterministics`` -- and a
#: refactor that only reorders operands is exactly what this must tolerate. A
#: changed scale, a swapped operand or a dropped term moves it far further than
#: this.
LOGP_RTOL = 1e-9

_MODEL_KEYS = sorted(CATALOGUE)


@pytest.fixture(scope="session")
def baseline() -> dict:
    if not BASELINE_PATH.is_file():
        pytest.fail(
            f"{BASELINE_PATH} is missing; regenerate it with "
            "tests/support/regenerate_graph_baseline.py"
        )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def reference_points() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


class _LazyBuilds(Mapping):
    """Each registered graph, built on first use and cached for the module's tests."""

    def __init__(self, output_dir, monkeypatch):
        self._output_dir = output_dir
        self._monkeypatch = monkeypatch
        self._cache: dict = {}

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = build_registered_model(
                key, output_dir=self._output_dir, monkeypatch=self._monkeypatch
            )
        return self._cache[key]

    def __iter__(self):
        return iter(_MODEL_KEYS)

    def __len__(self):
        return len(_MODEL_KEYS)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every registered model's graph, built once each and shared by the tests."""
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    root = str(tmp_path_factory.mktemp("graph-equivalence"))
    try:
        yield _LazyBuilds(root, patcher)
    finally:
        patcher.undo()


def test_the_baseline_covers_exactly_the_registered_models(baseline, reference_points):
    """A model registered without a baseline entry would be unguarded."""
    assert sorted(reference_points) == _MODEL_KEYS
    assert sorted(baseline) == _MODEL_KEYS, (
        f"baseline entries without a registered model: "
        f"{sorted(set(baseline) - set(_MODEL_KEYS))}; registered models with no "
        f"baseline entry: {sorted(set(_MODEL_KEYS) - set(baseline))}. Regenerate "
        "with tests/support/regenerate_graph_baseline.py."
    )


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_free_random_variables_keep_their_names_and_order(model_key, built, baseline):
    """Order matters as much as membership: it fixes the sampler's RNG stream.

    Two models with the same free variables created in a different order draw
    different values from the same seed, so a reordering is a change to every
    fit even though nothing about the distribution moved.
    """
    actual = [rv.name for rv in built[model_key].model.free_RVs]
    assert actual == baseline[model_key]["free_RVs"]


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_deterministics_keep_their_names_and_order(model_key, built, baseline):
    actual = [d.name for d in built[model_key].model.deterministics]
    assert actual == baseline[model_key]["deterministics"]


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_likelihood_factorisation_is_unchanged(model_key, built, baseline):
    """The observed variables are the likelihood's factors, in order."""
    actual = [rv.name for rv in built[model_key].model.observed_RVs]
    assert actual == baseline[model_key]["observed_RVs"]


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_dimensions_and_coordinates_are_unchanged(
    model_key, built, baseline, reference_points
):
    """A variable that keeps its name and loses its dims changes what readers get.

    Every consumer of a stored trace indexes by dimension -- the extractors, the
    summaries, the comparison suite -- so dims are part of the contract even
    when the values are not.
    """
    fingerprint = graph_fingerprint(built[model_key].model)
    assert fingerprint["dims"] == baseline[model_key]["dims"]
    assert fingerprint["coords"] == baseline[model_key]["coords"]
    actual = {
        name: None if values is None else np.asarray(values).tolist()
        for name, values in built[model_key].model.coords.items()
    }
    assert actual == reference_points[model_key]["coordinates"]


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_log_probability_at_saved_points_is_unchanged(
    model_key, built, baseline, reference_points
):
    """Compare at the old parameter values, even when the new prior moves."""
    reference = reference_points[model_key]
    assert reference["logps"][0] == pytest.approx(
        baseline[model_key]["logp_at_fixed_point"], rel=LOGP_RTOL
    )
    evaluate = built[model_key].model.compile_logp()
    for point, expected in zip(reference["points"], reference["logps"], strict=True):
        actual = float(
            evaluate({name: np.asarray(value) for name, value in point.items()})
        )
        assert actual == pytest.approx(expected, rel=LOGP_RTOL), (
            f"{model_key}'s log probability at a saved point moved: {expected!r} -> {actual!r}. "
            "Inspect the statistical change before updating either reference file."
        )


def test_the_fingerprint_is_stable_across_two_builds(built, reference_points):
    """A fingerprint that moved between two builds could never guard anything."""
    import tempfile

    from _pytest.monkeypatch import MonkeyPatch
    from support.synthetic_graphs import build_registered_model as build

    patcher = MonkeyPatch()
    try:
        with tempfile.TemporaryDirectory() as directory:
            for model_key in ("vg10", "vg22"):
                rebuilt = build(model_key, output_dir=directory, monkeypatch=patcher)
                assert graph_fingerprint(rebuilt.model) == graph_fingerprint(
                    built[model_key].model
                )
                point = {
                    name: np.asarray(value)
                    for name, value in reference_points[model_key]["points"][0].items()
                }
                assert fixed_point_logp(rebuilt.model, point) == fixed_point_logp(
                    built[model_key].model, point
                )
    finally:
        patcher.undo()


def test_saved_point_detects_a_changed_prior_scale():
    """Recomputing the point from each prior used to hide this change."""
    import pymc as pm
    from support.synthetic_graphs import fixed_point

    with pm.Model() as original:
        pm.HalfNormal("x", sigma=np.float64(1))
    point = fixed_point(original)
    with pm.Model() as changed:
        pm.HalfNormal("x", sigma=np.float64(2))
    assert fixed_point_logp(original, point) != pytest.approx(
        fixed_point_logp(changed, point)
    )
