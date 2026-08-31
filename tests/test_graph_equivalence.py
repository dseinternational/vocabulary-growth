# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A refactor must not move any registered model's graph.

Issue #273's structural work -- resolving the subject-effect variants into one
typed plan, splitting the 772-line bivariate random-effect builder, freezing the
definitions -- touches the code that writes the PyMC graph. Its own constraints
say what must survive: free random-variable names **and order**, deterministic
names required by consumers, dimensions, coordinates, likelihood factorisation,
and a fixed-point log probability to numerical tolerance. This module is that
check, standing rather than ad hoc.

Every registered model is built on one small deterministic synthetic frame
(``tests/support/synthetic_graphs``) and compared against a committed baseline.
Synthetic on purpose: the recorded fingerprint is then a function of the **code
alone**, so a legitimate data change does not present as a refactor failure and
a real refactor failure cannot hide inside one. Data changes are already
guarded, exactly, by ``data.analysis_frame_hash``.

**A deliberate statistical change is expected to fail this and then update the
baseline.** That is the point: the baseline diff is the change's own statement
of what moved in the graph, reviewable line by line beside the reasoning. What
it prevents is a refactor moving something silently.

Marked ``slow``: twenty graph builds plus twenty log-probability compilations is
about a hundred seconds of real numerical work. CI runs the slow job on every
pull request, so the guard is on every change; it is out of the fast local loop
only.

Regenerate the baseline with::

    uv run python tests/support/regenerate_graph_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support.synthetic_graphs import (
    build_registered_model,
    fixed_point_logp,
    graph_fingerprint,
)

from vocab_growth.models.catalogue import CATALOGUE

pytestmark = pytest.mark.slow

BASELINE_PATH = Path(__file__).parent / "support" / "graph_baseline.json"

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


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every registered model's graph, built once and shared by the tests below."""
    from _pytest.monkeypatch import MonkeyPatch

    patcher = MonkeyPatch()
    root = str(tmp_path_factory.mktemp("graph-equivalence"))
    try:
        models = {
            key: build_registered_model(key, output_dir=root, monkeypatch=patcher)
            for key in _MODEL_KEYS
        }
        yield models
    finally:
        patcher.undo()


def test_the_baseline_covers_exactly_the_registered_models(baseline):
    """A model registered without a baseline entry would be unguarded."""
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
def test_dimensions_and_coordinates_are_unchanged(model_key, built, baseline):
    """A variable that keeps its name and loses its dims changes what readers get.

    Every consumer of a stored trace indexes by dimension -- the extractors, the
    summaries, the comparison suite -- so dims are part of the contract even
    when the values are not.
    """
    fingerprint = graph_fingerprint(built[model_key].model)
    assert fingerprint["dims"] == baseline[model_key]["dims"]
    assert fingerprint["coords"] == baseline[model_key]["coords"]


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_log_probability_at_a_fixed_point_is_unchanged(model_key, built, baseline):
    """One float that moves if any expression in the graph moves.

    The structural checks above would not notice a changed prior scale, a
    swapped operand or a dropped term: the names, order and dims would all still
    match. This would.
    """
    actual = fixed_point_logp(built[model_key].model)
    expected = baseline[model_key]["logp_at_fixed_point"]
    assert actual == pytest.approx(expected, rel=LOGP_RTOL), (
        f"{model_key}'s log probability at the fixed point moved: "
        f"{expected!r} -> {actual!r}. If this is a deliberate statistical "
        "change, regenerate the baseline and say in the commit what moved and "
        "why. If it is not, the refactor changed the model."
    )


def test_the_fingerprint_is_stable_across_two_builds(built):
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
                assert fixed_point_logp(rebuilt.model) == fixed_point_logp(
                    built[model_key].model
                )
    finally:
        patcher.undo()
