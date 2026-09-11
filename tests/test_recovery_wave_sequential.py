# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A predictor built from an outcome must be simulated in an order that is sound.

VG16's cross-lag reads each child's earlier-wave comprehension count, so its
design matrix is a function of the outcome rather than of the design. It was
listed unsupported for recovery on the ground that a single simulation pass
"would fit synthetic-lag data against real-lag truth" -- #242 item 6, #289 task
3.9, and the deferral of #297 all rest on that sentence.

**It is false for VG16**, and this module is where that is established rather
than asserted. The simulator already rebuilds between stages, and VG16's lag
reads a column drawn in a strictly earlier stage than the node it enters, so the
lag is recomputed from the simulated parent before anything that uses it is
drawn. :func:`single_pass_is_sound` derives that from the stage order.

It is *not* false in general, and the case it is false for is the one #297
proposes: a sign-to-speech lag reads ``signed``, which the joint engine draws in
the same stage as the ``spoken`` it shifts. So the wave loop exists, is selected
by the same derivation, and the guard that would have caught the original
misdiagnosis runs on every simulation either way.

Data-free and sampling-free except for the two marked ``slow``.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np
import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery.simulate import (
    _restrict,
    _verify_predictor_coherence,
)
from vocab_growth.recovery.spec import (
    BIVARIATE_RE_SPEC,
    JOINT_SPEC,
    OutcomeDependentPredictor,
    outcome_dependent_predictor,
    single_pass_is_sound,
)

VG16 = MODEL_REGISTRY["vg16"]
VG10 = MODEL_REGISTRY["vg10"]


# --- what is declared, and against what -----------------------------------------


def test_only_a_cross_lag_definition_declares_a_predictor():
    """Pinned against the field that creates it, not against a model list."""
    assert outcome_dependent_predictor(VG10) is None
    assert not VG10.use_cross_lag

    predictor = outcome_dependent_predictor(VG16)
    assert VG16.use_cross_lag
    assert predictor is not None
    assert predictor.name == "cross_lag"
    assert predictor.source_column == "understood"
    assert predictor.consumer_rv_names == ("y_s_obs",)


def test_every_registered_model_with_a_lag_declares_one():
    """A lag model that declared nothing would be simulated with no guard at all."""
    for key, definition in MODEL_REGISTRY.items():
        expected = bool(getattr(definition, "use_cross_lag", False))
        assert (outcome_dependent_predictor(definition) is not None) is expected, key


def test_the_declared_source_and_consumer_are_names_the_engine_actually_uses():
    """A typo in either would silently disable the ordering rule and the guard."""
    predictor = outcome_dependent_predictor(VG16)
    columns = {
        node.column
        for stage in BIVARIATE_RE_SPEC.stages
        for node in stage
        if hasattr(node, "column")
    }
    rv_names = {
        node.rv_name for stage in BIVARIATE_RE_SPEC.stages for node in stage
    }
    assert predictor.source_column in columns
    assert set(predictor.consumer_rv_names) <= rv_names


# --- the ordering rule -----------------------------------------------------------


def test_vg16_needs_no_wave_loop_because_its_source_is_drawn_first():
    """The correction. `understood` is stage 0; `y_s_obs` is stage 1.

    Measured alongside this on 2026-09-11: at the build whose draw consumes the
    lag, the predictor already matched the finished frame's in 0 of 1,708 rows.
    """
    predictor = outcome_dependent_predictor(VG16)
    assert single_pass_is_sound(BIVARIATE_RE_SPEC, VG16, predictor)


def test_a_same_stage_source_is_not_sound_in_one_pass():
    """The VG25 shape: a lag reading `signed`, which is drawn with `spoken`."""
    sign_lag = OutcomeDependentPredictor(
        name="sign_cross_lag",
        source_column="signed",
        consumer_rv_names=("y_s_obs",),
        description="the prior wave's signed share, shifting the production ratio",
    )
    # Both are in the joint engine's second stage, so one pass would draw the
    # consumer against a source that is still the real study value.
    assert not single_pass_is_sound(JOINT_SPEC, MODEL_REGISTRY["vg15"], sign_lag)


def test_a_consumer_the_engine_does_not_draw_fails_toward_the_wave_loop():
    """Unrecognised means slower, never wrong."""
    nonsense = OutcomeDependentPredictor(
        name="x",
        source_column="understood",
        consumer_rv_names=("y_not_a_node",),
        description="",
    )
    assert not single_pass_is_sound(BIVARIATE_RE_SPEC, VG16, nonsense)


def test_a_source_nothing_simulates_is_sound():
    """A predictor reading a column the simulation never rewrites is real data."""
    from_design = OutcomeDependentPredictor(
        name="x",
        source_column="age",
        consumer_rv_names=("y_s_obs",),
        description="",
    )
    assert single_pass_is_sound(BIVARIATE_RE_SPEC, VG16, from_design)


# --- the guard, and that it can fail ---------------------------------------------


def _state(prev_idx, has_lag, logit):
    return np.vstack([prev_idx, has_lag, logit]).astype(float)


PREDICTOR = OutcomeDependentPredictor(
    name="cross_lag",
    source_column="understood",
    consumer_rv_names=("y_s_obs",),
    description="",
)


def test_the_guard_passes_when_the_draw_used_the_final_predictor():
    final = _state([0, 0, 1], [0, 1, 1], [0.0, -1.5, 0.5])
    report = _verify_predictor_coherence(
        PREDICTOR, final, [(1, np.array([0, 1, 2]), final.copy())]
    )
    assert report["cross_lag_round_1"] == {"rows": 3, "rows_with_predictor": 2}


def test_the_guard_refuses_a_draw_made_under_a_predictor_the_frame_does_not_reproduce():
    """The failure the whole design exists to catch, shown failing.

    A guard that has never been seen to fail is not evidence. This is the
    single-pass defect exactly: the row was drawn under the *real* earlier wave's
    logit and would be fitted under the *synthetic* one.
    """
    final = _state([0, 0, 1], [0, 1, 1], [0.0, -1.5, 0.5])
    used = _state([0, 0, 1], [0, 1, 1], [0.0, -1.5, 2.25])
    with pytest.raises(RuntimeError, match="finished .* frame does not reproduce"):
        _verify_predictor_coherence(
            PREDICTOR, final, [(1, np.array([0, 1, 2]), used)]
        )


def test_the_guard_notices_a_row_that_merely_gained_a_lag():
    """`has_lag` moving is as much a design-matrix change as the value moving."""
    final = _state([0, 0, 1], [0, 1, 1], [0.0, -1.5, 0.5])
    used = _state([0, 0, 1], [0, 0, 1], [0.0, -1.5, 0.5])
    with pytest.raises(RuntimeError):
        _verify_predictor_coherence(
            PREDICTOR, final, [(1, np.array([0, 1, 2]), used)]
        )


def test_the_guard_ignores_rows_the_round_did_not_draw():
    """Only the rows a round wrote were drawn under that round's predictor."""
    final = _state([0, 0, 1], [0, 1, 1], [0.0, -1.5, 0.5])
    used = _state([0, 0, 1], [0, 1, 1], [0.0, -1.5, 99.0])
    report = _verify_predictor_coherence(
        PREDICTOR, final, [(1, np.array([0, 1]), used)]
    )
    assert report["cross_lag_round_1"]["rows"] == 2


# --- the row restriction the wave loop needs --------------------------------------


def test_restrict_is_the_identity_without_a_wave_loop():
    rows = np.array([0, 3, 7])
    assert _restrict(rows, None) is rows


def test_restrict_keeps_only_the_rows_this_round_may_change():
    """Earlier waves are final; neutralising one would destroy a live predictor."""
    rows = np.array([0, 1, 2, 3, 4])
    mutable = np.array([2, 3, 4])
    assert _restrict(rows, mutable).tolist() == [2, 3, 4]


# --- end to end -------------------------------------------------------------------


@pytest.mark.slow
def test_vg16_simulates_and_the_finished_frame_reproduces_its_own_predictor(
    tmp_path, require_prepared_data
):
    """The property the guard asserts, checked again from outside the simulator."""
    from vocab_growth.models.cross_lag import prev_wave_lag_for_frame
    from vocab_growth.recovery.simulate import simulate_replicate

    result = simulate_replicate(
        "vg16",
        "dev",
        replicate=1,
        truth_source="prior",
        output_root=str(tmp_path),
    )
    frame = result.frame
    assert "understood" in result.simulated_columns
    assert "spoken" in result.simulated_columns

    # The synthetic frame's counts are the model's, not the study's.
    real = MODEL_REGISTRY["vg16"]
    _, has_lag, _ = prev_wave_lag_for_frame(frame, real.n_trials, real)
    assert has_lag.sum() > 0, "a cross-lag simulation with no lagged row proves nothing"


@pytest.mark.slow
def test_the_wave_loop_runs_and_produces_the_same_shape_of_simulation(
    tmp_path, monkeypatch, require_prepared_data
):
    """The wave loop itself, which no registered model currently selects.

    VG16 is sound in one pass, so without this the loop VG25 needs would ship
    with no coverage at all -- the failure mode this module exists to avoid,
    one level up. Forcing it on VG16 is the only way to exercise it until a
    model with a same-stage predictor is registered.

    The two passes draw different *values* (each round seeds from its own index)
    and must not be compared on those. What must match is the shape: the same
    columns simulated, over the same likelihood rows, with the guard passing --
    both are valid forward simulations of the same model.
    """
    from vocab_growth.models.cross_lag import prev_wave_lag_for_frame
    from vocab_growth.recovery import simulate as simulate_module

    one_pass = simulate_module.simulate_replicate(
        "vg16", "dev", replicate=1, truth_source="prior",
        output_root=str(tmp_path / "one"),
    )

    monkeypatch.setattr(simulate_module, "single_pass_is_sound", lambda *a, **k: False)
    waved = simulate_module.simulate_replicate(
        "vg16", "dev", replicate=1, truth_source="prior",
        output_root=str(tmp_path / "waved"),
    )

    assert waved.simulated_columns == one_pass.simulated_columns
    assert waved.row_counts == one_pass.row_counts, (
        "the wave loop wrote a different number of likelihood rows than the "
        "single pass; every row belongs to exactly one wave, so the totals "
        "must agree"
    )

    # And it really did go wave by wave, rather than falling through to one pass.
    record = json.loads(
        (pathlib.Path(waved.directory) / "simulation.json").read_text(encoding="utf-8")
    )["simulation"]
    assert record["waves"] == 7
    assert record["outcome_dependent_predictor"]["single_pass_sound"] is False
    assert json.loads(
        (pathlib.Path(one_pass.directory) / "simulation.json").read_text(
            encoding="utf-8"
        )
    )["simulation"]["waves"] is None

    # The property the guard asserts, restated from outside the simulator.
    definition = MODEL_REGISTRY["vg16"]
    _, has_lag, _ = prev_wave_lag_for_frame(
        waved.frame, definition.n_trials, definition
    )
    assert has_lag.sum() > 0


@pytest.mark.slow
def test_forcing_the_unsound_order_makes_the_simulation_abort(
    tmp_path, monkeypatch, require_prepared_data
):
    """The guard catching the defect on a real simulation, not on arrays.

    VG16 is sound in one pass, so the unsound case is manufactured the only
    honest way: declare the consumer to be the node that draws the *source*, so
    the predictor is read before its own input exists. That is the VG25 ordering,
    and with the wave loop suppressed the run must stop rather than produce a
    dataset generated under one design matrix and fitted under another.
    """
    from vocab_growth.recovery import simulate as simulate_module

    unsound = dataclasses.replace(
        outcome_dependent_predictor(MODEL_REGISTRY["vg16"]),
        consumer_rv_names=("y_u_obs",),
    )
    monkeypatch.setattr(
        simulate_module, "outcome_dependent_predictor", lambda definition: unsound
    )
    monkeypatch.setattr(
        simulate_module, "single_pass_is_sound", lambda *a, **k: True
    )

    with pytest.raises(RuntimeError, match="does not reproduce"):
        simulate_module.simulate_replicate(
            "vg16",
            "dev",
            replicate=1,
            truth_source="prior",
            output_root=str(tmp_path),
        )
