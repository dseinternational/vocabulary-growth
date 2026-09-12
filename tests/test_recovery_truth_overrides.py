# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Setting a parameter in a recovery truth draw.

Two of the three cells #297 check 4 asks for are parameter *settings* rather
than draws, and the same is true of #242 item 6 for VG16. What has to hold is
narrow and testable: the setting reaches the truth, everything downstream of it
is recomputed under it, a setting that cannot mean what it says is refused, and
a set truth can never be mistaken for -- or overwrite -- an unset one.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery.simulate import (
    recovery_config_name,
    simulation_dir,
)
from vocab_growth.recovery.truth_overrides import (
    STRUCTURAL_TRANSFORMS,
    TruthOverride,
    apply_truth_overrides,
    check_all_finite,
    override_tag,
    parse_truth_override,
    set_independent,
)

# ==========================================================================
# Reading a setting off the command line
# ==========================================================================


def test_a_number_and_a_transform_are_both_read_as_settings():
    assert parse_truth_override("beta_sign_lag=0") == TruthOverride(
        "beta_sign_lag", value=0.0
    )
    assert parse_truth_override(" beta_lag = -0.5 ") == TruthOverride(
        "beta_lag", value=-0.5
    )
    assert parse_truth_override("subject_re=independent") == TruthOverride(
        "subject_re", transform="independent"
    )


def test_an_unreadable_setting_names_the_transforms_rather_than_failing_bare():
    """The error has to say what *is* accepted; a bare parse failure does not."""
    with pytest.raises(ValueError, match="independent"):
        parse_truth_override("beta_sign_lag=nought")
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_truth_override("beta_sign_lag")
    with pytest.raises(ValueError, match="finite"):
        parse_truth_override("beta_sign_lag=inf")


def test_a_setting_carries_a_value_or_a_transform_and_never_both():
    with pytest.raises(ValueError):
        TruthOverride("beta_sign_lag")
    with pytest.raises(ValueError):
        TruthOverride("beta_sign_lag", value=0.0, transform="independent")


# ==========================================================================
# The setting is part of the run's identity
# ==========================================================================


def test_the_tag_is_the_same_however_the_settings_were_typed():
    """A cell names one directory, so the tag cannot depend on argument order."""
    first = parse_truth_override("beta_sign_lag=0")
    second = parse_truth_override("subject_re=independent")
    assert override_tag((first, second)) == override_tag((second, first))
    assert override_tag(()) == ""


def test_the_tag_spells_out_the_characters_a_config_name_uses():
    """`-` separates config-name tokens and `.` is awkward in a path."""
    assert override_tag((parse_truth_override("beta=-0.5"),)) == "set-beta-neg0p5"
    assert override_tag((parse_truth_override("beta=0.25"),)) == "set-beta-0p25"
    assert override_tag((parse_truth_override("beta=1.0"),)) == "set-beta-1"


def test_a_set_truth_cannot_land_in_or_be_scored_as_an_unset_one(tmp_path):
    """The `-under-` precedent (#226), for the same failure.

    Two cells of gate 4 differ only in what their truth was set to. If the
    setting did not reach the name, the second would simulate over the first's
    directory and be scored into the first's matrix -- one record where there
    are two.
    """
    definition = MODEL_REGISTRY["vg25"]
    overrides = (parse_truth_override("beta_sign_lag=0"),)

    plain = recovery_config_name(definition, 1)
    marked = recovery_config_name(definition, 1, truth_overrides=overrides)
    assert plain != marked
    assert marked.endswith("-recovery-r01")
    assert "set-beta_sign_lag-0" in marked

    root = str(tmp_path)
    assert simulation_dir(definition, 1, root) != simulation_dir(
        definition, 1, root, truth_overrides=overrides
    )


def test_no_settings_leaves_every_existing_name_exactly_as_it_was():
    """Fail-closed in the direction that matters for output already on disk."""
    definition = MODEL_REGISTRY["vg25"]
    assert (
        recovery_config_name(definition, 3, truth_overrides=())
        == f"{definition.config_name}-recovery-r03"
    )


# ==========================================================================
# The packed Cholesky transform
# ==========================================================================


def test_independent_keeps_the_scales_and_zeroes_the_correlations_on_arrays():
    """The arithmetic, separately from the graph that consumes it."""
    order = 3
    factor = np.array([[2.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-1.0, 0.5, 1.2]])
    packed = factor[np.tril_indices(order)]
    scales = np.linalg.norm(factor, axis=1)

    replacement, described = set_independent(packed, name="subject_re")

    rebuilt = np.zeros((order, order))
    rebuilt[np.tril_indices(order)] = replacement
    covariance = rebuilt @ rebuilt.T
    assert np.allclose(np.sqrt(np.diag(covariance)), scales)
    off_diagonal = covariance[~np.eye(order, dtype=bool)]
    assert np.array_equal(off_diagonal, np.zeros_like(off_diagonal))
    assert "identity" in described


def test_a_value_that_is_not_a_packed_triangle_is_refused():
    with pytest.raises(ValueError, match="lower triangle"):
        set_independent(np.zeros(4), name="subject_re")
    with pytest.raises(ValueError, match="one-dimensional"):
        set_independent(np.zeros((2, 3)), name="subject_re")


def test_every_registered_transform_is_reachable_from_the_command_line():
    """A transform nobody can type is a transform that does not exist."""
    for name in STRUCTURAL_TRANSFORMS:
        assert parse_truth_override(f"subject_re={name}").transform == name


# ==========================================================================
# Applying a setting to a real graph
# ==========================================================================


@pytest.fixture(scope="module")
def vg25_graph(tmp_path_factory):
    """VG25's graph and one prior draw of its free variables."""
    import pymc as pm
    from _pytest.monkeypatch import MonkeyPatch
    from support.synthetic_graphs import build_registered_model

    patcher = MonkeyPatch()
    try:
        context = build_registered_model(
            "vg25", output_dir=str(tmp_path_factory.mktemp("vg25")), monkeypatch=patcher
        )
    finally:
        patcher.undo()
    model = context.model
    free = [rv.name for rv in model.free_RVs]
    prior = pm.sample_prior_predictive(
        draws=2,
        model=model,
        var_names=free,
        random_seed=7,
        compile_kwargs={"mode": "FAST_COMPILE"},
    )
    posterior = _dataset(prior["prior"])[free].isel(chain=[0], draw=[1])
    posterior = posterior.assign_coords(
        chain=np.zeros(1, dtype=int), draw=np.zeros(1, dtype=int)
    )
    return model, posterior


def _dataset(node) -> xr.Dataset:
    return node.to_dataset() if isinstance(node, xr.DataTree) else node


def _deterministics(posterior, model, names):
    import pymc as pm

    computed = pm.compute_deterministics(
        posterior,
        model=model,
        var_names=list(names),
        progressbar=False,
        compile_kwargs={"mode": "FAST_COMPILE"},
    )
    return {
        name: float(np.asarray(_dataset(computed)[name].values).ravel()[0])
        for name in names
    }


def test_the_coefficient_is_set_to_what_was_asked_for(vg25_graph):
    model, posterior = vg25_graph
    updated, applied = apply_truth_overrides(
        posterior, model, (parse_truth_override("beta_sign_lag=0"),)
    )
    assert float(np.asarray(updated["beta_sign_lag"].values).ravel()[0]) == 0.0
    assert applied == [
        {
            "name": "beta_sign_lag",
            "setting": "0",
            "shape": [1, 1],
            "applied": "every entry set to 0",
        }
    ]
    # Nothing else moved: this is one cell of a gate, not a different truth.
    for name in posterior.data_vars:
        if name != "beta_sign_lag":
            assert np.array_equal(updated[name].values, posterior[name].values)


def test_independent_zeroes_vg25s_three_correlations_and_keeps_its_three_scales(
    vg25_graph,
):
    """The `(beta != 0, rho = 0)` cell, end to end on the model it is for.

    `rho_sign_q` is what VG25 has to be told apart from, and the model does not
    sample it -- it samples the packed Cholesky factor `subject_re` and reads
    all three correlations and all three scales off it. So the check is that the
    correlations come back exactly zero *and* the scales come back unchanged:
    a transform that moved the scales would be a differently-scaled model rather
    than the same model with the correlation switched off.
    """
    model, posterior = vg25_graph
    names = (
        "tau_subj_u",
        "tau_subj_q",
        "tau_subj_sign",
        "rho_uq",
        "rho_u_sign",
        "rho_sign_q",
    )
    before = _deterministics(posterior, model, names)
    assert any(abs(before[name]) > 1e-6 for name in names[3:]), (
        "the drawn truth has no correlation to switch off, so this proves nothing"
    )

    updated, applied = apply_truth_overrides(
        posterior, model, (parse_truth_override("subject_re=independent"),)
    )
    after = _deterministics(updated, model, names)

    assert [after[name] for name in names[3:]] == [0.0, 0.0, 0.0]
    for name in names[:3]:
        assert after[name] == before[name]
    assert applied[0]["applied"].startswith("3x3 correlation set to the identity")


def test_setting_a_deterministic_says_what_to_set_instead(vg25_graph):
    """The trap the whole design turns on.

    `rho_sign_q` is the name a reader of #297 has in mind, and setting it would
    be silently undone when the truth's deterministics are recomputed from the
    graph. Refusing it is only half the job; the message has to name the free
    variable that does carry it.
    """
    model, posterior = vg25_graph
    with pytest.raises(ValueError, match="deterministic") as raised:
        apply_truth_overrides(
            posterior, model, (parse_truth_override("rho_sign_q=0"),)
        )
    assert "subject_re=independent" in str(raised.value)


def test_an_unknown_name_lists_the_free_variables(vg25_graph):
    model, posterior = vg25_graph
    with pytest.raises(ValueError, match="not a free variable") as raised:
        apply_truth_overrides(posterior, model, (parse_truth_override("beta_lag=0"),))
    assert "beta_sign_lag" in str(raised.value)


def test_a_free_variable_absent_from_the_draw_is_a_disagreement_not_a_skip(vg25_graph):
    model, posterior = vg25_graph
    with pytest.raises(ValueError, match="absent from the truth draw"):
        apply_truth_overrides(
            posterior.drop_vars("beta_sign_lag"),
            model,
            (parse_truth_override("beta_sign_lag=0"),),
        )


def test_nothing_is_touched_when_there_are_no_settings(vg25_graph):
    model, posterior = vg25_graph
    updated, applied = apply_truth_overrides(posterior, model, ())
    assert applied == []
    assert updated is posterior


# ==========================================================================
# The finiteness guard
# ==========================================================================


def test_a_setting_that_makes_a_reported_quantity_non_finite_is_refused():
    """Where a boundary setting shows up first, and what it must not do.

    A scale set to zero reaches a report through a division or a logarithm long
    before it reaches the sampler. Scoring against a truth holding a NaN is the
    failure this stops -- it would read as a coverage miss rather than as a bad
    setting.
    """
    posterior = xr.Dataset(
        {
            "fine": ("draw", np.array([1.0])),
            "broken": ("draw", np.array([np.nan])),
        }
    )
    with pytest.raises(ValueError, match="broken"):
        check_all_finite(posterior, context="The truth settings")
    check_all_finite(posterior[["fine"]], context="The truth settings")
