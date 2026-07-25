# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the parameter-recovery engine specifications (issue #163).

Data-free, so these run in CI. The load-bearing guards here are the ones that
catch *drift*: a recovery spec restates a model's cross-tab cell order and its
simulation order, and if either silently stops matching the engine the harness
would generate data under one decomposition and fit it under another. The
resulting "recovery failure" would look like a model defect rather than a harness
defect, which is the worst possible failure mode for validity evidence.
"""

import os

import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery import spec as recovery_spec
from vocab_growth.recovery.spec import (
    HEADLINE_MODELS,
    CompositionOutcome,
    CountOutcome,
    outcome_column,
    recovery_target,
    supported_models,
)


def test_headline_models_are_all_supported():
    # The three models issue #163 gates must be runnable, or the harness does not
    # answer the question it exists to answer.
    for key in HEADLINE_MODELS:
        assert key in supported_models()
        assert recovery_target(key).spec is not None


def test_supported_models_are_registered_and_keyed_consistently():
    for key in supported_models():
        assert key in MODEL_REGISTRY
        assert recovery_target(key).model_key == key


def test_unsupported_model_explains_itself():
    # VG16 is excluded on substance (outcome-dependent cross-lag design), so the
    # error has to say why rather than looking like an oversight.
    with pytest.raises(KeyError, match="cross-lag"):
        recovery_target("vg16")


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError, match="Unknown model"):
        recovery_target("vg99")


def test_every_nested_link_child_is_simulated_after_its_parent():
    """A child outcome must be drawn in a strictly later round than its parent.

    This is the whole reason the simulator has rounds. If a spec ever placed a
    nested child in the same round as its parent, the child would be drawn against
    the *previous* parent's denominators.
    """
    for key in supported_models():
        spec = recovery_target(key).spec
        definition = MODEL_REGISTRY[key]
        round_of = {}
        for index, stage in enumerate(spec.stages):
            for node in stage:
                if isinstance(node, CountOutcome):
                    round_of[outcome_column(definition, node.column)] = index
        for link in spec.nested_links:
            assert link.parent_column in round_of, (
                f"{key}: nested parent {link.parent_column} is never simulated"
            )
            assert link.child_column in round_of, (
                f"{key}: nested child {link.child_column} is never simulated"
            )
            assert round_of[link.child_column] > round_of[link.parent_column], (
                f"{key}: {link.child_column} must be simulated after "
                f"{link.parent_column}"
            )


def test_cross_tab_totals_tracking_a_parent_are_simulated_parents():
    for key in supported_models():
        spec = recovery_target(key).spec
        definition = MODEL_REGISTRY[key]
        simulated = {
            outcome_column(definition, node.column)
            for stage in spec.stages
            for node in stage
            if isinstance(node, CountOutcome)
        }
        for total_column, parent_column in spec.totals_tracking_parent:
            assert parent_column in simulated, (
                f"{key}: {total_column} tracks {parent_column}, which is never simulated"
            )


def test_joint_cross_tab_cell_order_matches_the_engine():
    """The four-cell and produced-cell orders must match the engine's own stacks.

    The spec names the cell columns positionally, and the engine stacks its cell
    probabilities in a fixed order. A reordering on either side would permute the
    simulated cells relative to the probabilities that generated them.
    """
    from vocab_growth.models import common_joint_modality as joint

    compositions = {
        node.rv_name: node
        for stage in recovery_spec.JOINT_SPEC.stages
        for node in stage
        if isinstance(node, CompositionOutcome)
    }

    # uk_02 within-understood four cells: the engine reads these columns in this
    # order and stacks [neither, sign_only, speak_only, both] against them.
    assert compositions["cells_obs"].columns == (
        "understood_only",
        "signed_only",
        "spoken_only",
        "signed_spoken",
    )
    assert joint.CELL_NAMES == ["neither", "sign_only", "speak_only", "both"]

    # nz_01 within-produced three cells: read straight from the engine constant.
    assert list(compositions["nz_prod_cells_obs"].columns) == joint.PROD_CELL_COLUMNS
    assert joint.PROD_CELL_NAMES == ["sign_only", "speak_only", "both"]


def test_univariate_outcome_placeholder_resolves_per_definition():
    # The univariate engine fits whichever single outcome its definition names, so
    # its spec cannot hard-code a column: VG11 is spoken, VG12 understood.
    assert outcome_column(MODEL_REGISTRY["vg11"], "__outcome__") == "spoken"
    assert outcome_column(MODEL_REGISTRY["vg12"], "__outcome__") == "understood"
    assert outcome_column(MODEL_REGISTRY["vg12"], "signed") == "signed"


def test_engine_stage_factories_resolve_and_start_with_data_preparation():
    """Each engine must expose a stage list whose first stage is the one replaced.

    The refit substitutes stage 0 with the synthetic-data loader; if an engine's
    stage order changed, the substitution would silently replace the wrong stage.
    """
    from vocab_growth.recovery.simulate import (
        BUILD_STAGE_NAME,
        PREPARE_STAGE_NAME,
        PRIORS_STAGE_NAME,
    )

    for key in supported_models():
        target = recovery_target(key)
        stages = target.resolve_stages(MODEL_REGISTRY[key])
        names = [name for name, _ in stages]
        assert names[0] == PREPARE_STAGE_NAME, f"{key}: unexpected first stage {names[0]!r}"
        assert PRIORS_STAGE_NAME in names, key
        assert BUILD_STAGE_NAME in names, key
        # The build must come after the priors, which come after preparation.
        assert names.index(PRIORS_STAGE_NAME) < names.index(BUILD_STAGE_NAME), key


def test_available_replicates_discovers_written_simulations(tmp_path):
    """The recovery matrix must cover every replicate on disk, not just the last run.

    ``--compare-only`` for one replicate of a staged run would otherwise rewrite
    the matrix with a single row and silently drop the rest.
    """
    from vocab_growth.recovery.simulate import (
        SIMULATION_FILENAME,
        available_replicates,
        simulation_dir,
    )

    definition = MODEL_REGISTRY["vg10"]
    root = str(tmp_path)
    assert available_replicates(definition, root) == []

    for replicate in (1, 3, 2):
        directory = simulation_dir(definition, replicate, root)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, SIMULATION_FILENAME), "w") as handle:
            handle.write("{}")

    assert available_replicates(definition, root) == [1, 2, 3]

    # A directory without a written simulation record is not a replicate.
    partial = simulation_dir(definition, 4, root)
    os.makedirs(partial, exist_ok=True)
    assert available_replicates(definition, root) == [1, 2, 3]

    # Another model's replicates are not counted.
    other = simulation_dir(MODEL_REGISTRY["vg12"], 9, root)
    os.makedirs(other, exist_ok=True)
    with open(os.path.join(other, SIMULATION_FILENAME), "w") as handle:
        handle.write("{}")
    assert available_replicates(definition, root) == [1, 2, 3]
    assert available_replicates(MODEL_REGISTRY["vg12"], root) == [9]
