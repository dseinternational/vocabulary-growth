# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The exact prepared-frame hash, and the builders it is computed from (#266).

Every fit manifest records ``data.analysis_frame_hash``, but nothing read it
back: validation compared only the raw-CSV fingerprint, so a change to the
loader's *rules* left stale posteriors accepted as current. Reading it back
needs the prepared frame to be recomputable outside a fit, which is what
``vocab_growth.analysis_frames`` provides.

That buys a second construction of each engine's frame, so the load-bearing
test here is the drift guard: for every engine, the pure builder must produce
exactly the frame its ``prepare_*_data`` stage sets on the fit context. If the
two diverge, validation compares a hash of one frame against a fit of another
and either rejects every fit or accepts a stale one.
"""

import os

import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import pytest

from vocab_growth.analysis_frames import (
    FRAME_BUILDERS,
    analysis_frame_hash,
    build_analysis_frame,
    expected_analysis_frame_hash,
)
from vocab_growth.models.definitions import MODEL_REGISTRY

# One representative model per engine. The builders are per-engine, so covering
# every registered model would re-test the same code paths at several minutes
# a run; the registry-coverage test below is what stops a new model slipping
# through without a builder.
ENGINE_REPRESENTATIVES = [
    ("vg01", "vocab_growth.models.common", "prepare_univariate_data"),
    ("vg05", "vocab_growth.models.common_bivariate", "prepare_bivariate_data"),
    ("vg07", "vocab_growth.models.common_bivariate_re", "prepare_bivariate_re_data"),
    ("vg11", "vocab_growth.models.common_univariate_re", "prepare_univariate_re_data"),
    ("vg14", "vocab_growth.models.common_trivariate", "prepare_trivariate_data"),
    ("vg15", "vocab_growth.models.common_joint_modality", "prepare_joint_data"),
]


def test_every_registered_model_has_a_frame_builder():
    """A model with no builder cannot have its fitted output validated."""
    missing = sorted(set(MODEL_REGISTRY) - set(FRAME_BUILDERS))
    assert not missing, (
        f"Models with no registered analysis-frame builder: {missing}. "
        "Add their engine's builder to FRAME_BUILDERS."
    )
    unknown = sorted(set(FRAME_BUILDERS) - set(MODEL_REGISTRY))
    assert not unknown, f"FRAME_BUILDERS names unregistered models: {unknown}."


def test_an_unregistered_model_is_refused_rather_than_guessed():
    with pytest.raises(KeyError, match="No analysis-frame builder"):
        build_analysis_frame("vg99", MODEL_REGISTRY["vg01"])


@pytest.mark.parametrize("model_key", sorted(MODEL_REGISTRY))
def test_each_builder_belongs_to_the_engine_its_model_actually_uses(model_key):
    """A model moved between engines must not keep the old engine's builder.

    The mapping is keyed by model rather than by definition class, because the
    engine choice lives in each ``model_vgNN`` module (VG05 and VG07 share a
    definition class on different engines). That makes it possible for the two
    to drift silently, so the module's own import is what this checks against.
    """
    import importlib

    module = importlib.import_module(f"vocab_growth.models.model_{model_key}")
    engine_module = FRAME_BUILDERS[model_key].partition(":")[0]
    imported = {
        value.__module__
        for value in vars(module).values()
        if callable(value) and getattr(value, "__module__", None)
    }
    assert engine_module in imported, (
        f"{model_key} is mapped to {engine_module}, but model_{model_key}.py "
        f"imports from {sorted(m for m in imported if 'common' in m)}."
    )


@pytest.mark.parametrize("model_key", ["vg01", "vg07", "vg11"])
def test_the_frame_hash_is_stable_across_rebuilds(model_key):
    """A hash that moved between two loads could never validate anything.

    This is what the deterministic row order in the loader exists for: the
    queries carry no ``ORDER BY``, so without it the hash follows the DuckDB
    scan order rather than the data.
    """
    definition = MODEL_REGISTRY[model_key]
    assert expected_analysis_frame_hash(
        model_key, definition
    ) == expected_analysis_frame_hash(model_key, definition)


def test_the_hash_moves_when_the_frame_moves():
    """A hash insensitive to row order would not catch a reordering rule."""
    frame, _ = build_analysis_frame("vg01", MODEL_REGISTRY["vg01"])
    reordered = frame.iloc[::-1]
    assert analysis_frame_hash(frame) != analysis_frame_hash(reordered)

    relabelled = frame.rename(columns={frame.columns[0]: "renamed"})
    assert analysis_frame_hash(frame) != analysis_frame_hash(relabelled)


def _context_for(tmp_path, model_key):
    """A fit context sufficient for a data-preparation stage to run into."""
    from vocab_growth.models.common import ModelFitContext

    context = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=f"TEST_{model_key.upper()}_FRAME",
            config_name="test",
            output_root_dir=str(tmp_path),
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration("test"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    return context


@pytest.mark.parametrize(
    ("model_key", "module_name", "prepare_name"),
    ENGINE_REPRESENTATIVES,
    ids=[key for key, _, _ in ENGINE_REPRESENTATIVES],
)
def test_the_builder_matches_the_prepare_stage(
    tmp_path, model_key, module_name, prepare_name
):
    """The drift guard: two constructions of one frame must not diverge.

    ``write_fit_manifest`` hashes ``context.analysis_df`` immediately after the
    data-preparation stage, so the frame this test compares against is exactly
    the one a fit records.
    """
    import importlib

    module = importlib.import_module(module_name)
    definition = MODEL_REGISTRY[model_key]

    context = _context_for(tmp_path, model_key)
    getattr(module, prepare_name)(context, definition)

    built, _ = build_analysis_frame(model_key, definition)
    assert analysis_frame_hash(built) == analysis_frame_hash(context.analysis_df), (
        f"{model_key}: the pure builder and {prepare_name} disagree, so a "
        "recomputed hash could never match a recorded one."
    )
