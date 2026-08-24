# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Simulating under one definition and refitting under another (issue #226).

VG15's recovery shows ``psi`` and ``psi_study`` biased low through an
underestimated ``tau_psi``, and the issue asks whether ``tau_psi ~
HalfNormal(1.0)`` is a cause. The harness could not answer that: ``--variant``
substituted one definition for *both* the simulation and the refit, so moving
the prior moved the truth with it and the comparison was never controlled.
These tests pin the seam that separates the two roles, and -- more importantly
-- pin that a cross-definition run cannot be mistaken for, or overwrite, a
self-recovery one.
"""

import dataclasses

import pytest

from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.recovery.refit import make_recovery_definition, recovery_fit_dir
from vocab_growth.recovery.simulate import (
    recovery_config_name,
    simulation_dir,
    truth_source_tag,
)
from vocab_growth.sensitivity.registry import build_variant


@pytest.fixture
def record():
    return MODEL_REGISTRY["vg15"]


@pytest.fixture
def narrow():
    return build_variant("vg15", "tau-psi-narrow")[0]


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------


def test_a_self_recovery_run_keeps_the_name_it_always_had(record):
    """Existing recovery output must not be renamed by the seam's arrival."""
    assert recovery_config_name(record, 1) == f"{record.config_name}-recovery-r01"
    assert (
        recovery_config_name(record, 1, truth_definition=record)
        == f"{record.config_name}-recovery-r01"
    )


def test_a_cross_definition_run_is_named_for_the_truth_it_chases(record, narrow):
    name = recovery_config_name(narrow, 2, truth_definition=record)
    assert name == f"{narrow.config_name}-under-record-recovery-r02"
    assert name != recovery_config_name(narrow, 2)


def test_the_truth_tag_names_the_tokens_the_truth_carries_beyond_the_shared_base(
    record, narrow
):
    broad = build_variant("vg15", "tau-psi-wide")[0]
    # The record is the variant's own base, so nothing is left over.
    assert truth_source_tag(record, narrow) == "record"
    # Two siblings share their base *and* the tokens their suffixes share, so
    # the tag is the part that actually distinguishes them. That is what makes
    # it readable in a directory listing -- "-narrow-under-wide" -- and it stays
    # unique, because two truths with the same remainder against the same fit
    # name would have to be the same definition.
    assert truth_source_tag(broad, narrow) == "wide"
    # And it is not symmetric: read it as "whose truth", not "what differs".
    assert truth_source_tag(narrow, record) == "tau-psi-narrow"


def test_the_two_runs_cannot_land_in_the_same_directory(tmp_path, record, narrow):
    """The safety property. A cross-definition fit answers a different question
    from self-recovery, and scoring one as the other would be silent."""
    root = str(tmp_path)
    self_recovery = recovery_fit_dir("vg15", 1, root, definition=narrow)
    cross = recovery_fit_dir(
        "vg15", 1, root, definition=narrow, truth_definition=record
    )
    assert self_recovery != cross


def test_the_simulation_directory_stays_keyed_on_the_definition_that_made_it(
    tmp_path, record, narrow
):
    """So a cross-definition refit reuses the record's existing simulated data
    rather than demanding a fresh, separately-seeded copy of it."""
    root = str(tmp_path)
    assert simulation_dir(record, 1, root) == simulation_dir(record, 1, root)
    assert simulation_dir(record, 1, root) != simulation_dir(narrow, 1, root)


# --------------------------------------------------------------------------
# What gets fitted
# --------------------------------------------------------------------------


def test_the_fitted_definition_is_the_one_that_is_fitted(record, narrow):
    """The seam is only worth having if the priors that reach the graph are the
    refit's, not the simulation's."""
    built = make_recovery_definition(narrow, 1, truth_definition=record)
    assert built.tau_psi_sigma == narrow.tau_psi_sigma
    assert built.tau_psi_sigma != record.tau_psi_sigma
    assert built.model_id == narrow.model_id


def test_the_banner_says_which_truth_a_cross_run_is_chasing(record, narrow):
    built = make_recovery_definition(narrow, 1, truth_definition=record)
    assert "under VG15" in built.banner
    assert record.config_name in built.banner


def test_a_self_recovery_banner_is_unchanged(record):
    with_truth = make_recovery_definition(record, 1, truth_definition=record)
    without = make_recovery_definition(record, 1)
    assert with_truth.banner == without.banner
    assert with_truth.config_name == without.config_name


def test_make_recovery_definition_still_rejects_a_zero_replicate(record):
    with pytest.raises(ValueError, match="1-based"):
        make_recovery_definition(record, 0)


# --------------------------------------------------------------------------
# The provenance guard the seam must not weaken
# --------------------------------------------------------------------------


def test_the_simulation_guard_still_checks_the_definition_that_made_the_frame(
    tmp_path, record, narrow
):
    """The seam relaxes *which model is fitted*, not *whether the data are the
    data they claim to be*. A frame simulated from the record must still be
    refused if the record itself has since moved."""
    import json

    import numpy as np
    import pandas as pd
    import xarray as xr

    from vocab_growth.fit_artifacts import normalise_for_json
    from vocab_growth.recovery.simulate import (
        SIMULATION_FILENAME,
        SYNTHETIC_FRAME_FILENAME,
        TRUTH_FILENAME,
        load_simulation,
    )

    pd.DataFrame({"age": [12.0, 24.0]}).to_parquet(
        tmp_path / SYNTHETIC_FRAME_FILENAME
    )
    xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset(
                {"psi": (("chain", "draw"), np.ones((1, 1)))}
            )
        }
    ).to_netcdf(tmp_path / TRUTH_FILENAME)
    (tmp_path / SIMULATION_FILENAME).write_text(
        json.dumps(
            {
                "simulation": {
                    "definition": normalise_for_json(record),
                    "truth_source": "posterior",
                    "frame_file": SYNTHETIC_FRAME_FILENAME,
                }
            }
        )
    )

    # The refit's own definition is irrelevant to this check ...
    frame, _truth, _record = load_simulation(
        str(tmp_path), expected_definition=record
    )
    assert len(frame) == 2

    # ... but the simulating definition having moved is still fatal.
    moved = dataclasses.replace(record, tau_psi_sigma=record.tau_psi_sigma + 1.0)
    with pytest.raises(ValueError, match="tau_psi_sigma"):
        load_simulation(str(tmp_path), expected_definition=moved)


# --------------------------------------------------------------------------
# The CLI pairing
# --------------------------------------------------------------------------


def _resolve_pair(*args):
    import importlib.util
    import sys

    if "fit_recovery_cli" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "fit_recovery_cli", "scripts/fit_recovery.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["fit_recovery_cli"] = module
        spec.loader.exec_module(module)
    return sys.modules["fit_recovery_cli"]._resolve_pair(*args)


def test_the_cli_refuses_a_fit_variant_that_is_the_simulating_one():
    with pytest.raises(ValueError, match="self-recovery run"):
        _resolve_pair("vg15", "tau-psi-narrow", "tau-psi-narrow")


def test_the_cli_pairs_the_record_with_a_variant_and_labels_it():
    truth, fitted, label = _resolve_pair("vg15", None, "tau-psi-narrow")
    assert truth is MODEL_REGISTRY["vg15"]
    assert fitted.tau_psi_sigma == 0.3
    assert label == "vg15-tau-psi-narrow-under-record"


def test_the_cli_pairs_a_variant_with_the_record_the_other_way_round():
    truth, fitted, label = _resolve_pair("vg15", "tau-psi-narrow", "record")
    assert truth.tau_psi_sigma == 0.3
    assert fitted is MODEL_REGISTRY["vg15"]
    assert label == "vg15-under-tau-psi-narrow"


def test_the_cli_leaves_a_plain_run_alone():
    truth, fitted, label = _resolve_pair("vg15", None, None)
    assert truth is fitted is MODEL_REGISTRY["vg15"]
    assert label == "vg15"
