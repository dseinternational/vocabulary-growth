# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What makes a fit the same fit (issue #273).

Fitted output is validated by comparing the definition recorded in its manifest
against the one registered today, and that comparison was raw dictionary
equality. One consequence has shaped the model API more than any statistical
consideration: **adding a field with a default invalidates every historical fit
of that dataclass**, even when the default reproduces exactly what those fits
did. VG19's child slope and Proposal A1's age-varying scale arrive through a
scalar field holding an object because of it; VG20's correlation and VG22's
factor live on sibling subclasses because of it.

The comparison is now field by field through a classified, versioned payload.
These tests pin the two properties that make that safe to rely on: the
classification is **complete** over the registry, and the comparison **fails
closed** — every difference is still an error, and the only excuse is an
explicit, justified backfill entry.
"""

from __future__ import annotations

import dataclasses

import pytest

from vocab_growth.fit_artifacts import normalise_for_json
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG10,
    VG19,
    VG20,
    VG22,
)
from vocab_growth.models.fit_identity import (
    BACKFILL_DEFAULTS,
    FIELD_ROLES,
    SEMANTIC_SCHEMA_VERSION,
    FieldRole,
    definition_differences,
    is_classified,
    role_of,
    semantic_payload,
)

_MODEL_KEYS = sorted(MODEL_REGISTRY)


# --- the classification is complete ---------------------------------------------


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_every_field_of_every_registered_model_is_classified(model_key):
    """An unclassified field is treated as graph-affecting, which is safe but silent.

    Refused outright here so the decision is made when the field is added rather
    than defaulted into.
    """
    definition = MODEL_REGISTRY[model_key]
    unclassified = sorted(
        item.name
        for item in dataclasses.fields(definition)
        if not is_classified(item.name)
    )
    assert not unclassified, (
        f"{model_key} has unclassified definition fields: {unclassified}. Add "
        "each to FIELD_ROLES (or, for a prior family, confirm it matches one of "
        "the graph prefixes) so a validation failure can say what kind of thing "
        "moved."
    )


def test_an_unclassified_field_falls_back_to_the_strictest_role():
    assert role_of("a_field_nobody_classified") is FieldRole.GRAPH


def test_no_classification_names_a_field_no_model_has():
    """A stale entry would describe a field that no longer exists."""
    live = set()
    for definition in MODEL_REGISTRY.values():
        live |= {item.name for item in dataclasses.fields(definition)}
    stale = sorted(set(FIELD_ROLES) - live)
    assert not stale, f"FIELD_ROLES names fields no registered model has: {stale}"


def test_the_roles_partition_the_fields():
    """Each field has exactly one role, so a payload cannot record it twice."""
    payload = semantic_payload(VG10)
    groups = [set(payload[role.value]) for role in FieldRole]
    for index, group in enumerate(groups):
        for other in groups[index + 1 :]:
            assert not group & other


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_the_payload_covers_the_definition_exactly(model_key):
    definition = MODEL_REGISTRY[model_key]
    payload = semantic_payload(definition)
    assert payload["schema_version"] == SEMANTIC_SCHEMA_VERSION
    covered = {
        name for role in FieldRole for name in payload[role.value]
    }
    assert covered == {item.name for item in dataclasses.fields(definition)}


def test_identity_holds_what_names_the_fit():
    payload = semantic_payload(VG10)
    assert set(payload["identity"]) == {"model_id", "config_name", "banner"}


def test_the_child_effect_structures_are_graph_affecting():
    """Each of these selects a different model, so none may be excusable."""
    for name in (
        "subject_re_correlation_eta",
        "subject_factor",
        "subject_variance_partition",
        "tau_subj_u_sigma",
        "subject_slope_ref_age_months",
    ):
        assert role_of(name) is FieldRole.GRAPH, name


def test_the_data_admission_switches_are_data_affecting():
    for name in (
        "population",
        "td_languages",
        "include_implausible_production",
        "include_same_day_disagreements",
        "exclude_us01_spoken_ceiling",
        "one_observation_per_subject",
        # Seeds the reproducible subsample, so it selects rows.
        "random_seed",
    ):
        assert role_of(name) is FieldRole.DATA, name


# --- the comparison fails closed ------------------------------------------------


@pytest.mark.parametrize("model_key", _MODEL_KEYS)
def test_a_definition_matches_its_own_record(model_key):
    definition = MODEL_REGISTRY[model_key]
    assert definition_differences(normalise_for_json(definition), definition) == []


def test_a_changed_prior_is_a_graph_difference():
    altered = dataclasses.replace(VG10, eta_u_sigma=0.61)
    (difference,) = definition_differences(normalise_for_json(VG10), altered)
    assert difference.field == "eta_u_sigma"
    assert difference.role is FieldRole.GRAPH


def test_a_changed_query_grid_is_a_reporting_difference_and_still_fatal():
    """The posterior is unaffected; the stored query outputs are not."""
    altered = dataclasses.replace(VG10, ages_query=VG10.ages_query[:-1])
    (difference,) = definition_differences(normalise_for_json(VG10), altered)
    assert difference.field == "ages_query"
    assert difference.role is FieldRole.REPORTING


def test_a_field_the_record_lacks_is_a_difference_without_a_backfill_entry():
    """The behaviour the backfill mechanism exists to make *changeable*.

    VG20 adds `subject_re_correlation_eta` to VG10's shape. Without an entry
    saying what its absence meant, a VG10-era record does not satisfy VG20.
    """
    recorded = normalise_for_json(VG10)
    differences = definition_differences(recorded, VG20)
    fields = {difference.field for difference in differences}
    assert "subject_re_correlation_eta" in fields


def test_a_backfill_entry_excuses_exactly_the_stated_value(monkeypatch):
    """And nothing else: the entry is a claim about one value, not a waiver."""
    import vocab_growth.models.fit_identity as fit_identity

    recorded = normalise_for_json(VG10)
    recorded.pop("dse_native_only")

    # Without an entry, the missing field is a difference.
    assert any(
        difference.field == "dse_native_only"
        for difference in definition_differences(recorded, VG10)
    )

    monkeypatch.setitem(
        fit_identity.BACKFILL_DEFAULTS, "dse_native_only", VG10.dse_native_only
    )
    assert definition_differences(recorded, VG10) == []

    # A definition carrying a *different* value is still a difference: the entry
    # says what the absence meant, not that the field may be anything.
    altered = dataclasses.replace(VG10, dse_native_only=not VG10.dse_native_only)
    assert any(
        difference.field == "dse_native_only"
        for difference in definition_differences(recorded, altered)
    )


def test_the_same_day_backfill_entry_is_the_loaders_own_default():
    """The third entry's claim, checked rather than asserted (#289 task 4.3).

    Every fit made before `include_same_day_disagreements` existed called the
    loader without that argument, so it ran at the loader's declared default.
    The entry is only true while it equals that default -- read off the
    signature, not restated here -- and only for the classes whose engines
    forward the field. A record without the field must validate against the
    registered definition, and a variant that sets the field must not.
    """
    import inspect

    from vocab_growth.data_utils import load_combined_data, load_data
    from vocab_growth.models.definitions import VG15

    for loader in (load_data, load_combined_data):
        parameter = inspect.signature(loader).parameters["include_same_day_disagreements"]
        assert parameter.default is BACKFILL_DEFAULTS["include_same_day_disagreements"]

    for definition in (VG10, VG15):
        recorded = normalise_for_json(definition)
        recorded.pop("include_same_day_disagreements")
        assert definition_differences(recorded, definition) == []
        altered = dataclasses.replace(definition, include_same_day_disagreements=True)
        (difference,) = definition_differences(recorded, altered)
        assert difference.field == "include_same_day_disagreements"
        assert difference.role is FieldRole.DATA


def test_backfill_entries_name_real_fields():
    live = set()
    for definition in MODEL_REGISTRY.values():
        live |= {item.name for item in dataclasses.fields(definition)}
    stale = sorted(set(BACKFILL_DEFAULTS) - live)
    assert not stale, f"BACKFILL_DEFAULTS names fields no model has: {stale}"


def test_a_field_the_registry_no_longer_has_is_a_difference():
    """A retired field left in an old record must not pass silently."""
    recorded = normalise_for_json(VG10)
    recorded["a_field_since_removed"] = 1
    (difference,) = definition_differences(recorded, VG10)
    assert difference.field == "a_field_since_removed"
    assert "no longer a field" in difference.reason


def test_a_missing_definition_is_a_difference_rather_than_a_pass():
    for absent in (None, "", [], 0):
        differences = definition_differences(absent, VG10)
        assert differences and differences[0].field == "<definition>"


def test_the_structural_variants_each_differ_from_their_parent_in_the_graph():
    """VG19, VG20 and VG22 are VG10 plus one structure, and it must be graph-class."""
    for variant in (VG19, VG20, VG22):
        differences = definition_differences(normalise_for_json(VG10), variant)
        graph = {
            difference.field
            for difference in differences
            if difference.role is FieldRole.GRAPH
        }
        assert graph, f"{variant.model_id} differs from VG10 in no graph field"


# --- what the manifest records --------------------------------------------------


def test_the_manifest_keeps_the_raw_definition_beside_the_payload():
    """Several readers index `model.definition` directly, and every fit has it."""
    import inspect

    from vocab_growth.models import common

    source = inspect.getsource(common.write_fit_manifest)
    assert '"definition": normalise_for_json(definition)' in source
    assert '"definition_payload": fit_identity.semantic_payload(definition)' in source
