# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What makes a fit the same fit: a versioned, classified definition payload.

A fit is validated by comparing the definition recorded in its manifest against
the one registered today. That comparison was raw dictionary equality over
``dataclasses.asdict``, which has one consequence that has shaped the model API
more than any statistical consideration: **adding a field with a default
invalidates every historical fit of that dataclass**, even when the default
reproduces exactly what those fits did.

That is why VG19's child slope and Proposal A1's age-varying scale arrive
through a scalar field that holds an object, why VG20's correlation and VG22's
factor live on sibling subclasses rather than on the shared base, and why
``CLAMP_Q_ONLY`` rides on ``clamp_mean_above_hi_anchor``. Each is a good local
decision forced by a comparison that cannot tell "this model has a new option,
set to the value that means what it always meant" from "this model changed".

This module makes it able to tell, without loosening anything:

* :data:`FIELD_ROLES` classifies every field of every registered definition
  class as graph-affecting, data-affecting, reporting or identity. The
  classification is **complete** -- ``tests/test_fit_identity.py`` checks that
  against the registry -- and it **fails closed**: a field with no entry is
  treated as graph-affecting, the strictest reading, so forgetting to classify a
  new field cannot make a fit validate that should not.

* :data:`BACKFILL_DEFAULTS` names the fields whose *absence* from an older
  manifest is equivalent to a stated value. An entry is a claim that every fit
  made before the field existed behaved exactly as a fit with the field set to
  that value, and it is the only thing that excuses a difference. Adding a field
  without an entry still invalidates history, which is the correct default.

**Every difference remains fatal, including reporting and identity ones.** The
classification's job here is to say *what kind* of thing moved, so a reader of
the failure can tell a changed prior from a changed query grid. Whether a
reporting-only difference should stop a fit being published is a separate
decision, with a real consequence -- a changed ``ages_query`` leaves the stored
query outputs describing ages the report no longer asks for -- and it is not
made here.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

from vocab_growth.models.likelihood_utils import SPOKEN_FALLBACK_PRODUCT

#: Version of the payload's own structure, recorded alongside it. Bumped when
#: the *shape* changes, not when a model does: a reader must be able to tell a
#: payload it can interpret from one it cannot.
SEMANTIC_SCHEMA_VERSION = 1


class FieldRole(Enum):
    """What a definition field controls."""

    GRAPH = "graph"
    """Changes the PyMC graph: a prior, a likelihood, a structural switch.
    A difference means the recorded posterior is from a different model."""

    DATA = "data"
    """Changes the prepared analysis frame: which rows, which population, which
    outcome. A difference means the recorded posterior saw different data.
    ``data.analysis_frame_hash`` catches these independently and exactly; the
    classification is here so a failure can say which of the two it is."""

    REPORTING = "reporting"
    """Changes only the grids and caps the fit *reports* on. The posterior is
    unaffected, but the stored query and plot outputs are computed on these, so
    a difference leaves those outputs describing something else."""

    IDENTITY = "identity"
    """Names the fit rather than describing the model: the model id, the config
    name that fixes the output directory, and the console banner."""


#: Field-name prefixes the completeness test accepts without an explicit
#: :data:`FIELD_ROLES` entry, because every field in these families is a prior and
#: therefore GRAPH. This is **not** a classifier -- :func:`role_of` returns GRAPH for
#: any unlisted name regardless -- it is an allowance that keeps
#: :func:`is_classified` from demanding a row per prior hyperparameter.
#:
#: That makes the broad tokens a real hazard rather than a convenience: ``use_`` and
#: ``lag_`` would silently absorb a future *data* or *reporting* switch and classify
#: it GRAPH, which fails safe for validation but mislabels the failure message. Add a
#: narrower prefix, or an explicit entry, rather than widening one of these.
_ASSUMED_GRAPH_PREFIXES = (
    "anchor_g",
    "beta_lag",
    "ell_months_range",
    "ell_unit",
    "eta_",
    "kappa",
    "lag_",
    "log_conc_",
    "log_psi_",
    "p_slope_",
    "sign_anchor_ages",
    "sign_peak_prior",
    "slope_anchors",
    "spoken_fallback",
    "subject_factor",
    "subject_re_correlation_eta",
    "subject_slope_ref_age_months",
    "subject_variance_partition",
    "tau_",
    "use_",
)

#: Every definition field, classified. Written out for the fields that are not
#: obviously a prior; the prior families are covered by
#: :data:`_ASSUMED_GRAPH_PREFIXES` instead -- the
#: ``p_slope_*``/``ell_unit_*``/``eta_*``/``tau_*``/``kappa*`` blocks are the
#: priors themselves and nothing else.
FIELD_ROLES: dict[str, FieldRole] = {
    # -- identity: names the fit, not the model ------------------------------
    "model_id": FieldRole.IDENTITY,
    "config_name": FieldRole.IDENTITY,
    "banner": FieldRole.IDENTITY,
    # -- data: which rows the model is fitted to -----------------------------
    "population": FieldRole.DATA,
    "outcome": FieldRole.DATA,
    "td_languages": FieldRole.DATA,
    "max_age_months": FieldRole.DATA,
    "min_study_observations": FieldRole.DATA,
    "sample_fraction": FieldRole.DATA,
    # Seeds the reproducible subsample `sample_fraction` takes, so it selects
    # rows rather than only sampler draws.
    "random_seed": FieldRole.DATA,
    "exclude_studies": FieldRole.DATA,
    "exclude_us01_spoken_ceiling": FieldRole.DATA,
    "include_implausible_production": FieldRole.DATA,
    "include_same_day_disagreements": FieldRole.DATA,
    "include_uk01_signed": FieldRole.DATA,
    "include_es01_cells": FieldRole.DATA,
    "include_nz01_cells": FieldRole.DATA,
    "include_uk07_cells": FieldRole.DATA,
    "dse_native_only": FieldRole.DATA,
    "one_observation_per_subject": FieldRole.DATA,
    # -- reporting: the grids and caps the fit reports on --------------------
    #
    # `ages_query` and `n_plot` are reporting rather than graph because the
    # likelihood never touches either: they build `pm.Data` grids the trajectory
    # deterministics are evaluated on. The posterior is identical. What changes
    # is the stored `*_query` and `*_plot` output, which is what the report
    # reads -- so a difference is still fatal here, it is simply a different
    # kind of stale.
    "ages_query": FieldRole.REPORTING,
    "n_plot": FieldRole.REPORTING,
    "report_max_age_understood": FieldRole.REPORTING,
    "report_max_age_signed": FieldRole.REPORTING,
    # -- graph: named individually where the prefixes do not reach -----------
    "n_trials": FieldRole.GRAPH,
    "gp_domain_months": FieldRole.GRAPH,
    "gp_anchor_age_months": FieldRole.GRAPH,
    "centred_study_re": FieldRole.GRAPH,
    "clamp_mean_above_hi_anchor": FieldRole.GRAPH,
}

#: Fields whose absence from an older manifest is equivalent to this value.
#:
#: An entry is a **claim about history**: that every fit made before the field
#: existed behaved exactly as a fit with the field set to this. It is the only
#: thing that excuses a field missing from a recorded definition, and it must be
#: justified where it is added -- the value alone does not show that the
#: pre-field behaviour matched it.
#:
#: The first two entries here are the mechanism's first use, and they are what
#: it was built for. Issue #266 finding 8 needed ``spoken_fallback`` on the
#: trivariate and joint definitions so VG14 and VG15 could run the sensitivity
#: the bivariate models have had since #240 -- and under raw dictionary equality
#: adding it would have invalidated every VG14 and VG15 fit ever made, for a
#: field whose default is what those fits already did.
#:
#: The claim each entry makes is checkable, not asserted:
#: ``likelihood_utils.resolve_fallback_treatment`` reads the field through
#: ``getattr`` with exactly this default, so an engine and a definition that
#: predate the field resolved to ``product_marginal``; and
#: ``spoken_fallback_kappa_sigma`` is read **only** under
#: ``separate_dispersion``, which no fit without the field could have selected,
#: so its value could not have affected one.
#:
#: The third, ``include_same_day_disagreements`` (#289 task 4.3, 2026-09-05),
#: makes a different kind of claim, checked a different way. The field is a
#: loader switch, not a graph choice: an engine forwards it to
#: ``data_utils.load_data`` as a keyword argument, and before the field existed
#: no engine passed that argument at all, so every one of those fits ran the
#: loader at its declared default. The entry's value must therefore equal the
#: loader's own default -- ``tests/test_fit_identity.py`` reads it off
#: ``load_data``'s signature and compares -- and the engines that forward the
#: field are the same three that forward ``include_implausible_production``,
#: which is the only path by which a definition field reaches the loader.
BACKFILL_DEFAULTS: dict[str, Any] = {
    "spoken_fallback": SPOKEN_FALLBACK_PRODUCT,
    "spoken_fallback_kappa_sigma": 0.5,
    "include_same_day_disagreements": False,
}


def role_of(field_name: str) -> FieldRole:
    """How ``field_name`` is classified.

    Unclassified fields are :attr:`FieldRole.GRAPH` -- the strictest reading --
    so a field added without an entry here is treated as changing the model.
    ``tests/test_fit_identity.py`` refuses an unclassified field outright, so
    this fallback is a safety net rather than a way of skipping the decision.
    """
    role = FIELD_ROLES.get(field_name)
    if role is not None:
        return role
    # Everything else is GRAPH, whether or not it matches a prefix. The prefixes
    # are not a classifier -- they are a completeness *allowance*, used only by
    # :func:`is_classified` (see the tuple's own comment).
    return FieldRole.GRAPH


def is_classified(field_name: str) -> bool:
    """Whether ``field_name`` has an explicit classification.

    Distinct from :func:`role_of`, which answers for every name because it has
    to fail closed. This is what the completeness test asks.
    """
    return field_name in FIELD_ROLES or field_name.startswith(_ASSUMED_GRAPH_PREFIXES)


def semantic_payload(definition) -> dict[str, Any]:
    """The versioned, classified payload recorded beside the raw definition.

    The raw ``model.definition`` dictionary stays in the manifest unchanged --
    every fit on disk carries it, several readers index it directly, and the
    report layer reads its own numbers out of it. This is written alongside, so
    a reader that wants the classification has it and a reader that does not is
    unaffected.
    """
    from vocab_growth.fit_artifacts import normalise_for_json

    grouped: dict[str, dict[str, Any]] = {role.value: {} for role in FieldRole}
    for item in fields(definition):
        value = getattr(definition, item.name)
        grouped[role_of(item.name).value][item.name] = normalise_for_json(value)
    return {"schema_version": SEMANTIC_SCHEMA_VERSION, **grouped}


class DefinitionDifference:
    """One field on which a recorded definition and the expected one disagree."""

    __slots__ = ("field", "role", "recorded", "expected", "reason")

    def __init__(self, field: str, role: FieldRole, recorded, expected, reason: str):
        self.field = field
        self.role = role
        self.recorded = recorded
        self.expected = expected
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DefinitionDifference {self.field} ({self.role.value}): {self.reason}>"

    def describe(self) -> str:
        return f"{self.field} ({self.role.value}): {self.reason}"


def definition_differences(recorded, expected) -> list[DefinitionDifference]:
    """Every way ``recorded`` differs from ``expected``, classified.

    ``recorded`` is the manifest's ``model.definition`` dictionary; ``expected``
    is a registered definition, or its already-normalised dictionary.

    Fail-closed by construction:

    * a field in one and not the other is a difference, **unless** it is absent
      from ``recorded`` and :data:`BACKFILL_DEFAULTS` states the value its
      absence means and ``expected`` carries that value;
    * any value difference is a difference, whatever its role;
    * a recorded definition that is not a mapping is a difference in itself,
      rather than something to compare leniently.
    """
    from vocab_growth.fit_artifacts import normalise_for_json

    if is_dataclass(expected) and not isinstance(expected, type):
        expected = normalise_for_json(expected)
    if not isinstance(recorded, dict):
        return [
            DefinitionDifference(
                "<definition>",
                FieldRole.GRAPH,
                recorded,
                expected,
                "the manifest records no definition object",
            )
        ]

    differences: list[DefinitionDifference] = []
    for name in sorted(set(recorded) | set(expected)):
        role = role_of(name)
        if name not in recorded:
            if name in BACKFILL_DEFAULTS and expected[name] == normalise_for_json(
                BACKFILL_DEFAULTS[name]
            ):
                # The field postdates this fit, and its value is the one that
                # reproduces what the fit did. Not a difference.
                continue
            differences.append(
                DefinitionDifference(
                    name,
                    role,
                    None,
                    expected[name],
                    "absent from the recorded definition, and no backfill "
                    "default states what its absence meant",
                )
            )
        elif name not in expected:
            differences.append(
                DefinitionDifference(
                    name,
                    role,
                    recorded[name],
                    None,
                    "recorded but no longer a field of the registered definition",
                )
            )
        elif recorded[name] != expected[name]:
            differences.append(
                DefinitionDifference(
                    name,
                    role,
                    recorded[name],
                    expected[name],
                    f"recorded {recorded[name]!r}, registered {expected[name]!r}",
                )
            )
    return differences
