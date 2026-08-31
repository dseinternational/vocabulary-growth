# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build alternative-prior variants of a model definition (issue #89 §7).

`make_variant` returns a copy of a committed definition (via
``dataclasses.replace``) whose ``config_name`` carries a suffix — so its fitted
output lands in a separate ``output/models/<model_id>-<config_name>-<suffix>/``
directory and never clobbers the model of record — with the named prior
hyperparameters overridden. The committed ``VGxx`` instances are never mutated,
and since issue #273 they are frozen, so they cannot be.
"""

from __future__ import annotations

import dataclasses

from vocab_growth.models.definitions import (
    KappaAnchorPriorParams,
    KappaPriorParams,
)


def replace_kappa(
    kappa: KappaPriorParams | KappaAnchorPriorParams, **overrides: float
) -> KappaPriorParams | KappaAnchorPriorParams:
    """Return a NEW kappa prior block with the given fields overridden.

    A fresh instance is required because ``dataclasses.replace`` on the parent
    definition copies the nested kappa objects by *reference*, so an override
    has to supply a new object rather than edit the shared one. Since issue #273
    froze :class:`~vocab_growth.models.definitions.KappaPriorParams` and
    :class:`~vocab_growth.models.definitions.KappaAnchorPriorParams` there is no
    longer an in-place edit to reach for -- the sharing is safe by construction
    and an accidental mutation raises rather than silently reaching the base
    definition -- but the reason this function exists is unchanged: a variant
    needs its own block.

    Field names are checked against whichever form the block uses, so a variant
    written for the legacy triple fails loudly on a migrated outcome instead of
    silently doing nothing — which is how a stale sensitivity variant would
    otherwise survive a migration.
    """
    valid = {f.name for f in dataclasses.fields(type(kappa))}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown {type(kappa).__name__} field(s): {sorted(unknown)}. "
            f"This block uses the "
            f"{'two-anchor' if isinstance(kappa, KappaAnchorPriorParams) else 'legacy'}"
            f" form, whose fields are {sorted(valid)}."
        )
    return dataclasses.replace(kappa, **overrides)


def make_variant(
    base,
    *,
    config_suffix: str,
    scalar_over: dict | None = None,
    kappa_over: dict[str, dict[str, float]] | None = None,
):
    """Return a prior-variant copy of ``base``.

    Parameters
    ----------
    base
        A committed model definition (``UnivariateModelDefinition``,
        ``BivariateModelDefinition``, ``TrivariateModelDefinition`` or
        ``JointModelDefinition``). It is not mutated.
    config_suffix
        Appended to ``config_name`` (and the banner) to isolate the variant's
        output directory, e.g. ``"psi-neutral"``.
    scalar_over
        Top-level hyperparameter overrides, e.g. ``{"eta_sign_sigma": 1.5}``.
        Unknown field names raise ``TypeError`` (via ``dataclasses.replace``).
    kappa_over
        Per-modality kappa overrides keyed by the nested attribute name, e.g.
        ``{"kappa_u": {"kappa_min_sigma": 1.0}, "kappa_s": {"a_kappa_mu": 0.0}}``.
    """
    if not config_suffix:
        raise ValueError("config_suffix must be a non-empty string.")
    over = dict(scalar_over or {})
    for attr, sub in (kappa_over or {}).items():
        if not hasattr(base, attr):
            raise ValueError(
                f"{type(base).__name__} has no kappa attribute {attr!r}."
            )
        over[attr] = replace_kappa(getattr(base, attr), **sub)
    return dataclasses.replace(
        base,
        config_name=f"{base.config_name}-{config_suffix}",
        banner=f"{base.banner} [sensitivity: {config_suffix}]",
        **over,
    )
