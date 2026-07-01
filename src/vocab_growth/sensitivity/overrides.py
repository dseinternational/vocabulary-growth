# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build alternative-prior variants of a model definition (issue #89 §7).

`make_variant` returns a copy of a committed definition (via
``dataclasses.replace``) whose ``config_name`` carries a suffix — so its fitted
output lands in a separate ``output/models/<model_id>-<config_name>-<suffix>/``
directory and never clobbers the model of record — with the named prior
hyperparameters overridden. The committed ``VGxx`` instances are never mutated.
"""

from __future__ import annotations

import dataclasses

from vocab_growth.models.definitions import KappaPriorParams


def replace_kappa(kappa: KappaPriorParams, **overrides: float) -> KappaPriorParams:
    """Return a NEW ``KappaPriorParams`` with the given fields overridden.

    A fresh instance (rather than an in-place mutation) is required because
    ``dataclasses.replace`` on the parent definition copies the nested kappa
    objects by *reference*; an override must therefore supply a new object so the
    variant and the base do not share (and accidentally alias) one kappa prior.
    """
    valid = {f.name for f in dataclasses.fields(KappaPriorParams)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown KappaPriorParams field(s): {sorted(unknown)}")
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
