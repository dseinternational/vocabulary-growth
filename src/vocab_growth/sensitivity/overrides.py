# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Build independent sensitivity definitions from registered models.

make_variant copies the definition and adds a config-name suffix for separate
fit output. It can change priors, data rules and supported structural fields.
Nested concentration blocks are replaced with new frozen instances.
"""

from __future__ import annotations

import dataclasses

from vocab_growth.models.definitions import (
    KappaAnchorPriorParams,
    KappaPriorParams,
    validate_model_definition,
)


def replace_kappa(
    kappa: KappaPriorParams | KappaAnchorPriorParams, **overrides: float
) -> KappaPriorParams | KappaAnchorPriorParams:
    """Return a new concentration-prior block with validated field overrides.

    A parent dataclass copy shares unchanged nested blocks. Replacing the block
    keeps the base prior intact. Unknown fields fail for either parameterisation.
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
    variant = dataclasses.replace(
        base,
        config_name=f"{base.config_name}-{config_suffix}",
        banner=f"{base.banner} [sensitivity: {config_suffix}]",
        **over,
    )
    # Check the overridden definition, including domain bounds and prior shapes.
    validate_model_definition(variant)
    return variant
