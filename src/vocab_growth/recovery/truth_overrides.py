# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Set free parameters in a recovery truth draw and record those settings.

The caller recomputes derived quantities after these overrides. Direct changes
to deterministics are refused because recomputation would overwrite them.

Numeric settings fill a free variable with one value. ``subject_re=independent``
replaces a packed Cholesky covariance factor with a diagonal factor, preserving
its row norms while removing correlations. This transform applies to packed
blocks; other correlation parameterisations may use named free scalars.

Truth settings form part of simulation, fit and score labels so distinct checks
have separate output identities.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

#: Marker introducing the settings in a recovery config name.
TAG_PREFIX = "set"

#: A setting's name must be a Python identifier -- it names a model variable --
#: and the value must be a number or a registered transform.
_SETTING = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.+)$")


@dataclass(frozen=True)
class TruthOverride:
    """One parameter setting applied to a truth draw before it is used."""

    name: str
    """The free variable to set."""

    value: float | None = None
    """The number to set it to, or ``None`` when :attr:`transform` is given."""

    transform: str | None = None
    """A registered structural setting, when a number cannot express it."""

    def __post_init__(self) -> None:
        if (self.value is None) == (self.transform is None):
            raise ValueError(
                "A truth override carries exactly one of a value or a transform."
            )

    @property
    def setting(self) -> str:
        """The setting as written, for banners and provenance."""
        return self.transform if self.transform is not None else _format(self.value)

    def __str__(self) -> str:
        return f"{self.name}={self.setting}"


def _format(value: float) -> str:
    """A filesystem-safe spelling of ``value`` that stays readable.

    Directory names carry these, so the decimal point and the minus sign -- which
    is also the config name's token separator -- are spelt out rather than used.
    """
    if value == int(value):
        text = str(abs(int(value)))
    else:
        text = f"{abs(value):g}".replace(".", "p")
    return f"neg{text}" if value < 0 else text


def parse_truth_override(text: str) -> TruthOverride:
    """Parse one ``NAME=VALUE`` setting from the command line.

    ``VALUE`` is a number, or the name of a registered structural transform.
    """
    match = _SETTING.match(text.strip())
    if match is None:
        raise ValueError(
            f"Cannot read {text!r} as a truth setting. Write NAME=VALUE, where "
            f"VALUE is a number or one of: {', '.join(sorted(STRUCTURAL_TRANSFORMS))}."
        )
    name, raw = match.group("name"), match.group("value").strip()
    if raw in STRUCTURAL_TRANSFORMS:
        return TruthOverride(name=name, transform=raw)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"Cannot read {raw!r} as a number or a registered transform for "
            f"{name!r}. The transforms are: "
            f"{', '.join(sorted(STRUCTURAL_TRANSFORMS))}."
        ) from None
    if not math.isfinite(value):
        raise ValueError(f"{name}={raw} is not a finite value.")
    return TruthOverride(name=name, value=value)


def override_tag(overrides) -> str:
    """A settings token sorted by variable name, or empty when none are supplied.

    Uses each override's filesystem spelling so argument order does not change
    the recovery name.
    """
    ordered = sorted(overrides, key=lambda item: item.name)
    if not ordered:
        return ""
    return "-".join([TAG_PREFIX, *(f"{o.name}-{o.setting}" for o in ordered)])


# ==========================================================================
# Structural transforms
# ==========================================================================


def _triangular_order(n_packed: int) -> int:
    """Matrix order whose lower triangle holds ``n_packed`` entries."""
    order = int((math.isqrt(8 * n_packed + 1) - 1) // 2)
    if order * (order + 1) // 2 != n_packed:
        raise ValueError(
            f"{n_packed} values are not the lower triangle of a square matrix, "
            "so this is not a packed Cholesky factor."
        )
    return order


def set_independent(values: np.ndarray, *, name: str) -> tuple[np.ndarray, str]:
    """Replace a packed covariance Cholesky factor with its row norms on the diagonal.

    The covariance is ``L @ L.T``, so each marginal standard deviation is a row
    norm of ``L``. The replacement has the same norms and zero correlations,
    up to floating-point evaluation; it does not set the scales to zero.
    """
    packed = np.asarray(values, dtype=float)
    if packed.ndim != 1:
        raise ValueError(
            f"{name} has shape {packed.shape}; 'independent' applies to a packed "
            "Cholesky factor, which is one-dimensional."
        )
    order = _triangular_order(packed.size)
    factor = np.zeros((order, order), dtype=float)
    factor[np.tril_indices(order)] = packed
    scales = np.linalg.norm(factor, axis=1)
    if not np.all(scales > 0):
        raise ValueError(
            f"{name} has a zero row norm, so it is not a usable covariance factor."
        )
    independent = np.zeros((order, order), dtype=float)
    independent[np.diag_indices(order)] = scales
    described = ", ".join(f"{scale:.4g}" for scale in scales)
    return independent[np.tril_indices(order)], (
        f"{order}x{order} correlation set to the identity, scales kept ({described})"
    )


#: Settings that no single number can express, by the name written on the
#: command line. A transform takes the drawn value and returns the replacement
#: plus a one-line description for the provenance record.
STRUCTURAL_TRANSFORMS: dict[str, Callable[..., tuple[np.ndarray, str]]] = {
    "independent": set_independent,
}


# ==========================================================================
# Application
# ==========================================================================


def _free_variable_names(model) -> list[str]:
    return [rv.name for rv in model.free_RVs]


def _deterministic_names(model) -> set[str]:
    return {variable.name for variable in model.deterministics}


def apply_truth_overrides(
    posterior: xr.Dataset, model, overrides
) -> tuple[xr.Dataset, list[dict[str, Any]]]:
    """Apply settings to a single-draw free-parameter dataset and record them.

    Numeric settings fill every entry of the named free variable. Structural
    settings transform its drawn array. The caller must recompute deterministics
    afterwards to propagate changes to reported quantities.
    """
    if not overrides:
        return posterior, []

    free_names = _free_variable_names(model)
    deterministics = _deterministic_names(model)
    updated = posterior.copy()
    applied: list[dict[str, Any]] = []

    for override in overrides:
        name = override.name
        if name not in free_names:
            if name in deterministics:
                raise ValueError(
                    f"{name!r} is a deterministic: the model computes it from its "
                    "free variables rather than sampling it, so setting it here "
                    "would be overwritten when the truth's deterministics are "
                    "recomputed. Set the free variable(s) it is computed from "
                    "instead -- a correlation read off a packed Cholesky factor "
                    "is set with, for example, "
                    f"'subject_re=independent'. Free variables: "
                    f"{', '.join(free_names)}."
                )
            raise ValueError(
                f"{name!r} is not a free variable of this model. Free variables: "
                f"{', '.join(free_names)}."
            )
        if name not in updated.data_vars:
            raise ValueError(
                f"{name!r} is a free variable of the model but is absent from the "
                "truth draw, which records the free variables it was built from. "
                "The truth and the model disagree; re-run the simulate step."
            )

        drawn = updated[name]
        if override.transform is not None:
            transform = STRUCTURAL_TRANSFORMS[override.transform]
            replacement, described = transform(
                np.asarray(drawn.values).reshape(-1), name=name
            )
            values = np.asarray(replacement).reshape(drawn.shape)
            updated[name] = xr.DataArray(
                values, dims=drawn.dims, coords=drawn.coords, attrs=drawn.attrs
            )
        else:
            described = f"every entry set to {override.value:g}"
            updated[name] = xr.full_like(drawn, float(override.value))

        applied.append(
            {
                "name": name,
                "setting": override.setting,
                "shape": [int(size) for size in drawn.shape],
                "applied": described,
            }
        )

    return updated, applied


def check_all_finite(posterior: xr.Dataset, *, context: str) -> None:
    """Reject non-finite values in the supplied truth dataset.

    Call after recomputing deterministics, since an override can produce a
    non-finite derived value. This checks finiteness, not distributional support
    or scientific plausibility.
    """
    offending = sorted(
        name
        for name, array in posterior.data_vars.items()
        if not np.all(np.isfinite(np.asarray(array.values, dtype=float)))
    )
    if offending:
        raise ValueError(
            f"{context} produced non-finite truth value(s) for "
            f"{', '.join(offending)}. A setting has put a parameter outside what "
            "the model's reported quantities can be computed at."
        )
