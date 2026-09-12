# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Set a parameter in a recovery truth draw instead of taking what a draw offers.

A truth draw comes from the model of record's posterior or from the model's own
prior, and in both cases every parameter takes whatever value that draw happened
to hold. That answers "does this model recover itself in the regime it reports
in", which is the question the harness was built for. It cannot answer a
question about a *designed* parameter setting -- and two of the three cells
[#297](https://github.com/dseinternational/vocabulary-growth/issues/297) check 4
asks for are exactly that. VG25 exists to show that a sign -> speech cross-lag
can be told apart from the persistent sign-speech correlation it sits beside, so
the cells are ``(beta = 0, rho != 0)``, ``(beta != 0, rho = 0)`` and both
nonzero. The third is a draw; the first two are settings. #242 item 6 asks the
same three of VG16.

**Free variables only, and on purpose.** An override names a variable the model
samples, never one it computes. The reported estimands -- the trajectories,
``rho_sign_q``, the child effects -- are deterministics, and
:func:`vocab_growth.recovery.simulate._with_deterministics` recomputes them from
the graph *after* the overrides are applied. So setting a free variable moves
every quantity downstream of it and the truth stays internally consistent;
setting a deterministic would be overwritten by that recomputation, and the run
would silently score against a truth the graph never held. Naming one is refused
with the reason.

**Two kinds of setting, because two kinds of parameter.** A scalar coefficient
takes a number: ``beta_sign_lag=0``. A correlation does not, because the model
does not sample one -- the joint and bivariate correlated blocks sample a packed
Cholesky factor of the child covariance (``subject_re``) and read every ``rho_*``
off it as a deterministic. "No correlation" is a statement about that factor's
*structure*, so it is a named transform rather than a number:
``subject_re=independent`` replaces the factor with the diagonal one carrying the
same scales. The scales are the factor's own row norms, so ``tau_subj_*`` come
back bit-identical and only the correlations move -- which is what makes it the
``rho = 0`` cell of the same model rather than a different model.

**The setting is part of the run's identity.** Two cells of the same gate differ
only in their truth, so they must not share a simulation directory, a fit
directory or a recovery matrix. :func:`override_tag` goes into the recovery
config name the way ``-under-`` does for a cross-definition run (#226), for the
same reason: a name that does not distinguish them lets one overwrite the other,
or be scored as the other.
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
    """The token a set of settings contributes to a recovery config name.

    Sorted by variable name, so the same set always names the same directory
    however it was typed. Empty for no settings, which is what keeps every
    existing recovery output at the name it already has.
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
    """Replace a packed Cholesky covariance factor with the diagonal one.

    ``LKJCholeskyCov`` samples the packed lower triangle of the Cholesky factor
    of the child covariance, and the model reads both the scales and the
    correlations off it -- ``tau_subj_*`` are the factor's row norms and each
    ``rho_*`` an entry of the implied correlation matrix. Replacing the factor
    with ``diag(row norms)`` therefore sets every correlation to exactly zero and
    leaves every scale untouched, which is the ``rho = 0`` cell of the *same*
    model rather than a differently-scaled one.
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
    """Return ``posterior`` with each setting applied, and what was applied.

    ``posterior`` holds a single ``(chain, draw)`` of the model's free variables.
    Deterministics are **not** recomputed here -- the caller does that afterwards,
    which is what propagates a setting into every reported quantity downstream of
    it.
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
    """Refuse a truth carrying a non-finite value.

    Run after the deterministics are recomputed, which is where a setting on a
    boundary shows up: a scale set to zero reaches the report as a division or a
    logarithm of it long before it reaches the sampler, and a truth that is not
    finite cannot be scored against. It is a guard on the settings rather than on
    the draw -- a value inside a variable's support that the model nonetheless
    dislikes is the fit's business, not this check's.
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
