# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rebuild posterior deterministics a stored trace does not carry.

Since 2026-08-23 the engines' ``sample`` stage tells the sampler not to store
the observation-sized deterministics (``f_obs``, ``p_obs``, ``kappa_obs``,
their per-outcome counterparts and the concatenated ``*_all`` grids — see
:func:`vocab_growth.fit_artifacts.sampled_variable_names`). The model graph
still defines every one of them, so a reader that needs one rebuilds the model
and computes it from the stored free parameters with ``pm.compute_deterministics``.
This module is that one path, so the few readers that need it
(``scripts/loso_compare.py``) do not each grow their own.

A trace written before the change still carries the variables, and
:func:`with_deterministics` leaves anything already present alone — so the same
call serves old and new fits, and computes nothing when nothing is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pymc as pm
import xarray as xr


def missing_deterministics(posterior: xr.Dataset, names: Sequence[str]) -> list[str]:
    """The subset of ``names`` that ``posterior`` does not carry, in order."""
    return [name for name in names if name not in posterior.data_vars]


def with_deterministics(
    posterior: xr.Dataset,
    model: Any,
    names: Sequence[str],
    *,
    progressbar: bool = False,
) -> xr.Dataset:
    """Return ``posterior`` carrying every name in ``names``.

    Names already present are used as they are; the rest are computed from
    ``model`` — which must be the graph the posterior was sampled from, rebuilt
    on the same data, so that observation order and every data rule match
    (callers check that with ``fit_artifacts.validate_fit_output`` before
    aligning anything by row). The input is not modified.

    Pass a thinned posterior (``posterior.isel(draw=...)``) when only a subset
    of draws is wanted: the cost is ``len(names) x n_obs x draws``, which for a
    reporting fit is exactly the memory the change exists to avoid holding.
    """
    missing = missing_deterministics(posterior, names)
    unknown = [
        name
        for name in missing
        if name not in {deterministic.name for deterministic in model.deterministics}
    ]
    if unknown:
        raise KeyError(
            f"{unknown} are neither in the posterior nor deterministics of the "
            "model; they cannot be recomputed."
        )
    if not missing:
        return posterior
    computed = pm.compute_deterministics(
        posterior,
        var_names=missing,
        model=model,
        progressbar=progressbar,
    )
    # `compute_deterministics` relabels the sample dimensions 0..n-1, so on a
    # thinned posterior (draw labels 0, 36, 72, ...) the output would not align
    # with the input. The sizes are equal by construction — it computes one
    # value per input draw — so carry the input's own labels over before
    # merging, and merge with join="exact" so that if the sizes ever differ the
    # merge fails rather than fabricating all-missing draws (the same guard
    # recovery/simulate.py uses).
    for dim in ("chain", "draw"):
        if dim in posterior.dims and dim in computed.dims:
            if computed.sizes[dim] != posterior.sizes[dim]:
                raise RuntimeError(
                    f"Computed deterministics have {computed.sizes[dim]} {dim}s "
                    f"for a posterior with {posterior.sizes[dim]}."
                )
            computed = computed.assign_coords({dim: posterior[dim].values})
    return xr.merge([posterior, computed], join="exact")
