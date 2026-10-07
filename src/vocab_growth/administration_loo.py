# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Combine likelihood factors for leave-one-administration-out scoring.

Each analysis-frame row is one held-out case. Its score sums all available
outcome and composition factors, for example ``log p(U_i) + log p(S_i | U_i)``.
Separate outcome scores would still condition on part of the administration:
the spoken likelihood uses that row's observed comprehension as its trial count.

The engines' ``obs_*_mask`` arrays map likelihood rows to administrations. These
masks must mark rows used by the likelihood, not every row with a recorded
outcome. Missing factors prevent construction of the combined score.

Repeated administrations of one child remain separate cases. This score assesses
another administration like those in the frame. ``scripts/kfold_loso.py`` instead
assesses prediction for a new child.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from dse_research_utils.statistics.log_likelihood import (
    LogLikelihoodFactor as SharedLikelihoodFactor,
)
from dse_research_utils.statistics.log_likelihood import (
    aggregate_log_likelihood,
)

#: Combined pointwise likelihood and its administration-row dimension.
ADMINISTRATION_VAR = "y_administration"
ADMINISTRATION_DIM = "obs_joint"

#: Label in ``loo_summary.csv`` and the reports.
ADMINISTRATION_LABEL = "administration (all outcomes)"


@dataclass(frozen=True)
class LikelihoodFactor:
    """One likelihood term, and the mask locating its rows in the frame."""

    variable: str
    """Its name in the trace's ``log_likelihood`` group."""

    mask: str
    """The ``constant_data`` mask, over all ``n`` administration rows, marking
    the rows this factor covers, in the same order the factor's own rows are
    stored."""


def _factor_dim(array: xr.DataArray) -> str:
    """The row dimension of a pointwise log-likelihood.

    A composition factor is stored per row *and* per cell (``obs_cells_id`` x
    ``cell_id``); ArviZ sums the trailing dimensions into the pointwise value,
    and so does this -- the held-out unit is the administration, not one cell of
    its cross-tabulation.
    """
    dims = [dim for dim in array.dims if dim not in ("chain", "draw")]
    if not dims:
        raise ValueError(
            f"log-likelihood factor has no row dimension (dims {array.dims})."
        )
    return dims[0]


def administration_log_likelihood(
    trace, factors: tuple[LikelihoodFactor, ...]
) -> xr.DataArray | None:
    """Sum ``factors`` onto administration rows, or ``None`` if not derivable.

    Missing factors or masks return ``None``, including for older traces. A mask
    whose marked-row count differs from its factor raises, because that factor
    cannot be mapped to administrations safely.

    Repeated variable names, ``NaN`` and ``+inf`` raise. The shared aggregator
    retains ``-inf`` for impossible observations and sums values in float64.
    """
    log_likelihood = getattr(trace, "log_likelihood", None)
    constant_data = getattr(trace, "constant_data", None)
    if log_likelihood is None or constant_data is None:
        return None

    # Dataset indexing creates distinct array objects, so the shared helper's
    # duplicate-object check cannot detect one variable declared twice.
    names = [factor.variable for factor in factors]
    if len(set(names)) != len(names):
        raise ValueError(
            "A log-likelihood factor was declared more than once "
            f"({sorted(name for name in set(names) if names.count(name) > 1)}); "
            "its contribution would be summed onto the same administrations twice."
        )

    usable: list[tuple[xr.DataArray, np.ndarray]] = []
    for factor in factors:
        if factor.variable not in log_likelihood.data_vars:
            return None
        if factor.mask not in constant_data.data_vars:
            return None
        array = log_likelihood[factor.variable]
        mask = np.asarray(constant_data[factor.mask].values, dtype=bool)
        dim = _factor_dim(array)
        if int(mask.sum()) != array.sizes[dim]:
            raise ValueError(
                f"{factor.mask} marks {int(mask.sum())} rows but "
                f"{factor.variable} stores {array.sizes[dim]}; the factor "
                "cannot be mapped to administrations. This is the shape of "
                "issue #266 finding 3 -- a mask recording observed rows rather "
                "than likelihood rows."
            )
        usable.append((array, mask))

    if not usable:
        return None

    any_mask = np.zeros_like(usable[0][1], dtype=bool)
    for _, mask in usable:
        any_mask |= mask
    if not any_mask.any():
        return None

    # Use positions in the full analysis frame as administration identifiers.
    # Factor-specific ranks would map different factors to different rows.
    combined = aggregate_log_likelihood(
        [
            SharedLikelihoodFactor(
                values=array,
                row_dim=_factor_dim(array),
                row_unit_ids=np.flatnonzero(mask),
                # A composition factor is stored per row *and* per cell; the
                # cells are summed within the row, exactly as ArviZ folds a
                # pointwise likelihood's trailing dimensions.
                event_dims=tuple(
                    dim
                    for dim in array.dims
                    if dim not in ("chain", "draw", _factor_dim(array))
                ),
            )
            for array, mask in usable
        ],
        unit_ids=np.flatnonzero(any_mask),
        unit_dim=ADMINISTRATION_DIM,
    )
    # Store the same variable name in both the DataArray and the trace Dataset.
    return combined.rename(ADMINISTRATION_VAR)


def attach_administration_log_likelihood(
    trace, factors: tuple[LikelihoodFactor, ...]
) -> bool:
    """Add :data:`ADMINISTRATION_VAR` to ``trace``'s log-likelihood group.

    Returns whether it was added. Idempotent: an already-attached score is left
    alone, so re-running diagnostics on a trace does not recompute it.
    """
    log_likelihood = getattr(trace, "log_likelihood", None)
    if log_likelihood is None:
        return False
    if ADMINISTRATION_VAR in log_likelihood.data_vars:
        return True
    combined = administration_log_likelihood(trace, factors)
    if combined is None:
        return False
    trace.log_likelihood = log_likelihood.assign({ADMINISTRATION_VAR: combined})
    return True
