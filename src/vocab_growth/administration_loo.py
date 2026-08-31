# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Leave-one-**administration**-out, as the reports have always claimed.

Issue #266 finding 4. The multi-outcome engines compute a separate PSIS-LOO for
each outcome, while the report describes the predictive unit as a complete
checklist administration. Those are not the same score, and the difference is
not presentational:

* the spoken likelihood's trial count **is the same row's observed
  comprehension**, so holding out a spoken term scores prediction *conditional
  on* that comprehension rather than prediction of a withheld administration;
* holding out an understood term leaves its own observed value in the spoken
  term's denominator, so the held-out value has not really been withheld;
* a paired administration becomes two held-out cases with two importance
  weights, which is not what "one observation" means anywhere in the reports.

The coherent unit is the administration: **sum** every likelihood factor
belonging to one row of the analysis frame into one pointwise entry, so a
held-out case is ``log p(U_i) + log p(S_i | U_i)`` where both exist, and the
single factor where only one does. For VG15 that includes the two composition
terms -- the four-cell cross-tabulation and the ``nz_01`` produced cells --
which identify its headline association ``psi`` and which the per-outcome scores
omit entirely, so its LOO never scored the thing the model exists to estimate.

The mapping from each factor's likelihood rows back to administration rows is
the ``obs_*_mask`` constant data every engine already stores. That is why the
mask defect (finding 3) had to be fixed first: a mask that marked recorded rows
rather than likelihood rows would sum the wrong factors onto the wrong
administrations, silently.

The mechanics are ``scripts/loo_compare.py``'s ``_attach_joint_log_likelihood``,
which has computed this correctly for the bivariate case since #236 -- generalised
to any number of factors, including matrix-valued ones, and moved where the fit
pipeline itself can use it.

**Repeated administrations of the same child remain separate cases.** This
scores prediction of another administration like those in the frame, not
generalisation to a new child; ``scripts/kfold_loso.py`` is what answers the
latter. Said here because "leave one out" invites the other reading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

#: Name of the combined pointwise likelihood attached to the trace, and the
#: dimension it is indexed by. ``obs_joint`` is the name
#: ``scripts/loo_compare.py`` has used since #236; kept so the two agree.
ADMINISTRATION_VAR = "y_administration"
ADMINISTRATION_DIM = "obs_joint"

#: How the combined score is labelled in `loo_summary.csv` and the reports.
#: Spelled out rather than abbreviated: the whole finding is that "LOO" alone
#: was read as this when it was not.
ADMINISTRATION_LABEL = "administration (all outcomes)"


@dataclass(frozen=True)
class LikelihoodFactor:
    """One likelihood term, and the mask locating its rows in the frame."""

    variable: str
    """Its name in the trace's ``log_likelihood`` group."""

    mask: str
    """The ``constant_data`` mask, over all ``n`` administration rows, marking
    the rows this factor covers -- in the same order the factor's own rows are
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

    Returns ``None`` -- rather than raising -- when a factor or its mask is
    absent, because a caller may legitimately be looking at a trace written
    before this existed or by an engine that stores only one factor. A factor
    present with a mask that does not match its rows **does** raise: that is the
    finding-3 defect, and silently summing the wrong rows onto the wrong
    administrations is exactly what must not happen.
    """
    log_likelihood = getattr(trace, "log_likelihood", None)
    constant_data = getattr(trace, "constant_data", None)
    if log_likelihood is None or constant_data is None:
        return None

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

    # Position of each administration among the rows the combined score keeps.
    position = np.cumsum(any_mask) - 1
    template = usable[0][0]
    n_chain = template.sizes["chain"]
    n_draw = template.sizes["draw"]
    combined = np.zeros((n_chain, n_draw, int(any_mask.sum())), dtype=float)

    for array, mask in usable:
        dim = _factor_dim(array)
        values = array.transpose("chain", "draw", dim, ...).values
        if values.ndim > 3:
            # A composition factor: sum its cells into the row's contribution.
            values = values.reshape(n_chain, n_draw, values.shape[2], -1).sum(axis=-1)
        np.add.at(combined, (slice(None), slice(None), position[mask]), values)

    return xr.DataArray(
        combined,
        dims=("chain", "draw", ADMINISTRATION_DIM),
        coords={
            "chain": template["chain"].values,
            "draw": template["draw"].values,
            ADMINISTRATION_DIM: np.flatnonzero(any_mask),
        },
    )


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
