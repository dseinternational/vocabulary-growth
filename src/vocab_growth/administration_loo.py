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
pipeline itself can use it. Since the shared library's 0.14.0 release the
summation itself is
:func:`dse_research_utils.statistics.log_likelihood.aggregate_log_likelihood`,
which takes the output units *explicitly* rather than inferring them from a
mask. What stays here is everything that decides which unit a row belongs to:
the factor/mask pairs each engine declares, the refusal to score a partial set
of factors, and the finding-3 check below.

**Repeated administrations of the same child remain separate cases.** This
scores prediction of another administration like those in the frame, not
generalisation to a new child; ``scripts/kfold_loso.py`` is what answers the
latter. Said here because "leave one out" invites the other reading.
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

    The shared aggregator adds two refusals this had none of. A likelihood
    containing ``NaN`` or ``+inf`` raises instead of propagating into a
    plausible-looking score, while ``-inf`` is retained because an impossible
    observation genuinely has that log likelihood. Values are summed in
    float64, which every engine here already stores. Declaring one trace
    variable as two factors is refused here rather than by the shared helper --
    see the comment on the check.
    """
    log_likelihood = getattr(trace, "log_likelihood", None)
    constant_data = getattr(trace, "constant_data", None)
    if log_likelihood is None or constant_data is None:
        return None

    # Selecting non-overlapping factors is the caller's responsibility, and the
    # shared aggregator cannot check it for us: indexing a Dataset twice yields
    # two distinct objects, so its own duplicate-array guard does not see a
    # variable named twice. One repeated name is the whole of the overlap this
    # repository can have -- the factor tuples are engine-declared constants
    # over disjoint trace variables -- and it would double the term rather than
    # fail, producing a total that is simply wrong with nothing in the result
    # to show it.
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

    # The unit is the administration row of the analysis frame, named by its
    # own position in that frame. Handing the shared helper the frame positions
    # -- rather than a mask, or a rank among the kept rows -- is what makes the
    # mapping legible: `obs_joint` then *is* `np.flatnonzero(any_mask)`, the
    # same coordinate the combined score has carried since #266, and a factor
    # whose mask disagrees with its rows can no longer land on a neighbour.
    combined = aggregate_log_likelihood(
        [
            SharedLikelihoodFactor(
                values=array,
                row_dim=_factor_dim(array),
                # Rows of this factor, in the order it stores them, named by
                # the administration each covers.
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
    # The shared helper names its result `log_likelihood`; this one is stored
    # under `ADMINISTRATION_VAR`, and a name that says otherwise would be read
    # back from the trace.
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
