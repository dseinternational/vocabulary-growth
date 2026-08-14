# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Project-wide reporting age policy — the single source of truth.

Every figure and summary table stops where its own outcome's evidence stops.
The rule is per **quantity**, not per figure, because a single figure can carry
two outcomes with different support: the joint trajectory plots draw understood
and spoken together, and they must be trimmed independently.

Policy
------
==========================  ===  =========================================
quantity                    cap  rationale
==========================  ===  =========================================
understood                   84  25 rows / 20 children / 5 studies in the
                                 72-84 band; at or above 84 only 13 rows
                                 from 11 children, and 84 is the high trend
                                 anchor above which the mean is levelled
                                 off rather than fitted.
ratio of understood          84  ``q`` (spoken given understood), ``r``
                                 (fraction of understood signed) and
                                 ``p_any`` are all conditioned on
                                 understood, so they inherit its cap.
signed                       84  Its own field since #212; see below.
spoken                       90  The top of the query grid.
==========================  ===  =========================================

Why ``spoken`` is derived rather than declared
----------------------------------------------
The two declared caps live on the **model definition**, which is fingerprinted
into ``fit_manifest.json`` and compared as whole-object equality. That is
deliberate for ``report_max_age_understood``: change it and every affected model
of record is correctly marked stale.

It also means *adding* a field is not free. ``normalise_for_json`` serialises
the definition with ``asdict``, so a new field appears as a new key in every
model's definition and invalidates all fifteen at once -- including models
mid-fit. A spoken cap declared that way would have cost a full refit to express
a number we already know.

So the spoken cap is derived from ``ages_query``, which is already on the
definition. This keeps it fingerprint-backed *indirectly*: extend the query grid
and the definition changes, models go stale, and a refit follows -- which is
right, because ``ages_query`` is a real model grid rather than a reporting
choice. It also makes the coupling honest, in that spoken cannot be reported
past the ages the model was asked to report on.

The asymmetry is the cost, and it is deliberate rather than overlooked: the
understood cap is gate-protected and independently settable, the spoken cap is
neither. If spoken ever needs a cap that is *not* the top of the query grid, it
needs a real definition field and the refit that implies.

Typically-developing models are unaffected either way: ``TD_POOL_AGE_MONTHS`` is
``(8, 30)`` and their GP domains stop at 30 or 18, so no cap here can bind.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = ["ReportedQuantity", "max_age_for", "quantity_for_outcome"]


class ReportedQuantity(Enum):
    """What a figure or table reports, for the purpose of trimming its ages.

    Call sites name the quantity rather than reaching for a cap attribute.
    Picking the wrong attribute is exactly the defect fixed in ``4ff48e5``,
    where VG14's sign-derived figures were trimmed by the comprehension cap and
    therefore moved when an unrelated comprehension decision was taken.
    """

    UNDERSTOOD = "understood"
    SPOKEN = "spoken"
    SIGNED = "signed"
    #: Conditioned on understood: ``q``, ``r``, ``p_any``, comprehension gaps.
    RATIO_OF_UNDERSTOOD = "ratio_of_understood"


def quantity_for_outcome(outcome: Any) -> ReportedQuantity:
    """Map a single-outcome model's ``Outcome`` to its reported quantity.

    ``definitions.Outcome`` has only ``SPOKEN`` and ``UNDERSTOOD``; this keeps
    the single-outcome engines from having to know the mapping, and raises on
    anything unexpected rather than silently reporting the whole grid.
    """
    name = getattr(outcome, "value", outcome)
    if name == "spoken":
        return ReportedQuantity.SPOKEN
    if name == "understood":
        return ReportedQuantity.UNDERSTOOD
    raise ValueError(f"No reporting quantity for outcome {outcome!r}")


def max_age_for(config: Any, quantity: ReportedQuantity) -> float | None:
    """Return the reporting age cap for ``quantity``, or ``None`` if uncapped.

    ``config`` is a model's ``ModelConfiguration``. ``None`` means "report the
    whole grid", which is the correct answer for the typically-developing models
    and for any model that declines to set a cap.
    """
    if quantity is ReportedQuantity.SPOKEN:
        ages = getattr(config, "ages_query", None)
        return float(max(ages)) if ages else None

    if quantity is ReportedQuantity.SIGNED:
        signed = getattr(config, "report_max_age_signed", None)
        if signed is not None:
            return float(signed)
        # A trivariate model that declines a signed cap falls back to the
        # comprehension cap rather than the full grid, because r(a) is a
        # fraction of understood and would otherwise outrun its denominator.
        understood = getattr(config, "report_max_age_understood", None)
        return None if understood is None else float(understood)

    # UNDERSTOOD and RATIO_OF_UNDERSTOOD share the comprehension cap.
    understood = getattr(config, "report_max_age_understood", None)
    return None if understood is None else float(understood)
