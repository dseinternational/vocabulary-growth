# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reporting-age limits by quantity.

Understood counts and shares of understood words use the definition's
``report_max_age_understood``. Signed counts use ``report_max_age_signed``
when set, otherwise the comprehension cap. Spoken counts use the upper end of
``ages_query``. Sign-bearing shares and the expressive union use the tighter
of the comprehension and signing caps.

Current DS reporting limits are 72 months for comprehension and its shares,
84 for signed counts and 90 for spoken counts. The comprehension cap also
limits the spoken share because older-age estimates depend on the assumed
child-effect structure. See ``notes/202608221200-reporting-source-by-quantity.md``
and its September successor check. A cap does not establish adequate data at
every age below it.

TD limits also depend on the outcome and model. VG04 and VG12 cap comprehension
at 25 months because the forms with genuine comprehension measurements stop
there; production-only forms extend further. Joint TD models use shorter
windows. Pool-wide age limits alone cannot establish support for each outcome.

Declared caps are part of the serialised definition. Spoken uses the existing
query grid to avoid adding a definition field solely for its current cap.
A separate spoken limit would require a compatibility decision for stored fits.
Changing a cap affects post-processing, not posterior sampling, but invalidates
summary outputs recorded under the old definition. See ``docs/models/README.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "ReportedQuantity",
    "max_age_for",
    "max_age_for_sign_ratio",
    "quantity_for_outcome",
]


class ReportedQuantity(Enum):
    """What a figure or table reports, for the purpose of trimming its ages.

    Call sites name the quantity so each outcome uses the appropriate cap.
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

    ``config`` supplies the definition's age grid and reporting caps. ``None``
    means report the whole grid for a quantity with no applicable cap.
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


def max_age_for_sign_ratio(config: Any) -> float | None:
    """The cap for sign-bearing ratios of understood: ``r``, ``p_any``.

    These quantities depend on both comprehension and signing, so the tighter
    cap applies. ``q`` involves no signing and instead takes
    ``max_age_for(config, ReportedQuantity.RATIO_OF_UNDERSTOOD)`` alone.
    """
    caps = [
        cap
        for cap in (
            max_age_for(config, ReportedQuantity.RATIO_OF_UNDERSTOOD),
            max_age_for(config, ReportedQuantity.SIGNED),
        )
        if cap is not None
    ]
    return min(caps) if caps else None
