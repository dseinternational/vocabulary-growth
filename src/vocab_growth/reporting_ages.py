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
understood                   72  Lowered from 84 on 2026-08-22. Not for
                                 want of data — the 72-84 band holds 25
                                 rows from 20 children across 5 studies —
                                 but because the number there is set by
                                 the child-effect structure rather than by
                                 the data. See below.
ratio of understood          72  ``q`` (spoken given understood), ``r``
                                 (fraction of understood signed) and
                                 ``p_any`` are all conditioned on
                                 understood, so they inherit its cap.
                                 ``q`` is the quantity that now *binds*
                                 it — see below.
signed                       84  Its own field since #212; see below.
spoken                       90  The top of the query grid.
==========================  ===  =========================================

Why comprehension stops at 72 rather than 84
--------------------------------------------
The 2026-08-13 raise to 84 asked the right question for its purpose — is the
72-84 band observed rather than extrapolated? — and answered it correctly. This
cap applies a second and stricter test that the raise did not: **is the number
in that band determined by the data, or by the model?**

VG19 and VG20 differ only in how a child departs from the population trajectory,
and are indistinguishable out of sample (k-fold LOSO, +0.93 SE over 767
children). They nevertheless put ``q`` at 0.75 against 0.85 at 72 months and
0.83 against 0.94 at 84 — gaps of 0.89 and 0.93 of VG20's *own* 89% ETI width.
Twenty-five comprehension observations cannot separate the two structures, so a
number quoted above 72 reports a modelling choice. Below 60 months the same
comparison never exceeds 0.15 interval widths.

Two consequences worth stating, because neither is obvious from the number.

**The binding quantity is now ``q``, not ``understood``.** The three models agree
on the understood curve itself to within 0.15 interval widths at every age to 84,
so understood alone would still support 84. It is trimmed with ``q`` because both
ride this one definition field, and giving ``q`` its own would invalidate all
seventeen models to express a cap this field already expresses correctly, if
conservatively. The policy is per quantity; the *mechanism* is not, here, and
this is where that costs something.

**"Enough data" is no longer the test for raising it again.** The trigger is
whether the 72-84 band can *distinguish* the child structures, which is answered
by rerunning ``scripts/experiments/model_dependence_of_reported_quantities.py``
on new older-child comprehension data, not by recounting rows.

Recorded in ``notes/202608221200-reporting-source-by-quantity.md``.

Why ``p_any`` stops at the tighter of its components
-----------------------------------------------------
``p_any`` is a union over speaking and signing, whose caps differ (90 and 84).
It takes the **tighter** of the two: past the signing cap one of its two
components is no longer reported, and a union of a reported and an unreported
quantity is not a quantity this project publishes. Study owner, 2026-08-16.

The two arguments used to agree — both gave 84 — and they no longer do. On
2026-08-22 the comprehension cap moved to 72 without the signing cap moving with
it, which is precisely the case this paragraph was written to anticipate. The
conditioning rule now gives 72 and the components rule 84, so **the conditioning
rule binds** and ``p_any`` stops at 72, which is what ``max_age_for`` returns
because ``p_any`` is a ``RATIO_OF_UNDERSTOOD``. Note the anticipation was
half-right: it assumed the components rule would be the binding one, which holds
only if the comprehension cap moves *up*. Whichever is tighter binds. Stated
here because it is a reporting decision, not an arithmetic consequence: it was
what let VG14's modality figure run to 115 months above a ``p_any`` table
trimmed to 84 without anything objecting.

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

Why the typically-developing models are not exempt
--------------------------------------------------
This section used to read: "Typically-developing models are unaffected either
way: ``TD_POOL_AGE_MONTHS`` is ``(8, 30)`` and their GP domains stop at 30 or 18,
so no cap here can bind." That is the mistake this whole module exists to
prevent, made about itself. ``TD_POOL_AGE_MONTHS`` is a property of the **pool**,
and the policy above is deliberately per **quantity** — because a pool's outcomes
do not share a support.

They do not share one here. Comprehension rides only on
``WORDBANK_BIVARIATE_FORMS``: on the other forms it is a production proxy, so
``load_data`` restricts to those forms whenever ``understood`` is requested. Those
forms stop at **25** months. Production keeps WS and does reach 30. So the pool
window is honest for spoken and five months too generous for understood, and
VG04 and VG12 published 27- and 30-month comprehension medians on zero
observations for as long as the exemption stood — while their own figures, which
are drawn over the observed support rather than the query grid, already stopped
at 25. Both now declare ``report_max_age_understood = 25`` (#228).

The general form, worth keeping in view whenever a scope rule is written: a
restriction justified by a property of the pool is only sound if that property
holds for every outcome the pool carries.

``TD_POOL_AGE_MONTHS``'s own upper bound was checked at the same time and left
alone. Extending it to 36 would admit 150 further spoken rows of 18,987 (0.8%,
from two datasets) and would require widening ``_TD_GP_DOMAIN_MONTHS``, which
VG03 and VG04 share — the same trade, and the same answer, as the five 7-month
Italian rows that gave the pool its lower bound.
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
