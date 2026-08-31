# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG18: study-adjusted contrast of DS *total expressive production* by sign-group.

CAUTION — THE SIGN-GROUP CONTRAST IS PARTLY MECHANICAL. Sign group is derived from
``signed``, and ``signed`` is a *component* of this model's ``produced`` outcome for
every union study (uk_01, uk_02, nz_01, es_01, uk_07, where ``produced`` counts words
a child can express by speech OR sign). A child is therefore classified a signer
*because* of the very words that raise its outcome: a signer with any signed-only
word has a ``produced`` count that mechanically exceeds their ``spoken`` count, and
the non-signer group is by construction the ``signed == 0`` group whose ``produced``
equals ``spoken``. Part of any positive signer-vs-non-signer estimate here is that
identity, not an effect of signing. VG18 is DESCRIPTIVE: read it as "how much larger
is total expressive production in the signing group", never as "how much does signing
increase vocabulary". For the causally interpretable comparison use VG17, whose
``spoken`` outcome does not contain ``signed``; for the modelled sign/speech overlap
use VG15's ``psi``.

Identical structure to VG17 (trend + HSGP + study REs + child REs + a 3-level
sign-group covariate, Beta-Binomial(810), 12-66 mo, VG01 priors), but the outcome is the
recorded ``produced`` count — the union of words a child can express by speech
OR sign (it de-duplicates words known in both modalities; where all of
spoken/signed/produced are present, ``produced`` always lies between ``spoken``
and ``spoken + signed``).

CAUTION — ``produced`` is NOT a uniform total-expressive measure across studies:
  * uk_01, uk_02, nz_01, es_01, uk_07: ``produced`` is a de-duplicated UNION (each
    word once).
    - uk_01: ``produced`` is the study's own total-production column, defined in the
      write-up as "vocalised and signed-only words" (spoken PLUS words signed-but-not-
      spoken); so uk_01's ``signed`` column is the *signed-only* count and
      ``produced == spoken + signed`` is the correct union (NOT a double-count).
    - uk_02, nz_01, uk_07: built from mutually-exclusive says-only / signs-only /
      both cells.
    - es_01: the source records the spoken-or-gestured union outright, so it is taken
      as given rather than reconstructed. Its non-vocal modality is a *symbolic*
      (referential) gesture lexicon scored per word, read here as ``signed``.
  * ie_02, uk_04, uk_05, uk_06: ``produced`` := ``spoken`` — signs EXCLUDED.
  * unknown group (no sign data): ``produced`` reflects spoken-only production.

NB the ``signed`` column is itself inconsistent across studies (uk_01 = signed-ONLY;
uk_02/nz_01/es_01/uk_07 = total signed incl. both) — relevant to the VG14/VG15
signed-ratio models.

So the family-wide ``produced`` mixes true-union and spoken-only definitions; the VG18
signer-vs-non-signer estimate is heterogeneous. For a clean de-duplicated total-
expressive contrast, restrict to the union studies uk_01 + uk_02 + nz_01 + es_01 +
uk_07 (fit(..., studies=("uk_01","uk_02","nz_01","es_01","uk_07"))), or use VG15's
modelled ``p_any`` (estimates the sign/speech overlap ``psi``). Note that uk_07's
34-95 month span sits largely outside VG17/VG18's 12-66 month window, so it
contributes only its younger assessments. Exploratory; not in MODEL_REGISTRY. Its output is not validatable and must not be published -- see `vocab_growth.models.exploratory` for what it does not carry.
"""

from vocab_growth.models.exploratory import vg17

CAUTION = (
    "CAUTION: the sign-group contrast is PARTLY MECHANICAL. Sign group is derived from "
    "`signed`, which is a COMPONENT of the `produced` outcome for the union studies "
    "(uk_01, uk_02, nz_01, es_01, uk_07): a child is classified a signer because of the "
    "very words that raise their outcome, and the non-signer group is by construction the "
    "`signed == 0` group whose `produced` equals `spoken`. Part of any positive "
    "signer-vs-non-signer estimate below is that identity, not an effect of signing. "
    "VG18 is DESCRIPTIVE - read it as 'how much larger is total expressive production in "
    "the signing group', NEVER as 'how much does signing increase vocabulary'. Use VG17 "
    "(spoken outcome, does not contain `signed`) for the causally interpretable contrast, "
    "or VG15's modelled overlap `psi`."
)


def fit(config: str = "test", studies=None):
    """Fit VG18. Pass studies=("uk_02", "nz_01", "es_01", "uk_07") for the clean
    de-duplicated-union total-expressive analysis (excludes uk_01, whose produced
    double-counts, and the signs-excluded studies).

    Read :data:`CAUTION` (also printed by every fit): the sign-group contrast is
    partly an identity, because sign group is derived from a component of the
    outcome. The contrast is retained deliberately, as a description.
    """
    subdir = "VG18-age-produced-ds-signgroup"
    if studies is not None:
        subdir += "-" + "-".join(studies)
    return vg17.fit(
        config,
        outcome="produced",
        label="VG18",
        subdir=subdir,
        studies=studies,
        caution=CAUTION,
    )


if __name__ == "__main__":
    import sys
    from multiprocessing import freeze_support

    from vocab_growth import environment as env

    freeze_support()
    env.set_output_root("/scratch/vg-output")
    fit(sys.argv[1] if len(sys.argv) > 1 else "test")
