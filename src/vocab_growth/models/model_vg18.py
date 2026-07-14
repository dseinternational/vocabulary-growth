# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG18: study-adjusted contrast of DS *total expressive production* by sign-group.

Identical structure to VG17 (trend + HSGP + study REs + a 3-level sign-group
covariate, Beta-Binomial(810), 12-66 mo, VG01 priors), but the outcome is the
recorded ``produced`` count — the union of words a child can express by speech
OR sign (it de-duplicates words known in both modalities; where all of
spoken/signed/produced are present, ``produced`` always lies between ``spoken``
and ``spoken + signed``).

CAUTION — ``produced`` is NOT a uniform total-expressive measure across studies:
  * uk_01, uk_02, nz_01: ``produced`` is a de-duplicated UNION (each word once).
    - uk_01: ``produced`` is the study's own total-production column, defined in the
      write-up as "vocalised and signed-only words" (spoken PLUS words signed-but-not-
      spoken); so uk_01's ``signed`` column is the *signed-only* count and
      ``produced == spoken + signed`` is the correct union (NOT a double-count).
    - uk_02, nz_01: built from mutually-exclusive says-only / signs-only / both cells.
  * ie_02, uk_04, uk_05, uk_06: ``produced`` := ``spoken`` — signs EXCLUDED.
  * unknown group (no sign data): ``produced`` reflects spoken-only production.

NB the ``signed`` column is itself inconsistent across studies (uk_01 = signed-ONLY;
uk_02/nz_01 = total signed incl. both) — relevant to the VG14/VG15 signed-ratio models.

So the family-wide ``produced`` mixes true-union and spoken-only definitions; the VG18
signer-vs-non-signer estimate is heterogeneous. For a clean de-duplicated total-
expressive contrast, restrict to the union studies uk_01 + uk_02 + nz_01
(fit(..., studies=("uk_01","uk_02","nz_01"))), or use VG15's modelled ``p_any``
(estimates the sign/speech overlap ``psi``). Exploratory; not in MODEL_REGISTRY.
"""

from vocab_growth.models import model_vg17


def fit(config: str = "test", studies=None):
    """Fit VG18. Pass studies=("uk_02", "nz_01") for the clean de-duplicated-union
    total-expressive analysis (excludes uk_01, whose produced double-counts, and the
    signs-excluded studies)."""
    subdir = "VG18-age-produced-ds-signgroup"
    if studies is not None:
        subdir += "-" + "-".join(studies)
    return model_vg17.fit(
        config,
        outcome="produced",
        label="VG18",
        subdir=subdir,
        studies=studies,
    )


if __name__ == "__main__":
    import sys
    from multiprocessing import freeze_support

    from vocab_growth import environment as env

    freeze_support()
    env.set_output_root("/scratch/vg-output")
    fit(sys.argv[1] if len(sys.argv) > 1 else "test")
