# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exploratory VG18: total production by recorded signing group.

Output from this custom fit path must not be published.
See vocab_growth.models.exploratory for its missing validation artefacts.

The structure follows VG17 but uses produced rather than spoken counts.
CAUTION: where produced is a speech-or-sign union, signing contributes to the outcome
used to compare signing groups. The contrast is therefore partly mechanical
and does not estimate an effect of signing.

Produced has different definitions across sources. Some record the union;
others record spoken words only. uk_01's signed field counts signed-only words,
so adding it to spoken gives a valid union without duplication. The shared
preparation masks that field for total-signing group classification, leaving
uk_01 in the unknown group. Unknown signing status therefore does not imply
a spoken-only produced outcome.

es_01 records item-specific symbolic gestures, including spontaneous gestures
and taught signs. The grouping does not establish construct equivalence with
other sources. Restricting studies changes the sample as well as the outcome
definition. All comparisons remain descriptive. VG15 models overlap through
psi rather than a signing-group contrast.
"""

from vocab_growth.models.exploratory import vg17

CAUTION = (
    "CAUTION: the sign-group contrast is partly mechanical. In the union studies "
    "uk_02, nz_01, es_01 and uk_07, the group comes from `signed`, while `produced` "
    "also includes signed words. The grouping and outcome share item counts. The "
    "recorded non-signer group has `signed == 0`, so its `produced` equals `spoken`. "
    "uk_01 is in the unknown group because it records signed-only words; its "
    "`produced` still includes those words. Unknown status does not imply a "
    "spoken-only outcome. VG18 is descriptive and does not estimate an effect of "
    "signing. VG17's spoken outcome gives a separate descriptive contrast; "
    "VG15 models speech-sign overlap through `psi`."
)


def fit(config: str = "test", studies=None):
    """Fit the exploratory total-production contrast with an optional study subset.

    For a union-only contrast with recorded total signing, pass studies containing
    uk_02, nz_01, es_01 and uk_07. uk_01 also has a valid union: its signed-only
    count can be added to spoken without duplication. Its total signing status
    cannot be recovered, so the shared preparation assigns it to the unknown group.

    Read CAUTION, which each fit prints. Group membership is derived from a
    component of the outcome, so the contrast is partly mechanical.
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

    freeze_support()
    fit(sys.argv[1] if len(sys.argv) > 1 else "test")
