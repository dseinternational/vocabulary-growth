# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Load study-specific sign-speech cross-tabulations and marginal-only rows.

Each loader reads its source CSV and applies its measurement rules. New sources
should cite their ``data/vocab_data_<study>.md`` record, document cell derivations
and retain rows without a usable composition for their available marginals.
The production-only ``nz_01`` loader returns one frame and drops zero-production
rows because they contribute neither composition nor marginal likelihoods.

The joint engine maps source columns into analysis-frame cells and controls
which sources enter each model. Changing that mapping or the frame's row order
changes the prepared-frame hash and can invalidate stored fits.

Model construction validates cell counts and totals. A swap of signed-only and
spoken-only cells would preserve the total, so each loader documents their
direction explicitly.
"""

import os

import pandas as pd

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env

# Three sources supply a four-cell within-understood cross-tabulation.
UK02_STUDY_ID = "uk_02"
# uk_02 ran two instruments. Its `form` column separates them, and only the DSE
# arm is native to the 810-item reference (the other is the 416-item Oxford CDI),
# which the DSE-native sensitivity needs to tell apart.
UK02_DSE_FORM = "DSE"
# uk_07 (PACT-DS) records comprehension alongside modality-exclusive expressive
# cells, so its four cells are derivable: understood_only = understood - produced.
UK07_STUDY_ID = "uk_07"
# es_01 (Galeote) records comprehension, a spoken total, a symbolic-gesture total
# and their recorded union, from which the same four cells follow by subtraction.
ES01_STUDY_ID = "es_01"
# nz_01 (Foster-Cohen) carries a production-only three-cell (within-produced)
# cross-tabulation: word-only, sign-only, both. No comprehension.
NZ01_STUDY_ID = "nz_01"


def load_uk02_four_cell():
    """Load uk_02 rows, split into four-cell (cross-tab) and marginal-only rows.

    Returns (four_cell_df, marginal_df). The four-cell rows are those that have
    all four cell counts recorded, whose signed and spoken margins reconcile
    with the cross-tab cells (signed == signed_only + signed_spoken,
    spoken == spoken_only + signed_spoken) and whose four cells sum to a
    positive total; they identify psi. For these rows the four-cell sum is
    treated as the authoritative understood total, so a small mismatch between
    the raw comprehension column and the cross-tab partition does not make the
    U likelihood and the Dirichlet-Multinomial likelihood disagree. The rest are
    marginal-only uk_02 rows (no usable cross-tab).

    Rows missing any cell cannot form the within-understood composition. They
    retain their recorded spoken and signed margins in the marginal-only set.
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_uk_02.csv")
    raw = pd.read_csv(path)
    cells = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
    raw["cell_total"] = raw[cells].sum(axis=1)
    reconciles = (
        raw[cells].notna().all(axis=1)
        & (raw["signed"] == raw["signed_only"] + raw["signed_spoken"])
        & (raw["spoken"] == raw["spoken_only"] + raw["signed_spoken"])
        & (raw["cell_total"] > 0)
    )
    four = raw[reconciles].copy()
    marg = raw[~reconciles].copy()
    return four, marg


def load_uk07_four_cell():
    """Load uk_07 (PACT-DS) rows as a four-cell within-understood cross-tab.

    uk_07 records comprehension per item alongside three modality-exclusive
    expressive cells (says-only, signs-only, both), so the fourth cell follows by
    subtraction: ``understood_only = understood - produced``, where ``produced`` is
    the source's own sum of the three expressive cells. That is the same
    within-understood partition uk_02 supplies, and it is what identifies psi.

    Incomplete rows, rows with production above comprehension and rows with no
    understood words enter the marginal-only set. These guards also apply to
    future source revisions. The known incoherent administration is withheld
    before the split (see ``data_utils.UK07_WITHHELD_ADMINISTRATIONS``).

    Returns ``(four_cell_df, marginal_df)`` with the source's exclusive cells.
    The joint engine derives any-modality marginals from those cells.
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_uk_07.csv")
    raw, _withheld = vocab_data_utils.drop_uk07_withheld_administrations(
        pd.read_csv(path)
    )
    raw["understood_only"] = raw["understood"] - raw["produced"]
    usable = (
        raw[["understood", "produced", "spoken", "signed", "spoken_signed"]]
        .notna()
        .all(axis=1)
        & (raw["understood_only"] >= 0)
        & (raw["understood"] > 0)
    )
    four = raw[usable].copy()
    marg = raw[~usable].copy()
    return four, marg


def load_es01_four_cell():
    """Load es_01 (Galeote) rows as a four-cell within-understood cross-tab.

    es_01 records four totals per child. In the original table they are labelled
    TOTAL COMPREHENSIÓN, TOTAL PRODUCTION, TOTAL GESTURES and WORD PRODUCED +
    GESTURES ONLY, the last being what Galeote et al. (2011) describe as "total
    lexical production combining the two modalities". So the third column is a
    *total* (words gestured whether or not also spoken) and the fourth is a
    de-duplicated union, and the four cells follow by subtraction::

        understood_only = understood        - union
        spoken_only     = union             - gestured
        signed_only     = union             - spoken
        signed_spoken   = spoken + gestured - union

    The cells sum to ``understood``. The interpretation as a union agrees with
    the source description and its overlapping spoken and gestured totals.

    Guards mirror ``load_uk07_four_cell``: a row with any negative cell, or with
    no understood words, carries no composition and is routed to the marginal set.
    The row with 1 spoken, 15 gestured and a union of 11 has a negative
    ``spoken_only`` cell. The joint engine retains its comprehension and spoken
    marginals and masks signing, as the ``vocab_combined`` view does.

    Returns ``(four_cell_df, marginal_df)`` for Down syndrome children only. The
    matched typically developing group stays out of this relation, as it does in
    the view.
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_es_01.csv")
    raw = pd.read_csv(path)
    raw = raw[raw["group"] == "DS"].copy()

    union = raw["spoken_or_gestured"]
    raw["understood_only"] = raw["understood"] - union
    raw["spoken_only"] = union - raw["gestured"]
    raw["signed_only"] = union - raw["spoken"]
    raw["signed_spoken"] = raw["spoken"] + raw["gestured"] - union

    cells = ["understood_only", "spoken_only", "signed_only", "signed_spoken"]
    usable = (
        raw[["understood", "spoken", "gestured", "spoken_or_gestured"]]
        .notna()
        .all(axis=1)
        & (raw[cells] >= 0).all(axis=1)
        & (raw["understood"] > 0)
    )
    four = raw[usable].copy()
    marg = raw[~usable].copy()
    return four, marg


def load_nz01_produced_cells():
    """Load nz_01 (Foster-Cohen) rows as a within-produced three-cell cross-tab.

    nz_01 is production-only (no comprehension). Its checklist codes partition ALL
    items into word-only (a), sign-only (b), both (c) and neither (d). The three
    produced cells {a, b, c} form a modality cross-tab *conditioned on production*,
    not on comprehension: nz_01 records no understood total, and its "neither"
    mixes understood-but-unproduced with not-understood, so it cannot fill uk_02's
    ``understood_only`` cell. Conditioning on produced cancels that cell (and the
    understood level), so these rows identify psi/q/r through a three-cell
    Dirichlet-Multinomial (see ``build_model``). Rows with no produced words
    (``prod_total == 0``) carry no composition and are dropped.
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_nz_01.csv")
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "study": NZ01_STUDY_ID,
            "age": raw["age"].to_numpy(dtype=float),
            "subject_id": raw["subject_id"].to_numpy(),
            # CSV columns are modality-exclusive: spoken=word-only, signed=sign-only,
            # spoken_signed=both. Marginal understood/spoken/signed stay NaN so these
            # rows feed only the produced DM (no double counting).
            "prod_spoken_only": raw["spoken"].to_numpy(dtype=float),
            "prod_signed_only": raw["signed"].to_numpy(dtype=float),
            "prod_signed_spoken": raw["spoken_signed"].to_numpy(dtype=float),
        }
    )
    out["prod_total"] = (
        out["prod_spoken_only"] + out["prod_signed_only"] + out["prod_signed_spoken"]
    )
    return out[out["prod_total"] > 0].reset_index(drop=True)
