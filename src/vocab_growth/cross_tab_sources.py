# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The four study sources that carry a modality cross-tabulation, and their cells.

Each loader opens one raw CSV, derives that source's cross-tab cells, and splits its
rows into the ones that carry a usable composition and the ones that can only
contribute margins. This is per-study measurement knowledge -- the same class of
thing as :mod:`vocab_growth.data_utils`' defect rules -- and it lived inside the
joint engine's PyMC module until 2026-09-01, where a reader looking for "what does a
cross-tab source have to supply" had no reason to look.

**Adding a fifth source.** Write a ``load_<study>_*`` function here that:

1. reads only its own CSV from :data:`~vocab_growth.environment.DATA_DIR`, and cites
   its ``data/vocab_data_<study>.md`` note for anything not obvious from the columns;
2. returns ``(four_cell_df, marginal_df)`` -- the rows whose cells are complete,
   non-negative and reconcile with the recorded margins, and everything else, which
   still informs the model through its margins. Never drop a row for want of a
   composition; route it to the marginal set. (``load_nz01_produced_cells`` is the
   one exception, and returns a single frame: nz_01 is production-only, so a row
   with no produced words has no margins to contribute either.);
3. states which guards can fire on the source *as it is today* and which are held
   for what it may become -- ``load_uk07_four_cell``'s two currently cannot fire,
   and saying so is what stops a later reader deleting them as dead;
4. leaves the source's own column names in place. They are mapped to the
   analysis-frame cells by the assembly blocks in
   ``models/common_joint_modality.build_joint_analysis_frame``, which is also where
   the study is gated behind its definition field. Normalising the names here is a
   sensible next step but not a free one: the concatenated frame's schema and row
   order are hashed into every VG14/VG15 fit's ``analysis_frame_hash``.

The cell counts are validated at build time -- ``build_model`` requires each row's
four cells to sum to its recorded total -- so a margin substituted for a cell
raises. A ``signed_only`` / ``spoken_only`` swap preserves that sum and does not;
the direction of each source's derivation is stated in its docstring for that
reason.
"""

import os

import pandas as pd

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env

# The two sources with the four-cell within-understood cross-tabulation.
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

    A row missing any cell — in particular ``understood_only`` (some uk_02 rows
    record a produced sign/speech cross-tab but no comprehension total) — cannot
    form the within-understood four-way composition, so it is routed to the
    marginal-only set, where its recorded spoken/signed margins still inform the
    model. (Without this guard a NaN cell casts to a negative integer and trips
    the four-cell count validation in ``build_model``.)
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

    uk_07 records comprehension per item alongside a three-way *modality-exclusive*
    expressive coding — says-only, signs-only, both — so the fourth cell follows by
    subtraction: ``understood_only = understood - produced``, where ``produced`` is
    the source's own sum of the three expressive cells. That is the same
    within-understood partition uk_02 supplies, and it is what identifies psi.

    Two guards, mirroring ``load_uk02_four_cell``. A row whose production exceeds
    its comprehension has no non-negative ``understood_only`` cell, and a row with
    no understood words carries no composition; both are routed to the marginal
    set, where the recorded spoken/signed margins still inform the model. Neither
    fires on the current source: the one administration that would have failed the
    first is withheld before this point (see
    ``data_utils.UK07_WITHHELD_ADMINISTRATIONS``). They are kept so the guarantee
    holds for whatever the source becomes, rather than for what it is today.

    Returns ``(four_cell_df, marginal_df)`` with the any-modality marginals
    re-derived on the marginal rows exactly as ``vocab_combined`` does them.
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
    GESTURES ONLY — the last being what Galeote et al. (2011) describe as "total
    lexical production combining the two modalities". So the third column is a
    *total* (words gestured whether or not also spoken) and the fourth is a
    de-duplicated union, and the four cells follow by subtraction::

        understood_only = understood        - union
        spoken_only     = union             - gestured
        signed_only     = union             - spoken
        signed_spoken   = spoken + gestured - union

    which sum to ``understood`` identically. That the fourth column is a union
    rather than a disjoint cell is not an assumption: a disjoint reading forces
    ``union == spoken + gestured`` on every row, and 134 of the 186 Down syndrome
    rows have a union strictly smaller than that sum.

    Guards mirror ``load_uk07_four_cell``: a row with any negative cell, or with
    no understood words, carries no composition and is routed to the marginal set.
    One row of 186 fails (1 spoken, 15 gestured, union 11 — a union smaller than
    one of its parts, so ``spoken_only`` is negative); its comprehension and spoken
    marginals still inform the model, and its ``signed`` is masked there on the
    same reasoning the ``vocab_combined`` view applies.

    Returns ``(four_cell_df, marginal_df)``. Down syndrome children only — the
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
