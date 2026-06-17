# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import os

import duckdb
import pandas as pd

import vocab_growth.environment as local_env
from vocab_growth.models.definitions import Population

VOCABULARY_DATA_PATH = os.path.join(local_env.DATA_DIR, "vocabulary.duckdb")

TD_BIVARIATE_FORMS = ("Oxford CDI", "WG")
"""Wordbank TD forms with independent comprehension and production measures."""

TD_SPOKEN_ONLY_FORMS = ("WS",)
"""Wordbank TD forms that contribute production observations only."""

ENGLISH_LANGUAGES = (
    "English (American)",
    "English (Australian)",
    "English (British)",
    "English (Irish)",
)
"""Wordbank ``language`` values treated as English — the current default scope.

The ``wordbank_child`` table now holds the full multi-language Wordbank export.
Queries restrict to these English variants by default; pass a wider ``languages``
set (or ``None`` for all languages) to the loaders to widen the scope later.
"""


def filter_studies_by_min_obs(
    df: pd.DataFrame,
    min_obs: int | None,
    study_col: str = "study",
) -> tuple[pd.DataFrame, list[str]]:
    """Drop studies (datasets) with fewer than ``min_obs`` observations.

    Used by the study-random-intercept models (e.g. VG11/VG12/VG13) to trim
    tiny studies that would otherwise each add a near-unidentified intercept
    without materially informing the estimates.

    Parameters
    ----------
    df
        Analysis frame (one row per observation), already filtered to the rows
        that inform the model.
    min_obs
        Minimum observations a study must have to be kept. ``None`` or ``0``
        keeps every study.
    study_col
        Column identifying the study/dataset grouping.

    Returns
    -------
    tuple[pandas.DataFrame, list[str]]
        The filtered frame (index reset) and the sorted list of dropped study
        labels.
    """
    if not min_obs:
        return df.reset_index(drop=True), []
    sizes = df.groupby(study_col).size()
    keep = sizes[sizes >= min_obs].index
    dropped = sorted(set(sizes.index) - set(keep))
    filtered = df[df[study_col].isin(keep)].reset_index(drop=True)
    return filtered, dropped


def load_combined_data(max_age_months=None):
    """
    Load the combined data from the DuckDB database.

    (Run ./scripts/prepare_data.py to create the database if it doesn't exist.)

    Parameters:
    -----------
        max_age_months (int, optional): The maximum age in months to include in the data. Defaults to None (no limit).

    Returns:
    --------
        pd.DataFrame: The combined data as a DataFrame.
    """
    age_limit = max_age_months if max_age_months is not None else 1200

    with duckdb.connect(VOCABULARY_DATA_PATH) as con:
        df = con.execute(
            """
            SELECT
                study,
                subject_id,
                sex,
                age,
                understood,
                spoken,
                signed
            FROM vocab_combined
            WHERE age <= $1
            """,
            [age_limit],
        ).df()

    return df


def load_data(
    population: Population,
    columns: list[str],
    sample_fraction: float = 1.0,
    random_seed: int = 47,
    max_age_months: int | None = None,
    languages: tuple[str, ...] | None = ENGLISH_LANGUAGES,
) -> pd.DataFrame:
    """
    Load vocabulary data for the specified population.

    Parameters
    ----------
    population : Population
        Which population to load data for.
    columns : list[str]
        Columns to select (e.g. ["age", "spoken"] or ["age", "understood", "spoken"]).
        For DS the ``study`` and ``subject_id`` columns are available.
        For TD the ``study`` column is aliased from ``dataset_name`` (the Wordbank
        dataset/lab identifier), along with ``subject_id``, ``form`` and ``language``.
    sample_fraction : float
        Fraction of data to subsample (TD only). 1.0 = no subsampling.
    random_seed : int
        Random seed for subsampling.
    max_age_months : int | None
        Upper bound on age (inclusive, months). None means no upper bound.
    languages : tuple[str, ...] | None
        Wordbank ``language`` values to include (TD only). Defaults to
        :data:`ENGLISH_LANGUAGES`. Pass a wider tuple to broaden the scope, or
        ``None`` to include all languages. Ignored for DS (the DS subset is
        fixed to English when the database is built).

    Returns
    -------
    pd.DataFrame
        DataFrame with the requested columns.
    """
    if population == Population.DOWN_SYNDROME:
        df = load_combined_data(max_age_months=max_age_months)
        return df[columns]

    # Typically developing — query wordbank_child directly.
    #
    # Wordbank's CDI: Words & Sentences (WS) rows contain valid production
    # counts, but their comprehension column is a production proxy. Keep WG
    # and Oxford CDI as bivariate observations, and include WS only when the
    # requested model can use spoken observations.
    needs_understood = "understood" in columns
    needs_spoken = "spoken" in columns

    td_forms = list(TD_BIVARIATE_FORMS)
    if needs_spoken or not needs_understood:
        td_forms.extend(TD_SPOKEN_ONLY_FORMS)

    age_upper = max_age_months if max_age_months is not None else 30

    params: list = [td_forms, age_upper]
    language_clause = ""
    if languages is not None:
        params.append(list(languages))
        language_clause = f"AND language IN ${len(params)}"

    with duckdb.connect(VOCABULARY_DATA_PATH) as con:
        td_df = (
            con.execute(
                f"""
            SELECT
                form,
                language,
                dataset_name                       as study,
                concat('id_', hex(hash(child_id))) as subject_id,
                age,
                CASE
                    WHEN form IN ('Oxford CDI', 'WG') THEN comprehension
                    ELSE NULL
                END                                as understood,
                production                         as spoken,
                typically_developing,
                health_conditions
            FROM wordbank_child
            WHERE typically_developing = true
                AND age <= $2
                AND health_conditions IS NULL
                AND form IN $1
                {language_clause}
            """,
                params,
            )
            .df()
        )

    if sample_fraction < 1.0:
        td_df = (
            td_df.sample(frac=sample_fraction, random_state=random_seed)
            .reset_index(drop=True)
        )

    return td_df[columns]
