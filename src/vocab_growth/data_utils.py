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
        dataset/lab identifier), along with ``subject_id`` and ``form``.
    sample_fraction : float
        Fraction of data to subsample (TD only). 1.0 = no subsampling.
    random_seed : int
        Random seed for subsampling.
    max_age_months : int | None
        Upper bound on age (inclusive, months). None means no upper bound.

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

    with duckdb.connect(VOCABULARY_DATA_PATH) as con:
        td_df = (
            con.execute(
                """
            SELECT
                form,
                dataset_name,
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
            """,
                [td_forms, age_upper],
            )
            .df()
        )

    if sample_fraction < 1.0:
        td_df = (
            td_df.sample(frac=sample_fraction, random_state=random_seed)
            .reset_index(drop=True)
        )

    return td_df[columns]
