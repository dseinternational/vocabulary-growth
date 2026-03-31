# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import duckdb

import vocab_growth.environment as local_env

VOCABULARY_DATA_PATH = os.path.join(local_env.DATA_DIR, "vocabulary.duckdb")


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

    con = duckdb.connect(VOCABULARY_DATA_PATH)

    df = con.execute(
        f"""
        SELECT
            study,
            sex,
            age,
            understood,
            spoken,
            signed
        FROM vocab_combined
        WHERE age <= {max_age_months if max_age_months is not None else 1200}
        """
    ).df()

    con.close()

    return df
