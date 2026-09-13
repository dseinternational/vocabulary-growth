# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recover inventory sizes for the joint model's cross-form lag check.

Inventory size distinguishes the forms used within each current study. Equal
sizes alone do not establish measurement equivalence between studies. Unknown
or ambiguous provenance remains missing, so it cannot certify a same-form lag.
"""

import numpy as np
import pandas as pd

from vocab_growth import data_utils
from vocab_growth.cross_tab_sources import UK02_STUDY_ID, load_uk02_four_cell


def _unique_size(values) -> float:
    sizes = pd.Series(values).dropna().unique()
    return float(sizes[0]) if len(sizes) == 1 else np.nan


def joint_inventory_sizes(frame: pd.DataFrame, definition) -> np.ndarray:
    """Return one inventory size per row without changing the prepared frame.

    Ordinary rows match the loader's study, child and age metadata. The UK02
    cross-tab path replaces comprehension with the cell total, so those rows
    match its source loader on all the counts the joint frame actually retains.
    """
    metadata = data_utils.load_data(
        population=definition.population,
        columns=["study", "subject_id", "age", "survey_vocab_max"],
        include_implausible_production=definition.include_implausible_production,
        include_same_day_disagreements=definition.include_same_day_disagreements,
    )
    keys = ["study", "subject_id", "age"]
    metadata["subject_id"] = metadata["subject_id"].astype(str)
    rows = frame[keys].copy()
    rows["subject_id"] = rows["subject_id"].astype(str)
    by_wave = metadata.groupby(keys, dropna=False)["survey_vocab_max"].agg(_unique_size)
    joined = rows.merge(
        by_wave, on=keys, how="left", validate="many_to_one", sort=False
    )
    by_study = metadata.groupby("study")["survey_vocab_max"].agg(_unique_size)
    sizes = (
        joined["survey_vocab_max"]
        .fillna(joined["study"].map(by_study))
        .to_numpy(copy=True)
    )

    uk02 = frame["study"].eq(UK02_STUDY_ID).to_numpy()
    if not uk02.any():
        return sizes
    four, marginal = load_uk02_four_cell()
    four = four.assign(understood=four["cell_total"], spoken=np.nan, signed=np.nan)
    marginal = marginal.assign(understood=marginal["comprehension"])
    cell_columns = [
        "understood_only",
        "signed_only",
        "spoken_only",
        "signed_spoken",
        "cell_total",
    ]
    marginal[cell_columns] = np.nan
    source = pd.concat([four, marginal], ignore_index=True)
    # The same form labels and sizes appear in data_utils' vocab_combined view.
    form_sizes = {"DSE": 810.0, "Oxford_CDI": 416.0}
    unknown = set(source["form"].dropna()) - form_sizes.keys()
    if unknown:
        raise ValueError(
            f"UK02 has unrecognised forms; check their inventories: {sorted(unknown)}."
        )
    source["survey_vocab_max"] = source["form"].map(form_sizes)
    keys = ["subject_id", "age", "understood", "spoken", "signed", *cell_columns]
    source["subject_id"] = source["subject_id"].astype(str)
    candidates = source.groupby(keys, dropna=False)["survey_vocab_max"].agg(
        _unique_size
    )
    rows = frame.loc[uk02, keys].copy()
    rows["subject_id"] = rows["subject_id"].astype(str)
    matched = rows.merge(
        candidates, on=keys, how="left", validate="many_to_one", sort=False
    )
    sizes[uk02] = matched["survey_vocab_max"].to_numpy()
    return sizes
