# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The cross-form alignment table behind the methods chapter's nesting check.

``td_form_alignment_table`` tabulates, per age and form, the median count as
recorded and as a proportion of the form's own word-item count;
``form_alignment_spread`` says which scale the form medians agree on more
closely. The synthetic frame below is built so that the answer is known: at 10
months the two forms record the same raw medians (what nesting predicts), at
11 months the same proportions (what proportional sampling predicts).
"""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.data_utils import WORDBANK_FORM_ITEMS
from vocab_growth.descriptive import form_alignment_spread, td_form_alignment_table

ITEMS = {("Long", "WG"): 400, ("Short", "WG"): 300}


def _frame() -> pd.DataFrame:
    rows = []
    # 10 months: identical raw counts on both forms.
    for count in (10, 20, 30):
        rows.append(("Long", "WG", 10, count))
        rows.append(("Short", "WG", 10, count))
    # 11 months: identical proportions (10% of each form).
    for _ in range(3):
        rows.append(("Long", "WG", 11, 40))
        rows.append(("Short", "WG", 11, 30))
    # Outside the window, or with no count: must not contribute.
    rows.append(("Long", "WG", 7, 5))
    rows.append(("Short", "WG", 16, 200))
    rows.append(("Short", "WG", 10, np.nan))
    return pd.DataFrame(rows, columns=["language", "form", "age", "understood"])


def test_table_medians_on_both_scales():
    table = td_form_alignment_table(_frame(), age_range=(8, 15), form_items=ITEMS)
    assert list(table.columns) == [
        "age", "language", "form", "n_items", "n", "median_count", "median_proportion",
    ]
    assert sorted(table["age"].unique()) == [10, 11]
    at10 = table[table["age"] == 10].set_index("language")
    assert at10.loc["Long", "median_count"] == 20 and at10.loc["Short", "median_count"] == 20
    assert at10.loc["Long", "median_proportion"] == pytest.approx(20 / 400)
    assert at10.loc["Short", "median_proportion"] == pytest.approx(20 / 300)
    assert at10.loc["Short", "n"] == 3  # the NaN row is dropped, not counted
    at11 = table[table["age"] == 11].set_index("language")
    assert at11.loc["Long", "median_proportion"] == pytest.approx(0.10)
    assert at11.loc["Short", "median_proportion"] == pytest.approx(0.10)
    assert (table["n_items"] == table["language"].map({"Long": 400, "Short": 300})).all()


def test_spread_names_the_tighter_scale_and_the_lowest_form():
    spread = form_alignment_spread(
        td_form_alignment_table(_frame(), form_items=ITEMS)
    ).set_index("age")
    assert spread.loc[10, "cv_count"] == 0 and spread.loc[10, "cv_proportion"] > 0
    assert spread.loc[10, "tighter"] == "count"
    assert spread.loc[11, "cv_proportion"] == pytest.approx(0) and spread.loc[11, "cv_count"] > 0
    assert spread.loc[11, "tighter"] == "proportion"
    assert spread.loc[11, "lowest_count_form"] == "Short WG"
    assert (spread["n_forms"] == 2).all()


def test_unknown_form_raises_rather_than_guessing_a_ceiling():
    frame = _frame()
    frame.loc[frame["language"] == "Short", "language"] = "Unlisted"
    with pytest.raises(KeyError, match="Unlisted"):
        td_form_alignment_table(frame, form_items=ITEMS)


def test_default_item_counts_are_the_project_ceilings():
    # The admitted comprehension forms, at the ceilings the rest of the
    # codebase scores them against (Oxford CDI 416, not the 418 word rows in
    # the Wordbank definition file).
    assert WORDBANK_FORM_ITEMS[("English (American)", "WG")] == 396
    assert WORDBANK_FORM_ITEMS[("English (British)", "Oxford CDI")] == 416
    assert WORDBANK_FORM_ITEMS[("Italian", "WG")] == 408
    assert WORDBANK_FORM_ITEMS[("Spanish (European)", "WG")] == 309
    frame = pd.DataFrame(
        {"language": ["Spanish (European)"] * 2, "form": ["WG"] * 2, "age": [12, 12], "understood": [100, 120]}
    )
    table = td_form_alignment_table(frame)
    assert table.loc[0, "n_items"] == 309
    assert table.loc[0, "median_proportion"] == pytest.approx(110 / 309)
