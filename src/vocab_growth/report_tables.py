# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import pandas as pd


def print_table(df, caption, label):
    _print_small_table(df, caption, label, font_size="0.85em")


def print_small_table(df, caption, label):
    _print_small_table(df, caption, label, font_size="0.75em")


def _print_small_table(df, caption, label, font_size):
    """A Markdown table without the index, at reduced size in HTML and PDF."""
    print(f'::: {{style="font-size: {font_size}"}}\n')
    print(r"```{=latex}" + "\n" + r"\begingroup\footnotesize" + "\n```\n")
    print(df.to_markdown(index=False))
    print(f'\n: {caption} {{#{label} tbl-colwidths="false"}}\n')
    print(r"```{=latex}" + "\n" + r"\endgroup" + "\n```\n")
    print(":::\n")


TD_FORMS_ALIGNMENT_TABLE_LABELS = {
    ("English (American)", "WG"): "English WG (396)",
    ("English (British)", "Oxford CDI"): "Oxford CDI (416)",
    ("Italian", "WG"): "Italian WG (408)",
    ("Spanish (European)", "WG"): "Spanish WG (309)",
}


def td_forms_alignment_table(table_path, spread_path):
    t, s = Path(table_path), Path(spread_path)
    if not (t.exists() and s.exists()):
        return pd.DataFrame(
            {"note": ["Run scripts/generate_descriptive_report.py first."]}
        )
    table, spread = pd.read_csv(t), pd.read_csv(s)
    table["column"] = [
        TD_FORMS_ALIGNMENT_TABLE_LABELS.get((lang, form), f"{lang} {form} ({items})")
        for lang, form, items in zip(table["language"], table["form"], table["n_items"])
    ]
    table["cell"] = [
        f"{count:g} ({100 * prop:.1f}%)"
        for count, prop in zip(table["median_count"], table["median_proportion"])
    ]
    wide = table.pivot(index="age", columns="column", values="cell").fillna("—")
    order = [c for c in TD_FORMS_ALIGNMENT_TABLE_LABELS.values() if c in wide.columns]
    wide = wide[order + [c for c in wide.columns if c not in order]]
    wide["CV, counts"] = spread.set_index("age")["cv_count"].map("{:.2f}".format)
    wide["CV, proportions"] = spread.set_index("age")["cv_proportion"].map(
        "{:.2f}".format
    )
    wide.columns.name = None
    return wide.reset_index().rename(columns={"age": "Age (months)"})
