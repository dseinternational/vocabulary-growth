# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Display helpers for the report's tables.

The Quarto chapters call these from ``#| output: asis`` cells. A table is emitted
as Markdown rather than returned as a DataFrame so that the pandas index never
renders, the size can be set per format (a styled div for HTML, a LaTeX size
group for PDF), and the caption carries the cross-reference label. The builders
read the CSV artefacts the ``descriptives`` stage of
``scripts/prepare_report_figures.py`` writes into
the report's figure cache and return display-ready frames of strings.
"""

from pathlib import Path

import pandas as pd

PENDING_NOTE = "Run scripts/prepare_report_figures.py descriptives first."


def print_table(df, caption, label, colwidths="false"):
    _print_small_table(df, caption, label, font_size="0.85em", colwidths=colwidths)


def print_small_table(df, caption, label, colwidths="false"):
    _print_small_table(df, caption, label, font_size="0.75em", colwidths=colwidths)


def print_wide_table(df, caption, label, colwidths=None):
    """For tables too wide to sit at natural width on the PDF page.

    Smaller still (``scriptsize`` in PDF, 0.7em in HTML), with the LaTeX cell
    padding halved, and proportional column widths so that long headers wrap
    rather than the table running into the margin. Pass ``colwidths`` as
    relative widths tuned to each column's longest value (see
    :data:`SUMMARY_TABLE_COLWIDTHS`); ``None`` leaves the split to Quarto.
    """
    _print_small_table(
        df, caption, label, font_size="0.7em", colwidths=colwidths,
        latex_size="scriptsize", tabcolsep="3pt",
    )


SUMMARY_TABLE_COLWIDTHS = [13, 7, 5, 6, 5, 12, 7, 5, 14, 6, 6, 14]
"""Relative column widths for :func:`summary_table` output under :func:`print_wide_table`.

Proportioned to each column's longest value -- a 13-character dataset name, the
``mean (SD)`` cells -- so that values keep to one line on the A4 page while the
headers wrap. Checked by rendering the two study-summary tables to PDF.
"""


def _colwidths_attr(colwidths):
    """The ``tbl-colwidths`` caption attribute: natural widths by default.

    ``"false"`` (the default) stops pandoc assigning proportional column widths
    to a wide pipe table, so PDF cells never wrap mid-value; a list of relative
    widths pins them instead; ``None`` omits the attribute and leaves the choice
    to Quarto.
    """
    if colwidths is None:
        return ""
    if not isinstance(colwidths, str):
        colwidths = "[" + ",".join(str(w) for w in colwidths) + "]"
    return f' tbl-colwidths="{colwidths}"'


def _print_small_table(
    df, caption, label, font_size, colwidths="false", latex_size="footnotesize", tabcolsep=None,
):
    """A Markdown table without the index, at reduced size in HTML and PDF."""
    latex_open = "\\begingroup\\" + latex_size
    if tabcolsep is not None:
        latex_open += "\\setlength{\\tabcolsep}{" + tabcolsep + "}"
    print(f'::: {{style="font-size: {font_size}"}}\n')
    print(r"```{=latex}" + "\n" + latex_open + "\n```\n")
    print(df.to_markdown(index=False))
    print(f"\n: {caption} {{#{label}{_colwidths_attr(colwidths)}}}\n")
    print(r"```{=latex}" + "\n" + r"\endgroup" + "\n```\n")
    print(":::\n")


def _pending():
    return pd.DataFrame({"note": [PENDING_NOTE]})


def summary_table(path, group_label):
    """A per-group descriptive summary as display strings.

    ``path`` is one of the ``summary_table_*.csv`` files the descriptive report
    writes (``vocab_growth.descriptive.summary_table_by_group`` output);
    ``group_label`` heads the first column -- "Study" for the Down syndrome pool,
    "Dataset" for the typically-developing one. Ranges print as ``min–max``,
    medians as integers where they are whole, means as ``mean (SD)``; a blank
    cell means the group records no such measure (or the loader masks it).
    """
    p = Path(path)
    if not p.exists():
        return _pending()
    d = pd.read_csv(p)

    def rng(prefix):
        return [
            "" if pd.isna(lo) else f"{lo:.0f}–{hi:.0f}"
            for lo, hi in zip(d[f"{prefix}_min"], d[f"{prefix}_max"], strict=True)
        ]

    def med(prefix):
        return [
            "" if pd.isna(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}")
            for v in d[f"{prefix}_median"]
        ]

    def mean_sd(prefix):
        return [
            "" if pd.isna(m) else (f"{m:.1f} ({s:.1f})" if pd.notna(s) else f"{m:.1f}")
            for m, s in zip(d[f"{prefix}_mean"], d[f"{prefix}_sd"], strict=True)
        ]

    return pd.DataFrame(
        {
            group_label: d["study"],
            "Children": d["n_subjects"].astype(int),
            "Obs.": d["n_observations"].astype(int),
            "Age range": rng("age"),
            "Age mdn": med("age"),
            "Age mean (SD)": mean_sd("age"),
            "Und. range": rng("understood"),
            "Und. mdn": med("understood"),
            "Und. mean (SD)": mean_sd("understood"),
            "Spoken range": rng("spoken"),
            "Spoken mdn": med("spoken"),
            "Spoken mean (SD)": mean_sd("spoken"),
        }
    )


TD_FORMS_ALIGNMENT_TABLE_LABELS = {
    ("English (American)", "WG"): "English WG (396)",
    ("English (British)", "Oxford CDI"): "Oxford CDI (416)",
    ("Italian", "WG"): "Italian WG (408)",
    ("Spanish (European)", "WG"): "Spanish WG (309)",
}


def td_forms_alignment_table(table_path, spread_path):
    """The cross-form alignment check as one row per age.

    ``table_path`` and ``spread_path`` are ``td_form_alignment.csv`` and
    ``td_form_alignment_spread.csv`` (``vocab_growth.descriptive``
    ``td_form_alignment_table`` / ``form_alignment_spread`` output). Each form
    present becomes a column of ``median count (percent of own form)``, in
    :data:`TD_FORMS_ALIGNMENT_TABLE_LABELS` order, with a dash where the form was
    not administered at that age; the two coefficient-of-variation columns follow.
    """
    t, s = Path(table_path), Path(spread_path)
    if not (t.exists() and s.exists()):
        return _pending()
    table, spread = pd.read_csv(t), pd.read_csv(s)
    table["column"] = [
        TD_FORMS_ALIGNMENT_TABLE_LABELS.get((lang, form), f"{lang} {form} ({items})")
        for lang, form, items in zip(
            table["language"], table["form"], table["n_items"], strict=True
        )
    ]
    table["cell"] = [
        f"{count:g} ({100 * prop:.1f}%)"
        for count, prop in zip(table["median_count"], table["median_proportion"], strict=True)
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
