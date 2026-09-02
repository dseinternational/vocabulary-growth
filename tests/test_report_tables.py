# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The report's table display layer (``vocab_growth.report_tables``).

``print_small_table`` writes a Markdown table for an ``output: asis`` cell, and
the two builders turn the descriptive CSV artefacts into display-ready frames of
strings. The fixtures are tiny hand-written CSVs in the artefacts' schemas, so
the tests pin the formatting without touching the figure cache.
"""

import numpy as np
import pandas as pd
import pytest

from vocab_growth import report_tables as rt


@pytest.fixture
def frame():
    return pd.DataFrame({"Age (months)": [8, 9], "Form A": ["16 (4.0%)", "26 (6.6%)"]})


def test_print_small_table_emits_an_index_free_markdown_table(frame, capsys):
    rt.print_small_table(frame, "A caption.", "tbl-example")
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == '::: {style="font-size: 0.75em"}'
    assert out.rstrip().endswith(":::")
    assert frame.to_markdown(index=False) in out
    assert frame.to_markdown() not in out  # the index would add an unnamed first column
    assert ': A caption. {#tbl-example tbl-colwidths="false"}' in lines
    # The LaTeX size group brackets the table, and the caption follows the table.
    open_at = lines.index("\\begingroup\\footnotesize")
    header_at = next(i for i, line in enumerate(lines) if line.startswith("|") and "Age (months)" in line)
    caption_at = next(i for i, line in enumerate(lines) if line.startswith(": A caption."))
    close_at = lines.index("\\endgroup")
    assert open_at < header_at < caption_at < close_at
    assert lines[open_at - 1] == "```{=latex}" and lines[open_at + 1] == "```"
    assert lines[close_at - 1] == "```{=latex}" and lines[close_at + 1] == "```"


def test_print_table_is_the_larger_size(frame, capsys):
    rt.print_table(frame, "c", "tbl-x")
    assert capsys.readouterr().out.splitlines()[0] == '::: {style="font-size: 0.85em"}'


def test_print_wide_table_shrinks_further_and_pins_proportional_widths(frame, capsys):
    rt.print_wide_table(frame, "c", "tbl-x", colwidths=rt.SUMMARY_TABLE_COLWIDTHS)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == '::: {style="font-size: 0.7em"}'
    assert "\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3pt}" in lines
    assert ': c {#tbl-x tbl-colwidths="[13,7,5,6,5,12,7,5,14,6,6,14]"}' in lines
    assert sum(rt.SUMMARY_TABLE_COLWIDTHS) == 100 and len(rt.SUMMARY_TABLE_COLWIDTHS) == 12


def test_print_wide_table_without_widths_leaves_the_split_to_quarto(frame, capsys):
    rt.print_wide_table(frame, "c", "tbl-x")
    assert ": c {#tbl-x}" in capsys.readouterr().out.splitlines()


@pytest.mark.parametrize(
    ("colwidths", "expected"),
    [
        ("false", '{#tbl-x tbl-colwidths="false"}'),
        ([10, 20, 30], '{#tbl-x tbl-colwidths="[10,20,30]"}'),
        (None, "{#tbl-x}"),
    ],
)
def test_colwidths_attribute(frame, capsys, colwidths, expected):
    rt.print_small_table(frame, "c", "tbl-x", colwidths=colwidths)
    assert f": c {expected}" in capsys.readouterr().out.splitlines()


def _write_alignment_csvs(tmp_path):
    table = pd.DataFrame(
        [
            (8, "English (American)", "WG", 396, 255, 16.0, 16 / 396),
            (8, "Italian", "WG", 408, 34, 11.5, 11.5 / 408),
            (12, "English (American)", "WG", 396, 563, 62.0, 62 / 396),
            (12, "English (British)", "Oxford CDI", 416, 27, 45.0, 45 / 416),
        ],
        columns=["age", "language", "form", "n_items", "n", "median_count", "median_proportion"],
    )
    spread = pd.DataFrame(
        [
            (8, 2, 0.15330, 0.25316, "count", "Italian WG"),
            (12, 2, 0.2, 0.3, "count", "English (British) Oxford CDI"),
        ],
        columns=["age", "n_forms", "cv_count", "cv_proportion", "tighter", "lowest_count_form"],
    )
    t, s = tmp_path / "td_form_alignment.csv", tmp_path / "td_form_alignment_spread.csv"
    table.to_csv(t, index=False)
    spread.to_csv(s, index=False)
    return t, s


def test_alignment_table_is_one_row_per_age_in_form_order(tmp_path):
    t, s = _write_alignment_csvs(tmp_path)
    out = rt.td_forms_alignment_table(t, s)
    assert list(out.columns) == [
        "Age (months)", "English WG (396)", "Oxford CDI (416)", "Italian WG (408)",
        "CV, counts", "CV, proportions",
    ]
    by_age = out.set_index("Age (months)")
    at8 = by_age.loc[8]
    assert at8["English WG (396)"] == "16 (4.0%)"
    assert at8["Italian WG (408)"] == "11.5 (2.8%)"
    assert at8["Oxford CDI (416)"] == "—"  # not administered at 8 months
    assert at8["CV, counts"] == "0.15" and at8["CV, proportions"] == "0.25"
    at12 = by_age.loc[12]
    assert at12["Oxford CDI (416)"] == "45 (10.8%)" and at12["Italian WG (408)"] == "—"


def test_alignment_table_is_pending_without_the_artefacts(tmp_path):
    out = rt.td_forms_alignment_table(tmp_path / "missing.csv", tmp_path / "missing_spread.csv")
    assert list(out.columns) == ["note"] and out.loc[0, "note"] == rt.PENDING_NOTE


def _summary_row(study, **values):
    row = {"study": study}
    for prefix in ("age", "understood", "spoken"):
        for stat in ("min", "max", "median", "mean", "sd"):
            row[f"{prefix}_{stat}"] = values.get(f"{prefix}_{stat}", np.nan)
    row.update({k: v for k, v in values.items() if k in ("n_subjects", "n_observations")})
    return row


def test_summary_table_formats_ranges_medians_and_means(tmp_path):
    csv = tmp_path / "summary_table_ds.csv"
    pd.DataFrame(
        [
            _summary_row(
                "es_01", n_subjects=186, n_observations=186,
                age_min=11.0, age_max=71.0, age_median=32.0, age_mean=34.3118, age_sd=14.9955,
                understood_min=7.0, understood_max=651.0, understood_median=266.5,
                understood_mean=272.9086, understood_sd=182.1045,
                spoken_min=0.0, spoken_max=637.0, spoken_median=26.0, spoken_mean=104.5591, spoken_sd=156.77,
            ),
            # A production-only study with a single-observation SD: no understood
            # statistics at all, and a mean without an SD.
            _summary_row(
                "uk_05", n_subjects=12, n_observations=30,
                age_min=24.0, age_max=60.0, age_median=40.5, age_mean=41.0,
                spoken_min=3.0, spoken_max=400.0, spoken_median=120.0, spoken_mean=150.25, spoken_sd=90.5,
            ),
        ]
    ).to_csv(csv, index=False)
    out = rt.summary_table(csv, "Study")
    assert list(out.columns) == [
        "Study", "Children", "Obs.", "Age range", "Age mdn", "Age mean (SD)",
        "Und. range", "Und. mdn", "Und. mean (SD)", "Spoken range", "Spoken mdn", "Spoken mean (SD)",
    ]
    by_study = out.set_index("Study")
    es = by_study.loc["es_01"]
    assert es["Children"] == 186 and es["Obs."] == 186
    assert es["Age range"] == "11–71" and es["Age mdn"] == "32" and es["Age mean (SD)"] == "34.3 (15.0)"
    assert es["Und. mdn"] == "266.5" and es["Und. mean (SD)"] == "272.9 (182.1)"
    uk = by_study.loc["uk_05"]
    assert uk["Age mdn"] == "40.5" and uk["Age mean (SD)"] == "41.0"
    assert uk["Und. range"] == "" and uk["Und. mdn"] == "" and uk["Und. mean (SD)"] == ""
    assert uk["Spoken range"] == "3–400" and uk["Spoken mean (SD)"] == "150.2 (90.5)"


def test_summary_table_is_pending_without_the_artefact(tmp_path):
    out = rt.summary_table(tmp_path / "nope.csv", "Study")
    assert out.loc[0, "note"] == rt.PENDING_NOTE
