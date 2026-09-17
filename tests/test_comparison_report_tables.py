# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Execute the report's table cells to check support and missing-result handling."""

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vocab_growth.comparison import summarise_draws

ROOT = Path(__file__).resolve().parents[1]


def _cells(path):
    return re.findall(r"```\{python\}[^\n]*\n(.*?)```", (ROOT / path).read_text(), re.S)


@pytest.fixture
def comparison():
    cells = _cells("docs/comparison/index.qmd")
    namespace = {}
    exec(cells[0], namespace)
    return cells, namespace


@pytest.mark.parametrize("age", [7, 30, np.nan])
def test_comparison_does_not_label_an_endpoint_as_an_unsupported_age(comparison, age):
    _, namespace = comparison
    frame = pd.DataFrame({"age": [8, 18, 25], "median": [1, 100, 200]})
    row = namespace["_nearest"](frame, "age", age)
    assert row.isna().all()
    assert namespace["_number"](row["median"]) == "—"


def test_comparison_keeps_supported_endpoints_and_skips_missing_grid_rows(comparison):
    _, namespace = comparison
    frame = pd.DataFrame({"age": [np.nan, 25, 8], "median": [999, 200, 1]})
    assert namespace["_nearest"](frame, "age", 25)["median"] == 200
    assert namespace["_nearest"](frame, "age", 8)["median"] == 1
    assert namespace["_nearest"](frame.iloc[:0], "age", 18).isna().all()


@pytest.mark.parametrize("coverage", [0.79, np.nan, None])
def test_comparison_hides_bands_when_coverage_is_low_or_unknown(comparison, coverage):
    _, namespace = comparison
    row = {"words": 100, "median": 20, "ci_lo": 10, "ci_hi": 30}
    if coverage is not None:
        row["coverage"] = coverage
    frame = pd.DataFrame([row])
    assert namespace["plain_at"](frame, "words", 100) == "—"
    prefixed = frame.rename(columns={key: f"d_{key}" for key in row if key != "words"})
    assert namespace["band_at"](prefixed, "words", 100, "d", cov_col="d_coverage") == "—"


def test_comparison_displays_a_supported_band_but_not_a_partial_interval(comparison):
    _, namespace = comparison
    frame = pd.DataFrame([{"words": 100, "median": 20, "ci_lo": 10, "ci_hi": 30,
                           "coverage": 0.8}])
    assert namespace["plain_at"](frame, "words", 100) == "20 [10, 30]"
    frame["ci_hi"] = np.nan
    assert namespace["plain_at"](frame, "words", 100) == "—"


@pytest.mark.parametrize(
    ("filename", "axis", "prefix", "probability_column"),
    [
        ("ds_td_expressive_equivalent_age.csv", "age_months", "dexpAge", "P(> 0)"),
        ("ds_td_comprehension_q_at_U.csv", "words", "dq", "P(Δq>0)"),
    ],
)
def test_comparison_hides_probability_with_its_unsupported_contrast(
    comparison, monkeypatch, filename, axis, prefix, probability_column
):
    cells, namespace = comparison
    data = {axis: [24, 36] if axis == "age_months" else [50, 100, 150, 200, 300]}
    for name in (prefix, "q_TD", "q_DS"):
        for suffix, value in (("median", 0.4), ("ci_lo", 0.3), ("ci_hi", 0.5),
                              ("coverage", 0.5), ("p_gt0", 1.0)):
            data[f"{name}_{suffix}"] = value

    def read_csv(path):
        if path == filename:
            return pd.DataFrame(data)
        raise FileNotFoundError(path)

    monkeypatch.setattr(pd, "read_csv", read_csv)
    cell = next(cell for cell in cells if f'pd.read_csv("{filename}")' in cell)
    exec(cell, namespace)
    assert all(row[probability_column] == "—" for row in namespace["rows"])


def test_total_spread_cell_reads_the_producer_schema(comparison, tmp_path, monkeypatch, capsys):
    cells, namespace = comparison
    monkeypatch.chdir(tmp_path)
    for outcome in ("understood", "spoken"):
        for suffix, axis in (("total_spread", "age_months"), ("total_spread_at_level", "words")):
            grid = np.array([12, 24]) if axis == "age_months" else np.array([50, 100])
            frame = pd.DataFrame({axis: grid})
            for prefix, value in (("sd_DS", 20.0), ("sd_TD", 30.0), ("dsd", 10.0)):
                summary = summarise_draws(np.full((10, 2), value), grid, axis)
                for column in summary:
                    if column != axis:
                        frame[f"{prefix}_{column}"] = summary[column]
            frame.to_csv(f"ds_td_{outcome}_re_{suffix}.csv", index=False)
    cell = next(cell for cell in cells if 'for _outcome in ("understood", "spoken")' in cell)
    exec(cell, namespace)
    output = capsys.readouterr().out
    assert output.count("### Words") == 4
    assert output.count("Difference (TD minus DS)") == 4
    assert "20 [20, 20]" in output and "30 [30, 30]" in output and "10 [10, 10]" in output
    assert "Pending" not in output

    for path in tmp_path.glob("*.csv"):
        path.unlink()
    exec(cell, namespace)
    assert capsys.readouterr().out.count("Pending.") == 4


@pytest.mark.parametrize("age", [7, 30, np.nan])
def test_paper_does_not_quote_an_endpoint_for_an_unsupported_age(age):
    source = "\n".join(_cells("docs/paper/_paper_data.qmd"))
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "comp_row")
    namespace = {"pd": pd, "comp_load": lambda name: pd.DataFrame(
        {"age": [8, 18, 25], "d_median": [1, 100, 200]}
    )}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "paper-comp-row", "exec"), namespace)
    assert namespace["comp_row"]("fixture", "age", age) is None
    assert namespace["comp_row"]("fixture", "age", 25)["d_median"] == 200
