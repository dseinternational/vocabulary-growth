# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Audit data coverage by study × age bin for the Down syndrome dataset.

The Simpson's-paradox investigation (see
`notes/202604121055-understood-ds-decline.md`) showed that knowing
*which* studies contribute observations at *which* ages is essential
for interpreting any pooled trajectory. This script produces a
permanent coverage audit so any future compositional artefact can be
diagnosed immediately.

Outputs (under `output/data/`):

- `coverage_understood.csv`, `coverage_spoken.csv`
  Count of non-null observations per (study, age_bin).
- `coverage_understood.{png,svg}`, `coverage_spoken.{png,svg}`
  Heatmaps of the same.
- `coverage_summary.csv`
  Totals and the percentage of observations with each outcome.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import pandas as pd

from vocab_growth.data_utils import load_combined_data

OUT_DIR = "output/data"

AGE_BIN_EDGES = list(range(0, 121, 6))  # 0, 6, 12, …, 120 months


def _make_pivot(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Counts of non-null `column` rows per (study, age_bin)."""
    work = df[["study", "age", column]].dropna(subset=[column]).copy()
    work["age_bin"] = pd.cut(
        work["age"], bins=AGE_BIN_EDGES, right=False, include_lowest=True,
    )
    pivot = (
        work.groupby(["study", "age_bin"], observed=False)
        .size()
        .unstack("age_bin", fill_value=0)
    )
    pivot.columns = [
        f"{int(iv.left)}-{int(iv.right)}" for iv in pivot.columns
    ]
    pivot.loc["TOTAL"] = pivot.sum()
    pivot["total"] = pivot.sum(axis=1)
    return pivot


def _heatmap(pivot: pd.DataFrame, title: str, out_base: str) -> None:
    data = pivot.drop(columns=["total"], errors="ignore").drop(
        index=["TOTAL"], errors="ignore",
    )
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    im = ax.imshow(data.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"Study {s}" for s in data.index])
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data.values[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v < data.values.max() * 0.5 else "black")
    ax.set_xlabel("Age bin (months)")
    ax.set_ylabel("Study")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Observation count")
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".svg")
    plt.close(fig)


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_combined_data()
    print(f"Loaded {len(df):,} DS observations")

    for outcome in ("understood", "spoken"):
        pivot = _make_pivot(df, outcome)
        pivot.to_csv(os.path.join(OUT_DIR, f"coverage_{outcome}.csv"))
        _heatmap(
            pivot,
            f"Observations of '{outcome}' by study × age bin (DS)",
            os.path.join(OUT_DIR, f"coverage_{outcome}"),
        )

    rows = []
    for outcome in ("understood", "spoken"):
        non_null = df[outcome].notna().sum()
        rows.append({
            "outcome": outcome,
            "n_observations": int(non_null),
            "n_total_rows": int(len(df)),
            "pct_with_outcome": round(100 * non_null / len(df), 1),
        })

    paired = df[["understood", "spoken"]].notna().all(axis=1).sum()
    rows.append({
        "outcome": "both_understood_and_spoken",
        "n_observations": int(paired),
        "n_total_rows": int(len(df)),
        "pct_with_outcome": round(100 * paired / len(df), 1),
    })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "coverage_summary.csv"),
                              index=False)

    print(f"Wrote coverage CSVs + heatmaps under: {OUT_DIR}")


if __name__ == "__main__":
    main()
