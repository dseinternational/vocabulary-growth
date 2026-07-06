# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dataset-level descriptive summaries and study-coloured scatter plots.

These helpers describe the *observed data* (not model output): a per-study
summary table (n, age range, and the mean/median/quartiles of each vocabulary
measure) and scatter plots with each study drawn in a distinct colour. They are
used by ``scripts/generate_descriptive_report.py`` to populate the report's
"Data and measures" chapter.

Note: ``summarise_by_group``, ``scatter_by_group`` and ``categorical_palette``
are deliberately generic (no vocabulary-growth specifics) and are candidates for
promotion to ``dse_research_utils`` (``statistics.descriptive`` / ``plot``) so
other DSE projects can reuse them; kept local here until that shared release.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEASURES = ("understood", "spoken", "signed")


def categorical_palette(n: int) -> list:
    """Return ``n`` distinct colours from a qualitative matplotlib colormap."""
    cmap = plt.get_cmap("tab20" if n > 10 else "tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def summarise_by_group(
    df: pd.DataFrame,
    group: str = "study",
    measures: tuple[str, ...] = MEASURES,
) -> pd.DataFrame:
    """Per-group data inventory: counts, age range, and per-measure summaries.

    One row per group. Columns: ``{group}``, ``n_observations``, ``n_subjects``,
    ``age_min``/``age_max``/``age_median``, and for each present measure
    ``{m}_n``/``{m}_mean``/``{m}_median``/``{m}_q1``/``{m}_q3``. Measures absent
    from ``df`` (or all-NA within a group) are skipped / left NA.
    """
    present = [m for m in measures if m in df.columns]
    rows = []
    for name, g in df.groupby(group, dropna=False):
        row: dict[str, object] = {group: name, "n_observations": len(g)}
        if "subject_id" in g.columns:
            row["n_subjects"] = int(g["subject_id"].nunique())
        if "age" in g.columns:
            age = g["age"].dropna()
            row["age_min"] = age.min() if len(age) else np.nan
            row["age_max"] = age.max() if len(age) else np.nan
            row["age_median"] = age.median() if len(age) else np.nan
        for m in present:
            s = g[m].dropna()
            row[f"{m}_n"] = int(s.shape[0])
            if s.shape[0]:
                row[f"{m}_mean"] = round(float(s.mean()), 1)
                row[f"{m}_median"] = float(s.median())
                row[f"{m}_q1"] = float(s.quantile(0.25))
                row[f"{m}_q3"] = float(s.quantile(0.75))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(group).reset_index(drop=True)
    return out


def scatter_by_group(
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str = "study",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9, 6),
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Scatter of ``y`` vs ``x`` with points coloured by ``group``.

    Rows missing ``x`` or ``y`` are dropped. Saves ``.png``/``.svg`` and a
    ``.csv`` of the plotted points when ``output_dir``/``filename`` are given.
    """
    sub = df[[x, y, group]].dropna(subset=[x, y])
    groups = sorted(sub[group].dropna().unique(), key=str)
    palette = categorical_palette(len(groups))

    fig, ax = plt.subplots(figsize=figsize)
    for k, gname in enumerate(groups):
        m = sub[group] == gname
        ax.scatter(
            sub.loc[m, x], sub.loc[m, y],
            s=14, alpha=0.55, color=palette[k], edgecolors="none", label=str(gname),
        )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize="small", title=group, ncol=2)

    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches="tight")
        sub.rename(columns={x: "x", y: "y", group: "group"}).to_csv(
            os.path.join(output_dir, f"{filename}.csv"), index=False
        )
    return fig
