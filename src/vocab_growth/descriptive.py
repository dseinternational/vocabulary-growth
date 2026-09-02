# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dataset-level descriptive summaries and study-coloured scatter plots.

These helpers describe the *observed data* (not model output): a per-study
summary table (n, age range, and the mean/median/quartiles of each vocabulary
measure) and scatter plots with each study drawn in a distinct colour.
:func:`write_descriptive_artefacts` runs the whole set; it is the
``descriptives`` stage of ``scripts/prepare_report_figures.py``, which populates
the standalone descriptive report and the main report's "Data" chapter.

Note: ``categorical_palette`` moved to
``dse_research_utils.plot.styles`` in v0.12.0 (merged with the other repo's
variant, which also samples continuous colormaps evenly) and is re-exported
below. ``summarise_by_group`` and ``scatter_by_group`` are also generic (no
vocabulary-growth specifics apart from the ``MEASURES`` default and the
save-side-effect convention) and remain candidates for promotion.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dse_research_utils.plot.styles import categorical_palette
from matplotlib.colors import to_rgb

from vocab_growth.data_utils import WORDBANK_FORM_ITEMS, load_combined_data, load_data

MEASURES = ("understood", "spoken", "signed")

# The pooled-summary overlay the repeated-measures plots draw: pure red so it
# reads as one family over any categorical palette, with a near-transparent
# interquartile band.
_SUMMARY_COLOUR = "red"
_IQR_ALPHA = 0.08


def drawable_groups(
    df: pd.DataFrame,
    outcomes: tuple[str, ...],
    *,
    group: str = "study",
    subject_col: str = "subject_id",
    form_col: str = "survey_vocab_max",
) -> list[str]:
    """Groups contributing at least one drawable repeated-measures fragment.

    A fragment is two or more observations of one subject on one form for any
    of ``outcomes``. Callers building a shared group -> colour mapping across
    several figures should assign colours over this union, so a group keeps
    one colour everywhere and the palette is not wasted on groups that never
    draw a line.
    """
    drawn: set[str] = set()
    for outcome in outcomes:
        obs = df.dropna(subset=["age", outcome])
        frag = (
            obs[group].astype(str)
            + "::"
            + obs[subject_col].astype(str)
            + "@"
            + obs[form_col].astype(str)
        )
        counts = frag.groupby(frag).transform("size")
        drawn |= set(obs.loc[counts >= 2, group].astype(str).unique())
    return sorted(drawn)


def summary_table_by_group(
    age_frame: pd.DataFrame,
    measure_frames: dict[str, pd.DataFrame],
    *,
    group: str = "study",
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Per-group range/median/mean/SD summary table, plus a pooled ``All`` row.

    Age statistics and the observation/child counts come from ``age_frame``;
    each measure's statistics come from its own frame's non-missing values.
    The split matters for the typically-developing pool, whose comprehension
    and production are loaded from different form sets — a single frame would
    either drop the Words & Sentences production rows or misstate the ages the
    pool covers. For the Down syndrome pool, pass the same frame throughout.

    Columns: ``{group}``, ``n_subjects``, ``n_observations``, and
    ``{m}_{stat}`` for ``age`` and each measure, with stat in
    min/max/median/mean/sd (sd with ddof=1; all NaN where a group records no
    such measure).
    """

    def stats(series: pd.Series) -> dict[str, float]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if not len(s):
            return {k: np.nan for k in ("min", "max", "median", "mean", "sd")}
        return {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)),
        }

    def sub(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return frame if name == "All" else frame[frame[group].astype(str) == name]

    names = sorted(
        set(age_frame[group].astype(str).unique()).union(
            *[set(f[group].astype(str).unique()) for f in measure_frames.values()]
        )
    )
    rows = []
    for name in [*names, "All"]:
        ages = sub(age_frame, name)
        row: dict[str, object] = {
            group: name,
            "n_subjects": int(ages[subject_col].nunique()),
            "n_observations": int(len(ages)),
        }
        for stat, value in stats(ages["age"]).items():
            row[f"age_{stat}"] = value
        for measure, frame in measure_frames.items():
            for stat, value in stats(sub(frame, name)[measure]).items():
                row[f"{measure}_{stat}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def td_form_alignment_table(
    td: pd.DataFrame,
    *,
    age_range: tuple[int, int] = (8, 15),
    outcome: str = "understood",
    form_items: dict[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    """Per-age, per-form medians of a typically-developing count on two scales.

    Every model scores checklist counts against the fixed 810-item reference
    inventory, which assumes the shorter forms are *nested* -- each omits the
    rarer, later-acquired words -- rather than proportional samples of one
    word universe. Under nesting, forms of different length record similar raw
    counts at a given age; under proportional sampling the shorter forms sit
    systematically lower on raw counts and align instead on the proportion of
    their own inventory. This table lets a reader see which holds: for each age
    in ``age_range`` (inclusive) and each ``(language, form)`` present, the
    median ``outcome`` count as recorded (``median_count``) and as a proportion
    of the form's word-item count (``median_proportion``), with ``n``
    administrations.

    ``td`` is a typically-developing frame from
    :func:`~vocab_growth.data_utils.load_data` carrying ``language``, ``form``,
    ``age`` and ``outcome``. ``form_items`` maps ``(language, form)`` to the
    form's item count and defaults to
    :data:`~vocab_growth.data_utils.WORDBANK_FORM_ITEMS`; a form absent from the
    map raises, because a silently wrong ceiling would defeat the check.

    Columns: ``age``, ``language``, ``form``, ``n_items``, ``n``,
    ``median_count``, ``median_proportion``. :func:`form_alignment_spread`
    compares the two scales age by age. Reproduces the check in
    ``notes/202608031500-td-romance-extension.md``.
    """
    items = WORDBANK_FORM_ITEMS if form_items is None else form_items
    lo, hi = age_range
    obs = td.dropna(subset=["age", outcome])
    obs = obs[(obs["age"] >= lo) & (obs["age"] <= hi)]
    missing = sorted(set(zip(obs["language"], obs["form"], strict=True)) - set(items))
    if missing:
        raise KeyError(f"no word-item count for form(s) {missing}; add them to WORDBANK_FORM_ITEMS")
    rows = []
    for (age, language, form), g in obs.groupby(["age", "language", "form"], sort=True):
        n_items = int(items[(language, form)])
        counts = pd.to_numeric(g[outcome], errors="coerce").dropna()
        median = float(counts.median())
        rows.append(
            {
                "age": int(age),
                "language": language,
                "form": form,
                "n_items": n_items,
                "n": int(len(counts)),
                "median_count": median,
                "median_proportion": median / n_items,
            }
        )
    return pd.DataFrame(rows)


def form_alignment_spread(table: pd.DataFrame) -> pd.DataFrame:
    """Per-age dispersion of the form medians on each scale.

    From a :func:`td_form_alignment_table`: for each age, the number of forms,
    the coefficient of variation (population SD over mean) of the form medians
    on raw counts (``cv_count``) and on proportion of own form
    (``cv_proportion``), which scale is tighter (``tighter``), and the form
    recording the lowest raw median (``lowest_count_form``). Raw counts
    aligning more closely than proportions is what nesting predicts; the
    shortest form recording the lowest counts is what proportional sampling
    would predict.
    """

    def cv(values: pd.Series) -> float:
        x = np.asarray(values, dtype=float)
        return float(x.std(ddof=0) / x.mean()) if len(x) and x.mean() else np.nan

    rows = []
    for age, g in table.groupby("age", sort=True):
        cv_count, cv_proportion = cv(g["median_count"]), cv(g["median_proportion"])
        lowest = g.loc[g["median_count"].idxmin()]
        rows.append(
            {
                "age": int(age),
                "n_forms": int(len(g)),
                "cv_count": cv_count,
                "cv_proportion": cv_proportion,
                "tighter": "count" if cv_count < cv_proportion else "proportion",
                "lowest_count_form": f"{lowest['language']} {lowest['form']}",
            }
        )
    return pd.DataFrame(rows)


def _binned_outcome_summary(
    obs: pd.DataFrame,
    outcome: str,
    lo: float,
    hi: float,
    bin_width: int,
    min_bin_n: int,
) -> pd.DataFrame:
    """Pooled per-age-bin summary (n, median, mean, quartiles) of ``outcome``."""
    edges = np.arange(lo - lo % bin_width, hi + bin_width, bin_width)
    binned = obs.groupby(pd.cut(obs["age"], edges), observed=False)[outcome]
    summary = pd.DataFrame(
        {
            "age_mid": edges[:-1] + bin_width / 2,
            "n": binned.size().to_numpy(),
            "median": binned.median().to_numpy(),
            "mean": binned.mean().to_numpy(),
            "q25": binned.quantile(0.25).to_numpy(),
            "q75": binned.quantile(0.75).to_numpy(),
        }
    )
    return summary[summary["n"] >= min_bin_n]


def _draw_pooled_summary(
    ax, summary: pd.DataFrame, bin_width: int, *, centre_lines: bool = True
) -> None:
    """The red pooled overlay: IQR band, plus solid median and dashed mean
    unless ``centre_lines`` is False (the observation scatters draw the band
    alone — a fitted-looking centre line overstates what a raw-data figure
    shows)."""
    ax.fill_between(
        summary["age_mid"],
        summary["q25"],
        summary["q75"],
        color=_SUMMARY_COLOUR,
        alpha=_IQR_ALPHA,
        linewidth=0,
        zorder=2,
        label="Pooled IQR (25th–75th percentile)",
    )
    if not centre_lines:
        return
    ax.plot(
        summary["age_mid"],
        summary["median"],
        color=_SUMMARY_COLOUR,
        lw=2.5,
        zorder=3,
        label=f"Pooled median ({bin_width}-month bins)",
    )
    ax.plot(
        summary["age_mid"],
        summary["mean"],
        color=_SUMMARY_COLOUR,
        lw=1.8,
        ls="--",
        zorder=3,
        label=f"Pooled mean ({bin_width}-month bins)",
    )


def plot_observations_by_group(
    df: pd.DataFrame,
    outcome: str,
    *,
    group: str = "study",
    subject_col: str = "subject_id",
    age_range: tuple[float, float | None] = (8, None),
    bin_width: int = 6,
    min_bin_n: int = 5,
    point_alpha: float = 0.45,
    point_size: float = 12,
    group_colors: dict | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9, 6),
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Every observation of ``outcome`` by age, coloured by group, with the
    pooled IQR band.

    The scatter companion to :func:`plot_repeat_measures_by_group`: the same
    age window, binning and per-group colouring, but overlaying only the
    pooled interquartile band (no median/mean centre lines — a fitted-looking
    centre line overstates what a raw-data figure shows) and showing
    ALL observations as points — single-visit children included — rather than
    only the repeat-measures subset. An ``age_range`` upper bound of None means
    the data's own maximum, so the scatter can show where the pool thins while
    the windowed trajectory figures stay within the reporting range.
    ``group_colors`` maps group name to colour; pass the same mapping to both
    plot types so a group keeps one colour across the figures. Saves
    ``.png``/``.svg`` and the binned summary as ``.csv`` when
    ``output_dir``/``filename`` are given.
    """
    lo, hi = age_range
    obs = df.dropna(subset=["age", outcome]).copy()
    if hi is None:
        hi = float(obs["age"].max())
    obs = obs[(obs["age"] >= lo) & (obs["age"] <= hi)]
    n_subjects = (
        obs[group].astype(str) + "::" + obs[subject_col].astype(str)
    ).nunique()

    summary = _binned_outcome_summary(obs, outcome, lo, hi, bin_width, min_bin_n)

    groups = sorted(obs[group].astype(str).unique())
    if group_colors is None:
        group_colors = dict(zip(groups, categorical_palette(len(groups)), strict=True))

    fig, ax = plt.subplots(figsize=figsize)
    for name in groups:
        rows = obs[obs[group].astype(str) == name]
        ax.scatter(
            rows["age"],
            rows[outcome],
            s=point_size,
            alpha=point_alpha,
            color=group_colors[name],
            edgecolors="none",
            zorder=1,
        )

    _draw_pooled_summary(ax, summary, bin_width, centre_lines=False)

    per_group = obs.groupby(obs[group].astype(str))[subject_col].nunique()
    for name in group_colors:
        if name in per_group.index:
            ax.scatter(
                [], [],
                s=30,
                color=group_colors[name],
                label=f"{name} (n = {per_group[name]})",
            )

    ax.set_xlabel("Age (months)")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(f"{title} — {n_subjects:,} children, {len(obs):,} observations")
    ax.set_ylim(bottom=0)
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize="small")
    ax.grid(True, alpha=0.25)

    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches="tight")
        summary.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    return fig


def plot_monthly_violins(
    df: pd.DataFrame,
    outcome: str,
    *,
    age_range: tuple[float, float | None] = (8, None),
    min_month_n: int = 5,
    body_colour: str = "#4878a8",
    ylabel: str | None = None,
    figsize: tuple[float, float] = (9, 6),
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Per-month distributions of ``outcome`` as violins, with the pooled
    monthly median and interquartile band.

    Built for pools too dense for a scatter — the typically-developing
    Wordbank pool records hundreds of administrations at each integer age, so
    a scatter saturates into stripes while violins show the distribution's
    shape: the mass on zero before production starts, the growing right tail,
    and the accumulation at form ceilings. Ages are rounded to the nearest
    month; months with fewer than ``min_month_n`` observations are omitted.
    Each violin's density estimate is clipped to that month's observed range,
    so no mass is drawn below zero or beyond the ceilings. An ``age_range``
    upper bound of None means the data's own maximum, so the axis ends where
    the measure's forms do. Saves ``.png``/``.svg`` and the monthly summary as
    ``.csv`` when ``output_dir``/``filename`` are given.
    """
    lo, hi = age_range
    obs = df.dropna(subset=["age", outcome]).copy()
    if hi is None:
        hi = float(obs["age"].max())
    obs = obs[(obs["age"] >= lo) & (obs["age"] <= hi)]
    obs["_month"] = obs["age"].round().astype(int)

    months, groups = [], []
    for month, rows in obs.groupby("_month"):
        if len(rows) >= min_month_n:
            months.append(int(month))
            groups.append(rows[outcome].to_numpy())

    summary = pd.DataFrame(
        {
            "age_months": months,
            "n": [len(g) for g in groups],
            "median": [float(np.median(g)) for g in groups],
            "q25": [float(np.quantile(g, 0.25)) for g in groups],
            "q75": [float(np.quantile(g, 0.75)) for g in groups],
        }
    )

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(
        groups, positions=months, widths=0.85, showmedians=False, showextrema=False
    )
    for body, values in zip(parts["bodies"], groups, strict=True):
        body.set_facecolor(body_colour)
        body.set_alpha(0.55)
        body.set_edgecolor("none")
        path = body.get_paths()[0]
        path.vertices[:, 1] = np.clip(
            path.vertices[:, 1], float(values.min()), float(values.max())
        )

    ax.fill_between(
        summary["age_months"],
        summary["q25"],
        summary["q75"],
        color=_SUMMARY_COLOUR,
        alpha=_IQR_ALPHA,
        linewidth=0,
        zorder=2,
        label="Pooled IQR (25th–75th percentile)",
    )
    ax.plot(
        summary["age_months"],
        summary["median"],
        color=_SUMMARY_COLOUR,
        lw=2.5,
        zorder=3,
        label="Pooled median (monthly)",
    )

    ax.set_xlabel("Age (months)")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(months) - 1, max(months) + 1)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.25)

    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches="tight")
        summary.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    return fig


def plot_repeat_measures_by_group(
    df: pd.DataFrame,
    outcome: str,
    *,
    group: str = "study",
    subject_col: str = "subject_id",
    form_col: str = "survey_vocab_max",
    age_range: tuple[float, float] = (8, 72),
    bin_width: int = 6,
    min_bin_n: int = 5,
    line_alpha: float = 0.45,
    group_colors: dict | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (9, 6),
    output_dir: str | None = None,
    filename: str | None = None,
):
    """Repeated-measures trajectories by group, with a pooled median/IQR overlay.

    Every subject with two or more observations of ``outcome`` on one form is
    drawn as a line joining those visits, coloured by ``group``. Over the fan
    sit the pooled median (solid red), mean (dashed red) and interquartile band
    (translucent red), computed in ``bin_width``-month age bins over ALL
    observations in ``df`` within ``age_range`` — single-visit subjects
    included, so the central trajectory describes the whole pool rather than
    the repeat-measures subset.

    Lines join observations of one subject on ONE form only (``form_col``): a
    same-day dual-form pair or a cross-form transition is a change of
    measurement scale, not development, so those segments are never drawn.

    ``age_range`` bounds the observations used (excluded, not merely clipped
    from view), and bins with fewer than ``min_bin_n`` observations are not
    summarised. ``group_colors`` maps group name to colour; when None the
    colours come from ``categorical_palette`` over the groups that contribute a
    fragment — pass a shared mapping (see :func:`drawable_groups`) when several
    figures must keep group colours aligned. Saves ``.png``/``.svg`` and the
    binned summary as ``.csv`` when ``output_dir``/``filename`` are given.
    """
    lo, hi = age_range
    obs = df.dropna(subset=["age", outcome]).copy()
    obs = obs[(obs["age"] >= lo) & (obs["age"] <= hi)]

    obs["_subject"] = obs[group].astype(str) + "::" + obs[subject_col].astype(str)
    obs["_fragment"] = obs["_subject"] + "@" + obs[form_col].astype(str)
    n_visits = obs.groupby("_fragment")["age"].transform("size")
    repeats = obs[n_visits >= 2].sort_values(["_fragment", "age"])
    n_subjects = repeats["_subject"].nunique()

    summary = _binned_outcome_summary(obs, outcome, lo, hi, bin_width, min_bin_n)

    if group_colors is None:
        groups = drawable_groups(
            df, (outcome,), group=group, subject_col=subject_col, form_col=form_col
        )
        palette = categorical_palette(len(groups))
        group_colors = dict(zip(groups, palette, strict=True))

    fig, ax = plt.subplots(figsize=figsize)
    for _, rows in repeats.groupby("_fragment"):
        colour = group_colors[str(rows[group].iloc[0])]
        ax.plot(
            rows["age"],
            rows[outcome],
            color=colour,
            alpha=line_alpha,
            lw=1.0,
            marker="o",
            ms=2.5,
            markerfacecolor=colour,
            markeredgewidth=0,
            zorder=1,
        )

    _draw_pooled_summary(ax, summary, bin_width)

    per_group = repeats.groupby(repeats[group].astype(str))["_subject"].nunique()
    for name in group_colors:
        if name in per_group.index:
            ax.plot(
                [], [],
                color=group_colors[name],
                alpha=0.8,
                lw=2,
                label=f"{name} (n = {per_group[name]})",
            )

    ax.set_xlabel("Age (months)")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(f"{title} — {n_subjects:,} children, linked within form")
    ax.set_ylim(bottom=0)
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize="small")
    ax.grid(True, alpha=0.25)

    if output_dir is not None and filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(output_dir, f"{filename}.svg"), bbox_inches="tight")
        summary.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    return fig


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




# --------------------------------------------------------------------------- #
# The full descriptive artefact set, as ``scripts/prepare_report_figures.py``
# writes it (its ``descriptives`` stage)
# --------------------------------------------------------------------------- #

SCATTERS = [
    # (x, y, xlabel, ylabel, filename)
    ("age", "understood", "Age (months)", "Words understood", "scatter_age_understood_ds"),
    ("age", "spoken", "Age (months)", "Words spoken", "scatter_age_spoken_ds"),
    ("understood", "spoken", "Words understood", "Words spoken", "scatter_understood_spoken_ds"),
    ("age", "signed", "Age (months)", "Words signed", "scatter_age_signed_ds"),
]

REPEAT_OUTCOMES = [
    # (outcome, ylabel)
    ("understood", "Words understood (raw count)"),
    ("spoken", "Words spoken (raw count)"),
]

# The Down syndrome trajectory figures stop at 72 months — the reporting
# window; the sparse older visits are excluded rather than clipped. The
# typically-developing pool is bounded at 30 months by its own admission
# window, and its density supports finer bins.
DS_REPEAT_AGE_RANGE = (8, 72)
TD_REPEAT_AGE_RANGE = (8, 30)
TD_REPEAT_BIN_WIDTH = 3


def _shared_group_colours(groups):
    """One group -> colour mapping shared by every figure of a population.

    Built over ALL groups the population's frames contain, so a study keeps
    one colour across the observation scatters and the repeated-measures
    trajectories alike (each figure's legend lists only the groups it draws).
    Reddish palette entries are skipped: the pooled-summary overlay every
    figure draws is pure red, and a red-toned study would read as part of it.
    """
    groups = sorted(set(groups))
    colours: list = []
    extra = 0
    while len(colours) < len(groups):
        candidates = categorical_palette(len(groups) + extra)
        colours = [c for c in candidates if not _is_reddish(c)]
        extra += 2
    mapping = dict(zip(groups, colours[: len(groups)], strict=True))
    # teal, used by no palette entry, was chosen for uk_01 when its default
    # colour collided with the overlay; keep that choice stable.
    if "uk_01" in mapping:
        mapping["uk_01"] = "teal"
    return mapping


def _is_reddish(colour) -> bool:
    r, g, b = to_rgb(colour)
    return r > 0.55 and g < 0.45 and b < 0.45


def write_descriptive_artefacts(out_dir: str) -> None:
    """Write every descriptive table and figure into ``out_dir``.

    Reads the prepared dataset (``scripts/prepare_data.py``), so it is not
    importable-cheap: the model definitions are imported here rather than at
    module level to keep ``import vocab_growth.descriptive`` light for the
    helpers above.
    """
    from vocab_growth.models.definitions import (
        ENGLISH_AND_ROMANCE_LANGUAGES,
        Population,
    )

    os.makedirs(out_dir, exist_ok=True)

    # Pooled Down syndrome data (one row per observation), with per-study labels.
    df = load_combined_data()

    summary = summarise_by_group(df, group="study")
    summary.to_csv(os.path.join(out_dir, "dataset_summary_ds.csv"), index=False)
    print(f"Wrote dataset summary for {len(summary)} studies to {out_dir}")

    # Report summary table: per-study range/median/mean/SD of age and both
    # outcomes, plus a pooled All row.
    summary_table_by_group(df, {"understood": df, "spoken": df}).to_csv(
        os.path.join(out_dir, "summary_table_ds.csv"), index=False
    )
    print("Wrote summary_table_ds")

    for x, y, xlabel, ylabel, filename in SCATTERS:
        if y not in df.columns or not df[y].notna().any():
            print(f"Skipping {filename}: no '{y}' observations")
            continue
        fig = scatter_by_group(
            df, x, y, group="study",
            xlabel=xlabel, ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

    # Age-trajectory views of the Down syndrome pool against the pooled
    # median/IQR: every observation as a study-coloured point over the FULL
    # age range (so where the data thins is visible), and the repeat-measures
    # children linked within form over the reporting window.
    ds = load_combined_data(max_age_months=DS_REPEAT_AGE_RANGE[1])
    ds_colours = _shared_group_colours(df["study"].unique())
    for outcome, ylabel in REPEAT_OUTCOMES:
        filename = f"observations_age_{outcome}_ds"
        fig = plot_observations_by_group(
            df, outcome,
            age_range=(DS_REPEAT_AGE_RANGE[0], None),
            group_colors=ds_colours,
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

        filename = f"repeat_measures_age_{outcome}_ds"
        fig = plot_repeat_measures_by_group(
            ds, outcome,
            age_range=DS_REPEAT_AGE_RANGE,
            group_colors=ds_colours,
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

    # The same view of the typically-developing Wordbank reference pool. Each
    # outcome is loaded on its own: requesting understood restricts the TD
    # loader to the bivariate forms, so a joint frame would silently drop the
    # Words & Sentences spoken observations. The ~1,000 repeat-measures
    # children need fainter lines.
    td_frames = {
        outcome: load_data(
            Population.TYPICALLY_DEVELOPING,
            ["study", "subject_id", "form", "age", outcome],
        )
        for outcome, _ in REPEAT_OUTCOMES
    }
    td_colours = _shared_group_colours(
        set().union(*[set(frame["study"].unique()) for frame in td_frames.values()])
    )

    # The TD summary table: the spoken frame covers every admitted form, so it
    # supplies the age statistics and counts; understood comes from its own
    # bivariate-form frame.
    summary_table_by_group(
        td_frames["spoken"],
        {"understood": td_frames["understood"], "spoken": td_frames["spoken"]},
    ).to_csv(os.path.join(out_dir, "summary_table_td.csv"), index=False)
    print("Wrote summary_table_td")

    # The cross-form alignment check behind the methods chapter's "raw counts
    # versus proportions" reassurance. It uses the hierarchical models' full
    # typically-developing scope (English plus Italian and Spanish), so the
    # 309-item Spanish form is in the comparison, over the 8-15 month window
    # the comprehension forms share; the frames above are English-only.
    alignment = td_form_alignment_table(
        load_data(
            Population.TYPICALLY_DEVELOPING,
            ["study", "subject_id", "form", "language", "age", "understood"],
            languages=ENGLISH_AND_ROMANCE_LANGUAGES,
        )
    )
    alignment.to_csv(os.path.join(out_dir, "td_form_alignment.csv"), index=False)
    form_alignment_spread(alignment).to_csv(
        os.path.join(out_dir, "td_form_alignment_spread.csv"), index=False
    )
    print("Wrote td_form_alignment")
    for outcome, ylabel in REPEAT_OUTCOMES:
        # The TD pool is too dense for a scatter (hundreds of administrations
        # at each integer age), so the by-age view is monthly violins. The
        # axis ends at the measure's own data — comprehension stops where the
        # bivariate forms do, before the pool's 30-month bound.
        filename = f"violins_age_{outcome}_td"
        fig = plot_monthly_violins(
            td_frames[outcome], outcome,
            age_range=(TD_REPEAT_AGE_RANGE[0], None),
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

        filename = f"repeat_measures_age_{outcome}_td"
        fig = plot_repeat_measures_by_group(
            td_frames[outcome], outcome,
            form_col="form",
            age_range=TD_REPEAT_AGE_RANGE,
            bin_width=TD_REPEAT_BIN_WIDTH,
            line_alpha=0.15,
            group_colors=td_colours,
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")
