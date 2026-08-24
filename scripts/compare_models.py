# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Cross-model comparison overlays (CSV-based, from per-model output tables).

Produces figures under ``output/comparisons/``:

- ``ds_td_spoken_by_age.{png,svg}`` — VG01 (DS) vs VG03 (TD) — words spoken
- ``ds_td_understood_by_age.{png,svg}`` — VG02 (DS) vs VG04 (TD) — understood
- ``vg05_vs_vg07_{understood,spoken}.{png,svg}`` — study-RE effect in VG07
- ``ds_td_q_vs_understood.{png,svg}`` (+ ``ds_td_q_crossings.csv``) — headline
  matched-comprehension q overlay (DS VG09 / TD VG13, VG07 dashed reference)
- ``vg07_vg09_vg10_q_by_age.{png,svg}`` — q(age) three-way overlay
- ``ds_td_q_by_age_vg20.{png,svg}`` — q(age) DS (VG20) vs TD (VG13)
- ``ds_td_q_vs_understood_vg20.{png,svg}`` — matched-comprehension q with VG20
- ``ds_td_spoken_vs_understood_vg20.{png,svg}`` (+ ``.csv``) — the same
  matched-comprehension comparison in words spoken rather than the ratio

Shared helpers (``first_crossing``, ``overlay_age_curves``) and model-path
resolution (``model_dir``) come from ``vocab_growth.comparison``.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vocab_growth import environment as env
from vocab_growth.comparison import first_crossing, model_dir, overlay_age_curves

OUT_DIR = env.comparisons_output_dir()

DS_COLOUR = plot_styles.COLOUR_BLUE
TD_COLOUR = plot_styles.COLOUR_ORANGE
NO_RE_COLOUR = plot_styles.COLOUR_RED
RE_COLOUR = plot_styles.COLOUR_GREEN


def _read(key: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(model_dir(key), filename))


def ds_td_spoken_by_age() -> None:
    overlay_age_curves(
        "Expected words spoken by age — DS (VG01) vs TD (VG03)",
        [("DS (VG01)", _read("vg01", "posterior_summary.csv"), DS_COLOUR),
         ("TD (VG03)", _read("vg03", "posterior_summary.csv"), TD_COLOUR)],
        os.path.join(OUT_DIR, "ds_td_spoken_by_age"),
        ylabel="Expected words spoken",
    )


def ds_td_understood_by_age() -> None:
    overlay_age_curves(
        "Expected words understood by age — DS (VG02) vs TD (VG04)",
        [("DS (VG02)", _read("vg02", "posterior_summary.csv"), DS_COLOUR),
         ("TD (VG04)", _read("vg04", "posterior_summary.csv"), TD_COLOUR)],
        os.path.join(OUT_DIR, "ds_td_understood_by_age"),
        ylabel="Expected words understood",
    )


def vg05_vs_vg07() -> None:
    for outcome, suffix in (("understood", "u"), ("spoken", "s")):
        overlay_age_curves(
            f"VG05 (no study RE) vs VG07 (with study RE) — words {outcome} (DS)",
            [("VG05 (no RE)", _read("vg05", f"posterior_summary_{suffix}.csv"), NO_RE_COLOUR),
             ("VG07 (study RE)", _read("vg07", f"posterior_summary_{suffix}.csv"), RE_COLOUR)],
            os.path.join(OUT_DIR, f"vg05_vs_vg07_{outcome}"),
            ylabel=f"Expected words {outcome}",
        )


def _q_vs_understood_crossings(series: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows = []
    for pop, df in series:
        for thresh in (0.25, 0.5, 0.75, 0.90):
            rows.append({
                "population": pop,
                "threshold": thresh,
                "n_understood_at_median": first_crossing(
                    df["words_understood"].to_numpy(), df["q_median"].to_numpy(), thresh),
                "n_understood_at_ci_lo": first_crossing(
                    df["words_understood"].to_numpy(), df["ci_lo"].to_numpy(), thresh),
                "n_understood_at_ci_hi": first_crossing(
                    df["words_understood"].to_numpy(), df["ci_hi"].to_numpy(), thresh),
            })
    return pd.DataFrame(rows)


def ds_td_q_vs_understood() -> None:
    """Headline matched-comprehension q overlay: DS (VG09) vs TD (VG13), VG07 dashed."""
    ds_vg09 = _read("vg09", "production_rate_by_understood.csv")
    ds_vg07 = _read("vg07", "production_rate_by_understood.csv")
    td = _read("vg13", "production_rate_by_understood.csv")

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    ax.fill_between(td["words_understood"], td["ci_lo"], td["ci_hi"],
                    color=TD_COLOUR, alpha=0.18, linewidth=0, label="TD 89% interval")
    ax.fill_between(ds_vg09["words_understood"], ds_vg09["ci_lo"], ds_vg09["ci_hi"],
                    color=DS_COLOUR, alpha=0.18, linewidth=0, label="DS 89% interval")
    ax.plot(td["words_understood"], td["q_median"], color=TD_COLOUR, lw=2.5,
            label="TD median q (VG13)")
    ax.plot(ds_vg09["words_understood"], ds_vg09["q_median"], color=DS_COLOUR, lw=2.5,
            label="DS median q (VG09)")
    ax.plot(ds_vg07["words_understood"], ds_vg07["q_median"], color=DS_COLOUR,
            lw=1.5, linestyle="--", alpha=0.7, label="DS median q (VG07, no subject RE)")
    for thresh in (0.5, 0.9):
        ax.axhline(thresh, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
    ax.set_xlim(0, max(td["words_understood"].max(), ds_vg09["words_understood"].max()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected words understood")
    ax.set_ylabel(r"Production ratio  q = $p_S$ / $p_U$")
    ax.set_title("Production ratio against words understood — DS (VG09) vs TD (VG13)")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood.png"))
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood.svg"))
    plt.close(fig)

    _q_vs_understood_crossings(
        [("DS (VG09)", ds_vg09), ("DS (VG07)", ds_vg07), ("TD (VG13)", td)]
    ).to_csv(os.path.join(OUT_DIR, "ds_td_q_crossings.csv"), index=False)


def _merge_q_by_age(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    merged = None
    for tag, df in frames:
        cols = df[["age_months", "q_median", "q_ci_lo", "q_ci_hi"]].rename(
            columns={"q_median": f"{tag}_median", "q_ci_lo": f"{tag}_ci_lo",
                     "q_ci_hi": f"{tag}_ci_hi"})
        merged = cols if merged is None else merged.merge(cols, on="age_months", how="outer")
    return merged.sort_values("age_months")


def vg07_vg09_vg10_q_by_age() -> None:
    """Three-way overlay of q(age) for VG07, VG09 and VG10."""
    series = [
        ("VG07 (no subject RE on q)", _read("vg07", "posterior_summary_q.csv"), plot_styles.COLOUR_PURPLE),
        ("VG09 (subj REs, GP unanchored)", _read("vg09", "posterior_summary_q.csv"), plot_styles.COLOUR_BLUE),
        ("VG10 (subj REs, GP anchored at 54mo)", _read("vg10", "posterior_summary_q.csv"), plot_styles.COLOUR_GREEN),
    ]
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    for label, df, colour in series:
        ax.fill_between(df["age_months"], df["q_ci_lo"], df["q_ci_hi"],
                        color=colour, alpha=0.15, linewidth=0, label=f"{label} 89% interval")
        ax.plot(df["age_months"], df["q_median"], color=colour, lw=2.5, label=f"{label} median")
    ax.axhline(0.5, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
    ax.axhline(0.9, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(r"Production ratio  q = $p_S$ / $p_U$")
    ax.set_ylim(0, 1)
    ax.set_title("Production ratio q(age) — VG07 vs VG09 vs VG10 (DS, rep config)")
    ax.legend(loc="lower right", frameon=True, fontsize="small")
    fig.savefig(os.path.join(OUT_DIR, "vg07_vg09_vg10_q_by_age.png"))
    fig.savefig(os.path.join(OUT_DIR, "vg07_vg09_vg10_q_by_age.svg"))
    plt.close(fig)

    _merge_q_by_age([("vg07", series[0][1]), ("vg09", series[1][1]), ("vg10", series[2][1])]).to_csv(
        os.path.join(OUT_DIR, "vg07_vg09_vg10_q_by_age.csv"), index=False)


def ds_td_q_by_age_vg20() -> None:
    """DS (VG20) vs TD (VG13) production-ratio overlay against age."""
    ds = _read("vg20", "posterior_summary_q.csv")
    td = _read("vg13", "posterior_summary_q.csv")
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    ax.fill_between(td["age_months"], td["q_ci_lo"], td["q_ci_hi"],
                    color=TD_COLOUR, alpha=0.18, linewidth=0, label="TD 89% interval")
    ax.fill_between(ds["age_months"], ds["q_ci_lo"], ds["q_ci_hi"],
                    color=DS_COLOUR, alpha=0.18, linewidth=0, label="DS 89% interval")
    ax.plot(td["age_months"], td["q_median"], color=TD_COLOUR, lw=2.5, label="TD median q (VG13)")
    ax.plot(ds["age_months"], ds["q_median"], color=DS_COLOUR, lw=2.5, label="DS median q (VG20)")
    for thresh in (0.5, 0.9):
        ax.axhline(thresh, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
    ax.set_xlim(min(ds["age_months"].min(), td["age_months"].min()),
                max(ds["age_months"].max(), td["age_months"].max()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(r"Production ratio  q = $p_S$ / $p_U$")
    ax.set_title("Production ratio by age — DS (VG20) vs TD (VG13)")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_by_age_vg20.png"))
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_by_age_vg20.svg"))
    plt.close(fig)

    _merge_q_by_age([("td", td), ("ds", ds)]).to_csv(
        os.path.join(OUT_DIR, "ds_td_q_by_age_vg20.csv"), index=False)


def ds_td_q_vs_understood_vg20() -> None:
    """DS (VG20) vs TD (VG13) production-ratio against words understood."""
    ds = _read("vg20", "production_rate_by_understood.csv")
    td = _read("vg13", "production_rate_by_understood.csv")
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(td["words_understood"], td["ci_lo"], td["ci_hi"],
                    color=TD_COLOUR, alpha=0.15, linewidth=0, label="TD (VG13) 89% interval")
    ax.fill_between(td["words_understood"], td["ci50_lo"], td["ci50_hi"],
                    color=TD_COLOUR, alpha=0.30, linewidth=0, label="TD (VG13) 50% interval")
    ax.plot(td["words_understood"], td["q_median"], color=TD_COLOUR, lw=2.5, label="TD (VG13) median")

    ax.fill_between(ds["words_understood"], ds["ci_lo"], ds["ci_hi"],
                    color=DS_COLOUR, alpha=0.15, linewidth=0, label="DS (VG20) 89% interval")
    ax.fill_between(ds["words_understood"], ds["ci50_lo"], ds["ci50_hi"],
                    color=DS_COLOUR, alpha=0.30, linewidth=0, label="DS (VG20) 50% interval")
    ax.plot(ds["words_understood"], ds["q_median"], color=DS_COLOUR, lw=2.5, label="DS (VG20) median")

    for thresh in (0.5, 0.9):
        ax.axhline(thresh, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
    ax.set_xlim(0, max(td["words_understood"].max(), ds["words_understood"].max()))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Expected words understood")
    ax.set_ylabel(r"Production ratio  q = $E[S] / E[U]$")
    ax.set_title("Production ratio against words understood — DS (VG20) vs TD (VG13)")
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood_vg20.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood_vg20.svg"))
    plt.close(fig)

    print(
        f"DS (VG20) U range covered: {ds['words_understood'].min():.0f} – "
        f"{ds['words_understood'].max():.0f}"
    )
    print(
        f"TD (VG13) U range covered: {td['words_understood'].min():.0f} – "
        f"{td['words_understood'].max():.0f}"
    )


def ds_td_spoken_vs_understood_vg20() -> None:
    """The same matched-comprehension comparison in words rather than a ratio.

    ``E[S] = q(U) * U``, and at each grid point ``U`` is a constant — so every
    posterior quantile of the spoken count is that quantile of ``q`` multiplied
    by ``U``. Quantiles are equivariant under multiplication by a positive
    constant, so this is the *exact* posterior summary of words spoken, not an
    approximation, and it needs no trace: the median and both interval bounds
    rescale together.

    It carries the same information as the ratio plot and puts it in the form a
    family or teacher is more likely to want. Because both curves are multiplied
    by the same ``U``, their ratio at any point is unchanged — the vertical scale
    changes, the finding does not.

    **Both sides are population quantities, and the comparison is between
    developmental stages (issue #233).** ``production_rate_by_understood.csv``
    is written from ``p_u_plot`` and ``q_plot``, both evaluated at zero study and
    zero child effects, so ``U`` on the grid is the population median expected
    comprehension at some age and ``q(U)`` is the population ratio at that same
    age. Neither model conditions its child effects on observed comprehension.
    So this answers "at the stage where children of each population typically
    understand U words, what share do they typically say?", NOT "of two children
    who each understand U words, one with Down syndrome and one without, what
    does each say?". The second is a conditional expectation that needs the joint
    child-effect posterior integrated through the comprehension likelihood, and
    it is not what is computed here. Label it accordingly wherever it is
    published.
    """
    ds = _read("vg20", "production_rate_by_understood.csv")
    td = _read("vg13", "production_rate_by_understood.csv")

    for frame in (ds, td):
        u = frame["words_understood"]
        for col in ("q_median", "ci50_lo", "ci50_hi", "ci_lo", "ci_hi"):
            frame[f"s_{col}"] = frame[col] * u

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(td["words_understood"], td["s_ci_lo"], td["s_ci_hi"],
                    color=TD_COLOUR, alpha=0.15, linewidth=0, label="TD (VG13) 89% interval")
    ax.fill_between(td["words_understood"], td["s_ci50_lo"], td["s_ci50_hi"],
                    color=TD_COLOUR, alpha=0.30, linewidth=0, label="TD (VG13) 50% interval")
    ax.plot(td["words_understood"], td["s_q_median"], color=TD_COLOUR, lw=2.5,
            label="TD (VG13) median")

    ax.fill_between(ds["words_understood"], ds["s_ci_lo"], ds["s_ci_hi"],
                    color=DS_COLOUR, alpha=0.15, linewidth=0, label="DS (VG20) 89% interval")
    ax.fill_between(ds["words_understood"], ds["s_ci50_lo"], ds["s_ci50_hi"],
                    color=DS_COLOUR, alpha=0.30, linewidth=0, label="DS (VG20) 50% interval")
    ax.plot(ds["words_understood"], ds["s_q_median"], color=DS_COLOUR, lw=2.5,
            label="DS (VG20) median")

    # The 1:1 line is the ceiling: a child cannot say more words than they
    # understand, so every curve must lie on or below it. Drawing it stops the
    # eye reading the gap between the two curves as larger than the space
    # available for it.
    upper = max(td["words_understood"].max(), ds["words_understood"].max())
    ax.plot([0, upper], [0, upper], color=plot_styles.LINE_COLOUR, lw=0.8,
            linestyle=":", label="says everything understood (1:1)")

    ax.set_xlim(0, upper)
    ax.set_ylim(0, None)
    ax.set_xlabel("Expected words understood")
    ax.set_ylabel("Expected words spoken")
    ax.set_title("Words spoken against words understood — DS (VG20) vs TD (VG13)")
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_spoken_vs_understood_vg20.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_spoken_vs_understood_vg20.svg"))
    plt.close(fig)

    # The two models are summarised on different `words_understood` grids, so an
    # outer merge on that column yields a frame where no row carries both
    # populations — every lookup returns NaN for one side. Interpolate each onto
    # a shared grid instead, leaving NaN only outside a model's own support
    # (VG13 covers 8-18 months, so its curve genuinely stops early).
    cols = ["s_q_median", "s_ci50_lo", "s_ci50_hi", "s_ci_lo", "s_ci_hi"]
    grid = pd.Series(sorted(set(range(10, 531, 5))), name="words_understood")
    merged = pd.DataFrame({"words_understood": grid})
    for tag, frame in (("DS", ds), ("TD", td)):
        lo, hi = frame["words_understood"].min(), frame["words_understood"].max()
        inside = (grid >= lo) & (grid <= hi)
        for col in cols:
            values = np.interp(grid, frame["words_understood"], frame[col])
            merged[f"{tag}_{col}"] = np.where(inside, values, np.nan)
    both = merged["DS_s_q_median"].notna() & merged["TD_s_q_median"].notna()
    merged["TD_minus_DS_median"] = np.where(
        both, merged["TD_s_q_median"] - merged["DS_s_q_median"], np.nan)
    merged.to_csv(os.path.join(OUT_DIR, "ds_td_spoken_vs_understood_vg20.csv"), index=False)


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    ds_td_spoken_by_age()
    ds_td_understood_by_age()
    vg05_vs_vg07()
    ds_td_q_vs_understood()
    vg07_vg09_vg10_q_by_age()
    ds_td_q_by_age_vg20()
    ds_td_q_vs_understood_vg20()
    ds_td_spoken_vs_understood_vg20()
    print(f"Comparisons written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
