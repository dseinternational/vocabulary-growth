# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Cross-model comparison overlays.

Produces the following figures under `output/comparisons/`:

- `ds_td_spoken_by_age.{png,svg}` — VG01 (DS) vs VG03 (TD) — words spoken
- `ds_td_understood_by_age.{png,svg}` — VG02 (DS) vs VG04 (TD) — words understood
- `vg05_vs_vg07_understood.{png,svg}` — Simpson's-paradox fix in VG07
- `vg05_vs_vg07_spoken.{png,svg}` — sanity check that the fix preserves
  the spoken trajectory
- `ds_td_q_vs_understood.{png,svg}` — q vs words understood (DS VG09 / TD
  VG06) — the headline matched-comprehension comparison; the
  corresponding crossings CSV lives in this directory too.

CSVs of the underlying data are also written.

Replaces the earlier one-off `compare_ds_td.py`.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS_DIR = "output/models"
OUT_DIR = "output/comparisons"

DS_COLOUR = plot_styles.COLOUR_BLUE
TD_COLOUR = plot_styles.COLOUR_ORANGE
NO_RE_COLOUR = plot_styles.COLOUR_RED
RE_COLOUR = plot_styles.COLOUR_GREEN


def overlay_age_curves(
    title: str,
    series: list[tuple[str, pd.DataFrame, str]],
    out_base: str,
    *,
    ylabel: str = "Expected words",
) -> None:
    """Overlay Y_median + Y_hdi bands from posterior_summary-shaped frames.

    Each series tuple is (label, dataframe, colour). The dataframe must
    have columns `age_months`, `Ey_median`, `Ey_hdi_lo`, `Ey_hdi_hi`.
    """
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    for label, df, colour in series:
        ax.fill_between(
            df["age_months"], df["Ey_hdi_lo"], df["Ey_hdi_hi"],
            color=colour, alpha=0.18, linewidth=0,
            label=f"{label} 90% HDI",
        )
        ax.plot(
            df["age_months"], df["Ey_median"], color=colour, lw=2.5,
            label=f"{label} median",
        )
    ax.set_xlabel("Age (months)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".svg")
    plt.close(fig)


def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    above = y >= threshold
    if not above.any():
        return None
    if above.all():
        return float(x[0])
    i = int(np.argmax(above))
    if i == 0:
        return float(x[0])
    x0, x1 = float(x[i - 1]), float(x[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return x1
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def ds_td_spoken_by_age() -> None:
    ds = pd.read_csv(os.path.join(MODELS_DIR, "VG01-age-spoken-ds", "posterior_summary.csv"))
    td = pd.read_csv(os.path.join(MODELS_DIR, "VG03-age-spoken-td", "posterior_summary.csv"))
    overlay_age_curves(
        "Expected words spoken by age — DS (VG01) vs TD (VG03)",
        [("DS (VG01)", ds, DS_COLOUR), ("TD (VG03)", td, TD_COLOUR)],
        os.path.join(OUT_DIR, "ds_td_spoken_by_age"),
        ylabel="Expected words spoken",
    )


def ds_td_understood_by_age() -> None:
    ds = pd.read_csv(os.path.join(MODELS_DIR, "VG02-age-understood-ds", "posterior_summary.csv"))
    td = pd.read_csv(os.path.join(MODELS_DIR, "VG04-age-understood-td", "posterior_summary.csv"))
    overlay_age_curves(
        "Expected words understood by age — DS (VG02) vs TD (VG04)",
        [("DS (VG02)", ds, DS_COLOUR), ("TD (VG04)", td, TD_COLOUR)],
        os.path.join(OUT_DIR, "ds_td_understood_by_age"),
        ylabel="Expected words understood",
    )


def vg05_vs_vg07() -> None:
    for outcome, suffix in (("understood", "u"), ("spoken", "s")):
        no_re = pd.read_csv(
            os.path.join(MODELS_DIR, "VG05-age-understood-spoken-ds",
                         f"posterior_summary_{suffix}.csv")
        )
        re = pd.read_csv(
            os.path.join(MODELS_DIR, "VG07-age-understood-spoken-ds-re",
                         f"posterior_summary_{suffix}.csv")
        )
        overlay_age_curves(
            f"VG05 (no study RE) vs VG07 (with study RE) — words {outcome} (DS)",
            [("VG05 (no RE)", no_re, NO_RE_COLOUR),
             ("VG07 (study RE)", re, RE_COLOUR)],
            os.path.join(OUT_DIR, f"vg05_vs_vg07_{outcome}"),
            ylabel=f"Expected words {outcome}",
        )


def ds_td_q_vs_understood() -> None:
    """DS-vs-TD matched-comprehension production-ratio overlay.

    Headline DS model is VG09 (study + subject REs on both U and q).
    VG07 is plotted as a dashed reference line so the shift from the
    pre-subject-RE estimate is visible.
    """
    ds_vg09 = pd.read_csv(
        os.path.join(MODELS_DIR, "VG09-age-understood-spoken-ds-re-subj-uq",
                     "production_rate_by_understood.csv")
    )
    ds_vg07 = pd.read_csv(
        os.path.join(MODELS_DIR, "VG07-age-understood-spoken-ds-re",
                     "production_rate_by_understood.csv")
    )
    td = pd.read_csv(
        os.path.join(MODELS_DIR, "VG06-age-understood-spoken-td",
                     "production_rate_by_understood.csv")
    )

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    ax.fill_between(td["words_understood"], td["hdi_lo"], td["hdi_hi"],
                    color=TD_COLOUR, alpha=0.18, linewidth=0, label="TD 90% HDI")
    ax.fill_between(ds_vg09["words_understood"], ds_vg09["hdi_lo"], ds_vg09["hdi_hi"],
                    color=DS_COLOUR, alpha=0.18, linewidth=0, label="DS 90% HDI")
    ax.plot(td["words_understood"], td["q_median"], color=TD_COLOUR, lw=2.5,
            label="TD median q (VG06)")
    ax.plot(ds_vg09["words_understood"], ds_vg09["q_median"], color=DS_COLOUR, lw=2.5,
            label="DS median q (VG09)")
    ax.plot(ds_vg07["words_understood"], ds_vg07["q_median"], color=DS_COLOUR,
            lw=1.5, linestyle="--", alpha=0.7,
            label="DS median q (VG07, no subject RE)")

    for thresh in (0.5, 0.9):
        ax.axhline(thresh, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")

    ax.set_xlim(0, max(td["words_understood"].max(),
                       ds_vg09["words_understood"].max()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected words understood")
    ax.set_ylabel(r"Production ratio  q = $p_S$ / $p_U$")
    ax.set_title("Production ratio against words understood — DS (VG09) vs TD (VG06)")
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood.png"))
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood.svg"))
    plt.close(fig)

    rows = []
    for pop, df in (("DS (VG09)", ds_vg09), ("DS (VG07)", ds_vg07), ("TD (VG06)", td)):
        for thresh in (0.25, 0.5, 0.75, 0.90):
            rows.append({
                "population": pop,
                "threshold": thresh,
                "n_understood_at_median": first_crossing(
                    df["words_understood"].to_numpy(), df["q_median"].to_numpy(), thresh),
                "n_understood_at_hdi_lo": first_crossing(
                    df["words_understood"].to_numpy(), df["hdi_lo"].to_numpy(), thresh),
                "n_understood_at_hdi_hi": first_crossing(
                    df["words_understood"].to_numpy(), df["hdi_hi"].to_numpy(), thresh),
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "ds_td_q_crossings.csv"),
                              index=False)


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    ds_td_spoken_by_age()
    ds_td_understood_by_age()
    vg05_vs_vg07()
    ds_td_q_vs_understood()

    print(f"Comparisons written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
