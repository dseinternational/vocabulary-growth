# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Overlay DS (VG10) and TD (VG06) median production ratio q(U) = E[S]/E[U]
as a function of words understood, with HDI bands. This is the bivariate
equivalent of Figure 25 in each model's report (`production_rate_by_understood`),
overlaid on a single axis.

Reads `production_rate_by_understood.csv` directly from each model's
output directory.

Output:
- `output/comparisons/ds_td_q_vs_understood_vg10.{png,svg}`
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import pandas as pd

from vocab_growth import comparison

DS_DIR = comparison.model_dir("vg10")
TD_DIR = comparison.model_dir("vg06")
OUT_DIR = "output/comparisons"


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = pd.read_csv(os.path.join(DS_DIR, "production_rate_by_understood.csv"))
    td = pd.read_csv(os.path.join(TD_DIR, "production_rate_by_understood.csv"))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ds_colour = plot_styles.COLOUR_BLUE
    td_colour = plot_styles.COLOUR_ORANGE

    ax.fill_between(
        td["words_understood"], td["hdi_lo"], td["hdi_hi"],
        color=td_colour, alpha=0.15, linewidth=0,
        label="TD (VG06) 90% HDI",
    )
    ax.fill_between(
        td["words_understood"], td["hdi50_lo"], td["hdi50_hi"],
        color=td_colour, alpha=0.30, linewidth=0,
        label="TD (VG06) 50% HDI",
    )
    ax.plot(
        td["words_understood"], td["q_median"],
        color=td_colour, lw=2.5, label="TD (VG06) median",
    )

    ax.fill_between(
        ds["words_understood"], ds["hdi_lo"], ds["hdi_hi"],
        color=ds_colour, alpha=0.15, linewidth=0,
        label="DS (VG10) 90% HDI",
    )
    ax.fill_between(
        ds["words_understood"], ds["hdi50_lo"], ds["hdi50_hi"],
        color=ds_colour, alpha=0.30, linewidth=0,
        label="DS (VG10) 50% HDI",
    )
    ax.plot(
        ds["words_understood"], ds["q_median"],
        color=ds_colour, lw=2.5, label="DS (VG10) median",
    )

    for thresh in (0.5, 0.9):
        ax.axhline(thresh, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")

    ax.set_xlim(0, max(ds["words_understood"].max(), td["words_understood"].max()))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Expected words understood")
    ax.set_ylabel(r"Production ratio  q = $E[S] / E[U]$")
    ax.set_title(
        "Production ratio against words understood — DS (VG10) vs TD (VG06)"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood_vg10.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood_vg10.svg"))
    plt.close(fig)

    print(
        f"Saved:\n"
        f"  {os.path.join(OUT_DIR, 'ds_td_q_vs_understood_vg10.png')}\n"
        f"  {os.path.join(OUT_DIR, 'ds_td_q_vs_understood_vg10.svg')}"
    )
    print(
        f"\nDS U range covered: {ds['words_understood'].min():.0f} – "
        f"{ds['words_understood'].max():.0f}"
    )
    print(
        f"TD U range covered: {td['words_understood'].min():.0f} – "
        f"{td['words_understood'].max():.0f}"
    )


if __name__ == "__main__":
    main()
