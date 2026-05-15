# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Builds overlay comparisons between the Down-syndrome bivariate model with
study random intercepts (VG07) and the typically-developing bivariate
model (VG06).

Reads `production_rate_by_understood.csv` from both model output
directories and produces:

- `output/comparisons/ds_td_q_vs_understood.{svg,png}` — overlay plot of the
  production ratio q against words understood for both populations.
- `output/comparisons/ds_td_q_crossings.csv` — words understood at which each
  population reaches q = 0.25, 0.50, 0.75, 0.90, with the same threshold
  applied to the 5% and 95% HDI bands.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VG06_DIR = "output/models/VG06-age-understood-spoken-td"
VG07_DIR = "output/models/VG07-age-understood-spoken-ds-re"
OUT_DIR = "output/comparisons"

DS_COLOUR = plot_styles.COLOUR_BLUE
DS_FILL = plot_styles.COLOUR_BLUE
TD_COLOUR = plot_styles.COLOUR_ORANGE
TD_FILL = plot_styles.COLOUR_ORANGE


def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Return the x value at which a monotone-ish curve y first crosses
    `threshold`, using linear interpolation. Returns None if no crossing."""
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


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = pd.read_csv(os.path.join(VG07_DIR, "production_rate_by_understood.csv"))
    td = pd.read_csv(os.path.join(VG06_DIR, "production_rate_by_understood.csv"))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        td["words_understood"],
        td["hdi_lo"],
        td["hdi_hi"],
        color=TD_FILL,
        alpha=0.18,
        linewidth=0,
        label="TD 90% HDI",
    )
    ax.fill_between(
        ds["words_understood"],
        ds["hdi_lo"],
        ds["hdi_hi"],
        color=DS_FILL,
        alpha=0.18,
        linewidth=0,
        label="DS 90% HDI",
    )
    ax.plot(
        td["words_understood"], td["q_median"], color=TD_COLOUR, lw=2.5,
        label="TD median q (VG06)",
    )
    ax.plot(
        ds["words_understood"], ds["q_median"], color=DS_COLOUR, lw=2.5,
        label="DS median q (VG07)",
    )

    for thresh in (0.5, 0.9):
        ax.axhline(thresh, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")

    ax.set_xlim(0, max(td["words_understood"].max(), ds["words_understood"].max()))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected words understood")
    ax.set_ylabel(r"Production ratio  q = $p_S$ / $p_U$")
    ax.set_title(
        "Production ratio against words understood — DS (VG07) vs TD (VG06)"
    )
    ax.legend(loc="lower right", frameon=True)

    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood.png"))
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_vs_understood.svg"))

    rows = []
    for label, df in (("DS (VG07)", ds), ("TD (VG06)", td)):
        for thresh in (0.25, 0.5, 0.75, 0.90):
            rows.append({
                "population": label,
                "threshold": thresh,
                "n_understood_at_median": first_crossing(
                    df["words_understood"].to_numpy(),
                    df["q_median"].to_numpy(),
                    thresh,
                ),
                "n_understood_at_hdi_lo": first_crossing(
                    df["words_understood"].to_numpy(),
                    df["hdi_lo"].to_numpy(),
                    thresh,
                ),
                "n_understood_at_hdi_hi": first_crossing(
                    df["words_understood"].to_numpy(),
                    df["hdi_hi"].to_numpy(),
                    thresh,
                ),
            })
    crossings = pd.DataFrame(rows)
    crossings.to_csv(os.path.join(OUT_DIR, "ds_td_q_crossings.csv"), index=False)

    print(crossings.to_string(index=False))
    print(
        "\nSaved: "
        f"{os.path.join(OUT_DIR, 'ds_td_q_vs_understood.svg')}\n"
        f"       {os.path.join(OUT_DIR, 'ds_td_q_vs_understood.png')}\n"
        f"       {os.path.join(OUT_DIR, 'ds_td_q_crossings.csv')}"
    )


if __name__ == "__main__":
    main()
