# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Side-by-side DS (VG10) vs TD (VG06) comparison plots for the joint
trajectory (Figure 22 equivalent) and the comprehension-production gap
(Figure 27 equivalent).

Reads the per-model CSVs already produced by the model fit pipeline:

- `joint_trajectory.csv` — posterior predictive medians + 5/25/75/95 bands
  for words understood and words spoken.
- `comprehension_production_gap.csv` — posterior median + HDI bands for the
  expected gap (p_U - p_S) * n_trials.

Outputs (in `output/comparisons/`):

- `ds_td_joint_trajectory.{png,svg}` — two panels, shared x-axis 0-115 months.
- `ds_td_comprehension_production_gap.{png,svg}` — two panels, shared x-axis.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import pandas as pd

DS_DIR = "output/models/VG10-age-understood-spoken-ds-re-subj-uq-anchored"
TD_DIR = "output/models/VG06-age-understood-spoken-td"
OUT_DIR = "output/comparisons"

UNDERSTOOD_COLOUR = "C0"
SPOKEN_COLOUR = "C1"

X_MAX = 115.0
N_TRIALS = 800


def plot_joint_panel(ax, df: pd.DataFrame, title: str) -> None:
    ax.fill_between(
        df["age_months"], df["understood_p05"], df["understood_p95"],
        alpha=0.15, color=UNDERSTOOD_COLOUR,
    )
    ax.fill_between(
        df["age_months"], df["understood_p25"], df["understood_p75"],
        alpha=0.25, color=UNDERSTOOD_COLOUR,
    )
    ax.plot(
        df["age_months"], df["understood_median"],
        lw=3, color=UNDERSTOOD_COLOUR, label="Words understood (median)",
    )

    ax.fill_between(
        df["age_months"], df["spoken_p05"], df["spoken_p95"],
        alpha=0.15, color=SPOKEN_COLOUR,
    )
    ax.fill_between(
        df["age_months"], df["spoken_p25"], df["spoken_p75"],
        alpha=0.25, color=SPOKEN_COLOUR,
    )
    ax.plot(
        df["age_months"], df["spoken_median"],
        lw=3, color=SPOKEN_COLOUR, label="Words spoken (median)",
    )

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-20, N_TRIALS + 50)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Word count")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)


def plot_gap_panel(ax, df: pd.DataFrame, title: str, hdi_prob: int = 90) -> None:
    ax.fill_between(
        df["age_months"], df["hdi_lo"], df["hdi_hi"],
        alpha=0.20, color="C2", label=f"{hdi_prob}% HDI",
    )
    ax.fill_between(
        df["age_months"], df["hdi50_lo"], df["hdi50_hi"],
        alpha=0.30, color="C2", label="50% HDI",
    )
    ax.plot(df["age_months"], df["gap_median"], lw=3, color="C2", label="Median gap")

    ax.set_xlim(0, X_MAX)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("E[understood] - E[spoken] (words)")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True)


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    ds_joint = pd.read_csv(os.path.join(DS_DIR, "joint_trajectory.csv"))
    td_joint = pd.read_csv(os.path.join(TD_DIR, "joint_trajectory.csv"))
    ds_gap = pd.read_csv(os.path.join(DS_DIR, "comprehension_production_gap.csv"))
    td_gap = pd.read_csv(os.path.join(TD_DIR, "comprehension_production_gap.csv"))

    # ---- Joint trajectory: DS vs TD ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    plot_joint_panel(axes[0], ds_joint, "Down syndrome (VG10)")
    plot_joint_panel(axes[1], td_joint, "Typically developing (VG06)")
    fig.suptitle(
        "Joint posterior predictive trajectory — words understood vs words spoken",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_joint_trajectory.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_joint_trajectory.svg"))
    plt.close(fig)

    # ---- Comprehension-production gap: DS vs TD ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_gap_panel(axes[0], ds_gap, "Down syndrome (VG10)")
    plot_gap_panel(axes[1], td_gap, "Typically developing (VG06)")
    fig.suptitle(
        "Comprehension-production gap — E[understood] - E[spoken]",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_comprehension_production_gap.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_comprehension_production_gap.svg"))
    plt.close(fig)

    print(
        "Saved:\n"
        f"  {os.path.join(OUT_DIR, 'ds_td_joint_trajectory.png')}\n"
        f"  {os.path.join(OUT_DIR, 'ds_td_joint_trajectory.svg')}\n"
        f"  {os.path.join(OUT_DIR, 'ds_td_comprehension_production_gap.png')}\n"
        f"  {os.path.join(OUT_DIR, 'ds_td_comprehension_production_gap.svg')}"
    )


if __name__ == "__main__":
    main()
