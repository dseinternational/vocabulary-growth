# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Compute time-to-milestone curves for every fitted model.

Given the posterior predictive percentile curves in
`posterior_summary*.csv`:

- `Y_hdi_lo(age)`  → slow-acquiring (5th-percentile) child
- `Y_median(age)`  → typical (50th-percentile) child
- `Y_hdi_hi(age)`  → fast-acquiring (95th-percentile) child

we invert each curve to answer: **at what age does the corresponding
percentile child first reach a target word count?**

Targets: 25, 50, 100, 200, 400 words.

Output per model:

- `output/models/<MODEL>/time_to_milestone[_<u|s>].csv`
- `output/models/<MODEL>/time_to_milestone[_<u|s>].png/.svg`

The same CSVs are also concatenated into
`output/comparisons/time_to_milestone_all.csv` for cross-population
contrasts.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import pandas as pd

from vocab_growth import environment as env
from vocab_growth.comparison import invert_curve

MODELS_DIR = "output/models"
COMPARE_DIR = "output/comparisons"

TARGETS = [25, 50, 100, 200, 400]

UNIVARIATE = {
    "VG01": ("VG01-age-spoken-ds", "spoken", "DS"),
    "VG02": ("VG02-age-understood-ds", "understood", "DS"),
    "VG03": ("VG03-age-spoken-td", "spoken", "TD"),
    "VG04": ("VG04-age-understood-td", "understood", "TD"),
    "VG11": ("VG11-age-spoken-td-re", "spoken", "TD"),
    "VG12": ("VG12-age-understood-td-re", "understood", "TD"),
}

BIVARIATE = {
    "VG05": ("VG05-age-understood-spoken-ds", "DS"),
    "VG06": ("VG06-age-understood-spoken-td", "TD"),
    "VG07": ("VG07-age-understood-spoken-ds-re", "DS"),
    "VG08": ("VG08-age-understood-spoken-ds-re-subj", "DS"),
    "VG09": ("VG09-age-understood-spoken-ds-re-subj-uq", "DS"),
    "VG10": ("VG10-age-understood-spoken-ds-re-subj-uq-anchored", "DS"),
    "VG13": ("VG13-age-understood-spoken-td-re-young", "TD"),
}


def plot_milestone(df: pd.DataFrame, title: str, out_base: str) -> None:
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    age = df["age_months"].to_numpy()

    ax.fill_between(
        df["Y_hdi_lo"], age, age * 0,
        where=df["Y_hdi_lo"].notna(),
        color="white", alpha=0,
    )  # establish y-axis baseline
    ax.plot(df["Y_hdi_hi"], age, color=plot_styles.COLOUR_GREEN, lw=2,
            label="Fast child (95th percentile)")
    ax.plot(df["Y_median"], age, color=plot_styles.COLOUR_BLUE, lw=2.5,
            label="Typical child (50th)")
    ax.plot(df["Y_hdi_lo"], age, color=plot_styles.COLOUR_RED, lw=2,
            label="Slow child (5th percentile)")

    for t in TARGETS:
        ax.axvline(t, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
        ax.text(t, age.min() + (age.max() - age.min()) * 0.96, f" {t} words",
                rotation=90, va="top", ha="left", fontsize=8,
                color=plot_styles.LINE_COLOUR)

    ax.set_xlim(0, max(df["Y_hdi_hi"].max(), max(TARGETS) * 1.05))
    ax.set_xlabel("Words")
    ax.set_ylabel("Age (months)")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".svg")
    plt.close(fig)


def process_univariate(short: str, label: str, outcome: str, pop: str,
                       merged_rows: list[dict]) -> None:
    model_dir = os.path.join(MODELS_DIR, label)
    summary_path = os.path.join(model_dir, "posterior_summary.csv")
    if not os.path.exists(summary_path):
        print(f"  skip {short}: not fitted yet ({summary_path} absent)")
        return
    summary = pd.read_csv(summary_path)
    inv = invert_curve(summary, TARGETS)
    inv.insert(0, "outcome", outcome)
    inv.insert(0, "population", pop)
    inv.insert(0, "model", short)
    inv.to_csv(os.path.join(model_dir, "time_to_milestone.csv"), index=False)
    plot_milestone(
        summary,
        f"Time-to-milestone — {short} ({pop}, {outcome})",
        os.path.join(model_dir, "time_to_milestone"),
    )
    merged_rows.extend(inv.to_dict("records"))


def process_bivariate(short: str, label: str, pop: str,
                      merged_rows: list[dict]) -> None:
    model_dir = os.path.join(MODELS_DIR, label)
    for outcome, suffix in (("understood", "u"), ("spoken", "s")):
        summary_path = os.path.join(model_dir, f"posterior_summary_{suffix}.csv")
        if not os.path.exists(summary_path):
            print(f"  skip {short} ({outcome}): not fitted yet ({summary_path} absent)")
            continue
        summary = pd.read_csv(summary_path)
        inv = invert_curve(summary, TARGETS)
        inv.insert(0, "outcome", outcome)
        inv.insert(0, "population", pop)
        inv.insert(0, "model", short)
        inv.to_csv(
            os.path.join(model_dir, f"time_to_milestone_{suffix}.csv"),
            index=False,
        )
        plot_milestone(
            summary,
            f"Time-to-milestone — {short} ({pop}, {outcome})",
            os.path.join(model_dir, f"time_to_milestone_{suffix}"),
        )
        merged_rows.extend(inv.to_dict("records"))


def main() -> None:
    env.preflight_disk(2.0, env.OUTPUT_DIR, label="milestone outputs")
    plot_styles.set_matplotlib_default_style()
    os.makedirs(COMPARE_DIR, exist_ok=True)

    merged: list[dict] = []
    for short, (label, outcome, pop) in UNIVARIATE.items():
        process_univariate(short, label, outcome, pop, merged)
    for short, (label, pop) in BIVARIATE.items():
        process_bivariate(short, label, pop, merged)

    merged_df = pd.DataFrame(merged)
    merged_df.to_csv(os.path.join(COMPARE_DIR, "time_to_milestone_all.csv"),
                     index=False)
    n_models = len(UNIVARIATE) + len(BIVARIATE)
    print(f"Wrote per-model CSV+plot for {n_models} models.")
    print(f"Combined: {os.path.join(COMPARE_DIR, 'time_to_milestone_all.csv')}")


if __name__ == "__main__":
    main()
