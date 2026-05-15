# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Summarise the per-study random-intercept posteriors from VG07.

Loads `output/models/VG07-age-understood-spoken-ds-re/trace.nc`,
extracts `delta_u[study]`, `delta_q[study]`, `tau_u`, `tau_q`, and
writes:

- `output/models/VG07-age-understood-spoken-ds-re/study_effects.csv`
- `output/models/VG07-age-understood-spoken-ds-re/study_effects.{png,svg}`
  Forest plot showing each study's logit-scale offset on the understood
  and production-ratio trajectories.

The study_id → study label mapping is reconstructed by replicating the
sorting used in `common_bivariate_re.prepare_bivariate_re_data`.
"""

from __future__ import annotations

import os

import arviz as az
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vocab_growth.data_utils import load_data
from vocab_growth.models.definitions import Population

MODEL_DIR = "output/models/VG07-age-understood-spoken-ds-re"
HDI_PROB = 0.90


def study_labels() -> list[str]:
    df = load_data(Population.DOWN_SYNDROME,
                   ["age", "understood", "spoken", "study"])
    df = df.dropna(subset=["age"])
    df = df[df["understood"].notna() | df["spoken"].notna()]
    return sorted(df["study"].unique())


def _summary_row(label: str, samples: np.ndarray) -> dict:
    hdi = az.hdi(samples, prob=HDI_PROB)
    return {
        "parameter": label,
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "sd": float(np.std(samples)),
        "hdi_lo": float(hdi[0]),
        "hdi_hi": float(hdi[1]),
    }


def main() -> None:
    plot_styles.set_matplotlib_default_style()

    print("Loading VG07 trace …", flush=True)
    idata = az.from_netcdf(os.path.join(MODEL_DIR, "trace.nc"))
    post = idata.posterior

    labels = study_labels()
    assert len(labels) == post.sizes["study_id"], (
        f"Expected {post.sizes['study_id']} studies, got {len(labels)}"
    )

    delta_u = post["delta_u"].stack(sample=("chain", "draw"))
    delta_q = post["delta_q"].stack(sample=("chain", "draw"))
    tau_u = post["tau_u"].stack(sample=("chain", "draw")).values
    tau_q = post["tau_q"].stack(sample=("chain", "draw")).values

    rows: list[dict] = []
    rows.append(_summary_row("tau_u (SD of study intercepts, understood)", tau_u))
    rows.append(_summary_row("tau_q (SD of study intercepts, prod ratio)", tau_q))

    for i, name in enumerate(labels):
        u_samples = delta_u.isel(study_id=i).values
        q_samples = delta_q.isel(study_id=i).values
        rows.append(_summary_row(f"delta_u[{name}]", u_samples))
        rows.append(_summary_row(f"delta_q[{name}]", q_samples))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(MODEL_DIR, "study_effects.csv"), index=False)

    # Forest plot
    fig, axes = plt.subplots(
        ncols=2, sharey=True,
        figsize=(plot_styles.FIGSIZE_XL[0], plot_styles.FIGSIZE_XL[1] * 1.1),
    )
    y_positions = np.arange(len(labels))

    for ax, samples_da, title, colour in (
        (axes[0], delta_u, "delta_u — understood-trajectory shift (logit)",
         plot_styles.COLOUR_BLUE),
        (axes[1], delta_q, "delta_q — production-ratio shift (logit)",
         plot_styles.COLOUR_ORANGE),
    ):
        for i in range(len(labels)):
            samples = samples_da.isel(study_id=i).values
            median = np.median(samples)
            hdi = az.hdi(samples, prob=HDI_PROB)
            ax.errorbar(
                median, i,
                xerr=[[median - hdi[0]], [hdi[1] - median]],
                fmt="o", color=colour, ecolor=colour, capsize=3,
            )
        ax.axvline(0, color=plot_styles.LINE_COLOUR, lw=0.6, linestyle="--")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels)
        ax.set_title(title)
        ax.set_xlabel("Posterior median (90% HDI)")

    fig.suptitle(
        "VG07 — study random intercepts (DS, n=10 studies)", fontweight="bold",
    )
    fig.savefig(os.path.join(MODEL_DIR, "study_effects.png"))
    fig.savefig(os.path.join(MODEL_DIR, "study_effects.svg"))
    plt.close(fig)

    print(f"Wrote: {os.path.join(MODEL_DIR, 'study_effects.csv')}")
    print(f"       {os.path.join(MODEL_DIR, 'study_effects.png')}")
    print(f"       {os.path.join(MODEL_DIR, 'study_effects.svg')}")


if __name__ == "__main__":
    main()
