# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Reframe the comprehension-production gap as a learn-to-say latency.

For each population (DS via VG10, TD via VG06) and each target vocabulary
count N:

- a_U(N) = first age at which the latent expected U(a) reaches N
- a_S(N) = first age at which the latent expected S(a) reaches N
- DA(N)    = a_S(N) - a_U(N)        (months between understanding and saying N)
- extra(N) = U(a_S(N)) - N          (extra words understood when S first hits N)

Both quantities are computed per posterior draw on each model's population-
level latent trajectory (no study or subject REs). Results are summarised as
median + 50% / 90% HDI across draws and written as CSVs, then plotted as a
two-panel DS-vs-TD figure.

The numerical helpers live in ``vocab_growth.comparison``; this script is a thin
CLI wrapper. Model pairs are resolved from ``MODEL_REGISTRY`` keys, so changing
the compared models is a one-line edit here (or add a new registry entry).

Outputs in ``output/comparisons/``:
- ``ds_td_learn_to_say_latency_DA.csv``
- ``ds_td_learn_to_say_latency_extra.csv``
- ``ds_td_learn_to_say_latency.{png,svg}``
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vocab_growth import comparison

# Re-exported for backward compatibility with verify_ds_td_latency.py and
# compare_ds_td_q_overlap.py, which import these names from this module.
from vocab_growth.comparison import (  # noqa: F401
    compute_latency,
    evaluate_at_ages,
    first_crossing_age,
    hdi_from_samples,
    load_population_trajectory,
    summarise_per_N,
)

DS_KEY = "vg10"
TD_KEY = "vg06"

DS_DIR = comparison.model_dir(DS_KEY)
TD_DIR = comparison.model_dir(TD_KEY)
OUT_DIR = os.path.join("output", "comparisons")

N_TRIALS_DS = comparison.n_trials(DS_KEY)
N_TRIALS_TD = comparison.n_trials(TD_KEY)
MIN_COVERAGE = 0.80
N_GRID = np.array(
    [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200,
     250, 300, 350, 400, 450, 500, 550, 600],
    dtype=float,
)


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading traces …", flush=True)
    ages_ds, U_ds, S_ds = load_population_trajectory(os.path.join(DS_DIR, "trace.nc"), N_TRIALS_DS)
    ages_td, U_td, S_td = load_population_trajectory(os.path.join(TD_DIR, "trace.nc"), N_TRIALS_TD)
    print(f"  DS: {U_ds.shape[0]} draws, ages {ages_ds.min():.1f}-{ages_ds.max():.1f}, n_trials={N_TRIALS_DS}")
    print(f"  TD: {U_td.shape[0]} draws, ages {ages_td.min():.1f}-{ages_td.max():.1f}, n_trials={N_TRIALS_TD}")

    print("Computing DS latency …", flush=True)
    da_ds, extra_ds = compute_latency(ages_ds, U_ds, S_ds, N_GRID)
    print("Computing TD latency …", flush=True)
    da_td, extra_td = compute_latency(ages_td, U_td, S_td, N_GRID)

    da = pd.concat(
        [da_ds.assign(population="DS"), da_td.assign(population="TD")], ignore_index=True,
    )
    extra = pd.concat(
        [extra_ds.assign(population="DS"), extra_td.assign(population="TD")], ignore_index=True,
    )
    da.to_csv(os.path.join(OUT_DIR, "ds_td_learn_to_say_latency_DA.csv"), index=False)
    extra.to_csv(os.path.join(OUT_DIR, "ds_td_learn_to_say_latency_extra.csv"), index=False)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    comparison.plot_summary_band(ax, da_ds, "N", "DS (VG10)", plot_styles.COLOUR_BLUE, min_coverage=MIN_COVERAGE)
    comparison.plot_summary_band(ax, da_td, "N", "TD (VG06)", plot_styles.COLOUR_ORANGE, min_coverage=MIN_COVERAGE)
    ax.set_xlabel("Vocabulary count N (words)")
    ax.set_ylabel(r"$\Delta A(N) = a_S(N) - a_U(N)$  (months)")
    ax.set_title("Age lag between understanding and saying N words")
    ax.set_xscale("log")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    comparison.plot_summary_band(ax, extra_ds, "N", "DS (VG10)", plot_styles.COLOUR_BLUE, min_coverage=MIN_COVERAGE)
    comparison.plot_summary_band(ax, extra_td, "N", "TD (VG06)", plot_styles.COLOUR_ORANGE, min_coverage=MIN_COVERAGE)
    ax.set_xlabel("Spoken count N (words)")
    ax.set_ylabel("Extra words understood when first saying N  (= U(a_S(N)) - N)")
    ax.set_title("Vocabulary lag at production-matched points")
    ax.set_xscale("log")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Modeling the gap as a learn-to-say latency — DS (VG10) vs TD (VG06)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_learn_to_say_latency.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_learn_to_say_latency.svg"))
    plt.close(fig)

    print("\nSaved:")
    for f in [
        "ds_td_learn_to_say_latency_DA.csv",
        "ds_td_learn_to_say_latency_extra.csv",
        "ds_td_learn_to_say_latency.png",
        "ds_td_learn_to_say_latency.svg",
    ]:
        print(f"  {os.path.join(OUT_DIR, f)}")

    for name, df in (("DS DA(N)", da_ds), ("TD DA(N)", da_td),
                     ("DS extra(N)", extra_ds), ("TD extra(N)", extra_td)):
        print(f"\n--- {name} summary ---")
        print(df[df["coverage"] >= MIN_COVERAGE]
              [["N", "median", "hdi50_lo", "hdi50_hi", "coverage"]].to_string(index=False))


if __name__ == "__main__":
    main()
