# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Population-level posterior overlap of q(a) = S/U for DS (VG10) and TD (VG06).

Two views:

  (A) q at matched **comprehension** level U = N — strips age out, asking
      "given that a child understands N words, what fraction do they say?".
      This is the natural quantitative complement to Figure 28.

  (B) q at matched **age** a — direct chronological comparison.

For each posterior draw and each grid point (N or a), we compute q as a
deterministic function of the latent expected probabilities (no Beta-
Binomial sampling and no subject REs — this is **population-level**
posterior overlap, not predictive overlap of individual children). For
predictive overlap of individuals, a TD bivariate model with subject
random effects would be needed first.

Outputs in `output/comparisons/`:
  - `ds_td_q_at_U_summary.csv` and `ds_td_q_at_age_summary.csv`
  - `ds_td_q_overlap.{png,svg}` — two-panel median+HDI + P(DS>TD) panel
  - `ds_td_q_density_slices.{png,svg}` — small multiples of full posterior
    densities at selected N values
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from compare_ds_td_latency import (
    DS_DIR,
    N_TRIALS_DS,
    N_TRIALS_TD,
    TD_DIR,
    evaluate_at_ages,
    first_crossing_age,
    load_population_trajectory,
    summarise_per_N,
)
from scipy.stats import gaussian_kde

OUT_DIR = "output/comparisons"

# Grid for the q(U=N) view. Avoid the very small N tail (S/U noisy when U
# is barely above N) and the very large tail where TD coverage drops.
N_GRID_Q = np.array(
    [10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200,
     250, 300, 350, 400, 450],
    dtype=float,
)

# Grid for the q(a) view. TD model fits ages 8-30 months only.
AGE_GRID_Q = np.linspace(8.0, 30.0, 45)

# N values at which to render full posterior density slices.
N_SLICES = [25.0, 50.0, 100.0, 200.0, 400.0]

MIN_COVERAGE = 0.80
RNG = np.random.default_rng(0)


def compute_q_at_U(ages: np.ndarray, U: np.ndarray, S: np.ndarray,
                   N_grid: np.ndarray) -> np.ndarray:
    """Per-draw q at comprehension level N. Shape (n_draw, n_N). NaN where
    U(a) never reaches N for that draw."""
    n_draw = U.shape[0]
    q = np.full((n_draw, len(N_grid)), np.nan)
    for i, N in enumerate(N_grid):
        a_U = first_crossing_age(U, ages, N)
        S_at_a_U = evaluate_at_ages(S, ages, a_U)
        # Where a_U is the left edge (already-above case) compute q anyway
        with np.errstate(invalid="ignore"):
            q[:, i] = S_at_a_U / N
    return q


def compute_q_at_age(ages: np.ndarray, U: np.ndarray, S: np.ndarray,
                     age_grid: np.ndarray) -> np.ndarray:
    """Per-draw q at chronological age a. Shape (n_draw, n_age_grid)."""
    n_draw = U.shape[0]
    q = np.full((n_draw, len(age_grid)), np.nan)
    for j, a in enumerate(age_grid):
        target = np.full(n_draw, a)
        U_at = evaluate_at_ages(U, ages, target)
        S_at = evaluate_at_ages(S, ages, target)
        with np.errstate(invalid="ignore", divide="ignore"):
            q[:, j] = S_at / U_at
    return q


def prob_a_greater_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each column index i, MC estimate of P(a[:, i] > b[:, i]) treating
    a and b as independent posteriors. NaN where either column has no data."""
    n_iter = 20000
    p = np.full(a.shape[1], np.nan)
    for i in range(a.shape[1]):
        a_i = a[~np.isnan(a[:, i]), i]
        b_i = b[~np.isnan(b[:, i]), i]
        if a_i.size == 0 or b_i.size == 0:
            continue
        a_s = RNG.choice(a_i, size=n_iter, replace=True)
        b_s = RNG.choice(b_i, size=n_iter, replace=True)
        p[i] = float(np.mean(a_s > b_s))
    return p


def plot_overlay_panel(ax, df: pd.DataFrame, x_col: str, label: str,
                       colour: str) -> None:
    df_ok = df[df["coverage"] >= MIN_COVERAGE]
    if df_ok.empty:
        return
    ax.fill_between(df_ok[x_col], df_ok["hdi90_lo"], df_ok["hdi90_hi"],
                    alpha=0.15, color=colour, linewidth=0,
                    label=f"{label} 90% HDI")
    ax.fill_between(df_ok[x_col], df_ok["hdi50_lo"], df_ok["hdi50_hi"],
                    alpha=0.30, color=colour, linewidth=0,
                    label=f"{label} 50% HDI")
    ax.plot(df_ok[x_col], df_ok["median"], color=colour, lw=2.5,
            label=f"{label} median")


def main() -> None:
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading traces …", flush=True)
    ages_ds, U_ds, S_ds = load_population_trajectory(
        os.path.join(DS_DIR, "trace.nc"), N_TRIALS_DS)
    ages_td, U_td, S_td = load_population_trajectory(
        os.path.join(TD_DIR, "trace.nc"), N_TRIALS_TD)

    print("Computing q(U=N) for DS …", flush=True)
    qU_ds = compute_q_at_U(ages_ds, U_ds, S_ds, N_GRID_Q)
    print("Computing q(U=N) for TD …", flush=True)
    qU_td = compute_q_at_U(ages_td, U_td, S_td, N_GRID_Q)
    print("Computing q(a) for DS …", flush=True)
    qa_ds = compute_q_at_age(ages_ds, U_ds, S_ds, AGE_GRID_Q)
    print("Computing q(a) for TD …", flush=True)
    qa_td = compute_q_at_age(ages_td, U_td, S_td, AGE_GRID_Q)

    print("Summarising and computing P(DS > TD) …", flush=True)
    qU_ds_sum = summarise_per_N(qU_ds, N_GRID_Q).rename(columns={"N": "N"})
    qU_td_sum = summarise_per_N(qU_td, N_GRID_Q).rename(columns={"N": "N"})
    qa_ds_sum = summarise_per_N(qa_ds, AGE_GRID_Q).rename(columns={"N": "age_months"})
    qa_td_sum = summarise_per_N(qa_td, AGE_GRID_Q).rename(columns={"N": "age_months"})

    pU = prob_a_greater_b(qU_ds, qU_td)
    pa = prob_a_greater_b(qa_ds, qa_td)

    qU = pd.concat([
        qU_ds_sum.assign(population="DS"),
        qU_td_sum.assign(population="TD"),
    ], ignore_index=True)
    qU.to_csv(os.path.join(OUT_DIR, "ds_td_q_at_U_summary.csv"), index=False)

    qa = pd.concat([
        qa_ds_sum.assign(population="DS"),
        qa_td_sum.assign(population="TD"),
    ], ignore_index=True)
    qa.to_csv(os.path.join(OUT_DIR, "ds_td_q_at_age_summary.csv"), index=False)

    pd.DataFrame({"N": N_GRID_Q, "P(q_DS_gt_q_TD)": pU}).to_csv(
        os.path.join(OUT_DIR, "ds_td_q_at_U_P_DS_gt_TD.csv"), index=False)
    pd.DataFrame({"age_months": AGE_GRID_Q, "P(q_DS_gt_q_TD)": pa}).to_csv(
        os.path.join(OUT_DIR, "ds_td_q_at_age_P_DS_gt_TD.csv"), index=False)

    # ============================================================
    # Figure 1: median + HDI overlay + P(DS > TD)
    # ============================================================
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.28, wspace=0.22)

    ax_qU = fig.add_subplot(gs[0, 0])
    plot_overlay_panel(ax_qU, qU_ds_sum, "N", "DS (VG10)", plot_styles.COLOUR_BLUE)
    plot_overlay_panel(ax_qU, qU_td_sum, "N", "TD (VG06)", plot_styles.COLOUR_ORANGE)
    ax_qU.set_xscale("log")
    ax_qU.set_xlabel("Comprehension N (words)")
    ax_qU.set_ylabel("q(U=N) = E[S] / N")
    ax_qU.set_title("Production ratio at matched comprehension")
    ax_qU.legend(loc="lower right", frameon=True, fontsize=9)
    ax_qU.grid(True, which="both", alpha=0.3)
    ax_qU.set_ylim(0, 1.05)

    ax_qa = fig.add_subplot(gs[0, 1])
    plot_overlay_panel(ax_qa, qa_ds_sum, "age_months", "DS (VG10)", plot_styles.COLOUR_BLUE)
    plot_overlay_panel(ax_qa, qa_td_sum, "age_months", "TD (VG06)", plot_styles.COLOUR_ORANGE)
    ax_qa.set_xlabel("Age (months)")
    ax_qa.set_ylabel("q(a) = E[S(a)] / E[U(a)]")
    ax_qa.set_title("Production ratio at matched age")
    ax_qa.legend(loc="lower right", frameon=True, fontsize=9)
    ax_qa.grid(True, alpha=0.3)
    ax_qa.set_ylim(0, 1.05)

    ax_pU = fig.add_subplot(gs[1, 0], sharex=ax_qU)
    ax_pU.plot(N_GRID_Q, pU, color="black", lw=2)
    ax_pU.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax_pU.set_xscale("log")
    ax_pU.set_xlabel("Comprehension N (words)")
    ax_pU.set_ylabel(r"$P(q_{DS} > q_{TD})$")
    ax_pU.set_ylim(0, 1)
    ax_pU.grid(True, which="both", alpha=0.3)

    ax_pa = fig.add_subplot(gs[1, 1], sharex=ax_qa)
    ax_pa.plot(AGE_GRID_Q, pa, color="black", lw=2)
    ax_pa.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax_pa.set_xlabel("Age (months)")
    ax_pa.set_ylabel(r"$P(q_{DS} > q_{TD})$")
    ax_pa.set_ylim(0, 1)
    ax_pa.grid(True, alpha=0.3)

    fig.suptitle(
        "Posterior overlap of population q = S/U — DS (VG10) vs TD (VG06)",
        fontsize=13,
    )
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_overlap.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_overlap.svg"), bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # Figure 2: full posterior densities of q at selected N
    # ============================================================
    fig, axes = plt.subplots(
        1, len(N_SLICES), figsize=(3.0 * len(N_SLICES), 4.5), sharey=True,
    )
    if len(N_SLICES) == 1:
        axes = [axes]
    q_grid = np.linspace(0.0, 1.0, 401)
    for ax, N in zip(axes, N_SLICES, strict=True):
        i = int(np.argmin(np.abs(N_GRID_Q - N)))
        for samples, colour, label in [
            (qU_ds[:, i], plot_styles.COLOUR_BLUE, "DS"),
            (qU_td[:, i], plot_styles.COLOUR_ORANGE, "TD"),
        ]:
            valid = samples[~np.isnan(samples)]
            if valid.size < 50:
                continue
            kde = gaussian_kde(valid, bw_method="scott")
            density = kde(q_grid)
            ax.fill_between(q_grid, 0, density, color=colour, alpha=0.35, label=label)
            ax.plot(q_grid, density, color=colour, lw=1.5)
            med = float(np.median(valid))
            ax.axvline(med, color=colour, lw=1, ls="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("q = S / U")
        ax.set_title(f"N = {int(N)}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Posterior density")
    axes[0].legend(loc="upper left", frameon=True, fontsize=9)
    fig.suptitle(
        "Posterior of q at matched comprehension levels — DS vs TD",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_density_slices.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "ds_td_q_density_slices.svg"))
    plt.close(fig)

    # ============================================================
    # Console summary
    # ============================================================
    print("\nSaved:")
    for f in [
        "ds_td_q_at_U_summary.csv",
        "ds_td_q_at_age_summary.csv",
        "ds_td_q_at_U_P_DS_gt_TD.csv",
        "ds_td_q_at_age_P_DS_gt_TD.csv",
        "ds_td_q_overlap.png",
        "ds_td_q_overlap.svg",
        "ds_td_q_density_slices.png",
        "ds_td_q_density_slices.svg",
    ]:
        print(f"  {os.path.join(OUT_DIR, f)}")

    print("\n--- DS q(U=N) summary ---")
    print(qU_ds_sum[qU_ds_sum["coverage"] >= MIN_COVERAGE]
          [["N", "median", "hdi50_lo", "hdi50_hi", "hdi90_lo", "hdi90_hi", "coverage"]]
          .to_string(index=False, float_format="%.3f"))
    print("\n--- TD q(U=N) summary ---")
    print(qU_td_sum[qU_td_sum["coverage"] >= MIN_COVERAGE]
          [["N", "median", "hdi50_lo", "hdi50_hi", "hdi90_lo", "hdi90_hi", "coverage"]]
          .to_string(index=False, float_format="%.3f"))
    print("\n--- P(q_DS > q_TD | U=N) ---")
    print(pd.DataFrame({"N": N_GRID_Q, "P(DS>TD)": pU})
          .to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
