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
level latent trajectory (no study or subject REs, i.e. matching Figure 22 /
27 in the report). Results are summarised as median + 50% / 90% HDI across
draws and written as CSVs, then plotted as a two-panel DS-vs-TD figure.

Outputs in `output/comparisons/`:
- `ds_td_learn_to_say_latency_DA.csv`
- `ds_td_learn_to_say_latency_extra.csv`
- `ds_td_learn_to_say_latency.{png,svg}`
"""

from __future__ import annotations

import os

import arviz as az
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DS_DIR = "output/models/VG10-age-understood-spoken-ds-re-subj-uq-anchored"
TD_DIR = "output/models/VG06-age-understood-spoken-td"
OUT_DIR = "output/comparisons"

# n_trials is model-specific. See src/vocab_growth/models/definitions.py.
# VG10: 800-item DS inventory.
# VG06: 800-item TD reference inventory. WG and Oxford CDI contribute
#       bivariate observations; WS is production-only in Wordbank and
#       contributes spoken observations only.
N_TRIALS_DS = 800
N_TRIALS_TD = 800
MIN_COVERAGE = 0.80  # require at least 80% of draws to have a valid crossing
N_GRID = np.array(
    [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 200,
     250, 300, 350, 400, 450, 500, 550, 600],
    dtype=float,
)


def load_population_trajectory(trace_path: str, n_trials: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ages, U, S) where U and S are (n_draw, n_age) word counts at the population level.

    ``n_trials`` must match the inventory size used when the model was fit
    (see src/vocab_growth/models/definitions.py).
    """
    d = az.from_netcdf(trace_path)
    p_u = d.posterior["p_u_plot"].values  # (chain, draw, n_age)
    p_s = d.posterior["p_s_plot"].values
    n_chain, n_draw, n_age = p_u.shape
    p_u = p_u.reshape(n_chain * n_draw, n_age)
    p_s = p_s.reshape(n_chain * n_draw, n_age)
    ages = d.constant_data["X_plot"].values.astype(float)
    U = p_u * n_trials
    S = p_s * n_trials
    # Ensure ages are sorted (they are by construction, but guard)
    order = np.argsort(ages)
    return ages[order], U[:, order], S[:, order]


def first_crossing_age(Y: np.ndarray, ages: np.ndarray, N: float) -> np.ndarray:
    """For each row of Y (a draw), return the first age where Y >= N.

    Linear interpolation between adjacent grid points. NaN where never reached.
    """
    mask = Y >= N  # (n_draw, n_age)
    any_above = mask.any(axis=1)
    first_idx = mask.argmax(axis=1)  # 0 if no crossing or already-above at idx 0

    j = first_idx
    j_prev = np.maximum(j - 1, 0)
    y0 = np.take_along_axis(Y, j_prev[:, None], axis=1).squeeze(1)
    y1 = np.take_along_axis(Y, j[:, None], axis=1).squeeze(1)
    a0 = ages[j_prev]
    a1 = ages[j]

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = y1 - y0
        interp = np.where(denom == 0, a1, a0 + (N - y0) * (a1 - a0) / denom)

    crossing = np.where(j == 0, ages[0], interp)
    crossing = np.where(any_above, crossing, np.nan)
    return crossing


def evaluate_at_ages(Y: np.ndarray, ages: np.ndarray, target_ages: np.ndarray) -> np.ndarray:
    """Linearly interpolate each row of Y at target_ages (per-row), returning NaN out of range."""
    n_draw, n_age = Y.shape
    idx = np.searchsorted(ages, target_ages, side="right")
    idx = np.clip(idx, 1, n_age - 1)
    a_lo = ages[idx - 1]
    a_hi = ages[idx]
    Y_lo = np.take_along_axis(Y, (idx - 1)[:, None], axis=1).squeeze(1)
    Y_hi = np.take_along_axis(Y, idx[:, None], axis=1).squeeze(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (target_ages - a_lo) / (a_hi - a_lo)
        out = Y_lo + t * (Y_hi - Y_lo)
    out_of_range = (target_ages < ages[0]) | (target_ages > ages[-1]) | np.isnan(target_ages)
    return np.where(out_of_range, np.nan, out)


def hdi_from_samples(x: np.ndarray, prob: float) -> tuple[float, float]:
    """Return HDI bounds of 1-D array x at given probability, ignoring NaN."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan
    x_sorted = np.sort(x)
    n = x_sorted.size
    k = int(np.floor(prob * n))
    if k >= n:
        return float(x_sorted[0]), float(x_sorted[-1])
    widths = x_sorted[k:] - x_sorted[: n - k]
    i = int(np.argmin(widths))
    return float(x_sorted[i]), float(x_sorted[i + k])


def summarise_per_N(samples: np.ndarray, N_grid: np.ndarray) -> pd.DataFrame:
    """Median + 50% / 90% HDI across draws (axis=0), with coverage fraction."""
    rows = []
    n_draw = samples.shape[0]
    for i, N in enumerate(N_grid):
        col = samples[:, i]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        cov = n_valid / n_draw
        if n_valid == 0:
            rows.append({"N": N, "coverage": cov, "median": np.nan,
                         "hdi50_lo": np.nan, "hdi50_hi": np.nan,
                         "hdi90_lo": np.nan, "hdi90_hi": np.nan})
            continue
        med = float(np.nanmedian(col))
        l50, u50 = hdi_from_samples(col, 0.50)
        l90, u90 = hdi_from_samples(col, 0.90)
        rows.append({"N": N, "coverage": cov, "median": med,
                     "hdi50_lo": l50, "hdi50_hi": u50,
                     "hdi90_lo": l90, "hdi90_hi": u90})
    return pd.DataFrame(rows)


def compute_latency(ages: np.ndarray, U: np.ndarray, S: np.ndarray,
                    N_grid: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute DA(N) and extra(N) per draw, then summarise."""
    n_draw = U.shape[0]
    n_N = len(N_grid)
    DA = np.full((n_draw, n_N), np.nan)
    extra = np.full((n_draw, n_N), np.nan)
    for i, N in enumerate(N_grid):
        a_U = first_crossing_age(U, ages, N)
        a_S = first_crossing_age(S, ages, N)
        DA[:, i] = a_S - a_U
        U_at_a_S = evaluate_at_ages(U, ages, a_S)
        extra[:, i] = U_at_a_S - N
    da_df = summarise_per_N(DA, N_grid)
    extra_df = summarise_per_N(extra, N_grid)
    return da_df, extra_df


def plot_population_panel(ax, df: pd.DataFrame, label: str, colour: str) -> None:
    df_ok = df[df["coverage"] >= MIN_COVERAGE].copy()
    if df_ok.empty:
        return
    ax.fill_between(
        df_ok["N"], df_ok["hdi90_lo"], df_ok["hdi90_hi"],
        color=colour, alpha=0.15, linewidth=0, label=f"{label} 90% HDI",
    )
    ax.fill_between(
        df_ok["N"], df_ok["hdi50_lo"], df_ok["hdi50_hi"],
        color=colour, alpha=0.30, linewidth=0, label=f"{label} 50% HDI",
    )
    ax.plot(df_ok["N"], df_ok["median"], color=colour, lw=2.5, label=f"{label} median")


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
    plot_population_panel(ax, da_ds, "DS (VG10)", plot_styles.COLOUR_BLUE)
    plot_population_panel(ax, da_td, "TD (VG06)", plot_styles.COLOUR_ORANGE)
    ax.set_xlabel("Vocabulary count N (words)")
    ax.set_ylabel(r"$\Delta A(N) = a_S(N) - a_U(N)$  (months)")
    ax.set_title("Age lag between understanding and saying N words")
    ax.set_xscale("log")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    plot_population_panel(ax, extra_ds, "DS (VG10)", plot_styles.COLOUR_BLUE)
    plot_population_panel(ax, extra_td, "TD (VG06)", plot_styles.COLOUR_ORANGE)
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

    print("\n--- DS DA(N) summary (months) ---")
    print(da_ds[da_ds["coverage"] >= MIN_COVERAGE]
          [["N", "median", "hdi50_lo", "hdi50_hi", "coverage"]].to_string(index=False))
    print("\n--- TD DA(N) summary (months) ---")
    print(da_td[da_td["coverage"] >= MIN_COVERAGE]
          [["N", "median", "hdi50_lo", "hdi50_hi", "coverage"]].to_string(index=False))
    print("\n--- DS extra(N) summary (words) ---")
    print(extra_ds[extra_ds["coverage"] >= MIN_COVERAGE]
          [["N", "median", "hdi50_lo", "hdi50_hi", "coverage"]].to_string(index=False))
    print("\n--- TD extra(N) summary (words) ---")
    print(extra_td[extra_td["coverage"] >= MIN_COVERAGE]
          [["N", "median", "hdi50_lo", "hdi50_hi", "coverage"]].to_string(index=False))


if __name__ == "__main__":
    main()
