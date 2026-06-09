# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared utilities for comparing fitted vocabulary-growth models.

This module consolidates the helpers that the ``scripts/compare_*`` and
``scripts/time_to_milestone`` tools previously each re-implemented (five copies
of ``first_crossing``, two trace loaders, ad-hoc HDI code). The comparison
scripts are now thin CLI wrappers around the functions here.

Everything is **model-agnostic** and **registry-parameterised**: a comparison
target is a ``MODEL_REGISTRY`` key (e.g. ``"vg06"``, ``"vg10"``). Output
directories, vocabulary-checklist sizes (``n_trials``) and populations are
resolved from the model definition rather than hardcoded paths, so a *new*
model pair -- for example a future TD model with study intercepts -- can be
compared by adding it to the registry and passing its key. No script edits
required.

Two complementary lenses are supported:

* **Age-aligned** -- trajectories / contrasts vs chronological age. Only valid
  over the age range where *both* models have data (the TD models are fit to
  8-30 months); callers must restrict to the overlap.
* **Comprehension-matched** -- the production ratio ``q = E[S]/E[U]`` and
  derived latencies as a function of *understood vocabulary*, which removes the
  TD/DS timescale difference.

The population-level trajectories (``p_u_plot``, ``p_s_plot``, ``q_plot`` over
``X_plot``) read here are the GP+linear means with study/subject random effects
excluded, and are emitted under the same names by every bivariate model
(plain, study-RE and subject-RE), so this code is unchanged across them.
"""

from __future__ import annotations

import os

import arviz as az
import numpy as np
import pandas as pd

from vocab_growth import environment as env
from vocab_growth.models.definitions import MODEL_REGISTRY

DEFAULT_MILESTONES = (25, 50, 100, 200, 400)
DEFAULT_MIN_COVERAGE = 0.80


# ----------------------------------------------------------------------------
# Registry resolution
# ----------------------------------------------------------------------------
def model_dir(key: str) -> str:
    """Output directory for a registry key, e.g. 'vg06' -> .../VG06-age-...-td."""
    d = MODEL_REGISTRY[key]
    return os.path.join(env.MODELS_OUTPUT_DIR, f"{d.model_id}-{d.config_name}")


def trace_path(key: str) -> str:
    return os.path.join(model_dir(key), "trace.nc")


def n_trials(key: str) -> int:
    return MODEL_REGISTRY[key].n_trials


def population(key: str) -> str:
    pop = MODEL_REGISTRY[key].population
    return pop.value if hasattr(pop, "value") else str(pop)


def model_label(key: str) -> str:
    """Short display label, e.g. 'VG06 (TD)'."""
    d = MODEL_REGISTRY[key]
    return f"{d.model_id} ({population(key).upper()})"


# ----------------------------------------------------------------------------
# Trace loading
# ----------------------------------------------------------------------------
def _dataset(idata: az.InferenceData, group: str):
    """Return a group as an xarray Dataset, robust to ArviZ DataTree backing."""
    node = getattr(idata, group)
    return node if hasattr(node, "data_vars") else node.to_dataset()


def load_population_trajectory(
    path: str, n_trials_: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ages, U, S)`` for a fitted model's population-level trajectory.

    ``U`` and ``S`` are ``(n_draw, n_age)`` word counts (proportion * n_trials)
    over the plot grid; ``ages`` is sorted ascending in months. ``n_trials_``
    must match the checklist size used at fit time (see definitions.py).
    """
    d = az.from_netcdf(path)
    post = _dataset(d, "posterior")
    cdata = _dataset(d, "constant_data")
    p_u = post["p_u_plot"].values  # (chain, draw, n_age)
    p_s = post["p_s_plot"].values
    n_chain, n_draw, n_age = p_u.shape
    p_u = p_u.reshape(n_chain * n_draw, n_age)
    p_s = p_s.reshape(n_chain * n_draw, n_age)
    ages = np.asarray(cdata["X_plot"].values, dtype=float)
    order = np.argsort(ages)
    return ages[order], (p_u * n_trials_)[:, order], (p_s * n_trials_)[:, order]


def population_trajectory(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Registry-keyed convenience wrapper around :func:`load_population_trajectory`."""
    return load_population_trajectory(trace_path(key), n_trials(key))


# ----------------------------------------------------------------------------
# Crossing / interpolation / HDI helpers
# ----------------------------------------------------------------------------
def first_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Smallest x at which a monotone-ish 1-D curve y first reaches threshold.

    Linear interpolation between grid points; ``None`` if never reached. Used
    for summarising pre-computed median/HDI curves (CSV-based scripts).
    """
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


def first_crossing_age(Y: np.ndarray, ages: np.ndarray, N: float) -> np.ndarray:
    """Per-draw first age where each row of Y (n_draw, n_age) reaches N.

    Linear interpolation between adjacent grid points; NaN where never reached.
    """
    mask = Y >= N
    any_above = mask.any(axis=1)
    first_idx = mask.argmax(axis=1)
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
    return np.where(any_above, crossing, np.nan)


def evaluate_at_ages(
    Y: np.ndarray, ages: np.ndarray, target_ages: np.ndarray
) -> np.ndarray:
    """Per-row linear interpolation of Y (n_draw, n_age) at target_ages (n_draw,).

    NaN where a target age is outside the grid.
    """
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
    out_of_range = (
        (target_ages < ages[0]) | (target_ages > ages[-1]) | np.isnan(target_ages)
    )
    return np.where(out_of_range, np.nan, out)


def hdi_from_samples(x: np.ndarray, prob: float) -> tuple[float, float]:
    """Narrowest-interval HDI of a 1-D sample array, ignoring NaN."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan
    xs = np.sort(x)
    n = xs.size
    k = int(np.floor(prob * n))
    if k >= n:
        return float(xs[0]), float(xs[-1])
    widths = xs[k:] - xs[: n - k]
    i = int(np.argmin(widths))
    return float(xs[i]), float(xs[i + k])


def summarise_per_N(samples: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """Median + 50%/90% HDI across draws (axis 0) per grid column, with coverage."""
    rows = []
    n_draw = samples.shape[0]
    for i, N in enumerate(grid):
        col = samples[:, i]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        cov = n_valid / n_draw
        if n_valid == 0:
            rows.append(
                {"N": N, "coverage": cov, "median": np.nan,
                 "hdi50_lo": np.nan, "hdi50_hi": np.nan,
                 "hdi90_lo": np.nan, "hdi90_hi": np.nan}
            )
            continue
        l50, u50 = hdi_from_samples(col, 0.50)
        l90, u90 = hdi_from_samples(col, 0.90)
        rows.append(
            {"N": N, "coverage": cov, "median": float(np.nanmedian(col)),
             "hdi50_lo": l50, "hdi50_hi": u50, "hdi90_lo": l90, "hdi90_hi": u90}
        )
    return pd.DataFrame(rows)


def prob_a_greater_b(
    a: np.ndarray, b: np.ndarray, *, n_iter: int = 20000, rng=None
) -> np.ndarray:
    """Per-column MC estimate of P(a > b) treating columns as independent posteriors.

    NaN-aware; NaN where either column is empty.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    p = np.full(a.shape[1], np.nan)
    for i in range(a.shape[1]):
        a_i = a[~np.isnan(a[:, i]), i]
        b_i = b[~np.isnan(b[:, i]), i]
        if a_i.size == 0 or b_i.size == 0:
            continue
        a_s = rng.choice(a_i, size=n_iter, replace=True)
        b_s = rng.choice(b_i, size=n_iter, replace=True)
        p[i] = float(np.mean(a_s > b_s))
    return p


# ----------------------------------------------------------------------------
# Analyses (per-draw, population-level)
# ----------------------------------------------------------------------------
def compute_latency(
    ages: np.ndarray, U: np.ndarray, S: np.ndarray, N_grid: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Learn-to-say latency. Returns (DA_summary, extra_summary) per N.

    DA(N) = a_S(N) - a_U(N); extra(N) = U(a_S(N)) - N, per draw, then summarised.
    """
    n_draw = U.shape[0]
    DA = np.full((n_draw, len(N_grid)), np.nan)
    extra = np.full((n_draw, len(N_grid)), np.nan)
    for i, N in enumerate(N_grid):
        a_U = first_crossing_age(U, ages, N)
        a_S = first_crossing_age(S, ages, N)
        DA[:, i] = a_S - a_U
        extra[:, i] = evaluate_at_ages(U, ages, a_S) - N
    return summarise_per_N(DA, N_grid), summarise_per_N(extra, N_grid)


def compute_q_at_U(
    ages: np.ndarray, U: np.ndarray, S: np.ndarray, N_grid: np.ndarray
) -> np.ndarray:
    """Per-draw production ratio q at matched comprehension U=N: q = S(a_U(N)) / N."""
    q = np.full((U.shape[0], len(N_grid)), np.nan)
    for i, N in enumerate(N_grid):
        a_U = first_crossing_age(U, ages, N)
        with np.errstate(invalid="ignore"):
            q[:, i] = evaluate_at_ages(S, ages, a_U) / N
    return q


def compute_q_at_age(
    ages: np.ndarray, U: np.ndarray, S: np.ndarray, age_grid: np.ndarray
) -> np.ndarray:
    """Per-draw production ratio q at chronological age: q(a) = E[S(a)] / E[U(a)]."""
    n_draw = U.shape[0]
    q = np.full((n_draw, len(age_grid)), np.nan)
    for j, a in enumerate(age_grid):
        target = np.full(n_draw, a)
        with np.errstate(invalid="ignore", divide="ignore"):
            q[:, j] = evaluate_at_ages(S, ages, target) / evaluate_at_ages(
                U, ages, target
            )
    return q


def invert_curve(
    df: pd.DataFrame, targets=DEFAULT_MILESTONES
) -> pd.DataFrame:
    """Age at which the 5%/50%/95% percentile child first reaches each target.

    Inverts the pre-computed posterior-predictive percentile columns
    (``Y_hdi_lo``/``Y_median``/``Y_hdi_hi``) of a ``posterior_summary*`` frame.
    """
    age = df["age_months"].to_numpy(dtype=float)
    y_lo = df["Y_hdi_lo"].to_numpy(dtype=float)
    y_md = df["Y_median"].to_numpy(dtype=float)
    y_hi = df["Y_hdi_hi"].to_numpy(dtype=float)
    rows = []
    for target in targets:
        rows.append(
            {
                "target_words": target,
                "age_fast_child_p95": first_crossing(age, y_hi, target),
                "age_typical_child_p50": first_crossing(age, y_md, target),
                "age_slow_child_p5": first_crossing(age, y_lo, target),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def overlay_age_curves(title, series, out_base, *, ylabel="Expected words"):
    """Overlay Ey_median + Ey_hdi bands from posterior_summary-shaped frames.

    ``series`` is a list of ``(label, dataframe, colour)``; each frame needs
    ``age_months``, ``Ey_median``, ``Ey_hdi_lo``, ``Ey_hdi_hi``.
    """
    import dse_research_utils.plot.styles as plot_styles
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    for label, df, colour in series:
        ax.fill_between(
            df["age_months"], df["Ey_hdi_lo"], df["Ey_hdi_hi"],
            color=colour, alpha=0.18, linewidth=0, label=f"{label} 90% HDI",
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


def plot_summary_band(
    ax, df: pd.DataFrame, x_col: str, label: str, colour: str,
    *, min_coverage: float = DEFAULT_MIN_COVERAGE, show_50: bool = True,
) -> None:
    """Plot a median line with 90% (and optionally 50%) HDI bands from a
    :func:`summarise_per_N`-shaped frame, dropping low-coverage grid points."""
    df_ok = df[df["coverage"] >= min_coverage] if "coverage" in df else df
    if df_ok.empty:
        return
    ax.fill_between(
        df_ok[x_col], df_ok["hdi90_lo"], df_ok["hdi90_hi"],
        color=colour, alpha=0.15, linewidth=0, label=f"{label} 90% HDI",
    )
    if show_50:
        ax.fill_between(
            df_ok[x_col], df_ok["hdi50_lo"], df_ok["hdi50_hi"],
            color=colour, alpha=0.30, linewidth=0, label=f"{label} 50% HDI",
        )
    ax.plot(df_ok[x_col], df_ok["median"], color=colour, lw=2.5, label=f"{label} median")
