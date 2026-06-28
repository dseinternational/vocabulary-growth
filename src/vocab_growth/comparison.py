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
from vocab_growth.models.definitions import MODEL_REGISTRY, ModelType

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

    Linear interpolation between grid points. Returns ``None`` if the threshold
    is never reached, *or* if it is already exceeded at the first grid point —
    then the true crossing lies below the observed range and the milestone is
    unidentified, not ``x[0]`` (e.g. a fast child already past 25 words at the
    youngest modelled age). Used for summarising pre-computed median/HDI curves
    (CSV-based scripts), notably the time-to-milestone inversions.
    """
    above = y >= threshold
    if not above.any():
        return None
    i = int(np.argmax(above))
    if i == 0:
        # Already at/above the threshold at the youngest grid point: a genuine
        # crossing only if it equals the threshold there, else below the range.
        return float(x[0]) if float(y[0]) == float(threshold) else None
    x0, x1 = float(x[i - 1]), float(x[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return x1
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def first_crossing_age(Y: np.ndarray, ages: np.ndarray, N: float) -> np.ndarray:
    """Per-draw first age where each row of Y (n_draw, n_age) reaches N.

    Linear interpolation between adjacent grid points. Returns NaN where the
    level is never reached, *and* where it is already exceeded at the youngest
    supported age: a "crossing" flagged at the first grid point is only real if
    the series equals N there, otherwise the true crossing lies below the grid
    and is unidentified. (Without this guard, evaluating S/U-style ratios at a
    level below what a short-support model reaches at its first age clamps the
    evaluation to ``ages[0]`` and fabricates a spurious ``S(ages[0]) / N``
    hyperbola — e.g. the TD comprehension-matched q below ~40 understood words.)
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
    below_support = (j == 0) & (Y[:, 0] > N)
    crossing = np.where(below_support, np.nan, crossing)
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
    if len(df_ok) == 1:
        # Too few points for a band/line — show the single identified estimate as
        # a point with its 90% HDI so the figure is never silently empty.
        r = df_ok.iloc[0]
        ax.errorbar(
            [r[x_col]], [r["median"]],
            yerr=[[r["median"] - r["hdi90_lo"]], [r["hdi90_hi"] - r["median"]]],
            fmt="o", color=colour, capsize=4, markersize=7,
            label=f"{label} median (90% HDI)",
        )
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


def save_panel(out_dir, filename, ax_setup, draw, *, figsize=(8.0, 5.0)) -> None:
    """Render one standalone figure (png + svg) to ``out_dir``.

    ``draw(ax)`` plots the content; ``ax_setup`` is forwarded to ``ax.set`` for
    labels/title/limits. One panel per figure, linear axes — so every comparison
    figure is usable on its own (no subplot grids).
    """
    import dse_research_utils.plot.styles as plot_styles
    import matplotlib.pyplot as plt

    plot_styles.set_matplotlib_default_style()
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax)
    ax.set(**ax_setup)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{filename}.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, f"{filename}.svg"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Cross-model population contrasts (separate-model, per-draw)
# ----------------------------------------------------------------------------
# These support contrasting a DS RE-model against a TD RE-model. Because the DS
# and TD datasets are disjoint, the joint posterior factorises and any per-draw
# pairing is valid, so a difference-of-draws gives an *exact* credible interval
# for the contrast (no joint model required). All curves are read at the
# population level (study/subject random effects excluded) so the estimand is
# consistent on both sides. Contrasts are meaningful only over the age range
# where both models have data; callers restrict to that overlap. A joint/stacked
# model that makes the TD-DS gap a generative object is a separate exercise
# (the reserved VG16), not provided here.


def load_outcome_trajectory(
    key: str, outcome: str = "spoken"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(ages, p, kappa, n_trials)`` for one outcome's population curve.

    Dispatches on model type so the same call works for univariate RE models
    (VG11/VG12: ``p_plot`` / ``kappa_plot``) and bivariate RE models
    (VG07-VG10: ``p_{s,u}_plot`` / ``kappa_{s,u}_plot``). ``p`` and ``kappa``
    are ``(n_draw, n_age)`` over the model's own plot grid with random effects
    excluded; ``ages`` is sorted ascending (months). ``outcome`` is
    ``"spoken"`` or ``"understood"``.
    """
    d = MODEL_REGISTRY[key]
    mt = d.model_type
    if mt is ModelType.UNIVARIATE:
        if d.outcome.value != outcome:
            raise ValueError(
                f"{key} is a '{d.outcome.value}' model; cannot serve '{outcome}'."
            )
        p_name, k_name = "p_plot", "kappa_plot"
    elif mt is ModelType.BIVARIATE:
        if outcome == "spoken":
            p_name, k_name = "p_s_plot", "kappa_s_plot"
        elif outcome == "understood":
            p_name, k_name = "p_u_plot", "kappa_u_plot"
        else:
            raise ValueError(
                f"outcome must be 'spoken' or 'understood', got {outcome!r}."
            )
    else:
        raise ValueError(
            f"{key}: model_type {mt} is not supported by load_outcome_trajectory."
        )

    d_idata = az.from_netcdf(trace_path(key))
    post = _dataset(d_idata, "posterior")
    cdata = _dataset(d_idata, "constant_data")
    p = post[p_name].values  # (chain, draw, n_age)
    k = post[k_name].values
    n_chain, n_draw, n_age = p.shape
    p = p.reshape(n_chain * n_draw, n_age)
    k = k.reshape(n_chain * n_draw, n_age)
    ages = np.asarray(cdata["X_plot"].values, dtype=float)
    order = np.argsort(ages)
    return ages[order], p[:, order], k[:, order], n_trials(key)


def implied_sd_y(p: np.ndarray, kappa: np.ndarray, n: int) -> np.ndarray:
    """Beta-Binomial implied SD of the word count Y (words).

    For ``BetaBinomial(n, alpha=p*kappa, beta=(1-p)*kappa)`` the variance is
    ``n*p*(1-p)*(kappa+n)/(kappa+1)``; this returns its square root. This is the
    observable between-child spread at age ``a`` — closer to what clinicians see
    than ``kappa`` itself, but note it also moves with the mean level ``p``.
    """
    var = n * p * (1.0 - p) * (kappa + n) / (kappa + 1.0)
    return np.sqrt(var)


def overdispersion_factor(kappa: np.ndarray, n: int) -> np.ndarray:
    """Variance inflation vs a Binomial at the same mean: ``(kappa+n)/(kappa+1)``.

    Mean-independent (a function of ``kappa`` and ``n`` only), so contrasting it
    across populations isolates the pure concentration difference, unlike
    :func:`implied_sd_y` which is confounded by where each population sits on the
    mean-variance curve.
    """
    return (kappa + n) / (kappa + 1.0)


def align_draws(
    n_a: int, n_b: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays pairing two *independent* posteriors to a common draw count.

    DS and TD models are fit to disjoint data, so the joint posterior factorises
    and any pairing is valid; permute each and truncate to ``min(n_a, n_b)`` to
    get an unbiased paired sample for per-draw contrasts.
    """
    n = min(n_a, n_b)
    rng = np.random.default_rng(seed)
    return rng.permutation(n_a)[:n], rng.permutation(n_b)[:n]


def interp_draws(ages: np.ndarray, Y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linear-interpolate each row of ``Y`` (n_draw, n_age) onto a shared ``grid``."""
    out = np.empty((Y.shape[0], grid.size), dtype=float)
    for i in range(Y.shape[0]):
        out[i] = np.interp(grid, ages, Y[i])
    return out


def learning_rate(ages: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-draw derivative ``dY/d(age)`` via central differences (shape of ``Y``)."""
    return np.gradient(Y, ages, axis=1)


def summarise_draws(
    samples: np.ndarray,
    grid: np.ndarray,
    grid_name: str = "age_months",
    *,
    with_p_gt0: bool = False,
) -> pd.DataFrame:
    """Median + 50%/90% HDI per grid column (over axis 0), NaN-aware.

    Adds ``coverage`` (fraction of non-NaN draws) and, when ``with_p_gt0``,
    the posterior probability ``P(contrast > 0)``.
    """
    rows = []
    n_draw = samples.shape[0]
    for i, g in enumerate(grid):
        col = samples[:, i]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        row: dict[str, float] = {grid_name: float(g), "coverage": n_valid / n_draw}
        if n_valid == 0:
            row.update(
                median=np.nan, hdi50_lo=np.nan, hdi50_hi=np.nan,
                hdi90_lo=np.nan, hdi90_hi=np.nan,
            )
            if with_p_gt0:
                row["p_gt0"] = np.nan
        else:
            c = col[valid]
            l50, u50 = hdi_from_samples(c, 0.50)
            l90, u90 = hdi_from_samples(c, 0.90)
            row.update(
                median=float(np.median(c)), hdi50_lo=l50, hdi50_hi=u50,
                hdi90_lo=l90, hdi90_hi=u90,
            )
            if with_p_gt0:
                row["p_gt0"] = float(np.mean(c > 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Expressive-delay & distributional contrasts (per-draw, separate-model)
# ----------------------------------------------------------------------------
# These extend the per-draw DS-vs-TD contrast lens to the "expressive delay"
# question — is DS production delayed *beyond* its comprehension delay? — and to
# distributional (not just mean) contrasts. Everything here is a deterministic
# functional of the already-fitted, disjoint DS and TD posteriors (no joint
# model / VG16 required); callers pair draws with :func:`align_draws` first.


def attainment_ages(W: np.ndarray, ages: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Per-draw age at which trajectory ``W`` (n_draw, n_age) reaches each level.

    Returns ``(n_draw, n_level)``; NaN where a level is not reached on the grid.
    """
    return np.column_stack([first_crossing_age(W, ages, float(v)) for v in levels])


def expressive_specific_delay(
    ages_ds: np.ndarray, U_ds: np.ndarray, S_ds: np.ndarray,
    ages_td: np.ndarray, U_td: np.ndarray, S_td: np.ndarray,
    levels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Level-indexed expressive-specific delay (difference-in-differences).

    For each vocabulary level ``N`` and paired draw:

    * ``D_U(N)``     = a_U^DS(N) - a_U^TD(N)   — comprehension attainment delay
    * ``D_S(N)``     = a_S^DS(N) - a_S^TD(N)   — production attainment delay
    * ``delta_exp``  = D_S(N) - D_U(N)         — the *extra* production delay DS
      carries beyond its comprehension delay (== latency_DS - latency_TD).

    A ``delta_exp`` > 0 means DS production lags further behind TD than its
    comprehension does — an expressive-specific deficit, not just global slowing.
    All arrays are ``(n_draw, n_level)``. Inputs must be draw-paired and equal
    length (see :func:`align_draws`).
    """
    aU_ds = attainment_ages(U_ds, ages_ds, levels)
    aS_ds = attainment_ages(S_ds, ages_ds, levels)
    aU_td = attainment_ages(U_td, ages_td, levels)
    aS_td = attainment_ages(S_td, ages_td, levels)
    lat_ds = aS_ds - aU_ds
    lat_td = aS_td - aU_td
    return {
        "D_U": aU_ds - aU_td,
        "D_S": aS_ds - aS_td,
        "latency_ds": lat_ds,
        "latency_td": lat_td,
        "delta_exp": lat_ds - lat_td,
    }


def _invert_trajectory(
    ages: np.ndarray, W: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Per-draw age at which monotone ``W`` (n_draw, n_age) reaches ``targets``.

    ``targets`` is ``(n_draw, n_target)``. Linear interpolation on each draw's
    (age-increasing) curve; NaN where a target lies outside that draw's observed
    level range — i.e. where matching would require extrapolating the reference.
    """
    n_draw, n_target = targets.shape
    out = np.full((n_draw, n_target), np.nan)
    for d in range(n_draw):
        out[d] = np.interp(targets[d], W[d], ages, left=np.nan, right=np.nan)
    return out


def comprehension_equivalent_age(
    ages_ds: np.ndarray, U_ds: np.ndarray, S_ds: np.ndarray,
    ages_td: np.ndarray, U_td: np.ndarray, S_td: np.ndarray,
    age_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Age-indexed developmental-age view of the expressive delay.

    For DS evaluated at each chronological age ``a`` in ``age_grid``:

    * ``cea_U(a)`` — the TD age whose understood level equals DS's at ``a``
      (DS's *comprehension-equivalent* developmental age)
    * ``cea_S(a)`` — the TD age whose spoken level equals DS's at ``a``
      (DS's *production-equivalent* age)
    * ``delay_U(a) = a - cea_U(a)`` — receptive delay (months)
    * ``delay_S(a) = a - cea_S(a)`` — expressive delay (months)
    * ``delta_exp_age(a) = cea_U(a) - cea_S(a)`` — the extra expressive delay:
      DS's speech looks like a TD child *younger* than its comprehension age.

    All arrays ``(n_draw, n_age_grid)``; NaN where the DS level falls outside the
    TD level range (the reference is not extrapolated). Draw-paired inputs.
    """
    U_ds_g = interp_draws(ages_ds, U_ds, age_grid)
    S_ds_g = interp_draws(ages_ds, S_ds, age_grid)
    cea_U = _invert_trajectory(ages_td, U_td, U_ds_g)
    cea_S = _invert_trajectory(ages_td, S_td, S_ds_g)
    a = age_grid[None, :]
    return {
        "cea_U": cea_U,
        "cea_S": cea_S,
        "delay_U": a - cea_U,
        "delay_S": a - cea_S,
        "delta_exp_age": cea_U - cea_S,
    }


def fraction_below_reference_percentile(
    p_ds: np.ndarray, k_ds: np.ndarray,
    p_td: np.ndarray, k_td: np.ndarray,
    n_trials_: int, pct: float = 10.0,
) -> np.ndarray:
    """Per-draw fraction of the DS Beta-Binomial child distribution at/below the
    TD ``pct``-th percentile word count, on a common grid.

    A clinically legible "how atypical" estimand: at each age, what share of DS
    children fall below the TD ``pct``-th centile. ``p_*``/``k_*`` are
    ``(n_draw, n_grid)`` population mean proportions and Beta-Binomial
    concentrations. Returns ``(n_draw, n_grid)``.
    """
    from scipy.stats import betabinom

    a_td, b_td = p_td * k_td, (1.0 - p_td) * k_td
    a_ds, b_ds = p_ds * k_ds, (1.0 - p_ds) * k_ds
    thresh = betabinom.ppf(pct / 100.0, n_trials_, a_td, b_td)
    return betabinom.cdf(thresh, n_trials_, a_ds, b_ds)


def peak_growth_age(ages: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Per-draw age of maximum learning rate dW/da over the grid (n_draw,).

    Note: a value at the first/last grid age is *censored* — the true peak may
    lie outside the model's plotting range — so callers should report the share
    pinned at the boundary alongside the contrast.
    """
    rate = learning_rate(ages, W)
    return ages[np.argmax(rate, axis=1)]


def load_p_any_trajectory(
    path: str, n_trials_: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ages, p_any_words)`` for a trivariate/joint model's total
    expressive trajectory ``p_any = P(word produced in any modality)``.

    ``p_any_words`` is ``(n_draw, n_age)`` expected word counts over the plot
    grid (signing included), for the DS sign-inclusive expressive contrast.
    """
    d = az.from_netcdf(path)
    post = _dataset(d, "posterior")
    cdata = _dataset(d, "constant_data")
    p_any = post["p_any_plot"].values
    n_chain, n_draw, n_age = p_any.shape
    p_any = p_any.reshape(n_chain * n_draw, n_age)
    ages = np.asarray(cdata["X_plot"].values, dtype=float)
    order = np.argsort(ages)
    return ages[order], (p_any * n_trials_)[:, order]


def shade_unsupported(
    ax, support_lo: float, support_hi: float, *, colour: str = "0.85",
    label: str | None = "outside reference support",
) -> None:
    """Shade x-regions outside ``[support_lo, support_hi]`` (e.g. the TD age
    support) so extrapolated/unsupported regions are visually flagged."""
    x_lo, x_hi = ax.get_xlim()
    first = True
    if x_lo < support_lo:
        ax.axvspan(x_lo, support_lo, color=colour, alpha=0.5, lw=0,
                   label=label if first else None, zorder=0)
        first = False
    if x_hi > support_hi:
        ax.axvspan(support_hi, x_hi, color=colour, alpha=0.5, lw=0,
                   label=label if first else None, zorder=0)
    ax.set_xlim(x_lo, x_hi)
