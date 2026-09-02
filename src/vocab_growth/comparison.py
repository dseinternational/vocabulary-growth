# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared utilities for comparing fitted vocabulary-growth models.

This module consolidates the helpers that the ``scripts/compare_*`` and
``scripts/time_to_milestone`` tools previously each re-implemented (five copies
of ``first_crossing``, two trace loaders, ad-hoc HDI code). The comparison
scripts are now thin CLI wrappers around the functions here.

Everything is **model-agnostic** and **registry-parameterised**: a comparison
target is a ``MODEL_REGISTRY`` key (e.g. ``"vg11"``, ``"vg10"``). Output
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
import dse_research_utils.plot.io as plot_io
import dse_research_utils.statistics.intervals as shared_intervals
import numpy as np
import pandas as pd

from vocab_growth import environment as env
from vocab_growth import intervals
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    ModelType,
    subject_slope_spec,
)
from vocab_growth.models.subject_effects import DEFAULT_SLOPE_REF_AGE_MONTHS

DEFAULT_MILESTONES = (25, 50, 100, 200, 400)
DEFAULT_MIN_COVERAGE = 0.80


# ----------------------------------------------------------------------------
# Registry resolution
# ----------------------------------------------------------------------------
def model_dir(key: str) -> str:
    """Output directory for a registry key, e.g. 'vg11' -> .../VG11-age-...-td."""
    d = MODEL_REGISTRY[key]
    return os.path.join(env.models_output_dir(), f"{d.model_id}-{d.config_name}")


def trace_path(key: str) -> str:
    return os.path.join(model_dir(key), "trace.nc")


def n_trials(key: str) -> int:
    return MODEL_REGISTRY[key].n_trials


def population(key: str) -> str:
    pop = MODEL_REGISTRY[key].population
    return pop.value if hasattr(pop, "value") else str(pop)


def model_label(key: str) -> str:
    """Short display label, e.g. 'VG11 (TD)'."""
    d = MODEL_REGISTRY[key]
    return f"{d.model_id} ({population(key).upper()})"


# ----------------------------------------------------------------------------
# Trace loading
# ----------------------------------------------------------------------------
def _dataset(idata: az.InferenceData, group: str):
    """Return a group as an xarray Dataset, robust to ArviZ DataTree backing."""
    node = getattr(idata, group)
    return node if hasattr(node, "data_vars") else node.to_dataset()


def _load_reshaped_draws(
    path: str,
    var_names: tuple[str, ...],
    scalar_names: tuple[str, ...] = (),
) -> tuple[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
    """Load named posterior variables over the ``X_plot`` grid, draw-flattened.

    Shared tail of every population-trajectory loader below: opens the trace,
    reshapes each named posterior variable from ``(chain, draw, n_age)`` to
    ``(chain*draw, n_age)``, and age-sorts both the variables and the grid.
    Returns ``(ages_sorted, [var_sorted, ...], {scalar: draws})`` with the
    grid variables in the same order as ``var_names``.

    ``scalar_names`` additionally pulls per-draw *scalar* parameters (shape
    ``(chain, draw)``, e.g. a random-effect scale) and flattens them the same
    C-order way, so index ``i`` of a scalar and row ``i`` of a grid variable are
    the same posterior draw.
    """
    d = az.from_netcdf(path)
    post = _dataset(d, "posterior")
    cdata = _dataset(d, "constant_data")
    ages = np.asarray(cdata["X_plot"].values, dtype=float)
    order = np.argsort(ages)
    arrays = []
    for name in var_names:
        arr = post[name].values  # (chain, draw, n_age)
        n_chain, n_draw, n_age = arr.shape
        arrays.append(arr.reshape(n_chain * n_draw, n_age)[:, order])
    scalars = {}
    for name in scalar_names:
        if name not in post:
            raise KeyError(
                f"{os.path.basename(os.path.dirname(path))}: posterior has no "
                f"variable {name!r}."
            )
        scalars[name] = np.asarray(post[name].values, dtype=float).reshape(-1)
    return ages[order], arrays, scalars


def load_population_trajectory(
    path: str, n_trials_: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ages, U, S)`` for a fitted model's population-level trajectory.

    ``U`` and ``S`` are ``(n_draw, n_age)`` word counts (proportion * n_trials)
    over the plot grid; ``ages`` is sorted ascending in months. ``n_trials_``
    must match the checklist size used at fit time (see definitions.py).
    """
    ages, (p_u, p_s), _ = _load_reshaped_draws(path, ("p_u_plot", "p_s_plot"))
    return ages, p_u * n_trials_, p_s * n_trials_


def load_population_trajectory_weighted(
    path: str, n_trials_: int, frame, *, bandwidth: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(ages, U, S)`` for the administration-weighted child of a joint RE model.

    The counterpart of :func:`load_population_trajectory`, which returns the
    reference child (zero study and child effects: the child in the *average
    study*). Study effects are centred over studies, and studies are segregated
    by age, so at a given age the reference child can sit above or below every
    study sampled there -- 54 words below the Down syndrome pool's median child
    at 38 months, 46 above the typically developing pool's at 21 -- and a
    milestone or a delay read off it inherits that. This re-weights the same fit
    to the studies present at each age (a Gaussian kernel over ``frame``'s
    administrations, ``bandwidth`` months), which is the child the sample
    medians describe. Report both; the gap is the study-coverage sensitivity.
    """
    d = az.from_netcdf(path)
    post = _dataset(d, "posterior")
    cdata = _dataset(d, "constant_data")
    ages = np.asarray(cdata["X_plot"].values, dtype=float)
    order = np.argsort(ages)

    def flat(name):
        arr = post[name].values
        return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])

    f_u, h = flat("f_u_plot")[:, order], flat("h_plot")[:, order]  # (S, n_age)
    d_u, d_q = flat("delta_u"), flat("delta_q")  # (S, K)
    codes = np.asarray(frame["study_code"], dtype=int)
    obs_ages = np.asarray(frame["age"], dtype=float)
    ages_sorted = ages[order]
    kernel = np.exp(-0.5 * ((ages_sorted[:, None] - obs_ages[None, :]) / bandwidth) ** 2)
    K = int(codes.max()) + 1
    weights = np.stack([kernel[:, codes == k].sum(axis=1) for k in range(K)], axis=1)
    weights /= np.where(weights.sum(axis=1, keepdims=True) > 0, weights.sum(axis=1, keepdims=True), 1.0)

    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    U = np.zeros_like(f_u)
    S = np.zeros_like(f_u)
    for k in range(K):
        w = weights[:, k][None, :]
        if not np.any(w):
            continue
        pu_k = sig(f_u + d_u[:, k][:, None])
        U += w * pu_k
        S += w * pu_k * sig(h + d_q[:, k][:, None])
    return ages_sorted, U * n_trials_, S * n_trials_


#: Series the joint sign/speech engine (VG15) reports on the plot grid, as
#: fractions. ``pi_*`` are the four-cell composition **conditional on the word
#: being understood**, so they are scaled by ``p_u`` — not by ``n_trials`` alone —
#: to become word counts. Getting that wrong silently inflates every cell by
#: ``1 / p_u``, which at 12 months is a factor of fifty.
SIGN_SPEECH_SERIES = (
    "p_u_plot",
    "q_plot",
    "p_any_plot",
    "p_any_indep_plot",
    "r_plot",
    "pi_sign_only_plot",
    "pi_both_plot",
    "pi_speak_only_plot",
)


def load_sign_speech_trajectory(
    path: str, n_trials_: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``(ages, series)`` for the joint sign/speech engine's trajectory.

    ``series`` maps each name below to ``(n_draw, n_age)`` **word counts**, all
    from one posterior so they are draw-aligned with each other:

    ``understood``
        Expected words understood.
    ``spoken``
        ``p_u * q``. VG15 emits no ``p_s_plot`` — spoken is a ratio of
        understood in this engine, so it is reconstructed here rather than read.
    ``any``
        Total expressive vocabulary, in any modality, with the sign–speech
        association ``psi`` estimated from the data.
    ``any_indep``
        The same total computed **as if** sign and speech were independent given
        age — the assumption VG14 has no choice but to make. Shipping both makes
        the cost of that assumption a visible contrast rather than an argument.
    ``sign_only`` / ``both`` / ``speak_only``
        The composition of expressive vocabulary. ``sign_only`` is the count a
        speech-only assessment would miss entirely.

    ``r`` is returned separately as a **fraction** (of understood words signed),
    because it is a ratio by construction and a word count of it is meaningless.
    """
    ages, arrays, _ = _load_reshaped_draws(path, SIGN_SPEECH_SERIES)
    p_u, q, p_any, p_any_indep, r, sign_only, both, speak_only = arrays
    understood = p_u * n_trials_
    return ages, {
        "understood": understood,
        "spoken": p_u * q * n_trials_,
        "any": p_any * n_trials_,
        "any_indep": p_any_indep * n_trials_,
        "sign_only": p_u * sign_only * n_trials_,
        "both": p_u * both * n_trials_,
        "speak_only": p_u * speak_only * n_trials_,
        "r": r,
    }


def load_univariate_trajectory(
    path: str, n_trials_: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ages, W)`` for a single-outcome model's population trajectory.

    ``W`` is ``(n_draw, n_age)`` word counts (``p_plot`` * ``n_trials``) over the
    plot grid; ``ages`` is sorted ascending. The single-outcome analogue of
    :func:`load_population_trajectory`.
    """
    ages, (p,), _ = _load_reshaped_draws(path, ("p_plot",))
    return ages, p * n_trials_


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
    unidentified, not ``x[0]`` (e.g. a curve already past 25 words at the
    youngest modelled age). Used for summarising a single pre-computed median/HDI
    curve (CSV-based scripts, e.g. compare_models.py). For posterior milestone
    ages prefer the per-draw :func:`attainment_ages` / :func:`milestone_table`,
    which give the correct median-of-crossings rather than crossing-of-median.
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
    """Narrowest-interval HDI of a 1-D sample array, ignoring NaN.

    Delegates to the shared :func:`dse_research_utils.statistics.intervals.hdi_1d`
    (an identical ``floor(prob * n)`` construction); kept as a local name for the
    existing call sites.
    """
    return shared_intervals.hdi_1d(x, hdi_prob=prob)


def summarise_per_N(samples: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """Median + 50%/89% interval across draws (axis 0) per grid column, with coverage.

    Thin alias of :func:`summarise_draws` with ``grid_name="N"`` — kept as a
    separate name for the (many) call sites that read as "per N words".
    """
    return summarise_draws(samples, grid, "N")


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
    """Per-draw population production ratio at the age the population comprehension
    trajectory reaches N: q = S(a_U(N)) / N.

    This is the population q(age) re-indexed by population comprehension, not
    E[q_i | U_i = N]: it uses no child effects and no rho_uq. A child-level
    "given a child understands N words" statement would condition on the subject
    random effects (and, in VG20, their correlation), which this transformation
    does not touch.
    """
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


def milestone_table(
    W: np.ndarray,
    ages: np.ndarray,
    targets=DEFAULT_MILESTONES,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
) -> pd.DataFrame:
    """Posterior age at which the trajectory first reaches each target word count.

    ``W`` is the ``(n_draw, n_age)`` per-draw population count trajectory (from
    :func:`load_population_trajectory` / :func:`load_univariate_trajectory`). For
    each target this computes the crossing age *per draw* (:func:`attainment_ages`)
    and summarises that distribution — the correct **median-of-crossings**, not
    the age at which the median curve crosses the target (crossing-of-median),
    which the two differ for a nonlinear trajectory. The reported interval is the
    posterior HDI on the milestone age for the population trajectory; it is *not*
    a spread across individual "percentile children" (that would need new-child
    posterior-predictive draws — see the predictive-interval caveat in the report).

    ``prop_reaching`` is the fraction of draws that reach the target anywhere on
    the modelled age grid; the age summaries are over those draws only, so a low
    ``prop_reaching`` means the median/HDI ages are conditional and should be read
    with care.
    """
    A = attainment_ages(W, ages, np.asarray(list(targets), dtype=float))
    rows = []
    for j, target in enumerate(targets):
        a = A[:, j]
        reached = a[~np.isnan(a)]
        prop = float(reached.size) / float(a.size) if a.size else 0.0
        if reached.size == 0:
            rows.append({
                "target_words": target, "age_median": None,
                "age_ci50_lo": None, "age_ci50_hi": None,
                "age_ci_lo": None, "age_ci_hi": None, "prop_reaching": prop,
            })
            continue
        # Milestone ages are boundary-censored/skewed -> highest-density interval.
        lo, hi = hdi_from_samples(reached, ci_prob)
        lo50, hi50 = hdi_from_samples(reached, intervals.INNER_CI_PROB)
        rows.append({
            "target_words": target,
            "age_median": float(np.median(reached)),
            "age_ci50_lo": lo50, "age_ci50_hi": hi50,
            "age_ci_lo": lo, "age_ci_hi": hi, "prop_reaching": round(prop, 3),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def overlay_age_curves(title, series, out_base, *, ylabel="Expected words"):
    """Overlay Ey_median + Ey interval bands from posterior_summary-shaped frames.

    ``series`` is a list of ``(label, dataframe, colour)``; each frame needs
    ``age_months``, ``Ey_median``, ``Ey_ci_lo``, ``Ey_ci_hi``.
    """
    import dse_research_utils.plot.styles as plot_styles
    import matplotlib.pyplot as plt

    pct = int(round(intervals.DEFAULT_CI_PROB * 100))
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    for label, df, colour in series:
        ax.fill_between(
            df["age_months"], df["Ey_ci_lo"], df["Ey_ci_hi"],
            color=colour, alpha=0.18, linewidth=0, label=f"{label} {pct}% interval",
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
    plot_io.save_styled_figure(
        os.path.dirname(out_base), os.path.basename(out_base), fig=fig, bbox_inches=None
    )


def plot_summary_band(
    ax, df: pd.DataFrame, x_col: str, label: str, colour: str,
    *, min_coverage: float = DEFAULT_MIN_COVERAGE, show_50: bool = True,
) -> None:
    """Plot a median line with an 89% (and optionally 50%) interval band from a
    :func:`summarise_per_N`-shaped frame, dropping low-coverage grid points."""
    pct = int(round(intervals.DEFAULT_CI_PROB * 100))
    df_ok = df[df["coverage"] >= min_coverage] if "coverage" in df else df
    if df_ok.empty:
        return
    if len(df_ok) == 1:
        # Too few points for a band/line — show the single identified estimate as
        # a point with its interval so the figure is never silently empty.
        r = df_ok.iloc[0]
        ax.errorbar(
            [r[x_col]], [r["median"]],
            yerr=[[r["median"] - r["ci_lo"]], [r["ci_hi"] - r["median"]]],
            fmt="o", color=colour, capsize=4, markersize=7,
            label=f"{label} median ({pct}% interval)",
        )
        return
    ax.fill_between(
        df_ok[x_col], df_ok["ci_lo"], df_ok["ci_hi"],
        color=colour, alpha=0.15, linewidth=0, label=f"{label} {pct}% interval",
    )
    if show_50:
        ax.fill_between(
            df_ok[x_col], df_ok["ci50_lo"], df_ok["ci50_hi"],
            color=colour, alpha=0.30, linewidth=0, label=f"{label} 50% interval",
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
    plot_io.save_styled_figure(out_dir, filename, fig=fig, bbox_inches=None)


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

    ages, (p, k), _ = _load_reshaped_draws(trace_path(key), (p_name, k_name))
    return ages, p, k, n_trials(key)


def product_marginal_kappa(
    p_u: np.ndarray, kappa_u: np.ndarray, q: np.ndarray, kappa_s: np.ndarray
) -> np.ndarray:
    """Concentration of the Beta-Binomial matching the marginal spoken count's variance.

    NumPy port of ``likelihood_utils.product_marginal_concentration`` (the PyMC
    form the ``product_marginal`` fallback uses in the graph), kept in step by
    :func:`tests.test_comparison.test_product_marginal_kappa_matches_the_graph_form`.

    The joint models draw ``theta_U ~ Beta(p_U kappa_U)`` and ``theta_S ~ Beta(q
    kappa_S)`` independently, with ``S | U ~ Bin(U, theta_S)``, so the marginal
    spoken count is a Binomial mixed over the *product* of two Betas. That
    product has no Beta form but both moments are elementary::

        m         = p_U q
        E[theta^2] = p (p kappa + 1) / (kappa + 1)          for each factor
        var       = E[theta_U^2] E[theta_S^2] - m^2
        kappa_eff = m (1 - m) / var - 1

    This is the quantity a DS/TD **spoken** dispersion contrast has to use. VG20's
    ``kappa_s`` is the dispersion of the ratio ``q`` on the child's own understood
    count as denominator, and feeding it into ``(kappa + n)/(kappa + 1)`` with
    ``n = 810`` treats it as though it dispersed counts out of the item pool --
    which is what VG11's ``kappa`` does, and what the contrast then compared it
    with (``compare_ds_td_re.py``'s long-standing "known residual"). ``kappa_eff``
    is on the item-pool denominator and is comparable. It reduces to ``kappa_s``
    at ``kappa_U -> inf`` and ``p_U = 1``.
    """
    eps = 1e-9
    pu = np.clip(np.asarray(p_u, dtype=float), eps, 1 - eps)
    pc = np.clip(np.asarray(q, dtype=float), eps, 1 - eps)
    ku = np.asarray(kappa_u, dtype=float)
    ks = np.asarray(kappa_s, dtype=float)
    m = pu * pc
    e2_parent = pu * (pu * ku + 1.0) / (ku + 1.0)
    e2_child = pc * (pc * ks + 1.0) / (ks + 1.0)
    variance = np.maximum(e2_parent * e2_child - m * m, 1e-12)
    return np.maximum(m * (1.0 - m) / variance - 1.0, 1e-6)


def load_marginal_spoken_trajectory(
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(ages, p_s, kappa_eff, n_trials)`` for a joint model's marginal spoken count.

    The bivariate counterpart of :func:`load_outcome_trajectory` for the one case
    where that function's ``kappa_s_plot`` is the wrong object: ``p_s = p_u q`` is
    already marginal, but ``kappa_s`` is conditional, and a dispersion contrast
    against a univariate spoken model needs :func:`product_marginal_kappa`.
    """
    d = MODEL_REGISTRY[key]
    if d.model_type is not ModelType.BIVARIATE:
        raise ValueError(f"{key} is not a bivariate model; use load_outcome_trajectory.")
    ages, (p_u, k_u, q, k_s), _ = _load_reshaped_draws(
        trace_path(key), ("p_u_plot", "kappa_u_plot", "q_plot", "kappa_s_plot")
    )
    return ages, p_u * q, product_marginal_kappa(p_u, k_u, q, k_s), n_trials(key)


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

    A function of ``kappa`` and ``n`` only, so it removes the explicit ``p(1-p)``
    mean dependence that confounds :func:`implied_sd_y`, which is evaluated where
    each population sits on the mean-variance curve.

    That is the whole of the claim. It is **not** true that contrasting this factor
    across populations isolates a pure concentration difference: ``kappa`` is itself
    level-driven in this family, so a cross-population contrast still carries
    whatever part of the dispersion difference comes from the two populations being
    at different vocabulary levels. The reported ratio is robust; it simply does not
    isolate what the name suggests.
    """
    return (kappa + n) / (kappa + 1.0)


# ----------------------------------------------------------------------------
# Between-child heterogeneity (the subject random-effect scale)
# ----------------------------------------------------------------------------
# `kappa` — and therefore `overdispersion_factor` above — is an *observation*-level
# parameter, applied to a child-and-study-specific `p_obs`. In a model carrying
# subject random effects it is what is left after persistent between-child
# differences have been absorbed, so it does not answer "how much do children in
# this population differ from one another": that is the subject scale's job. The
# two are not merely different, they are complementary — in the TD models they are
# an explicit reparameterisation of one shared logit-scale scatter budget (see
# `models.gp_utils.build_variance_partition`), so reading either alone attributes
# the whole budget to whichever half is being looked at.
#
# The obstacle to contrasting the scales directly is that they do not all live on
# the same latent scale. The univariate TD models put one subject intercept on the
# logit of the outcome (`tau_subject`); the joint DS models put one on the logit of
# *understood* (`tau_subj_u`) and one on the logit of the production *ratio*
# (`tau_subj_q`), with spoken derived as p_u * q. So VG10 has no spoken subject
# scale to read off, and `tau_subj_q` is not VG11's `tau_subject` in different
# clothing. What both parameterisations *do* define is the between-child
# distribution of the child's own logit p for the outcome in question, which is a
# well-defined estimand in either. The functions below evaluate it — exactly for a
# single logit intercept, by quadrature for the product form.


def _gauss_hermite_standard_normal(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for ``E[g(Z)] = sum(w * g(x))`` under ``Z ~ Normal(0, 1)``."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    return x, w / w.sum()


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """``log(sigmoid(x))``, stable for large |x|."""
    return -np.logaddexp(0.0, -x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function, overflow-free at both tails (underflows to 0 / 1)."""
    return np.exp(_log_sigmoid(x))


def _logit_sigmoid_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``logit(sigmoid(a) * sigmoid(b))``, stable in both tails.

    Uses ``1 - s(a)s(b) = s(-a) + s(a)s(-b)`` so the upper tail is a log-sum-exp
    of two log-sigmoids rather than a cancelling subtraction. Needed because DS
    spoken proportions at the young end of the grid are small enough that a naive
    ``log(p) - log(1-p)`` on a clipped ``p`` would report the clip, not the model.
    """
    log_p = _log_sigmoid(a) + _log_sigmoid(b)
    log_1mp = np.logaddexp(_log_sigmoid(-a), _log_sigmoid(a) + _log_sigmoid(-b))
    return log_p - log_1mp


def child_scale_of_age(
    tau0: np.ndarray,
    tau1: np.ndarray,
    rho01: np.ndarray,
    ages: np.ndarray,
    *,
    ref_age_months: float = 36.0,
) -> np.ndarray:
    """Between-child scale at each age under a child intercept-and-slope block.

    VG19. Where the model of record gives each child one constant offset, the
    child-slope block gives each child ``b0 + b1 * D`` with
    ``D = (age - ref) / 12`` in years, so the between-child SD is no longer a
    number but a curve::

        sd(age) = sqrt(tau0^2 + 2 rho01 tau0 tau1 D + tau1^2 D^2)

    which is just ``Var(b0 + b1 D)`` with ``Cov(b0, b1) = rho01 tau0 tau1``.

    ``tau0``, ``tau1`` and ``rho01`` are per-draw scalars ``(n_draw,)``; ``ages``
    is the evaluation grid ``(n_age,)``. Returns ``(n_draw, n_age)``, ready to
    pass straight to :func:`child_spread_single` or :func:`child_spread_product`,
    both of which accept an age-varying scale in place of a constant one.

    Two properties worth stating because they are what makes the reported number
    interpretable. At ``age == ref_age_months`` the scale is exactly ``tau0``,
    which is why the reference age is a definition field rather than a constant
    — ``tau0`` is a spread with a stated age attached. And the curve is a
    parabola in age with its minimum at ``D = -rho01 tau0 / tau1``: a negative
    ``rho01`` puts the tightest point in the future and children fan out on
    both sides of it, which is a real qualitative claim the constant-offset model
    cannot make and should be read off the figure rather than assumed.
    """
    d_years = (np.asarray(ages, dtype=float) - float(ref_age_months)) / 12.0
    t0 = np.asarray(tau0, dtype=float)[:, None]
    t1 = np.asarray(tau1, dtype=float)[:, None]
    r = np.asarray(rho01, dtype=float)[:, None]
    var = t0 * t0 + 2.0 * r * t0 * t1 * d_years + (t1 * d_years) ** 2
    # A variance by construction; the clip guards floating point at the edge
    # where |rho01| -> 1 and the parabola touches zero.
    return np.sqrt(np.maximum(var, 0.0))


def _tau_to_draw_age(tau: np.ndarray, shape: tuple[int, ...], *, what: str):
    """Accept a per-draw constant scale or a per-draw, per-age one.

    ``(n_draw,)`` is the constant-offset case every model up to VG20 supplies,
    and is returned as ``(n_draw, 1)`` to broadcast over ages exactly as before.
    ``(n_draw, n_age)`` is the VG19 child-slope case from
    :func:`child_scale_of_age`, and is returned unchanged. Anything else is an
    error rather than a silent broadcast, because a wrong-shaped scale here
    produces a plausible curve instead of a failure.
    """
    t = np.asarray(tau, dtype=float)
    if t.ndim == 1:
        return t[:, None]
    if t.ndim == 2:
        if t.shape != shape:
            raise ValueError(
                f"age-varying {what} has shape {t.shape}, expected {shape} to match "
                "the population logit grid."
            )
        return t
    raise ValueError(f"{what} must be 1-D (n_draw) or 2-D (n_draw, n_age); got {t.ndim}-D.")


def child_spread_single(
    f: np.ndarray, tau: np.ndarray, n: int, *, n_nodes: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """Between-child spread when one Normal intercept sits on the outcome's logit.

    ``f`` is the population logit ``(n_draw, n_age)``, ``tau`` the per-draw subject
    scale ``(n_draw,)``; a child's logit is ``f + tau*Z``, ``Z ~ Normal(0, 1)``.
    Returns ``(tau_logit, sd_child_words)``, both ``(n_draw, n_age)``: the SD of the
    child's logit p — here exactly ``tau``, broadcast — and the SD across children
    of that child's *expected* word count ``n*p`` (Beta-Binomial noise excluded,
    since this is persistent between-child variation only).
    """
    x, w = _gauss_hermite_standard_normal(n_nodes)
    tau_col = _tau_to_draw_age(tau, f.shape, what="tau")
    m1 = np.zeros_like(f)
    m2 = np.zeros_like(f)
    for xi, wi in zip(x, w, strict=True):
        p = _sigmoid(f + tau_col * xi)
        m1 += wi * p
        m2 += wi * p * p
    sd_words = n * np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    return np.broadcast_to(tau_col, f.shape).copy(), sd_words


def child_spread_product(
    f_u: np.ndarray,
    h: np.ndarray,
    tau_u: np.ndarray,
    tau_q: np.ndarray,
    n: int,
    *,
    rho: np.ndarray | None = None,
    n_nodes: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Between-child spread of spoken in a joint ``p_s = p_u * q`` model.

    A child's spoken proportion is ``sigmoid(f_u + tau_u*Z1) * sigmoid(h + tau_q*Z2)``
    with standard Normal ``Z1``, ``Z2``, so the SD of the child's *spoken* logit is
    neither ``tau_u`` nor ``tau_q`` and is age-varying even though both scales are
    constants. Returns the same ``(tau_logit, sd_child_words)`` pair as
    :func:`child_spread_single`, evaluated on a tensor Gauss-Hermite grid.

    ``rho`` is the correlation between the child's two deviations, one value per
    draw. ``None`` means independent ``Z1``, ``Z2`` — the VG05–VG16 assumption and
    this function's historical behaviour. Where a model estimates the correlation
    (VG20's ``rho_uq``), passing it applies to the quadrature nodes the same
    Cholesky the model samples under, ``Z2 = rho*Z1 + sqrt(1 - rho^2)*Z2'``.

    The correction has a known direction: ``log p_S = log p_U + log q`` gains
    ``2 Cov``, so assuming independence when the correlation is positive
    **understates** the spoken between-child spread. That asymmetry was a
    disclosed limitation of the DS-versus-TD contrast for as long as no DS model
    estimated the correlation — the TD comparator's single spoken intercept
    absorbs it whether or not anyone models it.

    This is the quantity that is like-for-like with a univariate model's
    ``tau_subject``; contrasting ``tau_subj_q`` against it instead would compare the
    spread of a conversion ratio with the spread of a level.
    """
    x, w = _gauss_hermite_standard_normal(n_nodes)
    # Decide age-varying from the CALLER's input, not from the adapter's output:
    # the adapter returns (n_draw, 1) for a constant scale, which is also 2-D.
    age_varying = np.asarray(tau_u).ndim == 2 or np.asarray(tau_q).ndim == 2
    if rho is not None and age_varying:
        # The engine refuses to build this combination (see
        # `vocab_growth.models.subject_effects.resolve`), so reaching it means a
        # caller has
        # paired a child-slope scale with a cross-outcome correlation by hand.
        # `rho` would then be read as the intercept-intercept element of a 4x4
        # covariance that was never estimated.
        raise ValueError(
            "an age-varying child scale (VG19) cannot be combined with a "
            "cross-outcome correlation (VG20's rho_uq): that is a 4x4 covariance "
            "and this quadrature assumes a 2x2."
        )
    tu = _tau_to_draw_age(tau_u, f_u.shape, what="tau_u")
    tq = _tau_to_draw_age(tau_q, h.shape, what="tau_q")
    if rho is None:
        r = s = None
    else:
        r = np.asarray(rho, dtype=float)[:, None]
        # Clipped rather than trusted: rho lives on (-1, 1) by construction, but
        # a floating-point 1 - rho^2 can go very slightly negative at the edge.
        s = np.sqrt(np.maximum(1.0 - r * r, 0.0))
    p1 = np.zeros_like(f_u)
    p2 = np.zeros_like(f_u)
    l1 = np.zeros_like(f_u)
    l2 = np.zeros_like(f_u)
    for xi, wi in zip(x, w, strict=True):
        a = f_u + tu * xi
        for xj, wj in zip(x, w, strict=True):
            b = h + (tq * xj if r is None else tq * (r * xi + s * xj))
            weight = wi * wj
            p = _sigmoid(a) * _sigmoid(b)
            lg = _logit_sigmoid_product(a, b)
            p1 += weight * p
            p2 += weight * p * p
            l1 += weight * lg
            l2 += weight * lg * lg
    tau_logit = np.sqrt(np.maximum(l2 - l1 * l1, 0.0))
    sd_words = n * np.sqrt(np.maximum(p2 - p1 * p1, 0.0))
    return tau_logit, sd_words


def subject_heterogeneity(
    key: str,
    outcome: str = "spoken",
    *,
    ages: np.ndarray | None = None,
    draws: np.ndarray | None = None,
    n_nodes: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(ages, tau_logit, sd_child_words, n_trials)`` for one outcome.

    The between-child counterpart of :func:`load_outcome_trajectory`: how far
    children of the same age in this population sit from one another, on the
    outcome's own logit scale and in expected words. Study effects are excluded
    throughout, matching every other population-level curve in this module.

    Dispatches on model type, mirroring :func:`load_outcome_trajectory`: univariate
    RE models read ``f_plot`` + ``tau_subject``; bivariate RE models read
    ``f_u_plot`` + ``tau_subj_u`` for understood, and additionally ``h_plot`` +
    ``tau_subj_q`` for spoken, which needs :func:`child_spread_product`.

    Where a model gives its children a *rate* rather than a constant offset
    (VG19), the scale it reads is not a number but the curve
    :func:`child_scale_of_age` builds from ``tau_subj_*_0``, ``tau_subj_*_1`` and
    ``tau_subj_*_rho``, evaluated on the returned grid. Both quadratures accept a
    ``(n_draw, n_age)`` scale, so the dispatch is the only thing that changes.
    ``tau_logit`` is then age-varying for two reasons at once — the population
    logit moves and the child scale moves — where under a constant offset only
    the first applies.

    ``ages`` evaluates on a caller-supplied grid instead of the model's own plot
    grid. The population logits are interpolated *before* the quadrature (they are
    smooth in age; the derived SD need not be), which for a 0.5-month comparison
    grid also keeps the tensor-quadrature cost an order of magnitude down.

    ``draws`` selects posterior draws (an index array from :func:`align_draws`)
    *before* the quadrature rather than after. The result is identical either way —
    draws do not interact — but on a reporting-quality trace the tensor grid is the
    expensive part, so subsetting first is worth the argument.
    """
    d = MODEL_REGISTRY[key]
    mt = d.model_type
    n = n_trials(key)
    path = trace_path(key)

    def _prepare(native: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if draws is not None:
            Y = Y[draws]
        if ages is None:
            return native, Y
        out = np.asarray(ages, dtype=float)
        return out, interp_draws(native, Y, out)

    def _scale_names(base: str, slope) -> tuple[str, ...]:
        """Trace variables carrying one outcome's between-child scale."""
        if slope is None:
            return (base,)
        return (f"{base}_0", f"{base}_1", f"{base}_rho")

    def _scale(
        scal: dict[str, np.ndarray], base: str, slope, grid: np.ndarray
    ) -> np.ndarray:
        """One outcome's between-child scale: a per-draw scalar, or a curve in age.

        A model whose child effect carries a *rate* (VG19) has no single
        between-child scale: the spread is
        ``sqrt(tau0^2 + 2 rho01 tau0 tau1 D + tau1^2 D^2)`` at ``D`` years from
        the reference age. Reading its `tau_subj_*` Deterministic and stopping
        there would report the reference-age spread at every age, discarding
        `tau1` and `rho01` — the same defect #224 found in the subject-marginal
        predictive, where a fitted parameter was thrown away by the derived
        quantity that existed to use it. `tau_subj_*` is *present* in a VG19
        trace, so nothing would fail; the curve would just be silently flat.

        Keyed off the definition's scale field rather than off which variables
        the trace happens to contain, so a model that should carry a rate and
        does not fails loudly in :func:`_load_reshaped_draws`.

        Evaluated on ``grid`` — the caller's age grid once resolved, not the
        model's native one — because the scale is a function of age and must be
        computed at the ages actually reported, never interpolated from another
        grid.
        """
        if slope is None:
            tau = scal[base]
            return tau if draws is None else tau[draws]
        t0, t1, r = (scal[f"{base}_0"], scal[f"{base}_1"], scal[f"{base}_rho"])
        if draws is not None:
            t0, t1, r = t0[draws], t1[draws], r[draws]
        # Resolved exactly as `common_bivariate_re.build_model_re` resolves it,
        # default included: a reference age here that differs from the one the
        # fit used would silently shift the whole curve.
        ref = getattr(d, "subject_slope_ref_age_months", None)
        ref = float(DEFAULT_SLOPE_REF_AGE_MONTHS if ref is None else ref)
        return child_scale_of_age(t0, t1, r, grid, ref_age_months=ref)

    if mt is ModelType.UNIVARIATE:
        if d.outcome.value != outcome:
            raise ValueError(
                f"{key} is a '{d.outcome.value}' model; cannot serve '{outcome}'."
            )
        if not getattr(d, "use_subject_re", False):
            raise ValueError(
                f"{key} carries no subject random effect; it has no between-child "
                "scale to report. Use a model with use_subject_re=True."
            )
        # No univariate model carries a child rate; if one is ever added, this
        # branch must grow the same treatment rather than silently reporting the
        # reference-age spread at every age.
        if subject_slope_spec(getattr(d, "tau_subject_sigma", None)) is not None:
            raise NotImplementedError(
                f"{key} carries a child slope on a univariate engine; "
                "subject_heterogeneity has no age-varying path for it."
            )
        native, (f,), scal = _load_reshaped_draws(path, ("f_plot",), ("tau_subject",))
        grid, f = _prepare(native, f)
        tau = _scale(scal, "tau_subject", None, grid)
        tau_logit, sd_words = child_spread_single(f, tau, n, n_nodes=n_nodes)
        return grid, tau_logit, sd_words, n

    if mt is ModelType.BIVARIATE:
        if outcome == "understood":
            if not getattr(d, "use_subject_re_u", False):
                raise ValueError(
                    f"{key} carries no understood subject random effect "
                    "(use_subject_re_u=False)."
                )
            slope_u = subject_slope_spec(d.tau_subj_u_sigma)
            native, (f_u,), scal = _load_reshaped_draws(
                path, ("f_u_plot",), _scale_names("tau_subj_u", slope_u)
            )
            grid, f_u = _prepare(native, f_u)
            tau_u = _scale(scal, "tau_subj_u", slope_u, grid)
            tau_logit, sd_words = child_spread_single(f_u, tau_u, n, n_nodes=n_nodes)
            return grid, tau_logit, sd_words, n
        if outcome == "spoken":
            if not (
                getattr(d, "use_subject_re_u", False)
                and getattr(d, "use_subject_re_q", False)
            ):
                raise ValueError(
                    f"{key}: the spoken between-child scale is induced by the "
                    "understood *and* ratio subject effects; both use_subject_re_u "
                    "and use_subject_re_q must be set."
                )
            # A model that estimates the correlation between the two child
            # deviations must have it carried into the derived spoken scale;
            # otherwise the parameter is fitted and then thrown away here, which
            # is exactly the defect #224 found in the subject-marginal
            # predictive. Keyed off the definition field rather than off the
            # variable's presence in the trace, so a model that should carry a
            # correlation and does not fails loudly in _load_reshaped_draws.
            slope_u = subject_slope_spec(d.tau_subj_u_sigma)
            slope_q = subject_slope_spec(d.tau_subj_q_sigma)
            scalar_names = _scale_names("tau_subj_u", slope_u) + _scale_names(
                "tau_subj_q", slope_q
            )
            correlated = getattr(d, "subject_re_correlation_eta", None) is not None
            if correlated:
                scalar_names += ("rho_uq",)
            native, (f_u, h), scal = _load_reshaped_draws(
                path, ("f_u_plot", "h_plot"), scalar_names
            )
            grid, f_u = _prepare(native, f_u)
            _, h = _prepare(native, h)
            tau_u = _scale(scal, "tau_subj_u", slope_u, grid)
            tau_q = _scale(scal, "tau_subj_q", slope_q, grid)
            rho = None
            if correlated:
                rho = scal["rho_uq"]
                if draws is not None:
                    rho = rho[draws]
            tau_logit, sd_words = child_spread_product(
                f_u, h, tau_u, tau_q, n, rho=rho, n_nodes=n_nodes
            )
            return grid, tau_logit, sd_words, n
        raise ValueError(f"outcome must be 'spoken' or 'understood', got {outcome!r}.")

    raise ValueError(
        f"{key}: model_type {mt} is not supported by subject_heterogeneity."
    )


def subject_effect_correlation(
    key: str,
    *,
    names: tuple[str, str] = ("delta_subj_u", "delta_subj_q"),
    thin: int = 20,
) -> tuple[np.ndarray, int]:
    """Per-draw correlation *across children* between two subject random effects.

    The joint DS models give each child two deviations — one on comprehension
    (``delta_subj_u``) and one on the production ratio (``delta_subj_q``). In
    VG05–VG16 they are drawn as two *independent* standard Normal vectors, and
    :func:`child_spread_product` derives the spoken between-child scale on
    exactly that assumption; the univariate TD comparator places a single
    intercept on the spoken logit and so carries no such constraint, which made
    the assumption a live asymmetry in the DS-vs-TD ``tau`` contrast rather than
    an internal detail. VG20 estimates the correlation as a free parameter
    (``rho_uq``), which is the fix rather than the measurement.

    This function keeps both roles. On a model that does not estimate the
    correlation it is the only check available without a refit; on one that does,
    it is an *internal consistency check* — the realised deviations should
    reproduce the fitted ``rho_uq``, and on VG20 they do (0.371 against 0.369).

    Returns ``(correlations, n_children)``: one correlation per retained draw —
    computed across children within the draw — so the result carries posterior
    uncertainty, unlike a single correlation of the posterior *means*, which is
    inflated by the shrinkage the two effects share.

    Two cautions, and they apply only to the uncorrelated models. The estimate is
    shrunk toward zero by the independence prior, so its magnitude is a lower
    bound — VG10's realised +0.151 against VG20's fitted +0.369 measures how much
    that prior suppresses. And because ``log p_S = log p_U + log q``, a positive
    correlation means the independent-draw derivation **understates** the spoken
    between-child spread; a negative one means it overstates it.

    ``thin`` keeps every ``thin``-th draw: the correlation is over hundreds of
    children per draw, so a few thousand draws already resolve the interval.
    """
    d = az.from_netcdf(trace_path(key))
    post = _dataset(d, "posterior")
    for name in names:
        if name not in post:
            raise ValueError(
                f"{key}: {name!r} is not in the posterior. This check needs both "
                "per-child deviation vectors; a trace saved under a reduced "
                "persistence tier may not carry them."
            )
    a, b = (
        np.asarray(post[n].values, dtype=float).reshape(-1, post[n].values.shape[-1])[
            ::thin
        ]
        for n in names
    )
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    correlations = np.divide(
        (a * b).sum(axis=1),
        denominator,
        out=np.full(a.shape[0], np.nan),
        where=denominator > 0,
    )
    return correlations, int(a.shape[1])


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
        out[i] = np.interp(grid, ages, Y[i], left=np.nan, right=np.nan)
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
    """Median + inner-50%/outer-89% equal-tailed interval per grid column, NaN-aware.

    Contrasts and ratios are summarised with equal-tailed intervals (the project
    default). Adds ``coverage`` (fraction of non-NaN draws) and, when
    ``with_p_gt0``, the posterior probability ``P(contrast > 0)``.
    """
    outer, inner = intervals.DEFAULT_CI_PROB, intervals.INNER_CI_PROB
    rows = []
    n_draw = samples.shape[0]
    for i, g in enumerate(grid):
        col = samples[:, i]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        row: dict[str, float] = {grid_name: float(g), "coverage": n_valid / n_draw}
        if n_valid == 0:
            row.update(
                median=np.nan, ci50_lo=np.nan, ci50_hi=np.nan,
                ci_lo=np.nan, ci_hi=np.nan,
            )
            if with_p_gt0:
                row["p_gt0"] = np.nan
        else:
            c = col[valid]
            l50, u50 = intervals.interval_1d(c, inner, "eti")
            l89, u89 = intervals.interval_1d(c, outer, "eti")
            row.update(
                median=float(np.median(c)), ci50_lo=l50, ci50_hi=u50,
                ci_lo=l89, ci_hi=u89,
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


def _first_crossing_age_targets(
    W: np.ndarray,
    ages: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Per-draw first crossing age for one target per draw.

    ``np.interp(target, W, ages)`` only works when each trajectory is monotone.
    The fitted GP trajectories are not constrained that way, so this mirrors
    :func:`first_crossing_age` but lets the target vary by draw.
    """
    targets = np.asarray(targets, dtype=float)
    if targets.shape != (W.shape[0],):
        raise ValueError("targets must have shape (n_draw,).")

    out = np.full(W.shape[0], np.nan)
    valid = np.isfinite(targets)
    if not valid.any():
        return out

    W_valid = W[valid]
    targets_valid = targets[valid]
    mask = W_valid >= targets_valid[:, None]
    any_above = mask.any(axis=1)
    first_idx = mask.argmax(axis=1)
    j_prev = np.maximum(first_idx - 1, 0)

    y0 = np.take_along_axis(W_valid, j_prev[:, None], axis=1).squeeze(1)
    y1 = np.take_along_axis(W_valid, first_idx[:, None], axis=1).squeeze(1)
    a0 = ages[j_prev]
    a1 = ages[first_idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = y1 - y0
        interp = np.where(
            denom == 0,
            a1,
            a0 + (targets_valid - y0) * (a1 - a0) / denom,
        )
    crossing = np.where(first_idx == 0, ages[0], interp)
    below_support = (first_idx == 0) & (W_valid[:, 0] > targets_valid)
    crossing = np.where(below_support, np.nan, crossing)
    out[valid] = np.where(any_above, crossing, np.nan)
    return out


def _invert_trajectory(
    ages: np.ndarray, W: np.ndarray, targets: np.ndarray
) -> np.ndarray:
    """Per-draw first age at which ``W`` (n_draw, n_age) reaches ``targets``.

    ``targets`` is ``(n_draw, n_target)``. Linear interpolation on each draw's
    first crossing; NaN where a target lies outside that draw's observed level
    range — i.e. where matching would require extrapolating the reference.
    """
    n_draw, n_target = targets.shape
    out = np.full((n_draw, n_target), np.nan)
    for j in range(n_target):
        out[:, j] = _first_crossing_age_targets(W, ages, targets[:, j])
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
    ages, (p_any,), _ = _load_reshaped_draws(path, ("p_any_plot",))
    return ages, p_any * n_trials_


# ==========================================================================
# Reporting helpers for the matched-comprehension contrast
# ==========================================================================
#
# The findings chapter used to state this contrast's credible window, peak and
# direction as hand-typed prose, which outlived the fits that produced it: the
# quoted window and peak came from a superseded denominator and likelihood, and
# by the current fits the *sign of the trend* had changed too. Deriving those
# three facts here — from the same written table the chapter tabulates, filtered
# the same way — means the chapter cannot restate a superseded fit, and the
# derivation is unit-testable rather than living in a ``.qmd``.


def dq_contrast_facts(
    table: pd.DataFrame | None,
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    grid_col: str = "words",
    prefix: str = "dq",
) -> dict | None:
    """Summarise a matched-comprehension difference table for prose.

    ``table`` is a ``summarise_draws``-shaped frame for a difference (as written
    to ``ds_td_comprehension_q_at_U.csv``), carrying ``<prefix>_median``,
    ``<prefix>_ci_lo``/``_hi`` and optionally ``<prefix>_coverage``. Grid points
    whose coverage falls below ``min_coverage`` are dropped, because their
    summaries are conditional on the subset of draws that attain the level.

    Returns ``None`` when the table is absent or has no usable rows, so a report
    can degrade gracefully before the comparison has been run. Otherwise a dict
    with:

    ``table``
        The coverage-filtered, grid-sorted frame actually summarised.
    ``covered``
        ``(lo, hi)`` grid range retained after filtering.
    ``positive`` / ``negative``
        ``(lo, hi)`` grid sub-range where the interval excludes zero in that
        direction, or ``None``. These are the *extent* of credible points, not a
        guarantee that every point between them is credible.
    ``peak``
        The largest-magnitude row (a ``Series``), signed — not the largest
        positive row, so a contrast that is credibly negative reports honestly.
    ``rises``
        Whether the difference increases with the grid variable, by the sign of
        the Spearman correlation. ``None`` when fewer than three points remain,
        where a monotone direction is not meaningful.
    """
    if table is None:
        return None
    median, lo, hi = f"{prefix}_median", f"{prefix}_ci_lo", f"{prefix}_ci_hi"
    needed = {grid_col, median, lo, hi}
    if not needed.issubset(table.columns):
        return None

    rows = table.dropna(subset=[median, lo, hi])
    coverage = f"{prefix}_coverage"
    if coverage in rows.columns:
        rows = rows[rows[coverage] >= min_coverage]
    rows = rows.sort_values(grid_col)
    if rows.empty:
        return None

    positive = rows[rows[lo] > 0]
    negative = rows[rows[hi] < 0]

    def _extent(subset: pd.DataFrame) -> tuple[float, float] | None:
        if subset.empty:
            return None
        return float(subset[grid_col].min()), float(subset[grid_col].max())

    rises = None
    if len(rows) >= 3:
        correlation = rows[grid_col].corr(rows[median], method="spearman")
        if pd.notna(correlation):
            rises = bool(correlation > 0)

    return {
        "table": rows,
        "covered": (float(rows[grid_col].min()), float(rows[grid_col].max())),
        "positive": _extent(positive),
        "negative": _extent(negative),
        "peak": rows.loc[rows[median].abs().idxmax()],
        "rises": rises,
        "min_coverage": float(min_coverage),
    }


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
