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
from dataclasses import dataclass

import arviz as az
import dse_research_utils.plot.io as plot_io
import dse_research_utils.statistics.intervals as shared_intervals
import numpy as np
import pandas as pd

from vocab_growth import environment as env
from vocab_growth import intervals, reporting_ages
from vocab_growth.models import subject_effects
from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    ModelType,
    subject_slope_spec,
)
from vocab_growth.models.subject_effects import SubjectEffectKind, slope_reference_age

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
    ages_sorted = ages[order]
    weights = _study_weights(ages_sorted, frame, bandwidth)
    K = weights.shape[1]

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


def _study_weights(ages_sorted: np.ndarray, frame, bandwidth: float) -> np.ndarray:
    """``(n_age, K)`` study weights at each plot age: a Gaussian kernel over
    ``frame``'s administrations, normalised across studies at each age.

    Shared by the joint and single-outcome weighted loaders so the
    administration-weighted child means the same thing on both sides of a
    contrast. A study with no administration near an age gets zero weight
    there; an age with no administration anywhere keeps zero weights rather
    than dividing by zero.
    """
    codes = np.asarray(frame["study_code"], dtype=int)
    obs_ages = np.asarray(frame["age"], dtype=float)
    kernel = np.exp(-0.5 * ((ages_sorted[:, None] - obs_ages[None, :]) / bandwidth) ** 2)
    K = int(codes.max()) + 1
    weights = np.stack([kernel[:, codes == k].sum(axis=1) for k in range(K)], axis=1)
    weights /= np.where(weights.sum(axis=1, keepdims=True) > 0, weights.sum(axis=1, keepdims=True), 1.0)
    return weights


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


def load_univariate_trajectory_weighted(
    path: str, n_trials_: int, frame, *, bandwidth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """``(ages, W)`` for the administration-weighted child of a single-outcome RE model.

    The single-outcome analogue of :func:`load_population_trajectory_weighted`,
    for the typically developing comparators VG11 and VG12 (``f_plot`` plus the
    dataset-level study offsets ``delta``). Until #289 task 4.5 those had no
    weighted loader, so the weighted attainment delay was read against the
    joint comparator VG21 and stopped at its window's edge, about 100 spoken
    words, where the reference-child delay against VG11 runs on. Both delays
    now share a comparator. Same kernel, same normalisation, same frame guard
    as the joint loader; ``frame`` must carry ``study_code`` and ``age`` and
    should be the fit's own verified frame.
    """
    d = az.from_netcdf(path)
    post = _dataset(d, "posterior")
    cdata = _dataset(d, "constant_data")
    ages = np.asarray(cdata["X_plot"].values, dtype=float)
    order = np.argsort(ages)

    def flat(name):
        arr = post[name].values
        return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])

    f = flat("f_plot")[:, order]  # (S, n_age)
    d_s = flat("delta")  # (S, K)
    ages_sorted = ages[order]
    weights = _study_weights(ages_sorted, frame, bandwidth)
    if weights.shape[1] > d_s.shape[1]:
        raise ValueError(
            f"{os.path.basename(os.path.dirname(path))}: the frame codes "
            f"{weights.shape[1]} studies but the posterior carries offsets for "
            f"{d_s.shape[1]}; this is not the fit's own frame."
        )

    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    W = np.zeros_like(f)
    for k in range(weights.shape[1]):
        w = weights[:, k][None, :]
        if not np.any(w):
            continue
        W += w * sig(f + d_s[:, k][:, None])
    return ages_sorted, W * n_trials_


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
    ages, p, k = restrict_reporting_trajectory(key, outcome, ages, p, k)
    return ages, p, k, n_trials(key)


def restrict_reporting_trajectory(key, outcome, ages, *arrays):
    """Trim before inversion or differentiation, including an interpolated cap."""
    cap = reporting_ages.max_age_for(MODEL_REGISTRY[key], reporting_ages.quantity_for_outcome(outcome))
    ages = np.asarray(ages)
    if cap is None or cap >= ages[-1]:
        return (ages, *arrays)
    if cap < ages[0]:
        raise ValueError(f"{key} has no supported {outcome} grid below its reporting cap.")
    grid = np.unique(np.r_[ages[ages <= cap], cap])
    return (grid, *(interp_draws(ages, array, grid) for array in arrays))


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
    square root of a quadratic in age, with its minimum at
    ``D = -rho01 tau0 / tau1`` when ``tau1 > 0``: a negative
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
    p1 = np.zeros_like(f_u)
    p2 = np.zeros_like(f_u)
    l1 = np.zeros_like(f_u)
    l2 = np.zeros_like(f_u)
    for weight, a, b in _product_nodes(f_u, h, tau_u, tau_q, rho=rho, n_nodes=n_nodes):
        p = _sigmoid(a) * _sigmoid(b)
        lg = _logit_sigmoid_product(a, b)
        p1 += weight * p
        p2 += weight * p * p
        l1 += weight * lg
        l2 += weight * lg * lg
    tau_logit = np.sqrt(np.maximum(l2 - l1 * l1, 0.0))
    sd_words = n * np.sqrt(np.maximum(p2 - p1 * p1, 0.0))
    return tau_logit, sd_words


def _product_nodes(
    f_u: np.ndarray,
    h: np.ndarray,
    tau_u: np.ndarray,
    tau_q: np.ndarray,
    *,
    rho: np.ndarray | None,
    n_nodes: int,
):
    """Tensor Gauss-Hermite nodes over a child's two deviations in a ``p_u * q`` model.

    Validates eagerly and returns an iterator of ``(weight, a, b)``, where ``a``
    is the child's understood logit and ``b`` their production-ratio logit at one
    node pair, each ``(n_draw, n_age)``. Shared by :func:`child_spread_product`
    and :func:`total_spread_product` so that both integrate over the same
    children, correlation included.
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

    def nodes():
        for xi, wi in zip(x, w, strict=True):
            a = f_u + tu * xi
            for xj, wj in zip(x, w, strict=True):
                b = h + (tq * xj if r is None else tq * (r * xi + s * xj))
                yield wi * wj, a, b

    return nodes()


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
        ref = slope_reference_age(d)
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


# ----------------------------------------------------------------------------
# Total spread: how far apart children's counts sit, without splitting it
# ----------------------------------------------------------------------------
# The between-child contrast adopted for publication (#229 option 4, adopted
# 2026-09-14; #289 task 4.12). The functions above split a population's scatter
# into a persistent child scale and observation-level dispersion; in the typically
# developing models that split is identified by the Beta-Binomial's functional form
# rather than by repeat visits, so it is not reported as a cross-population
# contrast. What is reported instead is the spread of the counts themselves: the
# SD, in words, of one administration of one new child at a given age.
#
# The new child is the one every engine's own ``y_query`` draws: a fresh child
# effect, zero study effect (the average study) and sex contrast zero, so the
# population curves, ``subject_heterogeneity`` and the new-child predictive all
# describe the same child. The variance is exact, by the law of total variance
# over the child effect, with the Beta-Binomial's conditional variance in closed
# form:
#
#     Var(Y) = n^2 Var_child(E[theta]) + E_child[n m (1 - m) + n (n - 1) Var(theta)]
#
# where ``theta`` is the administration's Beta-distributed proportion and ``m``
# its mean. The nested spoken count needs no marginal approximation: binomial
# thinning makes ``S | theta_U, theta_S ~ Binomial(n, theta_U * theta_S)`` exactly,
# so only the product's first two moments are needed, and those are elementary.
#
# It is in words and on no transformed scale, by decision of 2026-09-14. On the
# 2026-09-08 fits the logit-scale versions disagreed about the direction of the
# Down syndrome / typically developing spoken contrast at the floor: the two that
# convert dispersion to logits were dominated by the tiny means there, and the
# exact SD of the observed log-odds rests on the continuity correction there. The
# mean dependence a transform was meant to remove is removed instead by comparing
# the two populations at the same vocabulary level (:func:`value_at_level`) as
# well as at the same age. See notes/202609141600-total-spread-estimand.md.


def _beta_second_moment(p: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """``E[theta^2]`` for ``theta ~ Beta(p kappa, (1 - p) kappa)``."""
    return p * (p * kappa + 1.0) / (kappa + 1.0)


def _binomial_mixture_variance(m: np.ndarray, second_moment: np.ndarray, n: int) -> np.ndarray:
    """``Var(Y)`` for ``Y | theta ~ Binomial(n, theta)`` given ``theta``'s first two moments."""
    return n * m * (1.0 - m) + n * (n - 1.0) * (second_moment - m * m)


def total_spread_single(
    f: np.ndarray,
    tau: np.ndarray | None,
    kappa: np.ndarray,
    n: int,
    *,
    n_nodes: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of a new child's count when one Normal intercept sits on the logit.

    ``f`` is the reference child's logit ``(n_draw, n_age)``, ``tau`` the child
    scale — ``(n_draw,)``, ``(n_draw, n_age)`` for an age-varying one, or ``None``
    for a model with no child effect — and ``kappa`` the concentration on the
    same grid. A child's count is ``BetaBinomial(n, p kappa, (1 - p) kappa)``
    with ``p = sigmoid(f + tau Z)``. Returns ``(mean_words, sd_words)``, both
    ``(n_draw, n_age)``; ``sd_words`` carries the child effect and the
    administration noise together. At ``tau = 0`` it is :func:`implied_sd_y`.
    """
    x, w = _gauss_hermite_standard_normal(n_nodes)
    kappa = np.asarray(kappa, dtype=float)
    tau_col = (
        np.zeros((f.shape[0], 1)) if tau is None else _tau_to_draw_age(tau, f.shape, what="tau")
    )
    m1 = np.zeros_like(f)
    m2 = np.zeros_like(f)
    within = np.zeros_like(f)
    for xi, wi in zip(x, w, strict=True):
        p = _sigmoid(f + tau_col * xi)
        m1 += wi * p
        m2 += wi * p * p
        within += wi * _binomial_mixture_variance(p, _beta_second_moment(p, kappa), n)
    variance = n * n * (m2 - m1 * m1) + within
    return n * m1, np.sqrt(np.maximum(variance, 0.0))


def total_spread_product(
    f_u: np.ndarray,
    h: np.ndarray,
    tau_u: np.ndarray | None,
    tau_q: np.ndarray | None,
    kappa_u: np.ndarray,
    kappa_s: np.ndarray,
    n: int,
    *,
    rho: np.ndarray | None = None,
    n_nodes: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SD of a new child's *spoken* count in a nested ``p_u * q`` model.

    The engines draw ``U ~ BetaBinomial(n, p_u kappa_u, ...)`` and then
    ``S | U ~ BetaBinomial(U, q kappa_s, ...)``, with the child's two deviations
    on the logits of ``p_u`` and ``q`` (correlated through ``rho`` where the model
    estimates it). Binomial thinning gives ``S | theta_U, theta_S ~ Binomial(n,
    theta_U theta_S)`` with independent Betas, so the count's conditional variance
    follows exactly from the product's first two moments, without the
    moment-matched concentration :func:`product_marginal_kappa` builds for the
    dispersion contrast. Arguments as :func:`child_spread_product`, plus the two
    concentrations on the same grid; a ``None`` scale means no child effect on
    that logit. Returns ``(mean_words, sd_words)``.
    """
    kappa_u = np.asarray(kappa_u, dtype=float)
    kappa_s = np.asarray(kappa_s, dtype=float)
    no_effect = np.zeros(f_u.shape[0])
    m1 = np.zeros_like(f_u)
    m2 = np.zeros_like(f_u)
    within = np.zeros_like(f_u)
    nodes = _product_nodes(
        f_u,
        h,
        no_effect if tau_u is None else tau_u,
        no_effect if tau_q is None else tau_q,
        rho=rho,
        n_nodes=n_nodes,
    )
    for weight, a, b in nodes:
        p_u = _sigmoid(a)
        q = _sigmoid(b)
        m = p_u * q
        second = _beta_second_moment(p_u, kappa_u) * _beta_second_moment(q, kappa_s)
        m1 += weight * m
        m2 += weight * m * m
        within += weight * _binomial_mixture_variance(m, second, n)
    variance = n * n * (m2 - m1 * m1) + within
    return n * m1, np.sqrt(np.maximum(variance, 0.0))


@dataclass(frozen=True)
class ChildScaleSource:
    """Where one logit's child scale lives in a trace, and what shape it takes."""

    kind: SubjectEffectKind
    names: tuple[str, ...]
    """One scalar (constant or partitioned scale), one grid variable (A1), or
    ``tau0``, ``tau1`` and their correlation (a child slope)."""


@dataclass(frozen=True)
class TotalSpreadPlan:
    """The trace variables one outcome's total spread is computed from.

    Resolved from the definition, never from which variables a trace happens to
    contain, for the reason :func:`subject_heterogeneity` gives: a child-slope
    model also emits ``tau_subj_u``, so reading whatever is present would
    silently report a flat scale. Variables are named on one grid (``plot`` or
    ``query``). Probabilities are read rather than logits because a recovery
    truth draw carries the probability-scale deterministics and not the logits.
    """

    outcome: str
    grid: str
    nested: bool
    """Spoken on a ``p_u * q`` model: two logits, two concentrations."""
    p_name: str
    kappa_name: str
    scale: ChildScaleSource | None
    q_name: str | None = None
    kappa_s_name: str | None = None
    scale_q: ChildScaleSource | None = None
    rho_name: str | None = None
    slope_ref_age_months: float = 36.0

    def _sources(self) -> tuple[ChildScaleSource, ...]:
        return tuple(s for s in (self.scale, self.scale_q) if s is not None)

    @property
    def grid_variables(self) -> tuple[str, ...]:
        names = [self.p_name, self.kappa_name]
        if self.nested:
            names += [self.q_name, self.kappa_s_name]
        names += [
            s.names[0] for s in self._sources() if s.kind is SubjectEffectKind.AGE_VARYING
        ]
        return tuple(names)

    @property
    def scalar_variables(self) -> tuple[str, ...]:
        names = [
            name
            for s in self._sources()
            if s.kind is not SubjectEffectKind.AGE_VARYING
            for name in s.names
        ]
        if self.rho_name is not None:
            names.append(self.rho_name)
        return tuple(names)


def _child_scale_source(effect, grid: str, where: str) -> ChildScaleSource | None:
    kind = effect.kind
    if kind is SubjectEffectKind.NONE:
        return None
    if kind in (SubjectEffectKind.CONSTANT, SubjectEffectKind.VARIANCE_PARTITION):
        return ChildScaleSource(kind, (effect.scale_name,))
    if kind is SubjectEffectKind.AGE_VARYING:
        return ChildScaleSource(kind, (f"{effect.scale_name}_{grid}",))
    if kind is SubjectEffectKind.CHILD_SLOPE:
        base = effect.scale_name
        return ChildScaleSource(kind, (f"{base}_0", f"{base}_1", f"{base}_rho"))
    raise NotImplementedError(
        f"{where}: total spread is not derived for a {kind.value} child effect. "
        "A low-rank factor (VG22) gives each child a level and a rate on both "
        "outcomes, which this quadrature does not integrate over."
    )


def total_spread_plan(definition, outcome: str, grid: str = "plot") -> TotalSpreadPlan:
    """Resolve which trace variables ``outcome``'s total spread reads.

    Supports the single-outcome random-effect models (VG11, VG12) and the
    bivariate ones (VG07-VG10, VG13, VG16, VG19-VG21, VG23, VG26), with constant,
    partitioned, age-varying (A1) and child-slope scales and VG20's correlation.
    Refuses the joint engine and the low-rank factor rather than returning a
    number for a structure it does not integrate over.
    """
    if grid not in ("plot", "query"):
        raise ValueError(f"grid must be 'plot' or 'query', got {grid!r}.")
    where = getattr(definition, "model_id", "definition")
    mt = definition.model_type
    plan = subject_effects.resolve(definition)
    if mt is ModelType.UNIVARIATE:
        if definition.outcome.value != outcome:
            raise ValueError(
                f"{where} is a '{definition.outcome.value}' model; cannot serve '{outcome}'."
            )
        effect = plan[subject_effects.UNIVARIATE_OUTCOME] if plan.effects else None
        return TotalSpreadPlan(
            outcome=outcome,
            grid=grid,
            nested=False,
            p_name=f"p_{grid}",
            kappa_name=f"kappa_{grid}",
            scale=None if effect is None else _child_scale_source(effect, grid, where),
            slope_ref_age_months=plan.slope_ref_age_months,
        )
    if mt is ModelType.BIVARIATE:
        scale_u = _child_scale_source(plan["u"], grid, where)
        if outcome == "understood":
            return TotalSpreadPlan(
                outcome=outcome,
                grid=grid,
                nested=False,
                p_name=f"p_u_{grid}",
                kappa_name=f"kappa_u_{grid}",
                scale=scale_u,
                slope_ref_age_months=plan.slope_ref_age_months,
            )
        if outcome == "spoken":
            return TotalSpreadPlan(
                outcome=outcome,
                grid=grid,
                nested=True,
                p_name=f"p_u_{grid}",
                kappa_name=f"kappa_u_{grid}",
                scale=scale_u,
                q_name=f"q_{grid}",
                kappa_s_name=f"kappa_s_{grid}",
                scale_q=_child_scale_source(plan["q"], grid, where),
                rho_name="rho_uq" if plan.correlation_eta is not None else None,
                slope_ref_age_months=plan.slope_ref_age_months,
            )
        raise ValueError(f"outcome must be 'spoken' or 'understood', got {outcome!r}.")
    raise ValueError(f"{where}: model_type {mt} is not supported by total_spread_plan.")


@dataclass(frozen=True)
class TotalSpread:
    """One outcome's total spread on an age grid, per posterior draw."""

    ages: np.ndarray
    """``(n_age,)``, months."""
    mean_words: np.ndarray
    """``(n_draw, n_age)``: the new child's expected count."""
    sd_words: np.ndarray
    """``(n_draw, n_age)``: SD of the new child's count, child effect and
    administration noise together."""
    reference_words: np.ndarray
    """``(n_draw, n_age)``: the reference child's count, the population curve
    every other contrast here reads, used to match populations on level."""
    n_trials: int


def _logit_of(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return np.log(p) - np.log1p(-p)


def _child_scale_values(
    source: ChildScaleSource | None,
    values: dict[str, np.ndarray],
    ages: np.ndarray,
    ref_age_months: float,
) -> np.ndarray | None:
    if source is None:
        return None
    if source.kind is SubjectEffectKind.CHILD_SLOPE:
        t0, t1, r = (values[name] for name in source.names)
        return child_scale_of_age(t0, t1, r, ages, ref_age_months=ref_age_months)
    return values[source.names[0]]


def total_spread_from_values(
    plan: TotalSpreadPlan,
    values: dict[str, np.ndarray],
    ages: np.ndarray,
    n_trials_: int,
    *,
    n_nodes: int = 21,
) -> TotalSpread:
    """Compute :class:`TotalSpread` from arrays keyed by the plan's variable names.

    Grid variables are ``(n_draw, n_age)`` on ``ages``; scalars are ``(n_draw,)``.
    Shared by :func:`total_spread`, which reads a fitted trace's plot grid, and
    by parameter-recovery scoring, which reads the query grid of a posterior and
    of a truth draw, so the contrast and its recovery check are one computation.
    """
    missing = [
        name for name in plan.grid_variables + plan.scalar_variables if name not in values
    ]
    if missing:
        raise KeyError(
            f"total spread for {plan.outcome!r} needs {', '.join(missing)}, "
            "which the supplied values do not carry."
        )
    ages = np.asarray(ages, dtype=float)
    ref = plan.slope_ref_age_months
    p = np.asarray(values[plan.p_name], dtype=float)
    kappa = values[plan.kappa_name]
    tau = _child_scale_values(plan.scale, values, ages, ref)
    if not plan.nested:
        mean, sd = total_spread_single(_logit_of(p), tau, kappa, n_trials_, n_nodes=n_nodes)
        return TotalSpread(ages, mean, sd, n_trials_ * p, n_trials_)
    q = np.asarray(values[plan.q_name], dtype=float)
    mean, sd = total_spread_product(
        _logit_of(p),
        _logit_of(q),
        tau,
        _child_scale_values(plan.scale_q, values, ages, ref),
        kappa,
        values[plan.kappa_s_name],
        n_trials_,
        rho=None if plan.rho_name is None else values[plan.rho_name],
        n_nodes=n_nodes,
    )
    return TotalSpread(ages, mean, sd, n_trials_ * p * q, n_trials_)


def total_spread(
    key: str,
    outcome: str = "spoken",
    *,
    ages: np.ndarray | None = None,
    draws: np.ndarray | None = None,
    n_nodes: int = 21,
) -> TotalSpread:
    """Total spread of ``outcome`` for registry model ``key``, from its fitted trace.

    The counterpart of :func:`subject_heterogeneity` for the contrast adopted for
    publication.
    ``ages`` evaluates on a caller-supplied grid, interpolating the plot-grid
    curves first (a child-slope scale is instead built on that grid directly);
    ``draws`` selects posterior draws before the quadrature, as there.
    """
    d = MODEL_REGISTRY[key]
    plan = total_spread_plan(d, outcome, "plot")
    native, arrays, scalars = _load_reshaped_draws(
        trace_path(key), plan.grid_variables, plan.scalar_variables
    )
    grid = native if ages is None else np.asarray(ages, dtype=float)
    values: dict[str, np.ndarray] = {}
    for name, array in zip(plan.grid_variables, arrays, strict=True):
        if draws is not None:
            array = array[draws]
        values[name] = array if ages is None else interp_draws(native, array, grid)
    for name, array in scalars.items():
        values[name] = array if draws is None else array[draws]
    return total_spread_from_values(plan, values, grid, n_trials(key), n_nodes=n_nodes)


def plot_ages(key: str) -> np.ndarray:
    """The ascending plot-grid ages (months) of ``key``'s fitted trace."""
    d = az.from_netcdf(trace_path(key))
    return np.sort(np.asarray(_dataset(d, "constant_data")["X_plot"].values, dtype=float))


def value_at_level(
    level_curve: np.ndarray,
    value: np.ndarray,
    ages: np.ndarray,
    levels: np.ndarray,
) -> np.ndarray:
    """Per draw, ``value`` read at the age ``level_curve`` first reaches each level.

    Both arrays are ``(n_draw, n_age)`` on ``ages``. The age comes from
    :func:`first_crossing_age`, so a level never reached, or already passed at
    the youngest age, gives NaN rather than an edge value. Returns
    ``(n_draw, n_level)``. This is how the two populations are compared at the
    same vocabulary level instead of the same age.
    """
    ages = np.asarray(ages, dtype=float)
    value = np.asarray(value, dtype=float)
    rows = np.arange(value.shape[0])
    out = np.full((value.shape[0], len(levels)), np.nan)
    for k, level in enumerate(levels):
        at = first_crossing_age(level_curve, ages, float(level))
        ok = np.isfinite(at)
        if not ok.any():
            continue
        hi = np.clip(np.searchsorted(ages, at[ok], side="right"), 1, ages.size - 1)
        lo = hi - 1
        span = ages[hi] - ages[lo]
        t = np.where(span > 0, (at[ok] - ages[lo]) / np.where(span > 0, span, 1.0), 0.0)
        t = np.clip(t, 0.0, 1.0)
        r = rows[ok]
        out[ok, k] = value[r, lo] * (1.0 - t) + value[r, hi] * t
    return out


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

    Returns ``(correlations, n_children)``: the empirical correlation across
    fitted child effects within each retained posterior draw. This includes
    uncertainty in those effects, but is not the population correlation parameter.
    A finite fitted sample need not reproduce that parameter exactly. Shrinkage,
    differing observation patterns and selection can move empirical correlations
    in either direction. They are not lower bounds on a population correlation.


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
    """Per-draw fraction of a simple Beta-Binomial distribution at/below the
    TD ``pct``-th percentile word count, on a common grid.

    This helper excludes child random effects and is not the nested speech
    distribution. Use ``new_child_percentile_fraction`` for those models.
    ``p_*``/``k_*`` are
    ``(n_draw, n_grid)`` population mean proportions and Beta-Binomial
    concentrations. Returns ``(n_draw, n_grid)``.
    """
    from scipy.stats import betabinom

    a_td, b_td = p_td * k_td, (1.0 - p_td) * k_td
    a_ds, b_ds = p_ds * k_ds, (1.0 - p_ds) * k_ds
    thresh = betabinom.ppf(pct / 100.0, n_trials_, a_td, b_td)
    return betabinom.cdf(thresh, n_trials_, a_ds, b_ds)


def predictive_count_inputs(key, outcome, grid, draws):
    """Pointwise new-child distribution at zero study effect and reference sex."""
    plan = total_spread_plan(MODEL_REGISTRY[key], outcome)
    native, arrays, scalars = _load_reshaped_draws(trace_path(key), plan.grid_variables, plan.scalar_variables)
    values = {name: interp_draws(native, array[draws], grid)
              for name, array in zip(plan.grid_variables, arrays, strict=True)}
    values.update({name: array[draws] for name, array in scalars.items()})
    return plan, values


def new_child_count_sample(plan, values, ages, n, draw, children, rng):
    """Pointwise count samples with child variation and nested observation noise.

    Samples across ages are not individual trajectories. The prediction fixes
    the study effect and sex at the values of the supplied reference curves.
    """
    shape = np.asarray(values[plan.p_name]).shape
    def scale(source):
        tau = _child_scale_values(source, values, ages, plan.slope_ref_age_months)
        return np.zeros(shape[1]) if tau is None else _tau_to_draw_age(tau, shape, what="predictive scale")[draw]
    def count(trials, p, k):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return rng.binomial(trials, rng.beta(p * k, (1 - p) * k))
    zu, zq = rng.normal(size=(2, children, len(ages)))
    p = _sigmoid(_logit_of(values[plan.p_name][draw]) + zu * scale(plan.scale))
    parent = count(n, p, values[plan.kappa_name][draw])
    if not plan.nested:
        return parent
    rho = 0.0 if plan.rho_name is None else float(values[plan.rho_name][draw])
    q = _sigmoid(_logit_of(values[plan.q_name][draw]) + scale(plan.scale_q) * (rho * zu + np.sqrt(max(0, 1 - rho**2)) * zq))
    return count(parent, q, values[plan.kappa_s_name][draw])


def new_child_percentile_fraction(ds_inputs, td_inputs, ages, n, *, pct=10.0, children=4096, seed=47):
    """Monte Carlo fractions at or below each draw's TD predictive percentile.

    Parameter uncertainty is represented by rows. Each row integrates child
    and observation variation by simulation, so it also has Monte Carlo error.
    For a fixed threshold its maximum binomial Monte Carlo SE is 0.5/sqrt(M);
    estimating the TD threshold adds uncertainty, especially with discrete ties.
    """
    if children < 2 or not 0 < pct < 100:
        raise ValueError("Need at least two children and a percentile between 0 and 100.")
    ds_plan, ds_values = ds_inputs
    td_plan, td_values = td_inputs
    nd = len(ds_values[ds_plan.p_name])
    if len(td_values[td_plan.p_name]) != nd:
        raise ValueError("Posterior draws must be paired before comparing percentiles.")
    rng = np.random.default_rng(seed)
    fractions = np.empty((nd, len(ages)))
    for draw in range(nd):
        td = new_child_count_sample(td_plan, td_values, ages, n, draw, children, rng)
        threshold = np.quantile(td, pct / 100, axis=0, method="inverted_cdf")
        ds = new_child_count_sample(ds_plan, ds_values, ages, n, draw, children, rng)
        fractions[draw] = np.mean(ds <= threshold, axis=0)
    return fractions


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
