# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from typing import Protocol

import dse_research_utils.plot.io as plot_io
import dse_research_utils.plot.predictive as plot_predictive
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

import vocab_growth.intervals as intervals


def _save_csv(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    """Save a DataFrame as CSV alongside the corresponding plot.

    Delegates to the shared writer; kept as a local name (and argument order)
    for this module's call sites.
    """
    plot_io.save_plot_data(output_dir, filename, df)


# ISO 216 A-series landscape aspect ratio (width : height = √2 : 1), used to size
# the individual per-age posterior-predictive count figures.
ISO_A_LANDSCAPE_RATIO = 2**0.5


def _iso_a_landscape_figsize(width: float = 7.5) -> tuple[float, float]:
    """Return an (width, height) tuple in ISO A-series landscape proportions."""
    return (width, width / ISO_A_LANDSCAPE_RATIO)


def _save_png_svg(
    fig: Figure, output_dir: str, filename: str, *, dpi: float = plot_styles.DPI_FILE, svg: bool = True
) -> None:
    """Save a figure as PNG (and, unless ``svg=False``, SVG) under one filename stem.

    Delegates to :func:`dse_research_utils.plot.io.save_styled_figure`.
    ``bbox_inches=None`` and ``close=False`` preserve this project's cropping
    and its figure-returning plot functions; the shared helper additionally
    creates ``output_dir`` if it is absent.
    """
    plot_io.save_styled_figure(
        output_dir,
        filename,
        fig=fig,
        dpi=dpi,
        bbox_inches=None,
        close=False,
        svg=svg,
    )


#: A child needs at least this many administrations before their observations are
#: joined into a trajectory. Two points are a segment, not a trajectory: they say
#: nothing about shape, and drawing them costs the ink that makes the children who
#: do have a shape readable.
MIN_ADMINISTRATIONS_FOR_TRAJECTORY = 3

#: An observation at or above this share of its **own** form's item count is
#: marked. Its value is compressed by the instrument rather than by the child, and
#: it is the usual explanation for an apparent plateau or reversal.
NEAR_CEILING_SHARE = 0.90

#: Neutral grey for the observed trajectories. They are data, not another modelled
#: series, and must not read as a category beside the predictive bands.
_TRAJECTORY_COLOUR = "0.30"

#: How many unseen-child trajectories to draw. Enough to read the spread, few
#: enough to see through to the bands and the observed lines underneath.
DEFAULT_PREDICTIVE_TRAJECTORIES = 60

#: The predictive trajectories take the median's own colour, thin and
#: translucent, because that is what they are: draws of the quantity the median
#: line summarises, not a separate series.
_PREDICTIVE_TRAJECTORY_COLOUR = plot_styles.COLOUR_DARK_BLUE


def _draw_subject_trajectories(
    x_obs,
    y_obs,
    subject_ids,
    form_max=None,
    min_administrations: int = MIN_ADMINISTRATIONS_FOR_TRAJECTORY,
) -> dict[str, int]:
    """Join each child's observations into a trajectory, on the current axes.

    Segments are drawn **solid within one recording form and dashed across a
    change of form**, because a change of form makes consecutive counts
    incomparable. The Down syndrome pool spans item counts from 396 to 810, and
    roughly half the children with three or more administrations are recorded on
    more than one form; a child near the ceiling of a short form can record
    *fewer* words on a longer form a month later. One real child in this pool
    scores 393 understood on a 416-item form at 47 months and 347 on an 810-item
    form at 48. Joining that with a plain line draws a developmental reversal
    that did not happen, which is the whole hazard of turning a scatter into a
    set of trajectories.

    Returns the counts the caller needs for the legend, so the figure states its
    own composition rather than relying on a caption written elsewhere.
    """
    frame = pd.DataFrame(
        {
            "subject": np.asarray(subject_ids),
            "age": np.asarray(x_obs, dtype=float),
            "count": np.asarray(y_obs, dtype=float),
        }
    )
    have_forms = form_max is not None
    frame["form"] = np.asarray(form_max, dtype=float) if have_forms else np.nan
    frame = frame.dropna(subset=["age", "count"])

    sizes = frame.groupby("subject")["age"].transform("size")
    frame = frame.loc[sizes >= min_administrations].sort_values(["subject", "age"])
    empty = {"children": 0, "form_changes": 0, "near_ceiling": 0}
    if frame.empty:
        return empty

    form_changes = 0
    for _, rows in frame.groupby("subject", sort=False):
        ages = rows["age"].to_numpy()
        counts = rows["count"].to_numpy()
        forms = rows["form"].to_numpy()
        for i in range(ages.size - 1):
            same_form = (not have_forms) or forms[i] == forms[i + 1]
            if not same_form:
                form_changes += 1
            plt.plot(
                ages[i : i + 2],
                counts[i : i + 2],
                color=_TRAJECTORY_COLOUR,
                lw=0.7,
                alpha=0.55 if same_form else 0.5,
                ls="-" if same_form else (0, (2, 2)),
                zorder=2.5,
            )

    near_ceiling = 0
    if have_forms:
        compressed = frame["count"] >= NEAR_CEILING_SHARE * frame["form"]
        near_ceiling = int(compressed.sum())
        if near_ceiling:
            plt.scatter(
                frame.loc[compressed, "age"],
                frame.loc[compressed, "count"],
                s=28,
                facecolors="none",
                edgecolors=_TRAJECTORY_COLOUR,
                linewidths=0.9,
                zorder=2.6,
            )

    return {
        "children": int(frame["subject"].nunique()),
        "form_changes": form_changes,
        "near_ceiling": near_ceiling,
    }


def _draw_predictive_trajectories(
    X_plot: np.ndarray,
    trajectory_samples: np.ndarray,
    n_trajectories: int,
    seed: int,
) -> int:
    """Draw a sample of unseen-child trajectories on the current axes.

    Each column of ``trajectory_samples`` is one posterior draw's expected curve
    for a child the model has not seen, built by the engine from a single child
    effect reused across the whole age grid. That coherence is the point: the
    predictive bands are pointwise quantiles, and a set of pointwise intervals
    does not tell you what a trajectory looks like -- it cannot say whether the
    spread comes from children differing in level, in rate, or in shape.

    Returns how many were drawn, for the legend.
    """
    samples = np.asarray(trajectory_samples)
    if samples.ndim != 2:
        raise ValueError("trajectory_samples must have shape (n_grid, n_samples).")
    if samples.shape[0] != X_plot.size:
        raise ValueError(
            f"trajectory_samples grid ({samples.shape[0]}) must match "
            f"X_plot ({X_plot.size})."
        )
    if samples.shape[1] == 0:
        return 0

    take = min(int(n_trajectories), samples.shape[1])
    columns = np.random.default_rng(seed).choice(samples.shape[1], take, replace=False)
    for column in columns:
        plt.plot(
            X_plot,
            samples[:, column],
            color=_PREDICTIVE_TRAJECTORY_COLOUR,
            lw=0.7,
            alpha=0.45,
            zorder=2.2,
        )
    return take


def _interval_by_sample(
    values: np.ndarray, prob: float, kind: intervals.IntervalKind = "eti"
) -> np.ndarray:
    """Per-plot-point credible interval over posterior samples (axis 0)."""
    return intervals.bands(values, prob, kind, sample_axis=0)


def _interval_by_row(
    values: np.ndarray, prob: float, kind: intervals.IntervalKind = "eti"
) -> np.ndarray:
    """Per-row credible interval over the second (sample) axis of ``values``."""
    return intervals.bands(values, prob, kind, sample_axis=1)


# ------------------------------------------------------------
# Prior predictive plots
# ------------------------------------------------------------

def plot_prior_samples(
    x: np.ndarray,
    y_samples: np.ndarray,
    x_obs: np.ndarray | pd.Series,
    y_obs: np.ndarray | pd.Series,
    n_trials: int = 810,
    n_curves: int = 1000,
    x_label: str = "x",
    y_label: str = "y",
    filename: str | None = None,
    output_dir: str | None = None,
    random_seed: int | None = None,
) -> Figure:
    """Prior-predictive spaghetti plot at this project's checklist size.

    A thin wrapper over the shared
    :func:`dse_research_utils.plot.predictive.plot_prior_samples_binomial` that
    carries the vocabulary-checklist defaults (``n_trials=810``,
    ``n_curves=1000``). ``random_seed`` selects which prior draws are drawn;
    left ``None`` the curve selection is unseeded, as before.
    """
    return plot_predictive.plot_prior_samples_binomial(
        x,
        y_samples,
        x_obs,
        y_obs,
        n_trials,
        n_curves,
        x_label,
        y_label,
        filename,
        output_dir,
        random_seed=random_seed,
    )


def plot_prior_samples_ratio(
    x: np.ndarray,
    ratio_samples: np.ndarray,
    *,
    y_label: str,
    filename: str | None = None,
    output_dir: str | None = None,
    colour: str = plot_styles.COLOUR_ORANGE,
    alpha: float = 0.1,
    lw: float = 1.0,
    n_curves: int = 500,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    x_label: str = "Age (months)",
) -> Figure:
    """
    Plot prior-sample curves for a production/signed ratio (q or r) in [0, 1].

    All curves share a single colour at a legible alpha — unlike matplotlib's
    default colour cycle at near-zero alpha, which renders the curves a faint,
    multi-coloured wash. Matches the single-colour convention of
    ``plot_prior_samples`` (the observed-data scatter elsewhere is blue, so the
    curves default to orange to contrast).

    Parameters
    ----------
    x
        Plotting grid (e.g. age in months), shape ``(n_plot,)``.
    ratio_samples
        Prior-sample ratio curves, shape ``(n_plot, n_samples)``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    n = min(n_curves, ratio_samples.shape[1])
    for i in range(n):
        ax.plot(x, ratio_samples[:, i], c=colour, alpha=alpha, lw=lw)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if filename is not None and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        _save_png_svg(fig, output_dir, filename)

    return fig


def plot_prior_predictions(
    x: np.ndarray,
    y_pred: np.ndarray,
    x_obs: np.ndarray | pd.Series,
    y_obs: np.ndarray | pd.Series,
    n_trials: int = 810,
    x_label: str = "x",
    y_label: str = "y",
    filename: str | None = None,
    output_dir: str | None = None,
) -> Figure:

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    # Seeded so the scatter of sampled prior-predictive draws is reproducible
    # (matches the seeded RNG used by the other spaghetti/scatter plots).
    rng = np.random.default_rng(42)
    for i in rng.integers(0, y_pred.shape[1], 500):
        plt.scatter(x, y_pred[:, i], color=plot_styles.COLOUR_ORANGE, alpha=0.01, s=12)

    plt.scatter(
        x_obs,
        y_obs,
        color=plot_styles.COLOUR_BLUE,
        alpha=0.4,
        s=10,
        label="Observed data",
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(-20, n_trials + 50)

    if filename is not None and output_dir is not None:
        # PNG only: the dense per-draw scatter produces a multi-megabyte SVG that
        # nothing embeds (reports use the PNG).
        _save_png_svg(plt.gcf(), output_dir, filename, svg=False)

    return plt.gcf()


def _draw_ppc_count_distribution(
    ax,
    age: float,
    draws: np.ndarray,
    n_trials: int,
    bin_width: int,
    ci_prob: float,
    interval_kind: intervals.IntervalKind,
    count_axis_max: int | None = None,
) -> None:
    """Draw one query age's posterior predictive count distribution on ``ax``.

    Shared by the combined grid figure and the individual per-age figures
    (issue #123). Does not set the x-axis label — the caller owns that, so the
    combined grid can label only its bottom row.

    ``count_axis_max`` caps the word-count axis (defaults to ``n_trials``); the
    label offsets scale with it so annotations stay on-plot when the axis is
    zoomed in.
    """
    axis_max = n_trials if count_axis_max is None else count_axis_max
    label_off = axis_max * 0.19
    med_off = axis_max * 0.14
    bins = np.arange(0, n_trials + bin_width + 1, bin_width)
    centres = (bins[:-1] + bins[1:]) / 2
    draws = draws.astype(int)

    counts, _ = np.histogram(draws, bins=bins)
    pmf_bins = counts / counts.sum()

    med = np.median(draws)
    kind_label = "HDI" if interval_kind == "hdi" else "ETI"
    pct = int(round(ci_prob * 100))
    lo, hi = intervals.interval_1d(draws, ci_prob, interval_kind)
    lo50, hi50 = intervals.interval_1d(draws, intervals.INNER_CI_PROB, interval_kind)

    ylim_max = max(pmf_bins.max() * 1.08, 0.25)

    # Outer + inner credible interval
    ax.fill_betweenx(
        [0, ylim_max * 0.96], lo, hi, color=plot_styles.COLOUR_GREEN, alpha=0.10
    )
    ax.fill_betweenx(
        [0, ylim_max * 0.96], lo50, hi50, color=plot_styles.COLOUR_GREEN, alpha=0.18
    )
    # Both bands are annotated. The inner one was drawn but unlabelled, which left
    # the reader to guess what the darker shading meant -- and the two are easy to
    # confuse precisely because an equal-tailed interval is asymmetric about the
    # median, so neither band's edges sit where an eye expects them to.
    inner_pct = int(round(intervals.INNER_CI_PROB * 100))
    ax.text(
        hi + label_off,
        ylim_max * 0.9,
        f"{pct}% {kind_label}: {lo:.0f} to {hi:.0f}",
        color=plot_styles.COLOUR_GREEN,
        ha="center",
    )
    ax.text(
        hi + label_off,
        ylim_max * 0.82,
        f"{inner_pct}% {kind_label}: {lo50:.0f} to {hi50:.0f}",
        color=plot_styles.COLOUR_GREEN,
        ha="center",
    )

    # PMF histogram
    ax.bar(centres, pmf_bins, width=bin_width, color=plot_styles.COLOUR_BLUE, align="center")

    # median
    ax.axvline(med, lw=2, ls="--", color=plot_styles.COLOUR_RED)
    ax.text(
        med + med_off,
        ylim_max * 0.98,
        f"median: {med:.0f}",
        color=plot_styles.COLOUR_RED,
        ha="center",
    )

    ax.set_title(f"{age:.1f} months")
    ax.set_ylim(0, ylim_max)
    ax.set_ylabel(f"Posterior predictive probability mass (bin width {bin_width})")
    ax.set_xlim(0, axis_max)
    tick_step = 100 if axis_max > 400 else 50
    ax.set_xticks(np.arange(0, axis_max + 1, tick_step))


def plot_posterior_predictive_count_distributions_by_query_age(
    X_query: np.ndarray,
    y_query: np.ndarray,
    n_trials: int,
    bin_width: int = 5,
    plot_cols: int = 2,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    x_label: str = "Word count",
    count_axis_max: int | None = None,
    max_age_months: float | None = None,
) -> Figure:
    """
    For each query age, plot the posterior predictive distribution of counts, as a histogram.

    ``count_axis_max`` caps the word-count axis of every subplot (defaults to
    ``n_trials``, i.e. 810), so young ages whose mass sits well below the full
    inventory can be zoomed in.

    Returns the combined grid figure (one subplot per query age; built for the
    return value but no longer written to disk — the reports embed the per-age
    figures). When ``output_dir``/``filename`` are given, writes each age as its
    own ISO A-landscape file ``{filename}_{age}m.{png,svg}`` and the summary
    table ``{filename}.csv`` (issue #123).
    """
    # ``max_age_months`` drops query ages past the outcome's reporting cap. This
    # grid is the one place the two came apart: ``ages_query`` runs to 90, so the
    # understood panels were drawn at 90 while every other understood artefact
    # stopped at 84. See :mod:`vocab_growth.reporting_ages`.
    X_query = np.asarray(X_query, dtype=float).reshape(-1)
    y_query = np.asarray(y_query)
    if max_age_months is not None:
        keep = X_query <= max_age_months
        X_query = X_query[keep]
        y_query = y_query[keep, ...]

    nq = len(X_query)
    axis_max = n_trials if count_axis_max is None else count_axis_max

    plot_rows = int(np.ceil(nq / plot_cols))
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(10, 3.8 * plot_rows), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for j, age in enumerate(X_query):
        _draw_ppc_count_distribution(
            axes[j], age, y_query[j, :], n_trials, bin_width, ci_prob, interval_kind,
            axis_max,
        )

    for k in range(nq, len(axes)):
        axes[k].axis("off")

    for ax in axes[max(0, len(axes) - plot_cols) :]:
        ax.set_xlabel(f"{x_label} (bins of {bin_width})")

    fig.suptitle("Posterior predictive distributions at query ages", y=1.02)

    if filename is not None and output_dir is not None:
        # Clear the previous run's per-age figures before writing this run's.
        # Without this a tightened reporting cap leaves the old, wider set on
        # disk, and `ppc_count_distribution_gallery` globs whatever it finds:
        # VG02 kept publishing a 90-month comprehension figure against an
        # 84-month cap, and it propagated into docs/report/figures/. The
        # capped `{filename}.csv` written below has no matching row, so the
        # orphan was invisible to `tests/test_reporting_age_policy.py`, which
        # reads tables.
        import glob as _glob
        import re as _re

        # Anchored on the full basename: an unanchored `{filename}_*m.png` glob
        # also matches another outcome's files, so the understood writer would
        # delete the spoken figures when the prefixes share a stem.
        stale_pattern = _re.compile(
            rf"^{_re.escape(filename)}_\d+(?:\.\d+)?m\.(?:png|svg)$"
        )
        for stale in _glob.glob(os.path.join(output_dir, f"{filename}_*m.png")) + _glob.glob(
            os.path.join(output_dir, f"{filename}_*m.svg")
        ):
            if stale_pattern.match(os.path.basename(stale)):
                os.remove(stale)

        # The combined grid is no longer written — reports embed the per-age
        # figures below (via ppc_count_distribution_gallery). The grid is still
        # returned for callers/tests.
        # Individual per-age figures, in ISO A landscape (issue #123).
        for j, age in enumerate(X_query):
            fig_i, ax_i = plt.subplots(figsize=_iso_a_landscape_figsize())
            _draw_ppc_count_distribution(
                ax_i, age, y_query[j, :], n_trials, bin_width, ci_prob, interval_kind,
                axis_max,
            )
            ax_i.set_xlabel(f"{x_label} (bins of {bin_width})")
            fig_i.tight_layout()
            _save_png_svg(fig_i, output_dir, f"{filename}_{age:g}m")
            plt.close(fig_i)

        rows = []
        for j, age in enumerate(X_query):
            draws = y_query[j, :].astype(int)
            med = np.median(draws)
            lo50, hi50 = intervals.interval_1d(draws, intervals.INNER_CI_PROB, interval_kind)
            lo, hi = intervals.interval_1d(draws, ci_prob, interval_kind)
            rows.append({
                "age_months": age, "median": med,
                "ci50_lo": lo50, "ci50_hi": hi50, "ci_lo": lo, "ci_hi": hi,
            })
        _save_csv(pd.DataFrame(rows), output_dir, filename)

    return fig


def ppc_count_distribution_gallery(
    prefix: str, *, ncol: int = 2, directory: str = "."
) -> None:
    """Emit a Quarto column layout of the individual per-age posterior predictive
    count-distribution plots (issue #123).

    Intended for a model report cell with ``#| output: asis``. Globs
    ``{prefix}_<age>m.png`` in ``directory``, and prints one lightboxed image per
    age, sorted by age. Falls back to the combined grid figure ``{prefix}.png``
    when no per-age files are present (e.g. before a model is re-fit with the
    individual plots), so the report section is never empty. Prints nothing if
    neither is present.

    Ages absent from the companion table ``{prefix}.csv`` are skipped. That
    table is written under the outcome's reporting cap, so this keeps the
    gallery inside the cap even when a figure from an earlier, looser run is
    still on disk -- which is how VG02 came to publish a 90-month comprehension
    figure against an 84-month cap. The fit-time writer now clears stale
    figures, but this guard also protects fits produced before it did.
    """
    import glob
    import re

    # Anchored on the prefix so that a gallery for one outcome cannot pick up
    # another's files when the prefixes share a stem.
    age_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+(?:\.\d+)?)m\.png$")

    def _age(path: str) -> float:
        m = age_pattern.match(os.path.basename(path))
        return float(m.group(1)) if m else float("inf")

    permitted: set[float] | None = None
    table_path = os.path.join(directory, f"{prefix}.csv")
    if os.path.exists(table_path):
        try:
            table = pd.read_csv(table_path)
            if "age_months" in table:
                permitted = {float(age) for age in table["age_months"]}
        except (OSError, ValueError):
            permitted = None

    candidates = glob.glob(os.path.join(directory, f"{prefix}_*m.png"))
    files = [path for path in candidates if _age(path) != float("inf")]
    if permitted is not None:
        files = [path for path in files if _age(path) in permitted]
    files.sort(key=_age)
    if not files:
        combined = os.path.join(directory, f"{prefix}.png")
        if os.path.exists(combined):
            print(f'![]({os.path.basename(combined)}){{.lightbox fig-align="left"}}')
        return

    print(f"::: {{layout-ncol={ncol}}}")
    for path in files:
        print(f"![{_age(path):g} months]({os.path.basename(path)}){{.lightbox}}")
        print()
    print(":::")

def plot_posterior_predictive_pmf(
    X_query: np.ndarray,
    y_query: np.ndarray,
    n_trials: int,
    log_scale: bool = False,
    output_dir: str | None = None,
    filename: str | None = None,
    x_label: str = "Word count",
    max_age_months: float | None = None,
) -> Figure:
    """
    For each query age, plot the posterior predictive distribution of counts as a PMF on a common support.

    ``y_query`` carries the exact posterior-predictive draws at the query ages,
    shape ``(n_query, n_samples)`` — previously the nearest point on the plot
    grid was substituted, so each panel showed the distribution at an age up to
    half a grid step away from the one in its label (#234).

    ``max_age_months`` drops query ages past the outcome's reporting cap. Age
    lives in the *column names* here (``pmf_84m``), not in a column, so this
    table is easy to miss when auditing which artefacts are capped — it was.
    """
    X_query = np.asarray(X_query, dtype=float).reshape(-1)
    y_query = np.asarray(y_query)
    if y_query.shape[0] != X_query.shape[0]:
        raise ValueError(
            f"y_query has {y_query.shape[0]} rows but X_query has "
            f"{X_query.shape[0]} ages."
        )
    if max_age_months is not None:
        keep = X_query <= max_age_months
        X_query = X_query[keep]
        y_query = y_query[keep, :]

    draws_by_age = [y_query[i, :].astype(int) for i in range(X_query.size)]
    all_draws = np.concatenate(draws_by_age)
    x_lo, x_hi = np.quantile(all_draws, [0.01, 0.99])
    x_lo = int(max(0, np.floor(x_lo)))
    x_hi = int(min(n_trials, np.ceil(x_hi)))

    k = np.arange(x_lo, x_hi + 1)
    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    for a, draws in zip(X_query, draws_by_age, strict=True):
        # Empirical PMF on common support
        in_support = (draws >= x_lo) & (draws <= x_hi)
        counts = np.bincount(draws[in_support] - x_lo, minlength=len(k))
        pmf = counts[: len(k)] / draws.size
        # Step line (discrete PMF)
        plt.step(k, pmf, where="mid", lw=2, label=f"{a:.0f}m")

    plt.xlabel(x_label)
    plt.ylabel("Posterior predictive probability")
    plt.title("Posterior predictive PMF at selected ages")
    plt.xlim(x_lo, x_hi)

    if log_scale:
        plt.yscale("log")

    plt.legend(title="Age", ncol=2, frameon=True)

    if filename is not None and output_dir is not None:
        _save_png_svg(plt.gcf(), output_dir, filename)
        csv_data = {"word_count": k}
        for a, draws in zip(X_query, draws_by_age, strict=True):
            in_support = (draws >= x_lo) & (draws <= x_hi)
            counts = np.bincount(draws[in_support] - x_lo, minlength=len(k))
            pmf = counts[: len(k)] / draws.size
            csv_data[f"pmf_{a:.0f}m"] = pmf
        _save_csv(pd.DataFrame(csv_data), output_dir, filename)

    return plt.gcf()



def plot_posterior_predictive_cdf(
    X_query: np.ndarray,
    y_query: np.ndarray,
    n_trials: int,
    output_dir: str | None = None,
    filename: str | None = None,
    x_label: str = "Words spoken (count)",
    max_age_months: float | None = None,
) -> Figure:
    """For each query age, plot the posterior predictive CDF of counts.

    ``y_query`` carries the exact posterior-predictive draws at the query ages,
    shape ``(n_query, n_samples)`` (see :func:`plot_posterior_predictive_pmf`
    for why the nearest-plot-grid substitution was retired). ``max_age_months``
    drops query ages past the outcome's reporting cap; age is carried in the
    column names.
    """
    X_query = np.asarray(X_query, dtype=float).reshape(-1)
    y_query = np.asarray(y_query)
    if y_query.shape[0] != X_query.shape[0]:
        raise ValueError(
            f"y_query has {y_query.shape[0]} rows but X_query has "
            f"{X_query.shape[0]} ages."
        )
    if max_age_months is not None:
        keep = X_query <= max_age_months
        X_query = X_query[keep]
        y_query = y_query[keep, :]

    draws_by_age = [y_query[i, :].astype(int) for i in range(X_query.size)]

    # Choose a common x-range so curves are comparable (central 99% over all selected ages)
    all_draws = np.concatenate(draws_by_age)
    x_lo, x_hi = np.quantile(all_draws, [0.005, 0.995])
    x_lo = int(max(0, np.floor(x_lo)))
    x_hi = int(min(n_trials, np.ceil(x_hi)))
    k = np.arange(x_lo, x_hi + 1)

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    for a, draws in zip(X_query, draws_by_age, strict=True):
        # Empirical CDF on common support: F(k) = mean(draws <= k)
        # Vectorised computation:
        draws_sorted = np.sort(draws)
        cdf = np.searchsorted(draws_sorted, k, side="right") / draws_sorted.size
        plt.step(k, cdf, where="post", lw=2, label=f"{a:.0f}m")

    plt.xlabel(x_label)
    plt.ylabel("Posterior predictive CDF  P(Y ≤ k)")
    plt.title("Posterior predictive CDFs at selected ages")
    plt.xlim(x_lo, x_hi)
    plt.ylim(0, 1)
    plt.legend(title="Age", ncol=2, frameon=True)

    if filename is not None and output_dir is not None:
        _save_png_svg(plt.gcf(), output_dir, filename)
        csv_data = {"word_count": k}
        for a, draws in zip(X_query, draws_by_age, strict=True):
            draws_sorted = np.sort(draws)
            cdf = np.searchsorted(draws_sorted, k, side="right") / draws_sorted.size
            csv_data[f"cdf_{a:.0f}m"] = cdf
        _save_csv(pd.DataFrame(csv_data), output_dir, filename)

    return plt.gcf()


def _resolve_savgol_window_length(
    n: int,
    window_length: int | None,
    polyorder: int,
) -> int:
    """
    Resolve a valid Savitzky-Golay window length.

    Ensures:
    - odd integer
    - <= n
    - > polyorder
    """
    if n < 3:
        raise ValueError("Need at least 3 points to apply Savitzky-Golay smoothing.")

    if window_length is None:
        # Sensible default: modest smoothing, scale with grid size
        window_length = min(21, n)

    if window_length < 3:
        window_length = 3

    if window_length > n:
        window_length = n

    # Must be odd
    if window_length % 2 == 0:
        window_length -= 1

    # Must be greater than polyorder
    min_valid = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if window_length <= polyorder:
        window_length = min_valid

    if window_length > n:
        raise ValueError(
            f"Cannot choose a valid Savitzky-Golay window_length for n={n} and polyorder={polyorder}."
        )

    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        raise ValueError(
            f"window_length must be greater than polyorder. Got window_length={window_length}, polyorder={polyorder}."
        )

    return window_length


def _maybe_savgol(
    y: np.ndarray,
    smooth: bool,
    window_length: int | None,
    polyorder: int,
) -> np.ndarray:
    """
    Optionally apply Savitzky-Golay smoothing to a 1D array.
    """
    y = np.asarray(y, dtype=float)

    if not smooth:
        return y

    wl = _resolve_savgol_window_length(
        n=len(y),
        window_length=window_length,
        polyorder=polyorder,
    )

    return savgol_filter(y, window_length=wl, polyorder=polyorder)


def plot_posterior_predictive_median_trend(
    X_plot: np.ndarray,
    y_plot: np.ndarray,
    x_obs: np.ndarray | pd.Series,
    y_obs: np.ndarray | pd.Series,
    smooth: bool = False,
    savgol_window_length: int | None = None,
    savgol_polyorder: int = 3,
    smooth_intervals: bool = True,
    output_dir: str | None = None,
    filename: str | None = None,
    y_label: str = "Predicted word count",
    max_age_months: float | None = None,
    subject_ids=None,
    form_max=None,
    min_administrations: int = MIN_ADMINISTRATIONS_FOR_TRAJECTORY,
    trajectory_samples=None,
    n_trajectories: int = DEFAULT_PREDICTIVE_TRAJECTORIES,
    trajectory_seed: int = 0,
):
    """
    Plot the posterior predictive distribution of counts as a function of age,
    showing the predictive median and multiple predictive percentile intervals.

    When ``subject_ids`` is given, the observed points belonging to one child are
    joined, so the figure contrasts individual growth with the population median
    rather than showing an undifferentiated cloud. ``form_max`` supplies each
    observation's recording-form item count and should be passed whenever it is
    available: see :func:`_draw_subject_trajectories` for why a trajectory drawn
    without it can show a reversal that is an artefact of the instrument.

    Parameters
    ----------
    X_plot
        Age grid, shape (n_grid,) or (n_grid, 1).
    y_plot
        Posterior predictive samples, shape (n_grid, n_samples).
    x_obs
        Observed ages.
    y_obs
        Observed counts.
    subject_ids
        Per-observation child identifier. ``None`` (the default) draws the
        observations as a scatter only, which is the behaviour every caller had
        before trajectories existed.
    form_max
        Per-observation item count of the recording form.
    min_administrations
        A child is drawn as a trajectory only with at least this many
        observations.
    trajectory_samples
        Posterior predictive **expected counts for an unseen child**, shape
        ``(n_grid, n_samples)``. A sample of its columns is drawn as curves, so
        the figure shows the trajectories the predictive bands summarise rather
        than only their pointwise quantiles -- a band of pointwise intervals and
        a set of coherent trajectories are different objects, and only the second
        answers how much children differ.

        Pass the *expected* curve, not the predictive counts: the counts carry
        observation noise drawn independently at each grid point, which on a
        500-point grid swamps the trajectory it is scattered around.
    n_trajectories
        How many columns to draw.
    trajectory_seed
        Seed for choosing them, so a figure redrawn from the same trace is the
        same figure.
    output_dir
        Output directory for saved figure.
    filename
        Base filename for saved figure.
    smooth
        If True, apply Savitzky-Golay smoothing to plotted summaries.
    savgol_window_length
        Window length for Savitzky-Golay smoothing. Must be odd and > polyorder.
        If None, a default is chosen.
    savgol_polyorder
        Polynomial order for Savitzky-Golay smoothing.
    smooth_intervals
        If True, smooth interval bounds as well as the median curve.
        If False, smooth only the median.
    y_label
        Label for the y-axis.
    """
    X_plot = np.asarray(X_plot).reshape(-1)
    y_plot = np.asarray(y_plot)
    x_obs = np.asarray(x_obs).reshape(-1)
    y_obs = np.asarray(y_obs).reshape(-1)

    # Trim before smoothing, and trim the scattered observations with the curve.
    # Leaving the points in place would run the x axis past the reported range
    # and invite the trimmed curve to be read as a finding about the trajectory
    # rather than as the edge of what is reported.
    if max_age_months is not None:
        keep = X_plot <= max_age_months
        X_plot = X_plot[keep]
        y_plot = y_plot[keep, :] if y_plot.ndim == 2 else y_plot[keep]
        if trajectory_samples is not None:
            # Grid-indexed like y_plot, and must be cut with it.
            trajectory_samples = np.asarray(trajectory_samples)[keep, :]
        keep_obs = x_obs <= max_age_months
        x_obs = x_obs[keep_obs]
        y_obs = y_obs[keep_obs]
        # The trajectory inputs are per-observation and must be cut with the same
        # mask, or a child's points are joined to another child's.
        if subject_ids is not None:
            subject_ids = subject_ids[keep_obs]
        if form_max is not None:
            form_max = form_max[keep_obs]

    if y_plot.ndim != 2:
        raise ValueError("y_plot must have shape (n_grid, n_samples).")

    if len(X_plot) != y_plot.shape[0]:
        raise ValueError(
            f"X_plot length ({len(X_plot)}) must match y_plot.shape[0] ({y_plot.shape[0]})."
        )

    y_plot_samples_median = np.quantile(y_plot, 0.50, axis=1)

    # Predictive bands are equal-tailed (percentile) intervals; outer 89%, inner 50%.
    outer, inner = intervals.DEFAULT_CI_PROB, intervals.INNER_CI_PROB
    predictive_interval_outer = intervals.bands(y_plot, outer, "eti", sample_axis=1)
    predictive_interval_inner = intervals.bands(y_plot, inner, "eti", sample_axis=1)
    outer_pct = int(round(outer * 100))

    # Optional smoothing for display
    y_plot_samples_median_plot = _maybe_savgol(
        y_plot_samples_median,
        smooth=smooth,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
    )

    def _smooth_band(band: np.ndarray) -> np.ndarray:
        if not (smooth and smooth_intervals):
            return band
        return np.column_stack(
            [
                _maybe_savgol(
                    band[:, k],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                )
                for k in (0, 1)
            ]
        )

    predictive_interval_outer_plot = _smooth_band(predictive_interval_outer)
    predictive_interval_inner_plot = _smooth_band(predictive_interval_inner)

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    plt.fill_between(
        X_plot,
        predictive_interval_outer_plot[:, 0],
        predictive_interval_outer_plot[:, 1],
        alpha=0.20,
        label=f"{outer_pct}% predictive interval (equal-tailed)",
    )
    plt.fill_between(
        X_plot,
        predictive_interval_inner_plot[:, 0],
        predictive_interval_inner_plot[:, 1],
        alpha=0.30,
        label="50% predictive interval (equal-tailed)",
    )
    plt.plot(
        X_plot,
        y_plot_samples_median_plot,
        lw=3,
        label="Posterior median (predictive)",
    )
    predictive_trajectories = 0
    if trajectory_samples is not None:
        predictive_trajectories = _draw_predictive_trajectories(
            X_plot, trajectory_samples, n_trajectories, trajectory_seed
        )

    trajectories = None
    if subject_ids is not None:
        trajectories = _draw_subject_trajectories(
            x_obs,
            y_obs,
            subject_ids,
            form_max=form_max,
            min_administrations=min_administrations,
        )

    plt.scatter(
        x_obs,
        y_obs,
        s=15,
        alpha=0.25,
        label="Observed",
    )

    plt.xlabel("Age (months)")
    plt.ylabel(y_label)
    handles, labels = plt.gca().get_legend_handles_labels()
    if predictive_trajectories:
        handles.append(
            Line2D([], [], color=_PREDICTIVE_TRAJECTORY_COLOUR, lw=0.7, alpha=0.45)
        )
        labels.append(f"Predicted child ({predictive_trajectories} draws)")
    if trajectories and trajectories["children"]:
        handles.append(
            Line2D([], [], color=_TRAJECTORY_COLOUR, lw=0.7, alpha=0.55)
        )
        labels.append(
            f"Same child ({trajectories['children']} with {min_administrations}+)"
        )
        if trajectories["form_changes"]:
            handles.append(
                Line2D(
                    [], [], color=_TRAJECTORY_COLOUR, lw=0.7, alpha=0.5,
                    ls=(0, (2, 2)),
                )
            )
            labels.append(
                f"Form changed - not comparable ({trajectories['form_changes']})"
            )
        if trajectories["near_ceiling"]:
            handles.append(
                Line2D(
                    [], [], ls="none", marker="o", markerfacecolor="none",
                    markeredgecolor=_TRAJECTORY_COLOUR, markeredgewidth=0.9,
                )
            )
            labels.append(
                f"Near that form's ceiling ({trajectories['near_ceiling']})"
            )
    if len(labels) > 4:
        # More than the four the plain figure carries: shrink, or the box covers
        # a quarter of the axes.
        plt.legend(
            handles=handles, labels=labels, loc="upper left",
            frameon=True, fontsize="small",
        )
    else:
        plt.legend(handles=handles, labels=labels, loc="upper left", frameon=True)
    plt.ylim(-20, np.max(y_plot) + 50)

    if filename is not None and output_dir is not None:
        _save_png_svg(plt.gcf(), output_dir, filename)
        # The *plotted* arrays, so the table is the figure's own numbers -- see
        # the note in plot_expected_learning_rate. Without this the "smoothed"
        # sidecar is byte-identical to the unsmoothed one.
        _save_csv(pd.DataFrame({
            "age_months": X_plot,
            "median": y_plot_samples_median_plot,
            "ci50_lo": predictive_interval_inner_plot[:, 0],
            "ci50_hi": predictive_interval_inner_plot[:, 1],
            "ci_lo": predictive_interval_outer_plot[:, 0],
            "ci_hi": predictive_interval_outer_plot[:, 1],
        }), output_dir, filename)

    return plt.gcf()


def plot_expected_learning_rate(
    X_plot: np.ndarray,
    f_plot: np.ndarray,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    smooth: bool = False,
    savgol_window_length: int | None = None,
    savgol_polyorder: int = 3,
    smooth_intervals: bool = True,
    y_label: str = "Estimated word score gain per month",
    max_age_months: float | None = None,
):
    """
    Plot the posterior distribution of the estimated learning rate
    (estimated gain in spoken words per month) across age.

    ``max_age_months`` stops the curve where its outcome's evidence stops, and
    is applied *before* smoothing so the Savitzky-Golay window cannot pull
    values from beyond the cap back across it. See
    :mod:`vocab_growth.reporting_ages`.

    The is the derivative of the conditional expectation of the count given the
    latent function f. The posterior uncertainty bands show how that estimated
    rate varies across posterior draws of f:

        E[Y] = n_trials * sigmoid(f)
        dE[Y]/dx = n_trials * p * (1 - p) * df/dx

    Parameters
    ----------
    X_plot
        Age grid in months, shape ``(n_plot,)``.
    f_plot
        Posterior samples of the latent linear predictor at the plot grid,
        shape ``(n_plot, n_samples)``.
    n_trials
        Maximum count / checklist size.
    ci_prob
        Outer interval probability mass (default 0.89).
    interval_kind
        Interval convention: ``"eti"`` (equal-tailed, default) or ``"hdi"``.
    output_dir
        Directory to save figure files.
    filename
        Base filename (without extension) for saved figure.
    smooth
        If True, apply Savitzky-Golay smoothing to plotted summaries.
    savgol_window_length
        Window length for Savitzky-Golay smoothing. Must be odd and > polyorder.
        If None, a default is chosen.
    savgol_polyorder
        Polynomial order for Savitzky-Golay smoothing.
    smooth_intervals
        If True, smooth HDI bounds as well as the median curve.
        If False, smooth only the median.
    y_label
        Label for the y-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    x_plot_values = np.asarray(X_plot, dtype=float).reshape(-1)
    # f_plot arrives as (n_plot, n_samples); transpose to (n_samples, n_plot)
    f_plot_values = np.asarray(f_plot).T

    if max_age_months is not None:
        keep = x_plot_values <= max_age_months
        x_plot_values = x_plot_values[keep]
        f_plot_values = f_plot_values[:, keep]

    if f_plot_values.shape[1] != x_plot_values.shape[0]:
        raise ValueError(
            f"Shape mismatch: f_plot has {f_plot_values.shape[1]} plot points, "
            f"but X_plot has {x_plot_values.shape[0]}."
        )

    if x_plot_values.shape[0] < 3:
        raise ValueError("At least 3 plot points are required to compute a gradient.")

    # Derivative df/dx along age axis
    dfdx = np.gradient(
        f_plot_values,
        x_plot_values,
        axis=1,
        edge_order=2,
    )  # shape: (n_samples, n_plot)

    # Transform to estimated word-count learning rate
    #    E[Y] = N * sigmoid(f)
    #    dE[Y]/dx = N * p * (1 - p) * df/dx
    p = 1.0 / (1.0 + np.exp(-f_plot_values))
    rate = n_trials * p * (1.0 - p) * dfdx  # words per month

    # Posterior summaries across draws, per age point
    median_rate = np.median(rate, axis=0)
    ci_rate = _interval_by_sample(rate, ci_prob, interval_kind)
    ci50_rate = _interval_by_sample(rate, intervals.INNER_CI_PROB, interval_kind)

    # Optional smoothing for display
    median_rate_plot = _maybe_savgol(
        median_rate,
        smooth=smooth,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
    )

    def _smooth_band(band: np.ndarray) -> np.ndarray:
        if not (smooth and smooth_intervals):
            return band
        return np.column_stack(
            [
                _maybe_savgol(
                    band[:, k],
                    smooth=True,
                    window_length=savgol_window_length,
                    polyorder=savgol_polyorder,
                )
                for k in (0, 1)
            ]
        )

    ci_rate_plot = _smooth_band(ci_rate)
    ci50_rate_plot = _smooth_band(ci50_rate)
    pct = int(round(ci_prob * 100))

    plt.figure(figsize=plot_styles.FIGSIZE_XL)

    plt.fill_between(
        x_plot_values,
        ci_rate_plot[:, 0],
        ci_rate_plot[:, 1],
        alpha=0.20,
        label=f"{pct}% interval",
    )
    plt.fill_between(
        x_plot_values,
        ci50_rate_plot[:, 0],
        ci50_rate_plot[:, 1],
        alpha=0.30,
        label="50% interval",
    )
    plt.plot(
        x_plot_values,
        median_rate_plot,
        lw=3,
        label="Median estimated gain per month",
    )

    plt.xlabel("Age (months)")
    plt.ylabel(y_label)
    plt.legend(loc="upper left", frameon=True)

    if filename is not None and output_dir is not None:
        _save_png_svg(plt.gcf(), output_dir, filename)
        # The *plotted* arrays, so the table is the figure's own numbers. Saving
        # the pre-smoothing arrays made `expected_learning_rate_smoothed.csv`
        # byte-identical to `expected_learning_rate.csv`, so a reader who
        # downloaded the smoothed table to check the smoothed figure got the
        # unsmoothed series without being told.
        _save_csv(pd.DataFrame({
            "age_months": x_plot_values,
            "median_rate": median_rate_plot,
            "ci50_lo": ci50_rate_plot[:, 0],
            "ci50_hi": ci50_rate_plot[:, 1],
            "ci_lo": ci_rate_plot[:, 0],
            "ci_hi": ci_rate_plot[:, 1],
        }), output_dir, filename)
        if not smooth:
            # Draw-wise peak location over the (capped) grid, so the headline
            # "fastest growth" row can carry uncertainty in *where* the peak is
            # rather than only in the rate at one selected age (#234). A draw
            # whose maximum lands on the first or last grid age is
            # boundary-censored — its true peak lies at or beyond the edge of
            # the reported range — so the censored share is recorded alongside.
            # The unsmoothed draws are used; the smoothed call writes no
            # companion because it would duplicate this file byte for byte.
            peak_idx = np.argmax(rate, axis=1)
            peak_ages = x_plot_values[peak_idx]
            n_grid = rate.shape[1]
            boundary_share = float(
                np.mean((peak_idx == 0) | (peak_idx == n_grid - 1))
            )
            peak_lo, peak_hi = intervals.interval_1d(
                peak_ages, ci_prob, interval_kind
            )
            median_idx = int(np.argmax(median_rate))
            _save_csv(
                pd.DataFrame(
                    {
                        "age_min_months": [float(x_plot_values[0])],
                        "age_max_months": [float(x_plot_values[-1])],
                        "median_curve_peak_age_months": [
                            float(x_plot_values[median_idx])
                        ],
                        "median_curve_peak_at_boundary": [
                            bool(median_idx in (0, n_grid - 1))
                        ],
                        "peak_age_median_months": [float(np.median(peak_ages))],
                        "peak_age_ci_lo_months": [float(peak_lo)],
                        "peak_age_ci_hi_months": [float(peak_hi)],
                        "boundary_draw_share": [boundary_share],
                        "ci_prob": [float(ci_prob)],
                    }
                ),
                output_dir,
                f"{filename}_peak",
            )

    return plt.gcf()


def plot_posterior_kappa(
    X_plot: np.ndarray,
    kappa_plot: np.ndarray,
    X_query: np.ndarray,
    kappa_query: np.ndarray,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "hdi",
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
) -> tuple[Figure, pd.DataFrame, pd.DataFrame]:
    """
    Plot the posterior distribution of κ(age) on the plot grid, and return
    summary DataFrames for both the plot grid and query ages.

    ``max_age_months`` stops the curve where its outcome's evidence stops. κ is
    the dispersion *of one outcome*, so it takes that outcome's cap -- see
    :mod:`vocab_growth.reporting_ages`. It trims the query grid as well as the
    plot grid, and the saved CSVs with the figure, so none of them can disagree.

    Parameters
    ----------
    X_plot
        Age grid in months for the plot points, shape ``(n_plot,)``.
    kappa_plot
        Posterior samples of κ at the plot grid,
        shape ``(n_plot, n_samples)``.
    X_query
        Age grid in months for the query points, shape ``(n_query,)``.
    kappa_query
        Posterior samples of κ at the query ages,
        shape ``(n_query, n_samples)``.
    n_trials
        Maximum count / checklist size.
    ci_prob
        Interval probability mass (default 0.89).
    interval_kind
        Interval convention: ``"hdi"`` (default for κ) or ``"eti"``.
    output_dir
        Directory to save figure files.
    filename
        Base filename (without extension) for saved figure.

    Returns
    -------
    tuple[Figure, pd.DataFrame, pd.DataFrame]
        ``(fig, df_kappa_plot, df_kappa_query)`` where each DataFrame contains
        columns for age, median κ, credible interval bounds, intra-cluster
        correlation ρ, and variance inflation factor.
    """
    X_plot = np.asarray(X_plot, dtype=float).reshape(-1)
    kappa_plot_samps = np.asarray(kappa_plot)
    X_query = np.asarray(X_query, dtype=float).reshape(-1)
    kappa_query_samps = np.asarray(kappa_query)

    if max_age_months is not None:
        keep_plot = X_plot <= max_age_months
        X_plot = X_plot[keep_plot]
        kappa_plot_samps = kappa_plot_samps[keep_plot, :]
        keep_query = X_query <= max_age_months
        X_query = X_query[keep_query]
        kappa_query_samps = kappa_query_samps[keep_query, :]

    inner = intervals.INNER_CI_PROB
    kind_label = "HDI" if interval_kind == "hdi" else "interval"
    pct = int(round(ci_prob * 100))

    # --- Plot grid ---
    kappa_plot_med = np.quantile(kappa_plot_samps, 0.5, axis=1)
    kappa_plot_ci = _interval_by_row(kappa_plot_samps, ci_prob, interval_kind)
    kappa_plot_ci50 = _interval_by_row(kappa_plot_samps, inner, interval_kind)
    kappa_plot_lo = kappa_plot_ci[:, 0]
    kappa_plot_hi = kappa_plot_ci[:, 1]

    df_kappa_plot = pd.DataFrame(
        {
            "age_months": X_plot,
            "kappa_ci50_lo": kappa_plot_ci50[:, 0],
            "kappa_ci50_hi": kappa_plot_ci50[:, 1],
            "kappa_ci_lo": kappa_plot_lo,
            "kappa_median": kappa_plot_med,
            "kappa_ci_hi": kappa_plot_hi,
            "rho_median": 1.0 / (kappa_plot_med + 1.0),
            "vif_median": (n_trials + kappa_plot_med) / (1.0 + kappa_plot_med),
        }
    )

    # --- Query ages ---

    kappa_query_med = np.quantile(kappa_query_samps, 0.5, axis=1)
    kappa_query_ci = _interval_by_row(kappa_query_samps, ci_prob, interval_kind)
    kappa_query_ci50 = _interval_by_row(kappa_query_samps, inner, interval_kind)
    kappa_query_lo = kappa_query_ci[:, 0]
    kappa_query_hi = kappa_query_ci[:, 1]

    df_kappa_query = pd.DataFrame(
        {
            "age_months": X_query,
            "kappa_ci50_lo": kappa_query_ci50[:, 0],
            "kappa_ci50_hi": kappa_query_ci50[:, 1],
            "kappa_ci_lo": kappa_query_lo,
            "kappa_median": kappa_query_med,
            "kappa_ci_hi": kappa_query_hi,
            "rho_median": np.median(1.0 / (kappa_query_samps + 1.0), axis=1),
            "vif_median": np.median(
                (n_trials + kappa_query_samps) / (1.0 + kappa_query_samps), axis=1
            ),
        }
    )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        kappa_plot_lo,
        kappa_plot_hi,
        alpha=0.18,
        label=f"{pct}% {kind_label}",
    )
    ax.fill_between(
        X_plot,
        kappa_plot_ci50[:, 0],
        kappa_plot_ci50[:, 1],
        alpha=0.28,
        label=f"50% {kind_label}",
    )
    ax.plot(X_plot, kappa_plot_med, lw=2.5, label="Median κ(age)")

    ax.set_yscale("log")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("κ(age) (log scale)")
    ax.set_title("Posterior κ(age) with credible band")
    ax.legend(frameon=True)

    if filename is not None and output_dir is not None:
        _save_png_svg(fig, output_dir, filename)
        _save_csv(df_kappa_plot, output_dir, filename)
        # Draw-wise endpoint contrast, so the headline "spread widens/narrows"
        # claim carries a posterior sign probability and an interval instead of
        # a comparison of two plug-in medians (#234). The variance-inflation
        # factor is computed per draw at the youngest and oldest (capped) grid
        # ages, and the ratio old/young summarised across draws.
        vif_young = (n_trials + kappa_plot_samps[0, :]) / (
            1.0 + kappa_plot_samps[0, :]
        )
        vif_old = (n_trials + kappa_plot_samps[-1, :]) / (
            1.0 + kappa_plot_samps[-1, :]
        )
        vif_ratio = vif_old / vif_young
        ratio_lo, ratio_hi = intervals.interval_1d(vif_ratio, ci_prob, interval_kind)
        _save_csv(
            pd.DataFrame(
                {
                    "age_young_months": [float(X_plot[0])],
                    "age_old_months": [float(X_plot[-1])],
                    "vif_young_median": [float(np.median(vif_young))],
                    "vif_old_median": [float(np.median(vif_old))],
                    "vif_ratio_median": [float(np.median(vif_ratio))],
                    "vif_ratio_ci_lo": [float(ratio_lo)],
                    "vif_ratio_ci_hi": [float(ratio_hi)],
                    "p_widens": [float(np.mean(vif_ratio > 1.0))],
                    "ci_prob": [float(ci_prob)],
                }
            ),
            output_dir,
            f"{filename}_trend",
        )

    return fig, df_kappa_plot, df_kappa_query


class _RatioGapSamples(Protocol):
    """Structural type for the two joint-trajectory plots below.

    Both ``BivariateModelSamples`` and ``TrivariateModelSamples`` satisfy this;
    declaring it here (instead of importing those dataclasses) keeps ``plotting``
    free of an import cycle with the engine modules.
    """

    X_plot: np.ndarray
    q_plot: np.ndarray
    p_u_plot: np.ndarray
    p_s_plot: np.ndarray


def plot_production_rate(
    samples: _RatioGapSamples,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
) -> Figure:
    """Plot the posterior of the production ratio q(a) = p_S(a) / p_U(a) over age.

    ``max_age_months`` stops the curve where the comprehension evidence stops.
    ``q`` is a ratio *of* comprehension, so it inherits the narrower of the two
    outcomes' age ranges, not the plot grid's — which spans the spoken data. It
    is the model definition's ``report_max_age_understood``, and it trims the
    saved CSV with the figure so the two cannot disagree.
    """
    X_plot = samples.X_plot
    q_plot = samples.q_plot

    if max_age_months is not None:
        keep = X_plot <= max_age_months
        X_plot = X_plot[keep]
        q_plot = q_plot[keep, :]

    q_median = np.median(q_plot, axis=1)
    q_ci = intervals.bands(q_plot, ci_prob, interval_kind, sample_axis=1)
    q_ci50 = intervals.bands(q_plot, intervals.INNER_CI_PROB, interval_kind, sample_axis=1)
    pct = int(round(ci_prob * 100))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        q_ci[:, 0],
        q_ci[:, 1],
        alpha=0.20,
        label=f"{pct}% interval",
    )
    ax.fill_between(
        X_plot,
        q_ci50[:, 0],
        q_ci50[:, 1],
        alpha=0.30,
        label="50% interval",
    )
    ax.plot(X_plot, q_median, lw=3, label="Median q(a)")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("q(a) = p_S(a) / p_U(a)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Production ratio q(a)")

    if output_dir is not None and filename is not None:
        _save_png_svg(fig, output_dir, filename)
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "q_median": q_median,
                    "ci50_lo": q_ci50[:, 0],
                    "ci50_hi": q_ci50[:, 1],
                    "ci_lo": q_ci[:, 0],
                    "ci_hi": q_ci[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_comprehension_production_gap(
    samples: _RatioGapSamples,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    interval_kind: intervals.IntervalKind = "eti",
    output_dir: str | None = None,
    filename: str | None = None,
    max_age_months: float | None = None,
) -> Figure:
    """Plot the posterior of the comprehension-production gap (p_U - p_S) over age.

    ``max_age_months`` stops the curve where the comprehension evidence stops, for
    the same reason as :func:`plot_production_rate`: the gap is a *difference from*
    comprehension, so it inherits comprehension's narrower age range rather than
    the plot grid's, which spans the spoken data.
    """
    X_plot = samples.X_plot
    gap = (samples.p_u_plot - samples.p_s_plot) * n_trials  # in word count units

    if max_age_months is not None:
        keep = X_plot <= max_age_months
        X_plot = X_plot[keep]
        gap = gap[keep, :]

    gap_median = np.median(gap, axis=1)
    gap_ci = intervals.bands(gap, ci_prob, interval_kind, sample_axis=1)
    gap_ci50 = intervals.bands(gap, intervals.INNER_CI_PROB, interval_kind, sample_axis=1)
    pct = int(round(ci_prob * 100))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        X_plot,
        gap_ci[:, 0],
        gap_ci[:, 1],
        alpha=0.20,
        label=f"{pct}% interval",
    )
    ax.fill_between(
        X_plot,
        gap_ci50[:, 0],
        gap_ci50[:, 1],
        alpha=0.30,
        label="50% interval",
    )
    ax.plot(X_plot, gap_median, lw=3, label="Median gap")

    ax.set_xlabel("Age (months)")
    ax.set_ylabel("E[understood] - E[spoken] (words)")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Comprehension-production gap")

    if output_dir is not None and filename is not None:
        _save_png_svg(fig, output_dir, filename)
        _save_csv(
            pd.DataFrame(
                {
                    "age_months": X_plot,
                    "gap_median": gap_median,
                    "ci50_lo": gap_ci50[:, 0],
                    "ci50_hi": gap_ci50[:, 1],
                    "ci_lo": gap_ci[:, 0],
                    "ci_hi": gap_ci[:, 1],
                }
            ),
            output_dir,
            filename,
        )

    return fig


def plot_expected_counts_by_month(
    monthly: pd.DataFrame,
    n_trials: int,
    ci_prob: float = intervals.DEFAULT_CI_PROB,
    output_dir: str | None = None,
    filename: str | None = None,
    y_label: str = "Word count",
    outcome_label: str = "words",
    show_predictive: bool = True,
):
    """Plot expected counts at whole-month resolution, with the predictive range.

    Driven by the DataFrame from
    :func:`vocab_growth.posterior_analysis.monthly_summary_table`, so the figure
    and its companion CSV cannot disagree.

    Two estimands are drawn, kept visually distinct because they are routinely
    confused and quoting one for the other misleads badly:

    - the **expected** count ``Ey`` (filled bands, monthly markers) — the mean
      trajectory with parameter uncertainty only;
    - the **predictive** count ``Y`` for an individual child (dashed outline,
      unfilled) — which is much wider, because it also carries between-child and
      occasion-level dispersion.

    Parameters
    ----------
    monthly
        Whole-month summary table; needs ``age_months``, ``Ey_median``,
        ``Ey_ci*`` and (when ``show_predictive``) ``Y_ci_lo`` / ``Y_ci_hi``.
    n_trials
        Maximum count, used only to set the y-axis bound.
    ci_prob
        Outer interval mass, for the legend label (default 0.89).
    show_predictive
        Draw the individual-child predictive interval alongside the expected
        count. Pass False for a figure about the mean trajectory alone.
    """
    required = {"age_months", "Ey_median", "Ey_ci50_lo", "Ey_ci50_hi", "Ey_ci_lo", "Ey_ci_hi"}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(
            "monthly summary is missing required columns: " + ", ".join(sorted(missing))
        )

    age = monthly["age_months"].to_numpy(dtype=float)
    pct = int(round(ci_prob * 100))

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)

    ax.fill_between(
        age,
        monthly["Ey_ci_lo"].to_numpy(dtype=float),
        monthly["Ey_ci_hi"].to_numpy(dtype=float),
        alpha=0.18,
        color="C0",
        label=f"Expected count ({pct}% interval)",
    )
    ax.fill_between(
        age,
        monthly["Ey_ci50_lo"].to_numpy(dtype=float),
        monthly["Ey_ci50_hi"].to_numpy(dtype=float),
        alpha=0.32,
        color="C0",
        label="Expected count (50% interval)",
    )
    ax.plot(
        age,
        monthly["Ey_median"].to_numpy(dtype=float),
        lw=2.5,
        color="C0",
        marker="o",
        markersize=3,
        label=f"Expected {outcome_label} (median, by month)",
    )

    if show_predictive and {"Y_ci_lo", "Y_ci_hi"} <= set(monthly.columns):
        for column, label in (
            ("Y_ci_lo", f"Individual child ({pct}% predictive interval)"),
            ("Y_ci_hi", None),
        ):
            ax.plot(
                age,
                monthly[column].to_numpy(dtype=float),
                lw=1.4,
                ls="--",
                color="C3",
                label=label,
            )

    ax.set_xlabel("Age (months)")
    ax.set_ylabel(y_label)
    ax.set_xlim(age.min(), age.max())
    ax.set_ylim(-20, n_trials + 50)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(f"Expected {outcome_label} by month of age")

    if output_dir is not None and filename is not None:
        _save_png_svg(fig, output_dir, filename)
        # No sidecar CSV. Unlike every other figure here, this one is handed a
        # frame the caller has *already* written under its own canonical name,
        # `posterior_summary_monthly[_<outcome>].csv` -- the name the reports
        # read, the model inventory documents and the figure sync ships. A
        # sidecar would be a byte-identical second copy under a second name,
        # which is how a reader ends up unsure which of two files is current.
        # See `emit_monthly_summary` in models/common.py.

    return fig
