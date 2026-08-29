# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import re
from pathlib import Path

import dse_research_utils.environment.setup as setup
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from scipy.stats import beta, binom

import vocab_growth.environment as local_env

RANDOM_SEED = 47
FIGURE_REF_RE = re.compile(r"!\[[^\n]*?\]\((figures/[^)\s]+)\)")


def plot_bayes_update(n, rng, true_p, alpha0, beta0, filename):
    data = rng.binomial(n=1, p=true_p, size=n)
    k = int(data.sum())

    print(f"Simulated data: {k} successes out of {n} trials (true p = {true_p:.2f})")

    alpha_post = alpha0 + k
    beta_post = beta0 + (n - k)

    p_grid = np.linspace(0.0, 1.0, 1000)

    prior_pdf = beta.pdf(p_grid, alpha0, beta0)
    post_pdf = beta.pdf(p_grid, alpha_post, beta_post)

    likelihood = binom.pmf(k, n, p_grid)
    likelihood_norm = likelihood / np.trapezoid(likelihood, p_grid)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(p_grid, prior_pdf, label="Prior", c="#0080e0")
    ax.plot(
        p_grid, likelihood_norm, label=f"Observed ($n = {n}$, $k = {k}$)", c="#009900"
    )
    ax.plot(p_grid, post_pdf, label="Posterior", c="#ff7800")

    ax.axvline(
        k / n,
        linestyle="--",
        linewidth=1,
        c="#009900",
        label=f"Observed mean = {k/n:.2f}",
    )
    ax.axvline(
        true_p,
        linestyle=":",
        linewidth=1,
        c="#ff0033",
        label=f"True $p$ (simulated) = {true_p:.2f}",
    )

    prior_mean = alpha0 / (alpha0 + beta0)
    ax.axvline(
        prior_mean,
        linestyle="--",
        linewidth=1,
        c="#0080e0",
        label=f"Prior mean = {prior_mean:.2f}",
    )

    posterior_mean = alpha_post / (alpha_post + beta_post)
    ax.axvline(
        posterior_mean,
        linestyle="-.",
        linewidth=1,
        c="#ff7800",
        label=f"Posterior mean = {posterior_mean:.2f}",
    )

    ax.set_xlabel("$p$")
    ax.set_ylabel("Density / scaled likelihood")

    ax.set_ylim(0, 9)

    ax.legend()

    fig.savefig(os.path.join(local_env.REPORT_FIGS_DIR, f"{filename}.png"), dpi=300)
    fig.savefig(os.path.join(local_env.REPORT_FIGS_DIR, f"{filename}.svg"))
    pd.DataFrame({
        "p": p_grid,
        "prior_pdf": prior_pdf,
        "likelihood_norm": likelihood_norm,
        "posterior_pdf": post_pdf,
    }).to_csv(os.path.join(local_env.REPORT_FIGS_DIR, f"{filename}.csv"), index=False)


def plot_count_dispersion(rng, n, p, kappa, n_draws, filename):
    """Simulated counts under a Binomial and a mean-matched Beta-Binomial.

    Both panels share the mean np; the Beta-Binomial splits the concentration
    kappa as alpha = p * kappa, beta = (1 - p) * kappa, the parameterisation the
    models use, so the only difference on display is the dispersion.
    """
    alpha_bb = p * kappa
    beta_bb = (1.0 - p) * kappa

    y_binom = rng.binomial(n=n, p=p, size=n_draws)
    p_obs = rng.beta(alpha_bb, beta_bb, size=n_draws)
    y_betabinom = rng.binomial(n=n, p=p_obs)

    print(
        f"Binomial(n={n}, p={p}): mean {y_binom.mean():.1f}, sd {y_binom.std():.1f}; "
        f"Beta-Binomial(alpha={alpha_bb:g}, beta={beta_bb:g}): "
        f"mean {y_betabinom.mean():.1f}, sd {y_betabinom.std():.1f}"
    )

    bins = np.arange(0, n + 16, 15)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for ax, draws, label, colour in (
        (axes[0], y_binom, f"Binomial($n = {n}$, $p = {p}$)", "#0080e0"),
        (
            axes[1],
            y_betabinom,
            f"Beta-Binomial($n = {n}$, $p = {p}$, $\\kappa = {kappa:g}$)",
            "#ff7800",
        ),
    ):
        ax.hist(draws, bins=bins, color=colour, label=label)
        ax.axvline(
            n * p,
            linestyle="--",
            linewidth=1,
            c="#555555",
            label=f"Mean $np = {n * p:.0f}$",
        )
        ax.set_ylabel(f"Draws (of {n_draws})")
        ax.legend()

    axes[1].set_xlabel(f"Count of words checked (out of $n = {n}$)")
    axes[1].set_xlim(0, n)

    fig.savefig(os.path.join(local_env.REPORT_FIGS_DIR, f"{filename}.png"), dpi=300)
    fig.savefig(os.path.join(local_env.REPORT_FIGS_DIR, f"{filename}.svg"))
    pd.DataFrame({
        "binomial": y_binom,
        "beta_binomial": y_betabinom,
    }).to_csv(os.path.join(local_env.REPORT_FIGS_DIR, f"{filename}.csv"), index=False)


def prepare_report_figures():
    rng = np.random.default_rng(RANDOM_SEED)
    true_p = 0.65
    alpha0, beta0 = 2.0, 2.0

    plot_bayes_update(20, rng, true_p, alpha0, beta0, "bayes_update")

    plot_bayes_update(100, rng, true_p, alpha0, beta0, "bayes_update_2")

    plot_bayes_update(250, rng, true_p, alpha0, beta0, "bayes_update_3")

    plot_count_dispersion(
        rng, n=810, p=0.3, kappa=4.0, n_draws=1000,
        filename="binomial_betabinomial_draws",
    )


def _figure_path_for_ref(ref: str) -> Path:
    """Filesystem path for a Markdown figure reference in the report."""
    path = Path(local_env.REPORT_DIR) / ref
    return path if path.suffix else path.with_suffix(".png")


def report_figure_refs() -> list[Path]:
    """Figure paths referenced by report QMD files.

    Quarto's report config sets ``default-image-extension: png``, so extensionless
    references resolve to ``.png`` during PDF rendering.
    """
    paths = set()
    for qmd in Path(local_env.REPORT_DIR).glob("*.qmd"):
        text = qmd.read_text(encoding="utf-8")
        paths.update(
            _figure_path_for_ref(match.group(1))
            for match in FIGURE_REF_RE.finditer(text)
        )
    return sorted(paths)


def write_pending_figure(path: Path) -> None:
    """Create an obvious placeholder for a figure that is not synced yet."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rel = path.relative_to(local_env.REPORT_DIR).as_posix()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            fill=False,
            linewidth=1.5,
            edgecolor="#777777",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.58,
        "Pending figure",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        rel,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        wrap=True,
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prepare_pending_figures() -> int:
    """Create placeholders for report figures that are not available locally."""
    written = 0
    for path in report_figure_refs():
        if not path.exists():
            write_pending_figure(path)
            written += 1
    return written


if __name__ == "__main__":
    setup.init_script()

    os.makedirs(local_env.REPORT_FIGS_DIR, exist_ok=True)

    print(f"Writing figures to: {local_env.REPORT_FIGS_DIR}")
    print()

    prepare_report_figures()
    n_pending = prepare_pending_figures()
    print(f"Pending figure placeholders written: {n_pending}")
