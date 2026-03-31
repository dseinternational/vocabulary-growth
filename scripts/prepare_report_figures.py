# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

# run from repository root with `python scripts/prepare_report_figures.py`

import os

import dse_research_utils.environment.setup as setup
import matplotlib.pyplot as plt
import numpy as np
from rich import print
from scipy.stats import beta, binom

import vocab_growth.environment as local_env

RANDOM_SEED = 47

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


def prepare_report_figures():
    rng = np.random.default_rng(RANDOM_SEED)
    true_p = 0.65
    alpha0, beta0 = 2.0, 2.0

    plot_bayes_update(20, rng, true_p, alpha0, beta0, "bayes_update")

    plot_bayes_update(100, rng, true_p, alpha0, beta0, "bayes_update_2")

    plot_bayes_update(250, rng, true_p, alpha0, beta0, "bayes_update_3")


if __name__ == "__main__":
    setup.init_script()
    np.random.seed(RANDOM_SEED)

    os.makedirs(local_env.REPORT_FIGS_DIR, exist_ok=True)

    print(f"Writing figures to: {local_env.REPORT_FIGS_DIR}")
    print()

    prepare_report_figures()
