# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Post-process VG09's trace to produce marginal posterior-predictive count
distributions that integrate over the subject random-effect priors.

VG09's `y_u_query`, `y_u_plot`, `y_s_query`, `y_s_plot` are Beta-Binomial
samples at the *population-level* logits (study and subject REs both = 0).
With substantial subject-RE SDs (`τ^{subj}_U ≈ 0.84`, `τ^{subj}_q ≈ 1.20`)
this systematically understates the spread of counts across the DS
population: the population mean is much sharper than any single child's
expected count.

This script reads VG09's saved trace and, for each posterior draw, draws
one subject RE for U and one for q from their priors, computes the
marginal `p_u` and `p_s` at every plot and query age, samples a single
Beta-Binomial count, and re-runs the existing posterior-predictive count
plot functions with these marginal samples. The output files
(`posterior_predictive_count_distributions_{u,s}.*`,
`posterior_predictive_pmf_{u,s}.*`, `posterior_predictive_cdf_{u,s}.*`,
`posterior_predictive_median_trend_{u,s}.*` and their smoothed variants)
are overwritten in the VG09 output directory.
"""

from __future__ import annotations

import os

import arviz as az
import dse_research_utils.environment.setup as setup
import numpy as np
import pandas as pd
from scipy.stats import betabinom

import vocab_growth.data_utils as data_utils
import vocab_growth.plotting as plotting
from vocab_growth import environment as env
from vocab_growth.models.definitions import VG09

VG09_DIR = os.path.join(env.models_output_dir(), "VG09-age-understood-spoken-ds-re-subj-uq")
N_TRIALS = VG09.n_trials  # trial count from the model this script post-processes
EPSILON = 1e-12
SEED = 47


def _load_vg09_observed_counts() -> pd.DataFrame:
    """Reload the DS analysis frame as VG09 used it, for the overlay scatter."""
    df = data_utils.load_combined_data()
    df = df[["age", "understood", "spoken"]].copy()
    df = df.dropna(subset=["age"])
    df = df[df["understood"].notna() | df["spoken"].notna()].reset_index(drop=True)
    return df


def main() -> None:
    setup.init_script()

    print(f"Loading VG09 trace from {VG09_DIR}/trace.nc …", flush=True)
    idata = az.from_netcdf(os.path.join(VG09_DIR, "trace.nc"))

    post = idata.posterior
    n_chain = post.sizes["chain"]
    n_draw = post.sizes["draw"]
    N = n_chain * n_draw

    # Population-level latent quantities (REs = 0).
    f_u_plot = post["f_u_plot"].values.reshape(N, -1)      # (N, n_plot)
    h_plot = post["h_plot"].values.reshape(N, -1)
    f_u_query = post["f_u_query"].values.reshape(N, -1)    # (N, n_query)
    h_query = post["h_query"].values.reshape(N, -1)
    kappa_u_plot = post["kappa_u_plot"].values.reshape(N, -1)
    kappa_u_query = post["kappa_u_query"].values.reshape(N, -1)
    kappa_s_plot = post["kappa_s_plot"].values.reshape(N, -1)
    kappa_s_query = post["kappa_s_query"].values.reshape(N, -1)
    tau_subj_u = post["tau_subj_u"].values.reshape(N)
    tau_subj_q = post["tau_subj_q"].values.reshape(N)

    n_plot = f_u_plot.shape[1]
    n_query = f_u_query.shape[1]
    constant = idata.constant_data
    X_plot = constant["X_plot"].values  # (n_plot,)
    X_query = constant["X_query"].values  # (n_query,)

    print(
        f"  N draws = {N} ({n_chain} chains × {n_draw}) | "
        f"n_plot = {n_plot} | n_query = {n_query}"
    )
    print(
        f"  posterior mean tau_subj_u = {tau_subj_u.mean():.3f} | "
        f"tau_subj_q = {tau_subj_q.mean():.3f}"
    )

    rng = np.random.default_rng(SEED)

    print("Sampling marginal subject REs and generating marginal counts …", flush=True)
    # One subject RE sample per posterior draw.
    delta_u_marg = rng.normal(0.0, tau_subj_u)   # (N,)
    delta_q_marg = rng.normal(0.0, tau_subj_q)   # (N,)

    # Build marginal logits/probs at plot and query ages.
    f_u_plot_marg = f_u_plot + delta_u_marg[:, None]
    f_u_query_marg = f_u_query + delta_u_marg[:, None]
    h_plot_marg = h_plot + delta_q_marg[:, None]
    h_query_marg = h_query + delta_q_marg[:, None]

    p_u_plot_marg = 1.0 / (1.0 + np.exp(-f_u_plot_marg))
    p_u_query_marg = 1.0 / (1.0 + np.exp(-f_u_query_marg))
    q_plot_marg = 1.0 / (1.0 + np.exp(-h_plot_marg))
    q_query_marg = 1.0 / (1.0 + np.exp(-h_query_marg))
    p_s_plot_marg = p_u_plot_marg * q_plot_marg
    p_s_query_marg = p_u_query_marg * q_query_marg

    # Sample one BetaBinomial count per (draw, age) under the marginal probability.
    def _sample_bb(p, kappa):
        p_c = np.clip(p, EPSILON, 1 - EPSILON)
        alpha = p_c * kappa
        beta = (1 - p_c) * kappa
        return betabinom.rvs(N_TRIALS, alpha, beta, random_state=rng)

    y_u_plot = _sample_bb(p_u_plot_marg, kappa_u_plot)    # (N, n_plot)
    y_u_query = _sample_bb(p_u_query_marg, kappa_u_query)  # (N, n_query)
    y_s_plot = _sample_bb(p_s_plot_marg, kappa_s_plot)
    y_s_query = _sample_bb(p_s_query_marg, kappa_s_query)

    # Plot functions expect shape (n_grid, n_samples).
    y_u_plot_T = y_u_plot.T
    y_u_query_T = y_u_query.T
    y_s_plot_T = y_s_plot.T
    y_s_query_T = y_s_query.T

    obs_df = _load_vg09_observed_counts()

    print("Regenerating posterior-predictive count plots (understood) …", flush=True)
    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=X_query,
        y_query=y_u_query_T,
        n_trials=N_TRIALS,
        output_dir=VG09_DIR,
        filename="posterior_predictive_count_distributions_u",
        x_label="Words understood (count)",
    )
    plotting.plot_posterior_predictive_pmf(
        X_query=X_query,
        X_plot=X_plot,
        y_plot=y_u_plot_T,
        n_trials=N_TRIALS,
        output_dir=VG09_DIR,
        filename="posterior_predictive_pmf_u",
        x_label="Words understood (count)",
    )
    plotting.plot_posterior_predictive_cdf(
        X_query=X_query,
        X_plot=X_plot,
        y_plot=y_u_plot_T,
        n_trials=N_TRIALS,
        output_dir=VG09_DIR,
        filename="posterior_predictive_cdf_u",
        x_label="Words understood (count)",
    )
    plotting.plot_posterior_predictive_median_trend(
        X_plot=X_plot,
        y_plot=y_u_plot_T,
        x_obs=obs_df["age"].to_numpy(),
        y_obs=obs_df["understood"].to_numpy(),
        smooth=False,
        output_dir=VG09_DIR,
        filename="posterior_predictive_median_trend_u",
        y_label="Words understood (predicted count)",
    )
    plotting.plot_posterior_predictive_median_trend(
        X_plot=X_plot,
        y_plot=y_u_plot_T,
        x_obs=obs_df["age"].to_numpy(),
        y_obs=obs_df["understood"].to_numpy(),
        smooth=True,
        output_dir=VG09_DIR,
        filename="posterior_predictive_median_trend_u_smoothed",
        y_label="Words understood (predicted count)",
    )

    print("Regenerating posterior-predictive count plots (spoken) …", flush=True)
    plotting.plot_posterior_predictive_count_distributions_by_query_age(
        X_query=X_query,
        y_query=y_s_query_T,
        n_trials=N_TRIALS,
        output_dir=VG09_DIR,
        filename="posterior_predictive_count_distributions_s",
        x_label="Words spoken (count)",
    )
    plotting.plot_posterior_predictive_pmf(
        X_query=X_query,
        X_plot=X_plot,
        y_plot=y_s_plot_T,
        n_trials=N_TRIALS,
        output_dir=VG09_DIR,
        filename="posterior_predictive_pmf_s",
        x_label="Words spoken (count)",
    )
    plotting.plot_posterior_predictive_cdf(
        X_query=X_query,
        X_plot=X_plot,
        y_plot=y_s_plot_T,
        n_trials=N_TRIALS,
        output_dir=VG09_DIR,
        filename="posterior_predictive_cdf_s",
        x_label="Words spoken (count)",
    )
    plotting.plot_posterior_predictive_median_trend(
        X_plot=X_plot,
        y_plot=y_s_plot_T,
        x_obs=obs_df["age"].to_numpy(),
        y_obs=obs_df["spoken"].to_numpy(),
        smooth=False,
        output_dir=VG09_DIR,
        filename="posterior_predictive_median_trend_s",
        y_label="Words spoken (predicted count)",
    )
    plotting.plot_posterior_predictive_median_trend(
        X_plot=X_plot,
        y_plot=y_s_plot_T,
        x_obs=obs_df["age"].to_numpy(),
        y_obs=obs_df["spoken"].to_numpy(),
        smooth=True,
        output_dir=VG09_DIR,
        filename="posterior_predictive_median_trend_s_smoothed",
        y_label="Words spoken (predicted count)",
    )

    # Save the marginal arrays for downstream use / inspection.
    np.savez(
        os.path.join(VG09_DIR, "marginal_posterior_predictive.npz"),
        X_plot=X_plot,
        X_query=X_query,
        y_u_plot=y_u_plot,
        y_u_query=y_u_query,
        y_s_plot=y_s_plot,
        y_s_query=y_s_query,
    )

    # Update posterior_summary_{u,s}.csv with marginal-predictive Y quantiles
    # and tail probabilities. Keep the conditional p_*/Ey_* columns as-is —
    # they describe the latent population probability, which is unchanged.
    print("Updating posterior_summary CSVs with marginal Y quantiles …", flush=True)
    cutoffs = [0, 5, 10, 25, 50, 100, 200, 400]

    def _augment_summary(summary_path: str, y_arr: np.ndarray) -> None:
        df = pd.read_csv(summary_path)
        # y_arr: (N, n_query). Quantiles across draws per age. Round to the
        # nearest integer count rather than truncating (floor), which would
        # bias every reported count downward.
        # Equal-tailed 89% (outer) and 50% (inner) intervals, matching the
        # posterior-summary schema (see vocab_growth.intervals).
        q_lo = (1.0 - 0.89) / 2.0
        y_median = np.rint(np.median(y_arr, axis=0)).astype(int)
        df["Y_median"] = y_median
        df["Y_ci50_lo"] = np.rint(np.quantile(y_arr, 0.25, axis=0)).astype(int)
        df["Y_ci50_hi"] = np.rint(np.quantile(y_arr, 0.75, axis=0)).astype(int)
        df["Y_ci_lo"] = np.rint(np.quantile(y_arr, q_lo, axis=0)).astype(int)
        df["Y_ci_hi"] = np.rint(np.quantile(y_arr, 1.0 - q_lo, axis=0)).astype(int)
        for c in cutoffs:
            col = f"P(Y={c})" if c == 0 else f"P(Y<={c})"
            df[col] = (y_arr <= c).mean(axis=0)
        df["P(Y>400)"] = (y_arr > 400).mean(axis=0)
        df.to_csv(summary_path, index=False)
        print(f"  wrote {summary_path}")

    _augment_summary(
        os.path.join(VG09_DIR, "posterior_summary_u.csv"), y_u_query
    )
    _augment_summary(
        os.path.join(VG09_DIR, "posterior_summary_s.csv"), y_s_query
    )

    print(
        "\nDone. VG09's posterior-predictive count plots and summary tables "
        "now reflect the subject-marginalised distribution.",
        flush=True,
    )


if __name__ == "__main__":
    main()
