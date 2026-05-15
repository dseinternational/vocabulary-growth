# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Leave-one-subject-out (LOSO) PSIS comparison of VG07 vs VG08.

Standard PSIS-LOO leaves out one observation at a time, which is unreliable
for VG08 because singleton subjects make the leave-one-out
log-predictive-density unstable (the subject RE is informed by that single
observation). The correct comparison aggregates observations to the subject
level.

Two variants are computed for VG08:

1. **Conditional LOSO** — sum the per-observation log-likelihoods within
   each subject, holding the subject RE at its posterior estimate. This is
   what `az.loo` on a subject-aggregated log-likelihood would compute. It
   biases the comparison in favour of VG08 because each subject's RE was
   estimated from that subject's own data.

2. **Marginal LOSO** — for VG08, integrate the subject RE over its prior
   `Normal(0, tau_subj_u)` via Monte Carlo, so the predictive log density
   reflects the situation where the model has not seen the held-out
   subject. For VG07 there is no subject RE so conditional == marginal.

Outputs:

- `output/comparisons/loso_loo_vg07.csv`
- `output/comparisons/loso_loo_vg08_conditional.csv`
- `output/comparisons/loso_loo_vg08_marginal.csv`
- `output/comparisons/loso_compare_vg07_vs_vg08.csv`
"""

from __future__ import annotations

import math
import os

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import betabinom

import vocab_growth.data_utils as data_utils

EPSILON = 1e-12
N_TRIALS = 800

MODELS_DIR = "output/models"
OUT_DIR = "output/comparisons"
VG07_TRACE = os.path.join(MODELS_DIR, "VG07-age-understood-spoken-ds-re", "trace.nc")
VG08_TRACE = os.path.join(
    MODELS_DIR, "VG08-age-understood-spoken-ds-re-subj", "trace.nc"
)


# ============================================================
# Data — reconstruct the analysis frame and subject mapping
# ============================================================

def load_analysis_frame() -> pd.DataFrame:
    """Reload the DS analysis frame in the same order the models used."""
    df = data_utils.load_combined_data()
    analysis_df = df[["age", "understood", "spoken", "study", "subject_id"]].copy()
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)
    subj_keys = (
        analysis_df["study"].astype(str)
        + "::"
        + analysis_df["subject_id"].astype(str)
    )
    unique_subjects = sorted(subj_keys.unique())
    subject_map = {s: i for i, s in enumerate(unique_subjects)}
    analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
    return analysis_df


# ============================================================
# Conditional LOSO — sum per-obs log_likelihoods within subjects
# ============================================================

def aggregate_to_subject(idata: az.InferenceData, analysis_df: pd.DataFrame) -> az.InferenceData:
    """Build a new InferenceData whose log_likelihood is keyed by subject_id.

    Sums per-observation log-likelihoods across both outcomes within each
    subject, giving a coherent subject-level log predictive density.
    """
    ll = idata.log_likelihood
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values
    subj_u = analysis_df.loc[has_u, "subject_code"].to_numpy(int)
    subj_s = analysis_df.loc[has_s, "subject_code"].to_numpy(int)
    n_subjects = int(analysis_df["subject_code"].max()) + 1

    ll_u = ll["y_u_obs"].values  # (chain, draw, obs_u_id)
    ll_s = ll["y_s_obs"].values  # (chain, draw, obs_s_id)
    n_chain, n_draw = ll_u.shape[:2]
    out = np.zeros((n_chain, n_draw, n_subjects), dtype=ll_u.dtype)
    np.add.at(out, (slice(None), slice(None), subj_u), ll_u)
    np.add.at(out, (slice(None), slice(None), subj_s), ll_s)

    obs_ll = xr.DataArray(
        out,
        dims=("chain", "draw", "subject_id"),
        coords={"subject_id": np.arange(n_subjects)},
    )
    log_lik_ds = xr.Dataset({"y_subj": obs_ll})

    observed = xr.Dataset(
        {"y_subj": xr.DataArray(np.zeros(n_subjects), dims=("subject_id",))}
    )

    return az.InferenceData(
        posterior=idata.posterior,
        log_likelihood=log_lik_ds,
        observed_data=observed,
    )


# ============================================================
# Marginal LOSO for VG08 — integrate the subject RE over its prior
# ============================================================

def marginal_subject_loglik_vg08(
    idata: az.InferenceData,
    analysis_df: pd.DataFrame,
    n_re_samples: int = 500,
    thin: int = 36,
    seed: int = 47,
) -> np.ndarray:
    """Compute marginal per-subject log-likelihood for VG08.

    For each (thinned) posterior draw and each subject, draws K samples
    from the subject RE prior Normal(0, tau_subj_u[draw]) and Monte-Carlo
    averages the conditional log-likelihood over both outcomes.

    Returns an array of shape (n_chain, n_draw_thinned, n_subjects).
    """
    rng = np.random.default_rng(seed)

    post = idata.posterior
    # Thin the draws to keep the computation tractable.
    draws_idx = np.arange(0, post.sizes["draw"], thin)
    n_chain = post.sizes["chain"]
    n_draw = len(draws_idx)
    n_subjects = int(analysis_df["subject_code"].max()) + 1

    # Pull the per-observation conditional linear predictors WITHOUT the
    # subject RE. f_u_obs is the population trajectory; we add the study RE.
    f_u_obs = post["f_u_obs"].values[:, draws_idx, :]   # (chain, draw, obs)
    h_obs = post["h_obs"].values[:, draws_idx, :]       # (chain, draw, obs)
    delta_u = post["delta_u"].values[:, draws_idx, :]   # (chain, draw, study)
    delta_q = post["delta_q"].values[:, draws_idx, :]   # (chain, draw, study)
    tau_subj_u = post["tau_subj_u"].values[:, draws_idx]   # (chain, draw)
    kappa_u_obs = post["kappa_u_obs"].values[:, draws_idx, :]
    kappa_s_obs = post["kappa_s_obs"].values[:, draws_idx, :]

    study_codes = analysis_df["study_code"].to_numpy() if "study_code" in analysis_df.columns else None
    if study_codes is None:
        unique_studies = sorted(analysis_df["study"].unique())
        study_map = {s: i for i, s in enumerate(unique_studies)}
        study_codes = analysis_df["study"].map(study_map).astype(int).to_numpy()

    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values
    y_u = analysis_df.loc[has_u, "understood"].to_numpy(int)
    y_s = analysis_df.loc[has_s, "spoken"].to_numpy(int)
    obs_idx_u = np.where(has_u)[0]
    obs_idx_s = np.where(has_s)[0]
    study_u = study_codes[obs_idx_u]
    study_s = study_codes[obs_idx_s]
    subj_u = analysis_df.loc[has_u, "subject_code"].to_numpy(int)
    subj_s = analysis_df.loc[has_s, "subject_code"].to_numpy(int)

    # Group observation indices by subject for fast aggregation later.
    subj_to_u_ix: dict[int, np.ndarray] = {}
    subj_to_s_ix: dict[int, np.ndarray] = {}
    for s in range(n_subjects):
        subj_to_u_ix[s] = np.where(subj_u == s)[0]
        subj_to_s_ix[s] = np.where(subj_s == s)[0]

    marginal_ll = np.zeros((n_chain, n_draw, n_subjects), dtype=np.float64)

    # Common per-draw normalisations for logsumexp average.
    log_K = math.log(n_re_samples)

    for c in range(n_chain):
        for d in range(n_draw):
            tau = float(tau_subj_u[c, d])
            # Conditional logit at each obs, including study RE (NOT subject).
            f_obs_u_d = f_u_obs[c, d, obs_idx_u] + delta_u[c, d, study_u]
            h_obs_d = h_obs[c, d, obs_idx_s] + delta_q[c, d, study_s]

            # Per-observation kappa_u and kappa_s for this draw.
            kappa_u_d = kappa_u_obs[c, d, obs_idx_u]
            kappa_s_d = kappa_s_obs[c, d, obs_idx_s]

            # Draw K samples of subject RE under the prior.
            re_samples = rng.normal(0.0, tau, size=n_re_samples)

            for s in range(n_subjects):
                u_ix = subj_to_u_ix[s]
                s_ix = subj_to_s_ix[s]

                # For this subject, evaluate the K conditional log-likelihoods.
                # Outer K × n_obs_subject grid.
                if len(u_ix) > 0:
                    f_u_s = f_obs_u_d[u_ix]                                # (n_u_s,)
                    f_u_grid = f_u_s[None, :] + re_samples[:, None]        # (K, n_u_s)
                    p_u_grid = 1.0 / (1.0 + np.exp(-f_u_grid))
                    p_u_grid = np.clip(p_u_grid, EPSILON, 1 - EPSILON)
                    kappa_u_s = kappa_u_d[u_ix]                            # (n_u_s,)
                    alpha = p_u_grid * kappa_u_s[None, :]
                    beta = (1 - p_u_grid) * kappa_u_s[None, :]
                    ll_u_grid = betabinom.logpmf(
                        y_u[u_ix][None, :], N_TRIALS, alpha, beta
                    )
                    ll_u_sum = ll_u_grid.sum(axis=1)   # (K,)
                else:
                    ll_u_sum = np.zeros(n_re_samples)

                if len(s_ix) > 0:
                    h_s = h_obs_d[s_ix]                                    # (n_s_s,)
                    # Production rate: p_s = sigmoid(f_u) * sigmoid(h).
                    # With subject RE on U only, the spoken side also depends
                    # on the same RE through p_u.
                    f_u_for_s = f_u_obs[c, d, obs_idx_s[s_ix]] + delta_u[c, d, study_s[s_ix]]
                    f_u_for_s_grid = f_u_for_s[None, :] + re_samples[:, None]  # (K, n_s_s)
                    p_u_for_s = 1.0 / (1.0 + np.exp(-f_u_for_s_grid))
                    q_for_s = 1.0 / (1.0 + np.exp(-h_s[None, :]))
                    p_s_grid = p_u_for_s * q_for_s
                    p_s_grid = np.clip(p_s_grid, EPSILON, 1 - EPSILON)
                    kappa_s_s = kappa_s_d[s_ix]                            # (n_s_s,)
                    alpha = p_s_grid * kappa_s_s[None, :]
                    beta = (1 - p_s_grid) * kappa_s_s[None, :]
                    ll_s_grid = betabinom.logpmf(
                        y_s[s_ix][None, :], N_TRIALS, alpha, beta
                    )
                    ll_s_sum = ll_s_grid.sum(axis=1)
                else:
                    ll_s_sum = np.zeros(n_re_samples)

                subject_ll_K = ll_u_sum + ll_s_sum
                marginal_ll[c, d, s] = logsumexp(subject_ll_K) - log_K

        print(f"  chain {c+1}/{n_chain} done", flush=True)

    return marginal_ll


def vg08_marginal_idata(
    idata: az.InferenceData,
    analysis_df: pd.DataFrame,
    n_re_samples: int = 500,
    thin: int = 36,
) -> az.InferenceData:
    print(
        f"  computing marginal subject log-likelihoods "
        f"(thin={thin}, K={n_re_samples}) …",
        flush=True,
    )
    ll = marginal_subject_loglik_vg08(idata, analysis_df, n_re_samples, thin)
    n_chain, n_draw, n_subjects = ll.shape
    # Build a thinned posterior to match the log-likelihood draws.
    post_thin = idata.posterior.isel(draw=slice(0, None, thin))
    log_lik_ds = xr.Dataset(
        {
            "y_subj": xr.DataArray(
                ll,
                dims=("chain", "draw", "subject_id"),
                coords={"subject_id": np.arange(n_subjects)},
            )
        }
    )
    observed = xr.Dataset(
        {"y_subj": xr.DataArray(np.zeros(n_subjects), dims=("subject_id",))}
    )
    return az.InferenceData(
        posterior=post_thin,
        log_likelihood=log_lik_ds,
        observed_data=observed,
    )


# ============================================================
# Driver
# ============================================================


def _summary_row(label: str, loo) -> dict:
    k = loo.pareto_k.values if hasattr(loo, "pareto_k") else loo.diagnostics.values
    return {
        "label": label,
        "elpd_loo": float(loo.elpd_loo),
        "se": float(loo.se),
        "p_loo": float(loo.p_loo),
        "pareto_k_gt_0.7": int((k > 0.7).sum()),
        "n_subjects": int(k.size),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reloading DS analysis frame …", flush=True)
    analysis_df = load_analysis_frame()
    n_subjects = analysis_df["subject_code"].nunique()
    n_obs = len(analysis_df)
    print(f"  {n_obs} observations / {n_subjects} subjects")

    print("\nLoading VG07 trace …", flush=True)
    vg07 = az.from_netcdf(VG07_TRACE)
    print("Aggregating VG07 log-likelihoods to subject level …", flush=True)
    vg07_subj = aggregate_to_subject(vg07, analysis_df)
    loo07 = az.loo(vg07_subj, var_name="y_subj", pointwise=True)
    print(f"VG07 LOSO: {loo07}\n")
    pd.DataFrame([_summary_row("VG07", loo07)]).to_csv(
        os.path.join(OUT_DIR, "loso_loo_vg07.csv"), index=False
    )

    print("Loading VG08 trace …", flush=True)
    vg08 = az.from_netcdf(VG08_TRACE)

    print("Conditional LOSO for VG08 (RE held at posterior estimate) …", flush=True)
    vg08_cond = aggregate_to_subject(vg08, analysis_df)
    loo08_cond = az.loo(vg08_cond, var_name="y_subj", pointwise=True)
    print(f"VG08 conditional LOSO: {loo08_cond}\n")
    pd.DataFrame([_summary_row("VG08_conditional", loo08_cond)]).to_csv(
        os.path.join(OUT_DIR, "loso_loo_vg08_conditional.csv"), index=False
    )

    print("Marginal LOSO for VG08 (RE integrated over prior) …", flush=True)
    vg08_marg = vg08_marginal_idata(vg08, analysis_df, n_re_samples=500, thin=36)
    loo08_marg = az.loo(vg08_marg, var_name="y_subj", pointwise=True)
    print(f"VG08 marginal LOSO: {loo08_marg}\n")
    pd.DataFrame([_summary_row("VG08_marginal", loo08_marg)]).to_csv(
        os.path.join(OUT_DIR, "loso_loo_vg08_marginal.csv"), index=False
    )

    print("\n=== az.compare on LOSO (subject-level) ===")
    df_cond = az.compare(
        {"VG07": vg07_subj, "VG08_conditional": vg08_cond},
        ic="loo",
        var_name="y_subj",
    )
    print("Conditional comparison:")
    print(df_cond.to_string())
    df_cond.to_csv(os.path.join(OUT_DIR, "loso_compare_conditional.csv"))

    df_marg = az.compare(
        {"VG07": vg07_subj, "VG08_marginal": vg08_marg},
        ic="loo",
        var_name="y_subj",
    )
    print("\nMarginal comparison (the honest one):")
    print(df_marg.to_string())
    df_marg.to_csv(os.path.join(OUT_DIR, "loso_compare_marginal.csv"))

    # Combined summary table for the report.
    summary = pd.DataFrame(
        [
            _summary_row("VG07", loo07),
            _summary_row("VG08_conditional", loo08_cond),
            _summary_row("VG08_marginal", loo08_marg),
        ]
    )
    summary["elpd_diff_vs_vg07"] = summary["elpd_loo"] - loo07.elpd_loo
    summary.to_csv(
        os.path.join(OUT_DIR, "loso_compare_vg07_vs_vg08.csv"), index=False
    )
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
