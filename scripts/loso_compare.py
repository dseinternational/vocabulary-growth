# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Leave-one-subject-out (LOSO) PSIS comparison across VG07, VG08 and VG09.

- VG07 has no subject RE — conditional == marginal.
- VG08 has a subject RE on the understood logit only.
- VG09 has subject REs on both the understood logit and the production ratio.

For each model we compute:

1. **Conditional LOSO** — sum per-observation log-likelihoods within each
   subject, holding any subject RE at its posterior estimate. Biased
   toward models that include subject REs.

2. **Marginal LOSO** — for each posterior draw and each subject, sample
   K replicates from each subject RE's prior `Normal(0, tau_subj_*)` and
   Monte-Carlo integrate the conditional log-likelihood over those
   replicates. This is the honest "predict an unseen subject" answer.

Outputs:

- `output/comparisons/loso_loo_<MODEL>.csv` — per-model LOSO summary.
- `output/comparisons/loso_compare_conditional.csv`
- `output/comparisons/loso_compare_marginal.csv`
- `output/comparisons/loso_compare_summary.csv` — combined table.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import betabinom

import vocab_growth.data_utils as data_utils
from vocab_growth import environment as env

EPSILON = 1e-12
# TODO(#131): derive from definition.n_trials (this script keys off trace
# folders via ModelSpec and has no model definition object in scope).
N_TRIALS = 800

MODELS_DIR = env.models_output_dir()
OUT_DIR = env.comparisons_output_dir()


@dataclass(frozen=True)
class ModelSpec:
    short: str
    folder: str
    use_subject_re_u: bool
    use_subject_re_q: bool


SPECS = [
    ModelSpec("VG07", "VG07-age-understood-spoken-ds-re", False, False),
    ModelSpec("VG08", "VG08-age-understood-spoken-ds-re-subj", True, False),
    ModelSpec("VG09", "VG09-age-understood-spoken-ds-re-subj-uq", True, True),
]


def load_analysis_frame() -> pd.DataFrame:
    df = data_utils.load_combined_data()
    analysis_df = df[["age", "understood", "spoken", "study", "subject_id"]].copy()
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)

    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    subj_keys = (
        analysis_df["study"].astype(str) + "::" + analysis_df["subject_id"].astype(str)
    )
    unique_subjects = sorted(subj_keys.unique())
    subject_map = {s: i for i, s in enumerate(unique_subjects)}
    analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
    return analysis_df


# ============================================================
# Conditional LOSO (any model)
# ============================================================

def aggregate_to_subject(
    idata: xr.DataTree, analysis_df: pd.DataFrame
) -> xr.DataTree:
    ll = idata.log_likelihood
    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values
    subj_u = analysis_df.loc[has_u, "subject_code"].to_numpy(int)
    subj_s = analysis_df.loc[has_s, "subject_code"].to_numpy(int)
    n_subjects = int(analysis_df["subject_code"].max()) + 1

    ll_u = ll["y_u_obs"].values
    ll_s = ll["y_s_obs"].values
    # The trace's per-observation log-likelihoods are aligned to the freshly
    # re-queried frame purely by row position, and load_combined_data() has no
    # ORDER BY. Guard against a frame/trace row-count mismatch before aligning
    # (mirrors common_joint_modality.sample_posterior_predictive).
    assert ll_u.shape[-1] == len(subj_u), (
        f"understood log-likelihood obs dim ({ll_u.shape[-1]}) does not match "
        f"the re-queried frame's understood-row count ({len(subj_u)}); the "
        "trace and analysis frame are misaligned."
    )
    assert ll_s.shape[-1] == len(subj_s), (
        f"spoken log-likelihood obs dim ({ll_s.shape[-1]}) does not match "
        f"the re-queried frame's spoken-row count ({len(subj_s)}); the "
        "trace and analysis frame are misaligned."
    )
    n_chain, n_draw = ll_u.shape[:2]
    out = np.zeros((n_chain, n_draw, n_subjects), dtype=ll_u.dtype)
    np.add.at(out, (slice(None), slice(None), subj_u), ll_u)
    np.add.at(out, (slice(None), slice(None), subj_s), ll_s)

    obs_ll = xr.DataArray(
        out,
        dims=("chain", "draw", "subject_id"),
        coords={"subject_id": np.arange(n_subjects)},
    )
    return xr.DataTree.from_dict({
        "posterior": idata.posterior,
        "log_likelihood": xr.Dataset({"y_subj": obs_ll}),
        "observed_data": xr.Dataset(
            {"y_subj": xr.DataArray(np.zeros(n_subjects), dims=("subject_id",))}
        ),
    })


# ============================================================
# Marginal LOSO (model-aware)
# ============================================================

def marginal_subject_loglik(
    idata: xr.DataTree,
    analysis_df: pd.DataFrame,
    spec: ModelSpec,
    n_re_samples: int = 500,
    thin: int = 36,
    seed: int = 47,
) -> np.ndarray:
    """Compute marginal per-subject log-likelihood under spec's RE structure.

    For each (thinned) posterior draw and subject, this draws K samples from
    each active subject-RE prior and Monte-Carlo integrates the conditional
    log-likelihood over them (``logsumexp(...) - log K``).

    The path is unified across models: a subject RE that ``spec`` does not
    enable contributes an all-zero draw vector, so its K samples are identical
    and the average collapses exactly to the population+study conditional
    log-likelihood. For VG07 (no subject RE) both REs are zero, so this returns
    the conditional == marginal value — correct, though it evaluates the same
    conditional likelihood K times. (A no-RE short-circuit that instead
    aggregates the stored ``idata.log_likelihood`` directly would be faster and
    read fewer posterior variables; left as a future optimisation since it
    changes an offline analysis path not exercised by the fit tests.)
    """
    rng = np.random.default_rng(seed)
    post = idata.posterior
    draws_idx = np.arange(0, post.sizes["draw"], thin)
    n_chain = post.sizes["chain"]
    n_draw = len(draws_idx)
    n_subjects = int(analysis_df["subject_code"].max()) + 1

    has_u = analysis_df["understood"].notna().values
    has_s = analysis_df["spoken"].notna().values
    y_u = analysis_df.loc[has_u, "understood"].to_numpy(int)
    y_s = analysis_df.loc[has_s, "spoken"].to_numpy(int)
    obs_idx_u = np.where(has_u)[0]
    obs_idx_s = np.where(has_s)[0]
    study_codes = analysis_df["study_code"].to_numpy()
    study_u = study_codes[obs_idx_u]
    study_s = study_codes[obs_idx_s]
    subj_u = analysis_df.loc[has_u, "subject_code"].to_numpy(int)
    subj_s = analysis_df.loc[has_s, "subject_code"].to_numpy(int)

    f_u_obs = post["f_u_obs"].values[:, draws_idx, :]
    h_obs = post["h_obs"].values[:, draws_idx, :]
    delta_u = post["delta_u"].values[:, draws_idx, :]
    delta_q = post["delta_q"].values[:, draws_idx, :]
    kappa_u_obs = post["kappa_u_obs"].values[:, draws_idx, :]
    kappa_s_obs = post["kappa_s_obs"].values[:, draws_idx, :]

    if spec.use_subject_re_u:
        tau_subj_u = post["tau_subj_u"].values[:, draws_idx]
    if spec.use_subject_re_q:
        tau_subj_q = post["tau_subj_q"].values[:, draws_idx]

    subj_to_u_ix = {s: np.where(subj_u == s)[0] for s in range(n_subjects)}
    subj_to_s_ix = {s: np.where(subj_s == s)[0] for s in range(n_subjects)}

    marginal_ll = np.zeros((n_chain, n_draw, n_subjects), dtype=np.float64)
    log_K = math.log(n_re_samples)

    for c in range(n_chain):
        for d in range(n_draw):
            f_obs_u_d = f_u_obs[c, d, obs_idx_u] + delta_u[c, d, study_u]
            h_obs_d = h_obs[c, d, obs_idx_s] + delta_q[c, d, study_s]
            f_u_for_s_d = (
                f_u_obs[c, d, obs_idx_s] + delta_u[c, d, study_s]
            )
            kappa_u_d = kappa_u_obs[c, d, obs_idx_u]
            kappa_s_d = kappa_s_obs[c, d, obs_idx_s]

            # Draw K samples from each RE prior (or zeros for VG07).
            if spec.use_subject_re_u:
                re_u = rng.normal(0.0, float(tau_subj_u[c, d]), size=n_re_samples)
            else:
                re_u = np.zeros(n_re_samples)
            if spec.use_subject_re_q:
                re_q = rng.normal(0.0, float(tau_subj_q[c, d]), size=n_re_samples)
            else:
                re_q = np.zeros(n_re_samples)

            for s in range(n_subjects):
                u_ix = subj_to_u_ix[s]
                s_ix = subj_to_s_ix[s]
                if len(u_ix) > 0:
                    f_u_s = f_obs_u_d[u_ix]
                    f_u_grid = f_u_s[None, :] + re_u[:, None]
                    p_u_grid = 1.0 / (1.0 + np.exp(-f_u_grid))
                    p_u_grid = np.clip(p_u_grid, EPSILON, 1 - EPSILON)
                    kappa_u_s = kappa_u_d[u_ix]
                    alpha = p_u_grid * kappa_u_s[None, :]
                    beta = (1 - p_u_grid) * kappa_u_s[None, :]
                    ll_u_grid = betabinom.logpmf(
                        y_u[u_ix][None, :], N_TRIALS, alpha, beta
                    )
                    ll_u_sum = ll_u_grid.sum(axis=1)
                else:
                    ll_u_sum = np.zeros(n_re_samples)

                if len(s_ix) > 0:
                    f_u_for_s = f_u_for_s_d[s_ix]
                    f_u_for_s_grid = f_u_for_s[None, :] + re_u[:, None]
                    p_u_for_s = 1.0 / (1.0 + np.exp(-f_u_for_s_grid))
                    h_grid = h_obs_d[s_ix][None, :] + re_q[:, None]
                    q_grid = 1.0 / (1.0 + np.exp(-h_grid))
                    p_s_grid = p_u_for_s * q_grid
                    p_s_grid = np.clip(p_s_grid, EPSILON, 1 - EPSILON)
                    kappa_s_s = kappa_s_d[s_ix]
                    alpha_s = p_s_grid * kappa_s_s[None, :]
                    beta_s = (1 - p_s_grid) * kappa_s_s[None, :]
                    ll_s_grid = betabinom.logpmf(
                        y_s[s_ix][None, :], N_TRIALS, alpha_s, beta_s
                    )
                    ll_s_sum = ll_s_grid.sum(axis=1)
                else:
                    ll_s_sum = np.zeros(n_re_samples)

                subject_ll_K = ll_u_sum + ll_s_sum
                marginal_ll[c, d, s] = logsumexp(subject_ll_K) - log_K

        print(f"    chain {c+1}/{n_chain} done", flush=True)

    return marginal_ll


def to_marginal_idata(
    idata: xr.DataTree,
    analysis_df: pd.DataFrame,
    spec: ModelSpec,
    n_re_samples: int = 500,
    thin: int = 36,
) -> xr.DataTree:
    print(
        f"  computing marginal subject log-lik for {spec.short} "
        f"(thin={thin}, K={n_re_samples}) …",
        flush=True,
    )
    ll = marginal_subject_loglik(
        idata, analysis_df, spec, n_re_samples=n_re_samples, thin=thin
    )
    n_chain, n_draw, n_subjects = ll.shape
    post_thin = idata.posterior.isel(draw=slice(0, None, thin))
    return xr.DataTree.from_dict({
        "posterior": post_thin,
        "log_likelihood": xr.Dataset(
            {
                "y_subj": xr.DataArray(
                    ll,
                    dims=("chain", "draw", "subject_id"),
                    coords={"subject_id": np.arange(n_subjects)},
                )
            }
        ),
        "observed_data": xr.Dataset(
            {"y_subj": xr.DataArray(np.zeros(n_subjects), dims=("subject_id",))}
        ),
    })


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
    print(f"  {len(analysis_df)} observations / {n_subjects} subjects\n")

    conditional_idatas: dict[str, xr.DataTree] = {}
    marginal_idatas: dict[str, xr.DataTree] = {}
    summary_rows: list[dict] = []

    for spec in SPECS:
        trace_path = os.path.join(MODELS_DIR, spec.folder, "trace.nc")
        if not os.path.exists(trace_path):
            print(f"  {spec.short}: trace not found at {trace_path} — skipping")
            continue
        print(f"Loading {spec.short} trace …", flush=True)
        idata = az.from_netcdf(trace_path)

        print(f"  conditional LOSO for {spec.short} …", flush=True)
        subj_idata = aggregate_to_subject(idata, analysis_df)
        loo_cond = az.loo(subj_idata, var_name="y_subj", pointwise=True)
        print(f"    {loo_cond}\n")
        conditional_idatas[spec.short] = subj_idata
        summary_rows.append(_summary_row(f"{spec.short}_conditional", loo_cond))

        print(f"  marginal LOSO for {spec.short} …", flush=True)
        marg_idata = to_marginal_idata(idata, analysis_df, spec)
        loo_marg = az.loo(marg_idata, var_name="y_subj", pointwise=True)
        print(f"    {loo_marg}\n")
        marginal_idatas[spec.short] = marg_idata
        summary_rows.append(_summary_row(f"{spec.short}_marginal", loo_marg))

        pd.DataFrame(
            [
                _summary_row(f"{spec.short}_conditional", loo_cond),
                _summary_row(f"{spec.short}_marginal", loo_marg),
            ]
        ).to_csv(os.path.join(OUT_DIR, f"loso_loo_{spec.short}.csv"), index=False)

    # Pairwise comparisons.
    if len(conditional_idatas) >= 2:
        print("\n=== Conditional comparison ===")
        df_cond = az.compare(conditional_idatas, var_name="y_subj")
        print(df_cond.to_string())
        df_cond.to_csv(os.path.join(OUT_DIR, "loso_compare_conditional.csv"))

        print("\n=== Marginal comparison (honest one) ===")
        df_marg = az.compare(marginal_idatas, var_name="y_subj")
        print(df_marg.to_string())
        df_marg.to_csv(os.path.join(OUT_DIR, "loso_compare_marginal.csv"))

    summary_df = pd.DataFrame(summary_rows)
    # Compute elpd_diff relative to VG07 marginal (no subject RE).
    base_marg = next(
        (
            r["elpd_loo"]
            for r in summary_rows
            if r["label"] == "VG07_marginal"
        ),
        None,
    )
    if base_marg is not None:
        summary_df["elpd_diff_vs_vg07_marginal"] = (
            summary_df["elpd_loo"] - base_marg
        )
    summary_df.to_csv(
        os.path.join(OUT_DIR, "loso_compare_summary.csv"), index=False
    )
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
