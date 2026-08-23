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
import tempfile
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import betabinom

import vocab_growth.data_utils as data_utils
from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    require_full_trace,
    source_data_hash,
    validate_fit_output,
)
from vocab_growth.models.common_bivariate_re import rebuild_model_context
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.posterior_recompute import missing_deterministics, with_deterministics

EPSILON = 1e-12
# TODO(#131): derive from definition.n_trials (this script keys off trace
# folders via ModelSpec and has no model definition object in scope).
N_TRIALS = 810

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

#: The observation-level posterior the marginal LOSO reconstructs the held-out
#: predictive from. Fits made since 2026-08-23 do not store these (the sampler
#: is told not to -- fit_artifacts.sampled_variable_names), so they are
#: recomputed from the stored free parameters on a rebuilt model graph.
OBS_LEVEL_POSTERIOR = ("f_u_obs", "h_obs", "kappa_u_obs", "kappa_s_obs")


def rebuilt_model_for(spec: ModelSpec, fit_dir: str, scratch_dir: str):
    """The fit's model graph, rebuilt on the current data, for recomputation.

    Recomputing an observation-level quantity from a stored posterior is only
    right if the rebuilt graph is the one the posterior came from, on the same
    rows in the same order; so the fit is first checked against the registered
    definition and the current source-data hash, and refused by name otherwise.
    """
    definition = MODEL_REGISTRY[spec.short.lower()]
    errors = validate_fit_output(
        fit_dir,
        expected_definition=definition,
        expected_source_data_hash=source_data_hash(env.DATA_DIR),
    )
    if errors:
        raise RuntimeError(
            f"Cannot recompute {spec.short}'s observation-level posterior: the fit "
            f"at {fit_dir} does not match the registered definition on the current "
            "data -- " + "; ".join(errors)
        )
    print(f"  rebuilding {spec.short}'s model graph to recompute "
          f"{', '.join(OBS_LEVEL_POSTERIOR)} …", flush=True)
    return rebuild_model_context(definition, output_dir=scratch_dir).model


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
    model=None,
) -> np.ndarray:
    """Compute marginal per-subject log-likelihood under spec's RE structure.

    ``model`` is the fit's graph rebuilt on the same data
    (``rebuild_model_context``), needed to recompute the observation-level
    posterior for a trace that does not store it; a trace that does (one written
    before 2026-08-23) is read as it is and ``model`` is not consulted.

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
    # Thin first, then fill in what the trace does not carry: the
    # observation-level posterior is recomputed on the thinned draws only, which
    # is the whole point of not storing it (fit_artifacts.sampled_variable_names).
    post_thin = (post.to_dataset() if hasattr(post, "to_dataset") else post).isel(
        draw=draws_idx
    )
    if missing_deterministics(post_thin, OBS_LEVEL_POSTERIOR):
        if model is None:
            raise RuntimeError(
                f"{spec.short}'s trace does not carry {OBS_LEVEL_POSTERIOR} and no "
                "rebuilt model was supplied to recompute them."
            )
        post_thin = with_deterministics(post_thin, model, OBS_LEVEL_POSTERIOR)
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

    f_u_obs = post_thin["f_u_obs"].values
    h_obs = post_thin["h_obs"].values
    delta_u = post_thin["delta_u"].values
    delta_q = post_thin["delta_q"].values
    kappa_u_obs = post_thin["kappa_u_obs"].values
    kappa_s_obs = post_thin["kappa_s_obs"].values

    if spec.use_subject_re_u:
        tau_subj_u = post_thin["tau_subj_u"].values
    if spec.use_subject_re_q:
        tau_subj_q = post_thin["tau_subj_q"].values

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
    model=None,
) -> xr.DataTree:
    print(
        f"  computing marginal subject log-lik for {spec.short} "
        f"(thin={thin}, K={n_re_samples}) …",
        flush=True,
    )
    ll = marginal_subject_loglik(
        idata, analysis_df, spec, n_re_samples=n_re_samples, thin=thin, model=model
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
        "elpd_loo": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
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

    # Scratch for the rebuilt models' preparation-stage output (a descriptive
    # statistics CSV); nothing in it is read back.
    scratch = tempfile.TemporaryDirectory(prefix="loso-rebuild-")
    scratch_dir = scratch.name

    for spec in SPECS:
        trace_path = os.path.join(MODELS_DIR, spec.folder, "trace.nc")
        if not os.path.exists(trace_path):
            print(f"  {spec.short}: trace not found at {trace_path} — skipping")
            continue
        # Marginal LOSO reconstructs the held-out predictive from the
        # observation-level posterior (f_u_obs, h_obs, kappa_*_obs, delta_*) and
        # aggregates the stored log-likelihood, none of which a compacted fit
        # carries. Checked before the read: these traces run to tens of
        # gigabytes, and a KeyError after loading one is a poor way to learn it.
        require_full_trace(
            os.path.dirname(trace_path),
            purpose=f"Leave-one-study-out for {spec.short}",
        )
        print(f"Loading {spec.short} trace …", flush=True)
        idata = az.from_netcdf(trace_path)
        model = None
        posterior = idata.posterior
        posterior = (
            posterior.to_dataset() if hasattr(posterior, "to_dataset") else posterior
        )
        if missing_deterministics(posterior, OBS_LEVEL_POSTERIOR):
            model = rebuilt_model_for(spec, os.path.dirname(trace_path), scratch_dir)

        print(f"  conditional LOSO for {spec.short} …", flush=True)
        subj_idata = aggregate_to_subject(idata, analysis_df)
        loo_cond = az.loo(subj_idata, var_name="y_subj", pointwise=True)
        print(f"    {loo_cond}\n")
        conditional_idatas[spec.short] = subj_idata
        summary_rows.append(_summary_row(f"{spec.short}_conditional", loo_cond))

        print(f"  marginal LOSO for {spec.short} …", flush=True)
        marg_idata = to_marginal_idata(idata, analysis_df, spec, model=model)
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
