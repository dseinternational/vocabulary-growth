# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG17: study-adjusted contrast of DS *spoken* vocabulary by sign-group.

A single-outcome (words spoken) DS model on the logit scale:

    f(a, s, g) = mean_trend(a) + gp(a) + delta[s] + beta_sign[g]
    delta[s]     ~ Normal(0, tau)        study (dataset) random intercepts
    beta_sign[g] : g in {unknown (ref, 0), non-signer, signer}

restricted to ages 12-66 months (the dense window). The study random intercepts
absorb between-cohort level differences, so `beta_sign` estimates the *residual*
sign-group difference after cohort adjustment. Because sign data was collected
almost entirely at the study level (only uk_02 has both sign and no-sign rows),
the **recorded-vs-unknown** contrast is largely confounded with study and is only
weakly identified once study REs are present; the **signer-vs-non-signer**
contrast varies *within* the sign-recorded studies and is the cleanly identified
one. Both are reported.

Reuses the family's HSGP trend + age-varying Beta-Binomial machinery
(`gp_utils`, `build_utils`); the sign-group covariate is the only structural
addition. Exploratory: self-contained (not routed through the generic engine and
not yet in `MODEL_REGISTRY` / the model inventory) so it cannot perturb the model
family. Folding the covariate into `common_univariate_re` would be the
productionisation step.
"""

import os

import arviz as az
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

import vocab_growth.data_utils as vocab_data_utils
from vocab_growth import environment as env
from vocab_growth import intervals
from vocab_growth.models.build_utils import (
    construct_age_grids,
    slope_anchor_logit_coeffs,
    standardize_ages,
    validate_ell_bounds,
)
from vocab_growth.models.common import ModelConfiguration, get_hsgp_hyperparams
from vocab_growth.models.definitions import VG01
from vocab_growth.models.gp_utils import GPGrid, build_kappa_of_z, trend_and_gp

EPS = 1e-6
N_TRIALS = 810
AGE_LO, AGE_HI = 12.0, 66.0
SIGN_GROUPS = ["unknown", "non_signer", "signer"]  # index 0 is the reference


def _prepare(outcome="spoken", studies=None):
    """Load DS `outcome` data, 12-66 mo, with study and 3-level sign-group codes.

    ``studies`` (optional) restricts to a subset, e.g. ("uk_02", "nz_01") for the
    de-duplicated-union total-expressive analysis (see model_vg18 docstring).
    """
    import duckdb

    with duckdb.connect(vocab_data_utils.VOCABULARY_DATA_PATH, read_only=True) as con:
        df = con.execute(
            "SELECT study, subject_id, age, spoken, signed, produced, survey_vocab_max "
            "FROM vocab_combined"
        ).df()
    # This module reads the view directly rather than through load_combined_data
    # (it needs `produced`), so it must apply the same partial-administration mask
    # the shared loader applies — otherwise an exploratory comparison would use
    # counts that are not on the 810-item reference scale.
    df, _ = vocab_data_utils.mask_incomplete_administrations(df)
    df = df[df[outcome].notna() & df["age"].between(AGE_LO, AGE_HI)].copy()
    if studies is not None:
        df = df[df["study"].isin(list(studies))].copy()
    # Classify sign groups only from sources whose field represents total sign
    # use.  uk_01 is signed-only and uk_06 is not source-verified; both remain in
    # the outcome model as the explicit "unknown" reference group.
    df, _ = vocab_data_utils.mask_incomparable_signed_outcomes(df)
    # sign group: unknown (no sign data) / non-signer (signed==0) / signer (signed>0)
    sg = np.where(df["signed"].isna(), 0, np.where(df["signed"] > 0, 2, 1))
    df["sign_group"] = sg.astype(int)
    studies = sorted(df["study"].unique())
    study_map = {s: i for i, s in enumerate(studies)}
    df["study_code"] = df["study"].map(study_map).astype(int)
    return df.reset_index(drop=True), studies


def _config() -> ModelConfiguration:
    """Reuse VG01's DS-spoken priors (trend/GP/kappa)."""
    b = VG01
    kp = b.kappa
    return ModelConfiguration(
        slope_anchors=b.slope_anchors,
        ell_months_range=b.ell_months_range,
        p_slope_low_dist=pz.Beta(alpha=b.p_slope_low_alpha, beta=b.p_slope_low_beta),
        p_slope_hi_dist=pz.Beta(alpha=b.p_slope_hi_alpha, beta=b.p_slope_hi_beta),
        ell_unit_dist=pz.Beta(alpha=b.ell_unit_alpha, beta=b.ell_unit_beta),
        eta_dist=pz.HalfNormal(sigma=b.eta_sigma),
        kappa_min_dist=pz.LogNormal(mu=kp.kappa_min_mu, sigma=kp.kappa_min_sigma),
        a_kappa_dist=pz.Normal(mu=kp.a_kappa_mu, sigma=kp.a_kappa_sigma),
        b_kappa_mag_dist=pz.HalfNormal(sigma=kp.b_kappa_mag_sigma),
        n_plot=b.n_plot,
        ages_query=b.ages_query,
    )


def _build(df, studies, config, y_col="spoken", tau_study_sigma=0.5, beta_sign_sigma=1.0):
    X_obs = np.asarray(df["age"], dtype=float).reshape(-1, 1)
    y_obs = np.asarray(df[y_col], dtype=int)
    study_codes = np.asarray(df["study_code"], dtype=int)
    sign_codes = np.asarray(df["sign_group"], dtype=int)
    n = len(X_obs)
    n_studies = len(studies)

    X_mean, X_std, X_z = standardize_ages(X_obs)
    grids = construct_age_grids(
        X_obs, X_z, X_obs_mean=X_mean, X_obs_std=X_std,
        n_plot=config.n_plot, ages_query=config.ages_query,
        slope_anchors=config.slope_anchors, use_gp_anchor=False,
        gp_anchor_age_months=None,
    )
    X_plot, X_query, X_all_z = grids.X_plot, grids.X_query, grids.X_all_z
    n_plot, n_query, n_all = grids.n_plot, grids.n_query, grids.n_all

    ell_low_m, ell_high_m = validate_ell_bounds(config.ell_months_range)
    ell_low_z, ell_high_z = ell_low_m / X_std, ell_high_m / X_std
    L, M = get_hsgp_hyperparams(X_all_z, (ell_low_z, ell_high_z))
    sa_z, sb_z = slope_anchor_logit_coeffs(config.slope_anchors, X_obs_mean=X_mean, X_obs_std=X_std)

    i_o0, i_o1 = 0, n
    i_p0, i_p1 = i_o1, i_o1 + n_plot
    i_q0, i_q1 = i_p1, i_p1 + n_query

    coords = {
        "all_id": np.arange(n_all), "obs_id": np.arange(n),
        "plot_id": np.arange(n_plot), "query_id": np.arange(n_query),
        "study_id": np.arange(n_studies), "x_dim": np.arange(1),
        "sign_id": SIGN_GROUPS, "sign_free": SIGN_GROUPS[1:],
    }

    with pm.Model(coords=coords) as model:
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))
        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))
        sign_obs = pm.Data("sign_obs", sign_codes, dims=("obs_id",))
        pm.Data("age_query", X_query.flatten(), dims=("query_id",))
        pm.Data("age_plot", X_plot.flatten(), dims=("plot_id",))

        f_all = trend_and_gp(
            cfg_low=config.p_slope_low_dist, cfg_hi=config.p_slope_hi_dist,
            cfg_ell=config.ell_unit_dist, cfg_eta=config.eta_dist, suffix="",
            X_all_z_data=X_all_z_data,
            grid=GPGrid(sa_z=sa_z, sb_z=sb_z, ell_low_z=ell_low_z, ell_high_z=ell_high_z, M=M, L=L),
            store_deterministic=True, latent_name="f_all", anchor_idx=None,
        )

        tau = pm.HalfNormal("tau", sigma=tau_study_sigma)
        delta = pm.Deterministic("delta", tau * pm.Normal("delta_raw", 0.0, 1.0, dims="study_id"), dims="study_id")

        # sign-group offsets, unknown = reference (0)
        beta_free = pm.Normal("beta_sign_free", 0.0, beta_sign_sigma, dims="sign_free")
        beta_sign = pm.Deterministic("beta_sign", pt.concatenate([pt.zeros(1), beta_free]), dims="sign_id")

        # contrasts of interest
        pm.Deterministic("nonsigner_vs_unknown", beta_free[0])
        pm.Deterministic("signer_vs_unknown", beta_free[1])
        pm.Deterministic("signer_vs_nonsigner", beta_free[1] - beta_free[0])

        f_obs = f_all[i_o0:i_o1] + delta[study_obs] + beta_sign[sign_obs]
        p_obs = pm.math.sigmoid(f_obs)

        kappa_of_z = build_kappa_of_z(config.kappa_min_dist, config.a_kappa_dist, config.b_kappa_mag_dist)
        kappa_obs = kappa_of_z(X_all_z_data[i_o0:i_o1, 0])

        # population trajectory (delta=0) per sign group, at plot + query ages
        f_pop_plot = f_all[i_p0:i_p1]
        f_pop_query = f_all[i_q0:i_q1]
        pm.Deterministic("p_plot_by_group",
                         pm.math.sigmoid(f_pop_plot[None, :] + beta_sign[:, None]), dims=("sign_id", "plot_id"))
        pm.Deterministic("p_query_by_group",
                         pm.math.sigmoid(f_pop_query[None, :] + beta_sign[:, None]), dims=("sign_id", "query_id"))

        pc = pm.math.clip(p_obs, EPS, 1 - EPS)
        pm.BetaBinomial("y_obs", n=N_TRIALS, alpha=pc * kappa_obs, beta=(1 - pc) * kappa_obs,
                        observed=y_obs, dims=("obs_id",))

    return model, X_plot.flatten()


def fit(config: str = "test", outcome="spoken", label="VG17", subdir="VG17-age-spoken-ds-signgroup",
        studies=None):
    sc = sampling.get_sampling_configuration(config)
    df, studies_list = _prepare(outcome, studies=studies)
    studies = studies_list
    n_by = df.groupby("sign_group").size().reindex(range(3), fill_value=0)
    print(f"[{label}] DS {outcome}, {AGE_LO:.0f}-{AGE_HI:.0f} mo: n={len(df)} obs, {df.subject_id.nunique()} children, "
          f"{len(studies)} studies; sign-group n = unknown {n_by[0]}, non-signer {n_by[1]}, signer {n_by[2]}",
          flush=True)

    model, plot_ages = _build(df, studies, _config(), y_col=outcome)
    with model:
        idata = pm.sample(draws=sc.draws, tune=sc.tune, chains=sc.chains, cores=sc.cores,
                          target_accept=sc.target_accept, random_seed=sc.random_seed,
                          nuts_sampler="nutpie", progressbar=False)

    out_dir = os.path.join(env.output_root(), "models", subdir)
    os.makedirs(out_dir, exist_ok=True)
    idata.to_netcdf(os.path.join(out_dir, "trace.nc"))

    # convergence
    ndiv = int(idata.sample_stats["diverging"].values.sum())
    contrasts = ["nonsigner_vs_unknown", "signer_vs_unknown", "signer_vs_nonsigner"]
    summ = az.summary(idata, var_names=contrasts, round_to="none", ci_kind="eti")
    max_rhat = float(np.nanmax(summ["r_hat"].values))
    print(f"[{label}] sampled: divergences={ndiv}, contrasts max R-hat={max_rhat:.4f}", flush=True)

    # contrasts on logit + odds-ratio scale, 89% equal-tailed interval
    post = idata.posterior
    rows = []
    for c in contrasts:
        v = post[c].values.ravel()
        lo, hi = intervals.interval_1d(v, intervals.DEFAULT_CI_PROB, "eti")
        rows.append((c, v.mean(), lo, hi, np.exp(v.mean()),
                     float((v > 0).mean())))
    tab = pd.DataFrame(rows, columns=["contrast", "logit_mean", "ci_lo", "ci_hi", "odds_ratio", "P(>0)"])
    tab.to_csv(os.path.join(out_dir, "sign_group_contrasts.csv"), index=False)
    print(f"\n[{label}] sign-group contrasts (logit; +ve = more {outcome} than unknown reference):")
    print(tab.to_string(index=False), flush=True)

    # implied expected count (x810) by group at query ages
    pq = post["p_query_by_group"].mean(dim=("chain", "draw")).values * N_TRIALS  # (3, n_query)
    ages_q = _config().ages_query
    eq = pd.DataFrame(pq.T, columns=SIGN_GROUPS)
    eq.insert(0, "age", ages_q)
    eq = eq[(eq.age >= AGE_LO) & (eq.age <= AGE_HI)]
    eq.to_csv(os.path.join(out_dir, "expected_by_group.csv"), index=False)
    print(f"\n[{label}] population expected {outcome} (of 810) by sign group and age:")
    print(eq.round(1).to_string(index=False), flush=True)
    print(f"\n[{label}] outputs -> {out_dir}", flush=True)
    return idata


if __name__ == "__main__":
    import sys
    from multiprocessing import freeze_support

    freeze_support()
    env.set_output_root("/scratch/vg-output")
    fit(sys.argv[1] if len(sys.argv) > 1 else "test")
