# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Model VG17: study-adjusted contrast of DS *spoken* vocabulary by sign-group.

**EXPLORATORY. Its output is not validatable and must not be published.** See
:mod:`vocab_growth.models.exploratory` for what a `fit()` here does not produce.

A single-outcome (words spoken) DS model on the logit scale:

    f(a, s, c, g) = mean_trend(a) + gp(a) + delta[s] + delta_subj[c] + beta_sign[g]
    delta[s]      ~ Normal(0, tau)       study (dataset) random intercepts
    delta_subj[c] ~ Normal(0, tau_subj)  child random intercepts
    beta_sign[g] : g in {unknown (ref, 0), non-signer, signer}

restricted to ages 12-66 months (the dense window). Data is loaded through the
canonical `load_combined_data`, so the DS pool's cleaning rules all apply; see
`_prepare`. The study random intercepts absorb between-cohort level differences
and the child intercepts absorb the repeated-measures correlation (most
observations come from children with more than one visit, and some change sign
group between visits), so `beta_sign` estimates the *residual* sign-group
difference after cohort and child adjustment. Because sign data was collected
almost entirely at the study level (only uk_02 has both sign and no-sign rows),
the **recorded-vs-unknown** contrast is largely confounded with study and is only
weakly identified once study REs are present; the **signer-vs-non-signer**
contrast varies *within* the sign-recorded studies and is the cleanly identified
one. Both are reported.

Reuses the family's HSGP trend + age-varying Beta-Binomial machinery
(`gp_utils`, `build_utils`); the sign-group covariate and the study/child
intercepts are the structural additions.

**Exploratory. Its output is not validatable and must not be published.** It is
self-contained -- not routed through a shared engine, not in `MODEL_REGISTRY`,
not in the model inventory -- so it cannot perturb the model family, and since
issue #273 it lives in `vocab_growth.models.exploratory`, whose docstring lists
what its `fit()` does not produce (no manifest, no staged promotion, no
predictive checks, no calibration, no LOO, no convergence gate). Every output
directory it writes carries an `exploratory_output.json` saying so.

Folding the covariate into `common_univariate_re` would be the productionisation
step, and it is a **statistical** decision rather than a packaging one: that
engine constrains its study effects to sum to zero while this model uses
unconstrained offsets, so routing it through would change the model rather than
move it. That work belongs with #266.
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
from vocab_growth.fit_artifacts import save_trace
from vocab_growth.models.build_utils import (
    construct_age_grids,
    standardize_ages,
    standardize_anchor_ages,
    validate_ell_bounds,
)
from vocab_growth.models.common import (
    ModelConfiguration,
    build_kappa_for_config,
    get_hsgp_hyperparams,
    kappa_config_fields,
)
from vocab_growth.models.definitions import VG01
from vocab_growth.models.exploratory import write_exploratory_marker
from vocab_growth.models.gp_utils import GPGrid, trend_and_gp

EPS = 1e-6
N_TRIALS = 810
AGE_LO, AGE_HI = 12.0, 66.0
SIGN_GROUPS = ["unknown", "non_signer", "signer"]  # index 0 is the reference
# Child random-intercept scale. HalfNormal(1.5) is the family's convention for
# child scales (`tau_subj_u_sigma` / `tau_subject_sigma` in models/definitions.py).
TAU_SUBJECT_SIGMA = 1.5


def _prepare(outcome="spoken", studies=None):
    """Load DS `outcome` data, 12-66 mo, with study, child and sign-group codes.

    Data comes from the canonical loader, so every DS-pool cleaning rule the rest
    of the family applies is applied here too (ceiling-only children, below-form-
    floor administrations, duplicate administrations, partial administrations,
    duplicated outcome columns, implausible production, comprehension below
    production). ``include_produced=True`` retains the ``produced`` union column
    that VG18 uses as its outcome; it is the only reason this module ever read the
    view directly, and reading it directly silently skipped five of the seven
    rules (issue #266 finding 6).

    :func:`~vocab_growth.data_utils.mask_incomparable_signed_outcomes` is applied
    on top, and deliberately here rather than in the loader: it is specific to the
    signing models, and the canonical loader leaves ``signed`` alone. For the
    ``produced`` outcome,
    :func:`~vocab_growth.data_utils.drop_ungroupable_produced_unions` is applied
    as well, for the same reason and on the same principle.

    ``studies`` (optional) restricts to a subset, e.g. ("uk_02", "nz_01", "es_01")
    for the de-duplicated-union total-expressive analysis (see model_vg18 docstring).
    """
    df = vocab_data_utils.load_combined_data(include_produced=True)
    df = df[df[outcome].notna() & df["age"].between(AGE_LO, AGE_HI)].copy()
    if studies is not None:
        df = df[df["study"].isin(list(studies))].copy()
    # Classify sign groups only from sources whose field represents total sign
    # use.  uk_01 is signed-only and uk_06 is not source-verified; both remain in
    # the outcome model as the explicit "unknown" reference group.
    df, _ = vocab_data_utils.mask_incomparable_signed_outcomes(df)
    if outcome == "produced":
        # A source whose produced union hides its own sign component cannot be
        # placed in a sign group: it would land in `unknown` while its outcome
        # contains the exposure. See PRODUCED_UNION_WITHOUT_SIGN_DETAIL. Only
        # the produced outcome is affected -- `spoken` does not contain `signed`,
        # which is the whole reason VG17 is the interpretable contrast.
        df, _ = vocab_data_utils.drop_ungroupable_produced_unions(df)
    # sign group: unknown (no sign data) / non-signer (signed==0) / signer (signed>0)
    sg = np.where(df["signed"].isna(), 0, np.where(df["signed"] > 0, 2, 1))
    df["sign_group"] = sg.astype(int)
    studies = sorted(df["study"].unique())
    study_map = {s: i for i, s in enumerate(studies)}
    df["study_code"] = df["study"].map(study_map).astype(int)
    # Child codes, on the family's `subject_key` convention: `subject_id` is only
    # unique within a study.
    subject_keys = df["study"].astype(str) + "::" + df["subject_id"].astype(str)
    df["subject_key"] = subject_keys
    subjects = sorted(subject_keys.unique())
    subject_map = {s: i for i, s in enumerate(subjects)}
    df["subject_code"] = subject_keys.map(subject_map).astype(int)
    return df.reset_index(drop=True), studies, subjects


def _config() -> ModelConfiguration:
    """Reuse VG01's DS-spoken priors (trend/GP/kappa).

    The kappa block follows whichever parameterisation VG01 carries, so this
    stays a genuine reuse rather than a copy that silently diverges.
    """
    b = VG01
    # The engines' own translation, not a copy of it: `kappa_config_fields` is the
    # pure half of `common.configure_kappa_priors`, which cannot be called here
    # because it emits prior figures into a `ModelFitContext` that does not exist at
    # module level. The hand-written copy this replaces is what the docstring above
    # was promising against.
    kappa_fields = kappa_config_fields(b.kappa)
    return ModelConfiguration(
        slope_anchors=b.slope_anchors,
        ell_months_range=b.ell_months_range,
        p_slope_low_dist=pz.Beta(alpha=b.p_slope_low_alpha, beta=b.p_slope_low_beta),
        p_slope_hi_dist=pz.Beta(alpha=b.p_slope_hi_alpha, beta=b.p_slope_hi_beta),
        ell_unit_dist=pz.Beta(alpha=b.ell_unit_alpha, beta=b.ell_unit_beta),
        eta_dist=pz.HalfNormal(sigma=b.eta_sigma),
        n_plot=b.n_plot,
        # VG01's grid runs to 90 months; this model observes 12-66 and takes
        # that as its GP domain, so the ages above 66 made `construct_age_grids`
        # refuse to build at all -- `fit()` raised on its default path (issue
        # #273 finding 4). Clipped rather than widening the domain: the fit
        # already discarded every out-of-window age when it wrote
        # `expected_by_group.csv`, so nothing above 66 was ever reported, and
        # AGE_LO/AGE_HI are a deliberate restriction to the dense window.
        # Widening the domain to 12-90 would extrapolate the HSGP two years past
        # any observation and change L and M, which is a statistical change and
        # not this one (study owner's decision, 2026-08-31).
        ages_query=tuple(age for age in b.ages_query if AGE_LO <= age <= AGE_HI),
        **kappa_fields,
    )


def _build(df, studies, config, y_col="spoken", tau_study_sigma=0.5, beta_sign_sigma=1.0,
           subjects=None, tau_subject_sigma=TAU_SUBJECT_SIGMA):
    X_obs = np.asarray(df["age"], dtype=float).reshape(-1, 1)
    y_obs = np.asarray(df[y_col], dtype=int)
    study_codes = np.asarray(df["study_code"], dtype=int)
    sign_codes = np.asarray(df["sign_group"], dtype=int)
    subject_codes = np.asarray(df["subject_code"], dtype=int)
    n = len(X_obs)
    n_studies = len(studies)
    n_subjects = len(subjects) if subjects is not None else int(subject_codes.max()) + 1

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
    sa_z, sb_z = standardize_anchor_ages(config.slope_anchors, X_obs_mean=X_mean, X_obs_std=X_std)

    i_o0, i_o1 = 0, n
    i_p0, i_p1 = i_o1, i_o1 + n_plot
    i_q0, i_q1 = i_p1, i_p1 + n_query

    coords = {
        "all_id": np.arange(n_all), "obs_id": np.arange(n),
        "plot_id": np.arange(n_plot), "query_id": np.arange(n_query),
        "study_id": np.arange(n_studies), "x_dim": np.arange(1),
        "subject_id": np.arange(n_subjects),
        "sign_id": SIGN_GROUPS, "sign_free": SIGN_GROUPS[1:],
    }

    with pm.Model(coords=coords) as model:
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))
        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))
        subject_obs = pm.Data("subject_obs", subject_codes, dims=("obs_id",))
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

        # Child random intercepts. Most observations come from children with more
        # than one visit, and sign group itself changes across visits for some of
        # them, so without this the repeated measurements are treated as
        # independent and `beta_sign` borrows their within-child correlation
        # (issue #266 finding 6).
        tau_subj = pm.HalfNormal("tau_subj", sigma=tau_subject_sigma)
        delta_subj = pm.Deterministic(
            "delta_subj",
            tau_subj * pm.Normal("delta_subj_raw", 0.0, 1.0, dims="subject_id"),
            dims="subject_id",
        )

        # sign-group offsets, unknown = reference (0)
        beta_free = pm.Normal("beta_sign_free", 0.0, beta_sign_sigma, dims="sign_free")
        beta_sign = pm.Deterministic("beta_sign", pt.concatenate([pt.zeros(1), beta_free]), dims="sign_id")

        # contrasts of interest
        pm.Deterministic("nonsigner_vs_unknown", beta_free[0])
        pm.Deterministic("signer_vs_unknown", beta_free[1])
        pm.Deterministic("signer_vs_nonsigner", beta_free[1] - beta_free[0])

        f_obs = f_all[i_o0:i_o1] + delta[study_obs] + delta_subj[subject_obs] + beta_sign[sign_obs]
        p_obs = pm.math.sigmoid(f_obs)

        kappa_of_z = build_kappa_for_config(config, X_obs_mean=X_mean, X_obs_std=X_std)
        kappa_obs = kappa_of_z(X_all_z_data[i_o0:i_o1, 0])

        # population trajectory (delta=0, delta_subj=0) per sign group, at plot + query ages
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
        studies=None, caution=None):
    sc = sampling.get_sampling_configuration(config)
    df, studies_list, subjects = _prepare(outcome, studies=studies)
    studies = studies_list
    if caution:
        print(f"\n[{label}] {caution}\n", flush=True)
    n_by = df.groupby("sign_group").size().reindex(range(3), fill_value=0)
    per_child = df.groupby("subject_code").size()
    n_repeated = int((per_child > 1).sum())
    n_repeated_rows = int(per_child[per_child > 1].sum())
    print(f"[{label}] DS {outcome}, {AGE_LO:.0f}-{AGE_HI:.0f} mo: n={len(df)} obs, {len(subjects)} children, "
          f"{len(studies)} studies; sign-group n = unknown {n_by[0]}, non-signer {n_by[1]}, signer {n_by[2]}",
          flush=True)
    print(f"[{label}] repeated measures: {n_repeated} of {len(subjects)} children contribute "
          f"more than one observation ({n_repeated_rows} of {len(df)} rows); "
          f"child random intercepts tau_subj ~ HalfNormal({TAU_SUBJECT_SIGMA})", flush=True)
    print(f"[{label}] rows are from the canonical loader (all DS-pool cleaning rules) "
          f"plus the signing-source mask", flush=True)

    model, plot_ages = _build(df, studies, _config(), y_col=outcome, subjects=subjects)
    with model:
        idata = pm.sample(draws=sc.draws, tune=sc.tune, chains=sc.chains, cores=sc.cores,
                          target_accept=sc.target_accept, random_seed=sc.random_seed,
                          nuts_sampler="nutpie", progressbar=False)

    out_dir = os.path.join(env.output_root(), "models", subdir)
    os.makedirs(out_dir, exist_ok=True)
    # Before the trace, so an interrupted run still leaves the directory
    # labelled. This output carries no manifest, no staged promotion and no
    # convergence gate; without the marker it is shaped exactly like a
    # publishable fit (issue #273 finding 4).
    write_exploratory_marker(out_dir, model_label=label, note=caution or None)
    save_trace(idata, out_dir)

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
    if caution:
        print(f"\n[{label}] {caution}", flush=True)

    # implied expected count (x810) by group at query ages
    pq = post["p_query_by_group"].mean(dim=("chain", "draw")).values * N_TRIALS  # (3, n_query)
    ages_q = _config().ages_query
    eq = pd.DataFrame(pq.T, columns=SIGN_GROUPS)
    eq.insert(0, "age", ages_q)
    # No age filter here: `_config()` clips the query grid to the observation
    # window, so this is already in range. Filtering again would give the window
    # two definitions, which is how the grid and the domain came apart.
    eq.to_csv(os.path.join(out_dir, "expected_by_group.csv"), index=False)
    print(f"\n[{label}] population expected {outcome} (of 810) by sign group and age:")
    print(eq.round(1).to_string(index=False), flush=True)
    print(f"\n[{label}] outputs -> {out_dir}", flush=True)
    return idata


if __name__ == "__main__":
    import sys
    from multiprocessing import freeze_support

    freeze_support()
    fit(sys.argv[1] if len(sys.argv) > 1 else "test")
