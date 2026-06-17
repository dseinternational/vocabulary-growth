# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the joint sign/speech modality
model (VG15): issue #49 Option 3.

VG15 extends the trivariate VG14 with two things VG14 assumed away:

1.  A within-understood sign-speech ASSOCIATION (a single scalar Plackett odds
    ratio ``psi``), identified from the uk_02 four-cell cross-tabulation
    (sign-only / sign+speech / speech-only / understood-only). This replaces
    VG14's independence-based ``p_any`` upper bound with a *data-identified*
    total expressive vocabulary.
2.  STUDY random intercepts on each latent trajectory (the VG07-VG10 pattern),
    so the age curve is separated from study composition (which made VG14's
    signed peak unidentifiable).

Latent scale (all out of N = 800 checklist words):

    p_U(a)  = sigmoid(f_U(a))                 # proportion understood
    r(a)    = sigmoid(g(a))                   # P(sign  | understood)
    q(a)    = sigmoid(h(a))                   # P(speak | understood)

    pi_both     = Plackett(r, q; psi)         # P(sign & speak | understood)
    pi_signonly = r - pi_both
    pi_speakonly= q - pi_both
    pi_neither  = 1 - r - q + pi_both
    p_any(a)    = p_U(a) * (r + q - pi_both)   # total expressive (data-identified)

Likelihoods:
    - understood ~ BetaBinomial(800, p_U)              (all DS studies)
    - spoken     ~ BetaBinomial(800, p_U * q)          (marginal-only rows)
    - signed     ~ BetaBinomial(800, p_U * r)          (uk_01/04/05/06 + uk_02 marginal-only)
    - uk_02 four cells ~ DirichletMultinomial(total, conc * [pi_*])  (identifies psi)

This is a self-contained module (like common_trivariate.py); it does not import
from or modify the bivariate / trivariate engines. The full-grid intermediates
are kept as plain tensors (only obs/plot/query slices are stored), following the
VG14 memory discipline.
"""

import os
import shutil
import time
from dataclasses import dataclass

import arviz as az
import dse_research_utils.environment.info as env_info
import dse_research_utils.math.constants as math_constants
import dse_research_utils.metadata.packages as package_metadata
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
from preliz.distributions.distributions import Continuous

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.reporting as vg_reporting
from vocab_growth.models.common import (
    PACKAGE_LIST,
    BaseModelConfiguration,
    ModelFitContext,
    _plot_and_print_dist,
    _report_diagnostic_warnings,
    get_hsgp_hyperparams,
    report,
)
from vocab_growth.models.definitions import JointModelDefinition
from vocab_growth.models.diagnostics_utils import capped_plot_var_names
from vocab_growth.plotting import _save_csv
from vocab_growth.reporting import (
    config_table,
    console,
    dataframe_table,
    heading,
    key_value_table,
    pipeline_summary,
    run_banner,
    section,
)

EPSILON = math_constants.EPSILON

# uk_02 is the only source with the four-cell sign/speech cross-tabulation.
UK02_STUDY_ID = "uk_02"

# Order of the four mutually-exclusive within-understood cells.
CELL_NAMES = ["neither", "sign_only", "speak_only", "both"]


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class JointModelConfiguration(BaseModelConfiguration):
    """Configuration for the joint sign/speech modality model (VG15)."""

    # Understood (U) trajectory priors
    p_slope_low_u_dist: Continuous
    p_slope_hi_u_dist: Continuous
    ell_unit_u_dist: Continuous
    eta_u_dist: Continuous

    # Speak-given-understood ratio (q) priors
    p_slope_low_q_dist: Continuous
    p_slope_hi_q_dist: Continuous
    ell_unit_q_dist: Continuous
    eta_q_dist: Continuous

    # Sign-given-understood ratio (r) priors — intercept-only mean (no age slope)
    intercept_sign_dist: Continuous
    ell_unit_sign_dist: Continuous
    eta_sign_dist: Continuous

    # Kappa priors (age-varying dispersion) — understood / spoken / signed
    kappa_min_u_dist: Continuous
    a_kappa_u_dist: Continuous
    b_kappa_mag_u_dist: Continuous
    kappa_min_s_dist: Continuous
    a_kappa_s_dist: Continuous
    b_kappa_mag_s_dist: Continuous
    kappa_min_sign_dist: Continuous
    a_kappa_sign_dist: Continuous
    b_kappa_mag_sign_dist: Continuous

    # Association (Plackett log odds-ratio) and Dirichlet-Multinomial concentration
    log_psi_dist: Continuous
    log_conc_dist: Continuous

    # Study random-intercept scales
    tau_u_sigma: float
    tau_q_sigma: float
    tau_sign_sigma: float


@dataclass
class JointModelSamples:
    """Posterior and predictive samples from the joint model (population level)."""

    X_plot: np.ndarray
    X_query: np.ndarray

    # Population trajectories (no study effect), plot grid
    p_u_plot: np.ndarray
    q_plot: np.ndarray
    r_plot: np.ndarray
    pi_both_plot: np.ndarray
    p_any_plot: np.ndarray
    p_any_indep_plot: np.ndarray
    # Four-cell composition (fractions of understood), plot grid
    pi_neither_plot: np.ndarray
    pi_sign_only_plot: np.ndarray
    pi_speak_only_plot: np.ndarray

    # Query grid
    p_u_query: np.ndarray
    q_query: np.ndarray
    r_query: np.ndarray
    p_any_query: np.ndarray
    p_any_indep_query: np.ndarray

    # Association scalar
    psi: np.ndarray  # shape (n_samples,)

    # uk_02 four-cell posterior predictive (counts) and observed
    cell_obs: np.ndarray  # (n_cells_obs, 4) observed
    cell_pred: np.ndarray  # (n_cells_obs, 4, n_samples) predicted
    cell_ages: np.ndarray  # (n_cells_obs,)


JointContext = ModelFitContext[JointModelConfiguration, JointModelSamples]


# ============================================================
# Data preparation
# ============================================================


def _load_uk02_four_cell():
    """Load uk_02 rows, split into four-cell (cross-tab) and marginal-only rows.

    Returns (four_cell_df, marginal_df). The four-cell rows are those whose
    signed and spoken margins reconcile with the cross-tab cells
    (signed == signed_only + signed_spoken, spoken == spoken_only + signed_spoken)
    and whose four cells sum to a positive total; they identify psi. The rest
    are marginal-only uk_02 rows (no usable cross-tab).
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_uk_02.csv")
    raw = pd.read_csv(path)
    cells = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
    raw["cell_total"] = raw[cells].sum(axis=1)
    reconciles = (
        (raw["signed"] == raw["signed_only"] + raw["signed_spoken"])
        & (raw["spoken"] == raw["spoken_only"] + raw["signed_spoken"])
        & (raw["cell_total"] > 0)
    )
    four = raw[reconciles].copy()
    marg = raw[~reconciles].copy()
    return four, marg


def prepare_joint_data(
    context: JointContext,
    definition: JointModelDefinition,
):
    """Load and prepare data for the joint model.

    Non-uk_02 DS studies contribute understood/spoken/signed marginals (from the
    merged view). uk_02 is taken from its raw CSV and split into four-cell rows
    (Dirichlet-Multinomial) and marginal-only rows (marginal likelihoods).
    """
    # Subject random intercepts (issue #59) need a per-child identifier in both
    # data sources (the merged view and the raw uk_02 cross-tab CSV).
    use_subject_codes = (
        definition.use_subject_re_u
        or definition.use_subject_re_q
        or definition.use_subject_re_sign
    )
    merged_columns = ["study", "age", "understood", "spoken", "signed"]
    if use_subject_codes:
        merged_columns = merged_columns + ["subject_id"]

    merged = vocab_data_utils.load_data(
        population=definition.population,
        columns=merged_columns,
    )
    other = merged[merged["study"] != UK02_STUDY_ID].copy()

    four, marg = _load_uk02_four_cell()
    include_uk06 = definition.include_uk06

    # uk_02 four-cell rows: understood for the U likelihood; cells for the DM;
    # marginal spoken/signed set NaN (subsumed by the DM, avoids double counting).
    four_cols = {
        "study": UK02_STUDY_ID,
        "age": four["age"].to_numpy(dtype=float),
        "understood": four["comprehension"].to_numpy(dtype=float),
        "spoken": np.nan,
        "signed": np.nan,
        "understood_only": four["understood_only"].to_numpy(dtype=float),
        "signed_only": four["signed_only"].to_numpy(dtype=float),
        "spoken_only": four["spoken_only"].to_numpy(dtype=float),
        "signed_spoken": four["signed_spoken"].to_numpy(dtype=float),
        "cell_total": four["cell_total"].to_numpy(dtype=float),
    }

    # uk_02 marginal-only rows: ordinary marginals.
    marg_cols = {
        "study": UK02_STUDY_ID,
        "age": marg["age"].to_numpy(dtype=float),
        "understood": marg["comprehension"].to_numpy(dtype=float),
        "spoken": marg["spoken"].to_numpy(dtype=float),
        "signed": marg["signed"].to_numpy(dtype=float),
    }
    if use_subject_codes:
        four_cols["subject_id"] = four["subject_id"].to_numpy()
        marg_cols["subject_id"] = marg["subject_id"].to_numpy()

    four_df = pd.DataFrame(four_cols)
    marg_df = pd.DataFrame(marg_cols)

    analysis_df = pd.concat([other, marg_df, four_df], ignore_index=True)
    analysis_df = analysis_df.dropna(subset=["age"]).reset_index(drop=True)

    # Drop uk_06 signed unless included (its understood/spoken stay).
    if not include_uk06:
        m = (analysis_df["study"] == "uk_06") & analysis_df["signed"].notna()
        analysis_df.loc[m, "signed"] = np.nan

    # Integer study codes (sorted for stability).
    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    # Integer subject codes, namespaced by study so identifiers never collide.
    n_subjects: int | None = None
    if use_subject_codes:
        subj_keys = (
            analysis_df["study"].astype(str) + "::" + analysis_df["subject_id"].astype(str)
        )
        analysis_df["subject_key"] = subj_keys
        unique_subjects = sorted(subj_keys.unique())
        subject_map = {s: i for i, s in enumerate(unique_subjects)}
        analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
        n_subjects = len(unique_subjects)

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_sign = int(analysis_df["signed"].notna().sum())
    n_cells = int(analysis_df["signed_spoken"].notna().sum())

    counts: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Studies", f"{len(unique_studies)} ({', '.join(unique_studies)})"),
        ("Understood observed", n_u),
        ("Spoken observed (marginal)", n_s),
        ("Signed observed (marginal)", n_sign),
        ("uk_02 four-cell rows (DM)", n_cells),
        ("include_uk06", include_uk06),
    ]
    if n_subjects is not None:
        n_singletons = int((analysis_df.groupby("subject_code").size() == 1).sum())
        # Subjects contributing at least one signed observation (the modality that
        # most stresses subject-RE identification).
        sign_subj = analysis_df.loc[analysis_df["signed"].notna(), "subject_code"]
        n_sign_rep = int((sign_subj.value_counts() > 1).sum())
        counts.append(("Subjects", n_subjects))
        counts.append(("Subjects with single observation", n_singletons))
        counts.append(("Subjects with repeated observations", n_subjects - n_singletons))
        counts.append(("Subjects with repeated SIGNED observations", n_sign_rep))
    key_value_table("Observation counts", counts)

    desc = descriptive_stats.describe_all(
        analysis_df[["age", "understood", "spoken", "signed"]], alpha=0.05
    )
    dataframe_table(desc, title="Descriptive statistics")

    X_obs = np.asarray(analysis_df["age"], dtype=float).reshape(-1, 1)
    y_u_valid = analysis_df.loc[analysis_df["understood"].notna(), "understood"]
    y_obs_placeholder = np.zeros(n, dtype=int)
    y_obs_placeholder[analysis_df["understood"].notna().values] = y_u_valid.values.astype(
        int
    )
    bmd = model_data.BinomialModelData(
        X_obs=X_obs, y_obs=y_obs_placeholder, n_trials=definition.n_trials
    )

    context.set_model_data(bmd, analysis_df)
    context.dataframes["descriptive_stats"] = desc
    desc.to_csv(
        os.path.join(context.reporting.output_dir, "descriptive_statistics.csv"),
        index=True,
    )


# ============================================================
# Prior configuration
# ============================================================


def configure_joint_priors(context: JointContext, definition: JointModelDefinition):
    """Configure priors from the joint model definition."""

    def beta(a, b, name):
        d = pz.Beta(alpha=a, beta=b)
        _plot_and_print_dist(context, d, name)
        return d

    def halfnormal(sigma, name):
        d = pz.HalfNormal(sigma=sigma)
        _plot_and_print_dist(context, d, name)
        return d

    heading("Understood trajectory priors", style="bold cyan")
    ell_unit_u_dist = beta(definition.ell_unit_u_alpha, definition.ell_unit_u_beta, "ell_unit_u_dist")
    eta_u_dist = halfnormal(definition.eta_u_sigma, "eta_u_dist")
    p_slope_low_u_dist = beta(definition.p_slope_low_u_alpha, definition.p_slope_low_u_beta, "p_slope_low_u_dist")
    p_slope_hi_u_dist = beta(definition.p_slope_hi_u_alpha, definition.p_slope_hi_u_beta, "p_slope_hi_u_dist")

    heading("Speak-given-understood (q) priors", style="bold cyan")
    ell_unit_q_dist = beta(definition.ell_unit_q_alpha, definition.ell_unit_q_beta, "ell_unit_q_dist")
    eta_q_dist = halfnormal(definition.eta_q_sigma, "eta_q_dist")
    p_slope_low_q_dist = beta(definition.p_slope_low_q_alpha, definition.p_slope_low_q_beta, "p_slope_low_q_dist")
    p_slope_hi_q_dist = beta(definition.p_slope_hi_q_alpha, definition.p_slope_hi_q_beta, "p_slope_hi_q_dist")

    heading("Sign-given-understood (r) priors", style="bold cyan")
    ell_unit_sign_dist = beta(definition.ell_unit_sign_alpha, definition.ell_unit_sign_beta, "ell_unit_sign_dist")
    eta_sign_dist = halfnormal(definition.eta_sign_sigma, "eta_sign_dist")
    # Intercept-only signed mean (no age slope): one weakly-informative intercept
    # on the logit scale; the study REs carry between-study level, the GP the hump.
    intercept_sign_dist = pz.Normal(
        mu=definition.intercept_sign_mu, sigma=definition.intercept_sign_sigma
    )
    _plot_and_print_dist(context, intercept_sign_dist, "intercept_sign_dist")

    def kappa_block(kp, suffix):
        heading(f"Kappa priors — {suffix}", style="bold cyan")
        kmin = pz.LogNormal(mu=kp.kappa_min_mu, sigma=kp.kappa_min_sigma)
        _plot_and_print_dist(context, kmin, f"kappa_min_{suffix}_dist")
        a = pz.Normal(mu=kp.a_kappa_mu, sigma=kp.a_kappa_sigma)
        _plot_and_print_dist(context, a, f"a_kappa_{suffix}_dist")
        b = pz.HalfNormal(sigma=kp.b_kappa_mag_sigma)
        _plot_and_print_dist(context, b, f"b_kappa_mag_{suffix}_dist")
        return kmin, a, b

    kappa_min_u_dist, a_kappa_u_dist, b_kappa_mag_u_dist = kappa_block(definition.kappa_u, "u")
    kappa_min_s_dist, a_kappa_s_dist, b_kappa_mag_s_dist = kappa_block(definition.kappa_s, "s")
    kappa_min_sign_dist, a_kappa_sign_dist, b_kappa_mag_sign_dist = kappa_block(definition.kappa_sign, "sign")

    heading("Association (psi) and Dirichlet-Multinomial concentration", style="bold cyan")
    log_psi_dist = pz.Normal(mu=definition.log_psi_mu, sigma=definition.log_psi_sigma)
    _plot_and_print_dist(context, log_psi_dist, "log_psi_dist")
    log_conc_dist = pz.Normal(mu=definition.log_conc_mu, sigma=definition.log_conc_sigma)
    _plot_and_print_dist(context, log_conc_dist, "log_conc_dist")

    config = JointModelConfiguration(
        slope_anchors=definition.slope_anchors,
        ell_months_range=definition.ell_months_range,
        n_plot=definition.n_plot,
        ages_query=definition.ages_query,
        p_slope_low_u_dist=p_slope_low_u_dist,
        p_slope_hi_u_dist=p_slope_hi_u_dist,
        ell_unit_u_dist=ell_unit_u_dist,
        eta_u_dist=eta_u_dist,
        p_slope_low_q_dist=p_slope_low_q_dist,
        p_slope_hi_q_dist=p_slope_hi_q_dist,
        ell_unit_q_dist=ell_unit_q_dist,
        eta_q_dist=eta_q_dist,
        intercept_sign_dist=intercept_sign_dist,
        ell_unit_sign_dist=ell_unit_sign_dist,
        eta_sign_dist=eta_sign_dist,
        kappa_min_u_dist=kappa_min_u_dist,
        a_kappa_u_dist=a_kappa_u_dist,
        b_kappa_mag_u_dist=b_kappa_mag_u_dist,
        kappa_min_s_dist=kappa_min_s_dist,
        a_kappa_s_dist=a_kappa_s_dist,
        b_kappa_mag_s_dist=b_kappa_mag_s_dist,
        kappa_min_sign_dist=kappa_min_sign_dist,
        a_kappa_sign_dist=a_kappa_sign_dist,
        b_kappa_mag_sign_dist=b_kappa_mag_sign_dist,
        log_psi_dist=log_psi_dist,
        log_conc_dist=log_conc_dist,
        tau_u_sigma=definition.tau_u_sigma,
        tau_q_sigma=definition.tau_q_sigma,
        tau_sign_sigma=definition.tau_sign_sigma,
    )
    context.set_model_config(config)


# ============================================================
# Plackett association helper (PyTensor)
# ============================================================


def _plackett_pi_both(r, q, psi):
    """P(both | understood) under a Plackett copula with odds ratio psi.

    Closed-form root, falling back to independence at psi == 1, then clipped to
    the Frechet bounds [max(0, r+q-1), min(r, q)].
    """
    S = 1.0 + (r + q) * (psi - 1.0)
    disc = pm.math.sqrt(pm.math.maximum(S * S - 4.0 * psi * (psi - 1.0) * r * q, 0.0))
    denom = 2.0 * (psi - 1.0)
    pi_root = (S - disc) / denom
    pi_both = pm.math.switch(abs(psi - 1.0) < 1e-6, r * q, pi_root)
    lo = pm.math.maximum(0.0, r + q - 1.0)
    hi = pm.math.minimum(r, q)
    return pm.math.clip(pi_both, lo, hi)


# ============================================================
# Model building
# ============================================================


def build_model(context: JointContext, definition: JointModelDefinition):
    """Build the joint sign/speech PyMC model with study + subject random intercepts."""
    config = context.model_config
    df = context.analysis_df
    n_trials = context.model_data.n_trials

    has_u = df["understood"].notna().values
    has_s = df["spoken"].notna().values
    has_sign = df["signed"].notna().values
    has_cells = df["signed_spoken"].notna().values

    # Optional held-out rows (K-fold LOSO): kept in obs space so their latents are
    # still computed, but excluded from every likelihood. A held-out subject's RE
    # offset is then drawn from the population prior. No holdout column => standard
    # fit (the posterior-predictive plot/query grids are population-level either way).
    if "holdout" in df.columns:
        holdout = df["holdout"].fillna(False).astype(bool).values
    else:
        holdout = np.zeros(len(df), dtype=bool)
    has_u_t = has_u & ~holdout
    has_s_t = has_s & ~holdout
    has_sign_t = has_sign & ~holdout
    has_cells_t = has_cells & ~holdout

    idx_u = np.where(has_u_t)[0]
    idx_s = np.where(has_s_t)[0]
    idx_sign = np.where(has_sign_t)[0]
    idx_cells = np.where(has_cells_t)[0]

    y_u = np.asarray(df.loc[has_u_t, "understood"], dtype=int)
    y_s = np.asarray(df.loc[has_s_t, "spoken"], dtype=int)
    y_sign = np.asarray(df.loc[has_sign_t, "signed"], dtype=int)

    cell_counts = np.asarray(
        df.loc[has_cells_t, ["understood_only", "signed_only", "spoken_only", "signed_spoken"]],
        dtype=int,
    )
    cell_total = np.asarray(df.loc[has_cells_t, "cell_total"], dtype=int)

    # Validate
    for arr, nm in [(y_u, "understood"), (y_s, "spoken"), (y_sign, "signed")]:
        if arr.size and (arr.min() < 0 or arr.max() > n_trials):
            raise ValueError(f"{nm} outside [0, n_trials].")
    if cell_counts.size:
        if cell_counts.min() < 0:
            raise ValueError("negative four-cell count.")
        if not np.array_equal(cell_counts.sum(axis=1), cell_total):
            raise ValueError("four-cell counts do not sum to cell_total.")
        if cell_total.max() > n_trials:
            raise ValueError(f"four-cell total exceeds n_trials ({n_trials}).")

    study_codes = np.asarray(df["study_code"], dtype=int)
    n_studies = int(study_codes.max()) + 1

    use_subject_re_u = bool(definition.use_subject_re_u)
    use_subject_re_q = bool(definition.use_subject_re_q)
    use_subject_re_sign = bool(definition.use_subject_re_sign)
    use_subject_codes = use_subject_re_u or use_subject_re_q or use_subject_re_sign
    if use_subject_codes:
        subject_codes = np.asarray(df["subject_code"], dtype=int)
        n_subjects = int(subject_codes.max()) + 1
    else:
        subject_codes = None
        n_subjects = 0

    X_obs = np.asarray(df["age"], dtype=float).reshape(-1, 1)
    n = len(X_obs)
    X_mean = float(np.mean(X_obs))
    X_std = float(np.std(X_obs, ddof=1))

    X_obs_z = (X_obs - X_mean) / X_std
    X_plot = np.linspace(X_obs.min(), X_obs.max(), config.n_plot).reshape(-1, 1)
    X_plot_z = (X_plot - X_mean) / X_std
    X_query = np.array(config.ages_query).reshape(-1, 1)
    X_query_z = (X_query - X_mean) / X_std
    n_plot = X_plot_z.shape[0]
    n_query = X_query_z.shape[0]

    # Option D — per-draw GP anchor at a reference age. Append one extra grid row
    # (the reference age) so each anchored GP can be centred to pass through zero
    # there for every draw, removing the GP<->intercept level redundancy.
    anchor_g_u = bool(definition.anchor_g_u_at_ref)
    anchor_g_q = bool(definition.anchor_g_q_at_ref)
    anchor_g_sign = bool(definition.anchor_g_sign_at_ref)
    use_gp_anchor = anchor_g_u or anchor_g_q or anchor_g_sign
    if use_gp_anchor:
        if definition.gp_anchor_age_months is not None:
            anchor_age_months = float(definition.gp_anchor_age_months)
        else:
            anchor_age_months = (
                float(config.slope_anchors[0]) + float(config.slope_anchors[1])
            ) / 2.0
        X_anchor_z = (np.array([[anchor_age_months]], dtype=float) - X_mean) / X_std
        X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z, X_anchor_z])
        i_anchor = n + n_plot + n_query
    else:
        anchor_age_months = None
        X_all_z = np.vstack([X_obs_z, X_plot_z, X_query_z])
        i_anchor = None
    n_all = X_all_z.shape[0]

    ell_low_z = float(config.ell_months_range[0]) / X_std
    ell_high_z = float(config.ell_months_range[1]) / X_std
    L, M = get_hsgp_hyperparams(X_obs_z, (ell_low_z, ell_high_z))

    sa = float(config.slope_anchors[0])
    sb = float(config.slope_anchors[1])
    sa_z = (sa - X_mean) / X_std
    sb_z = (sb - X_mean) / X_std

    build_cfg: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Studies", n_studies),
        ("Understood / spoken / signed / cells", f"{len(idx_u)} / {len(idx_s)} / {len(idx_sign)} / {len(idx_cells)}"),
        ("n_trials", n_trials),
        ("Age mean / std", (round(X_mean, 1), round(X_std, 1))),
        ("HSGP m / L", (M, L)),
    ]
    if use_subject_codes:
        build_cfg.append(
            ("Subject REs (u/q/sign)", f"{use_subject_re_u} / {use_subject_re_q} / {use_subject_re_sign}")
        )
        build_cfg.append(("n_subjects", n_subjects))
    if use_gp_anchor:
        build_cfg.append(
            ("GP anchor age (months)", f"{anchor_age_months:g} (u={anchor_g_u}, q={anchor_g_q}, sign={anchor_g_sign})")
        )
    key_value_table("Build configuration", build_cfg)

    i_obs0, i_obs1 = 0, n
    i_plot0, i_plot1 = n, n + n_plot
    i_query0, i_query1 = n + n_plot, n + n_plot + n_query

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
        "obs_u_id": np.arange(len(idx_u)),
        "obs_s_id": np.arange(len(idx_s)),
        "obs_sign_id": np.arange(len(idx_sign)),
        "obs_cells_id": np.arange(len(idx_cells)),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "study_id": np.arange(n_studies),
        "cell_id": CELL_NAMES,
        "x_dim": np.arange(1),
    }
    if use_subject_codes:
        coords["subject_id"] = np.arange(n_subjects)

    def trend_and_gp(cfg_low, cfg_hi, cfg_ell, cfg_eta, suffix, X_all_z_data, anchor_idx=None):
        """Build a logit-linear trend + HSGP deviation; return the full-grid latent.

        If ``anchor_idx`` is given (Option D), the GP is centred to pass through
        zero at that grid row for every draw, so the linear trend alone sets the
        level there and the GP only carries deviations.
        """
        p_lo = cfg_low.to_pymc(f"p_slope_low_{suffix}")
        p_hi = cfg_hi.to_pymc(f"p_slope_hi_{suffix}")
        slope = pm.Deterministic(
            f"slope_{suffix}",
            (pymc_utils.logit(p_hi) - pymc_utils.logit(p_lo)) / (sb_z - sa_z),
        )
        intercept = pm.Deterministic(
            f"intercept_{suffix}", pymc_utils.logit(p_lo) - slope * sa_z
        )
        mean_trend = intercept + slope * X_all_z_data[:, 0]
        ell_unit = cfg_ell.to_pymc(f"ell_unit_{suffix}")
        ell = pm.Deterministic(
            f"ell_{suffix}", ell_low_z + (ell_high_z - ell_low_z) * ell_unit
        )
        eta = cfg_eta.to_pymc(f"eta_{suffix}")
        cov = pm.gp.cov.ExpQuad(1, ls=ell)
        hsgp = pm.gp.HSGP(cov_func=cov, m=M, L=L)
        g_unit = hsgp.prior(f"g_unit_{suffix}", X=X_all_z_data, dims="all_id")
        if anchor_idx is not None:
            g_unit = g_unit - g_unit[anchor_idx]
        return mean_trend + eta * g_unit  # plain tensor (n_all,)

    def intercept_and_gp(cfg_intercept, cfg_ell, cfg_eta, suffix, X_all_z_data, anchor_idx=None):
        """Intercept-only mean (no age slope) + HSGP deviation; full-grid latent.

        Used for the signed ratio: a free age slope would extrapolate the ratio
        below the data floor (< ~18 mo), so the mean is intercept-only and the GP
        carries the age-varying (rise-then-fall) shape. The study random
        intercept is added at observation level by the caller. If ``anchor_idx``
        is given (Option D), the GP passes through zero at that grid row per draw.
        """
        intercept = cfg_intercept.to_pymc(f"intercept_{suffix}")
        ell_unit = cfg_ell.to_pymc(f"ell_unit_{suffix}")
        ell = pm.Deterministic(
            f"ell_{suffix}", ell_low_z + (ell_high_z - ell_low_z) * ell_unit
        )
        eta = cfg_eta.to_pymc(f"eta_{suffix}")
        cov = pm.gp.cov.ExpQuad(1, ls=ell)
        hsgp = pm.gp.HSGP(cov_func=cov, m=M, L=L)
        g_unit = hsgp.prior(f"g_unit_{suffix}", X=X_all_z_data, dims="all_id")
        if anchor_idx is not None:
            g_unit = g_unit - g_unit[anchor_idx]
        return intercept + eta * g_unit  # plain tensor (n_all,)

    with pm.Model(coords=coords) as model_pm:
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))
        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))
        if use_subject_codes:
            subject_obs = pm.Data("subject_obs", subject_codes, dims=("obs_id",))

        # Latent full-grid trajectories (plain tensors). Option D anchors each GP
        # (per-draw zero at the reference age) when the matching flag is set.
        f_u_all = trend_and_gp(
            config.p_slope_low_u_dist, config.p_slope_hi_u_dist,
            config.ell_unit_u_dist, config.eta_u_dist, "u", X_all_z_data,
            anchor_idx=i_anchor if anchor_g_u else None,
        )
        h_all = trend_and_gp(
            config.p_slope_low_q_dist, config.p_slope_hi_q_dist,
            config.ell_unit_q_dist, config.eta_q_dist, "q", X_all_z_data,
            anchor_idx=i_anchor if anchor_g_q else None,
        )
        # Signed marginal: intercept-only mean (no age slope) + GP hump; the
        # study random intercept delta_sign is added at obs level below.
        g_all = intercept_and_gp(
            config.intercept_sign_dist,
            config.ell_unit_sign_dist, config.eta_sign_dist, "sign", X_all_z_data,
            anchor_idx=i_anchor if anchor_g_sign else None,
        )

        # Study random intercepts (non-centred), applied at obs level only.
        tau_u = pm.HalfNormal("tau_u", sigma=config.tau_u_sigma)
        tau_q = pm.HalfNormal("tau_q", sigma=config.tau_q_sigma)
        tau_sign = pm.HalfNormal("tau_sign", sigma=config.tau_sign_sigma)
        delta_u = pm.Deterministic(
            "delta_u", tau_u * pm.Normal("z_u", 0.0, 1.0, dims="study_id"), dims="study_id"
        )
        delta_q = pm.Deterministic(
            "delta_q", tau_q * pm.Normal("z_q", 0.0, 1.0, dims="study_id"), dims="study_id"
        )
        delta_sign = pm.Deterministic(
            "delta_sign", tau_sign * pm.Normal("z_sign", 0.0, 1.0, dims="study_id"), dims="study_id"
        )

        # Subject random intercepts (non-centred), applied at obs level only. Each
        # is gated by its flag so the sign-RE can be dropped via config alone.
        def subject_shift(flag, tau_sigma, suffix):
            if not flag:
                return 0.0
            tau = pm.HalfNormal(f"tau_subj_{suffix}", sigma=tau_sigma)
            z = pm.Normal(f"z_subj_{suffix}", 0.0, 1.0, dims="subject_id")
            delta = pm.Deterministic(f"delta_subj_{suffix}", tau * z, dims="subject_id")
            return delta[subject_obs]

        subject_shift_u = subject_shift(use_subject_re_u, definition.tau_subj_u_sigma, "u")
        subject_shift_q = subject_shift(use_subject_re_q, definition.tau_subj_q_sigma, "q")
        subject_shift_sign = subject_shift(use_subject_re_sign, definition.tau_subj_sign_sigma, "sign")

        # Standardised observed ages (used by the age-varying kappa functions).
        z_obs = pm.Deterministic("z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims="obs_id")

        # --- obs-level latents WITH study + subject shifts (marginal likelihoods) ---
        f_u_obs = f_u_all[i_obs0:i_obs1] + delta_u[study_obs] + subject_shift_u
        h_obs = h_all[i_obs0:i_obs1] + delta_q[study_obs] + subject_shift_q
        g_obs = g_all[i_obs0:i_obs1] + delta_sign[study_obs] + subject_shift_sign
        p_u_obs = pm.math.sigmoid(f_u_obs)
        q_obs = pm.math.sigmoid(h_obs)
        r_obs = pm.math.sigmoid(g_obs)

        # --- population+study marginals (NO subject shift) for the four-cell DM ---
        # psi is identified from the ~62 uk_02 cross-tab rows (34 children, ~2 rows
        # each). The per-child sign offset is co-identified with psi from those same
        # rows, so letting it into the four-cell composition makes psi pivot on a
        # thinly-identified RE (psi 1.78 -> ~2.8; the move is driven almost entirely
        # by the sign subject RE — see notes/202606171200-vg15-subject-re-stabilisation).
        # We therefore keep psi a *population-conditioned* within-understood
        # association (comparable to the study-RE-only VG15) by feeding the DM the
        # study-level marginals only; subject REs still enter every marginal
        # likelihood. When subject REs are off these equal r_obs/q_obs exactly.
        q_obs_pop = pm.math.sigmoid(h_all[i_obs0:i_obs1] + delta_q[study_obs])
        r_obs_pop = pm.math.sigmoid(g_all[i_obs0:i_obs1] + delta_sign[study_obs])

        # --- population-level latents (no study shift), plot + query ---
        p_u_plot = pm.Deterministic("p_u_plot", pm.math.sigmoid(f_u_all[i_plot0:i_plot1]), dims="plot_id")
        q_plot = pm.Deterministic("q_plot", pm.math.sigmoid(h_all[i_plot0:i_plot1]), dims="plot_id")
        r_plot = pm.Deterministic("r_plot", pm.math.sigmoid(g_all[i_plot0:i_plot1]), dims="plot_id")
        p_u_query = pm.Deterministic("p_u_query", pm.math.sigmoid(f_u_all[i_query0:i_query1]), dims="query_id")
        q_query = pm.Deterministic("q_query", pm.math.sigmoid(h_all[i_query0:i_query1]), dims="query_id")
        r_query = pm.Deterministic("r_query", pm.math.sigmoid(g_all[i_query0:i_query1]), dims="query_id")

        # --- association ---
        log_psi = config.log_psi_dist.to_pymc("log_psi")
        psi = pm.Deterministic("psi", pm.math.exp(log_psi))
        log_conc = config.log_conc_dist.to_pymc("log_conc")
        conc = pm.Deterministic("conc", pm.math.exp(log_conc))

        # --- kappa functions ---
        def kappa_fn(kmin_d, a_d, bmag_d, suffix):
            kmin = kmin_d.to_pymc(f"kappa_min_{suffix}")
            a = a_d.to_pymc(f"a_kappa_{suffix}")
            bmag = bmag_d.to_pymc(f"b_kappa_mag_{suffix}")
            b = pm.Deterministic(f"b_kappa_{suffix}", -bmag)
            return lambda z: kmin + pm.math.exp(a + b * z)

        kappa_u_of_z = kappa_fn(config.kappa_min_u_dist, config.a_kappa_u_dist, config.b_kappa_mag_u_dist, "u")
        kappa_s_of_z = kappa_fn(config.kappa_min_s_dist, config.a_kappa_s_dist, config.b_kappa_mag_s_dist, "s")
        kappa_sign_of_z = kappa_fn(config.kappa_min_sign_dist, config.a_kappa_sign_dist, config.b_kappa_mag_sign_dist, "sign")

        kappa_u_obs = kappa_u_of_z(z_obs)
        kappa_s_obs = kappa_s_of_z(z_obs)
        kappa_sign_obs = kappa_sign_of_z(z_obs)

        # ============================================================
        # Likelihoods
        # ============================================================
        # Understood (all studies)
        p_u_sel = pm.math.clip(p_u_obs[idx_u], EPSILON, 1 - EPSILON)
        k_u = kappa_u_obs[idx_u]
        pm.BetaBinomial("y_u_obs", n=n_trials, alpha=p_u_sel * k_u, beta=(1 - p_u_sel) * k_u,
                        observed=y_u, dims="obs_u_id")

        # Spoken marginal (p_U * q)
        p_s_sel = pm.math.clip((p_u_obs * q_obs)[idx_s], EPSILON, 1 - EPSILON)
        k_s = kappa_s_obs[idx_s]
        pm.BetaBinomial("y_s_obs", n=n_trials, alpha=p_s_sel * k_s, beta=(1 - p_s_sel) * k_s,
                        observed=y_s, dims="obs_s_id")

        # Signed marginal (p_U * r)
        p_sign_sel = pm.math.clip((p_u_obs * r_obs)[idx_sign], EPSILON, 1 - EPSILON)
        k_sign = kappa_sign_obs[idx_sign]
        pm.BetaBinomial("y_sign_obs", n=n_trials, alpha=p_sign_sel * k_sign, beta=(1 - p_sign_sel) * k_sign,
                        observed=y_sign, dims="obs_sign_id")

        # uk_02 four cells (Dirichlet-Multinomial), within-understood composition.
        # Uses population+study marginals (r_obs_pop/q_obs_pop), so psi stays a
        # population-conditioned association decoupled from the per-child sign RE.
        r_c = pm.math.clip(r_obs_pop[idx_cells], EPSILON, 1 - EPSILON)
        q_c = pm.math.clip(q_obs_pop[idx_cells], EPSILON, 1 - EPSILON)
        pi_both_c = _plackett_pi_both(r_c, q_c, psi)
        pi_sign_c = pm.math.maximum(r_c - pi_both_c, EPSILON)
        pi_speak_c = pm.math.maximum(q_c - pi_both_c, EPSILON)
        pi_neither_c = pm.math.maximum(1 - r_c - q_c + pi_both_c, EPSILON)
        pi_both_c = pm.math.maximum(pi_both_c, EPSILON)
        pi_stack = pm.math.stack([pi_neither_c, pi_sign_c, pi_speak_c, pi_both_c], axis=1)
        pi_stack = pi_stack / pi_stack.sum(axis=1, keepdims=True)
        pm.DirichletMultinomial(
            "cells_obs", n=cell_total, a=conc * pi_stack, observed=cell_counts,
            dims=("obs_cells_id", "cell_id"),
        )

        # ============================================================
        # Reporting deterministics (population, plot/query): four-cell + p_any
        # ============================================================
        for grid, rg, qg, pug in [("plot", r_plot, q_plot, p_u_plot), ("query", r_query, q_query, p_u_query)]:
            pi_both = _plackett_pi_both(rg, qg, psi)
            pm.Deterministic(f"pi_both_{grid}", pi_both, dims=f"{grid}_id")
            pm.Deterministic(f"pi_sign_only_{grid}", pm.math.maximum(rg - pi_both, 0.0), dims=f"{grid}_id")
            pm.Deterministic(f"pi_speak_only_{grid}", pm.math.maximum(qg - pi_both, 0.0), dims=f"{grid}_id")
            pm.Deterministic(f"pi_neither_{grid}", pm.math.maximum(1 - rg - qg + pi_both, 0.0), dims=f"{grid}_id")
            # data-identified union and independence union (out of understood)
            pm.Deterministic(f"p_any_{grid}", pug * (rg + qg - pi_both), dims=f"{grid}_id")
            pm.Deterministic(f"p_any_indep_{grid}", pug * (1 - (1 - rg) * (1 - qg)), dims=f"{grid}_id")

        # kappa for plot grid (reporting)
        pm.Deterministic("kappa_sign_obs", kappa_sign_obs, dims="obs_id")

    pymc_utils.report_model_summary(model_pm)
    variables = pymc_utils.get_variables_dict(model_pm)
    try:
        digraph = pymc_utils.model_to_graphviz(model_pm)
        digraph.render(
            filename=os.path.join(context.reporting.output_dir, "gp_model_graph"),
            format="svg", cleanup=True,
        )
    except Exception as exc:  # graphviz 'dot' not on PATH — non-fatal
        console.print(f"[yellow]Skipped model graph: {exc}[/yellow]")

    context.set_model(model_pm, variables)


# ============================================================
# Pipeline
# ============================================================


def prior_predictive_checks(context: JointContext):
    """Light prior predictive check: r(a), q(a), psi spans independence."""
    with context.model:
        prior = pm.sample_prior_predictive(
            draws=500, random_seed=context.sampling.random_seed,
            compile_kwargs=dict(mode="FAST_COMPILE"),
        )
    context.set_prior_samples(prior)
    for var, ylab, fname in [("r_plot", "r(a) = P(sign | understood)", "prior_samples_r"),
                             ("q_plot", "q(a) = P(speak | understood)", "prior_samples_q")]:
        s = prior.prior[var].stack(sample=("chain", "draw")).transpose("plot_id", "sample")
        fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
        Xp = prior.constant_data["X_plot"].values
        for i in range(min(400, s.shape[1])):
            ax.plot(Xp, s.values[:, i], alpha=0.02)
        ax.set_xlabel("Age (months)")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
        fig.savefig(os.path.join(context.reporting.output_dir, f"{fname}.png"), dpi=300)
        fig.savefig(os.path.join(context.reporting.output_dir, f"{fname}.svg"))
        plt.close(fig)


def sample(context: JointContext):
    config_table("Sampling configuration", context.sampling)
    with context.model:
        trace = pm.sample(
            context.sampling.draws, tune=context.sampling.tune,
            chains=context.sampling.chains, cores=context.sampling.cores,
            target_accept=context.sampling.target_accept,
            nuts_sampler="nutpie", return_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )
    context.set_trace(trace)


def diagnostics(context: JointContext):
    var_names = [v.name for v in context.model.unobserved_RVs if v.size.eval() <= 2]
    diag = az.summary(context.trace, var_names=var_names, round_to=4,
                      ci_prob=context.reporting.hdi, ci_kind="hdi")
    diag.to_csv(os.path.join(context.reporting.output_dir, "diagnostics.csv"), index=True)
    dataframe_table(diag, title="Posterior diagnostics")
    _report_diagnostic_warnings(diag)
    tv = capped_plot_var_names(context.trace, var_names + ["psi", "conc"])
    az.plot_trace(context.trace, var_names=tv)
    plt.savefig(os.path.join(context.reporting.output_dir, "trace_plot.png"), dpi=300)
    plt.close()
    az.plot_energy(context.trace, figure_kwargs={"figsize": plot_styles.FIGSIZE_SM})
    plt.savefig(os.path.join(context.reporting.output_dir, "energy_plot.png"), dpi=300)
    plt.close()


def _extract(trace, name, dim):
    return np.array(trace.posterior[name].stack(sample=("chain", "draw")).transpose(dim, "sample").values)


def sample_posterior_predictive(context: JointContext, definition=None):
    """Posterior predictive for the uk_02 four cells (for the cell PPC)."""
    with context.model:
        trace = pm.sample_posterior_predictive(
            context.trace, var_names=["cells_obs"], extend_inferencedata=True,
            random_seed=context.sampling.random_seed,
        )
    context.set_trace(trace)
    trace.to_netcdf(os.path.join(context.reporting.output_dir, "trace.nc"))

    # Observed uk_02 four-cell counts / ages (recomputed from analysis_df).
    df = context.analysis_df
    has_cells = df["signed_spoken"].notna().values
    cell_counts = np.asarray(
        df.loc[has_cells, ["understood_only", "signed_only", "spoken_only", "signed_spoken"]],
        dtype=int,
    )
    cell_ages = np.asarray(df.loc[has_cells, "age"], dtype=float)

    samples = JointModelSamples(
        X_plot=np.array(trace.constant_data["X_plot"].values),
        X_query=np.array(trace.constant_data["X_query"].values),
        p_u_plot=_extract(trace, "p_u_plot", "plot_id"),
        q_plot=_extract(trace, "q_plot", "plot_id"),
        r_plot=_extract(trace, "r_plot", "plot_id"),
        pi_both_plot=_extract(trace, "pi_both_plot", "plot_id"),
        p_any_plot=_extract(trace, "p_any_plot", "plot_id"),
        p_any_indep_plot=_extract(trace, "p_any_indep_plot", "plot_id"),
        pi_neither_plot=_extract(trace, "pi_neither_plot", "plot_id"),
        pi_sign_only_plot=_extract(trace, "pi_sign_only_plot", "plot_id"),
        pi_speak_only_plot=_extract(trace, "pi_speak_only_plot", "plot_id"),
        p_u_query=_extract(trace, "p_u_query", "query_id"),
        q_query=_extract(trace, "q_query", "query_id"),
        r_query=_extract(trace, "r_query", "query_id"),
        p_any_query=_extract(trace, "p_any_query", "query_id"),
        p_any_indep_query=_extract(trace, "p_any_indep_query", "query_id"),
        psi=np.array(trace.posterior["psi"].stack(sample=("chain", "draw")).values),
        cell_obs=cell_counts,
        cell_pred=np.array(
            trace.posterior_predictive["cells_obs"]
            .stack(sample=("chain", "draw"))
            .transpose("obs_cells_id", "cell_id", "sample")
            .values
        ),
        cell_ages=cell_ages,
    )
    context.set_model_samples(samples)


def posterior_summary(context: JointContext):
    s = context.model_samples
    n_trials = context.model_data.n_trials
    hdi = context.reporting.hdi
    od = context.reporting.output_dir

    def ratio_summary(X, draws, prefix):
        med = np.median(draws, axis=1)
        h = az.hdi(draws, prob=hdi)
        d = pd.DataFrame({"age_months": X, f"{prefix}_median": med,
                          f"{prefix}_hdi_lo": h[:, 0], f"{prefix}_hdi_hi": h[:, 1]})
        d.to_csv(os.path.join(od, f"posterior_summary_{prefix}.csv"), index=False)
        return d

    ratio_summary(s.X_query, s.r_query, "r")
    ratio_summary(s.X_query, s.q_query, "q")

    # Data-identified p_any vs independence (expected counts)
    Ey = s.p_any_query * n_trials
    Ey_i = s.p_any_indep_query * n_trials
    pany = pd.DataFrame({
        "age_months": s.X_query,
        "p_any_median": np.median(s.p_any_query, axis=1),
        "p_any_hdi_lo": az.hdi(s.p_any_query, prob=hdi)[:, 0],
        "p_any_hdi_hi": az.hdi(s.p_any_query, prob=hdi)[:, 1],
        "Ey_any_median": np.median(Ey, axis=1),
        "p_any_indep_median": np.median(s.p_any_indep_query, axis=1),
        "Ey_any_indep_median": np.median(Ey_i, axis=1),
    })
    pany.to_csv(os.path.join(od, "posterior_summary_p_any.csv"), index=False)
    dataframe_table(pany.round(3), title="Total expressive p_any (identified vs independence)", show_index=False)

    # psi summary
    psi = s.psi
    psi_df = pd.DataFrame({
        "psi_median": [float(np.median(psi))],
        "psi_hdi_lo": [float(az.hdi(psi, prob=hdi)[0])],
        "psi_hdi_hi": [float(az.hdi(psi, prob=hdi)[1])],
        "P_psi_gt_1": [float((psi > 1).mean())],
    })
    psi_df.to_csv(os.path.join(od, "posterior_summary_psi.csv"), index=False)
    key_value_table("Association psi", [
        ("psi median", round(float(np.median(psi)), 3)),
        ("psi 90% HDI", (round(float(az.hdi(psi, prob=hdi)[0]), 3), round(float(az.hdi(psi, prob=hdi)[1]), 3))),
        ("P(psi > 1)", round(float((psi > 1).mean()), 3)),
    ])


# ============================================================
# Plots
# ============================================================


def _run_joint_plots(context: JointContext):
    s = context.model_samples
    n_trials = context.model_data.n_trials
    od = context.reporting.output_dir
    hdi = context.reporting.hdi
    X = s.X_plot

    # 1) Data-identified p_any vs independence upper bound (expected counts)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    id_med = np.median(s.p_any_plot, axis=1) * n_trials
    id_hdi = az.hdi(s.p_any_plot, prob=hdi) * n_trials
    ind_med = np.median(s.p_any_indep_plot, axis=1) * n_trials
    ax.fill_between(X, id_hdi[:, 0], id_hdi[:, 1], alpha=0.20, color="C0")
    ax.plot(X, id_med, lw=3, color="C0", label="Data-identified p_any (median)")
    ax.plot(X, ind_med, lw=2.5, ls="--", color="C3",
            label="Independence upper bound (p_U·(1-(1-r)(1-q)))")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Expected words produced (any modality)")
    ax.set_ylim(0, n_trials + 50)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Total expressive vocabulary: identified vs independence bound")
    fig.savefig(os.path.join(od, "p_any_identified_vs_bound.png"), dpi=300)
    fig.savefig(os.path.join(od, "p_any_identified_vs_bound.svg"))
    _save_csv(pd.DataFrame({"age_months": X, "identified_median": id_med,
                            "identified_hdi_lo": id_hdi[:, 0], "identified_hdi_hi": id_hdi[:, 1],
                            "independence_median": ind_med}), od, "p_any_identified_vs_bound")
    context.plots["p_any_identified_vs_bound"] = fig
    plt.close(fig)

    # 2) Four-cell composition trajectory (fractions of understood)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    comp = {
        "neither": (np.median(s.pi_neither_plot, axis=1), "C7"),
        "sign-only": (np.median(s.pi_sign_only_plot, axis=1), "C2"),
        "sign+speech": (np.median(s.pi_both_plot, axis=1), "C4"),
        "speak-only": (np.median(s.pi_speak_only_plot, axis=1), "C1"),
    }
    for lab, (med, c) in comp.items():
        ax.plot(X, med, lw=2.5, color=c, label=lab)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Fraction of understood words")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Within-understood composition (sign-only → both → speak-only)")
    fig.savefig(os.path.join(od, "four_cell_composition.png"), dpi=300)
    fig.savefig(os.path.join(od, "four_cell_composition.svg"))
    _save_csv(pd.DataFrame({"age_months": X, **{k: v[0] for k, v in comp.items()}}), od, "four_cell_composition")
    context.plots["four_cell_composition"] = fig
    plt.close(fig)

    # 3) psi posterior
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    ax.hist(s.psi, bins=60, color="C4", alpha=0.8)
    ax.axvline(1.0, ls=":", color="grey", label="independence (psi=1)")
    ax.axvline(float(np.median(s.psi)), ls="-", color="C0", label=f"median {np.median(s.psi):.2f}")
    ax.set_xlabel("psi (sign-speech odds ratio)")
    ax.set_ylabel("posterior draws")
    ax.legend(frameon=True)
    ax.set_title(f"Association psi — P(psi>1) = {(s.psi > 1).mean():.2f}")
    fig.savefig(os.path.join(od, "psi_posterior.png"), dpi=300)
    fig.savefig(os.path.join(od, "psi_posterior.svg"))
    context.plots["psi_posterior"] = fig
    plt.close(fig)

    # 4) signed rate r(a) (study REs absorb composition)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    r_med = np.median(s.r_plot, axis=1)
    r_hdi = az.hdi(s.r_plot, prob=hdi)
    q_med = np.median(s.q_plot, axis=1)
    ax.fill_between(X, r_hdi[:, 0], r_hdi[:, 1], alpha=0.18, color="C2")
    ax.plot(X, r_med, lw=3, color="C2", label="r(a) signed")
    ax.plot(X, q_med, lw=3, color="C1", label="q(a) spoken")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Fraction of understood words")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Signed vs spoken ratio (population, study REs marginalised)")
    fig.savefig(os.path.join(od, "signed_vs_spoken_rate.png"), dpi=300)
    fig.savefig(os.path.join(od, "signed_vs_spoken_rate.svg"))
    context.plots["signed_vs_spoken_rate"] = fig
    plt.close(fig)

    # 5) uk_02 four-cell PPC (observed vs predicted cell totals)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    obs_tot = s.cell_obs.sum(axis=0)  # (4,)
    pred_tot = s.cell_pred.sum(axis=0)  # (4, n_samples)
    pred_med = np.median(pred_tot, axis=1)
    lo, hi = 100 * (1 - hdi) / 2, 100 * (1 + hdi) / 2
    pred_lo = np.percentile(pred_tot, lo, axis=1)
    pred_hi = np.percentile(pred_tot, hi, axis=1)
    yerr = np.vstack([pred_med - pred_lo, pred_hi - pred_med])
    xpos = np.arange(4)
    ax.bar(xpos - 0.18, obs_tot, width=0.36, color="C0", label="observed")
    ax.bar(xpos + 0.18, pred_med, width=0.36, color="C3", alpha=0.7, label="predicted (median)",
           yerr=yerr, capsize=4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(CELL_NAMES)
    ax.set_ylabel("Total cell count (uk_02)")
    ax.legend(frameon=True)
    ax.set_title("uk_02 four-cell posterior predictive check")
    fig.savefig(os.path.join(od, "uk02_cell_ppc.png"), dpi=300)
    fig.savefig(os.path.join(od, "uk02_cell_ppc.svg"))
    context.plots["uk02_cell_ppc"] = fig
    plt.close(fig)


# ============================================================
# Fit orchestration
# ============================================================


def fit_joint_model(config: str, definition: JointModelDefinition) -> JointContext:
    """Shared fit pipeline for the joint sign/speech model (VG15)."""
    run_banner(definition.banner, subtitle=f"sampling config: {config}")
    env_info.report_environment_info()
    console.print()
    package_metadata.report_package_versions(PACKAGE_LIST)

    context: JointContext = ModelFitContext(
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=local_env.OUTPUT_DIR,
            hdi=0.90,
        ),
        sampling=sampling.get_sampling_configuration(config),
    )
    if os.path.exists(context.reporting.output_dir):
        shutil.rmtree(context.reporting.output_dir)
    os.makedirs(context.reporting.output_dir, exist_ok=True)

    timings = context.timings
    run_started = time.perf_counter()
    with section("Prepare data", timings=timings):
        prepare_joint_data(context, definition)
    with section("Priors and hyperparameters", timings=timings):
        configure_joint_priors(context, definition)
    with section("Model definition and initialisation", timings=timings):
        build_model(context, definition)
    with section("Prior predictive checks", timings=timings):
        prior_predictive_checks(context)
    with section("Posterior sampling", timings=timings):
        sample(context)
    with section("Diagnostics", timings=timings):
        diagnostics(context)
    with section("Posterior predictions", timings=timings):
        sample_posterior_predictive(context, definition)
    with section("Posterior summary", timings=timings):
        posterior_summary(context)
    with section("Plots", timings=timings):
        _run_joint_plots(context)
    with section("Report", timings=timings):
        report(context)

    pipeline_summary(f"Pipeline summary — {context.reporting.model_label}", timings)
    console.print(
        f"[dim]Total wall time: {vg_reporting.format_duration(time.perf_counter() - run_started)}[/dim]"
    )
    return context
