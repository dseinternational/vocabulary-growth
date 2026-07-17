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

Latent scale (all out of N = 810 checklist words, the DSE reference inventory):

    p_U(a)  = sigmoid(f_U(a))                 # proportion understood
    r(a)    = sigmoid(g(a))                   # P(sign  | understood)
    q(a)    = sigmoid(h(a))                   # P(speak | understood)

    pi_both     = Plackett(r, q; psi)         # P(sign & speak | understood)
    pi_signonly = r - pi_both
    pi_speakonly= q - pi_both
    pi_neither  = 1 - r - q + pi_both
    p_any(a)    = p_U(a) * (r + q - pi_both)   # total expressive (data-identified)

Likelihoods use the observed understood count as the denominator for spoken
and signed outcomes when the counts are jointly available and logically
nested. Rows without a usable understood count retain a marginal likelihood.
The uk_02 rows with a four-cell cross-tabulation use that joint composition
term instead of duplicate spoken and signed likelihood contributions:
    - understood ~ BetaBinomial(810, p_U)              (all DS studies)
    - spoken | understood ~ BetaBinomial(understood, q)
    - signed | understood ~ BetaBinomial(understood, r)
    - uk_02 four cells ~ DirichletMultinomial(total, conc * [pi_*])  (the one
      item-level joint term; identifies psi)

This is a self-contained module (like common_trivariate.py); it does not import
from or modify the bivariate / trivariate engines. The full-grid intermediates
are kept as plain tensors (only obs/plot/query slices are stored), following the
VG14 memory discipline.
"""

import os
import sys
from dataclasses import dataclass

import dse_research_utils.math.constants as math_constants
import dse_research_utils.plot.styles as plot_styles
import dse_research_utils.statistics.descriptive as descriptive_stats
import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.pymc_utils as pymc_utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import preliz as pz
import pymc as pm
from preliz.distributions.distributions import Continuous

import vocab_growth.data_utils as vocab_data_utils
import vocab_growth.environment as local_env
import vocab_growth.intervals as intervals
from vocab_growth.models.build_utils import (
    construct_age_grids,
    slope_anchor_logit_coeffs,
    standardize_ages,
    validate_ell_bounds,
)
from vocab_growth.models.calibration import write_trace_calibration
from vocab_growth.models.common import (
    BaseModelConfiguration,
    ModelFitContext,
    _plot_and_print_dist,
    get_hsgp_hyperparams,
    render_model_graph,
    report,
    run_fit_pipeline,
)
from vocab_growth.models.common import diagnostics as _shared_diagnostics
from vocab_growth.models.common import sample as _shared_sample
from vocab_growth.models.definitions import JointModelDefinition
from vocab_growth.models.gp_utils import (
    GPGrid,
    build_kappa_of_z,
    tent_and_gp,
    trend_and_gp,
)
from vocab_growth.models.likelihood_utils import nested_outcome_spec
from vocab_growth.plotting import (
    _save_csv,
    plot_prior_samples,
    plot_prior_samples_ratio,
)
from vocab_growth.posterior_analysis import extract_posterior as _extract
from vocab_growth.reporting import (
    dataframe_table,
    heading,
    key_value_table,
)

EPSILON = math_constants.EPSILON

# uk_02 is the only source with the four-cell within-understood cross-tabulation.
UK02_STUDY_ID = "uk_02"
# nz_01 (Foster-Cohen) carries a production-only three-cell (within-produced)
# cross-tabulation: word-only, sign-only, both. No comprehension.
NZ01_STUDY_ID = "nz_01"

# Order of the four mutually-exclusive within-understood cells.
CELL_NAMES = ["neither", "sign_only", "speak_only", "both"]
# Order of nz_01's three within-produced cells (the four-cell composition
# conditioned on produced, dropping the unobservable "neither"/understood-only).
PROD_CELL_NAMES = ["sign_only", "speak_only", "both"]
PROD_CELL_COLUMNS = ["prod_signed_only", "prod_spoken_only", "prod_signed_spoken"]


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

    # Sign-given-understood ratio (r) priors — three-anchor hump (young/peak/old)
    p_slope_low_sign_dist: Continuous
    p_slope_mid_sign_dist: Continuous
    p_slope_hi_sign_dist: Continuous
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

    # nz_01 produced-cell posterior predictive (counts) and observed
    prod_cell_obs: np.ndarray  # (n_prod_obs, 3) observed
    prod_cell_pred: np.ndarray  # (n_prod_obs, 3, n_samples) predicted
    prod_cell_ages: np.ndarray  # (n_prod_obs,)


JointContext = ModelFitContext[JointModelConfiguration, JointModelSamples]


# ============================================================
# Data preparation
# ============================================================


def _load_uk02_four_cell():
    """Load uk_02 rows, split into four-cell (cross-tab) and marginal-only rows.

    Returns (four_cell_df, marginal_df). The four-cell rows are those that have
    all four cell counts recorded, whose signed and spoken margins reconcile
    with the cross-tab cells (signed == signed_only + signed_spoken,
    spoken == spoken_only + signed_spoken) and whose four cells sum to a
    positive total; they identify psi. For these rows the four-cell sum is
    treated as the authoritative understood total, so a small mismatch between
    the raw comprehension column and the cross-tab partition does not make the
    U likelihood and the Dirichlet-Multinomial likelihood disagree. The rest are
    marginal-only uk_02 rows (no usable cross-tab).

    A row missing any cell — in particular ``understood_only`` (some uk_02 rows
    record a produced sign/speech cross-tab but no comprehension total) — cannot
    form the within-understood four-way composition, so it is routed to the
    marginal-only set, where its recorded spoken/signed margins still inform the
    model. (Without this guard a NaN cell casts to a negative integer and trips
    the four-cell count validation in ``build_model``.)
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_uk_02.csv")
    raw = pd.read_csv(path)
    cells = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
    raw["cell_total"] = raw[cells].sum(axis=1)
    reconciles = (
        raw[cells].notna().all(axis=1)
        & (raw["signed"] == raw["signed_only"] + raw["signed_spoken"])
        & (raw["spoken"] == raw["spoken_only"] + raw["signed_spoken"])
        & (raw["cell_total"] > 0)
    )
    four = raw[reconciles].copy()
    marg = raw[~reconciles].copy()
    return four, marg


def _load_nz01_produced_cells():
    """Load nz_01 (Foster-Cohen) rows as a within-produced three-cell cross-tab.

    nz_01 is production-only (no comprehension). Its checklist codes partition ALL
    items into word-only (a), sign-only (b), both (c) and neither (d). The three
    produced cells {a, b, c} form a modality cross-tab *conditioned on production*,
    not on comprehension: nz_01 records no understood total, and its "neither"
    mixes understood-but-unproduced with not-understood, so it cannot fill uk_02's
    ``understood_only`` cell. Conditioning on produced cancels that cell (and the
    understood level), so these rows identify psi/q/r through a three-cell
    Dirichlet-Multinomial (see ``build_model``). Rows with no produced words
    (``prod_total == 0``) carry no composition and are dropped.
    """
    path = os.path.join(local_env.DATA_DIR, "vocab_data_nz_01.csv")
    raw = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "study": NZ01_STUDY_ID,
            "age": raw["age"].to_numpy(dtype=float),
            "subject_id": raw["subject_id"].to_numpy(),
            # CSV columns are modality-exclusive: spoken=word-only, signed=sign-only,
            # spoken_signed=both. Marginal understood/spoken/signed stay NaN so these
            # rows feed only the produced DM (no double counting).
            "prod_spoken_only": raw["spoken"].to_numpy(dtype=float),
            "prod_signed_only": raw["signed"].to_numpy(dtype=float),
            "prod_signed_spoken": raw["spoken_signed"].to_numpy(dtype=float),
        }
    )
    out["prod_total"] = (
        out["prod_spoken_only"] + out["prod_signed_only"] + out["prod_signed_spoken"]
    )
    return out[out["prod_total"] > 0].reset_index(drop=True)


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
    if definition.exclude_us01_spoken_ceiling:
        merged_columns = merged_columns + ["survey_vocab_max"]

    merged = vocab_data_utils.load_data(
        population=definition.population,
        columns=merged_columns,
    )
    # uk_02 and nz_01 are handled via their cross-tab paths below, so exclude their
    # marginals from the merged view here to avoid double counting. (nz_01 is
    # dropped entirely when include_nz01_cells is False.)
    other = merged[~merged["study"].isin([UK02_STUDY_ID, NZ01_STUDY_ID])].copy()

    four, marg = _load_uk02_four_cell()
    # uk_02 four-cell rows: the four-cell sum is the authoritative understood
    # total; cells feed the DM; marginal spoken/signed are set NaN because they
    # are subsumed by the DM and would otherwise be double counted.
    four_cols = {
        "study": UK02_STUDY_ID,
        "age": four["age"].to_numpy(dtype=float),
        "understood": four["cell_total"].to_numpy(dtype=float),
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

    frames = [other, marg_df, four_df]
    # nz_01's (real-key) CSV is committed separately; tolerate its absence (CI,
    # unit tests, or a checkout predating the data) so the model still builds.
    nz01_csv = os.path.join(local_env.DATA_DIR, "vocab_data_nz_01.csv")
    if definition.include_nz01_cells and os.path.exists(nz01_csv):
        frames.append(_load_nz01_produced_cells())
    analysis_df = pd.concat(frames, ignore_index=True)
    analysis_df = analysis_df.dropna(subset=["age"]).reset_index(drop=True)
    ceiling_rows_excluded = 0
    if definition.exclude_us01_spoken_ceiling:
        analysis_df, ceiling_rows_excluded = (
            vocab_data_utils.exclude_us01_spoken_ceiling_rows(analysis_df)
        )

    analysis_df, sign_source_dropped = (
        vocab_data_utils.mask_incomparable_signed_outcomes(
            analysis_df,
            include_signed_only=definition.include_uk01_signed,
            include_uncertain=definition.include_uk06,
        )
    )

    has_prod_obs = (
        analysis_df["prod_signed_spoken"].notna()
        if "prod_signed_spoken" in analysis_df.columns
        else pd.Series(False, index=analysis_df.index)
    )
    has_any_observation = (
        analysis_df["understood"].notna()
        | analysis_df["spoken"].notna()
        | analysis_df["signed"].notna()
        | analysis_df["signed_spoken"].notna()
        | has_prod_obs
    )
    analysis_df = analysis_df[has_any_observation].reset_index(drop=True)
    if use_subject_codes:
        vocab_data_utils.validate_subject_ids(analysis_df)

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
    n_prod = (
        int(analysis_df["prod_signed_spoken"].notna().sum())
        if "prod_signed_spoken" in analysis_df.columns
        else 0
    )

    counts: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Studies", f"{len(unique_studies)} ({', '.join(unique_studies)})"),
        ("Understood observed", n_u),
        ("Spoken observed (marginal)", n_s),
        ("Signed observed (marginal)", n_sign),
        ("uk_02 four-cell rows (DM)", n_cells),
        ("nz_01 produced-cell rows (DM)", n_prod),
        ("include_uk01_signed", definition.include_uk01_signed),
        ("uk_01 signed-only rows dropped", sign_source_dropped.get("uk_01", 0)),
        ("include_uk06", definition.include_uk06),
        ("uk_06 unverified signed rows dropped", sign_source_dropped.get("uk_06", 0)),
        ("include_nz01_cells", definition.include_nz01_cells),
    ]
    if definition.exclude_us01_spoken_ceiling:
        counts.append(("us_01 WS-ceiling rows excluded", ceiling_rows_excluded))
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
    os.makedirs(context.reporting.output_dir, exist_ok=True)
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
    # Three-anchor hump signed mean (young/peak/old): Beta priors on r at three
    # reference ages, interpolated as a tent meeting at the peak (gp_utils.tent_and_gp).
    p_slope_low_sign_dist = beta(definition.p_slope_low_sign_alpha, definition.p_slope_low_sign_beta, "p_slope_low_sign_dist")
    p_slope_mid_sign_dist = beta(definition.p_slope_mid_sign_alpha, definition.p_slope_mid_sign_beta, "p_slope_mid_sign_dist")
    p_slope_hi_sign_dist = beta(definition.p_slope_hi_sign_alpha, definition.p_slope_hi_sign_beta, "p_slope_hi_sign_dist")

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
        p_slope_low_sign_dist=p_slope_low_sign_dist,
        p_slope_mid_sign_dist=p_slope_mid_sign_dist,
        p_slope_hi_sign_dist=p_slope_hi_sign_dist,
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
    # Numerically stable, branch-free form of the Plackett root. The textbook
    # expression ``(S - disc) / (2 (psi - 1))`` needs a ``switch`` fallback to
    # ``r*q`` at psi == 1 (0/0) and suffers catastrophic cancellation in the
    # whole psi->1 neighbourhood (S ~ disc ~ 1 while the denominator ~ 0), which
    # both distorts pi_both and destabilises the NUTS gradient. Rationalising by
    # ``(S + disc)`` cancels the ``(psi - 1)`` factor exactly:
    #     (S - disc) / (2 (psi - 1))  ==  2 psi r q / (S + disc),
    # since ``S^2 - disc^2 = 4 psi (psi - 1) r q``. The right-hand side has no
    # vanishing denominator (S + disc > 0 across the valid odds-ratio range) and
    # is continuous at psi == 1, where it returns exactly ``r*q`` — so no switch
    # is needed.
    S = 1.0 + (r + q) * (psi - 1.0)
    disc = pm.math.sqrt(pm.math.maximum(S * S - 4.0 * psi * (psi - 1.0) * r * q, 0.0))
    pi_both = 2.0 * psi * r * q / pm.math.maximum(S + disc, 1e-12)
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
    has_cells_t = has_cells & ~holdout

    idx_u = np.where(has_u_t)[0]
    idx_cells = np.where(has_cells_t)[0]

    y_u = np.asarray(df.loc[has_u_t, "understood"], dtype=int)
    marginal_outcome_eligible = ~holdout & ~has_cells
    spoken_spec = nested_outcome_spec(
        df,
        parent_col="understood",
        outcome_col="spoken",
        n_trials=n_trials,
        eligible_mask=marginal_outcome_eligible,
    )
    signed_spec = nested_outcome_spec(
        df,
        parent_col="understood",
        outcome_col="signed",
        n_trials=n_trials,
        eligible_mask=marginal_outcome_eligible,
    )
    expected_spoken = marginal_outcome_eligible & df["spoken"].notna().to_numpy()
    expected_signed = marginal_outcome_eligible & df["signed"].notna().to_numpy()
    if not np.array_equal(spoken_spec.indices, np.flatnonzero(expected_spoken)):
        raise ValueError("Spoken likelihood rows do not match the marginal-data mask.")
    if not np.array_equal(signed_spec.indices, np.flatnonzero(expected_signed)):
        raise ValueError("Signed likelihood rows do not match the marginal-data mask.")
    idx_s = spoken_spec.indices
    idx_sign = signed_spec.indices
    y_s = spoken_spec.observed
    y_sign = signed_spec.observed
    has_s_likelihood = np.zeros(len(df), dtype=bool)
    has_sign_likelihood = np.zeros(len(df), dtype=bool)
    has_s_likelihood[idx_s] = True
    has_sign_likelihood[idx_sign] = True

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

    # nz_01 produced three-cell cross-tab (order matches PROD_CELL_NAMES:
    # sign_only, speak_only, both). n is the observed produced total, not n_trials.
    has_prod = (
        df["prod_signed_spoken"].notna().values
        if "prod_signed_spoken" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    has_prod_t = has_prod & ~holdout
    idx_prod = np.where(has_prod_t)[0]
    if idx_prod.size:
        prod_counts = np.asarray(
            df.loc[has_prod_t, PROD_CELL_COLUMNS],
            dtype=int,
        )
        prod_total = np.asarray(df.loc[has_prod_t, "prod_total"], dtype=int)
        if prod_counts.min() < 0:
            raise ValueError("negative produced-cell count.")
        if not np.array_equal(prod_counts.sum(axis=1), prod_total):
            raise ValueError("produced-cell counts do not sum to prod_total.")
    else:
        prod_counts = np.zeros((0, 3), dtype=int)
        prod_total = np.zeros(0, dtype=int)

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
    X_mean, X_std, X_obs_z = standardize_ages(X_obs)

    # Plot / query grids (standardised), with the optional Option-D reference-age
    # anchor row — see models.build_utils.construct_age_grids. Option D centres
    # each anchored GP to pass through zero at the reference age for every draw,
    # removing the GP<->intercept level redundancy.
    anchor_g_u = bool(definition.anchor_g_u_at_ref)
    anchor_g_q = bool(definition.anchor_g_q_at_ref)
    anchor_g_sign = bool(definition.anchor_g_sign_at_ref)
    use_gp_anchor = anchor_g_u or anchor_g_q or anchor_g_sign
    grids = construct_age_grids(
        X_obs,
        X_obs_z,
        X_obs_mean=X_mean,
        X_obs_std=X_std,
        n_plot=config.n_plot,
        ages_query=config.ages_query,
        slope_anchors=config.slope_anchors,
        use_gp_anchor=use_gp_anchor,
        gp_anchor_age_months=definition.gp_anchor_age_months,
    )
    X_plot = grids.X_plot
    X_query = grids.X_query
    X_all_z = grids.X_all_z
    n_plot = grids.n_plot
    n_query = grids.n_query
    n_all = grids.n_all
    i_anchor = grids.i_anchor
    anchor_age_months = grids.anchor_age_months

    ell_low_months, ell_high_months = validate_ell_bounds(config.ell_months_range)
    ell_low_z = ell_low_months / X_std
    ell_high_z = ell_high_months / X_std
    L, M = get_hsgp_hyperparams(X_all_z, (ell_low_z, ell_high_z))

    sa_z, sb_z = slope_anchor_logit_coeffs(
        config.slope_anchors, X_obs_mean=X_mean, X_obs_std=X_std
    )

    build_cfg: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Studies", n_studies),
        ("Understood / spoken / signed / cells", f"{len(idx_u)} / {len(idx_s)} / {len(idx_sign)} / {len(idx_cells)}"),
        ("Spoken conditional / marginal", f"{spoken_spec.n_conditional} / {spoken_spec.n_marginal}"),
        ("Signed conditional / marginal", f"{signed_spec.n_conditional} / {signed_spec.n_marginal}"),
        ("Child > understood violations (spoken/signed)", f"{spoken_spec.n_parent_violations} / {signed_spec.n_parent_violations}"),
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
        "obs_prod_id": np.arange(len(idx_prod)),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "study_id": np.arange(n_studies),
        "cell_id": CELL_NAMES,
        "prod_cell_id": PROD_CELL_NAMES,
        "x_dim": np.arange(1),
    }
    if use_subject_codes:
        coords["subject_id"] = np.arange(n_subjects)

    with pm.Model(coords=coords) as model_pm:
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))
        study_obs = pm.Data("study_obs", study_codes, dims=("obs_id",))
        if use_subject_codes:
            subject_obs = pm.Data("subject_obs", subject_codes, dims=("obs_id",))
        _ = pm.Data("obs_cells_mask", has_cells_t.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_prod_mask", has_prod_t.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_u_mask", has_u_t.astype(int), dims=("obs_id",))
        _ = pm.Data("obs_s_mask", has_s_likelihood.astype(int), dims=("obs_id",))
        _ = pm.Data(
            "obs_sign_mask", has_sign_likelihood.astype(int), dims=("obs_id",)
        )
        s_likelihood_n = pm.Data(
            "s_likelihood_n", spoken_spec.trials, dims=("obs_s_id",)
        )
        s_is_conditional = pm.Data(
            "s_is_conditional",
            spoken_spec.is_conditional.astype(int),
            dims=("obs_s_id",),
        )
        sign_likelihood_n = pm.Data(
            "sign_likelihood_n", signed_spec.trials, dims=("obs_sign_id",)
        )
        sign_is_conditional = pm.Data(
            "sign_is_conditional",
            signed_spec.is_conditional.astype(int),
            dims=("obs_sign_id",),
        )

        # Latent full-grid trajectories (plain tensors), built by the shared
        # gp_utils helpers. Option D anchors each GP (per-draw zero at the
        # reference age) when the matching flag is set.
        gp_grid = GPGrid(
            sa_z=sa_z,
            sb_z=sb_z,
            ell_low_z=ell_low_z,
            ell_high_z=ell_high_z,
            M=M,
            L=L,
        )
        f_u_all = trend_and_gp(
            cfg_low=config.p_slope_low_u_dist,
            cfg_hi=config.p_slope_hi_u_dist,
            cfg_ell=config.ell_unit_u_dist,
            cfg_eta=config.eta_u_dist,
            suffix="_u",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=False,
            anchor_idx=i_anchor if anchor_g_u else None,
        )
        h_all = trend_and_gp(
            cfg_low=config.p_slope_low_q_dist,
            cfg_hi=config.p_slope_hi_q_dist,
            cfg_ell=config.ell_unit_q_dist,
            cfg_eta=config.eta_q_dist,
            suffix="_q",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=False,
            anchor_idx=i_anchor if anchor_g_q else None,
        )
        # Signed marginal: three-anchor "tent" hump mean (young/peak/old) + GP; the
        # study random intercept delta_sign is added at obs level below. The GP is
        # anchored at 54 mo (anchor_g_sign) so the tent supplies the hump and the GP
        # only deviates around it.
        sa_young, sa_peak, sa_old = definition.sign_anchor_ages
        g_all = tent_and_gp(
            cfg_low=config.p_slope_low_sign_dist,
            cfg_mid=config.p_slope_mid_sign_dist,
            cfg_hi=config.p_slope_hi_sign_dist,
            z_low=(sa_young - X_mean) / X_std,
            z_mid=(sa_peak - X_mean) / X_std,
            z_hi=(sa_old - X_mean) / X_std,
            cfg_ell=config.ell_unit_sign_dist,
            cfg_eta=config.eta_sign_dist,
            suffix="_sign",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=False,
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

        # --- kappa functions (shared helper — see models.gp_utils.build_kappa_of_z) ---
        kappa_u_of_z = build_kappa_of_z(
            config.kappa_min_u_dist, config.a_kappa_u_dist, config.b_kappa_mag_u_dist, suffix="_u"
        )
        kappa_s_of_z = build_kappa_of_z(
            config.kappa_min_s_dist, config.a_kappa_s_dist, config.b_kappa_mag_s_dist, suffix="_s"
        )
        kappa_sign_of_z = build_kappa_of_z(
            config.kappa_min_sign_dist, config.a_kappa_sign_dist, config.b_kappa_mag_sign_dist, suffix="_sign"
        )

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

        # Spoken: nested where U is usable, otherwise marginal over the inventory.
        p_s_sel = pm.math.switch(
            s_is_conditional,
            q_obs[idx_s],
            (p_u_obs * q_obs)[idx_s],
        )
        p_s_sel = pm.math.clip(p_s_sel, EPSILON, 1 - EPSILON)
        k_s = kappa_s_obs[idx_s]
        pm.BetaBinomial("y_s_obs", n=s_likelihood_n, alpha=p_s_sel * k_s, beta=(1 - p_s_sel) * k_s,
                        observed=y_s, dims="obs_s_id")

        # Signed: nested where U is usable, otherwise marginal over the inventory.
        p_sign_sel = pm.math.switch(
            sign_is_conditional,
            r_obs[idx_sign],
            (p_u_obs * r_obs)[idx_sign],
        )
        p_sign_sel = pm.math.clip(p_sign_sel, EPSILON, 1 - EPSILON)
        k_sign = kappa_sign_obs[idx_sign]
        pm.BetaBinomial("y_sign_obs", n=sign_likelihood_n, alpha=p_sign_sel * k_sign, beta=(1 - p_sign_sel) * k_sign,
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

        # nz_01 produced three cells (Dirichlet-Multinomial), within-PRODUCED
        # composition. nz_01 has no comprehension, so we condition on produced:
        # drop the unobservable "neither" (understood_only) cell from the Plackett
        # within-understood composition and renormalise over {sign_only, speak_only,
        # both}. Same psi/conc and population+study marginals as the uk_02 DM, so the
        # two cross-tab sources jointly identify psi.
        if idx_prod.size:
            r_p = pm.math.clip(r_obs_pop[idx_prod], EPSILON, 1 - EPSILON)
            q_p = pm.math.clip(q_obs_pop[idx_prod], EPSILON, 1 - EPSILON)
            pi_both_p = _plackett_pi_both(r_p, q_p, psi)
            pi_sign_p = pm.math.maximum(r_p - pi_both_p, EPSILON)
            pi_speak_p = pm.math.maximum(q_p - pi_both_p, EPSILON)
            pi_both_p = pm.math.maximum(pi_both_p, EPSILON)
            pi_prod = pm.math.stack([pi_sign_p, pi_speak_p, pi_both_p], axis=1)
            pi_prod = pi_prod / pi_prod.sum(axis=1, keepdims=True)
            pm.DirichletMultinomial(
                "nz_prod_cells_obs", n=prod_total, a=conc * pi_prod, observed=prod_counts,
                dims=("obs_prod_id", "prod_cell_id"),
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

        # Persist the obs-grid signed kappa in the trace for downstream
        # inspection. (It is not shown in the diagnostics() trace plot: that
        # plots only scalar unobserved RVs, and this is an obs_id-length
        # deterministic.)
        pm.Deterministic("kappa_sign_obs", kappa_sign_obs, dims="obs_id")

    pymc_utils.report_model_summary(model_pm)
    variables = pymc_utils.get_variables_dict(model_pm)
    render_model_graph(model_pm, context.reporting.output_dir)

    context.set_model(model_pm, variables)


# ============================================================
# Pipeline
# ============================================================


def prior_predictive_checks(context: JointContext):
    """Prior predictive checks for the joint sign/speech model."""
    with context.model:
        prior = pm.sample_prior_predictive(
            draws=500, random_seed=context.sampling.random_seed,
            compile_kwargs=dict(mode="FAST_COMPILE"),
        )
    context.set_prior_samples(prior)

    def prior_matrix(var: str) -> np.ndarray:
        return (
            prior.prior[var]
            .stack(sample=("chain", "draw"))
            .transpose("plot_id", "sample")
            .values
        )

    x_plot = prior.constant_data["X_plot"].values
    analysis_df = context.analysis_df

    p_u = prior_matrix("p_u_plot")
    q = prior_matrix("q_plot")
    r = prior_matrix("r_plot")
    p_s = p_u * q
    p_sign = p_u * r

    for y_col, samples, y_label, fname in [
        ("understood", p_u, "Words understood", "prior_samples_u"),
        ("spoken", p_s, "Words spoken", "prior_samples_s"),
        ("signed", p_sign, "Words signed", "prior_samples_sign"),
    ]:
        observed = analysis_df[analysis_df[y_col].notna()]
        fig = plot_prior_samples(
            x_plot,
            samples,
            observed["age"],
            observed[y_col],
            n_trials=context.model_data.n_trials,
            x_label="Age (months)",
            y_label=y_label,
            filename=fname,
            output_dir=context.reporting.output_dir,
        )
        plt.close(fig)

    for var, ylab, fname in [
        ("r_plot", "r(a) = P(sign | understood)", "prior_samples_r"),
        ("q_plot", "q(a) = P(speak | understood)", "prior_samples_q"),
        ("p_any_plot", "p_any(a) total expressive probability", "prior_samples_p_any"),
    ]:
        s = prior_matrix(var)
        fig = plot_prior_samples_ratio(
            x_plot,
            s,
            y_label=ylab,
            filename=fname,
            output_dir=context.reporting.output_dir,
        )
        plt.close(fig)


# ``sample`` is engine-agnostic (identical pm.sample() call in every engine) —
# reuse the shared implementation from common.py rather than redefining it.
sample = _shared_sample


def diagnostics(context: JointContext):
    """Run diagnostics on the posterior samples.

    Thin wrapper over the shared engine (common.py): joint prioritises
    ``psi``/``conc`` first in the pair/trace/posterior-density plots (they are
    the headline association/concentration parameters) and reports per-outcome
    LOO-CV for the three marginal likelihoods (the four-cell
    Dirichlet-Multinomial ``cells_obs`` is not a per-observation Beta-Binomial
    and is left out of LOO-CV, matching every other engine's scope).
    """
    posterior_vars = set(context.trace.posterior.data_vars)

    def _prioritise_psi_conc(
        names: list[str],
        priority: tuple[str, ...] = ("psi", "conc"),
    ) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in (*priority, *names):
            if name in posterior_vars and name not in seen:
                ordered.append(name)
                seen.add(name)
        return ordered

    _shared_diagnostics(
        context,
        var_names_fn=_prioritise_psi_conc,
        round_to=4,
        loo_var_names=(
            ("y_u_obs", "words understood"),
            ("y_s_obs", "words spoken"),
            ("y_sign_obs", "words signed"),
        ),
    )


def _extract_produced_cell_observations(
    df: pd.DataFrame,
    has_prod: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed nz_01 produced-cell counts and ages for rows flagged by ``has_prod``."""
    if not has_prod.any():
        return (
            np.zeros((0, len(PROD_CELL_NAMES)), dtype=int),
            np.zeros(0, dtype=float),
        )

    missing = [col for col in (*PROD_CELL_COLUMNS, "age") if col not in df.columns]
    if missing:
        raise ValueError(
            "obs_prod_mask marks produced-cell rows, but analysis_df is missing "
            f"columns: {', '.join(missing)}"
        )

    return (
        np.asarray(df.loc[has_prod, PROD_CELL_COLUMNS], dtype=int),
        np.asarray(df.loc[has_prod, "age"], dtype=float),
    )


def sample_posterior_predictive(context: JointContext, definition=None):
    """Posterior predictive for the observed cell-count likelihoods."""
    with context.model:
        # Include the three marginal word-count likelihoods alongside the
        # four-cell composition likelihoods so they are posterior-predictively
        # sampled too (they are unconditionally defined in the build above).
        var_names = ["cells_obs", "y_u_obs", "y_s_obs", "y_sign_obs"]
        if "nz_prod_cells_obs" in context.model.named_vars:
            var_names.append("nz_prod_cells_obs")
        trace = pm.sample_posterior_predictive(
            context.trace, var_names=var_names, extend_inferencedata=True,
            progressbar=sys.stdout.isatty(),
            random_seed=context.sampling.random_seed,
        )
    context.set_trace(trace)
    calibration_df = write_trace_calibration(
        trace,
        context.analysis_df,
        context.reporting.output_dir,
        (
            ("understood", "y_u_obs", "obs_u_mask"),
            ("spoken", "y_s_obs", "obs_s_mask"),
            ("signed", "y_sign_obs", "obs_sign_mask"),
        ),
    )
    context.dataframes["posterior_predictive_calibration"] = calibration_df
    trace.to_netcdf(os.path.join(context.reporting.output_dir, "trace.nc"))

    # Observed uk_02 four-cell counts / ages. Use the stored training mask so
    # held-out four-cell rows stay aligned with posterior_predictive["cells_obs"].
    df = context.analysis_df
    has_cells = np.array(trace.constant_data["obs_cells_mask"].values, dtype=bool)
    cell_counts = np.asarray(
        df.loc[has_cells, ["understood_only", "signed_only", "spoken_only", "signed_spoken"]],
        dtype=int,
    )
    cell_ages = np.asarray(df.loc[has_cells, "age"], dtype=float)
    cell_pred = np.array(
        trace.posterior_predictive["cells_obs"]
        .stack(sample=("chain", "draw"))
        .transpose("obs_cells_id", "cell_id", "sample")
        .values
    )
    if int(has_cells.sum()) != cell_pred.shape[0]:
        raise ValueError(
            f"obs_cells_mask count ({int(has_cells.sum())}) does not match "
            f"posterior predictive cells_obs length ({cell_pred.shape[0]}); "
            "stored mask and likelihood rows are misaligned."
        )

    has_prod = np.array(trace.constant_data["obs_prod_mask"].values, dtype=bool)
    prod_counts, prod_ages = _extract_produced_cell_observations(df, has_prod)
    if "nz_prod_cells_obs" in trace.posterior_predictive:
        prod_pred = np.array(
            trace.posterior_predictive["nz_prod_cells_obs"]
            .stack(sample=("chain", "draw"))
            .transpose("obs_prod_id", "prod_cell_id", "sample")
            .values
        )
        if int(has_prod.sum()) != prod_pred.shape[0]:
            raise ValueError(
                f"obs_prod_mask count ({int(has_prod.sum())}) does not match "
                f"posterior predictive nz_prod_cells_obs length ({prod_pred.shape[0]}); "
                "stored mask and likelihood rows are misaligned."
            )
    else:
        prod_pred = np.zeros((0, len(PROD_CELL_NAMES), 0), dtype=int)

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
        cell_pred=cell_pred,
        cell_ages=cell_ages,
        prod_cell_obs=prod_counts,
        prod_cell_pred=prod_pred,
        prod_cell_ages=prod_ages,
    )
    context.set_model_samples(samples)


def posterior_summary(context: JointContext):
    s = context.model_samples
    n_trials = context.model_data.n_trials
    ci_prob = context.reporting.ci_prob
    ci_kind = context.reporting.interval_kind
    inner = intervals.INNER_CI_PROB
    od = context.reporting.output_dir

    def probability_summary(X, draws, prefix, label):
        Ey = draws * n_trials
        p_o = intervals.bands(draws, ci_prob, ci_kind, sample_axis=1)
        p_i = intervals.bands(draws, inner, ci_kind, sample_axis=1)
        Ey_o = intervals.bands(Ey, ci_prob, ci_kind, sample_axis=1)
        Ey_i = intervals.bands(Ey, inner, ci_kind, sample_axis=1)
        d = pd.DataFrame({
            "age_months": X,
            f"p_{prefix}_median": np.median(draws, axis=1),
            f"p_{prefix}_ci50_lo": p_i[:, 0],
            f"p_{prefix}_ci50_hi": p_i[:, 1],
            f"p_{prefix}_ci_lo": p_o[:, 0],
            f"p_{prefix}_ci_hi": p_o[:, 1],
            f"Ey_{prefix}_median": np.median(Ey, axis=1),
            f"Ey_{prefix}_ci50_lo": Ey_i[:, 0],
            f"Ey_{prefix}_ci50_hi": Ey_i[:, 1],
            f"Ey_{prefix}_ci_lo": Ey_o[:, 0],
            f"Ey_{prefix}_ci_hi": Ey_o[:, 1],
        })
        d.to_csv(os.path.join(od, f"posterior_summary_{prefix}.csv"), index=False)
        dataframe_table(d.round(3), title=label, show_index=False)
        return d

    def ratio_summary(X, draws, prefix):
        d = intervals.summarise(
            draws, X, name=f"{prefix}_query", outer=ci_prob, sample_axis=1
        ).rename(
            columns={
                "median": f"{prefix}_median",
                "ci50_lo": f"{prefix}_ci50_lo",
                "ci50_hi": f"{prefix}_ci50_hi",
                "ci_lo": f"{prefix}_ci_lo",
                "ci_hi": f"{prefix}_ci_hi",
            }
        )
        d.to_csv(os.path.join(od, f"posterior_summary_{prefix}.csv"), index=False)
        return d

    probability_summary(s.X_query, s.p_u_query, "u", "Words understood")
    probability_summary(s.X_query, s.p_u_query * s.q_query, "s", "Words spoken")
    probability_summary(s.X_query, s.p_u_query * s.r_query, "sign", "Words signed")
    ratio_summary(s.X_query, s.r_query, "r")
    ratio_summary(s.X_query, s.q_query, "q")

    # Data-identified p_any vs independence (expected counts)
    Ey = s.p_any_query * n_trials
    Ey_i = s.p_any_indep_query * n_trials
    p_any_o = intervals.bands(s.p_any_query, ci_prob, ci_kind, sample_axis=1)
    pany = pd.DataFrame({
        "age_months": s.X_query,
        "p_any_median": np.median(s.p_any_query, axis=1),
        "p_any_ci_lo": p_any_o[:, 0],
        "p_any_ci_hi": p_any_o[:, 1],
        "Ey_any_median": np.median(Ey, axis=1),
        "p_any_indep_median": np.median(s.p_any_indep_query, axis=1),
        "Ey_any_indep_median": np.median(Ey_i, axis=1),
    })
    pany.to_csv(os.path.join(od, "posterior_summary_p_any.csv"), index=False)
    dataframe_table(pany.round(3), title="Total expressive p_any (identified vs independence)", show_index=False)

    # psi summary (HDI: psi is a right-skewed association ratio)
    psi = s.psi
    psi_lo, psi_hi = intervals.interval_1d(psi, ci_prob, "hdi")
    pct = int(round(ci_prob * 100))
    psi_df = pd.DataFrame({
        "psi_median": [float(np.median(psi))],
        "psi_ci_lo": [float(psi_lo)],
        "psi_ci_hi": [float(psi_hi)],
        "P_psi_gt_1": [float((psi > 1).mean())],
    })
    psi_df.to_csv(os.path.join(od, "posterior_summary_psi.csv"), index=False)
    key_value_table("Association psi", [
        ("psi median", round(float(np.median(psi)), 3)),
        (f"psi {pct}% HDI", (round(float(psi_lo), 3), round(float(psi_hi), 3))),
        ("P(psi > 1)", round(float((psi > 1).mean()), 3)),
    ])


# ============================================================
# Plots
# ============================================================


def _run_joint_plots(context: JointContext):
    s = context.model_samples
    n_trials = context.model_data.n_trials
    od = context.reporting.output_dir
    ci_prob = context.reporting.ci_prob
    ci_kind = context.reporting.interval_kind
    X = s.X_plot

    # 1) Data-identified p_any vs independence upper bound (expected counts)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    id_med = np.median(s.p_any_plot, axis=1) * n_trials
    id_hdi = intervals.bands(s.p_any_plot * n_trials, ci_prob, ci_kind, sample_axis=1)
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
                            "identified_ci_lo": id_hdi[:, 0], "identified_ci_hi": id_hdi[:, 1],
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

    # 4) signed rate r(a) and spoken rate q(a), population level
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    r_med = np.median(s.r_plot, axis=1)
    r_hdi = intervals.bands(s.r_plot, ci_prob, ci_kind, sample_axis=1)
    q_med = np.median(s.q_plot, axis=1)
    ax.fill_between(X, r_hdi[:, 0], r_hdi[:, 1], alpha=0.18, color="C2")
    ax.plot(X, r_med, lw=3, color="C2", label="r(a) signed")
    ax.plot(X, q_med, lw=3, color="C1", label="q(a) spoken")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Fraction of understood words")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Signed vs spoken ratio (population-level)")
    fig.savefig(os.path.join(od, "signed_vs_spoken_rate.png"), dpi=300)
    fig.savefig(os.path.join(od, "signed_vs_spoken_rate.svg"))
    context.plots["signed_vs_spoken_rate"] = fig
    plt.close(fig)

    # 5) uk_02 four-cell PPC (observed vs predicted cell totals)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    obs_tot = s.cell_obs.sum(axis=0)  # (4,)
    pred_tot = s.cell_pred.sum(axis=0)  # (4, n_samples)
    pred_med = np.median(pred_tot, axis=1)
    lo, hi = 100 * (1 - ci_prob) / 2, 100 * (1 + ci_prob) / 2
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

    # 6) nz_01 produced-cell PPC (observed vs predicted produced-cell totals)
    if s.prod_cell_obs.size and s.prod_cell_pred.size:
        fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
        obs_tot = s.prod_cell_obs.sum(axis=0)  # (3,)
        pred_tot = s.prod_cell_pred.sum(axis=0)  # (3, n_samples)
        pred_med = np.median(pred_tot, axis=1)
        lo, hi = 100 * (1 - ci_prob) / 2, 100 * (1 + ci_prob) / 2
        pred_lo = np.percentile(pred_tot, lo, axis=1)
        pred_hi = np.percentile(pred_tot, hi, axis=1)
        yerr = np.vstack([pred_med - pred_lo, pred_hi - pred_med])
        xpos = np.arange(len(PROD_CELL_NAMES))
        ax.bar(xpos - 0.18, obs_tot, width=0.36, color="C0", label="observed")
        ax.bar(
            xpos + 0.18,
            pred_med,
            width=0.36,
            color="C3",
            alpha=0.7,
            label="predicted (median)",
            yerr=yerr,
            capsize=4,
        )
        ax.set_xticks(xpos)
        ax.set_xticklabels(PROD_CELL_NAMES)
        ax.set_ylabel("Total produced-cell count (nz_01)")
        ax.legend(frameon=True)
        ax.set_title("nz_01 produced-cell posterior predictive check")
        fig.savefig(os.path.join(od, "nz01_produced_cell_ppc.png"), dpi=300)
        fig.savefig(os.path.join(od, "nz01_produced_cell_ppc.svg"))
        context.plots["nz01_produced_cell_ppc"] = fig
        plt.close(fig)


# ============================================================
# Fit orchestration
# ============================================================


def fit_joint_model(config: str, definition: JointModelDefinition) -> JointContext:
    """Shared fit pipeline for the joint sign/speech model (VG15)."""
    return run_fit_pipeline(
        config,
        definition,
        stages=[
            ("Prepare data", lambda ctx: prepare_joint_data(ctx, definition)),
            (
                "Priors and hyperparameters",
                lambda ctx: configure_joint_priors(ctx, definition),
            ),
            (
                "Model definition and initialisation",
                lambda ctx: build_model(ctx, definition),
            ),
            ("Prior predictive checks", prior_predictive_checks),
            ("Posterior sampling", sample),
            ("Diagnostics", diagnostics),
            (
                "Posterior predictions",
                lambda ctx: sample_posterior_predictive(ctx, definition),
            ),
            ("Posterior summary", posterior_summary),
            ("Plots", _run_joint_plots),
            ("Report", report),
        ],
    )
