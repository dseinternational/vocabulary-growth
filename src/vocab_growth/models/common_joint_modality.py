# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Shared dataclasses and pipeline functions for the joint sign/speech modality
model (VG15): issue #49 Option 3.

VG15 extends the trivariate VG14 with two things VG14 assumed away:

1.  A within-understood sign-speech ASSOCIATION (a Plackett odds ratio ``psi``
    with a study-level random intercept ``delta_psi``), identified from four
    cross-tabulation sources: the uk_02, uk_07 and es_01 four-cell
    within-understood cross-tabs (sign-only / sign+speech / speech-only /
    understood-only) and nz_01's three-cell within-produced cross-tab. This
    replaces VG14's ``p_any`` calculated under independence with a
    *data-identified* total expressive vocabulary. The reported population
    ``psi`` is a shrunk centre over sources that disagree; the per-study values
    are the primary read.
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
The rows with a four-cell cross-tabulation use that joint composition term
instead of duplicate spoken and signed likelihood contributions:
    - understood ~ BetaBinomial(810, p_U)              (all DS studies)
    - spoken | understood ~ BetaBinomial(understood, q)
    - signed | understood ~ BetaBinomial(understood, r)
    - uk_02 / uk_07 / es_01 four cells ~ DirichletMultinomial(total, conc * [pi_*])
      (the within-understood joint term; identifies psi)
    - nz_01 three produced cells ~ DirichletMultinomial(produced, conc * [pi_*])
      (the within-produced term; also identifies psi, with no comprehension
      denominator — p_U cancels, so nz_01 carries no understood information)

This is a self-contained module (like common_trivariate.py); it does not import
from or modify the bivariate / trivariate engines. The full-grid intermediates
are kept as plain tensors (only obs/plot/query slices are stored), following the
VG14 memory discipline.
"""

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import dse_research_utils.math.constants as math_constants
import dse_research_utils.plot.io as plot_io
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
import vocab_growth.posterior_analysis as posterior_analysis
import vocab_growth.reporting_ages as reporting_ages
from vocab_growth.administration_loo import LikelihoodFactor
from vocab_growth.cross_tab_sources import (
    ES01_STUDY_ID,
    NZ01_STUDY_ID,
    UK02_DSE_FORM,
    UK02_STUDY_ID,
    UK07_STUDY_ID,
    load_es01_four_cell,
    load_nz01_produced_cells,
    load_uk02_four_cell,
    load_uk07_four_cell,
)
from vocab_growth.fit_artifacts import save_trace
from vocab_growth.models.build_reporting import BuildReport
from vocab_growth.models.build_utils import (
    construct_age_grids,
    standardize_ages,
    standardize_ages_to_z,
    standardize_anchor_ages,
    validate_ell_bounds,
)
from vocab_growth.models.calibration import write_trace_calibration
from vocab_growth.models.common import (
    AnchoredKappaPriors,
    BaseModelConfiguration,
    ModelFitContext,
    build_kappa_for_config,
    configure_kappa_priors,
    emit_monthly_summary,
    get_hsgp_hyperparams,
    kappa_anchor_derived_rows,
    plot_and_print_dist,
    report,
    report_model_build,
    run_fit_pipeline,
    validate_kappa_fields,
)
from vocab_growth.models.common import diagnostics as _shared_diagnostics
from vocab_growth.models.common import sample as _shared_sample
from vocab_growth.models.composition import (
    build_composition_likelihood,
    build_composition_parameters,
)
from vocab_growth.models.composition import (
    composition_probabilities as _composition_probabilities,
)
from vocab_growth.models.composition import (
    plackett_pi_both as _plackett_pi_both,
)
from vocab_growth.models.cross_lag import (
    prev_wave_sign_share_lag_for_frame,
    sign_cross_lag_audit_frame,
    sign_lag_same_form_only,
    validate_sign_cross_lag,
)
from vocab_growth.models.definitions import JointModelDefinition, clamp_targets
from vocab_growth.models.diagnostics_utils import pair_plot_var_names_fn
from vocab_growth.models.gp_utils import (
    GPGrid,
    tent_and_gp,
    trend_and_gp,
)
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PAIRED_ONLY,
    nested_outcome_alpha_beta,
)
from vocab_growth.models.observation_arrays import (
    CELL_COLUMNS as CELL_COLUMNS,
)
from vocab_growth.models.observation_arrays import (
    CELL_NAMES as CELL_NAMES,
)
from vocab_growth.models.observation_arrays import (
    PROD_CELL_COLUMNS as PROD_CELL_COLUMNS,
)
from vocab_growth.models.observation_arrays import (
    PROD_CELL_NAMES as PROD_CELL_NAMES,
)
from vocab_growth.models.observation_arrays import (
    prepare_joint_observations,
)
from vocab_growth.models.study_effects import informed_studies, zero_sum_study_offsets
from vocab_growth.models.subject_graphs import (
    SUBJECT_RE_CORRELATIONS as SUBJECT_RE_CORRELATIONS,
)
from vocab_growth.models.subject_graphs import (
    SUBJECT_RE_OUTCOMES as SUBJECT_RE_OUTCOMES,
)
from vocab_growth.models.subject_graphs import (
    build_joint_child_effects,
)
from vocab_growth.plotting import (
    plot_prior_samples,
    plot_prior_samples_ratio,
)
from vocab_growth.posterior_analysis import extract_posterior
from vocab_growth.reporting import (
    console,
    dataframe_table,
    heading,
    key_value_table,
)

EPSILON = math_constants.EPSILON





# ============================================================
# Dataclasses
# ============================================================


_QUANTITY_BY_SUFFIX = {
    "u": reporting_ages.ReportedQuantity.UNDERSTOOD,
    "s": reporting_ages.ReportedQuantity.SPOKEN,
    "sign": reporting_ages.ReportedQuantity.SIGNED,
}


#: A hand-over milestone is only read once the child has at least this much
#: expressive vocabulary. Below it the three cells are fractions of a word and
#: their ordering carries no information.
MIN_WORDS_FOR_MILESTONE = 1.0


def _signing_milestones(
    ages: np.ndarray,
    sign_only: np.ndarray,
    both: np.ndarray,
    speak_only: np.ndarray,
    ci_prob: float,
) -> pd.DataFrame:
    """Per-draw ages for the sign-to-speech hand-over (shared implementation).

    Thin wrapper over :func:`vocab_growth.posterior_analysis.signing_milestone_table`,
    which the DS/TD comparison script also uses — the two used to carry duplicate
    copies whose crossing rule reported *first age true* rather than a genuine
    false-to-true transition, and whose peak rule reported a grid-boundary
    maximum as reached (#238). See the shared helper for the corrected
    semantics, the censoring columns, and the HDI interval policy.
    """
    return posterior_analysis.signing_milestone_table(
        ages,
        sign_only,
        both,
        speak_only,
        ci_prob=ci_prob,
        min_words=MIN_WORDS_FOR_MILESTONE,
    )


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

    # Association (Plackett log odds-ratio) and Dirichlet-Multinomial concentration
    log_psi_dist: Continuous
    log_conc_dist: Continuous

    # Study random-intercept scales
    tau_u_sigma: float
    tau_q_sigma: float
    tau_sign_sigma: float
    tau_psi_sigma: float

    # Kappa priors (age-varying dispersion) — understood / spoken / signed.
    # Legacy triples; None on an outcome that uses the two-anchor form below.
    # These carry defaults, so they sit last in the field order.
    kappa_min_u_dist: Continuous | None = None
    a_kappa_u_dist: Continuous | None = None
    b_kappa_mag_u_dist: Continuous | None = None
    kappa_min_s_dist: Continuous | None = None
    a_kappa_s_dist: Continuous | None = None
    b_kappa_mag_s_dist: Continuous | None = None
    kappa_min_sign_dist: Continuous | None = None
    a_kappa_sign_dist: Continuous | None = None
    b_kappa_mag_sign_dist: Continuous | None = None

    # Two-anchor dispersion priors, in place of the triples above. Outcomes are
    # independent: VG15 anchors understood and spoken, whose scale the Down
    # syndrome joint calibration covers, and leaves the signed ratio on the
    # legacy form because nothing measures it.
    kappa_anchored_u: AnchoredKappaPriors | None = None
    kappa_anchored_s: AnchoredKappaPriors | None = None
    kappa_anchored_sign: AnchoredKappaPriors | None = None

    # Reporting only — the age at which understood and q stop being reported.
    report_max_age_understood: int | None = None
    report_max_age_signed: int | None = None
    """Highest query age at which signed quantities are reported. Same mechanism
    and the same caveats as ``report_max_age_understood`` -- post-processing only,
    but part of the recorded definition, so a change needs a refit."""

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_kappa_fields(self, suffixes=("_u", "_s", "_sign"))


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
    # Per-study association and its between-study SD. `psi_study` is (n_studies,
    # n_samples); studies with no cross-tab carry exactly the population value.
    # `tau_psi` is None when fewer than two studies inform psi (nothing to estimate).
    psi_study: np.ndarray
    psi_study_names: list
    # Which of `psi_study_names` actually contribute a cross-tabulation term, as
    # an explicit record rather than something inferred afterwards. Equality
    # between a study's draws and the population value cannot stand in for this:
    # with a single informed study the model takes its degenerate branch and
    # pins every delta_psi to zero, so the one study that does inform psi also
    # sits at exactly the population value and an equality test discards it
    # (#266). Down syndrome native-only is precisely that configuration.
    psi_informed_studies: tuple[str, ...]
    tau_psi: np.ndarray | None

    # Four-cell posterior predictive (counts) and observed, all sources
    cell_obs: np.ndarray  # (n_cells_obs, 4) observed
    cell_pred: np.ndarray  # (n_cells_obs, 4, n_samples) predicted
    cell_studies: np.ndarray  # (n_cells_obs,) source study of each row

    # nz_01 produced-cell posterior predictive (counts) and observed
    prod_cell_obs: np.ndarray  # (n_prod_obs, 3) observed
    prod_cell_pred: np.ndarray  # (n_prod_obs, 3, n_samples) predicted


JointContext = ModelFitContext[JointModelConfiguration, JointModelSamples]


# ============================================================
# Data preparation
# ============================================================


def build_joint_analysis_frame(
    definition: JointModelDefinition,
) -> tuple[pd.DataFrame, dict]:
    """The exact prepared frame the joint engine fits, with no side effects.

    Split out of :func:`prepare_joint_data` so fitted-output validation can
    recompute the frame (and its exact hash) without a fit context — see
    :mod:`vocab_growth.analysis_frames` (issue #266 finding 1).
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
    if definition.exclude_us01_spoken_ceiling or definition.dse_native_only:
        merged_columns = merged_columns + ["survey_vocab_max"]

    merged = vocab_data_utils.load_data(
        population=definition.population,
        columns=merged_columns,
        include_implausible_production=definition.include_implausible_production,
        include_same_day_disagreements=definition.include_same_day_disagreements,
    )
    # The DSE-native sensitivity drops every source on a shorter form, which
    # includes three of the four cross-tab sources. Their cell blocks are gated
    # off below rather than left to the row filter, because those blocks read
    # their own CSVs and the merged-view filter would never see them.
    native_only = definition.dse_native_only
    non_native_rows_excluded = 0
    if native_only:
        merged, non_native_rows_excluded = (
            vocab_data_utils.restrict_to_dse_native_administrations(merged)
        )
    use_uk07_cells = definition.include_uk07_cells and not native_only
    use_es01_cells = definition.include_es01_cells and not native_only
    use_nz01_cells = definition.include_nz01_cells and not native_only
    # uk_02, uk_07, es_01 and nz_01 are handled via their cross-tab paths below, so
    # exclude their marginals from the merged view here to avoid double counting.
    # uk_02 and nz_01 are unconditional; uk_07 and es_01 join the list only when
    # their include_* flags select the cross-tab path (see the appends below).
    # (nz_01 is dropped entirely when include_nz01_cells is False; uk_07 and es_01
    # fall back to their merged-view marginals when their flag is False, because
    # unlike nz_01 those marginals are usable on their own.)
    cross_tab_studies = [UK02_STUDY_ID, NZ01_STUDY_ID]
    if use_uk07_cells:
        cross_tab_studies.append(UK07_STUDY_ID)
    if use_es01_cells:
        cross_tab_studies.append(ES01_STUDY_ID)
    other = merged[~merged["study"].isin(cross_tab_studies)].copy()

    four, marg = load_uk02_four_cell()
    if native_only:
        # uk_02 ran both forms; only its DSE arm is on the native 810 reference.
        # All 56 four-cell rows are that arm, so the cross-tab survives intact
        # and only Oxford marginals leave.
        four = four[four["form"].eq(UK02_DSE_FORM)].copy()
        marg = marg[marg["form"].eq(UK02_DSE_FORM)].copy()
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

    # uk_07 (PACT-DS): the same within-understood four-cell partition, derived from
    # its modality-exclusive expressive cells. Its `understood` is the recorded
    # comprehension total and equals the four-cell sum by construction, so unlike
    # uk_02 there is no partition-versus-total mismatch to reconcile. Marginal
    # spoken/signed are NaN on the four-cell rows for the same no-double-counting
    # reason. See data/vocab_data_uk_07.md.
    if use_uk07_cells:
        four07, marg07 = load_uk07_four_cell()
        four07_cols = {
            "study": UK07_STUDY_ID,
            "age": four07["age"].to_numpy(dtype=float),
            "understood": four07["understood"].to_numpy(dtype=float),
            "spoken": np.nan,
            "signed": np.nan,
            "understood_only": four07["understood_only"].to_numpy(dtype=float),
            "signed_only": four07["signed"].to_numpy(dtype=float),
            "spoken_only": four07["spoken"].to_numpy(dtype=float),
            "signed_spoken": four07["spoken_signed"].to_numpy(dtype=float),
            "cell_total": four07["understood"].to_numpy(dtype=float),
        }
        # Marginal-only uk_07 rows (none in the current source) carry the
        # any-modality marginals the vocab_combined view derives.
        marg07_cols = {
            "study": UK07_STUDY_ID,
            "age": marg07["age"].to_numpy(dtype=float),
            "understood": marg07["understood"].to_numpy(dtype=float),
            "spoken": (marg07["spoken"] + marg07["spoken_signed"]).to_numpy(
                dtype=float
            ),
            "signed": (marg07["signed"] + marg07["spoken_signed"]).to_numpy(
                dtype=float
            ),
        }
        if use_subject_codes:
            four07_cols["subject_id"] = four07["subject_id"].to_numpy()
            marg07_cols["subject_id"] = marg07["subject_id"].to_numpy()
        frames.append(pd.DataFrame(four07_cols))
        frames.append(pd.DataFrame(marg07_cols))

    # es_01 (Galeote): the same within-understood partition, derived from the
    # source's recorded totals and their recorded union. Its non-vocal modality is
    # scored per lexical item, the same coding uk_02 and uk_07 apply to signs, and
    # its own association is carried by delta_psi — see
    # JointModelDefinition.include_es01_cells.
    if use_es01_cells:
        four_es, marg_es = load_es01_four_cell()
        four_es_cols = {
            "study": ES01_STUDY_ID,
            "age": four_es["age"].to_numpy(dtype=float),
            "understood": four_es["understood"].to_numpy(dtype=float),
            "spoken": np.nan,
            "signed": np.nan,
            "understood_only": four_es["understood_only"].to_numpy(dtype=float),
            "signed_only": four_es["signed_only"].to_numpy(dtype=float),
            "spoken_only": four_es["spoken_only"].to_numpy(dtype=float),
            "signed_spoken": four_es["signed_spoken"].to_numpy(dtype=float),
            "cell_total": four_es["understood"].to_numpy(dtype=float),
        }
        # Marginal-only es_01 rows keep understood and spoken. `signed` is left NaN
        # rather than passed through: a row only lands here because its cells do not
        # reconcile, which is exactly the condition under which the view masks its
        # gestural total as unusable.
        marg_es_cols = {
            "study": ES01_STUDY_ID,
            "age": marg_es["age"].to_numpy(dtype=float),
            "understood": marg_es["understood"].to_numpy(dtype=float),
            "spoken": marg_es["spoken"].to_numpy(dtype=float),
            "signed": np.nan,
        }
        if use_subject_codes:
            four_es_cols["subject_id"] = four_es["subject_id"].to_numpy()
            marg_es_cols["subject_id"] = marg_es["subject_id"].to_numpy()
        frames.append(pd.DataFrame(four_es_cols))
        frames.append(pd.DataFrame(marg_es_cols))

    # Fail closed on a missing nz_01 source: this block used to tolerate an
    # absent CSV so the model "still builds", which silently fitted VG15
    # without all 111 nz_01 composition observations while the build banner
    # reported include_nz01_cells=True (issue #266). The file has been
    # committed since `abfcc3b`, and tests that need a fixture pool write
    # their own CSV, so absence now means a broken checkout, not a supported
    # configuration.
    if use_nz01_cells:
        nz01_csv = os.path.join(local_env.DATA_DIR, "vocab_data_nz_01.csv")
        if not os.path.exists(nz01_csv):
            raise FileNotFoundError(
                f"include_nz01_cells is set but {nz01_csv} is absent; set "
                "include_nz01_cells=False to fit without nz_01's cross-tab."
            )
        frames.append(load_nz01_produced_cells())
    analysis_df = pd.concat(frames, ignore_index=True)
    analysis_df = analysis_df.dropna(subset=["age"]).reset_index(drop=True)
    ceiling_rows_excluded = 0
    if definition.exclude_us01_spoken_ceiling:
        analysis_df, ceiling_rows_excluded = (
            vocab_data_utils.exclude_us01_spoken_ceiling_rows(analysis_df)
        )

    # Leave-one-study-out (#297 check 5). On the assembled frame, not the merged
    # view, so a cross-tab study loses its cell rows along with its marginals.
    # Read through `getattr` with the empty default: that default is the claim
    # `fit_identity.BACKFILL_DEFAULTS` makes about every joint fit made before
    # the field existed, and `tests/test_fit_identity.py` reads it off this line.
    # The whole block is skipped when nothing is excluded, so the default frame
    # is byte-identical to the one this function built before the field.
    exclude_studies = tuple(getattr(definition, "exclude_studies", ()))
    excluded_study_rows = 0
    if exclude_studies:
        keep = ~analysis_df["study"].isin(exclude_studies)
        excluded_study_rows = int((~keep).sum())
        if excluded_study_rows == 0:
            raise ValueError(
                f"exclude_studies={exclude_studies!r} matched no rows. A "
                "leave-one-study-out check that removes nothing cannot fail, which "
                "is worse than not running it -- check the study codes."
            )
        analysis_df = analysis_df[keep].reset_index(drop=True)

    analysis_df, sign_source_dropped = (
        vocab_data_utils.mask_incomparable_signed_outcomes(
            analysis_df,
            include_signed_only=definition.include_uk01_signed,
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
            analysis_df["study"].astype(str)
            + "::"
            + analysis_df["subject_id"].astype(str)
        )
        analysis_df["subject_key"] = subj_keys
        unique_subjects = sorted(subj_keys.unique())
        subject_map = {s: i for i, s in enumerate(unique_subjects)}
        analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
        n_subjects = len(unique_subjects)

    if sign_lag_same_form_only(definition):
        from vocab_growth.sign_lag_forms import joint_inventory_sizes

        analysis_df["survey_vocab_max"] = joint_inventory_sizes(analysis_df, definition)

    return analysis_df, {
        "use_subject_codes": use_subject_codes,
        "native_only": native_only,
        "non_native_rows_excluded": non_native_rows_excluded,
        "ceiling_rows_excluded": ceiling_rows_excluded,
        "exclude_studies": exclude_studies,
        "excluded_study_rows": excluded_study_rows,
        "sign_source_dropped": sign_source_dropped,
        "use_uk07_cells": use_uk07_cells,
        "use_es01_cells": use_es01_cells,
        "use_nz01_cells": use_nz01_cells,
        "unique_studies": unique_studies,
        "n_subjects": n_subjects,
    }


def prepare_joint_data(
    context: JointContext,
    definition: JointModelDefinition,
):
    """Load and prepare data for the joint model.

    Studies without a cross-tab contribute understood/spoken/signed marginals
    (from the merged view). uk_02, uk_07 and es_01 are taken from their raw CSVs and
    each split into four-cell rows (Dirichlet-Multinomial) and marginal-only rows
    (marginal likelihoods); nz_01 contributes a within-produced three-cell DM.

    es_01's non-vocal modality is called gestural by its source but is scored per
    lexical item on an adapted CDI, so it is the same construct as the other
    sources' signed counts -- see ``JointModelDefinition.include_es01_cells``.
    """
    analysis_df, info = build_joint_analysis_frame(definition)
    native_only = info["native_only"]
    non_native_rows_excluded = info["non_native_rows_excluded"]
    ceiling_rows_excluded = info["ceiling_rows_excluded"]
    sign_source_dropped = info["sign_source_dropped"]
    use_uk07_cells = info["use_uk07_cells"]
    use_es01_cells = info["use_es01_cells"]
    use_nz01_cells = info["use_nz01_cells"]
    unique_studies = info["unique_studies"]
    n_subjects = info["n_subjects"]

    n = len(analysis_df)
    n_u = int(analysis_df["understood"].notna().sum())
    n_s = int(analysis_df["spoken"].notna().sum())
    n_sign = int(analysis_df["signed"].notna().sum())
    n_cells = int(analysis_df["signed_spoken"].notna().sum())
    has_cells = analysis_df["signed_spoken"].notna()
    n_cells_uk02 = int((has_cells & analysis_df["study"].eq(UK02_STUDY_ID)).sum())
    n_cells_uk07 = int((has_cells & analysis_df["study"].eq(UK07_STUDY_ID)).sum())
    n_cells_es01 = int((has_cells & analysis_df["study"].eq(ES01_STUDY_ID)).sum())
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
        ("Four-cell rows (DM, identify psi)", n_cells),
        ("  of which uk_02", n_cells_uk02),
        ("  of which uk_07", n_cells_uk07),
        ("  of which es_01", n_cells_es01),
        ("nz_01 produced-cell rows (DM)", n_prod),
        ("include_uk01_signed", definition.include_uk01_signed),
        ("uk_01 signed-only rows dropped", sign_source_dropped.get("uk_01", 0)),
        # The effective values, not the definition's: the DSE-native sensitivity
        # turns all three off regardless of what was requested, and a banner that
        # reported the request would contradict the row counts above it.
        ("include_nz01_cells", use_nz01_cells),
        ("include_uk07_cells", use_uk07_cells),
        ("include_es01_cells", use_es01_cells),
    ]
    if definition.exclude_us01_spoken_ceiling:
        counts.append(("us_01 WS-ceiling rows excluded", ceiling_rows_excluded))
    if info["exclude_studies"]:
        # Printed with its row count because the count is the check: a
        # leave-one-study-out arm is only as informative as what it removed.
        counts.append((
            f"Studies excluded ({', '.join(info['exclude_studies'])})",
            f"{info['excluded_study_rows']} rows",
        ))
    if native_only:
        # A zero here means the variant has stopped biting and is silently fitting
        # the model of record's data — a failure that looks exactly like a pass.
        counts.append(("Non-native-ceiling rows excluded", non_native_rows_excluded))
    if definition.include_implausible_production:
        # No age bound: this engine's load_data call above passes none, so the
        # reported count has to be taken over the same frame or it misstates what
        # the fit actually reinstated. The other flag is held at the definition's
        # value so the figure is this flag's own net reinstatement.
        counts.append((
            "us_01 implausible production reinstated",
            vocab_data_utils.count_reinstated_implausible_production(
                include_same_day_disagreements=(
                    definition.include_same_day_disagreements
                ),
            ),
        ))
    if definition.include_same_day_disagreements:
        # The rule's own catch. The ceiling-region counts it re-masks when the
        # implausible rule is lifted are counted under that rule's figure above
        # (11 with this flag set, 5 without), so the two lines partition the
        # combined variant's gain over the default pool rather than overlap.
        counts.append((
            "us_01 same-day production disagreements reinstated",
            vocab_data_utils.count_reinstated_same_day_disagreements(),
        ))
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
        plot_and_print_dist(context, d, name)
        return d

    def halfnormal(sigma, name):
        d = pz.HalfNormal(sigma=sigma)
        plot_and_print_dist(context, d, name)
        return d

    if context.report_build:
        heading("Understood trajectory priors", style="bold cyan")
    ell_unit_u_dist = beta(
        definition.ell_unit_u_alpha, definition.ell_unit_u_beta, "ell_unit_u_dist"
    )
    eta_u_dist = halfnormal(definition.eta_u_sigma, "eta_u_dist")
    p_slope_low_u_dist = beta(
        definition.p_slope_low_u_alpha,
        definition.p_slope_low_u_beta,
        "p_slope_low_u_dist",
    )
    p_slope_hi_u_dist = beta(
        definition.p_slope_hi_u_alpha, definition.p_slope_hi_u_beta, "p_slope_hi_u_dist"
    )

    if context.report_build:
        heading("Speak-given-understood (q) priors", style="bold cyan")
    ell_unit_q_dist = beta(
        definition.ell_unit_q_alpha, definition.ell_unit_q_beta, "ell_unit_q_dist"
    )
    eta_q_dist = halfnormal(definition.eta_q_sigma, "eta_q_dist")
    p_slope_low_q_dist = beta(
        definition.p_slope_low_q_alpha,
        definition.p_slope_low_q_beta,
        "p_slope_low_q_dist",
    )
    p_slope_hi_q_dist = beta(
        definition.p_slope_hi_q_alpha, definition.p_slope_hi_q_beta, "p_slope_hi_q_dist"
    )

    if context.report_build:
        heading("Sign-given-understood (r) priors", style="bold cyan")
    ell_unit_sign_dist = beta(
        definition.ell_unit_sign_alpha,
        definition.ell_unit_sign_beta,
        "ell_unit_sign_dist",
    )
    eta_sign_dist = halfnormal(definition.eta_sign_sigma, "eta_sign_dist")
    # Three-anchor hump signed mean (young/peak/old): Beta priors on r at three
    # reference ages, interpolated as a tent meeting at the peak (gp_utils.tent_and_gp).
    p_slope_low_sign_dist = beta(
        definition.p_slope_low_sign_alpha,
        definition.p_slope_low_sign_beta,
        "p_slope_low_sign_dist",
    )
    p_slope_mid_sign_dist = beta(
        definition.p_slope_mid_sign_alpha,
        definition.p_slope_mid_sign_beta,
        "p_slope_mid_sign_dist",
    )
    p_slope_hi_sign_dist = beta(
        definition.p_slope_hi_sign_alpha,
        definition.p_slope_hi_sign_beta,
        "p_slope_hi_sign_dist",
    )

    def kappa_block(kp, suffix):
        if context.report_build:
            heading(f"Kappa priors — {suffix}", style="bold cyan")
        return configure_kappa_priors(context, kp, f"_{suffix}")

    kappa_u_fields = kappa_block(definition.kappa_u, "u")
    kappa_s_fields = kappa_block(definition.kappa_s, "s")
    kappa_sign_fields = kappa_block(definition.kappa_sign, "sign")

    if context.report_build:
        heading(
            "Association (psi) and Dirichlet-Multinomial concentration",
            style="bold cyan",
        )
    log_psi_dist = pz.Normal(mu=definition.log_psi_mu, sigma=definition.log_psi_sigma)
    plot_and_print_dist(context, log_psi_dist, "log_psi_dist")
    log_conc_dist = pz.Normal(
        mu=definition.log_conc_mu, sigma=definition.log_conc_sigma
    )
    plot_and_print_dist(context, log_conc_dist, "log_conc_dist")

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
        **kappa_u_fields,
        **kappa_s_fields,
        **kappa_sign_fields,
        log_psi_dist=log_psi_dist,
        log_conc_dist=log_conc_dist,
        tau_u_sigma=definition.tau_u_sigma,
        tau_q_sigma=definition.tau_q_sigma,
        tau_sign_sigma=definition.tau_sign_sigma,
        tau_psi_sigma=definition.tau_psi_sigma,
        report_max_age_understood=definition.report_max_age_understood,
        report_max_age_signed=definition.report_max_age_signed,
    )
    context.set_model_config(config)


def build_model(context: JointContext, definition: JointModelDefinition):
    """Pipeline stage: construct the model, then write its build report."""
    details = build_model_graph(context, definition)
    report_model_build(context, details)


def build_model_graph(
    context: JointContext, definition: JointModelDefinition
) -> BuildReport:
    """Build the joint sign/speech PyMC model with study + subject random intercepts."""
    build_report = BuildReport()
    config = context.model_config
    df = context.analysis_df
    n_trials = context.model_data.n_trials

    use_subject_re_sign = bool(definition.use_subject_re_sign)
    use_subject_codes = any(
        (
            definition.use_subject_re_u,
            definition.use_subject_re_q,
            use_subject_re_sign,
        )
    )
    observations = prepare_joint_observations(
        df,
        definition,
        n_trials=n_trials,
        use_subject_codes=use_subject_codes,
    )
    if observations.spoken_fallback == SPOKEN_FALLBACK_PAIRED_ONLY:
        build_report.messages.append(
            f"Marginal fallback treatment {observations.spoken_fallback!r}: dropped "
            f"{observations.n_fallback_dropped} child-outcome row(s) with no usable "
            "understood count from the likelihood."
        )

    # Sign -> speech cross-lag (VG25, issue #297): the child's most recent
    # strictly earlier administration wave carrying a signed share of
    # comprehension is the lag source, computed over complete (subject, age)
    # wave groups as VG16's is (issue #242). x = 0 where there is no such wave,
    # which is most of the frame -- a child's first wave, a child with one wave,
    # and every nz_01 row. prev_idx/has_sign_lag_f/r_prev_logit are consumed
    # below when injecting beta_sign_lag * x into the q logit.
    use_sign_cross_lag = bool(getattr(definition, "use_sign_cross_lag", False))
    sign_lag_baseline = getattr(definition, "sign_lag_baseline", "within")
    sign_lag_in_cells = bool(getattr(definition, "sign_lag_in_cells", True))
    sign_prev_idx = np.zeros(len(df), dtype=int)
    has_sign_lag_f = np.zeros(len(df), dtype=float)
    r_prev_logit = np.zeros(len(df), dtype=float)
    if use_sign_cross_lag:
        validate_sign_cross_lag(sign_lag_baseline, use_subject_re_sign)
        sign_prev_idx, has_sign_lag_f, r_prev_logit = (
            prev_wave_sign_share_lag_for_frame(df, definition)
        )
        build_report.messages.append(
            f"Sign cross-lag ({sign_lag_baseline}, in_cells={sign_lag_in_cells}): "
            f"{int(has_sign_lag_f.sum())} of {len(df)} observations have a "
            "prior-wave signed-share source."
        )
        build_report.signed_lag_audit = sign_cross_lag_audit_frame(
            df,
            sign_prev_idx,
            has_sign_lag_f,
            spoken_indices=observations.spoken_spec.indices,
            spoken_is_conditional=observations.spoken_spec.is_conditional,
            cell_indices=observations.idx_cells,
            prod_indices=observations.idx_prod,
        )
        build_report.sign_lag_in_cells = sign_lag_in_cells
    X_obs = observations.X_obs
    n = observations.n
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
        gp_domain_months=definition.gp_domain_months,
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
    L, M = get_hsgp_hyperparams(grids.X_gp_domain_z, (ell_low_z, ell_high_z))

    sa_z, sb_z = standardize_anchor_ages(
        config.slope_anchors, X_obs_mean=X_mean, X_obs_std=X_std
    )

    build_cfg: list[tuple[str, object]] = [
        ("Total observations", n),
        ("Studies", observations.n_studies),
        (
            "Understood / spoken / signed / cells",
            f"{len(observations.idx_u)} / {len(observations.idx_s)} / {len(observations.idx_sign)} / {len(observations.idx_cells)}",
        ),
        (
            "Spoken conditional / marginal",
            f"{observations.spoken_spec.n_conditional} / {observations.spoken_spec.n_marginal}",
        ),
        (
            "Signed conditional / marginal",
            f"{observations.signed_spec.n_conditional} / {observations.signed_spec.n_marginal}",
        ),
        (
            "Child > understood violations (spoken/signed)",
            f"{observations.spoken_spec.n_parent_violations} / {observations.signed_spec.n_parent_violations}",
        ),
        ("n_trials", n_trials),
        ("Age mean / std", (round(X_mean, 1), round(X_std, 1))),
        ("HSGP m / L", (M, L)),
        *kappa_anchor_derived_rows(
            config, X_obs_mean=X_mean, X_obs_std=X_std, suffix="_u"
        ),
        *kappa_anchor_derived_rows(
            config, X_obs_mean=X_mean, X_obs_std=X_std, suffix="_s"
        ),
        *kappa_anchor_derived_rows(
            config, X_obs_mean=X_mean, X_obs_std=X_std, suffix="_sign"
        ),
    ]
    if use_subject_codes:
        build_cfg.append(
            (
                "Subject REs (u/q/sign)",
                f"{definition.use_subject_re_u} / {definition.use_subject_re_q} / {use_subject_re_sign}",
            )
        )
        build_cfg.append(("n_subjects", observations.n_subjects))
    if use_gp_anchor:
        build_cfg.append(
            (
                "GP anchor age (months)",
                f"{anchor_age_months:g} (u={anchor_g_u}, q={anchor_g_q}, sign={anchor_g_sign})",
            )
        )
    if use_sign_cross_lag:
        build_cfg.append(
            (
                "Sign cross-lag (baseline, in cells, lagged rows)",
                f"{sign_lag_baseline} / {sign_lag_in_cells} / "
                f"{int(has_sign_lag_f.sum())}",
            )
        )
    build_report.add_table("Build configuration", build_cfg)

    i_obs0, i_obs1 = grids.i_obs
    i_plot0, i_plot1 = grids.i_plot
    i_query0, i_query1 = grids.i_query

    coords = {
        "all_id": np.arange(n_all),
        "obs_id": np.arange(n),
        "obs_u_id": np.arange(len(observations.idx_u)),
        "obs_s_id": np.arange(len(observations.idx_s)),
        "obs_sign_id": np.arange(len(observations.idx_sign)),
        "obs_cells_id": np.arange(len(observations.idx_cells)),
        "obs_prod_id": np.arange(len(observations.idx_prod)),
        "plot_id": np.arange(n_plot),
        "query_id": np.arange(n_query),
        "study_id": np.arange(observations.n_studies),
        "cell_id": CELL_NAMES,
        "prod_cell_id": PROD_CELL_NAMES,
        "x_dim": np.arange(1),
    }
    if use_subject_codes:
        coords["subject_id"] = np.arange(observations.n_subjects)

    with pm.Model(coords=coords) as model_pm:
        X_all_z_data = pm.Data("X_all_z", X_all_z, dims=("all_id", "x_dim"))
        _ = pm.Data("X_plot", X_plot.flatten(), dims=("plot_id",))
        _ = pm.Data("X_query", X_query.flatten(), dims=("query_id",))
        study_obs = pm.Data("study_obs", observations.study_codes, dims=("obs_id",))
        if use_subject_codes:
            subject_obs = pm.Data(
                "subject_obs", observations.subject_codes, dims=("obs_id",)
            )
        _ = pm.Data(
            "obs_cells_mask",
            observations.has_cells_likelihood.astype(int),
            dims=("obs_id",),
        )
        _ = pm.Data(
            "obs_prod_mask",
            observations.has_prod_likelihood.astype(int),
            dims=("obs_id",),
        )
        _ = pm.Data(
            "obs_u_mask", observations.has_u_likelihood.astype(int), dims=("obs_id",)
        )
        _ = pm.Data(
            "obs_s_mask", observations.has_s_likelihood.astype(int), dims=("obs_id",)
        )
        _ = pm.Data(
            "obs_sign_mask",
            observations.has_sign_likelihood.astype(int),
            dims=("obs_id",),
        )
        s_likelihood_n = pm.Data(
            "s_likelihood_n", observations.spoken_spec.trials, dims=("obs_s_id",)
        )
        s_is_conditional = pm.Data(
            "s_is_conditional",
            observations.spoken_spec.is_conditional.astype(int),
            dims=("obs_s_id",),
        )
        sign_likelihood_n = pm.Data(
            "sign_likelihood_n", observations.signed_spec.trials, dims=("obs_sign_id",)
        )
        sign_is_conditional = pm.Data(
            "sign_is_conditional",
            observations.signed_spec.is_conditional.astype(int),
            dims=("obs_sign_id",),
        )

        # One flag, two means: see definitions.clamp_targets. 'q_only' is
        # truthy, so testing the raw value would clamp both.
        _clamp_u, _clamp_q = clamp_targets(definition.clamp_mean_above_hi_anchor)

        # Latent full-grid trajectories (plain tensors), built by the shared
        # gp_utils helpers. Option D anchors each GP (per-draw zero at the
        # reference age) when the matching flag is set.
        gp_grid = GPGrid.from_age_grids(
            grids,
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
            n_obs=n,
            clamp_above_hi=_clamp_u,
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
            n_obs=n,
            clamp_above_hi=_clamp_q,
        )
        # Signed marginal: three-anchor "tent" hump mean (young/peak/old) + GP; the
        # study random intercept delta_sign is added at obs level below. The GP is
        # anchored at 54 mo (anchor_g_sign) so the tent supplies the hump and the GP
        # only deviates around it.
        sa_young, sa_peak, sa_old = definition.sign_anchor_ages
        sa_young_z, sa_peak_z, sa_old_z = standardize_ages_to_z(
            (sa_young, sa_peak, sa_old), X_obs_mean=X_mean, X_obs_std=X_std
        )
        g_all = tent_and_gp(
            cfg_low=config.p_slope_low_sign_dist,
            cfg_mid=config.p_slope_mid_sign_dist,
            cfg_hi=config.p_slope_hi_sign_dist,
            z_low=sa_young_z,
            z_mid=sa_peak_z,
            z_hi=sa_old_z,
            # Optional: estimate the peak age rather than assert it. Read from the
            # definition so no configuration class changes -- adding a field to a
            # definition class invalidates every existing fit of that class, and
            # nothing here should do that until the change is chosen deliberately.
            cfg_peak=(
                pz.Beta(
                    alpha=definition.sign_peak_prior[0],
                    beta=definition.sign_peak_prior[1],
                )
                if getattr(definition, "sign_peak_prior", None) is not None
                else None
            ),
            cfg_ell=config.ell_unit_sign_dist,
            cfg_eta=config.eta_sign_dist,
            suffix="_sign",
            X_all_z_data=X_all_z_data,
            grid=gp_grid,
            store_deterministic=False,
            anchor_idx=i_anchor if anchor_g_sign else None,
            n_obs=n,
        )

        # This engine centres each latent over studies whose likelihood uses it.
        # Comprehension is informed by U and marginal fallback outcomes, but not
        # by conditional speech/sign counts or the within-understood cells.
        # See study_effects.py for how this reference differs from all-study centring.
        tau_u = pm.HalfNormal("tau_u", sigma=config.tau_u_sigma)
        tau_q = pm.HalfNormal("tau_q", sigma=config.tau_q_sigma)
        tau_sign = pm.HalfNormal("tau_sign", sigma=config.tau_sign_sigma)

        u_informed = informed_studies(
            observations.study_codes,
            observations.idx_u,
            observations.idx_s[~observations.spoken_spec.is_conditional],
            observations.idx_sign[~observations.signed_spec.is_conditional],
        )
        q_informed = informed_studies(
            observations.study_codes,
            observations.idx_s,
            observations.idx_cells,
            observations.idx_prod,
        )
        # Sign-informed studies: those contributing a signing marginal (idx_sign) or
        # a within-understood/produced cross-tab (idx_cells / idx_prod).
        sign_informed = informed_studies(
            observations.study_codes,
            observations.idx_sign,
            observations.idx_cells,
            observations.idx_prod,
        )

        delta_u = zero_sum_study_offsets(
            "delta_u",
            scale=tau_u,
            n_studies=observations.n_studies,
            raw_name="z_u",
            informed=u_informed,
        )
        delta_q = zero_sum_study_offsets(
            "delta_q",
            scale=tau_q,
            n_studies=observations.n_studies,
            raw_name="z_q",
            informed=q_informed,
        )
        delta_sign = zero_sum_study_offsets(
            "delta_sign",
            scale=tau_sign,
            n_studies=observations.n_studies,
            raw_name="z_sign",
            informed=sign_informed,
        )

        child_effects = build_joint_child_effects(
            definition, subject_obs if use_subject_codes else None
        )

        # Standardised observed ages (used by the age-varying kappa functions).
        z_obs = pm.Deterministic("z_obs", X_all_z_data[i_obs0:i_obs1, 0], dims="obs_id")

        # --- obs-level latents WITH study + subject shifts (marginal likelihoods) ---
        f_u_obs = f_u_all[i_obs0:i_obs1] + delta_u[study_obs] + child_effects.understood
        h_obs = h_all[i_obs0:i_obs1] + delta_q[study_obs] + child_effects.spoken_ratio
        g_obs = (
            g_all[i_obs0:i_obs1] + delta_sign[study_obs] + child_effects.signed_ratio
        )

        # Sign -> speech cross-lag (VG25, issue #297). The child's prior-wave
        # signed share of comprehension, as a residual from the signed-ratio
        # trajectory at that wave, shifts their current production ratio q.
        # `beta_sign_lag > 0` means a child who signed a larger share of what
        # they understood then says a larger share of it now; x is 0 with no
        # prior wave, which is most rows.
        #
        # The baseline is a residual from `g`, not from `f_u`, because the
        # predictor is a signed RATIO. `within` leaves the child's own
        # persistent signing standing in the baseline and so subtracts it from
        # the predictor -- the prospective, net-of-standing quantity, and the one
        # VG24's `rho_sign_q` does not already carry. `population` removes the
        # subject shift from the baseline and so retains that standing in the
        # predictor, which with `rho_sign_q` in the model makes it a second,
        # noisier reading of the same thing; it is the registered sensitivity for
        # exactly that reason.
        if use_sign_cross_lag:
            beta_sign_lag = pm.Normal(
                "beta_sign_lag",
                mu=definition.beta_sign_lag_mu,
                sigma=definition.beta_sign_lag_sigma,
            )
            sign_lag_base = g_obs[sign_prev_idx]
            if sign_lag_baseline == "population":
                sign_lag_base = (
                    sign_lag_base - child_effects.signed_ratio[sign_prev_idx]
                )
            x_sign_lag = has_sign_lag_f * (r_prev_logit - sign_lag_base)
            q_sign_lag_term = beta_sign_lag * x_sign_lag
            h_obs = h_obs + q_sign_lag_term
        else:
            q_sign_lag_term = 0.0

        p_u_obs = pm.math.sigmoid(f_u_obs)
        q_obs = pm.math.sigmoid(h_obs)
        r_obs = pm.math.sigmoid(g_obs)

        # --- population+study marginals (NO subject shift) for the cell DMs ---
        # psi is identified from the cross-tab rows (uk_02/uk_07/es_01 four-cell
        # and nz_01 three-cell). The per-child sign offset is co-identified with
        # psi from those same rows, so letting it into the composition makes psi
        # pivot on a thinly-identified RE (measured when uk_02 was the only
        # source: psi 1.78 -> ~2.8, driven almost entirely by the sign subject
        # RE — see notes/202606171200-vg15-subject-re-stabilisation). We
        # therefore keep psi a *population-conditioned* within-understood
        # association by feeding the DMs the study-level marginals only; subject
        # REs still enter every marginal likelihood. When subject REs are off
        # these equal r_obs/q_obs exactly.
        #
        # Two honest consequences, not incidental details (#238). First, this
        # DECOUPLES psi from the child effects rather than empirically
        # separating the two: the cell likelihood is built so they cannot
        # compete, so a sharp child-sign scale posterior is evidence from the
        # marginal rows, not evidence that the cells distinguish association
        # from child heterogeneity. Second, repeated cross-tab visits by one
        # child are conditionally independent in the cell likelihood — no term
        # here carries within-child dependence — so psi's uncertainty may be
        # understated and children with more visits weigh more. The
        # repeated-child sensitivity is tracked in #238.
        #
        # The sign cross-lag is the one term that does cross this line, when
        # `sign_lag_in_cells` is set, and the distinction is worth stating rather
        # than inferring from the code. What is kept out above is a FREE PER-CHILD
        # quantity, which on these thin rows is co-identified with psi and pulled
        # it from 1.78 to about 2.8 when it was let in. `beta_sign_lag` is one
        # scalar multiplying a covariate fixed by the data: it adds a single
        # dimension, not one per child, and it is what brings uk_07's cross-tab
        # children into the coefficient's support at all (191 supporting
        # observations from 129 children against 111 from 80).
        #
        # The honest caveat, because it is not nothing. Under the `within`
        # baseline the predictor itself contains `subject_shift_sign` at the
        # PRIOR wave, so an estimated per-child quantity does reach the
        # composition -- through one scalar coefficient, on lagged rows only,
        # rather than as a free offset per row. The `sign-lag-population` arm is
        # the one in which no estimated per-child quantity reaches the cells at
        # all, since that baseline subtracts the shift back out; reading the two
        # together is what says whether psi moved and why.
        #
        # Added under the flag rather than as `+ (term or 0.0)`, so a model
        # without the lag emits the ops it always did rather than gaining an
        # addition of zero -- which `tests/test_graph_equivalence.py` would see
        # for every other joint model.
        h_obs_pop = h_all[i_obs0:i_obs1] + delta_q[study_obs]
        if use_sign_cross_lag and sign_lag_in_cells:
            h_obs_pop = h_obs_pop + q_sign_lag_term
        q_obs_pop = pm.math.sigmoid(h_obs_pop)
        r_obs_pop = pm.math.sigmoid(g_all[i_obs0:i_obs1] + delta_sign[study_obs])

        # --- population-level latents (no study shift), plot + query ---
        p_u_plot = pm.Deterministic(
            "p_u_plot", pm.math.sigmoid(f_u_all[i_plot0:i_plot1]), dims="plot_id"
        )
        q_plot = pm.Deterministic(
            "q_plot", pm.math.sigmoid(h_all[i_plot0:i_plot1]), dims="plot_id"
        )
        r_plot = pm.Deterministic(
            "r_plot", pm.math.sigmoid(g_all[i_plot0:i_plot1]), dims="plot_id"
        )
        p_u_query = pm.Deterministic(
            "p_u_query", pm.math.sigmoid(f_u_all[i_query0:i_query1]), dims="query_id"
        )
        q_query = pm.Deterministic(
            "q_query", pm.math.sigmoid(h_all[i_query0:i_query1]), dims="query_id"
        )
        r_query = pm.Deterministic(
            "r_query", pm.math.sigmoid(g_all[i_query0:i_query1]), dims="query_id"
        )

        composition = build_composition_parameters(config, observations)

        # --- kappa functions (shared helper — see models.common.build_kappa_for_config) ---
        kappa_u_of_z = build_kappa_for_config(
            config, X_obs_mean=X_mean, X_obs_std=X_std, suffix="_u"
        )
        kappa_s_of_z = build_kappa_for_config(
            config, X_obs_mean=X_mean, X_obs_std=X_std, suffix="_s"
        )
        kappa_sign_of_z = build_kappa_for_config(
            config, X_obs_mean=X_mean, X_obs_std=X_std, suffix="_sign"
        )

        kappa_u_obs = kappa_u_of_z(z_obs)
        kappa_s_obs = kappa_s_of_z(z_obs)
        kappa_sign_obs = kappa_sign_of_z(z_obs)

        # ============================================================
        # Likelihoods
        # ============================================================
        # Understood (all studies)
        p_u_sel = pm.math.clip(p_u_obs[observations.idx_u], EPSILON, 1 - EPSILON)
        k_u = kappa_u_obs[observations.idx_u]
        pm.BetaBinomial(
            "y_u_obs",
            n=n_trials,
            alpha=p_u_sel * k_u,
            beta=(1 - p_u_sel) * k_u,
            observed=observations.y_u,
            dims="obs_u_id",
        )

        # Spoken: nested where U is usable, otherwise marginal over the
        # inventory. Through the shared builder since issue #266 finding 8, so
        # this engine can run the marginal fallback sensitivity the bivariate
        # engines have had since #240. Under the default `product_marginal` it
        # emits the ops it always did, which `tests/test_graph_equivalence.py`
        # checks.
        alpha_s, beta_s = nested_outcome_alpha_beta(
            treatment=observations.spoken_fallback,
            is_conditional=s_is_conditional,
            conditional_p=q_obs[observations.idx_s],
            marginal_p=(p_u_obs * q_obs)[observations.idx_s],
            parent_p=p_u_obs[observations.idx_s],
            parent_kappa=kappa_u_obs[observations.idx_s],
            kappa=kappa_s_obs[observations.idx_s],
            epsilon=EPSILON,
            outcome="s",
            fallback_kappa_sigma=definition.spoken_fallback_kappa_sigma,
        )
        pm.BetaBinomial(
            "y_s_obs",
            n=s_likelihood_n,
            alpha=alpha_s,
            beta=beta_s,
            observed=observations.y_s,
            dims="obs_s_id",
        )

        # Signed: the same treatment. Signing is nested inside comprehension
        # exactly as speech is, so exposing the choice for one outcome and not
        # the other would leave half the exposure unmeasurable.
        alpha_sign, beta_sign = nested_outcome_alpha_beta(
            treatment=observations.spoken_fallback,
            is_conditional=sign_is_conditional,
            conditional_p=r_obs[observations.idx_sign],
            marginal_p=(p_u_obs * r_obs)[observations.idx_sign],
            parent_p=p_u_obs[observations.idx_sign],
            parent_kappa=kappa_u_obs[observations.idx_sign],
            kappa=kappa_sign_obs[observations.idx_sign],
            epsilon=EPSILON,
            outcome="sign",
            fallback_kappa_sigma=definition.spoken_fallback_kappa_sigma,
        )
        pm.BetaBinomial(
            "y_sign_obs",
            n=sign_likelihood_n,
            alpha=alpha_sign,
            beta=beta_sign,
            observed=observations.y_sign,
            dims="obs_sign_id",
        )

        build_composition_likelihood(
            observations,
            composition,
            signed_ratio=r_obs_pop,
            spoken_ratio=q_obs_pop,
        )

        # ============================================================
        # Reporting deterministics (population, plot/query): four-cell + p_any
        # ============================================================
        # These use the POPULATION psi (no study shift), matching p_u_plot/q_plot/
        # r_plot above: the reported composition and p_any are population quantities.
        # Per-study associations are reported separately from `psi_study`.
        for grid, rg, qg, pug in [
            ("plot", r_plot, q_plot, p_u_plot),
            ("query", r_query, q_query, p_u_query),
        ]:
            pi_both = _plackett_pi_both(rg, qg, composition.association)
            pm.Deterministic(f"pi_both_{grid}", pi_both, dims=f"{grid}_id")
            pm.Deterministic(
                f"pi_sign_only_{grid}",
                pm.math.maximum(rg - pi_both, 0.0),
                dims=f"{grid}_id",
            )
            pm.Deterministic(
                f"pi_speak_only_{grid}",
                pm.math.maximum(qg - pi_both, 0.0),
                dims=f"{grid}_id",
            )
            pm.Deterministic(
                f"pi_neither_{grid}",
                pm.math.maximum(1 - rg - qg + pi_both, 0.0),
                dims=f"{grid}_id",
            )
            # data-identified union and independence union (out of understood)
            pm.Deterministic(
                f"p_any_{grid}", pug * (rg + qg - pi_both), dims=f"{grid}_id"
            )
            pm.Deterministic(
                f"p_any_indep_{grid}",
                pug * (1 - (1 - rg) * (1 - qg)),
                dims=f"{grid}_id",
            )

        # ============================================================
        # Per-row quantities, named so a reader outside the fit can reach them
        # ============================================================
        # These per-row outputs support held-out scoring. Sampling excludes them
        # by default to limit trace size; explicit consumers can request them.
        # Cells use population + study ratios, without direct child offsets.
        pm.Deterministic("p_u_obs", p_u_obs, dims="obs_id")
        pm.Deterministic("q_obs", q_obs, dims="obs_id")
        pm.Deterministic("r_obs", r_obs, dims="obs_id")
        pm.Deterministic("kappa_u_obs", kappa_u_obs, dims="obs_id")
        pm.Deterministic("kappa_s_obs", kappa_s_obs, dims="obs_id")
        # Use the same probability calculation for likelihood and held-out rows.
        pm.Deterministic(
            "pi_cells_obs",
            _composition_probabilities(
                pm.math.clip(r_obs_pop, EPSILON, 1 - EPSILON),
                pm.math.clip(q_obs_pop, EPSILON, 1 - EPSILON),
                pm.math.exp(composition.log_association_obs),
            ),
            dims=("obs_id", "cell_id"),
        )
        pm.Deterministic("kappa_sign_obs", kappa_sign_obs, dims="obs_id")

    variables = pymc_utils.get_variables_dict(model_pm)

    context.set_model(model_pm, variables)
    return build_report


# ============================================================
# Pipeline
# ============================================================


def prior_predictive_checks(context: JointContext):
    """Prior predictive checks for the joint sign/speech model."""
    with context.model:
        # No ``mode="FAST_COMPILE"`` -- it makes this fixed-cost stage 20-40x
        # slower for the same draws. See the note at the same call in
        # ``common.py`` and notes/202608251100-prior-predictive-compile-mode.md.
        prior = pm.sample_prior_predictive(
            draws=500, random_seed=context.sampling.random_seed,
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


def diagnostics(context: JointContext, definition: JointModelDefinition):
    """Run diagnostics on the posterior samples.

    The shared engine prioritises ``psi`` and ``conc`` in parameter plots.
    Per-outcome LOO scores cover the three marginal count likelihoods. They
    hold out one likelihood term, with the other terms' denominators fixed.
    They therefore do not hold out a complete administration.

    The separate administration-level score includes every factor, including
    ``cells_obs`` and ``nz_prod_cells_obs``. It assesses the predictive
    contribution of the composition association as well as the count models.

    **A sign cross-lag definition (VG25) suppresses the scores that leak.** Its
    predictor reads an earlier wave's ``signed`` and ``understood`` counts as
    fixed covariates of every later row it feeds, so leaving one of those
    likelihood terms out does not remove that count from the model: the
    "held-out" score still conditions on the held-out outcome, and Pareto-k
    checks the importance-sampling approximation rather than this leakage, so it
    cannot flag it. ``y_u_obs`` and ``y_sign_obs`` are therefore not computed,
    and neither is the administration-level score, which bundles both of them
    with the two composition terms the lag also enters. What is kept is
    ``y_s_obs``, labelled for what it estimates: prediction of a spoken count
    conditional on the child's observed sign history, not unconditional
    new-observation prediction. This is #242's finding for VG16 on the other
    engine, and the same remedy -- ``scripts/wave_forward_score.py`` is the
    forward-chaining score that replaces what is suppressed here.

    ``definition`` is taken for the pair plot's ordering, which is issue #233's
    problem on this engine: ArviZ caps the grid at ``floor(sqrt(max_subplots))``
    -- six variables -- and this engine led with ``psi`` and ``conc`` and then
    took build order, which puts the mean functions first. So VG25's
    ``beta_sign_lag``, built eighteenth, never rendered, and neither did any of
    VG24's three child correlations -- while both reports send the reader to the
    pair plot to inspect exactly those. The ordering now comes from
    :func:`~vocab_growth.models.diagnostics_utils.pair_plot_priority`, one
    implementation shared with the bivariate engine.
    """
    _prioritise = pair_plot_var_names_fn(
        definition, set(context.trace.posterior.data_vars)
    )

    if not getattr(definition, "use_sign_cross_lag", False):
        _shared_diagnostics(
            context,
            var_names_fn=_prioritise,
            round_to=4,
            loo_var_names=(
                ("y_u_obs", "words understood"),
                ("y_s_obs", "words spoken"),
                ("y_sign_obs", "words signed"),
            ),
            # Every factor of an administration, the two composition terms
            # included (issue #266 finding 4). Those terms are what identify
            # `psi`, this model's headline association, and the per-outcome
            # scores above omit them entirely -- so before this the LOO never
            # scored the quantity the model exists to estimate.
            administration_factors=(
                LikelihoodFactor("y_u_obs", "obs_u_mask"),
                LikelihoodFactor("y_s_obs", "obs_s_mask"),
                LikelihoodFactor("y_sign_obs", "obs_sign_mask"),
                LikelihoodFactor("cells_obs", "obs_cells_mask"),
                LikelihoodFactor("nz_prod_cells_obs", "obs_prod_mask"),
            ),
        )
        return

    console.print(
        "[yellow]Understood and signed LOO are not computed for this model: the "
        "sign cross-lag predictor embeds an earlier wave's observed signed and "
        "understood counts, so a pointwise leave-one-out score would still "
        "condition on the held-out count through the later terms it feeds "
        "(issue #242). The administration-level score is suppressed for the "
        "same reason -- it bundles both of those factors with the two "
        "composition terms the lag also enters. The spoken score below is "
        "prediction conditional on the child's observed sign history; "
        "`scripts/wave_forward_score.py` is the forward-chaining "
        "replacement.[/yellow]"
    )
    # No `administration_factors`, for the reason the message gives: with the
    # lag in the graph no pointwise hold-out of an administration is clean, so
    # the score would read as leave-one-administration-out while conditioning on
    # the administration it claims to have left out. Confining the lag to the
    # spoken marginal (`sign_lag_in_cells=False`) does not change this: the
    # source wave's counts still reach later rows through the predictor, which
    # is where the leak is.
    _shared_diagnostics(
        context,
        var_names_fn=_prioritise,
        round_to=4,
        loo_var_names=(
            (
                "y_s_obs",
                "words spoken (conditional on the child's observed sign history)",
            ),
        ),
    )


def _extract_produced_cell_observations(
    df: pd.DataFrame,
    has_prod: np.ndarray,
) -> np.ndarray:
    """Observed nz_01 produced-cell counts for the rows flagged by ``has_prod``.

    Counts only. It returned the matching ages as well until 2026-09-01, and no
    caller ever read them -- the tuple target hid that from ruff's unused-variable
    rule, which is why they outlived the ``prod_cell_ages`` *field* they were the
    source of.
    """
    if not has_prod.any():
        return np.zeros((0, len(PROD_CELL_NAMES)), dtype=int)

    missing = [col for col in PROD_CELL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "obs_prod_mask marks produced-cell rows, but analysis_df is missing "
            f"columns: {', '.join(missing)}"
        )

    return np.asarray(df.loc[has_prod, PROD_CELL_COLUMNS], dtype=int)


def sample_posterior_predictive(
    context: JointContext, definition: JointModelDefinition
):
    """Posterior predictive for the observed cell-count likelihoods.

    ``definition`` is required but unread. Every engine's predictive stage is called
    with the same two arguments so the catalogue can describe them uniformly, and a
    default of ``None`` here only hid that no caller ever omitted it. Keeping the
    parameter is the contract; defaulting it was the defect.
    """
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
    save_trace(trace, context.reporting.output_dir)

    # Observed four-cell counts / ages, over every within-understood cross-tab
    # source the fit admitted (uk_02, and uk_07 / es_01 when enabled) — not uk_02
    # alone. Use the stored training mask so held-out four-cell rows stay aligned
    # with posterior_predictive["cells_obs"].
    df = context.analysis_df
    has_cells = np.array(trace.constant_data["obs_cells_mask"].values, dtype=bool)
    cell_counts = np.asarray(
        df.loc[has_cells, ["understood_only", "signed_only", "spoken_only", "signed_spoken"]],
        dtype=int,
    )
    cell_studies = np.asarray(df.loc[has_cells, "study"], dtype=str)
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
    prod_counts = _extract_produced_cell_observations(df, has_prod)
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

    # The studies that identify psi, mirroring `psi_informed` in build_model:
    # the within-understood four-cell rows plus nz_01's within-produced
    # three-cell rows. `obs_cells_mask` and `obs_prod_mask` are the very masks
    # that produced `idx_cells`/`idx_prod` there, so this cannot drift from the
    # likelihood, and it respects a K-fold holdout for the same reason.
    psi_informed_studies = tuple(
        sorted(set(df.loc[has_cells | has_prod, "study"].astype(str)))
    )

    samples = JointModelSamples(
        X_plot=np.array(trace.constant_data["X_plot"].values),
        X_query=np.array(trace.constant_data["X_query"].values),
        p_u_plot=extract_posterior(trace, "p_u_plot", "plot_id"),
        q_plot=extract_posterior(trace, "q_plot", "plot_id"),
        r_plot=extract_posterior(trace, "r_plot", "plot_id"),
        pi_both_plot=extract_posterior(trace, "pi_both_plot", "plot_id"),
        p_any_plot=extract_posterior(trace, "p_any_plot", "plot_id"),
        p_any_indep_plot=extract_posterior(trace, "p_any_indep_plot", "plot_id"),
        pi_neither_plot=extract_posterior(trace, "pi_neither_plot", "plot_id"),
        pi_sign_only_plot=extract_posterior(trace, "pi_sign_only_plot", "plot_id"),
        pi_speak_only_plot=extract_posterior(trace, "pi_speak_only_plot", "plot_id"),
        p_u_query=extract_posterior(trace, "p_u_query", "query_id"),
        q_query=extract_posterior(trace, "q_query", "query_id"),
        r_query=extract_posterior(trace, "r_query", "query_id"),
        p_any_query=extract_posterior(trace, "p_any_query", "query_id"),
        p_any_indep_query=extract_posterior(trace, "p_any_indep_query", "query_id"),
        psi=np.array(trace.posterior["psi"].stack(sample=("chain", "draw")).values),
        psi_study=np.array(
            trace.posterior["psi_study"].stack(sample=("chain", "draw")).values
        ),
        psi_study_names=sorted(context.analysis_df["study"].unique()),
        psi_informed_studies=psi_informed_studies,
        tau_psi=(
            np.array(trace.posterior["tau_psi"].stack(sample=("chain", "draw")).values)
            if "tau_psi" in trace.posterior
            else None
        ),
        cell_obs=cell_counts,
        cell_pred=cell_pred,
        cell_studies=cell_studies,
        prod_cell_obs=prod_counts,
        prod_cell_pred=prod_pred,
    )
    context.set_model_samples(samples)


def summarise_psi_by_study(
    psi_study: np.ndarray,
    psi_study_names: Sequence[str],
    psi_informed_studies: Sequence[str],
    *,
    ci_prob: float,
    inner: float,
) -> pd.DataFrame:
    """Per-study association rows, for the studies that actually inform psi.

    Membership of ``psi_informed_studies`` is the criterion, and it has to be:
    a study with no cross-tabulation carries ``delta_psi == 0`` and therefore
    sits at exactly the population value, but so does *every* study when only
    one of them is informed, because the model then takes its degenerate branch
    and pins the offsets to zero rather than estimating a contrast. Testing
    ``psi_study == psi`` as a proxy for "uninformed" therefore discarded the one
    row worth printing under a single-source pool -- Down syndrome native-only
    -- so no per-study table was written for the only study that informs the
    association (#266).

    Returns an empty frame when no study informs the association, so the caller
    writes no file rather than an empty one.
    """
    informed = set(psi_informed_studies)
    rows = []
    for i, name in enumerate(psi_study_names):
        if name not in informed:
            continue
        draws = np.asarray(psi_study[i])
        lo50, hi50 = intervals.interval_1d(draws, inner, "hdi")
        lo, hi = intervals.interval_1d(draws, ci_prob, "hdi")
        rows.append({
            "study": name,
            "psi_median": float(np.median(draws)),
            "psi_ci50_lo": float(lo50), "psi_ci50_hi": float(hi50),
            "psi_ci_lo": float(lo), "psi_ci_hi": float(hi),
            "P_psi_gt_1": float((draws > 1).mean()),
        })
    return pd.DataFrame(rows)


def posterior_summary(context: JointContext):
    s = context.model_samples
    n_trials = context.model_data.n_trials
    ci_prob = context.reporting.ci_prob
    ci_kind = context.reporting.interval_kind
    inner = intervals.INNER_CI_PROB
    od = context.reporting.output_dir
    # Per-quantity reporting caps (vocab_growth.reporting_ages). r and p_any are
    # ratios of understood built from the signed ratio, so the tighter of the
    # comprehension and signing caps binds; until #238 both tables carried the
    # signing cap alone, so they ran to 84 months beside figures that (correctly)
    # stopped at 72.
    config = context.model_config
    understood_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.UNDERSTOOD
    )
    signed_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.SIGNED
    )
    ratio_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.RATIO_OF_UNDERSTOOD
    )
    sign_ratio_cap = reporting_ages.max_age_for_sign_ratio(config)

    def probability_summary(X, draws, prefix, label, max_age=None):
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
        d = posterior_analysis.trim_reported_ages(d, max_age)
        d.to_csv(os.path.join(od, f"posterior_summary_{prefix}.csv"), index=False)
        dataframe_table(d.round(3), title=label, show_index=False)
        return d

    def ratio_summary(X, draws, prefix, max_age=None):
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
        d = posterior_analysis.trim_reported_ages(d, max_age)
        d.to_csv(os.path.join(od, f"posterior_summary_{prefix}.csv"), index=False)
        return d

    probability_summary(s.X_query, s.p_u_query, "u", "Words understood", understood_cap)
    probability_summary(s.X_query, s.p_u_query * s.q_query, "s", "Words spoken")
    probability_summary(
        s.X_query, s.p_u_query * s.r_query, "sign", "Words signed", signed_cap
    )
    ratio_summary(s.X_query, s.r_query, "r", sign_ratio_cap)
    ratio_summary(s.X_query, s.q_query, "q", ratio_cap)

    # Whole-month companions to the tables above. This engine draws no predictive
    # counts on the plot grid, so these carry the expected count only — matching
    # what probability_summary reports at query ages, which likewise has no Y_*
    # or bucket columns.
    #
    # n_obs must count the administrations that observed *this* outcome, as the
    # other engines do by passing their per-outcome x_obs; the three outcomes
    # have different coverage here, so the whole frame's ages would report the
    # same total for all three and overstate every one of them.
    analysis_df = context.analysis_df
    for draws, suffix, column, label in (
        (s.p_u_plot, "u", "understood", "words understood"),
        (s.p_u_plot * s.q_plot, "s", "spoken", "words spoken"),
        (s.p_u_plot * s.r_plot, "sign", "signed", "words signed"),
    ):
        observed_ages = (
            analysis_df.loc[analysis_df[column].notna(), "age"]
            if column in analysis_df.columns
            else None
        )
        emit_monthly_summary(
            output_dir=od,
            X_plot=s.X_plot,
            p_plot=draws,
            y_plot=None,
            X_obs=observed_ages,
            n_trials=n_trials,
            ci_prob=ci_prob,
            interval_kind=ci_kind,
            suffix=suffix,
            outcome_label=label,
            y_label=f"Expected {label}",
            dataframes=context.dataframes,
            plots=context.plots,
            max_age_months=reporting_ages.max_age_for(
                context.model_config, _QUANTITY_BY_SUFFIX[suffix]
            ),
        )

    # Data-identified p_any vs independence (expected counts)
    Ey = s.p_any_query * n_trials
    Ey_i = s.p_any_indep_query * n_trials
    p_any_o = intervals.bands(s.p_any_query, ci_prob, ci_kind, sample_axis=1)
    p_any_in = intervals.bands(s.p_any_query, inner, ci_kind, sample_axis=1)
    Ey_o = intervals.bands(Ey, ci_prob, ci_kind, sample_axis=1)
    Ey_in = intervals.bands(Ey, inner, ci_kind, sample_axis=1)
    pany = pd.DataFrame({
        "age_months": s.X_query,
        "p_any_median": np.median(s.p_any_query, axis=1),
        "p_any_ci50_lo": p_any_in[:, 0],
        "p_any_ci50_hi": p_any_in[:, 1],
        "p_any_ci_lo": p_any_o[:, 0],
        "p_any_ci_hi": p_any_o[:, 1],
        "Ey_any_median": np.median(Ey, axis=1),
        "Ey_any_ci50_lo": Ey_in[:, 0],
        "Ey_any_ci50_hi": Ey_in[:, 1],
        "Ey_any_ci_lo": Ey_o[:, 0],
        "Ey_any_ci_hi": Ey_o[:, 1],
        "p_any_indep_median": np.median(s.p_any_indep_query, axis=1),
        "Ey_any_indep_median": np.median(Ey_i, axis=1),
    })
    # Total expressive is conditioned on understood and a function of the
    # signed ratio, so the tighter of those two caps binds
    # (reporting_ages.max_age_for_sign_ratio).
    pany = posterior_analysis.trim_reported_ages(pany, sign_ratio_cap)
    pany.to_csv(os.path.join(od, "posterior_summary_p_any.csv"), index=False)
    dataframe_table(pany.round(3), title="Total expressive p_any (identified vs independence)", show_index=False)

    # psi summary (HDI: psi is a right-skewed association ratio)
    psi = s.psi
    psi_lo50, psi_hi50 = intervals.interval_1d(psi, inner, "hdi")
    psi_lo, psi_hi = intervals.interval_1d(psi, ci_prob, "hdi")
    pct = int(round(ci_prob * 100))
    psi_df = pd.DataFrame({
        "psi_median": [float(np.median(psi))],
        "psi_ci50_lo": [float(psi_lo50)],
        "psi_ci50_hi": [float(psi_hi50)],
        "psi_ci_lo": [float(psi_lo)],
        "psi_ci_hi": [float(psi_hi)],
        "P_psi_gt_1": [float((psi > 1).mean())],
    })
    psi_df.to_csv(os.path.join(od, "posterior_summary_psi.csv"), index=False)
    key_value_table("Association psi (population)", [
        ("psi median", round(float(np.median(psi)), 3)),
        (f"psi {pct}% HDI", (round(float(psi_lo), 3), round(float(psi_hi), 3))),
        ("P(psi > 1)", round(float((psi > 1).mean()), 3)),
    ])

    # Per-study association. This is the primary read on a parameter the sources
    # disagree about: the population value is a shrunk centre, not a consensus.
    psi_study_df = summarise_psi_by_study(
        s.psi_study,
        s.psi_study_names,
        s.psi_informed_studies,
        ci_prob=ci_prob,
        inner=inner,
    )
    if not psi_study_df.empty:
        psi_study_df.to_csv(
            os.path.join(od, "posterior_summary_psi_study.csv"), index=False
        )
        dataframe_table(
            psi_study_df.round(3),
            title="Association psi by study (cross-tab sources only)",
            show_index=False,
        )
    if s.tau_psi is not None:
        tau_lo, tau_hi = intervals.interval_1d(s.tau_psi, ci_prob, "hdi")
        key_value_table("Between-study SD of log psi", [
            ("tau_psi median", round(float(np.median(s.tau_psi)), 3)),
            (f"tau_psi {pct}% HDI", (round(float(tau_lo), 3), round(float(tau_hi), 3))),
        ])
    elif not psi_study_df.empty:
        # No tau_psi means fewer than two informed studies, so the model took its
        # degenerate branch and pinned every delta_psi to zero. The row above is
        # then the population value repeated, not an independently estimated
        # per-study one, and saying so is the difference between reporting a
        # single source's association and appearing to report heterogeneity.
        console.print(
            "Only one study informs the association, so there is no between-study "
            "contrast to estimate: delta_psi is pinned to zero and the per-study "
            "value above equals the population value by construction."
        )


# ============================================================
# Plots
# ============================================================


def run_joint_plots(context: JointContext):
    s = context.model_samples
    n_trials = context.model_data.n_trials
    od = context.reporting.output_dir
    ci_prob = context.reporting.ci_prob
    ci_kind = context.reporting.interval_kind
    X = s.X_plot

    # Every curve below is a function of comprehension, and the sign-bearing ones
    # are functions of the signed ratio too, so each inherits the reporting age of
    # its narrowest input (vocab_growth.reporting_ages). Without this the tables
    # stop at the cap while the figures beside them run to the end of the plot
    # grid, and the pair disagree in print about where the evidence ends.
    config = context.model_config
    ratio_cap = reporting_ages.max_age_for(
        config, reporting_ages.ReportedQuantity.RATIO_OF_UNDERSTOOD
    )
    sign_ratio_cap = reporting_ages.max_age_for_sign_ratio(config)

    def _keep(max_age):
        if max_age is None:
            return np.ones(len(X), dtype=bool)
        return np.asarray(X) <= max_age

    # p_any, the four-cell composition and r(a) all involve the signed ratio, so
    # they stop at the tighter of the comprehension and signing caps.
    keep_sign = _keep(sign_ratio_cap)
    keep_u = _keep(ratio_cap)
    X_sign = np.asarray(X)[keep_sign]
    X_u = np.asarray(X)[keep_u]

    # Expected union counts under the fitted association and under independence.
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    id_med = np.median(s.p_any_plot[keep_sign, :], axis=1) * n_trials
    id_hdi = intervals.bands(
        s.p_any_plot[keep_sign, :] * n_trials, ci_prob, ci_kind, sample_axis=1
    )
    ind_med = np.median(s.p_any_indep_plot[keep_sign, :], axis=1) * n_trials
    ax.fill_between(X_sign, id_hdi[:, 0], id_hdi[:, 1], alpha=0.20, color="C0")
    ax.plot(X_sign, id_med, lw=3, color="C0", label="Data-identified p_any (median)")
    ax.plot(X_sign, ind_med, lw=2.5, ls="--", color="C3", label="Assuming independence")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Expected words produced (any modality)")
    ax.set_ylim(0, n_trials + 50)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title(
        "Total expressive vocabulary under fitted association and independence"
    )
    fig.savefig(os.path.join(od, "p_any_identified_vs_bound.png"), dpi=300)
    fig.savefig(os.path.join(od, "p_any_identified_vs_bound.svg"))
    plot_io.save_plot_data(
        od,
        "p_any_identified_vs_bound",
        pd.DataFrame(
            {
                "age_months": X_sign,
                "identified_median": id_med,
                "identified_ci_lo": id_hdi[:, 0],
                "identified_ci_hi": id_hdi[:, 1],
                "independence_median": ind_med,
            }
        ),
    )
    context.plots["p_any_identified_vs_bound"] = fig
    plt.close(fig)

    # 2) Four-cell composition trajectory (fractions of understood)
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    comp = {
        "neither": (np.median(s.pi_neither_plot[keep_sign, :], axis=1), "C7"),
        "sign-only": (np.median(s.pi_sign_only_plot[keep_sign, :], axis=1), "C2"),
        "sign+speech": (np.median(s.pi_both_plot[keep_sign, :], axis=1), "C4"),
        "speak-only": (np.median(s.pi_speak_only_plot[keep_sign, :], axis=1), "C1"),
    }
    for lab, (med, c) in comp.items():
        ax.plot(X_sign, med, lw=2.5, color=c, label=lab)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Fraction of understood words")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("Within-understood composition (sign-only → both → speak-only)")
    fig.savefig(os.path.join(od, "four_cell_composition.png"), dpi=300)
    fig.savefig(os.path.join(od, "four_cell_composition.svg"))
    plot_io.save_plot_data(
        od,
        "four_cell_composition",
        pd.DataFrame({"age_months": X_sign, **{k: v[0] for k, v in comp.items()}}),
    )
    context.plots["four_cell_composition"] = fig
    plt.close(fig)

    # 2b) What signing is worth to the child, and when speech takes over.
    #
    # The four-cell composition above is a set of *fractions of understood*, which
    # answers "what proportion of the words a child knows can they sign?" but not
    # "how much does signing add to what this child can express?". That second
    # question is the one a parent, teacher or therapist asks, and it needs word
    # counts and a denominator of the child's own expressive vocabulary rather
    # than of comprehension. The two give very different-looking answers from the
    # same posterior, which is exactly why both belong here.
    #
    # The pi_* cells are conditional on the word being understood, so they are
    # scaled by p_u to become counts. Reading them as unconditional inflates every
    # cell by 1/p_u -- a factor of about fifty at the youngest modelled ages.
    spoken_w = s.p_u_plot[keep_sign, :] * s.q_plot[keep_sign, :] * n_trials
    any_w = s.p_any_plot[keep_sign, :] * n_trials
    sign_only_w = (
        s.p_u_plot[keep_sign, :] * s.pi_sign_only_plot[keep_sign, :] * n_trials
    )
    both_w = s.p_u_plot[keep_sign, :] * s.pi_both_plot[keep_sign, :] * n_trials
    speak_only_w = (
        s.p_u_plot[keep_sign, :] * s.pi_speak_only_plot[keep_sign, :] * n_trials
    )

    eps = 1e-9
    uplift = any_w / np.maximum(spoken_w, eps)
    sign_only_share = sign_only_w / np.maximum(any_w, eps)

    def _band(arr):
        return np.median(arr, axis=1), intervals.bands(
            arr, ci_prob, ci_kind, sample_axis=1
        )

    up_med, up_ci = _band(uplift)
    sh_med, sh_ci = _band(sign_only_share)
    so_med, so_ci = _band(sign_only_w)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    for arr, lab, c in (
        (speak_only_w, "Speech only", "C1"),
        (both_w, "Both sign and speech", "C4"),
        (sign_only_w, "Sign only", "C2"),
    ):
        med, hdi = _band(arr)
        ax.fill_between(X_sign, hdi[:, 0], hdi[:, 1], alpha=0.18, color=c)
        ax.plot(X_sign, med, lw=2.5, color=c, label=lab)
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Words")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("How expressive vocabulary is expressed")
    fig.savefig(os.path.join(od, "signing_composition_words.png"), dpi=300)
    fig.savefig(os.path.join(od, "signing_composition_words.svg"))
    context.plots["signing_composition_words"] = fig
    plt.close(fig)

    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    ax.fill_between(X_sign, up_ci[:, 0], up_ci[:, 1], alpha=0.20, color="C0")
    ax.plot(
        X_sign,
        up_med,
        lw=3,
        color="C0",
        label="Expressive vocabulary as a multiple of spoken",
    )
    ax.axhline(1.0, ls=":", color="grey", label="speech alone")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("p_any / spoken")
    ax.legend(loc="upper right", frameon=True)
    ax.set_title("What counting sign adds to a child's own expressive vocabulary")
    fig.savefig(os.path.join(od, "signing_uplift.png"), dpi=300)
    fig.savefig(os.path.join(od, "signing_uplift.svg"))
    context.plots["signing_uplift"] = fig
    plt.close(fig)

    plot_io.save_plot_data(
        od,
        "signing_profile",
        pd.DataFrame(
            {
                "age_months": X_sign,
                "spoken_median": np.median(spoken_w, axis=1),
                "any_median": np.median(any_w, axis=1),
                "sign_only_median": so_med,
                "sign_only_ci_lo": so_ci[:, 0],
                "sign_only_ci_hi": so_ci[:, 1],
                "both_median": np.median(both_w, axis=1),
                "speak_only_median": np.median(speak_only_w, axis=1),
                "uplift_median": up_med,
                "uplift_ci_lo": up_ci[:, 0],
                "uplift_ci_hi": up_ci[:, 1],
                "sign_only_share_median": sh_med,
                "sign_only_share_ci_lo": sh_ci[:, 0],
                "sign_only_share_ci_hi": sh_ci[:, 1],
            }
        ),
    )

    # These two summary TABLES are written from the plot stage, not from
    # `posterior_summary`, and that placement is load-bearing rather than untidy:
    # `scripts/regenerate_plots.py` re-runs only the plot stage, so a table written
    # here can be refreshed by a replot, while the same table moved into
    # `posterior_summary` could only be refreshed by a reporting-quality refit. The
    # next time a reporting cap moves, that is the difference between minutes and
    # multiple gigabytes. (Recorded until now only in a `KNOWN_STALE` comment inside
    # tests/test_reporting_age_policy.py.)
    _signing_milestones(X_sign, sign_only_w, both_w, speak_only_w, ci_prob).to_csv(
        os.path.join(od, "signing_milestones.csv"), index=False
    )

    # 3) psi posterior
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
    ax.hist(s.psi, bins=60, color="C4", alpha=0.8)
    ax.axvline(1.0, ls=":", color="grey", label="independence (psi=1)")
    ax.axvline(
        float(np.median(s.psi)),
        ls="-",
        color="C0",
        label=f"median {np.median(s.psi):.2f}",
    )
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
    r_med = np.median(s.r_plot[keep_sign, :], axis=1)
    r_hdi = intervals.bands(s.r_plot[keep_sign, :], ci_prob, ci_kind, sample_axis=1)
    q_med = np.median(s.q_plot[keep_u, :], axis=1)
    ax.fill_between(X_sign, r_hdi[:, 0], r_hdi[:, 1], alpha=0.18, color="C2")
    ax.plot(X_sign, r_med, lw=3, color="C2", label="r(a) signed")
    ax.plot(X_u, q_med, lw=3, color="C1", label="q(a) spoken")
    ax.set_xlabel("Age (months)")
    ax.set_ylabel("Fraction of understood words")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Signed vs spoken ratio (population-level)")
    fig.savefig(os.path.join(od, "signed_vs_spoken_rate.png"), dpi=300)
    fig.savefig(os.path.join(od, "signed_vs_spoken_rate.svg"))
    context.plots["signed_vs_spoken_rate"] = fig
    plt.close(fig)

    # 4b) Per-draw peak of the FULL fitted signed-ratio curve r(a) over the
    # reported grid. `peak_unit_sign` locates the tent's movable middle knot;
    # the three anchor heights are sampled independently and a GP departure is
    # then added, so the knot need not be the maximum of the fitted curve
    # (#238). This table is the quantity to quote as "the age at which the
    # signed ratio peaks"; the knot position is a parameter, reported as such.
    # Schema matches expected_learning_rate_peak.csv (#234): median + HDI over
    # the draws whose maximum is interior, and the boundary share as the
    # censoring disclosure. multi_peak_draw_share flags GP-induced multimodality
    # (more than one local maximum), which would make "the" peak ill-defined.
    r_grid = s.r_plot[keep_sign, :]
    peak_idx = np.argmax(r_grid, axis=0)
    boundary = (peak_idx == 0) | (peak_idx == r_grid.shape[0] - 1)
    peak_ages = np.where(boundary, np.nan, np.asarray(X_sign)[peak_idx])
    interior = peak_ages[np.isfinite(peak_ages)]
    peak_lo, peak_hi = intervals.interval_1d(interior, ci_prob, "hdi")
    local_max = (r_grid[1:-1, :] > r_grid[:-2, :]) & (r_grid[1:-1, :] > r_grid[2:, :])
    pd.DataFrame(
        {
            "peak_age_median_months": [
                float(np.median(interior)) if interior.size else np.nan
            ],
            "peak_age_ci_lo_months": [float(peak_lo)],
            "peak_age_ci_hi_months": [float(peak_hi)],
            "boundary_draw_share": [float(boundary.mean())],
            "multi_peak_draw_share": [float((local_max.sum(axis=0) > 1).mean())],
            "support_lo_months": [float(np.asarray(X_sign)[0])],
            "support_hi_months": [float(np.asarray(X_sign)[-1])],
        }
    ).to_csv(os.path.join(od, "signed_ratio_peak.csv"), index=False)

    # 5) Four-cell PPC (observed vs predicted cell totals), one panel per
    # source. This was a single figure named and titled uk_02 until #238, while
    # its bars aggregated every within-understood four-cell source; the per-study
    # associations differ severalfold, so an aggregate can hide offsetting
    # source-specific errors as well as mislabelling what is shown.
    lo, hi = 100 * (1 - ci_prob) / 2, 100 * (1 + ci_prob) / 2
    sources = sorted(set(np.asarray(s.cell_studies, dtype=str).tolist()))
    if sources:
        fig, axes = plt.subplots(
            1, len(sources), figsize=plot_styles.FIGSIZE_XL, sharey=False
        )
        for panel, source in zip(np.atleast_1d(axes), sources, strict=True):
            rows = np.asarray(s.cell_studies, dtype=str) == source
            obs_tot = s.cell_obs[rows].sum(axis=0)  # (4,)
            pred_tot = s.cell_pred[rows].sum(axis=0)  # (4, n_samples)
            pred_med = np.median(pred_tot, axis=1)
            pred_lo = np.percentile(pred_tot, lo, axis=1)
            pred_hi = np.percentile(pred_tot, hi, axis=1)
            yerr = np.vstack([pred_med - pred_lo, pred_hi - pred_med])
            xpos = np.arange(4)
            panel.bar(xpos - 0.18, obs_tot, width=0.36, color="C0", label="observed")
            panel.bar(
                xpos + 0.18,
                pred_med,
                width=0.36,
                color="C3",
                alpha=0.7,
                label="predicted (median)",
                yerr=yerr,
                capsize=4,
            )
            panel.set_xticks(xpos)
            panel.set_xticklabels(CELL_NAMES, rotation=30, ha="right")
            panel.set_title(source)
        np.atleast_1d(axes)[0].set_ylabel("Total cell count")
        np.atleast_1d(axes)[0].legend(frameon=True)
        fig.suptitle("Four-cell posterior predictive check, by source")
        fig.tight_layout()
        fig.savefig(os.path.join(od, "four_cell_ppc_by_source.png"), dpi=300)
        fig.savefig(os.path.join(od, "four_cell_ppc_by_source.svg"))
        context.plots["four_cell_ppc_by_source"] = fig
        plt.close(fig)

    # 6) nz_01 produced-cell PPC (observed vs predicted produced-cell totals)
    if s.prod_cell_obs.size and s.prod_cell_pred.size:
        fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_MD)
        obs_tot = s.prod_cell_obs.sum(axis=0)  # (3,)
        pred_tot = s.prod_cell_pred.sum(axis=0)  # (3, n_samples)
        pred_med = np.median(pred_tot, axis=1)
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


def joint_stages(
    definition: JointModelDefinition,
) -> list[tuple[str, Callable[[JointContext], None]]]:
    """The ordered ``(stage name, stage fn)`` list for this engine's fit.

    Exposed separately from :func:`fit_joint_model` so a caller can substitute a
    single stage and still run the identical pipeline — the parameter-recovery
    harness swaps stage 0 (data preparation) for a loader that injects a
    simulated analysis frame (see :mod:`vocab_growth.recovery.refit`).
    """
    return [
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
        ("Diagnostics", lambda ctx: diagnostics(ctx, definition)),
        (
            "Posterior predictions",
            lambda ctx: sample_posterior_predictive(ctx, definition),
        ),
        ("Posterior summary", posterior_summary),
        ("Plots", run_joint_plots),
        ("Report", report),
    ]


def fit_joint_model(
    config: str,
    definition: JointModelDefinition,
) -> JointContext:
    """Shared fit pipeline for the joint sign/speech model (VG15)."""
    return run_fit_pipeline(config, definition, stages=joint_stages(definition))
