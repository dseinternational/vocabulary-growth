# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Declare prior, data and structural sensitivity variants.

Entries map a model and variant name to a suffix and field overrides.
build_variant creates new definitions; registration does not establish that a
variant has been fitted or validated, or that its base has a reporting role.

Move both parameters when changing a Beta anchor. Vary coupled assumptions
together when needed, and disclose variants that change several aspects of a
model. Data restrictions can also change study and age composition. Compare
their actual prepared frames and fitted outputs.
"""

from __future__ import annotations

import math

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    AgeVaryingSubjectScale,
    BivariateCorrelatedSubjectREModelDefinition,
    SubjectFactorPriorParams,
    _as_definition_subclass,
)
from vocab_growth.models.likelihood_utils import (
    LAG_ZERO_CLIP,
    LAG_ZERO_CONTINUITY,
    SPOKEN_FALLBACK_MOMENT_MATCHED,
    SPOKEN_FALLBACK_PAIRED_ONLY,
    SPOKEN_FALLBACK_SEPARATE_DISPERSION,
)
from vocab_growth.sensitivity.overrides import make_variant

# Derive age-scale anchors from the paired concentration prior so both use
# the same age coordinates. A1 also changes dispersion to be constant.
_VG10_KAPPA_U_ANCHORS = MODEL_REGISTRY["vg10"].kappa_u.anchor_ages
_VG10_KAPPA_S_ANCHORS = MODEL_REGISTRY["vg10"].kappa_s.anchor_ages

# (model_key, variant_name) -> {"suffix": str, "scalar"?: dict, "kappa"?: dict}
VARIANTS: dict[tuple[str, str], dict] = {
    # -- Target 1: VG10/VG15 DS-joint q anchors (weakly-informative, broadened off the VG07-posterior values by #155) --
    ("vg10", "q-broad"): {"suffix": "q-broad", "scalar": {
        "p_slope_low_q_alpha": 1.0, "p_slope_low_q_beta": 1.5,
        "p_slope_hi_q_alpha": 2.0, "p_slope_hi_q_beta": 1.2}},
    ("vg10", "q-wider"): {"suffix": "q-wider", "scalar": {
        "p_slope_low_q_alpha": 2.0, "p_slope_low_q_beta": 15.0,
        "p_slope_hi_q_alpha": 12.0, "p_slope_hi_q_beta": 3.0}},
    ("vg15", "q-broad"): {"suffix": "q-broad", "scalar": {
        "p_slope_low_q_alpha": 1.0, "p_slope_low_q_beta": 1.5,
        "p_slope_hi_q_alpha": 2.0, "p_slope_hi_q_beta": 1.2}},

    # -- Target 2: signed GP amplitude & length-scale (VG15) --
    # Compare signed GP amplitude separately from its length scale.
    ("vg15", "etasign-wide"): {"suffix": "etasign-wide", "scalar": {"eta_sign_sigma": 0.7}},
    ("vg15", "etasign-narrow"): {"suffix": "etasign-narrow", "scalar": {"eta_sign_sigma": 0.2}},
    ("vg15", "ellsign-beta33"): {"suffix": "ellsign-beta33", "scalar": {
        "ell_unit_sign_alpha": 3.0, "ell_unit_sign_beta": 3.0}},
    ("vg15", "ellsign-short"): {"suffix": "ellsign-short", "scalar": {
        "ell_unit_sign_alpha": 1.5, "ell_unit_sign_beta": 6.0}},

    # -- Target 3: signed hump anchors (VG15 three-anchor signed mean) --
    # Vary the middle signing height and the old height separately.
    ("vg15", "sign-peak-lo"): {"suffix": "sign-peak-lo", "scalar": {
        "p_slope_mid_sign_alpha": 2.0, "p_slope_mid_sign_beta": 6.0}},  # peak r ~0.26 (vs ~0.42)
    ("vg15", "sign-peak-hi"): {"suffix": "sign-peak-hi", "scalar": {
        "p_slope_mid_sign_alpha": 4.0, "p_slope_mid_sign_beta": 3.0}},  # peak r ~0.58
    # These priors move the middle trend knot. It is not necessarily the
    # combined curve's maximum. Stability under several priors is a limited
    # sensitivity result, not proof of identification.
    ("vg15", "sign-peak-age-uniform"): {"suffix": "sign-peak-age-uniform", "scalar": {
        "sign_peak_prior": (1.0, 1.0)}},   # peak age median 55.5 mo, 89% [19.5, 91.5]
    ("vg15", "sign-peak-age-early"): {"suffix": "sign-peak-age-early", "scalar": {
        "sign_peak_prior": (1.5, 6.0)}},   # peak age median 29.0 mo, 89% [17.4, 52.0]
    ("vg15", "sign-peak-age-late"): {"suffix": "sign-peak-age-late", "scalar": {
        "sign_peak_prior": (4.0, 3.0)}},   # peak age median 61.9 mo, 89% [37.6, 83.1]
    ("vg15", "sign-old-hi"): {"suffix": "sign-old-hi", "scalar": {
        "p_slope_hi_sign_alpha": 2.0, "p_slope_hi_sign_beta": 8.0}},  # old r ~0.18 (words plateau)
    ("vg15", "sign-include-uk01"): {"suffix": "sign-include-uk01", "scalar": {
        "include_uk01_signed": True}},

    # -- Target 4: kappa (dispersion): VG10 (U/S) and VG15 (adds sign) --
    #
    # kappa-flat lowers the anchor-excess centres, favouring more residual
    # variation. The values retain the historical sensitivity calibration.
    # kappa-const shifts only the old-anchor prior towards a historical young
    # centre. Independently sampled anchors do not enforce a constant curve,
    # and that historical centre differs from the current young-anchor centre.
    # VG15 signing retains the legacy concentration parameterisation.
    ("vg10", "kappa-broadfloor"): {"suffix": "kappa-broadfloor", "kappa": {
        "kappa_u": {"kappa_min_sigma": 1.5}, "kappa_s": {"kappa_min_sigma": 1.5}}},
    ("vg10", "kappa-flat"): {"suffix": "kappa-flat", "kappa": {
        "kappa_u": {"excess_young_mu": math.log(106.0 / 8),
                    "excess_old_mu": math.log(28.7 / 8)},
        "kappa_s": {"excess_young_mu": math.log(12.6 / 8),
                    "excess_old_mu": math.log(6.7 / 8)}}},
    ("vg10", "kappa-const"): {"suffix": "kappa-const", "kappa": {
        "kappa_u": {"excess_old_mu": math.log(106.0)},
        "kappa_s": {"excess_old_mu": math.log(12.6)}}},
    ("vg15", "kappa-broadfloor"): {"suffix": "kappa-broadfloor", "kappa": {
        "kappa_u": {"kappa_min_sigma": 1.5}, "kappa_s": {"kappa_min_sigma": 1.5},
        "kappa_sign": {"kappa_min_sigma": 1.0}}},

    # -- Target 5: random-effect scales --
    #
    # The subject scales are calibrated at HalfNormal(1.5) (note section 23), so
    # these bracket that rather than the old 0.5: wide is 3.0 and narrow 0.75,
    # keeping the factor of two either side the variants had before. The *study*
    # scales are still 0.5 and keep their original 1.0 / 0.25.
    ("vg10", "tau-wide"): {"suffix": "tau-wide", "scalar": {
        "tau_u_sigma": 1.0, "tau_q_sigma": 1.0,
        "tau_subj_u_sigma": 3.0, "tau_subj_q_sigma": 3.0}},
    ("vg10", "tau-narrow"): {"suffix": "tau-narrow", "scalar": {
        "tau_u_sigma": 0.25, "tau_q_sigma": 0.25,
        "tau_subj_u_sigma": 0.75, "tau_subj_q_sigma": 0.75}},
    ("vg10", "no-subject"): {"suffix": "no-subject", "scalar": {
        "use_subject_re_u": False, "use_subject_re_q": False}},
    ("vg11", "tau-wide"): {"suffix": "tau-wide", "scalar": {
        "tau_study_sigma": 1.0, "tau_subject_sigma": 3.0}},
    ("vg11", "tau-narrow"): {"suffix": "tau-narrow", "scalar": {
        "tau_study_sigma": 0.25, "tau_subject_sigma": 0.75}},
    # A variance partition needs an active child scale. Disable both together.
    ("vg11", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True, "use_subject_re": False,
        "subject_variance_partition": None}},
    ("vg12", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True, "use_subject_re": False,
        "subject_variance_partition": None}},
    ("vg13", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True,
        "use_subject_re_u": False, "use_subject_re_q": False}},
    ("vg15", "tau-wide"): {"suffix": "tau-wide", "scalar": {
        "tau_u_sigma": 1.0, "tau_q_sigma": 1.0, "tau_sign_sigma": 1.0}},
    ("vg15", "sign-study-only"): {"suffix": "sign-study-only", "scalar": {
        "use_subject_re_sign": False}},

    # Reinstatement checks the source records excluded by the implausibility
    # rule. The source audit motivated the mask but did not confirm every
    # original value. See notes/202607261245-edgin-duplicated-outcome-records.md.
    ("vg10", "us01-implausible-reinstated"): {
        "suffix": "us01-implausible-reinstated",
        "scalar": {"include_implausible_production": True},
    },
    ("vg15", "us01-implausible-reinstated"): {
        "suffix": "us01-implausible-reinstated",
        "scalar": {"include_implausible_production": True},
    },
    # Lift both production masks. Lifting only the implausibility rule can
    # leave reinstated records masked by the separate same-day rule.
    ("vg10", "us01-masked-production-reinstated"): {
        "suffix": "us01-masked-production-reinstated",
        "scalar": {
            "include_implausible_production": True,
            "include_same_day_disagreements": True,
        },
    },
    ("vg15", "us01-masked-production-reinstated"): {
        "suffix": "us01-masked-production-reinstated",
        "scalar": {
            "include_implausible_production": True,
            "include_same_day_disagreements": True,
        },
    },

    # ie_02 as a short form (2026-09-15). ie_02 administered DSE Checklists 1 + 2
    # only; the study owner kept its counts on the 810 scale with a recorded
    # ceiling of 476 instead of masking it as a partial administration, as
    # ie_01's baseline is (`data_utils.DSE_SHORT_FORM_CEILINGS`). The omitted
    # Checklist 3 holds harder words that matter for comprehension above about
    # 300 words, so this arm masks ie_02's comprehension counts and keeps its
    # spoken and signed counts: if the headline comprehension trajectory or `q`
    # moves, the short-form judgement is carrying weight. On VG10 and VG15 beside
    # the other data-handling arms, and on VG20, the model of record.
    ("vg10", "ie02-comprehension-masked"): {
        "suffix": "ie02-comprehension-masked",
        "scalar": {"mask_dse_short_form_comprehension": True},
    },
    ("vg15", "ie02-comprehension-masked"): {
        "suffix": "ie02-comprehension-masked",
        "scalar": {"mask_dse_short_form_comprehension": True},
    },
    ("vg20", "ie02-comprehension-masked"): {
        "suffix": "ie02-comprehension-masked",
        "scalar": {"mask_dse_short_form_comprehension": True},
    },

    # Native-form restriction checks dependence on the shared reference scale.
    # It changes study and age composition and the available cell sources.
    # It cannot isolate measurement equivalence or prove item comparability.
    # Inspect the fitted frame and support audit rather than historical counts.
    ("vg10", "dse-native-only"): {
        "suffix": "dse-native-only",
        "scalar": {"dse_native_only": True},
    },
    ("vg15", "dse-native-only"): {
        "suffix": "dse-native-only",
        "scalar": {"dse_native_only": True},
    },

    # -- Target 6: VG15 psi (association) --
    ("vg15", "psi-neutral"): {"suffix": "psi-neutral", "scalar": {"log_psi_mu": 0.0, "log_psi_sigma": 0.5}},
    ("vg15", "psi-broad"): {"suffix": "psi-broad", "scalar": {"log_psi_mu": 0.0, "log_psi_sigma": 1.0}},
    ("vg15", "psi-strong"): {"suffix": "psi-strong", "scalar": {"log_psi_mu": 0.6, "log_psi_sigma": 0.5}},

    # The between-study spread of psi. tau_psi_sigma = 1.0 was set from the
    # measured spread (an order of magnitude across four sources), which makes it
    # data-informed rather than independently justified — the condition that put
    # every trajectory anchor under Target 8. With only four informed studies
    # tau_psi is weakly identified, so the prior does real work on how far the
    # per-study values shrink toward the population centre, and therefore on the
    # headline psi itself. Narrow forces near-pooling (roughly the pre-2026-08-12
    # behaviour); wide lets each source sit where its own cells put it.
    ("vg15", "tau-psi-narrow"): {"suffix": "tau-psi-narrow", "scalar": {"tau_psi_sigma": 0.3}},
    ("vg15", "tau-psi-wide"): {"suffix": "tau-psi-wide", "scalar": {"tau_psi_sigma": 2.0}},

    # Source composition of psi. Each flag drops one source's cross-tab while
    # keeping its marginal counts. The joint refit can still change U, q and r
    # through shared parameters. es_01 supplies 185 of the 434 historical
    # psi-informing rows and the only source at independence, so "what is the
    # headline without Spain" is the first question the heterogeneity table
    # invites. uk_07 is registered alongside it because it is an intervention
    # trial, and its arrival is what moved psi from 1.80 to 2.49.
    ("vg15", "psi-drop-es01"): {"suffix": "psi-drop-es01", "scalar": {"include_es01_cells": False}},
    ("vg15", "psi-drop-uk07"): {"suffix": "psi-drop-uk07", "scalar": {"include_uk07_cells": False}},

    # -- Target 7: VG15 four-cell concentration --
    ("vg15", "conc-broad"): {"suffix": "conc-broad", "scalar": {"log_conc_sigma": 1.5}},
    ("vg15", "conc-lo"): {"suffix": "conc-lo", "scalar": {"log_conc_mu": 2.0}},
    ("vg15", "conc-hi"): {"suffix": "conc-hi", "scalar": {"log_conc_mu": 4.0}},

    # -- Target 8: young-age trajectory-anchor recalibration (#135/#138/#140/#142)
    #    The mean-function anchors (p_slope_*) and eta were re-centred toward the
    #    young-age empirical/normative band; Targets 1-7 never vary these. Each
    #    variant reverts to the pre-recalibration vague prior so the recalibration
    #    can be shown not to drive the young-age conclusions.
    # VG10 (DS joint understood anchors, aligned in #142): revert both understood
    # anchors to the pre-recalibration vague band, and un-widen eta_u (which was
    # raised 0.4 -> 0.6 specifically to offset the anchor pull-down).
    ("vg10", "u-anchor-broad"): {"suffix": "u-anchor-broad", "scalar": {
        "p_slope_low_u_alpha": 1.0, "p_slope_low_u_beta": 10.0,
        "p_slope_hi_u_alpha": 1.1, "p_slope_hi_u_beta": 1.1}},
    ("vg10", "eta-u-narrow"): {"suffix": "eta-u-narrow", "scalar": {"eta_u_sigma": 0.4}},

    # VG10 clamp scope. `clamp_mean_above_hi_anchor` was `True` -- levelling BOTH
    # the understood mean and `q` off above the 84 mo anchor -- until 2026-08-14,
    # when it became `CLAMP_Q_ONLY` for the whole DS joint family. Measurement
    # drove that: extrapolating VG10's own fitted anchors past the clamp gives
    # q = 0.996 at 115 mo with P(mean > 0.99) = 0.999, while understood reaches
    # 0.962 and never crosses 0.99 in any draw, so only `q` ever needed it. The
    # `clamp-q-only` variant that established this is now the model of record and
    # has been REPLACED by its inverse rather than retired: `clamp-both` restores
    # the old behaviour, so the decision keeps a check that can still fail. (A
    # variant that can no longer vary anything reads as robustness it has not
    # demonstrated -- the principle the retired ceiling and uk_06 variants went
    # out under.) See notes/202608141200-clamp-q-only.md.
    ("vg10", "clamp-both"): {"suffix": "clamp-both", "scalar": {
        "clamp_mean_above_hi_anchor": True}},

    # A1 varies child scales exponentially across each concentration block's
    # current anchor ages and holds concentration constant. It changes both
    # aspects of the base model. A zero log-scale ratio alone does not nest
    # the base when its concentration varies with age.
    # Positive scaling fixes each outcome's latent child ordering across ages;
    # observed counts can still change rank through residual variation.
    ("vg10", "a1-tau-age-varying"): {"suffix": "a1-tau-age-varying", "scalar": {
        "tau_subj_u_sigma": AgeVaryingSubjectScale(
            anchor_ages=_VG10_KAPPA_U_ANCHORS,
            young_sigma=1.5,
            log_ratio_sigma=0.5,
        ),
        "tau_subj_q_sigma": AgeVaryingSubjectScale(
            anchor_ages=_VG10_KAPPA_S_ANCHORS,
            young_sigma=1.5,
            log_ratio_sigma=0.5,
        )}},

    # VG11 (TD spoken anchors, #138): revert the (norm-anchored) spoken band.
    ("vg11", "anchor-broad"): {"suffix": "anchor-broad", "scalar": {
        "p_slope_low_alpha": 1.0, "p_slope_low_beta": 15.0,
        "p_slope_hi_alpha": 1.5, "p_slope_hi_beta": 1.1}},
    # The widened GP amplitude VG11 carried until 2026-09-16. This arm was
    # `eta-narrow` (0.4) until the model of record adopted 0.4 (#357), which made
    # it a no-op; it now keeps the former model of record reachable instead.
    ("vg11", "eta-wide"): {"suffix": "eta-wide", "scalar": {"eta_sigma": 0.5}},

    # VG12 (TD understood anchors, #138): the 12 mo LOW anchor is Wordbank-norm
    # matched (test it reverts cleanly); the 26 mo HIGH anchor has NO CDI
    # comprehension norm (WS is production-only) — its broad variant is the key
    # un-normed sensitivity test.
    ("vg12", "lo-anchor-broad"): {"suffix": "lo-anchor-broad", "scalar": {
        "p_slope_low_alpha": 1.0, "p_slope_low_beta": 20.0}},
    ("vg12", "hi-anchor-broad"): {"suffix": "hi-anchor-broad", "scalar": {
        "p_slope_hi_alpha": 1.1, "p_slope_hi_beta": 1.1}},
    ("vg12", "eta-narrow"): {"suffix": "eta-narrow", "scalar": {"eta_sigma": 0.4}},

    # -- VG12 free scales: the pre-partition parameterisation (#225) --
    #
    # Compare the free child/concentration scales with the variance partition.
    # Clearing the partition restores their separate priors. A few recovery
    # replicates cannot distinguish bias from weak information, and recovery
    # scoring requires satisfactory sampling. See #225 for the design history.
    ("vg12", "free-scales"): {"suffix": "free-scales", "scalar": {
        "subject_variance_partition": None}},

    # -- The spoken fallback branch (#233, #236) --
    #
    # Compare missing-parent treatments on VG10 and VG20.
    # paired-only changes the sample as well as removing the approximation;
    # it provides no bound or guaranteed interval-width direction.
    # fallback-dispersion adds a branch-specific concentration multiplier.
    # marginal-moments matches the paired marginal's first two moments but
    # remains an approximation to its full count distribution.
    # Each treatment requires its own fit.
    ("vg20", "paired-only"): {"suffix": "paired-only", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_PAIRED_ONLY}},
    ("vg20", "fallback-dispersion"): {"suffix": "fallback-dispersion", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_SEPARATE_DISPERSION}},
    ("vg20", "marginal-moments"): {"suffix": "marginal-moments", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_MOMENT_MATCHED}},
    ("vg10", "paired-only"): {"suffix": "paired-only", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_PAIRED_ONLY}},
    ("vg10", "fallback-dispersion"): {"suffix": "fallback-dispersion", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_SEPARATE_DISPERSION}},
    ("vg10", "marginal-moments"): {"suffix": "marginal-moments", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_MOMENT_MATCHED}},

    # The same three arms for the two signing models, registered once their
    # engines could run them (#266 finding 8). VG14 and VG15 hard-coded the
    # default, so the exposure the finding names -- an approximation that
    # preserves the mean but not the variance, on rows that are older and
    # clustered by study -- could not be measured on either at all. On these
    # engines the treatment applies to the SIGNED rows as well as the spoken
    # ones: signing is nested inside comprehension exactly as speech is, so
    # varying one and not the other would leave half the exposure in place.
    ("vg14", "paired-only"): {"suffix": "paired-only", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_PAIRED_ONLY}},
    ("vg14", "fallback-dispersion"): {"suffix": "fallback-dispersion", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_SEPARATE_DISPERSION}},
    ("vg14", "marginal-moments"): {"suffix": "marginal-moments", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_MOMENT_MATCHED}},
    ("vg15", "paired-only"): {"suffix": "paired-only", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_PAIRED_ONLY}},
    ("vg15", "fallback-dispersion"): {"suffix": "fallback-dispersion", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_SEPARATE_DISPERSION}},
    ("vg15", "marginal-moments"): {"suffix": "marginal-moments", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_MOMENT_MATCHED}},

    # -- VG13 observation window (#228) --
    #
    # Wider VG13 windows also change anchors, GP priors and reporting grids.
    # They are joint specification changes, not window-only contrasts.
    # High-anchor centres use the fitted data; dispersion magnitudes remain
    # inherited. The 22-month window reduces some late short-form ceiling
    # exposure but cannot certify an unbiased age curve.
    # See notes/202608211100-window-22-adopted.md.
    ("vg13", "window-25"): {"suffix": "window-25", "scalar": {
        "max_age_months": 25,
        "slope_anchors": (10, 24),
        "ages_query": (8, 10, 12, 14, 16, 18, 20, 22, 24,),
        "gp_domain_months": (8, 25),
        "gp_anchor_age_months": 17.0,
        # 24 mo in-sample medians: understood 0.415 of 810, q 0.675.
        "p_slope_hi_u_alpha": 2.0, "p_slope_hi_u_beta": 2.8,   # median 0.404
        "p_slope_hi_q_alpha": 2.0, "p_slope_hi_q_beta": 1.3,   # median 0.630
        "eta_q_sigma": 0.5},
     "kappa": {"kappa_u": {"anchor_ages": (12.0, 20.0)},
               "kappa_s": {"anchor_ages": (12.0, 20.0)}}},
    ("vg13", "window-22"): {"suffix": "window-22", "scalar": {
        "max_age_months": 22,
        "slope_anchors": (10, 21),
        "ages_query": (8, 10, 12, 14, 16, 18, 20, 22,),
        "gp_domain_months": (8, 22),
        "gp_anchor_age_months": 15.5,
        # 21 mo in-sample medians: understood 0.359 of 810, q 0.417.
        "p_slope_hi_u_alpha": 2.0, "p_slope_hi_u_beta": 3.2,   # median 0.369
        "p_slope_hi_q_alpha": 2.0, "p_slope_hi_q_beta": 2.6,   # median 0.425
        "eta_q_sigma": 0.5},
     "kappa": {"kappa_u": {"anchor_ages": (12.0, 20.0)},
               "kappa_s": {"anchor_ages": (12.0, 20.0)}}},

    # Widen and shift the two high-anchor priors in the 22-month specification.
    # This checks dependence on centres calibrated from the fitted data.
    # A stable result does not establish independence from all prior choices.
    ("vg13", "window-22-vague-anchors"): {"suffix": "window-22-vague-anchors", "scalar": {
        "max_age_months": 22,
        "slope_anchors": (10, 21),
        "ages_query": (8, 10, 12, 14, 16, 18, 20, 22,),
        "gp_domain_months": (8, 22),
        "gp_anchor_age_months": 15.5,
        "p_slope_hi_u_alpha": 1.2, "p_slope_hi_u_beta": 2.0,   # median 0.346
        "p_slope_hi_q_alpha": 1.3, "p_slope_hi_q_beta": 1.3,   # median 0.500
        "eta_q_sigma": 0.5},
     "kappa": {"kappa_u": {"anchor_ages": (12.0, 20.0)},
               "kappa_s": {"anchor_ages": (12.0, 20.0)}}},

    # -- Target 9: where the dispersion prior is placed (#229) --
    #
    # These three were registered on 2026-08-19 as `kappa-anchor-18-72`,
    # `kappa-floor-recentred` and `kappa-anchor-18-72-floor`, perturbing a base
    # anchored at (24, 48) with a floor prior median of 3.0. The combination was
    # promoted into `_DS_JOINT_*_KAPPA_RE` the same day, so they are restated
    # here as the inverse: a sensitivity variant has to perturb *away* from the
    # model of record, and after promotion the originals pointed at it. The
    # priors are unchanged, so the `test`-tier fits made under the old names are
    # these variants under new labels -- their output directories still carry the
    # old suffixes, and the numbers are in
    # notes/202608191800-kappa-components-not-estimands.md.
    #
    #   kappa-anchor-24-48   the old anchors, keeping the calibrated floor. Asks
    #                        whether anchoring outside the reporting range, so
    #                        that everything above 48 months is extrapolation
    #                        onto the asymptote, changes what is reported. On the
    #                        evidence so far it does not: under 1% on
    #                        comprehension at every age above 24 months.
    #   kappa-floor-generic  the old generic log(3.0) floor prior, keeping the
    #                        new anchors. This is the factor that does move the
    #                        old-age numbers -- 14.2% at 84 months -- though only
    #                        by about a fifth of that quantity's own 89%
    #                        interval, and notes/202608020829 §22 had already
    #                        judged the uncentred floor immaterial because only
    #                        the sum at the anchors is identified.
    #   kappa-pre-promotion  both, which is the dispersion block every DS joint
    #                        fit carried before 2026-08-19. Keeps the previous
    #                        model of record reachable as a registered variant
    #                        rather than only in git history.
    # Tail-leverage check: change only the admitted age maximum.
    # Keep the GP domain, anchors and priors so the below-cap likelihood has
    # the same parameterisation. Any refitted change can involve shared fixed
    # effects as well as child slopes; it is not uniquely attributable to the
    # longitudinal subset or proof that the omitted tail was harmless.
    ("vg19", "max-age-84"): {"suffix": "max-age-84", "scalar": {
        "max_age_months": 84}},

    ("vg20", "kappa-anchor-24-48"): {"suffix": "kappa-anchor-24-48", "kappa": {
        "kappa_u": {"anchor_ages": (24.0, 48.0),
                    "excess_young_mu": math.log(63.4),
                    "excess_old_mu": math.log(19.9)},
        "kappa_s": {"anchor_ages": (24.0, 48.0),
                    "excess_young_mu": math.log(7.6),
                    "excess_old_mu": math.log(2.2),
                    "excess_old_sigma": 1.0}}},
    ("vg20", "kappa-floor-generic"): {"suffix": "kappa-floor-generic", "kappa": {
        "kappa_u": {"kappa_min_mu": math.log(3.0),
                    "excess_young_mu": math.log(89.6),
                    "excess_old_mu": math.log(11.0)},
        "kappa_s": {"kappa_min_mu": math.log(3.0),
                    "excess_young_mu": math.log(15.2),
                    "excess_old_mu": math.log(5.4)}}},
    ("vg20", "kappa-pre-promotion"): {"suffix": "kappa-pre-promotion", "kappa": {
        "kappa_u": {"anchor_ages": (24.0, 48.0),
                    "kappa_min_mu": math.log(3.0),
                    "excess_young_mu": math.log(106.0),
                    "excess_old_mu": math.log(28.7)},
        "kappa_s": {"anchor_ages": (24.0, 48.0),
                    "kappa_min_mu": math.log(3.0),
                    "excess_young_mu": math.log(12.6),
                    "excess_old_mu": math.log(6.7),
                    "excess_old_sigma": 1.0}}},

    # Compare factor ranks one and two with the default rank three.
    # The rate-scale priors remain fixed, but covariance geometry changes with
    # rank. Rank one gives a common deviate with affine age effects, unlike A1's
    # exponential scaling. Historical residual likelihoods do not establish
    # rank identification under the fitted count model.
    # See notes/202608231420-vg22-factor-anchor-bimodality.md.
    ("vg22", "rank-1"): {"suffix": "rank-1", "scalar": {
        "subject_factor": SubjectFactorPriorParams(
            rank=1, tau1_u_sigma=0.5, tau1_q_sigma=0.5, ref_age_months=36.0)}},
    ("vg22", "rank-2"): {"suffix": "rank-2", "scalar": {
        "subject_factor": SubjectFactorPriorParams(
            rank=2, tau1_u_sigma=0.5, tau1_q_sigma=0.5, ref_age_months=36.0)}},

    # -- VG16: what the cross-lag coefficient survives (#242) --
    #
    # Compare VG16's coefficient as well as its trajectories. Prior-wave
    # comprehension can exist even when the current row uses the spoken fallback.
    # Removing fallback rows changes current-parent availability, study mix and
    # age support; coefficient movement does not isolate one cause.
    ("vg16", "conditional-only"): {"suffix": "conditional-only", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_PAIRED_ONLY}},
    #
    # `dse-native-only` keeps only administrations recorded natively on the 810
    # reference, so no count is scored against a denominator its form did not
    # use. The lag predictor is a logit of a *proportion*, understood / 810, so
    # a short-form source enters it already deflated -- the harmonisation acts
    # directly on the regressor here, not only on the outcome. Read the verdict
    # with its support in view: the same restriction keeps 153 of VG10's 1,707
    # fitted rows (2026-09-15, three studies once ie_02 left the native set), and
    # it changes study composition as well as size. (Until
    # #289 task 4.2 the plot-grid `gap` series was matched on exact ages, so a
    # restricted pool's different linspace made every such variant "partial
    # coverage" with nothing compared; it is now interpolated onto the
    # baseline's grid inside the variant's support.)
    ("vg16", "dse-native-only"): {
        "suffix": "dse-native-only",
        "scalar": {"dse_native_only": True},
    },

    #
    # The three below need the fields added on 2026-08-25. They are registered
    # in the same change as the fields, because a field with no variant using it
    # is dead weight in the fingerprint of all twelve bivariate models.
    #
    # `lag-gap-12` tests the constancy assumption directly. `beta_lag` is one
    # number for gaps running 1 to 28 months (median 6), and a prospective
    # association measured over two years is not the same quantity as one
    # measured over six months. A ceiling at 12 keeps the bulk and drops the
    # tail: 41 of 477 lagged rows on the current frame, with 9 above 18 months
    # and 4 above 24. Dropping a lag does not drop the row -- the observation
    # still enters both likelihoods, it just stops informing the coefficient.
    ("vg16", "lag-gap-12"): {"suffix": "lag-gap-12", "scalar": {
        "lag_max_gap_months": 12.0}},
    #
    # `no-us01` is the leave-one-study-out check with the most leverage. `us_01`
    # supplies 136 of VG16's 477 lagged rows, 28.5% of the evidence for a
    # coefficient reported as a property of children with Down syndrome rather
    # than of a study. `it_01` is the next largest at 106 and is one registry
    # line away; the field takes any tuple of study codes, so a full
    # leave-one-out sweep over the eight contributing studies needs no code.
    ("vg16", "no-us01"): {"suffix": "no-us01", "scalar": {
        "exclude_studies": ("us_01",)}},
    #
    # Boundary correction changes every proportion except one-half, with
    # larger logit changes near zero or one. Even positive counts close to the
    # boundary can change materially. See likelihood_utils.LAG_ZERO_CONTINUITY.
    ("vg16", "lag-continuity"): {"suffix": "lag-continuity", "scalar": {
        "lag_zero_handling": LAG_ZERO_CONTINUITY}},
    #
    # Require equal, known source and target inventory ceilings.
    # This checks size, not item identity. It disables the lag rather than
    # dropping the row or searching further back. Native-form restriction also
    # changes study composition; neither comparison bounds the scale effect.
    ("vg16", "lag-same-form"): {"suffix": "lag-same-form", "scalar": {
        "lag_same_form_only": True}},
    #
    # The coefficient-prior-scale pair #242 asks for, and the one VG16 variant
    # that needs no field: `beta_lag_sigma` has been on the definition since the
    # cross-lag was added. `Normal(0, 0.5)` is symmetric, which the review grants,
    # but symmetry is not calibration -- "posterior exclusion of zero is not
    # purely a data result merely because the prior did not prefer a sign".
    #
    # The prediction being tested is that these move nothing: the current fit's
    # posterior SD is 0.066 against a prior SD of 0.5, a contraction of 0.98, so
    # the prior is doing almost no work on the interval. That is an argument from
    # one number, which is exactly what this item exists to replace with a fit.
    # `beta-tight` is the arm that carries the question -- a prior pulled toward
    # zero is what would expose a prior-driven exclusion of it -- and `beta-wide`
    # is its companion, checking that the interval does not simply inflate with
    # whatever it is given.
    ("vg16", "beta-tight"): {"suffix": "beta-tight", "scalar": {
        "beta_lag_sigma": 0.25}},
    ("vg16", "beta-wide"): {"suffix": "beta-wide", "scalar": {
        "beta_lag_sigma": 1.0}},
    #
    # `corr` is #289 task 3.9's comparator: VG16 with VG20's correlated child
    # block (LKJ eta = 2) and nothing else, so VG16 is nested at rho_uq = 0. A
    # persistent correlation between a child's comprehension and conversion
    # standings is the rival explanation for a positive lag coefficient under
    # the population baseline VG16 registers (notes/202608151140 section 3, #297),
    # and only a model carrying both terms can say whether `beta_lag` survives
    # it. VG16 carries no sex covariate, so neither does this arm. It needs a
    # class promotion rather than a scalar override, because the correlation
    # field lives on the correlated subclass. Once it is fitted, the designed
    # (beta, rho) cells run on it through `fit_recovery.py vg16 --variant corr`
    # with `--set-truth beta_lag=0` or `--set-truth rho_uq_raw=0.5`
    # (rho_uq = 2 * rho_uq_raw - 1); scripts/vm/campaign.sh schedules all three.
    ("vg16", "corr"): {"suffix": "corr", "promote": (
        BivariateCorrelatedSubjectREModelDefinition,
        {"subject_re_correlation_eta": 2.0})},

    # -- VG21: the anchors it was promoted with (#228, #240) --
    #
    # Check the two high-anchor priors calibrated from the fitted data.
    # Values match VG13's wider-window sensitivity so the contrasts stay
    # comparable. Sensitivity does not make those priors independently sourced.
    ("vg21", "vague-anchors"): {"suffix": "vague-anchors", "scalar": {
        "p_slope_hi_u_alpha": 1.2, "p_slope_hi_u_beta": 2.0,   # median 0.346
        "p_slope_hi_q_alpha": 1.3, "p_slope_hi_q_beta": 1.3}},  # median 0.500

    # -- VG23: whether the correlation is evidenced or regularised (#229) --
    #
    # For the 2x2 block, eta=1 gives a uniform correlation prior and eta=2
    # favours values near zero. Compare posterior estimates and uncertainty;
    # a small shift under one alternative does not prove data identification.
    ("vg23", "eta-flat"): {"suffix": "eta-flat", "scalar": {
        "subject_re_correlation_eta": 1.0}},

    # -- VG25: the sign -> speech cross-lag (#297) --
    #
    # VG25 variants check boundary handling, likelihood scope, baseline,
    # gap support, study contribution and coefficient-prior scale.
    ("vg25", "sign-lag-clip"): {"suffix": "sign-lag-clip", "scalar": {
        "sign_lag_zero_handling": LAG_ZERO_CLIP}},
    #
    # `sign-lag-in-cells` replaced `sign-lag-marginal-only` on 2026-09-15, when
    # the headline itself moved to the spoken marginal. The headline had let the
    # lag into the cross-tab compositions on the argument that one scalar on a
    # fixed covariate cannot do to `psi` what a free per-child offset did. Under
    # the within-child baseline the covariate is not fixed -- it subtracts the
    # child's estimated signing intercept -- and the first rep fit was bimodal
    # (`beta_sign_lag` +0.69 / -0.50, R-hat 1.61). So the in-cells arm is
    # registered under the POPULATION baseline, the one combination in which the
    # argument holds; twelve-chain probes found it unimodal, as they found the
    # headline. It is what the 80 extra supporting observations and uk_07's 52
    # rows are worth: 190 from 128 children against the headline's 110 from 79.
    # Read `psi` from it too, since it is the arm in which the lag reaches the
    # rows that identify `psi`. notes/202609151930-vg25-lag-out-of-the-cells.md.
    ("vg25", "sign-lag-in-cells"): {"suffix": "sign-lag-in-cells", "scalar": {
        "sign_lag_baseline": "population", "sign_lag_in_cells": True}},
    #
    # The population baseline retains modelled child signing standing in the
    # predictor. It can share information with the child-correlation block,
    # but is not mathematically a second measure of that correlation.
    ("vg25", "sign-lag-population"): {"suffix": "sign-lag-population", "scalar": {
        "sign_lag_baseline": "population"}},
    #
    # `sign-lag-gap-12` tests the constancy assumption the way `lag-gap-12` does
    # for VG16: one coefficient is fitted across every gap the frame offers, and
    # a prospective association measured over a year is not self-evidently the
    # same quantity as one measured over three months. Registered with the model
    # rather than after a reviewer asks, which is what #242 asked for. Dropping a
    # lag does not drop the row.
    ("vg25", "sign-lag-gap-12"): {"suffix": "sign-lag-gap-12", "scalar": {
        "sign_lag_max_gap_months": 12.0}},
    # Isolate the seven DSE-to-Oxford lags in the 13 September 2026 frame.
    ("vg25", "sign-lag-same-form"): {
        "suffix": "sign-lag-same-form",
        "scalar": {"sign_lag_same_form_only": True},
    },
    #
    # Removing uk_07 cells returns its counts to marginal likelihoods and can
    # add spoken lag support while removing its direct cell-association evidence.
    # This changes likelihood allocation, not study inclusion. Leave-one-study
    # variants below remove all of the named studies' observations instead.
    # fit_identity records the historical empty exclude_studies default.
    ("vg25", "sign-lag-uk07-marginal"): {"suffix": "sign-lag-uk07-marginal", "scalar": {
        "include_uk07_cells": False}},
    #
    # The leave-one-study-out pair (#297 check 5), for the two studies the lag's
    # support rests on most. On the 2026-09-15 frame four studies supply its 110
    # supporting observations -- ie_02 42, uk_05 30, uk_04 25, uk_02 13 -- and
    # these two carry two thirds of it between them. `no-ie02` takes the support
    # to 68 observations from 37 children and `no-uk05` to 80 from 64, exactly
    # their own contributions; the other two are one registry line each.
    #
    # `no-uk05` replaced `no-uk07` on 2026-09-15. With the lag in the spoken
    # marginal only, uk_07 contributes no support at all (its rows carry no
    # spoken marginal), so leaving it out would no longer test the lag.
    #
    # Both remove **merged-view** sources with no cross-tabulation, so the
    # composition likelihood is untouched and a moved coefficient is a statement
    # about that study's children. uk_05 is also one of the five signing sources,
    # so a move in the sign trajectory there is expected and is not a lag result.
    ("vg25", "no-ie02"): {"suffix": "no-ie02", "scalar": {
        "exclude_studies": ("ie_02",)}},
    ("vg25", "no-uk05"): {"suffix": "no-uk05", "scalar": {
        "exclude_studies": ("uk_05",)}},
    #
    # The coefficient-prior pair, matching VG16's `beta-tight` / `beta-wide` and
    # for the same reason: a symmetric prior is not a calibrated one, and
    # "posterior exclusion of zero is not purely a data result merely because the
    # prior did not prefer a sign". `beta-sign-tight` is the arm that carries the
    # question; `beta-sign-wide` checks that the interval does not simply inflate
    # with whatever it is given.
    ("vg25", "beta-sign-tight"): {"suffix": "beta-sign-tight", "scalar": {
        "beta_sign_lag_sigma": 0.25}},
    ("vg25", "beta-sign-wide"): {"suffix": "beta-sign-wide", "scalar": {
        "beta_sign_lag_sigma": 1.0}},

    # -- #240: the typically developing variants its review asked for --
    #
    # Variants cover current TD references and the unclassified VG26 candidate.
    # Registration does not establish completed fits or a reporting role.
    ("vg11", "no-study-threshold"): {"suffix": "no-study-threshold", "scalar": {
        "min_study_observations": None}},
    ("vg12", "no-study-threshold"): {"suffix": "no-study-threshold", "scalar": {
        "min_study_observations": None}},
    ("vg21", "no-study-threshold"): {"suffix": "no-study-threshold", "scalar": {
        "min_study_observations": None}},
    ("vg23", "no-study-threshold"): {"suffix": "no-study-threshold", "scalar": {
        "min_study_observations": None}},
    ("vg26", "no-study-threshold"): {"suffix": "no-study-threshold", "scalar": {
        "min_study_observations": None}},
    #
    # Add study age slopes to test intercept-only structure. The slope scale
    # has a HalfNormal prior, not a hard upper bound. At zero scale the graph
    # nests the intercept-only structure; a positive-scale interval alone
    # does not establish evidence for nonzero variation.
    ("vg11", "study-age-slopes"): {"suffix": "study-age-slopes", "scalar": {
        "study_age_slope_sigma": 0.5}},
    ("vg12", "study-age-slopes"): {"suffix": "study-age-slopes", "scalar": {
        "study_age_slope_sigma": 0.5}},
    ("vg21", "study-age-slopes"): {"suffix": "study-age-slopes", "scalar": {
        "study_age_slope_sigma": 0.5}},
    ("vg23", "study-age-slopes"): {"suffix": "study-age-slopes", "scalar": {
        "study_age_slope_sigma": 0.5}},
    ("vg26", "study-age-slopes"): {"suffix": "study-age-slopes", "scalar": {
        "study_age_slope_sigma": 0.5}},
    #
    # A1 tests allocation of age-varying spread between child scale and
    # residual concentration. Short-form compression is one possible cause,
    # not a cause established by improved fit or by rescoring aggregate totals.
    # A1 holds concentration constant, so zero scale ratios do not generally
    # reproduce the base. Its positive scale preserves latent child ordering
    # within each outcome. Correlated A1 blocks are not implemented here.
    ("vg11", "a1-tau-age-varying"): {"suffix": "a1-tau-age-varying", "scalar": {
        "tau_subject_sigma": AgeVaryingSubjectScale(
            anchor_ages=MODEL_REGISTRY["vg11"].kappa.anchor_ages,
            young_sigma=1.5,
            log_ratio_sigma=0.5,
        )}},
    ("vg12", "a1-tau-age-varying"): {"suffix": "a1-tau-age-varying", "scalar": {
        "tau_subject_sigma": AgeVaryingSubjectScale(
            anchor_ages=MODEL_REGISTRY["vg12"].kappa.anchor_ages,
            young_sigma=1.5,
            log_ratio_sigma=0.5,
        )}},
    ("vg21", "a1-tau-age-varying"): {"suffix": "a1-tau-age-varying", "scalar": {
        "tau_subj_u_sigma": AgeVaryingSubjectScale(
            anchor_ages=MODEL_REGISTRY["vg21"].kappa_u.anchor_ages,
            young_sigma=1.5,
            log_ratio_sigma=0.5,
        ),
        "tau_subj_q_sigma": AgeVaryingSubjectScale(
            anchor_ages=MODEL_REGISTRY["vg21"].kappa_s.anchor_ages,
            young_sigma=1.5,
            log_ratio_sigma=0.5,
        )}},
    #
    # Widen q's GP-amplitude prior. Historical contraction statistics describe
    # those fits and do not prove that the parameter was or was not informed.
    ("vg21", "eta-q-wide"): {"suffix": "eta-q-wide", "scalar": {"eta_q_sigma": 1.0}},
    ("vg26", "eta-q-wide"): {"suffix": "eta-q-wide", "scalar": {"eta_q_sigma": 1.0}},
    ("vg23", "eta-q-wide"): {"suffix": "eta-q-wide", "scalar": {"eta_q_sigma": 0.5}},
    #
    # ITEM 6, VG13's DEBT, CARRIED TO ITS SUCCESSORS. VG13's `single-admin` and
    # `window-22-vague-anchors` were registered and never fitted, and VG21's page
    # names `single-admin` as the variant most worth carrying across: repeated
    # administrations are the mechanism behind both the child scales and the
    # energy caveat. One administration per child, child effects removed. On
    # VG26 the correlation goes with them, since it correlates the two child
    # blocks the variant removes. VG26's `vague-anchors` is VG21's entry
    # unchanged, so the two stay comparable.
    ("vg21", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True,
        "use_subject_re_u": False, "use_subject_re_q": False}},
    ("vg26", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True,
        "use_subject_re_u": False, "use_subject_re_q": False,
        "subject_re_correlation_eta": None}},
    ("vg26", "vague-anchors"): {"suffix": "vague-anchors", "scalar": {
        "p_slope_hi_u_alpha": 1.2, "p_slope_hi_u_beta": 2.0,   # median 0.346
        "p_slope_hi_q_alpha": 1.3, "p_slope_hi_q_beta": 1.3}},  # median 0.500
}


def variants_for(model_key: str) -> list[str]:
    """Variant names registered for a model, in registry order."""
    return [name for (m, name) in VARIANTS if m == model_key]


def build_variant(model_key: str, variant_name: str) -> list:
    """Materialise variant definition(s) for a model.

    ``variant_name="all"`` returns every registered variant for the model;
    otherwise a single-element list holding the named variant.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_key!r}.")
    names = variants_for(model_key) if variant_name == "all" else [variant_name]
    if not names:
        raise KeyError(f"No sensitivity variants registered for {model_key!r}.")
    base = MODEL_REGISTRY[model_key]
    out = []
    for name in names:
        spec = VARIANTS.get((model_key, name))
        if spec is None:
            raise KeyError(f"Unknown variant {name!r} for {model_key!r}.")
        # A variant adding a field its base's class lacks is rebuilt as the
        # subclass that carries it, the way the registry derives VG20 from VG10.
        variant_base = base
        promote = spec.get("promote")
        if promote is not None:
            cls, fields = promote
            variant_base = _as_definition_subclass(base, cls, **fields)
        out.append(
            make_variant(
                variant_base,
                config_suffix=spec["suffix"],
                scalar_over=spec.get("scalar"),
                kappa_over=spec.get("kappa"),
            )
        )
    return out
