# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sensitivity variant registry for model robustness analyses.

Each entry maps ``(model_key, variant_name)`` to a config-name suffix and the
hyperparameter overrides that define one alternative-prior variant of a model of
record. ``build_variant`` materialises the variant definition(s) via
:func:`make_variant`, so each variant's fit lands in its own output directory.

The matrix covers the seven §7 sensitivity targets plus Target 8 — the young-age
trajectory-anchor recalibration (#135/#138/#140/#142) — on the models of record
(VG10, VG11, VG12, VG13, VG15). VG12 (TD understood RE) is included for Target 8
because it carries the 26-month understood high anchor, which has no independent
CDI comprehension norm (WS is production-only). VG13 is included only for the
single-administration clustering sensitivity; this deliberately reduces rather
than multiplies the cost of its large repeated-measures fit.

Source-measurement variants also test the signing-source inclusion decisions and
the influence of the 18 us_01 Words & Sentences observations at their 680-word
form ceiling. Primary models continue to retain those valid ceiling counts.

Co-identified priors are kept as whole units per variant: a Beta anchor moves
both ``alpha`` and ``beta`` together (mean = α/(α+β), concentration = α+β), and
the signed GP amplitude (``eta_sign``) and length-scale (``ell_unit_sign``) are
varied in *separate* one-factor variants so the attribution stays clean.
"""

from __future__ import annotations

import math

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    AgeVaryingSubjectScale,
    SubjectFactorPriorParams,
)
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_MOMENT_MATCHED,
    SPOKEN_FALLBACK_PAIRED_ONLY,
    SPOKEN_FALLBACK_SEPARATE_DISPERSION,
)
from vocab_growth.sensitivity.overrides import make_variant

# A1's subject-scale anchors must be the paired kappa block's own anchors, or the
# two parameters contest different spans and the variant stops being one-factor.
# Read them off the model the variant is registered on rather than off the shared
# prior constants, so the coupling survives VG10 being given its own kappa block.
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
    # Bracket the new default eta_sign=0.4 (the three-anchor mean carries the hump,
    # so the GP amplitude is back to the standard scale).
    ("vg15", "etasign-wide"): {"suffix": "etasign-wide", "scalar": {"eta_sign_sigma": 0.7}},
    ("vg15", "etasign-narrow"): {"suffix": "etasign-narrow", "scalar": {"eta_sign_sigma": 0.2}},
    ("vg15", "ellsign-beta33"): {"suffix": "ellsign-beta33", "scalar": {
        "ell_unit_sign_alpha": 3.0, "ell_unit_sign_beta": 3.0}},
    ("vg15", "ellsign-short"): {"suffix": "ellsign-short", "scalar": {
        "ell_unit_sign_alpha": 1.5, "ell_unit_sign_beta": 6.0}},

    # -- Target 3: signed hump anchors (VG15 three-anchor signed mean) --
    # Peak level is the key uncertain quantity (peak AGE is only weakly identified);
    # the old anchor is the Miller-vs-uk_06 "words fall vs plateau" knob.
    ("vg15", "sign-peak-lo"): {"suffix": "sign-peak-lo", "scalar": {
        "p_slope_mid_sign_alpha": 2.0, "p_slope_mid_sign_beta": 6.0}},  # peak r ~0.26 (vs ~0.42)
    ("vg15", "sign-peak-hi"): {"suffix": "sign-peak-hi", "scalar": {
        "p_slope_mid_sign_alpha": 4.0, "p_slope_mid_sign_beta": 3.0}},  # peak r ~0.58
    # Peak AGE priors. The two `sign-peak-lo`/`-hi` variants above vary the peak's
    # HEIGHT; until 2026-08-06 the peak's age was a fixed anchor, so nothing could
    # vary it. These three span the plausible range so the adopted Beta(2, 4)
    # (median 40 mo) can be checked against a uniform prior and against priors
    # pulling early and late. `sign-peak-age-late` is the sharpest test: its 89%
    # interval starts at 37.6 months, above the fitted 29.4, so a posterior that
    # stays near 29-30 under it is data-driven rather than prior-driven.
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
    # VG10's and VG15's understood and spoken outcomes moved to the two-anchor
    # form (note section 22), so these are restated in its parameters. The two
    # questions each variant asks are unchanged:
    #
    #   kappa-flat   what if dispersion is much *lower* than calibrated? The
    #                legacy version set a_kappa_mu from log(8) to 0, an eight-fold
    #                cut in the age term; here both anchor medians are divided by
    #                eight, which is the same cut stated at the anchors.
    #   kappa-const  what if dispersion does not vary with age? The legacy version
    #                pinned b_kappa_mag near zero; under the anchored form the
    #                slope is derived, so a curve constant in age is one whose two
    #                anchors are equal.
    #
    # VG15's signed ratio stays on the legacy form and keeps the legacy override.
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
    # `subject_variance_partition` must be cleared alongside `use_subject_re`: the
    # partition allocates a shared scatter budget BETWEEN the subject scale and the
    # young kappa anchor, so with no subject scale there is nothing to allocate and
    # the engine rejects the combination. Adopting the partition on VG11/VG12
    # (2026-08-05) silently broke these two variants until 2026-08-07 -- the
    # registry test builds variant *definitions*, which still succeeded; the
    # failure only appears when a model graph is built from one.
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

    # The former ("vg10"/"vg15", "us01-ceiling-excluded") variants are retired.
    # They asked what changes if the us_01 Words & Sentences records at the 680-item
    # form ceiling are dropped, on the reading that those counts were valid but
    # right-censored. The Edgin audit (notes/202607261245-...) established that they
    # are not censored measurements but invalid values — a contiguous child-id block
    # uniformly at the form maximum, with seven of eight young records contradicted
    # 5-fold or more by the same child later — so they are now masked by default in
    # data_utils.mask_implausible_production_administrations. A sensitivity that
    # excludes records already excluded is a no-op, and a registered check that
    # cannot fail is worse than no check. The live question is the inverse — what
    # changes if the exclusion is wrong — and it is registered below.
    #
    # This inverse form is not optional garnish. The source author no longer holds
    # the original data files, so the 30 masked administrations can never be
    # confirmed as defective at source; these two variants are the only way to
    # report what the headline joint trajectories would have been had that
    # judgement been wrong. Both engines print the reinstated count, so a variant
    # that silently stopped biting would be visible rather than reassuring.
    ("vg10", "us01-implausible-reinstated"): {
        "suffix": "us01-implausible-reinstated",
        "scalar": {"include_implausible_production": True},
    },
    ("vg15", "us01-implausible-reinstated"): {
        "suffix": "us01-implausible-reinstated",
        "scalar": {"include_implausible_production": True},
    },

    # The 810-item reference denominator. Every model scores raw counts against
    # n_trials = 810, so counts from the 416-item Oxford CDI, the 396/680-item
    # MB-CDI forms, the 651-item CDI-Down, the 674-item Reading CDI and the
    # 675-item NZCDI all enter on a denominator their form did not use. That is
    # sound only if the shorter forms hold the easier items, and the sufficiency
    # result (notes/202607261540) proves no aggregate analysis of these data can
    # test it — a statistic sufficient for ability carries no information about
    # item composition. The assumption is therefore probed by deleting the rows
    # that need it, not by re-running the model differently.
    #
    # This is the widest-scoped variant registered: 278 of 1,521 rows survive.
    # On vg15 it also collapses psi to its single-study branch, because uk_02's
    # DSE arm is the only cross-tab source native to 810 — so read it for the
    # trajectory shapes, not for the association. Both engines print the excluded
    # count, so a variant that stopped biting shows up rather than passing quietly.
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
    # keeping its marginals, so U, q and r are unchanged and only the association
    # loses evidence. es_01 is the one that matters most: it is 185 of the 434
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

    # VG10 Proposal A1 -- the age variation moved off `kappa` and onto the
    # between-child scale. `tau_subj_*` becomes log-linear in age between the
    # SAME two anchors the kappa blocks use (24 and 48 months), and both kappa
    # blocks are held flat, so the two parameters contest one span rather than
    # one of them absorbing what the other cannot express. Registered on VG10
    # alone: it is the model of record whose `kappa` decline the report reads
    # developmentally, and the diagnostic only needs one model to answer how
    # much of that decline is misattributed widening.
    #
    # `log_tau_subj_*_ratio ~ Normal(0, 0.5)` is centred on the model of record
    # (ratio 1, no widening) and puts 89% of its mass on a 24->48 month ratio in
    # [0.48, 2.1]; the tracking note's non-measurement spread rises by about 1.4x
    # over that span on DS spoken, so the prior brackets the measured effect
    # without asserting it. The young anchor keeps HalfNormal(1.5) exactly, so
    # this is one factor: the constant scale becoming a slope.
    #
    # NOT a candidate model of record, and the reason is measured rather than
    # stylistic: scaling one per-child deviate by tau(age) imposes perfect rank
    # correlation across age, and the observed disattenuated correlation is 0.28
    # beyond two years. See notes/202607261540 §9 and notes/202608141600 §8.
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

    # VG11 (TD spoken anchors, #138): revert the (norm-anchored) spoken band and eta.
    ("vg11", "anchor-broad"): {"suffix": "anchor-broad", "scalar": {
        "p_slope_low_alpha": 1.0, "p_slope_low_beta": 15.0,
        "p_slope_hi_alpha": 1.5, "p_slope_hi_beta": 1.1}},
    ("vg11", "eta-narrow"): {"suffix": "eta-narrow", "scalar": {"eta_sigma": 0.4}},

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
    # Item 3 of #225. Recovery returns `tau_subject` below its truth in three
    # replicates of three, by about 5.8%, with the truth outside the 89%
    # interval every time, and `v_total` recovers well while
    # `subject_variance_share` carries the bias -- so the budget is estimated
    # and the *split* is not. That leaves two candidate diagnoses which call for
    # different fixes: the partition biases the split, or the split is not
    # identifiable however it is parameterised. This variant separates them by
    # reverting to the two free scales VG12 carried before 2026-08-05 and
    # scoring recovery of `tau_subject` against the record's.
    #
    # `None` is the whole override: `tau_subject_sigma = 1.5` is already on the
    # definition, inert under the partition (which is why `render_priors_table`
    # suppresses its row), and the engine falls back to sampling
    # `HalfNormal(tau_subject_sigma)` directly whenever the partition is absent.
    # `kappa` likewise reverts to its own prior rather than to an allocated
    # excess. This is a graph change, so it needs its own fit; expect the
    # sampling pathology the partition was introduced to fix to come back --
    # 59 divergences against 14, and `corr(tau_subject, kappa_young) = +0.755`.
    # A variant that samples badly still answers the recovery question, which is
    # about the estimator rather than about publishable geometry.
    ("vg12", "free-scales"): {"suffix": "free-scales", "scalar": {
        "subject_variance_partition": None}},

    # -- The spoken fallback branch (#233, #236) --
    #
    # 455 of the 1,428 spoken observations cannot condition on an observed
    # understood count, and have always been given S ~ BB(810, p_U*q, kappa_S) --
    # mean-correct, but not the paired model's marginal. Those rows are older and
    # concentrated by study, so the approximation is not ignorable and the two
    # audits both asked for it to be bounded rather than only disclosed.
    #
    # Three variants, each isolating a different thing the default could be
    # getting wrong, all registered on VG20 (the model of record for child-level
    # inference) and VG10 (the model everything else is compared against, and the
    # one whose q and dispersion the fallback rows most directly inform):
    #
    #   paired-only          drops the rows. The bound: whatever the
    #                        approximation is doing, it cannot be doing it here.
    #                        Expect wider intervals and a q that leans on the
    #                        younger, paired part of the frame -- read the
    #                        *direction* of the shift, not its size.
    #   fallback-dispersion  keeps the rows and lets the branch have its own
    #                        concentration. Nests the default at zero, so
    #                        `log_kappa_s_fallback` reads off both how much
    #                        dispersion the branch wants and which way -- the
    #                        default is not wrong in a fixed direction (it is
    #                        under-dispersed exactly when q*kappa_S > kappa_U),
    #                        so a signed readout is the point.
    #   marginal-moments     keeps the rows and gives them the concentration that
    #                        makes the likelihood exact in its first two moments.
    #                        Not a sensitivity in the usual sense -- it is the
    #                        better model, and the default's distance from it is
    #                        the size of the defect.
    #
    # `spoken_fallback` is a graph field, so each needs its own fit; none is a
    # prior tweak that `--render-only` could pick up.
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

    # -- VG13 observation window (#228) --
    #
    # These two are the widest-scoped variants registered on VG13, and the only
    # ones here that change the *data* rather than a prior. They exist because
    # VG13's 18-month cap turned out to rest on two reasons that no longer hold.
    #
    # The cap's code comment says it "avoids the WS bias (production proxy
    # comprehension) entirely". It does not do that work: `load_data` selects
    # `WORDBANK_BIVARIATE_FORMS` whenever `understood` is requested, so WS is
    # never loaded for a comprehension model at ANY age. The July review gave the
    # real reason -- above 18 months only Oxford CDI supplied bivariate rows, a
    # single study. The Romance extension of 2026-08-03 retired that objection
    # without anyone revisiting the cap: Italian Words & Gestures is registered
    # 7-24 months, so Caselli now sits alongside Floccia above 18.
    #
    # What the cap currently discards, measured on the loader's own code path in
    # VG13's language scope: 694 administrations from 323 children, all on forms
    # the pipeline already treats as genuinely bivariate. Raising the cap to 25
    # takes the pool from 6,358 rows to 7,052 -- which is VG12's pool exactly,
    # same six studies, same rows. VG12 already fits and reports these
    # observations; VG13 is the only model that drops them, and VG13 is the one
    # carrying the Down-syndrome-versus-typically-developing comparison.
    #
    # Why this costs more than a `max_age_months` override. Over 8-18 months the
    # production ratio `q` runs from a median of 0.04 to 0.22 -- the bottom limb
    # of its S, which is what justifies `eta_q_sigma = 0.20` and a logit-linear
    # trend between anchors at 10 and 16 months. Over 8-25 it runs to 0.83.
    # Extending the window alone would extrapolate that trend nine months past
    # its high anchor into the part of the curve that decelerates: on understood
    # it reaches p = 0.85 (687 words) at 25 months against an observed median of
    # 0.42. So each variant moves its high anchor into the new window, recentres
    # both high-anchor Betas on the in-sample median there, widens `eta_q` to
    # give the GP the curvature the longer window needs, and moves the GP domain,
    # GP anchor and query grid to match. Co-identified, therefore one unit --
    # the same discipline the Beta anchors are varied under above.
    #
    # The high-anchor Betas are recentred on IN-SAMPLE medians, not on published
    # norms, because no CDI comprehension norm exists above 18 months (this is
    # the same gap that makes VG12's 26-month anchor a named sensitivity target).
    # They sit just below the in-sample median, per the house convention that an
    # anchor is recentred toward the empirical level and not tightened onto it.
    #
    # `kappa_u`'s anchor ages move from (12, 17) to (12, 20) -- VG12's, since
    # VG12 is calibrated on exactly this comprehension pool over exactly this
    # window. `kappa_s` follows so the two blocks contest one span. The
    # magnitudes are inherited rather than recalibrated, and that is a named
    # limitation rather than a claim.
    #
    # Two windows rather than one, because the Oxford CDI's 418-item ceiling
    # bites unevenly. Below 19 months 1-5% of administrations sit above 90% of
    # their form's ceiling; at 19-22 it is 7-8%, and at 23-25 it jumps to 20-36%.
    # Ceiling compression biases the comprehension trend down where it bites, so
    # `window-22` is the ceiling-safe half of the pair and the difference between
    # the two measures the exposure rather than arguing about it. `window-25` is
    # the one to fit first: it is the one that answers the question, reaching a
    # median of 340 understood words against VG13's ~220.
    ("vg13", "window-25"): {"suffix": "window-25", "scalar": {
        "max_age_months": 25,
        "slope_anchors": (10, 24),
        "ages_query": [8, 10, 12, 14, 16, 18, 20, 22, 24],
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
        "ages_query": [8, 10, 12, 14, 16, 18, 20, 22],
        "gp_domain_months": (8, 22),
        "gp_anchor_age_months": 15.5,
        # 21 mo in-sample medians: understood 0.359 of 810, q 0.417.
        "p_slope_hi_u_alpha": 2.0, "p_slope_hi_u_beta": 3.2,   # median 0.369
        "p_slope_hi_q_alpha": 2.0, "p_slope_hi_q_beta": 2.6,   # median 0.425
        "eta_q_sigma": 0.5},
     "kappa": {"kappa_u": {"anchor_ages": (12.0, 20.0)},
               "kappa_s": {"anchor_ages": (12.0, 20.0)}}},

    # `window-22-vague-anchors` is `window-22` with its two HIGH anchors made
    # much less informative, and nothing else changed. It exists to answer one
    # objection to promoting `window-22` (notes/202608211100-window-22-adopted.md
    # §6): its 21-month anchors were recentred on the *in-sample* medians
    # (understood 0.359, q 0.417) because no CDI comprehension norm exists above
    # 18 months -- US-English WG stops there. VG13's own 16-month anchor is
    # norm-validated (prior median 0.228 against a Wordbank median of 0.222,
    # PRIORS.md "TD anchor priors vs Wordbank norms"), so the extension really
    # does sit on weaker footing, and this repository has already had to correct
    # one case of setting a prior from the data it is fitted to (the DS-joint
    # anchors read off the VG07 posterior; same PRIORS.md section).
    #
    # The finding at risk is specific: under `window-22` the DS/TD gap closes,
    # Delta-q = -0.00 at 300 words and +0.00 at 320 (P(TD>DS) = 0.49, 0.51),
    # where `window-25` has it reopening to +0.09. That closure needs TD q to
    # reach ~0.48 at U=320, and the q high anchor is what pins the level q can
    # reach. So the threatening displacement is UPWARD: if the anchor is holding
    # TD q down, a higher, wider one should let the gap reopen.
    #
    #   p_slope_hi_q  Beta(2, 2.6) -> Beta(1.3, 1.3)   median 0.425 -> 0.500,
    #       5-95% 0.110-0.795 -> 0.079-0.921. The centre moves off the in-sample
    #       value and the span roughly doubles. Beta(1.3, 1.3) is not invented
    #       here: it is the house broad high anchor already used by VG03, VG04,
    #       VG11 and VG12.
    #   p_slope_hi_u  Beta(2, 3.2) -> Beta(1.2, 2.0)   median 0.369 -> 0.346,
    #       5-95% 0.092-0.731 -> 0.044-0.803. Widened rather than displaced, on
    #       purpose. Beta(1.3, 1.3) would put the understood median at 0.500 =
    #       405 words, essentially at the Oxford CDI's own 418-item ceiling
    #       (0.516) -- PRIORS.md warns that a high understood anchor near 0.5
    #       "would sit against the WG ceiling and implicitly assume near-total
    #       WG comprehension". That is not a vague prior, it is a wrong one, and
    #       it would confound the test rather than sharpen it.
    #
    # Read it as a gate, not as a candidate for reporting: if the closure
    # survives, the prior limitation is a disclosure; if it does not, the
    # closure was prior-held and must not be published.
    ("vg13", "window-22-vague-anchors"): {"suffix": "window-22-vague-anchors", "scalar": {
        "max_age_months": 22,
        "slope_anchors": (10, 21),
        "ages_query": [8, 10, 12, 14, 16, 18, 20, 22],
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
    # VG19 gate G5b (notes/202608141900 SS G5b). A LEVERAGE diagnostic, not a
    # reportable narrower model, and the difference dictates the design: this
    # variant changes `max_age_months` and NOTHING else.
    #
    # The `window-22` precedent moves `gp_domain_months`, the anchors and the
    # slope priors along with the cap, because it exists to report over a narrow
    # window and the mean function has to be right there. Doing that here would
    # confound the measurement: `tau1` could then move because the population
    # curve changed rather than because the high-age observations left, and the
    # whole point is to attribute its movement to one cause. Holding the GP
    # domain at (8, 115) leaves the basis and both slope anchors exactly where
    # the reference fit put them, so the likelihood below 84 months is the same
    # function of the same parameters. Above 84 the GP is unconstrained, which
    # costs nothing: no observation is there to be fitted.
    #
    # Reading it: `tau1` stable => the rate is carried by the longitudinal
    # subset and the concentration is harmless. `tau1` materially moved => it is
    # set by the sparse tail (50 spoken rows above 84 months, 3.5% of the data
    # carrying 28-35% of the cross-sectional information about it), and the
    # slope should be reported only over the range that supports it.
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

    # VG22's rank family. `notes/202608221000-four-by-four-gate1.md` §5 is
    # explicit that this analysis cannot choose `k`: rank 2 and rank 3 are 2.60
    # apart on 2 df and rank 3 and rank 4 are identical, so 2 and 3 are both
    # defensible and 4 is not. Registering the other two against the default is
    # the "honest way to settle it" the note asks for, and it is cheap because
    # the three differ by one column of L.
    #
    # The default moved from rank 2 to rank 3 on 2026-08-24, so the variants are
    # now 1 and 2. What decided it was the fits rather than the residual
    # likelihood: rank 3 cleared the gate on plain `rep` where rank 2 needed a
    # hightune, and the two disagree on the spoken child-slope scale by about 2.8
    # combined standard errors. See
    # notes/202608231420-vg22-factor-anchor-bimodality.md §5, §7.
    #
    # Rank 1 is not a throwaway lower bound. It is the rank-one case Proposal A1
    # assumes -- every child's four effects one deviate scaled four ways, so all
    # four correlate perfectly -- and Gate 1 rejects it decisively on residuals
    # (2 x delta logL = 221 on 3 df). Fitting it here tests that rejection under
    # the real likelihood rather than the Gaussian-on-logit approximation.
    #
    # The two rate scales are held at the default's 0.5 so the rank is the only
    # thing that moves, which is what keeps the family one-factor.
    ("vg22", "rank-1"): {"suffix": "rank-1", "scalar": {
        "subject_factor": SubjectFactorPriorParams(
            rank=1, tau1_u_sigma=0.5, tau1_q_sigma=0.5, ref_age_months=36.0)}},
    ("vg22", "rank-2"): {"suffix": "rank-2", "scalar": {
        "subject_factor": SubjectFactorPriorParams(
            rank=2, tau1_u_sigma=0.5, tau1_q_sigma=0.5, ref_age_months=36.0)}},

    # -- VG16: what the cross-lag coefficient survives (#242) --
    #
    # VG16 had no registered sensitivities at all, which #242 records as a
    # defect in its own right: `beta_lag` "is assumed constant across gaps of
    # 1-28 months, multiple studies and checklist-form transitions, without
    # registered VG16 sensitivities". These are the two of that list expressible
    # as field overrides; gap, leave-one-study-out and the zero-count boundary
    # need new fields on `BivariateModelDefinition`, which is shared by all
    # twelve bivariate models, so adding them re-stales every one and is a
    # decision rather than a registration.
    #
    # Read these against `beta_lag`, not against the trajectories. Until
    # 2026-08-25 `compare.py` scored only the eight trajectory series, so a
    # variant could halve the coefficient and still be called robust; the
    # coefficient is now loaded as a scalar from `diagnostics.csv`
    # (`compare.load_beta_lag`) alongside VG15's `psi`.
    #
    # `conditional-only` is the sharper of the two. VG16's cross-lag is a claim
    # about children whose earlier comprehension was measured, but 455 of 1,428
    # spoken rows enter through the fallback branch with no observed
    # comprehension parent at all -- 444 with no comprehension total and 11
    # where spoken exceeds understood. Dropping them leaves only rows whose
    # spoken count is conditioned on an observed understood count, which is the
    # population the coefficient is supposed to describe. If `beta_lag` moves
    # materially, the cross-lag is partly an artefact of the substitute
    # likelihood on rows that never observed the predictor's parent.
    ("vg16", "conditional-only"): {"suffix": "conditional-only", "scalar": {
        "spoken_fallback": SPOKEN_FALLBACK_PAIRED_ONLY}},
    #
    # `dse-native-only` keeps only administrations recorded natively on the 810
    # reference, so no count is scored against a denominator its form did not
    # use. The lag predictor is a logit of a *proportion*, understood / 810, so
    # a short-form source enters it already deflated -- the harmonisation acts
    # directly on the regressor here, not only on the outcome. Expect partial
    # coverage rather than a clean verdict: the same restriction keeps 278 of
    # VG10's 1,521 rows, and it changes study composition as well as size.
    ("vg16", "dse-native-only"): {
        "suffix": "dse-native-only",
        "scalar": {"dse_native_only": True},
    },

    # -- VG21: the anchors it was promoted with (#228, #240) --
    #
    # VG21 is the window-22 promotion, and its two HIGH anchors were recentred
    # on **in-sample medians** because no CDI comprehension norm exists above 18
    # months (#228). Anchors set from the data and then used to fit the data are
    # double-dipping unless the conclusions are shown not to turn on them --
    # which is exactly the test Target 8 applied to every other recalibrated
    # anchor in the project, all of which passed.
    #
    # This is `("vg13", "window-22-vague-anchors")` expressed against VG21's own
    # baseline: VG21 already carries the window, the anchor ages, the GP domain
    # and `eta_q_sigma`, so only the two high-anchor Betas move, and the widened
    # values are taken from that entry unchanged so the two remain comparable.
    # p_slope_hi_q is deliberately Beta(1.3, 1.3) rather than flatter: the note
    # on the VG13 entry records why a median at 0.5 would sit against the Oxford
    # CDI's own 418-item ceiling and confound the test rather than sharpen it.
    #
    # A gate, not a reporting candidate. If VG21's conclusions survive, the
    # in-sample anchors are a disclosure; if they do not, they are load-bearing
    # and must not be published without saying so.
    ("vg21", "vague-anchors"): {"suffix": "vague-anchors", "scalar": {
        "p_slope_hi_u_alpha": 1.2, "p_slope_hi_u_beta": 2.0,   # median 0.346
        "p_slope_hi_q_alpha": 1.3, "p_slope_hi_q_beta": 1.3}},  # median 0.500

    # -- VG23: whether the correlation is evidenced or regularised (#229) --
    #
    # VG23 exists to estimate `rho_uq` on the typically-developing pool, and its
    # LKJ concentration is `eta = 2` -- chosen to match VG20 so the two
    # populations are comparable, and deliberately a "gentle pull toward
    # independence, so a correlation has to be evidenced". That is a defensible
    # choice and an informative prior, which is precisely why it needs a check.
    #
    # `eta = 1` is the flat LKJ: uniform over correlation matrices, no pull in
    # either direction. If `rho_uq` barely moves, the correlation is carried by
    # the data and the matched-prior comparison with VG20 is sound. If it moves
    # materially, `eta = 2` is doing the work and the DS-versus-TD contrast is
    # partly a statement about two priors.
    ("vg23", "eta-flat"): {"suffix": "eta-flat", "scalar": {
        "subject_re_correlation_eta": 1.0}},
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
        out.append(
            make_variant(
                base,
                config_suffix=spec["suffix"],
                scalar_over=spec.get("scalar"),
                kappa_over=spec.get("kappa"),
            )
        )
    return out
