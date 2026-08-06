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

from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.sensitivity.overrides import make_variant

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
    ("vg15", "sign-include-uk06"): {"suffix": "sign-include-uk06", "scalar": {
        "include_uk06": True}},

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
    ("vg11", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True, "use_subject_re": False}},
    ("vg12", "single-admin"): {"suffix": "single-admin", "scalar": {
        "one_observation_per_subject": True, "use_subject_re": False}},
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

    # -- Target 6: VG15 psi (association) --
    ("vg15", "psi-neutral"): {"suffix": "psi-neutral", "scalar": {"log_psi_mu": 0.0, "log_psi_sigma": 0.5}},
    ("vg15", "psi-broad"): {"suffix": "psi-broad", "scalar": {"log_psi_mu": 0.0, "log_psi_sigma": 1.0}},
    ("vg15", "psi-strong"): {"suffix": "psi-strong", "scalar": {"log_psi_mu": 0.6, "log_psi_sigma": 0.5}},

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
