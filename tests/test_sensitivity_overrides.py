# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the prior-sensitivity override + registry tooling (issue #89).

These are pure/fast (no data, no sampling): they pin that a variant applies the
requested overrides, isolates its output via a suffixed ``config_name``, and
never mutates the committed model definitions (including the nested kappa
priors, which must be fresh objects rather than aliases of the base's).
"""

import dataclasses

import pytest

from vocab_growth.models.definitions import VG10, VG11, VG12, VG13, VG15, VG19, VG20
from vocab_growth.sensitivity.overrides import make_variant, replace_kappa
from vocab_growth.sensitivity.registry import VARIANTS, build_variant, variants_for


def test_make_variant_suffixes_config_name_and_leaves_base_untouched():
    base_low = (VG10.p_slope_low_q_alpha, VG10.p_slope_low_q_beta)
    v = make_variant(VG10, config_suffix="q-broad", scalar_over={
        "p_slope_low_q_alpha": 1.0, "p_slope_low_q_beta": 1.5})
    assert v.config_name == f"{VG10.config_name}-q-broad"
    assert v.model_id == VG10.model_id
    assert (v.p_slope_low_q_alpha, v.p_slope_low_q_beta) == (1.0, 1.5)
    # Base instance is untouched.
    assert (VG10.p_slope_low_q_alpha, VG10.p_slope_low_q_beta) == base_low
    assert VG10.config_name == "age-understood-spoken-ds-re-subj-uq-anchored"


def test_make_variant_nested_kappa_is_fresh_not_aliased():
    base_sigma = VG10.kappa_s.kappa_min_sigma
    v = make_variant(VG10, config_suffix="kappa-broadfloor", kappa_over={
        "kappa_u": {"kappa_min_sigma": 1.0}, "kappa_s": {"kappa_min_sigma": 1.0}})
    assert v.kappa_s.kappa_min_sigma == 1.0
    assert v.kappa_u.kappa_min_sigma == 1.0
    # The base's kappa objects are neither mutated nor shared with the variant.
    assert VG10.kappa_s.kappa_min_sigma == base_sigma
    assert v.kappa_s is not VG10.kappa_s
    assert v.kappa_u is not VG10.kappa_u


def test_make_variant_rejects_bad_input():
    with pytest.raises(ValueError):
        make_variant(VG10, config_suffix="")  # empty suffix
    with pytest.raises(TypeError):
        make_variant(VG10, config_suffix="x", scalar_over={"not_a_field": 1.0})
    with pytest.raises(ValueError):
        make_variant(VG10, config_suffix="x", kappa_over={"kappa_u": {"nope": 1.0}})
    with pytest.raises(ValueError):
        make_variant(VG10, config_suffix="x", kappa_over={"no_such_kappa": {"a_kappa_mu": 0.0}})


def test_replace_kappa_overrides_only_named_fields():
    kp = VG15.kappa_sign
    new = replace_kappa(kp, kappa_min_sigma=1.0)
    assert new.kappa_min_sigma == 1.0
    assert new.a_kappa_mu == kp.a_kappa_mu  # untouched
    assert new is not kp


def test_replace_kappa_checks_fields_against_the_form_in_use():
    """A variant written for one parameterisation must not survive a migration.

    VG15 carries both forms — anchored on understood, legacy on the signed ratio
    — so it exercises the dispatch in one object. Silently accepting a legacy
    field name on an anchored block would leave a registered sensitivity check
    quietly testing nothing.
    """
    with pytest.raises(ValueError, match="two-anchor form"):
        replace_kappa(VG15.kappa_u, a_kappa_mu=0.0)
    with pytest.raises(ValueError, match="legacy form"):
        replace_kappa(VG15.kappa_sign, excess_young_mu=0.0)

    # and each accepts its own
    assert replace_kappa(VG15.kappa_u, excess_young_mu=1.0).excess_young_mu == 1.0
    assert replace_kappa(VG15.kappa_sign, a_kappa_mu=1.0).a_kappa_mu == 1.0


def test_every_registered_variant_builds():
    """The registry is only useful if every entry in it can be materialised.

    Nothing else covers this: the variants are data, so a stale override survives
    import and lint and only fails when someone tries to fit it.
    """
    for key in VARIANTS:
        build_variant(*key)


def test_registry_counts_and_models():
    # 27 §7 targets + 7 Target-8 young-age anchor variants (#146), two
    # signing-source variants and three repeated-measures sensitivities.
    #
    # The two `us01-ceiling-excluded` variants were retired with the Edgin audit:
    # the records they excluded are now masked by default, so the variants could
    # only have excluded records already excluded. A registered check that cannot
    # fail is worse than no check — see the note in registry.py. They are replaced
    # by the inverse `us01-implausible-reinstated` pair, which asks what changes if
    # that default exclusion is mistaken — the only remaining check on it, the
    # source author no longer holding the original files.
    #
    # +3 on 2026-08-06: the three `sign-peak-age-*` variants. VG15's signed peak
    # age became a sampled parameter that day, so for the first time there is
    # something for a peak-age variant to vary — the existing `sign-peak-lo`/`-hi`
    # pair varies the peak's HEIGHT, and could not have covered this.
    #
    # -1 on 2026-08-12: `sign-include-uk06` retired on the same principle as the
    # ceiling variants. The source confirmed uk_06 used the standard DSE
    # checklists, whose column 2 is "understands and signs" — a total sign count —
    # so uk_06 is now included by default and the variant cannot vary anything.
    #
    # +6 on 2026-08-12, all closing gaps the uk_07/es_01 work opened or exposed:
    # the `dse-native-only` pair (the 810 reference denominator, #190 — the first
    # check on the harmonisation the sufficiency result proves no aggregate
    # analysis can test), the `tau-psi-*` pair (a data-informed prior on a newly
    # added, weakly identified parameter, the same condition that created
    # Target 8), and the `psi-drop-*` pair (psi's source composition, which both
    # inclusion flags advertise but nothing ran).
    #
    # +1 on 2026-08-14: `clamp-q-only`. `clamp_mean_above_hi_anchor` levels BOTH
    # the understood mean and `q` off above the 84 mo anchor, and spoken is
    # p_U * q, so the spoken trajectory inherits both — the corner at 84 months
    # is the sharpest feature of its whole trajectory. Measurement says the
    # saturation the flag was added for is `q`'s alone: extrapolating VG10's own
    # fitted anchors gives q = 0.996 at 115 mo with P(mean > 0.99) = 0.999,
    # while understood reaches 0.962 and never crosses 0.99 in any draw. This
    # variant makes that a fit rather than an argument. See
    # notes/202608141200-clamp-q-only.md.
    #
    # +1 on 2026-08-14: `a1-tau-age-varying`, Proposal A1 — the age variation
    # moved off `kappa` and onto the between-child scale, on VG10 only. Unlike
    # every variant above it this is a GRAPH change, carried on the existing
    # `tau_subj_*_sigma` fields (as CLAMP_Q_ONLY is carried on
    # `clamp_mean_above_hi_anchor`) so no definition gains a field and no
    # fingerprint moves. It is registered as a diagnostic, not a candidate model
    # of record: scaling one per-child deviate by tau(age) imposes perfect rank
    # correlation across age, which is measured at 0.28 beyond two years. See
    # notes/202607261540 §9 and notes/202608141600 §§8-10.
    #
    # +2 on 2026-08-17: the VG13 `window-*` pair (#228). Every variant above
    # varies a prior; these two vary the observation *window*, and they are the
    # first to do so. VG13's 18-month cap was justified in code by avoiding the
    # WS production-proxy bias — work the form filter in `load_data` already does
    # unconditionally — and in review by there being only one study above 18
    # months, which the Romance extension of 2026-08-03 retired by admitting
    # Italian Words & Gestures (registered 7-24). 694 admissible administrations
    # sit above the cap. A window change drags its co-identified anchors, GP
    # domain and query grid with it, so each is registered as one unit; see the
    # registry comment for the measurements.
    #
    # +3 on 2026-08-19: VG20's kappa placement trio (#229), and the first
    # variants registered against the model of record rather than against a
    # development step. Two of them vary where the dispersion prior is placed
    # rather than how wide it is; the third combines them. `anchor_ages` was
    # already an overridable field, but treating anchor *placement* as a
    # registered question is new, and it followed the measurement that
    # kappa_min carries 42.5% of reported kappa_u at 84 months and 95.2% of
    # kappa_s while recovery scores it at -40% to -60%.
    #
    # Restated the same day as `kappa-anchor-24-48`, `kappa-floor-generic` and
    # `kappa-pre-promotion` when the combination was promoted into
    # `_DS_JOINT_*_KAPPA_RE`: after promotion the originals perturbed toward the
    # model of record rather than away from it, and one of them had become a
    # literal no-op. The count is unchanged because each was inverted rather
    # than dropped.
    #
    # +1 on 2026-08-21: `window-22-vague-anchors`, gating the promotion of
    # `window-22` to a registered model. `window-22`'s 21-month anchors were
    # recentred on in-sample medians because no CDI comprehension norm exists
    # above 18 months, and the finding that rests on them -- the DS/TD gap
    # closing by 300 words understood -- is exactly what a level-pinning prior
    # could manufacture. The variant displaces the q high anchor upward and
    # widens both, so a surviving closure is the data's and not the prior's.
    # 58 with VG19's `max-age-84` (gate G5b): a leverage diagnostic that caps the
    # data at 84 months and changes nothing else, so any movement in `tau1` is
    # attributable to the discarded high-age rows rather than to a re-placed
    # mean function. See notes/202608141900 SS G5b.
    assert len(VARIANTS) == 58
    assert len(variants_for("vg19")) == 1
    assert len(variants_for("vg10")) == 14
    assert len(variants_for("vg11")) == 5
    assert len(variants_for("vg12")) == 4
    assert len(variants_for("vg13")) == 4
    assert len(variants_for("vg15")) == 27
    assert len(variants_for("vg20")) == 3


def test_td_models_account_for_repeated_children_by_default():
    assert VG11.use_subject_re
    assert VG12.use_subject_re
    assert VG13.use_subject_re_u
    assert VG13.use_subject_re_q

    (single_vg13,) = build_variant("vg13", "single-admin")
    assert single_vg13.one_observation_per_subject
    assert not single_vg13.use_subject_re_u
    assert not single_vg13.use_subject_re_q


def test_build_variant_all_and_named():
    all_vg15 = build_variant("vg15", "all")
    assert len(all_vg15) == 27
    # All distinct config_names, all still VG15.
    assert len({d.config_name for d in all_vg15}) == 27
    assert all(d.model_id == "VG15" for d in all_vg15)
    # psi-neutral applies both hyperparameters.
    (psi,) = build_variant("vg15", "psi-neutral")
    assert (psi.log_psi_mu, psi.log_psi_sigma) == (0.0, 0.5)

    # The retired ceiling variants must be gone from the registry, not merely
    # unused: a registered sensitivity whose records are already excluded by
    # default cannot fail, and would read as robustness it has not demonstrated.
    for model in ("vg10", "vg15"):
        with pytest.raises(KeyError, match="us01-ceiling-excluded"):
            build_variant(model, "us01-ceiling-excluded")

    # Same principle for sign-include-uk06. It asked "what if uk_06's signing IS
    # comparable?" — answered on 2026-08-12 when the source confirmed the standard
    # DSE checklists, after which uk_06 is included by default and the variant has
    # nothing left to vary. See data/vocab_data_uk_06.md and issue #211.
    with pytest.raises(KeyError, match="sign-include-uk06"):
        build_variant("vg15", "sign-include-uk06")


def test_implausible_production_reinstatement_is_registered_and_bites():
    """The inverse sensitivity must exist, flip the flag, and change the frame.

    The 30 masked administrations cannot be confirmed defective at source — the
    source author no longer holds the original files — so this variant is the only
    published check on that exclusion. It has to move real observations, or it
    repeats the fault of the variants it replaces.
    """
    for model, model_id in (("vg10", "VG10"), ("vg15", "VG15")):
        (variant,) = build_variant(model, "us01-implausible-reinstated")
        assert variant.include_implausible_production is True
        assert variant.model_id == model_id
        assert "us01-implausible-reinstated" in variant.config_name

    # The baselines must not carry the flag, or the variant would be a no-op.
    assert VG10.include_implausible_production is False
    assert VG15.include_implausible_production is False


def test_dse_native_variant_is_registered_and_bites():
    """The 810-denominator check must exist, flip the flag, and move real rows.

    This is the only check on the harmonisation that carries a 416-item Oxford
    count onto an 810-item denominator, and the sufficiency result
    (notes/202607261540) is the proof that no aggregate analysis of these data can
    test that assumption instead. A variant that silently stopped removing rows
    would read as robustness it has not demonstrated.
    """
    for model, model_id in (("vg10", "VG10"), ("vg15", "VG15")):
        (variant,) = build_variant(model, "dse-native-only")
        assert variant.dse_native_only is True
        assert variant.model_id == model_id
        assert "dse-native-only" in variant.config_name

    assert VG10.dse_native_only is False
    assert VG15.dse_native_only is False


def test_psi_variants_cover_the_scale_and_the_sources():
    """psi's two untested degrees of freedom after the 2026-08-12 study term.

    ``tau_psi_sigma`` was set from the measured between-study spread, which makes
    it data-informed rather than externally justified — the condition Target 8
    exists for — and with four informed studies it is weakly identified, so it
    governs how far the per-study values shrink and therefore the headline itself.
    Separately, both cross-tab inclusion flags document that setting them False
    isolates a source's pull on psi, which nothing ran until these variants.
    """
    (narrow,) = build_variant("vg15", "tau-psi-narrow")
    (wide,) = build_variant("vg15", "tau-psi-wide")
    assert narrow.tau_psi_sigma < VG15.tau_psi_sigma < wide.tau_psi_sigma

    # Each source variant drops exactly one cross-tab and leaves the other alone,
    # so the contrast attributes movement to that source rather than to "fewer
    # cells in general".
    (no_es01,) = build_variant("vg15", "psi-drop-es01")
    assert (no_es01.include_es01_cells, no_es01.include_uk07_cells) == (False, True)
    (no_uk07,) = build_variant("vg15", "psi-drop-uk07")
    assert (no_uk07.include_es01_cells, no_uk07.include_uk07_cells) == (True, False)

    # Both must be on in the model of record, or the variants are no-ops.
    assert VG15.include_es01_cells and VG15.include_uk07_cells


def test_build_variant_rejects_unknown():
    with pytest.raises(KeyError):
        build_variant("vg99", "q-broad")
    with pytest.raises(KeyError):
        build_variant("vg10", "no-such-variant")


def test_variants_are_single_factor_or_documented_pairs():
    # Every variant produces a definition whose model_type matches its base
    # (sanity that replace preserved the class), and changes at least one field.
    for (model_key, name) in VARIANTS:
        (v,) = build_variant(model_key, name)
        base = {
            "vg10": VG10,
            "vg11": VG11,
            "vg12": VG12,
            "vg13": VG13,
            "vg15": VG15,
            "vg19": VG19,
            "vg20": VG20,
        }[model_key]
        assert v.model_type == base.model_type
        assert dataclasses.asdict(v) != dataclasses.asdict(base)


def test_variants_that_disable_subject_effects_also_clear_the_partition():
    """A variant turning off subject effects must clear the variance partition.

    The partition allocates one scatter budget *between* the subject scale and the
    young kappa anchor, so without a subject scale there is nothing to allocate and
    the engine raises. Adopting the partition on VG11/VG12 broke both `single-admin`
    variants for two days without any test noticing, because `build_variant` only
    constructs the definition — the failure appears when a model graph is built
    from it, which nothing here does.
    """
    for model_key in ("vg11", "vg12"):
        (variant,) = build_variant(model_key, "single-admin")
        assert variant.use_subject_re is False, model_key
        assert variant.subject_variance_partition is None, (
            f"{model_key} single-admin leaves the variance partition set while "
            "disabling subject effects; the engine rejects that combination."
        )


def test_window_variants_stay_inside_their_own_gp_domain_and_anchors():
    """A window variant must carry its GP domain, anchors and grid with it.

    Pure checks, so they run without the database. The window is the only
    variant factor here that the *rest* of the definition has to agree with:
    the GP domain must contain the data (``build_utils`` refuses otherwise),
    the high slope anchor must sit inside the window or the logit-linear trend
    is extrapolated past its own anchor, and the query grid must not report
    ages the window excludes.
    """
    for name, cap in (("window-25", 25), ("window-22", 22)):
        (v,) = build_variant("vg13", name)
        lo_domain, hi_domain = v.gp_domain_months
        assert v.max_age_months == cap, name
        assert hi_domain == cap, f"{name}: GP domain stops at {hi_domain}, window at {cap}"
        assert lo_domain == 8, name
        lo_anchor, hi_anchor = v.slope_anchors
        assert lo_anchor >= lo_domain and hi_anchor <= cap, (
            f"{name}: slope anchors {v.slope_anchors} are not inside the window"
        )
        assert hi_anchor > VG13.slope_anchors[1], (
            f"{name}: the high anchor did not move into the new window, so the "
            "trend extrapolates past it"
        )
        assert v.gp_anchor_age_months == (lo_anchor + hi_anchor) / 2, name
        assert max(v.ages_query) <= cap, name
        assert max(v.ages_query) > max(VG13.ages_query), name
        # eta_q was 0.20 because only the bottom limb of q's S was in view.
        assert v.eta_q_sigma > VG13.eta_q_sigma, name
        # The kappa anchor ages must sit inside the window too, or the "old"
        # anchor describes dispersion somewhere the model no longer stops.
        for block in (v.kappa_u, v.kappa_s):
            assert max(block.anchor_ages) <= cap, name
            assert max(block.anchor_ages) > max(VG13.kappa_u.anchor_ages), name


def test_window_variants_admit_the_rows_the_cap_discards():
    """The point of the pair: more data, the same six studies, no WS.

    Builds the real graphs, because the failure mode this guards against —
    a variant whose definition is valid but whose engine rejects it — only
    appears at build time. That gap is what let the ``single-admin`` variants
    sit broken for two days.
    """
    import contextlib
    import io
    import os
    import tempfile

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models import common_bivariate as cb
    from vocab_growth.models import common_bivariate_re as cbr
    from vocab_growth.models.common import ModelFitContext

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def prepared(definition, root):
        ctx = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name=definition.model_id,
                config_name=definition.config_name,
                output_root_dir=root,
                ci_prob=0.90,
                interval_kind="hdi",
            ),
            sampling=sampling.get_sampling_configuration("dev"),
        )
        os.makedirs(ctx.reporting.output_dir, exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            cbr.prepare_bivariate_re_data(ctx, definition)
            cb.configure_bivariate_priors(ctx, definition)
            cbr.build_model_re(ctx, definition)
        frame = next(v for v in vars(ctx).values() if hasattr(v, "columns"))
        return ctx, frame

    with tempfile.TemporaryDirectory() as root:
        base_ctx, base = prepared(VG13, root)
        rows = {"base": len(base)}
        studies = {"base": set(base["study"].unique())}
        free_rvs = {"base": len(base_ctx.model.free_RVs)}
        for name in ("window-22", "window-25", "window-22-vague-anchors"):
            (variant,) = build_variant("vg13", name)
            ctx, frame = prepared(variant, root)
            rows[name] = len(frame)
            studies[name] = set(frame["study"].unique())
            free_rvs[name] = len(ctx.model.free_RVs)
            assert frame["age"].max() == variant.max_age_months, name

    # Strictly more data, and monotone in the window.
    assert rows["base"] < rows["window-22"] < rows["window-25"], rows
    # The window is the only factor: no study enters or leaves, and the graph
    # keeps exactly the structure of the model of record.
    assert studies["base"] == studies["window-22"] == studies["window-25"], studies
    assert len(set(free_rvs.values())) == 1, free_rvs
    # `window-22-vague-anchors` changes two prior HYPERparameters and nothing
    # else, so it must be indistinguishable from `window-22` on every structural
    # axis this test measures. If it ever differs here, the variant has stopped
    # being a clean prior-sensitivity check and its result cannot be read as one.
    assert rows["window-22-vague-anchors"] == rows["window-22"], rows
    assert studies["window-22-vague-anchors"] == studies["window-22"], studies
