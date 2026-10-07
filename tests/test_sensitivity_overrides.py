# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Check sensitivity overrides, output names and definition isolation.

Most tests construct definitions without sampling. Selected tests prepare data
or build graphs to ensure that registered variants change the intended inputs
and remain executable.
"""

import dataclasses

import pytest

from vocab_growth.models.definitions import (
    MODEL_REGISTRY,
    VG10,
    VG11,
    VG12,
    VG13,
    VG15,
    VG21,
)
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
    # Pin the variant count so additions require an explicit review here.

    assert len(VARIANTS) == 118
    assert len(variants_for("vg25")) == 10
    assert len(variants_for("vg14")) == 3
    assert len(variants_for("vg16")) == 9
    assert len(variants_for("vg21")) == 6
    assert len(variants_for("vg23")) == 4
    assert len(variants_for("vg26")) == 5
    assert len(variants_for("vg22")) == 2
    assert len(variants_for("vg19")) == 1
    assert len(variants_for("vg10")) == 19
    assert len(variants_for("vg11")) == 8
    assert len(variants_for("vg12")) == 8
    assert len(variants_for("vg13")) == 4
    assert len(variants_for("vg15")) == 32
    assert len(variants_for("vg20")) == 7


def test_vg25s_in_cells_arm_is_the_population_baseline_in_the_cells():
    """The within-child baseline in the cells was bimodal; the arm must not be it."""
    import dataclasses

    from vocab_growth.models.definitions import MODEL_REGISTRY

    base = MODEL_REGISTRY["vg25"]
    assert (base.sign_lag_baseline, base.sign_lag_in_cells) == ("within", False)
    (arm,) = build_variant("vg25", "sign-lag-in-cells")
    assert (arm.sign_lag_baseline, arm.sign_lag_in_cells) == ("population", True)
    changed = {
        item.name
        for item in dataclasses.fields(base)
        if getattr(base, item.name) != getattr(arm, item.name)
    }
    assert changed == {"sign_lag_baseline", "sign_lag_in_cells", "config_name", "banner"}
    assert "sign-lag-marginal-only" not in {name for _model, name in VARIANTS if _model == "vg25"}


def test_vg25s_leave_one_study_out_arms_remove_one_study_and_nothing_else():
    """The exclusion tuple is the whole of what each arm claims to test.

    A second study in the tuple, or any other field moving with it, would turn a
    leave-one-study-out check into something that cannot be read as one.
    """
    import dataclasses

    from vocab_growth.models.definitions import MODEL_REGISTRY

    base = MODEL_REGISTRY["vg25"]
    for variant, study in (("no-ie02", "ie_02"), ("no-uk05", "uk_05")):
        (arm,) = build_variant("vg25", variant)
        assert arm.exclude_studies == (study,)
        assert arm.config_name == f"{base.config_name}-{variant}"
        changed = {
            item.name
            for item in dataclasses.fields(base)
            if getattr(base, item.name) != getattr(arm, item.name)
        }
        # `config_name` and `banner` are the variant's identity, rewritten for
        # every registered arm; `exclude_studies` is the only substantive field.
        assert changed == {"exclude_studies", "config_name", "banner"}
        assert arm.banner.endswith(f"[sensitivity: {variant}]")


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
    assert len(all_vg15) == 32
    # All distinct config_names, all still VG15.
    assert len({d.config_name for d in all_vg15}) == 32
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
    """Require the reinstatement flag to change the prepared frame."""
    for model, model_id in (("vg10", "VG10"), ("vg15", "VG15")):
        (variant,) = build_variant(model, "us01-implausible-reinstated")
        assert variant.include_implausible_production is True
        assert variant.model_id == model_id
        assert "us01-implausible-reinstated" in variant.config_name

    # The baselines must not carry the flag, or the variant would be a no-op.
    assert VG10.include_implausible_production is False
    assert VG15.include_implausible_production is False


def test_ie02_comprehension_masked_arm_sets_its_flag_and_nothing_else():
    """The short-form arm (2026-09-15) changes one substantive field.

    It exists to show what keeping ie_02 on the 810 scale carries, so anything
    else moving with the flag would make it unreadable as that check.
    """
    from vocab_growth.models.definitions import MODEL_REGISTRY

    for model in ("vg10", "vg15", "vg20"):
        base = MODEL_REGISTRY[model]
        assert base.mask_dse_short_form_comprehension is False
        (arm,) = build_variant(model, "ie02-comprehension-masked")
        changed = {
            item.name
            for item in dataclasses.fields(base)
            if getattr(base, item.name) != getattr(arm, item.name)
        }
        assert changed == {"mask_dse_short_form_comprehension", "config_name", "banner"}
        assert arm.mask_dse_short_form_comprehension is True
        assert arm.config_name == f"{base.config_name}-ie02-comprehension-masked"


def test_masked_production_reinstatement_lifts_both_rules():
    """Keep the combined reinstatement distinct from lifting only one masking rule."""
    for model, model_id in (("vg10", "VG10"), ("vg15", "VG15")):
        (variant,) = build_variant(model, "us01-masked-production-reinstated")
        assert variant.include_implausible_production is True
        assert variant.include_same_day_disagreements is True
        assert variant.model_id == model_id
        assert "us01-masked-production-reinstated" in variant.config_name

        (one_factor,) = build_variant(model, "us01-implausible-reinstated")
        assert one_factor.include_same_day_disagreements is False

    assert VG10.include_same_day_disagreements is False
    assert VG15.include_same_day_disagreements is False


def test_dse_native_variant_is_registered_and_bites():
    """Require native-checklist restriction to remove the intended real rows."""
    for model, model_id in (("vg10", "VG10"), ("vg15", "VG15")):
        (variant,) = build_variant(model, "dse-native-only")
        assert variant.dse_native_only is True
        assert variant.model_id == model_id
        assert "dse-native-only" in variant.config_name

    assert VG10.dse_native_only is False
    assert VG15.dse_native_only is False


def test_psi_variants_cover_the_scale_and_the_sources():
    """Vary the study-scale prior and each composition-source gate separately."""
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
        # Resolved from the registry rather than from a literal map: the map
        # had to be edited by hand every time a model gained its first variant,
        # and a missing entry failed as a bare KeyError that reads like a broken
        # variant rather than a stale test.
        base = MODEL_REGISTRY[model_key]
        assert v.model_type == base.model_type
        assert dataclasses.asdict(v) != dataclasses.asdict(base)


def test_variants_that_disable_subject_effects_also_clear_the_partition():
    """Clear the variance partition when removing its child-scale component."""
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
    """Build wider-window variants and check added rows without new studies or forms."""
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


def test_vg12_free_scales_swaps_the_coordinates_and_nothing_else():
    """Switch VG12's scale parameterisation without changing other definition fields."""
    (variant,) = build_variant("vg12", "free-scales")

    assert variant.subject_variance_partition is None
    assert variant.use_subject_re is True, (
        "unlike `single-admin`, this variant keeps the subject effects -- it "
        "moves the scale out of the budget rather than removing it"
    )
    assert variant.tau_subject_sigma == VG12.tau_subject_sigma, (
        "the scale's own prior is already on the definition, inert under the "
        "partition; the variant makes it live rather than changing it"
    )

    changed = {
        field.name
        for field in dataclasses.fields(VG12)
        if getattr(VG12, field.name) != getattr(variant, field.name)
    }
    assert changed == {"subject_variance_partition", "config_name", "banner"}, changed


def test_vg12_free_scales_builds_a_real_graph():
    """Check the free-scale variable set, parameter count and finite initial log probability."""
    import contextlib
    import io as _io
    import os
    import tempfile

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling
    import numpy as np

    import vocab_growth.data_utils as vocab_data_utils
    import vocab_growth.models.common_univariate_re as engine
    from vocab_growth.models.common import ModelFitContext

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def build(definition, root):
        ctx = ModelFitContext(
            reporting=reporting.ReportingConfiguration(
                model_name=definition.model_id,
                config_name=definition.config_name,
                output_root_dir=root,
                ci_prob=0.89,
                interval_kind="eti",
            ),
            sampling=sampling.get_sampling_configuration("dev"),
        )
        os.makedirs(ctx.reporting.output_dir, exist_ok=True)
        with contextlib.redirect_stdout(_io.StringIO()):
            for name, stage in engine.univariate_re_stages(definition):
                if name == "Prior predictive checks":
                    break
                stage(ctx)
        return ctx.model

    (variant,) = build_variant("vg12", "free-scales")
    with tempfile.TemporaryDirectory() as root:
        record = build(VG12, root)
        free = build(variant, root)

        record_names = {rv.name for rv in record.free_RVs}
        variant_names = {rv.name for rv in free.free_RVs}
        assert record_names - variant_names == {"v_total", "subject_variance_share"}
        assert variant_names - record_names == {"tau_subject", "kappa_excess_young"}
        assert len(free.free_RVs) == len(record.free_RVs), (
            "a change of coordinates must not change the parameter count"
        )

        logp = free.compile_logp()(free.initial_point())
        assert np.isfinite(logp), "the variant's graph does not initialise"


def _prepared_bivariate(definition, root):
    """Build a variant's real graph, returning ``(context, analysis frame)``."""
    import contextlib
    import io
    import os

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models import common_bivariate as cb
    from vocab_growth.models import common_bivariate_re as cbr
    from vocab_growth.models.common import ModelFitContext

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


def test_vg16_scope_variants_build_and_actually_narrow_the_data():
    """VG16's two registered variants (#242), built rather than asserted.

    Both are *scope* changes, so the thing to check is that each really removes
    the rows it claims to and still produces a graph carrying ``beta_lag`` —
    a cross-lag variant that silently lost its coefficient would be scored as a
    robust result rather than as a broken variant.
    """
    import os
    import tempfile

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models.definitions import VG16

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    with tempfile.TemporaryDirectory() as root:
        base_ctx, base = _prepared_bivariate(VG16, root)
        seen = {"base": (len(base), set(base["study"].unique()))}
        for name in ("conditional-only", "dse-native-only"):
            (variant,) = build_variant("vg16", name)
            ctx, frame = _prepared_bivariate(variant, root)
            seen[name] = (len(frame), set(frame["study"].unique()))
            names = {v.name for v in ctx.model.free_RVs}
            assert "beta_lag" in names, f"{name} lost the cross-lag coefficient"

    # `dse-native-only` restricts the pool itself, so it drops both rows and
    # studies. `conditional-only` changes which likelihood branch the spoken
    # rows take, not which rows are loaded, so the frame is unchanged.
    assert seen["dse-native-only"][0] < seen["base"][0], seen
    assert seen["dse-native-only"][1] < seen["base"][1], seen
    assert seen["conditional-only"][0] == seen["base"][0], seen


def test_vg21_vague_anchors_moves_two_priors_and_nothing_structural():
    """Change both high-anchor priors while retaining VG21's structural settings."""
    import os
    import tempfile

    import vocab_growth.data_utils as vocab_data_utils

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    (variant,) = build_variant("vg21", "vague-anchors")
    assert (variant.p_slope_hi_u_alpha, variant.p_slope_hi_u_beta) == (1.2, 2.0)
    assert (variant.p_slope_hi_q_alpha, variant.p_slope_hi_q_beta) == (1.3, 1.3)
    # The window and its co-identified settings must be VG21's own, untouched.
    for field in ("max_age_months", "slope_anchors", "gp_domain_months",
                  "gp_anchor_age_months", "eta_q_sigma"):
        assert getattr(variant, field) == getattr(VG21, field), field

    with tempfile.TemporaryDirectory() as root:
        base_ctx, base = _prepared_bivariate(VG21, root)
        var_ctx, frame = _prepared_bivariate(variant, root)

    assert len(frame) == len(base)
    assert set(frame["study"].unique()) == set(base["study"].unique())
    assert len(var_ctx.model.free_RVs) == len(base_ctx.model.free_RVs)


def test_vg23_eta_flat_keeps_the_correlation_and_only_relaxes_its_prior():
    """Set LKJ eta to 1 while retaining the fitted correlation parameter."""
    import os
    import tempfile

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models.definitions import (
        VG23,
        BivariateCorrelatedSubjectREModelDefinition,
    )

    (variant,) = build_variant("vg23", "eta-flat")
    assert isinstance(variant, BivariateCorrelatedSubjectREModelDefinition)
    assert variant.subject_re_correlation_eta == 1.0
    assert VG23.subject_re_correlation_eta == 2.0     # base untouched

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    with tempfile.TemporaryDirectory() as root:
        base_ctx, base = _prepared_bivariate(VG23, root)
        var_ctx, frame = _prepared_bivariate(variant, root)

    names = {v.name for v in var_ctx.model.free_RVs}
    assert "rho_uq_raw" in names
    assert len(frame) == len(base)
    assert len(var_ctx.model.free_RVs) == len(base_ctx.model.free_RVs)


def test_vg16_lag_scope_variants_change_what_each_claims():
    """The three field-backed variants (#242), measured on the real frame.

    Each is registered on the promise of moving one specific thing; this checks
    it does, and that it leaves the others alone.
    """
    import os
    import tempfile

    import numpy as np

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models import cross_lag
    from vocab_growth.models.definitions import VG16

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def measure(definition, root):
        _, frame = _prepared_bivariate(definition, root)
        # The frame-level entry point rather than a hand-rebuilt copy of it: it
        # reads the same three columns and the same two settings off the same
        # definition, so this measures what a fit would build.
        _, has_lag, logits = cross_lag.prev_wave_lag_for_frame(
            frame, definition.n_trials, definition
        )
        lagged = has_lag > 0
        return {
            "rows": len(frame),
            "lagged": int(lagged.sum()),
            "studies": len(set(frame["study"])),
            "min_logit": float(np.min(logits[lagged])),
        }

    with tempfile.TemporaryDirectory() as root:
        base = measure(VG16, root)
        got = {}
        for name in ("lag-gap-12", "no-us01", "lag-continuity"):
            (variant,) = build_variant("vg16", name)
            got[name] = measure(variant, root)

    # A gap ceiling drops lags, not rows or studies.
    assert got["lag-gap-12"]["lagged"] < base["lagged"]
    assert got["lag-gap-12"]["rows"] == base["rows"]
    assert got["lag-gap-12"]["studies"] == base["studies"]

    # Leave-one-study-out drops rows and exactly one study.
    assert got["no-us01"]["studies"] == base["studies"] - 1
    assert got["no-us01"]["rows"] < base["rows"]
    assert got["no-us01"]["lagged"] < base["lagged"]

    # The continuity correction moves the predictor's boundary and nothing else.
    assert got["lag-continuity"]["rows"] == base["rows"]
    assert got["lag-continuity"]["lagged"] == base["lagged"]
    assert got["lag-continuity"]["min_logit"] > base["min_logit"] + 1.5


def test_excluding_a_study_that_matches_nothing_is_an_error():
    """A leave-one-out check that removes nothing cannot fail, which is the
    failure mode the retired `us01-ceiling-excluded` variants had: a registered
    check silently reporting robustness about an exclusion it never made."""
    import dataclasses
    import os
    import tempfile

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models.definitions import VG16

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    bogus = dataclasses.replace(
        VG16, config_name="probe-bogus-study", exclude_studies=("zz_99",)
    )
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(ValueError, match="matched no rows"):
            _prepared_bivariate(bogus, root)


def test_vg16_corr_adds_exactly_the_correlation_and_keeps_the_lag():
    """#289 task 3.9's comparator: VG16 nested at rho_uq = 0, nothing else moved.

    Builds both real graphs (no sampling), so it needs the prepared DuckDB.
    """
    import os
    import tempfile

    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling
    import numpy as np

    import vocab_growth.data_utils as vocab_data_utils
    from vocab_growth.models import common_bivariate as cb
    from vocab_growth.models import common_bivariate_re as cbr
    from vocab_growth.models.common import ModelFitContext
    from vocab_growth.models.definitions import (
        VG16,
        BivariateCorrelatedSubjectREModelDefinition,
    )

    (variant,) = build_variant("vg16", "corr")
    assert isinstance(variant, BivariateCorrelatedSubjectREModelDefinition)
    assert variant.subject_re_correlation_eta == 2.0
    assert variant.use_cross_lag and variant.lag_baseline == VG16.lag_baseline
    assert variant.config_name == f"{VG16.config_name}-corr"

    if not os.path.exists(vocab_data_utils.VOCABULARY_DATA_PATH):
        pytest.skip("prepared vocabulary DuckDB not available")

    def build(definition, root):
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
        cbr.prepare_bivariate_re_data(ctx, definition)
        cb.configure_bivariate_priors(ctx, definition)
        cbr.build_model_re(ctx, definition)
        return ctx.model

    with tempfile.TemporaryDirectory() as root:
        base = build(VG16, root)
        corr = build(variant, root)

    def names(model):
        return {v.name for v in model.free_RVs} | {v.name for v in model.deterministics}

    assert names(corr) - names(base) == {"rho_uq", "rho_uq_raw"}
    assert names(base) - names(corr) == set()
    assert "beta_lag" in names(corr)
    assert len(corr.free_RVs) == len(base.free_RVs) + 1
    assert np.isfinite(corr.point_logps()["y_s_obs"])
