# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The shared report blocks must describe the fit they are rendered against.

Every one of these tests exists because the hand-written block it replaces got
the answer wrong on a published report. The sampling banner told five models of
record they were not fitted at reporting quality when they were fitted at more
than the default reporting effort; the convergence callout told VG11 it had
cleared a gate it is published under an exception to; and the priors prose in
VG10 and VG15 described amplitudes that had been changed months earlier.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vocab_growth import glossary, report_cells
from vocab_growth.models.diagnostics_utils import render_convergence_caveats

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fit(tmp_path, *, definition=None, config="rep", parameters=("eta_u",), gate=None):
    """A minimal fitted-output directory: manifest, diagnostics, gate payload."""
    manifest = {
        "model": {"model_id": "VGXX", "definition": definition or {}},
        "sampling": {"configuration_name": config},
        "data": {"rows": 1431},
    }
    (tmp_path / "fit_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(index=list(parameters), data={"r_hat": [1.0] * len(parameters)}).to_csv(
        tmp_path / "diagnostics.csv"
    )
    if gate is not None:
        (tmp_path / "diagnostics_summary.json").write_text(json.dumps(gate))
    return tmp_path


# --------------------------------------------------------------------------
# Sampling banner
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config", ["rep", "rep-hightune", "rep-lite"])
def test_reporting_configs_are_not_called_approximate(tmp_path, capsys, config):
    """The defect this replaces: rep-hightune fits published a disclaimer.

    VG08, VG09, VG11, VG12 and VG13 were each fitted at 6 chains x 8,000-10,000
    draws -- more sampling effort than the 6 x 6,000 the old lookup table called
    "reporting" -- fell through it, and printed "It was not fitted in reporting
    mode" on a reporting-quality fit.
    """
    report_cells.render_sampling_banner(str(_fit(tmp_path, config=config)))
    out = capsys.readouterr().out
    assert "not** fitted at reporting quality" not in out
    assert config in out or report_cells.CONFIG_LABELS[config] in out


@pytest.mark.parametrize("config", ["dev", "test"])
def test_non_reporting_configs_are_flagged(tmp_path, capsys, config):
    report_cells.render_sampling_banner(str(_fit(tmp_path, config=config)))
    assert "not** fitted at reporting quality" in capsys.readouterr().out


def test_unrecorded_config_is_flagged_rather_than_assumed(tmp_path, capsys):
    (tmp_path / "fit_manifest.json").write_text(json.dumps({"sampling": {}}))
    report_cells.render_sampling_banner(str(tmp_path))
    assert "not** fitted at reporting quality" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Priors table
# --------------------------------------------------------------------------


def test_priors_table_reads_values_from_the_manifest(tmp_path, capsys):
    """A prior's value must come from the fit, never from prose in the template."""
    fit = _fit(
        tmp_path,
        definition={"eta_u_sigma": 0.8, "n_trials": 810},
        parameters=("eta_u",),
    )
    report_cells.render_priors_table(str(fit))
    assert "HalfNormal(0.8)" in capsys.readouterr().out


def test_priors_table_omits_parameters_the_fit_did_not_sample(tmp_path, capsys):
    """The gating that stops a report describing machinery its model lacks.

    The definition dataclass carries defaults for every field, so VG05's
    manifest records `tau_u_sigma` although VG05 has no study effects, and every
    bivariate manifest records `beta_lag_sigma` although only VG16 fits one.
    """
    fit = _fit(
        tmp_path,
        definition={"eta_u_sigma": 0.6, "tau_u_sigma": 0.5, "beta_lag_sigma": 0.5},
        parameters=("eta_u",),  # no tau_u, no beta_lag
    )
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "GP amplitude, understood" in out
    assert "Between-study SD" not in out
    assert "Cross-lag" not in out


def test_priors_table_carries_the_correlation_prior(tmp_path, capsys):
    """VG20's whole reason for existing had no row in its own priors table (#233).

    `subject_re_correlation_eta` is a top-level scalar rather than part of a
    subject-scale block, so neither the scalar HalfNormal path nor
    `_subject_scale_row` reached it and the table simply omitted it.
    """
    fit = _fit(
        tmp_path,
        definition={"subject_re_correlation_eta": 2.0},
        parameters=("rho_uq",),
    )
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "LKJ(2)" in out
    assert "Beta(2, 2)" in out
    assert "a correlation has to be evidenced" in out


def test_priors_table_omits_the_correlation_for_an_uncorrelated_model(tmp_path, capsys):
    """VG10 records the field's default in its manifest but never samples it."""
    fit = _fit(tmp_path, definition={"eta_u_sigma": 0.6}, parameters=("eta_u",))
    report_cells.render_priors_table(str(fit))
    assert "LKJ" not in capsys.readouterr().out


# --- VG22's factor block (issue #273) -------------------------------------------

#: VG22's real sampled parameter set at rank 3, as the graph builds it since
#: issue #266 finding 5: the shared bivariate-RE block, the four factor scales,
#: the designed `rho_uq_raw`, the six loading entries that remain, and the
#: per-child scores. `subject_factor_w_00`, `_w_20` and `_w_21` are gone --
#: the first anchor row is the constant e_0 and the second is set by `rho_uq`.
_VG22_PARAMETERS = (
    "p_slope_low_u", "p_slope_hi_u", "p_slope_low_q", "p_slope_hi_q",
    "eta_u", "eta_q", "tau_u", "tau_q",
    "tau_subj_u_0", "tau_subj_u_1", "tau_subj_q_0", "tau_subj_q_1",
    "rho_uq_raw",
    "subject_factor_w_10", "subject_factor_w_11", "subject_factor_w_12",
    "subject_factor_w_30", "subject_factor_w_31", "subject_factor_w_32",
    "subject_factor_z",
    # Deterministics the diagnostics table carries alongside them.
    "rho_uq", "tau_subj_u", "tau_subj_q", "subject_factor_corr",
    "subject_factor_loadings", "delta_subj_u", "b0_tau_subj_u",
)

_VG22_DEFINITION = {
    "n_trials": 810,
    "slope_anchors": [24.0, 84.0],
    "p_slope_low_u_alpha": 1.5, "p_slope_low_u_beta": 8.0,
    "p_slope_hi_u_alpha": 3.0, "p_slope_hi_u_beta": 1.3,
    "p_slope_low_q_alpha": 2.0, "p_slope_low_q_beta": 12.0,
    "p_slope_hi_q_alpha": 4.0, "p_slope_hi_q_beta": 1.2,
    "tau_subj_u_sigma": 1.5,
    "tau_subj_q_sigma": 1.5,
    "eta_u_sigma": 0.6,
    "eta_q_sigma": 0.8,
    "tau_u_sigma": 0.5,
    "tau_q_sigma": 0.5,
    "subject_factor": {
        "rank": 3,
        "tau1_u_sigma": 0.5,
        "tau1_q_sigma": 0.5,
        "ref_age_months": 36.0,
        "rho_uq_eta": 2.0,
    },
}


def test_priors_table_carries_the_factor_block(tmp_path, capsys):
    """VG22's page said the table "omits every prior this block adds" (#273).

    The four scales, the loading directions and the per-child factor scores had
    no entry in `_PRIOR_SPECS`, and nothing but that sentence recorded it -- so
    the rendered table described VG10's two child intercepts for a model that
    does not have them.
    """
    fit = _fit(tmp_path, definition=_VG22_DEFINITION, parameters=_VG22_PARAMETERS)
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out

    # The two level scales are the parent's own, at a stated reference age.
    assert "Between-child SD, understood — level" in out
    assert "at 36 months" in out
    # The two rate scales are per year, not per month, and are VG19's 0.5.
    assert "Between-child SD, understood — rate" in out
    assert "logits per year of age" in out
    assert out.count("HalfNormal(0.5)") >= 2
    # The loading family that remains after the two leading anchor rows.
    assert "Loading directions (6 sampled entries)" in out
    # The factor scores, and the rank.
    assert "Per-child factor scores $z$ (rank 3)" in out
    assert "Normal(0, I)" in out
    # rho_uq is designed now, under the prior VG20 places on the same quantity.
    assert "LKJ(2)" in out
    assert "The same prior VG20 places on the same quantity" in out
    # And the five that remain induced are named as such rather than left silent.
    assert "The other five implied correlations" in out
    assert "not separately chosen" in out
    # The arcsine is gone: it was the defect, not a property of the model.
    assert "arcsine" not in out


def test_the_factor_block_replaces_the_parent_child_intercept_rows(tmp_path, capsys):
    """`tau_subj_u` is a deterministic here, equal to the level scale.

    Rendering VG10's generic "Between-child SD, understood" row beside the four
    explicit ones would state the same prior twice under a name that no longer
    means what it means in VG10.
    """
    fit = _fit(tmp_path, definition=_VG22_DEFINITION, parameters=_VG22_PARAMETERS)
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "| Between-child SD, understood |" not in out
    assert "| Between-child SD, production ratio $q$ |" not in out


def test_a_model_without_a_factor_block_is_unchanged(tmp_path, capsys):
    """VG10's own table must not gain factor rows."""
    fit = _fit(
        tmp_path,
        definition={"tau_subj_u_sigma": 1.5, "tau_subj_q_sigma": 1.5},
        parameters=("tau_subj_u", "tau_subj_q"),
    )
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "| Between-child SD, understood |" in out
    assert "Loading directions" not in out
    assert "factor scores" not in out


def test_prior_coverage_reports_no_gap_for_the_factor_model(tmp_path):
    """The graph-to-report contract: every sampled family is rendered or exempt."""
    fit = _fit(tmp_path, definition=_VG22_DEFINITION, parameters=_VG22_PARAMETERS)
    coverage = report_cells.prior_coverage(str(fit))
    assert coverage["uncovered"] == []
    assert "subject_factor_z" in coverage["rendered"]
    assert "subject_factor_w_31" in coverage["rendered"]
    # The designed correlation is rendered under its own row, not left
    # to the induced-geometry caveat.
    assert "rho_uq_raw" in coverage["rendered"]
    assert "rho_uq" in coverage["rendered"]
    # Deterministics have no prior of their own and are exempt with a reason.
    assert "subject_factor_corr" in coverage["exempt"]
    assert report_cells._is_exempt("subject_factor_corr")


def test_the_table_discloses_its_own_gap(tmp_path, capsys):
    """A gap must announce itself on the page, not in a hand-written sentence.

    VG22's template carried the disclosure by hand, which is the same failure
    the whole module exists to replace: a fact about the fit, copied into prose,
    that nothing keeps true.
    """
    definition = dict(_VG22_DEFINITION)
    definition.pop("subject_factor")
    fit = _fit(tmp_path, definition=definition, parameters=_VG22_PARAMETERS)
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "Incomplete priors table" in out
    assert "`subject_factor_w_10`" in out
    assert "the gap is in the table, not in the fit" in out


def test_a_complete_table_says_nothing_about_gaps(tmp_path, capsys):
    fit = _fit(tmp_path, definition=_VG22_DEFINITION, parameters=_VG22_PARAMETERS)
    report_cells.render_priors_table(str(fit))
    assert "Incomplete priors table" not in capsys.readouterr().out


def test_prior_coverage_names_an_unrendered_family(tmp_path):
    """The check must fail loudly on the shape of the defect it exists for."""
    definition = dict(_VG22_DEFINITION)
    definition.pop("subject_factor")
    fit = _fit(tmp_path, definition=definition, parameters=_VG22_PARAMETERS)
    coverage = report_cells.prior_coverage(str(fit))
    assert "subject_factor_w_10" in coverage["uncovered"]
    assert "tau_subj_u_1" in coverage["uncovered"]


def test_priors_table_carries_the_dirichlet_multinomial_concentration(tmp_path, capsys):
    """VG15's concentration had a prior figure, a sensitivity mention, and no row.

    Found by the coverage check written for VG22's factor block: the same class
    of omission, in a different model, that nothing had noticed (#273).
    """
    fit = _fit(
        tmp_path,
        definition={"log_conc_mu": 3.0, "log_conc_sigma": 1.0},
        parameters=("log_conc",),
    )
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "Dirichlet-Multinomial concentration" in out
    assert "Normal(3, 1)" in out
    # Reported on the plain scale too: log 3 is not something a reader pictures.
    assert "concentration median 20" in out


def test_a_scale_that_is_inert_in_one_model_is_still_required_in_another(tmp_path):
    """`tau_subject` and `rho_uq` are sampled in some models, derived in others.

    An unconditional exemption for either would let the row that VG10 and VG20
    do need disappear unnoticed, which is how VG22's block was lost. The
    exemption must therefore come from the definition, not from the name.
    """
    assert report_cells._is_exempt("tau_subject") is None
    assert report_cells._is_exempt("rho_uq") is None
    assert report_cells._is_exempt("tau_subj_u") is None

    # Under a variance partition `tau_subject` is a deterministic function of the
    # budget, so it is covered without a HalfNormal row of its own.
    partitioned = _fit(
        tmp_path,
        definition={
            "subject_variance_partition": {
                "total_mu": 0.0,
                "total_sigma": 0.7,
                "share_alpha": 2.0,
                "share_beta": 2.0,
            }
        },
        parameters=("tau_subject", "v_total", "subject_variance_share"),
    )
    coverage = report_cells.prior_coverage(str(partitioned))
    assert coverage["uncovered"] == []
    assert "tau_subject" in coverage["rendered"]


def test_non_centred_offsets_are_exempt_with_a_reason(tmp_path):
    """`*_raw` and `*_z` carry no prior a reader would look for."""
    for name in ("rho_uq_raw", "delta_u_raw", "subject_factor_z"):
        assert report_cells._is_exempt(name), name
    assert report_cells._is_exempt("eta_u") is None


def test_signed_anchors_use_the_signed_anchor_ages(tmp_path, capsys):
    """Signing has its own anchor ages; labelling them with slope_anchors lies."""
    fit = _fit(
        tmp_path,
        definition={
            "p_slope_low_sign_alpha": 2.0,
            "p_slope_low_sign_beta": 20.0,
            "slope_anchors": [24, 84],
            "sign_anchor_ages": [15.0, 36.0, 96.0],
        },
        parameters=("p_slope_low_sign",),
    )
    report_cells.render_priors_table(str(fit))
    assert "(15 months)" in capsys.readouterr().out


def test_signed_peak_is_reported_as_estimated(tmp_path, capsys):
    """VG15's report claimed the signed peak was fixed by construction; it is not."""
    fit = _fit(
        tmp_path,
        definition={"sign_peak_prior": [2.0, 4.0], "sign_anchor_ages": [15.0, 36.0, 96.0]},
        parameters=("peak_unit_sign",),
    )
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "estimated, not fixed" in out
    assert "prior median 40 months" in out


def test_priors_table_says_so_when_there_is_no_manifest(tmp_path, capsys):
    report_cells.render_priors_table(str(tmp_path))
    assert "No fit manifest" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Convergence caveats
# --------------------------------------------------------------------------


def test_accepted_rhat_exception_is_disclosed(tmp_path, capsys):
    """VG11 is published under a recorded exception and said it had cleared the gate.

    The old per-template block read only `divergences` and `bfmi`, then printed
    a hard-coded sentence claiming the hard tier had been cleared.
    """
    gate = {
        "checks": {"rhat": False, "ess": True, "divergences": True, "bfmi": True},
        "max_rhat": 1.0125,
        "accepted_rhat_exception": {
            "parameters": ["g_unit_hsgp_coeffs[4]"],
            "observed_max_rhat": 1.0125,
            "reason": "One HSGP basis coefficient of sixteen.",
            "decided": "2026-08-15, study owner",
        },
    }
    fit = _fit(tmp_path, gate=gate)
    render_convergence_caveats(str(fit))
    out = capsys.readouterr().out
    assert "did not clear" in out
    assert "g_unit_hsgp_coeffs[4]" in out
    assert "cleared the hard convergence tier" not in out


def test_soft_tier_only_keeps_the_cleared_framing(tmp_path, capsys):
    gate = {
        "checks": {"rhat": True, "ess": True, "divergences": False, "bfmi": True},
        "divergences": 16,
    }
    render_convergence_caveats(str(_fit(tmp_path, gate=gate)))
    out = capsys.readouterr().out
    assert "cleared the hard convergence tier" in out
    assert "16 divergent" in out


def test_clean_fit_prints_nothing(tmp_path, capsys):
    gate = {"checks": {"rhat": True, "ess": True, "divergences": True, "bfmi": True}}
    render_convergence_caveats(str(_fit(tmp_path, gate=gate)))
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Glossary
# --------------------------------------------------------------------------


def test_unknown_glossary_term_raises(capsys):
    """A typo in a template must fail the render, not drop the definition."""
    with pytest.raises(KeyError):
        glossary.render_glossary(["Logit", "Not A Real Term"])


def test_glossary_renders_in_definition_order(capsys):
    glossary.render_glossary(["Concentration ($\\kappa$)", "Logit"])
    out = capsys.readouterr().out
    assert out.index("Logit") < out.index("Concentration")


def test_kappa_definition_states_the_counter_intuitive_direction():
    """Ten reviews each flagged that kappa is used without saying which way it runs."""
    assert "less" in glossary.GLOSSARY["Concentration ($\\kappa$)"]


def test_default_interval_convention_is_stated_somewhere():
    assert "89%" in glossary.GLOSSARY["Equal-tailed interval (ETI)"]


# --------------------------------------------------------------------------
# Model at a glance
# --------------------------------------------------------------------------


def test_glance_reports_no_hierarchy_when_the_fit_has_none(tmp_path, capsys):
    """VG05 announced study random intercepts it does not have.

    The definition dataclass carries a non-None default for every scale, so a
    definition-field test says "study random intercepts" for a model that never
    instantiates one -- contradicting the report's own prose two sections later.
    """
    fit = _fit(
        tmp_path,
        definition={"tau_u_sigma": 0.5, "tau_subj_u_sigma": 1.5},
        parameters=("eta_u",),  # no tau_u, no tau_subj_u
    )
    report_cells.render_model_at_a_glance(str(fit))
    assert "none — study and repeated-child effects are not modelled here" in capsys.readouterr().out


def test_glance_finds_hierarchy_under_the_univariate_parameter_names(tmp_path, capsys):
    """The error in the other direction: VG11 and VG12 have both levels.

    The univariate engine names them `tau` and `tau_subject`, not `tau_u` and
    `tau_subj_u`, so a test written against the joint names reported "none" for
    two models of record whose whole purpose is the study effect.
    """
    fit = _fit(
        tmp_path,
        definition={"tau_study_sigma": 0.5, "tau_subject_sigma": 1.5},
        parameters=("tau", "tau_subject"),
    )
    report_cells.render_model_at_a_glance(str(fit))
    out = capsys.readouterr().out
    assert "study" in out
    assert "child" in out
    assert "not modelled here" not in out


def test_glance_does_not_claim_spoken_ages_for_a_comprehension_model(tmp_path, capsys):
    """VG02 and VG04 both reported the query grid's maximum as a spoken cap."""
    manifest = {
        "model": {"definition": {"ages_query": [12, 84], "report_max_age_understood": 84}},
        "sampling": {"configuration_name": "rep"},
        "data": {"observed_outcome_counts": {"understood": 987}},
    }
    (tmp_path / "fit_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(index=["eta"], data={"r_hat": [1.0]}).to_csv(tmp_path / "diagnostics.csv")

    report_cells.render_model_at_a_glance(str(tmp_path))
    out = capsys.readouterr().out
    assert "understood and ratios to 84 months" in out
    assert "spoken" not in out


def test_glance_reports_both_caps_for_a_joint_model(tmp_path, capsys):
    manifest = {
        "model": {"definition": {"ages_query": [12, 90], "report_max_age_understood": 84}},
        "sampling": {"configuration_name": "rep"},
        "data": {"observed_outcome_counts": {"understood": 987, "spoken": 1428}},
    }
    (tmp_path / "fit_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame(index=["eta_u"], data={"r_hat": [1.0]}).to_csv(tmp_path / "diagnostics.csv")

    report_cells.render_model_at_a_glance(str(tmp_path))
    out = capsys.readouterr().out
    assert "understood and ratios to 84 months" in out
    assert "spoken to 90 months" in out


def test_variance_partition_replaces_the_inert_child_scale_prior(tmp_path, capsys):
    """VG11 and VG12 reparameterise the child scale into a shared budget.

    `tau_subject` is then a deterministic function of `v_total` and
    `subject_variance_share`, so its HalfNormal prior never enters the model.
    Reporting that HalfNormal described a prior with no effect on the fit.
    """
    fit = _fit(
        tmp_path,
        definition={
            "tau_subject_sigma": 1.5,
            "subject_variance_partition": {
                "total_mu": 0.0,
                "total_sigma": 0.8,
                "share_alpha": 3.9,
                "share_beta": 2.1,
                "reference_proportion": 0.0118,
            },
        },
        parameters=("tau_subject", "v_total", "subject_variance_share"),
    )
    report_cells.render_priors_table(str(fit))
    out = capsys.readouterr().out
    assert "HalfNormal(1.5)" not in out
    assert "LogNormal(0, 0.8)" in out
    assert "Beta(3.9, 2.1)" in out


def test_plain_child_scale_prior_survives_without_a_partition(tmp_path, capsys):
    fit = _fit(
        tmp_path,
        definition={"tau_subject_sigma": 1.5},
        parameters=("tau_subject",),
    )
    report_cells.render_priors_table(str(fit))
    assert "HalfNormal(1.5)" in capsys.readouterr().out


# --------------------------------------------------------------------------
# Leave-one-out
# --------------------------------------------------------------------------


def _loo_fit(tmp_path, rows, parameters=("eta",)):
    pd.DataFrame(rows).to_csv(tmp_path / "loo_summary.csv", index=False)
    pd.DataFrame(index=list(parameters), data={"r_hat": [1.0] * len(parameters)}).to_csv(
        tmp_path / "diagnostics.csv"
    )
    return tmp_path


_CLEAN_ROW = {
    "outcome": "all",
    "elpd_loo": -6985.6,
    "se": 76.8,
    "p_loo": 120.4,
    "n_data_points": 1521,
    "n_samples": 36000,
    "n_dropped_degenerate": 0,
    "pareto_k_good": 1521,
    "pareto_k_bad": 0,
    "pareto_k_very_bad": 0,
    "good_k_threshold": 0.7,
    "scale": "log",
}


def test_loo_section_says_so_when_the_table_is_absent(tmp_path, capsys):
    """The section must never render silently empty."""
    report_cells.render_loo_section(str(tmp_path))
    assert "No leave-one-out summary" in capsys.readouterr().out


def test_loo_section_reports_a_clean_fit_as_reliable(tmp_path, capsys):
    report_cells.render_loo_section(str(_loo_fit(tmp_path, [_CLEAN_ROW])))
    out = capsys.readouterr().out
    assert "every one of the 1,521 observations is within the threshold" in out
    assert "-6985.6" in out or "-6985.60" in out


def test_loo_section_flags_unreliable_importance_sampling(tmp_path, capsys):
    row = {**_CLEAN_ROW, "pareto_k_good": 1233, "pareto_k_bad": 258, "pareto_k_very_bad": 30}
    report_cells.render_loo_section(str(_loo_fit(tmp_path, [row])))
    out = capsys.readouterr().out
    assert "288 of 1,521 observations (19%) have non-finite Pareto $k$ or exceed the threshold" in out
    assert "30 finite values above 1" in out


def test_hierarchical_fits_get_the_wrong_unit_explanation(tmp_path, capsys):
    """A high k share under subject effects is expected, not evidence of misfit.

    VG10 has 20% of spoken and 31% of understood observations over the
    threshold. Reporting that as unreliability without saying why would send a
    reader looking for a modelling fault that is not there.
    """
    row = {**_CLEAN_ROW, "pareto_k_good": 1233, "pareto_k_bad": 258, "pareto_k_very_bad": 30}
    fit = _loo_fit(tmp_path, [row], parameters=("eta_u", "tau_subj_u"))
    report_cells.render_loo_section(str(fit))
    out = capsys.readouterr().out
    assert "wrong unit of prediction" in out
    assert "kfold_loso.py" in out


def test_non_hierarchical_fits_do_not_get_that_explanation(tmp_path, capsys):
    row = {**_CLEAN_ROW, "pareto_k_good": 1233, "pareto_k_bad": 258, "pareto_k_very_bad": 30}
    fit = _loo_fit(tmp_path, [row], parameters=("eta",))
    report_cells.render_loo_section(str(fit))
    assert "wrong unit of prediction" not in capsys.readouterr().out


def test_dropped_degenerate_observations_are_disclosed(tmp_path, capsys):
    row = {**_CLEAN_ROW, "n_dropped_degenerate": 12}
    report_cells.render_loo_section(str(_loo_fit(tmp_path, [row])))
    assert "12 observation(s) were excluded as degenerate" in capsys.readouterr().out


# A three-outcome table, as every joint/trivariate engine writes: one row per
# likelihood term, not one row per administration.
_MULTI_OUTCOME_ROWS = [
    {**_CLEAN_ROW, "outcome": "words understood"},
    {**_CLEAN_ROW, "outcome": "words spoken"},
    {**_CLEAN_ROW, "outcome": "words signed"},
]


def test_single_outcome_loo_keeps_the_administration_label(tmp_path, capsys):
    """A univariate fit has one likelihood over administration rows, so the
    administration wording is correct there and must survive the branch."""
    report_cells.render_loo_section(str(_loo_fit(tmp_path, [_CLEAN_ROW])))
    out = capsys.readouterr().out
    assert "so this is leave-one-administration-out" in out
    assert "leave-one-likelihood-term-out" not in out


def test_multi_outcome_loo_is_not_labelled_leave_one_administration_out(
    tmp_path, capsys
):
    """The label was wrong for every multi-outcome model (issue #266, finding 4).

    A per-outcome row holds out one likelihood term. The spoken likelihood's
    trial count is the same administration's observed understood count, so
    neither direction is a clean held-out administration: the spoken score is
    conditional on that observed comprehension, and the understood score leaves
    its own observed value in the spoken denominator.
    """
    fit = _loo_fit(tmp_path, _MULTI_OUTCOME_ROWS, parameters=("eta_u", "eta_s"))
    report_cells.render_loo_section(str(fit))
    out = capsys.readouterr().out
    assert (
        "each per-outcome row is leave-one-likelihood-term-out for that "
        "outcome, not leave-one-administration-out" in out
    )
    assert "conditional on the same administration's observed comprehension" in out
    assert "not independent held-out units" in out
    # Still usable for model comparison -- the finding is about the label, not
    # about the number, and a reader must not be sent away from a valid check.
    assert "Comparing models on these per-outcome numbers is still sound" in out
    assert "must not be read as whole-administration predictive accuracy" in out
    # This fixture carries no administration row, so the section must say so
    # rather than leave the conditional rows reading as the whole story.
    assert "no administration-level row" in out


def test_an_administration_row_is_named_as_the_one_to_read(tmp_path, capsys):
    """The score the reports always described, once a fit actually carries it.

    The per-outcome caveats stay -- they are still true of those rows -- but the
    reader is told which row is whole-administration predictive accuracy rather
    than left with three conditional ones and a warning.
    """
    from vocab_growth.administration_loo import ADMINISTRATION_LABEL

    rows = _MULTI_OUTCOME_ROWS + [{**_CLEAN_ROW, "outcome": ADMINISTRATION_LABEL}]
    fit = _loo_fit(tmp_path, rows, parameters=("eta_u", "eta_s"))
    report_cells.render_loo_section(str(fit))
    out = capsys.readouterr().out
    assert f"The **{ADMINISTRATION_LABEL}** row is the one that can be" in out
    assert "one importance weight rather than two" in out
    # And it must not claim generalisation it does not have.
    assert "not generalisation to a new child" in out
    assert "no administration-level row" not in out


def test_a_joint_fit_with_an_administration_row_scores_psi(tmp_path, capsys):
    """The composition terms identify psi and the per-outcome rows omit them.

    Summed into the administration case, they are scored -- which makes that row
    the only one that touches this model's headline association at all.
    """
    from vocab_growth.administration_loo import ADMINISTRATION_LABEL

    rows = _MULTI_OUTCOME_ROWS + [{**_CLEAN_ROW, "outcome": ADMINISTRATION_LABEL}]
    fit = _loo_fit(tmp_path, rows, parameters=("psi", "conc", "eta_u"))
    report_cells.render_loo_section(str(fit))
    out = capsys.readouterr().out
    assert "They **are** included in the administration row" in out
    assert "not scored by leave-one-out at all" not in out


def test_joint_composition_fits_say_psi_is_unscored(tmp_path, capsys):
    """VG15 excludes both Dirichlet-Multinomial terms from every LOO row, and
    those are the only terms that identify psi."""
    fit = _loo_fit(tmp_path, _MULTI_OUTCOME_ROWS, parameters=("psi", "conc", "eta_u"))
    report_cells.render_loo_section(str(fit))
    out = capsys.readouterr().out
    assert "$\\psi$ is not scored by leave-one-out at all" in out
    assert "composition factor in the conditioning set" in out


def test_multi_outcome_fits_without_composition_terms_omit_the_psi_note(
    tmp_path, capsys
):
    """VG14 and the bivariate engines have no composition likelihood, so the
    exclusion note would be false there."""
    fit = _loo_fit(tmp_path, _MULTI_OUTCOME_ROWS, parameters=("eta_u", "eta_s"))
    report_cells.render_loo_section(str(fit))
    assert "psi" not in capsys.readouterr().out


def test_priors_table_survives_an_overloaded_subject_scale_field():
    """A subject-scale field holding a block, not a float, must not break render.

    The subject-scale fields are overloaded: VG19 puts a child intercept-and-rate
    block there and Proposal A1 an age-varying scale. Once through `asdict` both
    are mappings, and the scalar path fed one straight to `scipy.stats.halfnorm
    .ppf(scale=...)`, which raises a bare `TypeError: '>' not supported between
    instances of 'dict' and 'int'` from inside scipy. Nothing surfaced until
    `quarto render` failed on the whole page.
    """
    import dataclasses as dc

    from vocab_growth.models.definitions import VG19, VG20

    row = report_cells._prior_row(
        "tau_subj_u", "Between-child scale", "tau_subj_u_sigma", "odds", dc.asdict(VG19)
    )
    assert row is not None, "the slope block must produce a row, not be dropped"
    label, distribution, reading = row
    # Both scales must be stated: a row naming only tau0 would read as though the
    # model had no rate at all.
    assert "HalfNormal(1.5)" in distribution and "HalfNormal(0.5)" in distribution
    assert "LKJ(2)" in distribution
    assert "per year" in reading

    # The scalar path is untouched: a model of record renders exactly as before.
    scalar = report_cells._prior_row(
        "tau_subj_u", "Between-child scale", "tau_subj_u_sigma", "odds", dc.asdict(VG20)
    )
    assert scalar == ("Between-child scale", "HalfNormal(1.5)", scalar[2])
    assert "odds" in scalar[2]


def test_priors_table_describes_an_age_varying_subject_scale():
    """Proposal A1's block reaches the same path and must also be described."""
    import dataclasses as dc

    from vocab_growth.models.definitions import AgeVaryingSubjectScale

    spec = dc.asdict(
        AgeVaryingSubjectScale(
            anchor_ages=(24.0, 72.0), young_sigma=1.5, log_ratio_sigma=0.4
        )
    )
    row = report_cells._prior_row(
        "tau_subj_u", "Between-child scale", "tau_subj_u_sigma", "odds",
        {"tau_subj_u_sigma": spec},
    )
    assert row is not None
    _, distribution, reading = row
    assert "HalfNormal(1.5)" in distribution
    assert "24 months" in reading



# --- render_headline_quantities: draw-wise summaries and boundary handling -----


def _write_learning_rate(directory, median_rate, peak_info=None):
    ages = np.linspace(12.0, 84.0, len(median_rate))
    pd.DataFrame(
        {
            "age_months": ages,
            "median_rate": median_rate,
            "ci_lo": np.asarray(median_rate) - 1.0,
            "ci_hi": np.asarray(median_rate) + 1.0,
        }
    ).to_csv(os.path.join(directory, "expected_learning_rate.csv"), index=False)
    if peak_info is not None:
        pd.DataFrame(peak_info).to_csv(
            os.path.join(directory, "expected_learning_rate_peak.csv"), index=False
        )


def test_headline_boundary_maximum_is_not_reported_as_fastest_growth(tmp_path, capsys):
    """A median-curve maximum on the grid edge locates no peak (#234)."""
    _write_learning_rate(str(tmp_path), [1.0, 2.0, 3.0, 4.0, 5.0])
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert "Fastest growth" not in out
    assert "boundary of the reported range" in out
    assert "not located" in out


def test_headline_interior_peak_reports_draw_wise_age_interval(tmp_path, capsys):
    _write_learning_rate(
        str(tmp_path),
        [1.0, 4.0, 9.0, 4.0, 1.0],
        peak_info={
            "peak_age_median_months": [47.0],
            "peak_age_ci_lo_months": [39.0],
            "peak_age_ci_hi_months": [55.0],
            "boundary_draw_share": [0.02],
        },
    )
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert "Fastest growth in words" in out
    assert "around 47 months" in out
    assert "peak age 39 – 55 months" in out
    # A negligible boundary share is not reported as a caveat.
    assert "range edge" not in out


def test_headline_interior_peak_without_peak_table_says_so(tmp_path, capsys):
    _write_learning_rate(str(tmp_path), [1.0, 4.0, 9.0, 4.0, 1.0])
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert "read off the median curve" in out
    assert "refit for peak-age uncertainty" in out


def _write_kappa(directory, trend=None):
    pd.DataFrame(
        {
            "age_months": [12.0, 48.0, 84.0],
            "vif_median": [12.0, 20.0, 40.0],
        }
    ).to_csv(os.path.join(directory, "posterior_kappa.csv"), index=False)
    if trend is not None:
        pd.DataFrame(trend).to_csv(
            os.path.join(directory, "posterior_kappa_trend.csv"), index=False
        )


_TREND = {
    "age_young_months": [12.0],
    "age_old_months": [84.0],
    "vif_young_median": [12.0],
    "vif_old_median": [40.0],
    "vif_ratio_median": [3.3],
    "vif_ratio_ci_lo": [2.1],
    "vif_ratio_ci_hi": [4.8],
    "p_widens": [0.99],
}


def test_headline_dispersion_uses_the_draw_wise_contrast(tmp_path, capsys):
    _write_kappa(str(tmp_path), trend=_TREND)
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    # kappa is marginal count dispersion, not a between-child quantity.
    assert "Spread between children" not in out
    assert "Spread across same-age administrations" in out
    assert "widens with age" in out
    assert "P(widens) = 0.99" in out
    assert "ratio ×3.30 (2.10 – 4.80)" in out


def test_headline_dispersion_with_unresolved_direction_says_so(tmp_path, capsys):
    trend = {**_TREND, "p_widens": [0.62]}
    _write_kappa(str(tmp_path), trend=trend)
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert "does not change clearly" in out
    assert "widens with age" not in out


def test_headline_dispersion_without_trend_table_is_hedged(tmp_path, capsys):
    _write_kappa(str(tmp_path))
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert "Spread between children" not in out
    assert "read off the median curve; refit for a draw-wise contrast" in out


def test_headline_dispersion_is_labelled_residual_with_child_effects(tmp_path, capsys):
    # With a child random effect in the fit, kappa is conditional on it — the
    # residual within-child spread — and must not be labelled as the spread
    # across (different children's) administrations, which is tau_subject's
    # job (#240).
    _write_kappa(str(tmp_path), trend=_TREND)
    pd.DataFrame({"mean": [0.9]}, index=["tau_subject"]).to_csv(
        os.path.join(str(tmp_path), "diagnostics.csv")
    )
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert "Within-child spread between same-age administrations" in out
    assert "Spread across same-age administrations" not in out


def test_headline_dispersion_labels_are_per_outcome(tmp_path, capsys):
    # VG08's shape: a child effect on understood only. kappa_u is conditional
    # on it (within-child); the nested spoken mean is p_u * q with no child
    # effect on q, so kappa_s keeps the marginal-style label.
    for suffix in ("u", "s"):
        pd.DataFrame(
            {"age_months": [12.0, 48.0, 84.0], "vif_median": [12.0, 20.0, 40.0]}
        ).to_csv(
            os.path.join(str(tmp_path), f"posterior_kappa_{suffix}.csv"), index=False
        )
    pd.DataFrame({"mean": [0.9]}, index=["tau_subj_u"]).to_csv(
        os.path.join(str(tmp_path), "diagnostics.csv")
    )
    report_cells.render_headline_quantities(str(tmp_path))
    out = capsys.readouterr().out
    assert (
        "Within-child spread between same-age administrations, words understood"
        in out
    )
    assert "Spread across same-age administrations, words spoken" in out


# --------------------------------------------------------------------------
# Variation table under a child intercept-and-rate block (#233)
# --------------------------------------------------------------------------


def _slope_fit(tmp_path, *, ref_age=36.0, with_rho=True):
    """A VG19-shaped fit: alias scales, the rate block, and a reporting cap.

    Values are VG19's own `rep` posterior means from
    notes/202608212000-vg19-gates-g2-g4-g5.md section 2.
    """
    index = ["tau_subj_u_0", "tau_subj_u_1", "tau_subj_q_0", "tau_subj_q_1",
             "tau_subj_u", "tau_subj_q"]
    mean = [0.751, 0.176, 1.207, 0.640, 0.751, 1.207]
    if with_rho:
        index += ["tau_subj_u_rho", "tau_subj_q_rho"]
        mean += [-0.219, 0.469]
    manifest = {
        "model": {
            "model_id": "VG19",
            "definition": {
                "subject_slope_ref_age_months": ref_age,
                "ages_query": [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90],
                "report_max_age_understood": 72,
            },
        },
        "sampling": {"configuration_name": "rep"},
    }
    (tmp_path / "fit_manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame({"mean": mean}, index=index).to_csv(tmp_path / "diagnostics.csv")
    return tmp_path


def test_variation_table_dates_the_alias_scales_under_a_rate(tmp_path, capsys):
    """`tau_subj_u` is tau0 at the reference age, not a spread that holds at every age.

    The alias exists so consumers written against VG10 keep working, and the
    cost is that a row labelled "Between children, understood" describes one age
    while implying all of them.
    """
    report_cells.render_variation_table(str(_slope_fit(tmp_path)))
    out = capsys.readouterr().out
    assert "Between children, understood (at 36 months)" in out
    assert "Between children, production ratio $q$ (at 36 months)" in out


def test_variation_table_reports_the_age_varying_child_scale(tmp_path, capsys):
    """The spread is a parabola in age; one number cannot state it.

    With rho01 = -0.219, tau0 = 0.751 and tau1 = 0.176 the comprehension scale
    has its minimum at D = -rho01 * tau0 / tau1 = +0.93 years, so it must fall
    from 12 months to the reference age and rise again by 72.
    """
    report_cells.render_variation_table(str(_slope_fit(tmp_path)))
    out = capsys.readouterr().out
    assert "| 12 mo | 30 mo | 36 mo | 54 mo | 72 mo |" in out
    assert "Plug-in, not a posterior summary" in out

    row = next(
        line for line in out.splitlines()
        if line.startswith("| Between children, understood |")
    )
    values = [float(cell) for cell in row.strip("| ").split(" | ")[1:]]
    assert values[0] > values[2], "the spread must be wider below the reference age"
    assert values[-1] > values[3], "and widen again above the parabola's minimum"
    assert values[2] == pytest.approx(0.751, abs=0.005), "tau0 at the reference age"


def test_variation_table_caps_the_age_columns_at_the_reporting_cap(tmp_path, capsys):
    """Both scales ride comprehension, so neither may outrun its reporting cap."""
    report_cells.render_variation_table(str(_slope_fit(tmp_path)))
    out = capsys.readouterr().out
    assert "84 mo" not in out
    assert "90 mo" not in out


def test_variation_table_declines_the_age_scale_without_a_named_correlation(
    tmp_path, capsys
):
    """VG22 carries the correlation inside a matrix; a guessed element is worse
    than no table."""
    report_cells.render_variation_table(str(_slope_fit(tmp_path, with_rho=False)))
    out = capsys.readouterr().out
    assert "Between children, understood (at 36 months)" in out
    assert "12 mo | 30 mo" not in out
    assert "not tabulated for this" in out


def test_variation_table_is_unchanged_for_a_constant_offset_model(tmp_path, capsys):
    """VG10 and every earlier model must render exactly as before."""
    (tmp_path / "fit_manifest.json").write_text(
        json.dumps({"model": {"definition": {"ages_query": [12, 24]}}})
    )
    pd.DataFrame(
        {"mean": [0.787, 1.286, 0.35]},
        index=["tau_subj_u", "tau_subj_q", "tau_u"],
    ).to_csv(tmp_path / "diagnostics.csv")
    report_cells.render_variation_table(str(tmp_path))
    out = capsys.readouterr().out
    assert "Between children, understood |" in out
    assert "(at " not in out
    assert "Plug-in" not in out


# --------------------------------------------------------------------------
# Reader-facing blocks (2026-09-02 template review)
# --------------------------------------------------------------------------


def _summary_frame(ages, *, population=True, predictive=True, prefix=""):
    """A posterior_summary table in the shape the engines write."""
    rows = []
    for i, age in enumerate(ages):
        row = {"age_months": age, f"Ey{prefix}_median": 10.0 * (i + 1)}
        row[f"Ey{prefix}_ci50_lo"] = 8.0 * (i + 1)
        row[f"Ey{prefix}_ci50_hi"] = 12.0 * (i + 1)
        row[f"Ey{prefix}_ci_lo"] = 5.0 * (i + 1)
        row[f"Ey{prefix}_ci_hi"] = 15.0 * (i + 1)
        if population:
            row["Ey_population_median"] = 11.0 * (i + 1)
        if predictive:
            row.update(
                {
                    "Y_ci50_lo": 4.0 * (i + 1),
                    "Y_ci50_hi": 20.0 * (i + 1),
                    "Y_ci_lo": 0.0,
                    "Y_ci_hi": 40.0 * (i + 1),
                    "P(Y=0)": 0.5 / (i + 1),
                    "P(Y<=10)": 0.7 / (i + 1),
                    "P(Y<=50)": 0.9,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def test_expectations_table_prefers_population_and_single_child_columns(tmp_path, capsys):
    fit = _fit(tmp_path, parameters=("eta_u", "tau_subj_u"))
    _summary_frame([12, 24, 36]).to_csv(fit / "posterior_summary_s.csv", index=False)
    pd.DataFrame({"age_months": [12, 24, 36], "n_obs": [30, 12, 0]}).to_csv(
        fit / "posterior_summary_monthly_s.csv", index=False
    )
    report_cells.render_expectations_table("s", directory=str(fit))
    out = capsys.readouterr().out
    assert "| 12 | 11 | 4–20 | 0–40 | 50% | 70% | 90% | 30 |" in out
    assert "Half of children" in out and "Nine in ten children" in out
    assert "**population-level** median" in out
    assert "**single-child** ranges" in out
    # 36 months has no nearby observations and is the last row: beyond the data.
    assert "| 36† |" in out
    assert "† Ages above 24 months" in out


def test_expectations_table_falls_back_when_the_engine_writes_no_predictive_columns(
    tmp_path, capsys
):
    """The joint modality engine writes `Ey_u_*` and no `Y_*`; say so rather than omit."""
    fit = _fit(tmp_path)
    _summary_frame([18, 30], population=False, predictive=False, prefix="_u").to_csv(
        fit / "posterior_summary_u.csv", index=False
    )
    report_cells.render_expectations_table("u", directory=str(fit))
    out = capsys.readouterr().out
    assert "| 18 | 10 | 8–12 | 5–15 |" in out
    assert "50% interval" in out and "89% interval" in out
    assert "no single-child predictive columns" in out
    assert "P(no words)" not in out


def test_expectations_table_says_so_when_the_summary_is_absent(tmp_path, capsys):
    report_cells.render_expectations_table(None, directory=str(_fit(tmp_path)))
    assert "cannot be shown" in capsys.readouterr().out


def test_diagnostic_verdict_names_the_extremes_and_the_effort(tmp_path, capsys):
    gate = {
        "passed": True,
        "checks": {"rhat": True, "ess": True, "divergences": True, "bfmi": True},
        "divergences": 0,
        "max_rhat": 1.00421,
        "min_ess": 1416.08,
        "bfmi_per_chain": [0.44, 0.47, 0.51],
        "thresholds": {"rhat_max": 1.01, "ess_threshold": 400, "bfmi_threshold": 0.3},
    }
    fit = _fit(tmp_path, gate=gate, parameters=("eta_u", "tau_subj_q"))
    pd.DataFrame(
        index=["eta_u", "tau_subj_q"],
        data={"r_hat": [1.00421, 1.001], "ess_bulk": [2017.0, 1416.08], "ess_tail": [3000.0, 2500.0]},
    ).to_csv(fit / "diagnostics.csv")
    manifest = json.loads((fit / "fit_manifest.json").read_text())
    manifest["sampling"]["parameters"] = {
        "chains": 6, "draws": 8000, "tune": 12000, "target_accept": 0.97
    }
    (fit / "fit_manifest.json").write_text(json.dumps(manifest))
    report_cells.render_diagnostic_verdict(str(fit))
    out = capsys.readouterr().out
    assert "| Largest R-hat | 1.004210 (`eta_u`) | ≤ 1.01 | hard | pass |" in out
    assert "| Smallest effective sample size | 1,416 (`tau_subj_q`) | ≥ 400 | hard | pass |" in out
    assert "| Smallest energy BFMI across chains | 0.440 | ≥ 0.3 | soft | pass |" in out
    assert "clears both tiers" in out
    assert "6 chains × 8,000 draws, after 12,000 tuning draws, at target acceptance 0.97" in out


def test_diagnostic_verdict_distinguishes_the_soft_tier(tmp_path, capsys):
    gate = {
        "passed": False,
        "checks": {"rhat": True, "ess": True, "divergences": False, "bfmi": False},
        "divergences": 4,
        "max_rhat": 1.0022,
        "min_ess": 2695.0,
        "bfmi_per_chain": [0.208, 0.31],
        "thresholds": {"rhat_max": 1.01, "ess_threshold": 400, "bfmi_threshold": 0.3},
    }
    report_cells.render_diagnostic_verdict(str(_fit(tmp_path, gate=gate)))
    out = capsys.readouterr().out
    assert "| Divergent transitions | 4 | 0 | soft | **fail** |" in out
    assert "| Smallest energy BFMI across chains | 0.208 | ≥ 0.3 | soft | **fail** |" in out
    assert "clears the **hard** tier" in out and "not the **soft** tier" in out


def test_diagnostic_verdict_says_so_when_the_summary_is_absent(tmp_path, capsys):
    report_cells.render_diagnostic_verdict(str(_fit(tmp_path)))
    assert "cannot be shown" in capsys.readouterr().out


def test_contraction_table_marks_prior_driven_and_pressing_parameters(tmp_path, capsys):
    fit = _fit(tmp_path)
    pd.DataFrame(
        [
            {"parameter": "eta_u", "posterior_mean": 0.9, "posterior_sd": 0.45,
             "prior_median": 0.4, "prior_sd": 0.36, "prior_cdf": 0.96, "contraction": -0.25},
            {"parameter": "p_slope_low_u", "posterior_mean": 0.14, "posterior_sd": 0.01,
             "prior_median": 0.134, "prior_sd": 0.11, "prior_cdf": 0.55, "contraction": 0.91},
            {"parameter": "ell_unit_sign", "posterior_mean": 0.5, "posterior_sd": 0.19,
             "prior_median": 0.5, "prior_sd": 0.19, "prior_cdf": 0.5, "contraction": 0.03},
        ]
    ).to_csv(fit / "prior_posterior_contraction.csv", index=False)
    report_cells.render_prior_posterior_contraction(str(fit))
    out = capsys.readouterr().out
    assert "| GP amplitude, understood | 0.4 | 0.36 | 0.9 | 0.45 | -0.25 | 0.96 | prior-driven |" in out
    assert "Understood proportion at the low age anchor" in out and "informed by the data" in out
    assert "Prior-driven here: `eta_u`, `ell_unit_sign`" in out
    # Sorted by contraction, so the least-informed parameter comes first.
    assert out.index("GP amplitude, understood") < out.index("GP length-scale, signing")


def test_contraction_table_says_how_to_produce_it_when_absent(tmp_path, capsys):
    report_cells.render_prior_posterior_contraction(str(_fit(tmp_path)))
    assert "prior_vs_posterior.py --table" in capsys.readouterr().out


def test_family_notes_read_population_and_hierarchy_from_the_fit(tmp_path, capsys):
    fit = _fit(
        tmp_path,
        definition={"population": "ds", "n_trials": 810},
        parameters=("eta_u", "tau_subj_u"),
    )
    report_cells.render_family_notes(str(fit))
    out = capsys.readouterr().out
    assert "810-word reference inventory" in out
    assert "single-child range" in out and "not unusual" in out
    assert "pooled research sample" in out and "not a norm" in out

    (tmp_path / "td").mkdir()
    fit = _fit(tmp_path / "td", definition={"population": "td", "n_trials": 810})
    report_cells.render_family_notes(str(fit))
    out = capsys.readouterr().out
    assert "spread of administrations" in out
    assert "*reference*" in out and "not a target" in out


def test_reading_routes_send_non_researchers_away_from_a_development_step(capsys):
    report_cells.render_reading_routes("development", instead="VG20")
    out = capsys.readouterr().out
    assert "**VG20**" in out
    assert "#sec-predictions" not in out


def test_reading_routes_offer_three_routes_on_a_model_of_record(capsys):
    report_cells.render_reading_routes(
        "record", joint=True, signing=True, robustness=True, recovery=True
    )
    out = capsys.readouterr().out
    for anchor in (
        "#sec-predictions", "#sec-one-child", "#sec-spoken-given-understood",
        "#what-signing-is-worth-to-the-child", "#sec-monthly", "#sec-priors", "#sec-diagnostics",
        "#sec-loo", "#sec-robustness", "#sec-limits",
    ):
        assert anchor in out, anchor
    assert "parameter recovery" in out


def test_reading_routes_reject_an_unknown_role():
    with pytest.raises(ValueError):
        report_cells.render_reading_routes("headline")


def test_frame_composition_prints_the_manifest_totals_without_a_rebuild(tmp_path, capsys, monkeypatch):
    fit = _fit(tmp_path)
    manifest = json.loads((fit / "fit_manifest.json").read_text())
    manifest["data"].update(
        {
            "rows": 1424,
            "children": 763,
            "observed_outcome_counts": {"spoken": 1421, "understood": 976},
            "source_row_counts": {"uk_01": 214, "es_01": 186, "it_01": 173},
            "analysis_frame_hash": "sha256:not-the-current-frame",
        }
    )
    (fit / "fit_manifest.json").write_text(json.dumps(manifest))
    report_cells.render_frame_composition(str(fit))
    out = capsys.readouterr().out
    assert "**1,424 administrations** from **763 children** (1.87 administrations per child" in out
    assert "spoken 1,421, understood 976" in out
    assert "| `uk_01` | 214 |" in out
    assert out.index("`uk_01`") < out.index("`es_01`") < out.index("`it_01`")
    assert "the totals above are exact" in out


def test_stage_report_sources_copies_the_template_and_every_shared_include(tmp_path):
    from vocab_growth.reporting import stage_report_sources

    docs = tmp_path / "docs"
    (docs / "models" / "vg99").mkdir(parents=True)
    (docs / "models" / "vg99" / "index.qmd").write_text("{{< include _shared.qmd >}}\n")
    (docs / "models" / "_shared.qmd").write_text("shared body\n")
    (docs / "models" / "_other.qmd").write_text("another include\n")
    out = tmp_path / "out"
    out.mkdir()
    staged = stage_report_sources("VG99", str(out), docs_dir=str(docs))
    assert [os.path.basename(p) for p in staged] == ["index.qmd", "_other.qmd", "_shared.qmd"]
    assert (out / "_shared.qmd").read_text() == "shared body\n"


def test_stage_report_sources_raises_when_the_template_is_missing(tmp_path):
    from vocab_growth.reporting import stage_report_sources

    (tmp_path / "docs" / "models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        stage_report_sources("VG99", str(tmp_path), docs_dir=str(tmp_path / "docs"))


def test_diagnostic_verdict_does_not_name_a_listed_parameter_for_an_unlisted_extreme(tmp_path, capsys):
    """The gate screens random-effect elements the table omits (VG20's min ESS)."""
    gate = {
        "passed": True,
        "checks": {"rhat": True, "ess": True, "divergences": True, "bfmi": True},
        "divergences": 0, "max_rhat": 1.0042, "min_ess": 1416.0, "bfmi_per_chain": [0.5],
        "thresholds": {"rhat_max": 1.01, "ess_threshold": 400, "bfmi_threshold": 0.3},
    }
    fit = _fit(tmp_path, gate=gate)
    pd.DataFrame(index=["eta_u"], data={"r_hat": [1.0042], "ess_bulk": [2017.0], "ess_tail": [3367.0]}).to_csv(
        fit / "diagnostics.csv"
    )
    report_cells.render_diagnostic_verdict(str(fit))
    out = capsys.readouterr().out
    assert "1,416 (an element the table does not list)" in out
    assert "1.004200 (`eta_u`)" in out


def test_emit_factor_correlation_summarises_the_trace_variable(tmp_path):
    """The 4x4 the factor model exists to estimate reaches a CSV the report can render."""
    import runpy

    import h5netcdf

    module = runpy.run_path(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "emit_factor_correlation.py")
    )
    rng = np.random.default_rng(0)
    values = np.tile(np.eye(4), (2, 5, 1, 1))
    values[..., 0, 2] = values[..., 2, 0] = 0.3 + rng.normal(0, 0.01, size=(2, 5))
    with h5netcdf.File(tmp_path / "trace.nc", "w") as handle:
        group = handle.create_group("posterior")
        group.dimensions = {"chain": 2, "draw": 5, "child_effect4": 4, "child_effect4_b": 4}
        var = group.create_variable(
            "subject_factor_corr", ("chain", "draw", "child_effect4", "child_effect4_b"), float
        )
        var[...] = values
        labels = group.create_variable("child_effect4", ("child_effect4",), dtype="S4")
        labels[...] = np.array([b"b0u", b"b1u", b"b0q", b"b1q"])
    path = module["summarise"](str(tmp_path))
    table = pd.read_csv(path)
    assert set(table["row"]) == {"b0u", "b1u", "b0q", "b1q"}
    cell = table[(table["row"] == "b0u") & (table["column"] == "b0q")].iloc[0]
    assert abs(cell["mean"] - 0.3) < 0.02
    diagonal = table[table["row"] == table["column"]]
    assert (diagonal["mean"] == 1.0).all()


def test_frame_composition_says_when_the_frame_carries_no_study_or_child_key(tmp_path, capsys, monkeypatch):
    """The plain bivariate engine's frame has age and outcomes only; that is a fact to state."""
    from vocab_growth import analysis_frames

    fit = _fit(tmp_path)
    manifest = json.loads((fit / "fit_manifest.json").read_text())
    manifest["model"]["model_id"] = "VG05"
    manifest["data"].update({"rows": 3, "analysis_frame_hash": "sha256:stub"})
    (fit / "fit_manifest.json").write_text(json.dumps(manifest))
    frame = pd.DataFrame({"age": [12.0, 30.0, 50.0], "understood": [5, 80, None], "spoken": [0, 20, 300]})
    monkeypatch.setattr(analysis_frames, "build_analysis_frame", lambda key, definition: (frame, {}))
    monkeypatch.setattr(analysis_frames, "analysis_frame_hash", lambda df: "sha256:stub")
    report_cells.render_frame_composition(str(fit))
    out = capsys.readouterr().out
    assert "**This frame carries no study or child key**" in out
    assert "| 0–24 | 1 | 1 |" in out
    assert "| 48–72 | 0 | 1 |" in out
    assert "seen more than once" not in out.split("**This frame")[0]


def test_frame_composition_tabulates_studies_when_only_the_child_key_is_absent(tmp_path, capsys, monkeypatch):
    """The trivariate frame keeps `study` for its signing masks but has no child key."""
    from vocab_growth import analysis_frames

    fit = _fit(tmp_path)
    manifest = json.loads((fit / "fit_manifest.json").read_text())
    manifest["model"]["model_id"] = "VG14"
    manifest["data"].update({"rows": 3, "analysis_frame_hash": "sha256:stub"})
    (fit / "fit_manifest.json").write_text(json.dumps(manifest))
    frame = pd.DataFrame(
        {"study": ["uk_02", "uk_02", "es_01"], "age": [20.0, 40.0, 30.0], "understood": [50, 200, 90], "spoken": [5, 60, 20]}
    )
    monkeypatch.setattr(analysis_frames, "build_analysis_frame", lambda key, definition: (frame, {}))
    monkeypatch.setattr(analysis_frames, "analysis_frame_hash", lambda df: "sha256:stub")
    report_cells.render_frame_composition(str(fit))
    out = capsys.readouterr().out
    assert "| `uk_02` | 2 | 20–40 |" in out
    assert "**This frame carries no child key**" in out
    assert "no study or child key" not in out


# --------------------------------------------------------------------------
# Dispersion scope
# --------------------------------------------------------------------------


def _kappa_fit(tmp_path, *, definition=None, contraction=None, curves=("u", "s")):
    """A fit directory carrying posterior kappa curves and a contraction table."""
    fit = _fit(tmp_path, definition=definition or {"n_trials": 810})
    for suffix in curves:
        name = "posterior_kappa" if suffix is None else f"posterior_kappa_{suffix}"
        pd.DataFrame(
            {"age_months": [8.0, 24.0], "kappa_median": [30.0, 60.0], "vif_median": [20.0, 10.0]}
        ).to_csv(fit / f"{name}.csv", index=False)
    if contraction is not None:
        pd.DataFrame(contraction).to_csv(fit / "prior_posterior_contraction.csv", index=False)
    return fit


def test_dispersion_scope_names_the_denominator_of_each_curve(tmp_path, capsys):
    """The defect: two kappa figures under near-identical headings, no scope stated.

    ``kappa_u`` disperses counts out of the item pool; ``kappa_s`` disperses the
    production ratio on the child's own understood count. Nothing on the page
    said so, which invited reading one level against the other.
    """
    report_cells.render_dispersion_scope(str(_kappa_fit(tmp_path)))
    out = capsys.readouterr().out
    assert "810-item reference inventory" in out
    assert "**conditional** ratio" in out
    assert "spoken among the words that child understands" in out
    assert "different denominators" in out and "These two curves" in out
    assert "never against another model's curve" in out


def test_dispersion_scope_counts_a_third_curve_in_words(tmp_path, capsys):
    report_cells.render_dispersion_scope(
        str(_kappa_fit(tmp_path, curves=("u", "s", "sign")))
    )
    out = capsys.readouterr().out
    assert "These three curves" in out
    assert "signed among the words that child understands" in out


def test_dispersion_scope_omits_the_comparison_for_a_single_outcome(tmp_path, capsys):
    report_cells.render_dispersion_scope(str(_kappa_fit(tmp_path, curves=(None,))))
    out = capsys.readouterr().out
    assert "810-item reference inventory" in out
    assert "different denominators" not in out


def test_dispersion_scope_maps_an_uninformed_kappa_to_its_anchor_age(tmp_path, capsys):
    """VG22's ``kappa_excess_young_s`` contracts to -0.23 on real data.

    The curve is still drawn there, so the page has to say which *end* of it the
    data never placed -- which means resolving the parameter to the reference age
    its prior is anchored at, not just naming the parameter.
    """
    fit = _kappa_fit(
        tmp_path,
        definition={"n_trials": 810, "kappa_s": {"anchor_ages": [18.0, 72.0]}},
        contraction=[
            {"parameter": "kappa_min_u", "posterior_mean": 8.6, "posterior_sd": 5.1,
             "prior_cdf": 0.55, "contraction": 0.50, "flags": ""},
            {"parameter": "kappa_excess_young_s", "posterior_mean": 49.6, "posterior_sd": 27.6,
             "prior_cdf": 0.94, "contraction": -0.2286, "flags": "uninformed"},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "the curve at and below 18 months" in out
    assert "not estimated from this data" in out and "-0.23" in out
    assert "callout-warning" in out
    # The informed parameter is not listed as a caveat.
    assert "kappa_min_u" not in out


def test_dispersion_scope_flags_a_prior_acting_as_a_floor(tmp_path, capsys):
    """The two-sided test. VG14's kappa parameters press the *lower* tail.

    ``prior_vs_posterior.py`` flagged only ``cdf >= 0.95`` until 2026-09-02, so a
    fit whose data wants a smaller dispersion than the prior offers carried an
    empty ``flags`` column. This block reads the numbers, so it is right against
    a table written either side of that fix.
    """
    fit = _kappa_fit(
        tmp_path,
        definition={"n_trials": 810, "kappa_u": {"anchor_ages": [18.0, 72.0]}},
        contraction=[
            {"parameter": "kappa_excess_young_u", "posterior_mean": 8.7, "posterior_sd": 1.15,
             "prior_cdf": 0.0114, "contraction": 0.9937, "flags": ""},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "pressing against its prior" in out
    assert "below what the prior comfortably allows" in out
    assert "the curve at and below 18 months" in out


def test_dispersion_scope_reports_every_curve_readable_when_nothing_is_flagged(
    tmp_path, capsys
):
    fit = _kappa_fit(
        tmp_path,
        contraction=[
            {"parameter": "kappa_min_u", "posterior_mean": 8.6, "posterior_sd": 5.1,
             "prior_cdf": 0.55, "contraction": 0.50, "flags": ""},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "both ends of each curve can be read" in out
    assert "callout-warning" not in out


def test_dispersion_scope_says_so_when_the_fit_writes_no_curve(tmp_path, capsys):
    report_cells.render_dispersion_scope(str(_fit(tmp_path)))
    assert "no posterior dispersion curve" in capsys.readouterr().out


def test_prior_vs_posterior_presses_on_both_tails():
    """A prior acting as a floor is the same finding as one acting as a ceiling.

    ``scripts/`` is not importable from the test run, and the assertion is about
    one condition, so this reads the source rather than adding path plumbing.
    """
    source = (REPO_ROOT / "scripts" / "prior_vs_posterior.py").read_text(encoding="utf-8")
    assert "CONFLICT_CDF = 0.95" in source
    assert "if cdf >= CONFLICT_CDF or cdf <= 1.0 - CONFLICT_CDF:" in source


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("kappa_min_u", ("min", "u")),
        ("kappa_min", ("min", None)),
        ("kappa_excess_young_s", ("excess_young", "s")),
        ("kappa_excess_old_sign", ("excess_old", "sign")),
        ("a_kappa_s", ("level", "s")),
        ("b_kappa_mag_sign", ("slope", "sign")),
        ("eta_u", (None, None)),
        ("ell_unit_q", (None, None)),
    ],
)
def test_kappa_role_parsing_covers_the_legacy_names(name, expected):
    """The defect: the first cut matched ``name.startswith("kappa")``.

    The legacy intercept-and-slope form names its parameters ``a_kappa_s`` and
    ``b_kappa_mag_s``, which do not begin with "kappa", so VG05, VG07 and VG08
    had every dispersion caveat silently dropped -- including ``b_kappa_mag_s``
    at 7.8 prior SDs, the strongest prior-data conflict in the suite.
    """
    assert report_cells._kappa_role_and_suffix(name) == expected


def test_dispersion_scope_reports_a_strained_legacy_curve_as_one_finding(tmp_path, capsys):
    """``kappa_min + exp(a - b_mag z)`` couples the intercept and the slope.

    Reporting them as two caveats would read as two problems where the fit has
    one, and would invite fixing the intercept prior when the slope prior is what
    is binding.
    """
    fit = _kappa_fit(
        tmp_path,
        contraction=[
            {"parameter": "a_kappa_s", "posterior_mean": 0.361, "posterior_sd": 0.167,
             "prior_cdf": 0.043, "contraction": 0.833, "flags": "pressing"},
            {"parameter": "b_kappa_mag_s", "posterior_mean": 1.419, "posterior_sd": 0.195,
             "prior_cdf": 1.0, "contraction": -0.080, "flags": "pressing+uninformed"},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert out.count("- **") == 1, "the coupled pair must be one bullet"
    assert "The shape of this curve" in out
    assert "`a_kappa_s`" in out and "`b_kappa_mag_s`" in out
    assert "one finding, not two" in out
    assert "steeper decline with age than the slope prior allows" in out and "pulled down to compensate" in out


def test_dispersion_scope_does_not_call_a_far_tail_posterior_uninformed(tmp_path, capsys):
    """Contraction cannot separate "data said nothing" from "said something far away".

    ``b_kappa_mag_s`` is 7.8 prior SDs out with a 13% relative posterior spread,
    and scores contraction -0.08. Reporting that as "not estimated from this
    data" would be the opposite of the truth.
    """
    fit = _kappa_fit(
        tmp_path,
        contraction=[
            {"parameter": "b_kappa_mag_s", "posterior_mean": 1.419, "posterior_sd": 0.195,
             "prior_cdf": 1.0, "contraction": -0.080, "flags": "pressing+uninformed"},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "pressing against its prior" in out
    assert "not estimated from this data" not in out
    assert "does not** mean the data was silent" in out.replace("**not**", "not**")


def test_dispersion_scope_still_calls_a_mid_prior_flat_posterior_unestimated(tmp_path, capsys):
    fit = _kappa_fit(
        tmp_path,
        definition={"n_trials": 810, "kappa_s": {"anchor_ages": [18.0, 72.0]}},
        contraction=[
            {"parameter": "kappa_excess_young_s", "posterior_mean": 49.6, "posterior_sd": 27.6,
             "prior_cdf": 0.62, "contraction": -0.2286, "flags": "uninformed"},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "not estimated from this data" in out and "mid-prior" in out
    assert "pressing against its prior" not in out


# --------------------------------------------------------------------------
# Conditional production check
# --------------------------------------------------------------------------


def _by_understood_fit(tmp_path, monkeypatch, *, frame):
    fit = _fit(tmp_path, definition={"n_trials": 810})
    pd.DataFrame(
        {"words_understood": [50.0, 100.0, 200.0, 300.0], "q_median": [0.05, 0.10, 0.22, 0.43],
         "ci_lo": [0.04, 0.09, 0.21, 0.40], "ci_hi": [0.06, 0.11, 0.23, 0.45]}
    ).to_csv(fit / "production_rate_by_understood.csv", index=False)
    monkeypatch.setattr(report_cells, "_verified_frame", lambda manifest: (frame, None))
    return fit


def test_conditional_production_check_sets_the_children_beside_the_curve(
    tmp_path, monkeypatch, capsys
):
    """The defect (#233): the curve is population q at the age where the population
    median reaches U, and three captions read it as E[q | understood = U]. At 300
    words the VG21 and VG22 curves both sit near 0.4 while the children who
    understood 300 words have median ratios of 0.27 and 0.13.
    """
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame(
        {"understood": np.full(n, 300.0), "spoken": 300.0 * rng.uniform(0.1, 0.4, n),
         "age": np.full(n, 38.0), "subject_key": [f"c{i}" for i in range(n)]}
    )
    report_cells.render_conditional_production_check(str(_by_understood_fit(tmp_path, monkeypatch, frame=frame)))
    out = capsys.readouterr().out
    assert "is **not** the share" in out
    assert "| 300 | 0.43 [0.40, 0.45] | 60 children |" in out
    assert "| 38 months |" in out
    # No children near 50, 100 or 200, so those rows are not shown.
    assert "| 50 |" not in out and "| 200 |" not in out
    assert "must be made in the right-hand column" in out


def test_conditional_production_check_says_why_when_the_frame_is_unavailable(
    tmp_path, monkeypatch, capsys
):
    fit = _fit(tmp_path)
    pd.DataFrame({"words_understood": [300.0], "q_median": [0.43]}).to_csv(
        fit / "production_rate_by_understood.csv", index=False
    )
    monkeypatch.setattr(report_cells, "_verified_frame", lambda manifest: (None, "the hash moved"))
    report_cells.render_conditional_production_check(str(fit))
    out = capsys.readouterr().out
    assert "is **not** the share" in out
    assert "because the hash moved" in out


def test_conditional_production_check_says_so_when_the_curve_is_absent(tmp_path, capsys):
    report_cells.render_conditional_production_check(str(_fit(tmp_path)))
    assert "no by-understood production curve" in capsys.readouterr().out


def test_observed_production_ratio_at_levels_windows_and_thresholds():
    """Shared by the page block and compare_ds_td_re, so one definition of "near"."""
    frame = pd.DataFrame(
        {
            "understood": [95, 100, 105, 110, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
            "spoken": [10, 20, 30, 40] + [30, 60, 90, 120, 150, 30, 60, 90, 120, 150, 90],
            "age": [12, 12, 13, 13] + [40] * 11,
            "subject_key": [f"c{i}" for i in range(14)] + ["c13"],
        }
    )
    table = report_cells.observed_production_ratio_at_levels(frame, [100, 200, 300])
    # 100 has four rows (below the ten-row floor); 200 has none; 300 has eleven.
    assert table["level"].tolist() == [300.0]
    row = table.iloc[0]
    assert row["n"] == 11 and row["children"] == 10
    assert abs(row["median"] - 0.3) < 1e-9
    assert row["median_age"] == 40.0


def test_dispersion_scope_pairs_only_two_pressing_parameters_and_reads_the_direction(
    tmp_path, capsys
):
    """A mid-prior, barely-contracted intercept beside a pressing slope is two facts.

    And the pair's sentence must follow the tails: a slope in its lower tail is a
    gentler decline, not the steeper one VG08 happens to show.
    """
    fit = _kappa_fit(
        tmp_path,
        contraction=[
            {"parameter": "a_kappa_s", "posterior_mean": 2.0, "posterior_sd": 1.0,
             "prior_cdf": 0.48, "contraction": 0.01, "flags": "uninformed"},
            {"parameter": "b_kappa_mag_s", "posterior_mean": 1.4, "posterior_sd": 0.2,
             "prior_cdf": 1.0, "contraction": -0.08, "flags": "pressing+uninformed"},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "The shape of this curve" not in out
    assert out.count("- **") == 2

    (tmp_path / "low").mkdir()
    fit = _kappa_fit(
        tmp_path / "low",
        contraction=[
            {"parameter": "a_kappa_u", "posterior_mean": 3.5, "posterior_sd": 0.2,
             "prior_cdf": 0.97, "contraction": 0.8, "flags": "pressing"},
            {"parameter": "b_kappa_mag_u", "posterior_mean": 0.01, "posterior_sd": 0.01,
             "prior_cdf": 0.02, "contraction": 0.9, "flags": ""},
        ],
    )
    report_cells.render_dispersion_scope(str(fit))
    out = capsys.readouterr().out
    assert "The shape of this curve" in out and "gentler decline" in out
    assert "pulled up to compensate" in out


def test_reference_child_calibration_sets_the_curve_beside_the_sample(tmp_path, monkeypatch, capsys):
    """The population curve is the child in the average study; the sample median is
    the children in the data. At ages covered by one or two studies they can sit
    far apart, and every milestone on the page is read off the former."""
    fit = _fit(tmp_path)
    ages = np.arange(8.0, 61.0)
    pd.DataFrame({"age_months": ages, "Ey_median": ages * 5}).to_csv(fit / "posterior_summary_monthly_u.csv", index=False)
    pd.DataFrame({"age_months": ages, "Ey_median": ages * 5 + 20}).to_csv(fit / "posterior_summary_monthly_weighted_u.csv", index=False)
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"age": rng.uniform(10, 58, 600)})
    frame["understood"] = frame["age"] * 6
    monkeypatch.setattr(report_cells, "_verified_frame", lambda manifest: (frame, None))
    report_cells.render_reference_child_calibration(str(fit))
    out = capsys.readouterr().out
    assert "| Age | Outcome | Reference child | Administration-weighted child | Sample median |" in out
    assert out.count("| understood |") == 3
    assert "largest gap between the reference child and the sample" in out
    assert "child in the *average study*" in out


def test_reference_child_calibration_says_so_without_a_monthly_summary(tmp_path, capsys):
    report_cells.render_reference_child_calibration(str(_fit(tmp_path)))
    assert "no monthly summary" in capsys.readouterr().out
