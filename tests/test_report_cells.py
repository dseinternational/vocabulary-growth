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

import pandas as pd
import pytest

from vocab_growth import glossary, report_cells
from vocab_growth.models.diagnostics_utils import render_convergence_caveats


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
    row = {**_CLEAN_ROW, "pareto_k_good": 1128, "pareto_k_bad": 258, "pareto_k_very_bad": 30}
    report_cells.render_loo_section(str(_loo_fit(tmp_path, [row])))
    out = capsys.readouterr().out
    assert "288 of 1,521 observations (19%) exceed the threshold" in out
    assert "30 of them above 1" in out


def test_hierarchical_fits_get_the_wrong_unit_explanation(tmp_path, capsys):
    """A high k share under subject effects is expected, not evidence of misfit.

    VG10 has 20% of spoken and 31% of understood observations over the
    threshold. Reporting that as unreliability without saying why would send a
    reader looking for a modelling fault that is not there.
    """
    row = {**_CLEAN_ROW, "pareto_k_good": 1128, "pareto_k_bad": 258, "pareto_k_very_bad": 30}
    fit = _loo_fit(tmp_path, [row], parameters=("eta_u", "tau_subj_u"))
    report_cells.render_loo_section(str(fit))
    out = capsys.readouterr().out
    assert "wrong unit of prediction" in out
    assert "kfold_loso.py" in out


def test_non_hierarchical_fits_do_not_get_that_explanation(tmp_path, capsys):
    row = {**_CLEAN_ROW, "pareto_k_good": 1128, "pareto_k_bad": 258, "pareto_k_very_bad": 30}
    fit = _loo_fit(tmp_path, [row], parameters=("eta",))
    report_cells.render_loo_section(str(fit))
    assert "wrong unit of prediction" not in capsys.readouterr().out


def test_dropped_degenerate_observations_are_disclosed(tmp_path, capsys):
    row = {**_CLEAN_ROW, "n_dropped_degenerate": 12}
    report_cells.render_loo_section(str(_loo_fit(tmp_path, [row])))
    assert "12 observation(s) were excluded as degenerate" in capsys.readouterr().out


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

