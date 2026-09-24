# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Numerical counterexamples from the September statistical review."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import betabinom

from vocab_growth.censored_ages import summarise_crossing_ages
from vocab_growth.loo_policy import suppressed_outcomes
from vocab_growth.predictive_mixtures import (
    conditional_log_predictive,
    conditioned_resample,
)


def script(name):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location("review_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_conditional_mixture_updates_component_weights():
    # Observing U=9 favours the component whose child also has higher speech.
    pu = np.array([0.1, 0.9])
    q = np.array([0.2, 0.8])
    lu = betabinom.logpmf(9, 10, pu * 20, (1 - pu) * 20)
    ls = betabinom.logpmf(8, 9, q * 20, (1 - q) * 20)
    exact = np.log(np.dot(np.exp(lu), np.exp(ls)) / np.exp(lu).sum())
    assert conditional_log_predictive(lu, ls) == pytest.approx(exact)
    assert exact > np.log(np.exp(ls).mean()) + 0.6
    values = np.tile(q, 1000)[:, None]
    weights = np.tile(lu, 1000)[:, None]
    draws = conditioned_resample(values, weights, np.random.default_rng(14))
    assert draws.mean() > 0.79


def test_censored_age_interval_retains_probability_outside_the_window():
    result = summarise_crossing_ages(np.r_[np.arange(20, 100), np.full(20, np.inf)], 120, 12)
    assert result["hi"] == np.inf
    assert result["frac_beyond_window"] == 0.2
    early = summarise_crossing_ages(np.full(10, -np.inf), 72, 12)
    assert early["before_floor"] and not early["beyond_cap"]
    assert early["median"] == -np.inf


@pytest.mark.parametrize("name", ["VG16", "vg16"])
def test_cross_lag_policy_is_case_independent(name):
    assert suppressed_outcomes(name) == {"y_u_obs"}


def test_forward_difference_se_uses_children():
    wf = script("wave_forward_score.py")
    frame = pd.DataFrame({"subject_code": [1, 1, 2, 2, 3, 3],
                          "has_lag": True, "source_in_training": True,
                          "elpd_spoken_lag": [1, 1, 3, 3, 5, 5],
                          "elpd_spoken_control": 0.0})
    result = wf.paired_difference(frame, "elpd_spoken", restriction="lagged-from-training")
    assert result["n_rows"] == 6 and result["n_children"] == 3
    assert result["se"] == pytest.approx(np.sqrt(3 * np.var([2, 6, 10], ddof=1)))


def test_icc_removes_noise_in_child_means():
    rank = script("experiments/rank_stability.py")
    rng = np.random.default_rng(231)
    frame = pd.DataFrame({"child": np.repeat(np.arange(10000), 2), "resid": rng.normal(size=20000)})
    assert rank.icc(frame) < 0.025
    effects = rng.normal(size=10000)
    frame["resid"] += np.repeat(effects, 2)
    assert rank.icc(frame) == pytest.approx(0.5, abs=0.025)


def test_oracle_slope_accounts_for_zero_sum_covariance_once():
    hierarchy = script("experiments/psi_hierarchy_simulation.py")
    assert hierarchy.oracle_slope(1, [1])[0] == pytest.approx(0.5)
    np.testing.assert_allclose(hierarchy.oracle_slope(1, np.ones(4)), 4 / 7)


def test_age_bands_reject_overlap_and_allow_gaps():
    checks = script("predict_new_study_checks.py")
    assert checks._parse_bands("29-35,17-25") == [(17, 26), (29, 36)]
    with pytest.raises(ValueError, match="overlap"):
        checks._parse_bands("17-25,24-30")


def test_nested_new_child_sample_matches_exact_speech_tail():
    from vocab_growth.comparison import TotalSpreadPlan, new_child_count_sample

    plan = TotalSpreadPlan("spoken", "plot", True, "p", "ku", None, "q", "ks")
    values = {name: np.array([[value]]) for name, value in {"p": .4, "ku": 3., "q": .6, "ks": 20.}.items()}
    sample = new_child_count_sample(plan, values, np.array([24.]), 810, 0, 200000, np.random.default_rng(129))
    counts = np.arange(811)
    exact = np.dot(betabinom.pmf(counts, 810, 1.2, 1.8), betabinom.cdf(100, counts, 12, 8))
    assert np.mean(sample <= 100) == pytest.approx(exact, abs=0.004)
    assert exact > 0.27  # The obsolete single Beta-Binomial gives less than 0.10.


def test_reporting_cap_is_applied_before_peak_or_crossing():
    from vocab_growth import comparison as c

    ages = np.array([12., 36., 60., 84., 108.])
    words = np.array([[10., 20., 30., 60., 160.]])
    grid, values = c.restrict_reporting_trajectory("vg20", "understood", ages, words)
    assert grid[-1] == 72
    assert values[0, -1] == 45
    assert np.isnan(c.first_crossing_age(values, grid, 50)[0])


def test_source_verification_refuses_a_missing_nonempty_zero_speech_row(tmp_path):
    source = script("build_us01_source.py")
    path = tmp_path / "export.csv"
    pd.DataFrame({"dataset_name": ["Edgin"] * 3, "form": ["WG"] * 3,
                  "health_conditions": ["down syndrome"] * 3,
                  "age": [12, 17, 18], "comprehension": [0, 0, 100],
                  "production": [0, 0, 0]}).to_csv(path, index=False)
    assert not source.verify([], str(path))
    records = [{"form": "WG", "dev_status": "down_syndrome", "in_norming_window": True,
                "age": 18, "comprehension": 100, "production": 0}]
    assert source.verify(records, str(path))


def test_loso_uses_the_joint_nested_likelihood():
    import xarray as xr
    from scipy.special import logit

    loso = script("loso_compare.py")
    frame = pd.DataFrame({"understood": [400., np.nan, 50.], "spoken": [200., 100., 100.],
                          "subject_code": [0, 1, 2], "study_code": [0, 0, 0]})
    data = {name: (("chain", "draw", "obs"), np.full((1, 1, 3), value))
            for name, value in {"f_u_obs": logit(.4), "h_obs": logit(.6),
                                "kappa_u_obs": 3., "kappa_s_obs": 20.}.items()}
    data.update({name: (("chain", "draw", "study"), np.zeros((1, 1, 1)))
                 for name in ("delta_u", "delta_q")})
    trace = xr.DataTree.from_dict({"posterior": xr.Dataset(data)})
    spec = loso.ModelSpec("VG07", "unused", False, False)
    actual = loso.marginal_subject_loglik(trace, frame, spec, n_re_samples=3, thin=1)
    expected = [betabinom.logpmf(400, 810, 1.2, 1.8) + betabinom.logpmf(200, 400, 12, 8),
                betabinom.logpmf(100, 810, .24 * 20, .76 * 20),
                betabinom.logpmf(50, 810, 1.2, 1.8) + betabinom.logpmf(100, 810, .24 * 20, .76 * 20)]
    np.testing.assert_allclose(actual[0, 0], expected)


def test_band_likelihood_ratio_uses_the_exact_union(monkeypatch):
    checks = script("predict_new_study_checks.py")
    frame = pd.DataFrame({"age": [18, 23, 27, 30, 34], "subject_id": range(5)})
    calls = []

    def offset(post, grid, rows, draws, n, definition):
        calls.append(rows["age"].tolist())
        return 0., 0., 0., -float(len(rows))

    monkeypatch.setattr(checks, "_offset", offset)
    monkeypatch.setattr(checks.pns, "child_structure", lambda *args: "constant")
    result = checks.offset_profile(None, None, frame, None, 810, None, ["17-25,29-35"], None)
    assert result["lr[17-25,29-35]"] == 0
    assert [18, 23, 30, 34] in calls


def test_null_simulation_does_not_add_form_censoring(monkeypatch):
    from types import SimpleNamespace

    checks = script("predict_new_study_checks.py")
    frame = pd.DataFrame({"age": [24.], "subject_id": [1], "survey_vocab_max": [10]})
    monkeypatch.setattr(checks.pns, "engine_profile", lambda post: {k: k for k in ("f", "h", "ku", "ks", "tau_u", "tau_q")})
    monkeypatch.setattr(checks.pns, "_flat", lambda post, name: np.array([0.]) if name.startswith("tau") else np.zeros((1, 2)))
    monkeypatch.setattr(checks.pns, "child_structure", lambda *args: "none")
    monkeypatch.setattr(checks.pns, "child_ref_age", lambda *args: 36)
    monkeypatch.setattr(checks.pns, "draw_child_params", lambda *args: np.zeros((1, 1, 4)))
    monkeypatch.setattr(checks.pns, "child_deltas", lambda *args: (0., 0.))
    monkeypatch.setattr(checks.pns, "_betabinom_draw", lambda rng, n, p, k: np.broadcast_to(np.asarray(n) // 2, np.shape(p)))
    simulated, _, _, exceeded = checks.simulate_study(None, [12., 36.], frame, SimpleNamespace(), 0, np.random.default_rng(3), 810)
    assert simulated["understood"].iloc[0] == 405
    assert simulated["spoken"].iloc[0] == 202
    assert exceeded == 1


def test_sex_shift_uses_product_of_shifted_probabilities():
    from types import SimpleNamespace

    from scipy.special import expit, logit

    shift = script("experiments/sex_shift_predictive.py")
    pred = SimpleNamespace(su=pd.DataFrame({"p_population_median": [.5, .6]}, index=[24., 30.]),
                           sq=pd.DataFrame({"q_median": [.5, .6]}, index=[24., 30.]))
    result = shift.constant_shift_in_words_and_months(pred, .2, .15)
    row = result[(result.outcome == "spoken") & (result.age == 24)].iloc[0]
    expected = 810 * (expit(.1) * expit(.075) - expit(-.1) * expit(-.075))
    assert row.gap_words == pytest.approx(expected)
    wrong = 810 * (expit(logit(.25) + .175) - expit(logit(.25) - .175))
    assert abs(row.gap_words - wrong) > 15


def test_rank_deficient_signing_regression_is_refused():
    audit = script("sign_speech_association_audit.py")
    x = np.linspace(0, 1, 20)
    frame = pd.DataFrame({"x": x, "y": x**2, "speech": 1 - x, "subject_id": np.repeat(np.arange(10), 2)})
    with pytest.raises(ValueError, match="Rank"):
        audit.ols_cluster(frame, "y", "x", ["speech"])


def test_weighted_cluster_standard_error_agrees_with_independent_implementation():
    import statsmodels.api as sm

    audit = script("psi_heterogeneity_audit.py")
    rng = np.random.default_rng(63)
    child = np.repeat(np.arange(25), 3)
    x = np.column_stack([np.ones(75), rng.normal(size=75)])
    y = x @ [.5, -.2] + np.repeat(rng.normal(size=25), 3) + rng.normal(size=75) * .1
    weights = rng.uniform(.5, 2, 75)
    beta, se, _ = audit._wls(y, x, weights, child)
    expected = sm.WLS(y, x, weights=weights).fit(cov_type="cluster", cov_kwds={"groups": child})
    np.testing.assert_allclose(beta, expected.params)
    np.testing.assert_allclose(se, expected.bse)


def test_history_updates_parameter_mixture_before_second_visit_prediction(monkeypatch):
    import xarray as xr
    from scipy.special import logit, logsumexp

    prediction = script("predict_new_study.py")
    original_interp = prediction._interp_draws
    interpolations = []

    def counted_interp(*args, **kwargs):
        interpolations.append(1)
        return original_interp(*args, **kwargs)

    monkeypatch.setattr(prediction, "_interp_draws", counted_interp)
    pu, q = np.array([.1, .9]), np.array([.2, .8])
    curves = {"f_u_plot": logit(pu), "h_plot": logit(q),
              "kappa_u_plot": [20., 20.], "kappa_s_plot": [20., 20.]}
    data = {name: (("chain", "draw", "age"), np.repeat(np.asarray(v)[None, :, None], 2, axis=2))
            for name, v in curves.items()}
    data.update({name: (("chain", "draw"), np.zeros((1, 2))) for name in ("tau_subj_u", "tau_subj_q")})
    post = xr.Dataset(data)
    frame = pd.DataFrame({"subject_id": [1, 1], "timepoint": ["t1", "t2"],
                          "age": [12., 24.], "understood": [9, 9], "spoken": [8, 8]})
    frame = pd.concat([frame, frame.assign(subject_id=2)], ignore_index=True)
    summary, scores = prediction.within_child(post, np.array([12., 24.]), frame, np.array([0, 1]),
                    np.random.default_rng(8), 10, None, n_candidates=3, chunk=1)
    lu = betabinom.logpmf(9, 10, pu * 20, (1 - pu) * 20)
    ls = betabinom.logpmf(8, 9, q * 20, (1 - q) * 20)
    history = lu + ls
    assert scores["lpd_understood_given_both"].iloc[0] == pytest.approx(logsumexp(history + lu) - logsumexp(history))
    assert scores["lpd_spoken_given_both"].iloc[0] == pytest.approx(logsumexp(history + lu + ls) - logsumexp(history + lu))
    assert len(interpolations) == 8
    assert "importance_weight_ess" in summary.columns
    assert len(scores.filter(like="importance_weight_ess").columns) == 4
    assert np.isfinite(scores.filter(like="importance_weight_ess").to_numpy()).all()


def test_publication_validation_happens_before_staging(monkeypatch):
    publisher = script("publish_comparison.py")
    monkeypatch.setattr(sys, "argv", ["publish_comparison.py", "--dry-run"])
    monkeypatch.setattr(publisher.env, "set_output_root", lambda _: None)
    monkeypatch.setattr(publisher, "validate_comparison_manifest", lambda *args, **kwargs: (["failed hard convergence"], []))
    monkeypatch.setattr(publisher, "source_data_hash", lambda _: "data")
    staged = []
    monkeypatch.setattr(publisher, "stage_inputs", staged.append)
    with pytest.raises(SystemExit, match="failed hard convergence"):
        publisher.main()
    assert not staged


@pytest.mark.parametrize("changed", ["input", "source", "html", "asset", "missing-receipt"])
def test_reusing_render_requires_matching_inputs_and_assets(tmp_path, monkeypatch, changed):
    from vocab_growth.fit_artifacts import write_json_atomic

    publisher = script("publish_comparison.py")
    book = tmp_path / "book"
    book.mkdir()
    files = {"index.qmd": "source", "summary.csv": "x\n1\n", "figure.svg": "<svg/>",
             "index.html": '<html><img src="figure.svg"></html>'}
    for name, text in files.items():
        (book / name).write_text(text, encoding="utf-8")
    for name, value in {"BOOK_DIR": str(book), "BOOK_SOURCE": str(book / "index.qmd"),
                        "BOOK_HTML": str(book / "index.html")}.items():
        monkeypatch.setattr(publisher, name, value)
    inputs = {name: publisher._file_sha256(str(book / name)) for name in ("summary.csv", "figure.svg")}
    receipt_path = book / "_publication_receipt.json"
    write_json_atomic(str(receipt_path), publisher.render_receipt(inputs))
    path = {"input": "summary.csv", "source": "index.qmd", "html": "index.html", "asset": "figure.svg"}.get(changed)
    if path:
        with (book / path).open("a", encoding="utf-8") as handle:
            handle.write(" changed")
    else:
        receipt_path.unlink()
    monkeypatch.setattr(sys, "argv", ["publish_comparison.py", "--no-render", "--dry-run"])
    monkeypatch.setattr(publisher.env, "set_output_root", lambda _: None)
    monkeypatch.setattr(publisher, "validate_inputs", lambda _, **kwargs: inputs)
    collected = []
    monkeypatch.setattr(publisher, "collect", collected.append)
    with pytest.raises(SystemExit, match="stage and render"):
        publisher.main()
    assert not collected


def test_matching_render_can_be_assembled_without_upload(tmp_path, monkeypatch):
    from vocab_growth.fit_artifacts import write_json_atomic

    publisher = script("publish_comparison.py")
    book = tmp_path / "book"
    book.mkdir()
    for name, contents in {"index.qmd": "source", "summary.csv": "x\n1\n",
                           "index.html": '<html><a href="summary.csv">Table</a></html>'}.items():
        (book / name).write_text(contents, encoding="utf-8")
    for name, value in {"BOOK_DIR": str(book), "BOOK_SOURCE": str(book / "index.qmd"),
                        "BOOK_HTML": str(book / "index.html")}.items():
        monkeypatch.setattr(publisher, name, value)
    inputs = {"summary.csv": publisher._file_sha256(str(book / "summary.csv"))}
    write_json_atomic(str(book / "_publication_receipt.json"), publisher.render_receipt(inputs))
    destination = tmp_path / "assembled"
    monkeypatch.setattr(sys, "argv", ["publish_comparison.py", "--no-render", "--dry-run",
                                     "--work-dir", str(destination)])
    monkeypatch.setattr(publisher.env, "set_output_root", lambda _: None)
    monkeypatch.setattr(publisher, "validate_inputs", lambda _, **kwargs: inputs)
    publisher.main()
    assert (destination / "index.html").read_bytes() == (book / "index.html").read_bytes()
    assert (destination / "summary.csv").read_bytes() == (book / "summary.csv").read_bytes()


def test_sex_group_replication_draws_comprehension_then_speech(monkeypatch):
    from types import SimpleNamespace

    import xarray as xr
    from scipy.special import expit

    sex = script("experiments/vg20_sex_arm.py")
    pns = script("predict_new_study.py")
    post = xr.Dataset({
        name: (("chain", "draw", "age"), np.full((1, 2, 2), value))
        for name, value in {"f_u_plot": 0., "h_plot": 0.,
                            "kappa_u_plot": 20., "kappa_s_plot": 20.}.items()
    })
    for name in ("delta_u", "delta_q"):
        post[name] = (("chain", "draw", "study"), np.zeros((1, 2, 1)))
    cd = xr.Dataset({"X_plot": ("age", [12., 24.]), "study_obs": ("obs", [0, 0, 0]),
                     "subject_obs": ("obs", [0, 0, 1]),
                     "s_is_conditional": ("speech_obs", [True, True, False])})
    frame = pd.DataFrame({"age": [12., 24., 24.]})
    class Posterior:
        def to_dataset(self):
            return post

        def __contains__(self, name):
            return name in post

    arm = SimpleNamespace(frame=frame, definition=None, idata=SimpleNamespace(
        posterior=Posterior(), constant_data=cd))
    monkeypatch.setattr(sex, "_predict_new_study", lambda: pns)
    # One effect per child and draw. Visits 0 and 1 belong to the same child.
    child = np.zeros((2, 2, 4))
    child[:, 0, 0] = [-1., 1.]
    monkeypatch.setattr(pns, "draw_child_params", lambda *args: child)
    calls = []

    def draw(rng, n, p, k):
        calls.append((np.asarray(n), p.copy()))
        return np.floor(np.asarray(n) * p).astype(int)

    monkeypatch.setattr(pns, "_betabinom_draw", draw)
    monkeypatch.setattr(sex, "_outcome_rows", lambda _: [
        ("understood", "y_u_obs", np.array([0, 1]), frame.iloc[:2], np.array([10, 10])),
        ("spoken", "y_s_obs", np.arange(3), frame, np.array([5, 5, 5]))])
    replicated = {}

    def cells(name, outcome, rows_df, y, rep):
        replicated[outcome] = rep
        return [{"outcome": outcome}]

    monkeypatch.setattr(sex, "_cell_rows", cells)
    result = sex.marginal_ppc_by_sex_table({"control": arm}, n_draws=2)
    u = np.floor(810 * expit([-1., 1.]))
    np.testing.assert_array_equal(replicated["understood"], np.column_stack([u, u]))
    np.testing.assert_array_equal(replicated["spoken"][:, :2], np.column_stack([u // 2, u // 2]))
    np.testing.assert_array_equal(calls[1][0][:, 2], [810, 810])
    np.testing.assert_array_equal(replicated["spoken"][:, 2], [202, 202])
    assert result.prediction_target.str.contains("joint counts").all()
