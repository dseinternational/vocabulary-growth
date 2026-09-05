#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the girl-boy vocabulary gap survive VG20's structure? Fit it and see (#295).

``notes/202609041206-sex-differences-in-vocabulary.md`` measured the sex
difference descriptively -- an empirical-logit regression on the merged view,
before the loader's masking rules and with no child or study random effects --
at about +0.2 logits on words understood and +0.35 on words spoken, girls ahead,
and recommended one exploratory variant of VG20 rather than a change to the
model of record. This harness is that variant. It fits three arms, every one
derived from ``VG20`` through ``_as_definition_subclass`` onto
``BivariateSexShiftModelDefinition`` and run through the engine's own pipeline
into a **separate output root**, so nothing here can become a model of record:

* ``control`` -- VG20 restricted to the administrations with a recorded sex
  (eight studies; 997 rows and 559 children on the 2026-09-04 database).
* ``sex`` -- the same rows, plus ``beta_sex_u`` and ``beta_sex_q``, each
  ``Normal(0, 0.5)``, multiplying a girls ``+1/2`` / boys ``-1/2`` contrast on
  the understood and production-ratio logits. Constant in age, for the reasons
  the note gives. ``beta_sex_*`` read directly as girl-minus-boy differences in
  logits, and the population curves stay the sex-balanced average.
* ``full`` -- VG20 on every row, as the inert subclass, so what the restriction
  costs (six studies, a quarter of the children) can be read at the same tier.
  Not needed for the effect itself, and the slowest arm.

``compare`` reads the finished arms and writes, under
``<output-root>/comparisons/sex-effect/``:

* ``vg20_sex_arm_parameters.csv`` -- the sex coefficients, ``rho_uq``, the
  child and study scales and the convergence figures, per arm; plus the
  coefficients as a share of the child scale and as a posterior probability
  of being positive.
* ``vg20_sex_arm_trajectories.csv`` -- the population curves at the canonical
  ages, per arm, with the control-to-sex and control-to-full differences
  against the control's own interval width.
* ``vg20_sex_arm_words.csv`` -- the fitted shift read as girls' and boys'
  expected words on the population curve, and their gap, at the canonical ages.
* ``vg20_sex_arm_ppc_by_sex.csv`` and ``vg20_sex_arm_ppc_marginal_by_sex.csv``
  -- the check the note asked for: observed mean counts by sex within age band
  against each arm's posterior predictive, and the girl-minus-boy difference in
  each band against its replicated distribution. The first uses the stored
  predictive, which conditions on each child's fitted random effect and so
  already absorbs most of a child-level covariate; the second draws a fresh
  child per child (the new-child predictive) and is the one to read. Under the
  control a real effect shows as girls above and boys below the predictive;
  under the sex arm both should be centred, in every band if the constant-in-age
  form is right.
* ``vg20_sex_arm_loo.csv`` -- paired leave-one-out comparison of the two arms
  per outcome, valid because they see identical rows.

The descriptive estimates the note reports are read from
``sex_effect_by_study.csv``, which ``sex_effect_by_study.py`` writes into the
same ``comparisons/sex-effect/`` directory, so the fitted values print beside
them without a hand-copied table that can go stale; when that file is absent
the reference table is skipped.

Usage::

    python scripts/experiments/vg20_sex_arm.py --output-dir /scratch/vg20-sex fit control
    python scripts/experiments/vg20_sex_arm.py --output-dir /scratch/vg20-sex fit sex
    python scripts/experiments/vg20_sex_arm.py --output-dir /scratch/vg20-sex fit full
    python scripts/experiments/vg20_sex_arm.py --output-dir /scratch/vg20-sex compare

``--output-dir`` and ``--sigma`` belong to the top-level parser, so they go
before the subcommand.

``--config`` defaults to ``test`` (4 chains x 2000 draws), the tier the note
recommended: ``dev`` under-converges the hierarchical models and a coefficient
this size needs an honest interval more than it needs speed. The one-off
harness conventions of ``scripts/experiments/README.md`` apply: this is a
record of how the numbers in ``notes/202609041530-vg20-sex-shift-arm.md`` were
obtained, not a maintained tool.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

N_TRIALS = 810
CANONICAL_AGES = (24, 36, 48, 60, 72)
AGE_BANDS = [(8, 30), (30, 42), (42, 54), (54, 72), (72, 116)]
MIN_CELL = 10
CI = (0.055, 0.945)

#: Overrides applied to VG20 for each arm.
ARMS: dict[str, dict] = {
    "control": {"sex_known_only": True},
    "sex": {"sex_known_only": True, "sex_effect_sigma": 0.5},
    "full": {},
}

def reference_table(output_root: str) -> pd.DataFrame:
    """The descriptive estimates of ``notes/202609041206``, as ``sex_effect_by_study.py`` wrote them.

    Read rather than transcribed: those numbers moved once already, when ie_02,
    uk_05 and uk_06 gained sex codes, and a hand-copied table would not have
    noticed. Empty when the sibling script has not been run on this output root.
    """
    path = os.path.join(output_root, "comparisons", "sex-effect", "sex_effect_by_study.csv")
    if not os.path.isfile(path):
        print(
            f"[vg20_sex_arm] no descriptive reference at {path}; run "
            "sex_effect_by_study.py to print it beside the fitted values."
        )
        return pd.DataFrame()
    return pd.read_csv(path)


def _predict_new_study():
    """``scripts/predict_new_study.py``, loaded by path as ``predict_new_study_checks.py`` does."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "predict_new_study.py")
    spec = importlib.util.spec_from_file_location("predict_new_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def arm_definition(arm: str, sigma: float):
    from vocab_growth.models import definitions as D

    overrides = dict(ARMS[arm])
    if "sex_effect_sigma" in overrides:
        overrides["sex_effect_sigma"] = sigma
    definition = D._as_definition_subclass(
        D.VG20,
        D.BivariateSexShiftModelDefinition,
        config_name=f"{D.VG20.config_name}-sex-{arm}",
        banner=f"Fitting VG20 sex arm '{arm}' (exploratory, issue #295)",
        **overrides,
    )
    # Validate at definition time, as `marginal_arm.py` and the sensitivity
    # overrides do, so a mis-specified arm fails here rather than after the
    # prepare and priors stages have written a manifest and a prior figure.
    D.validate_model_definition(definition)
    return definition


def _recorded_definition_differences(fit_dir: str, definition) -> list:
    """How the manifest in ``fit_dir`` differs from ``definition``; empty without a manifest."""
    from vocab_growth.models.fit_identity import definition_differences

    path = os.path.join(fit_dir, "fit_manifest.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return definition_differences(manifest["model"]["definition"], definition)


def _refuse_to_replace_a_different_fit(fit_dir: str, definition) -> None:
    """``--sigma`` is not in the directory name, so a second prior would replace the first."""
    diffs = _recorded_definition_differences(fit_dir, definition)
    if diffs:
        fields = ", ".join(f"{d.field} ({d.recorded!r} fitted, {d.expected!r} asked)" for d in diffs)
        raise SystemExit(
            f"{fit_dir} already holds a fit of a different definition: {fields}. "
            "Promotion would delete it; use another --output-dir for a different prior."
        )


def arm_dir(output_root: str, arm: str, sigma: float) -> str:
    definition = arm_definition(arm, sigma)
    return os.path.join(
        output_root, "models", f"{definition.model_id}-{definition.config_name}"
    )


def fit(args) -> int:
    import dse_research_utils.environment.setup as setup

    from vocab_growth import environment as env
    from vocab_growth.models.common import run_fit_pipeline
    from vocab_growth.models.common_bivariate_re import bivariate_re_stages
    from vocab_growth.models.exploratory import write_exploratory_marker

    definition = arm_definition(args.arm, args.sigma)
    env.set_output_root(args.output_dir)
    setup.init_script()
    print(
        f"[vg20_sex_arm] arm={args.arm} config={args.config} "
        f"sex_known_only={definition.sex_known_only} "
        f"sex_effect_sigma={definition.sex_effect_sigma} -> {arm_dir(args.output_dir, args.arm, args.sigma)}"
    )
    target = arm_dir(args.output_dir, args.arm, args.sigma)
    _refuse_to_replace_a_different_fit(target, definition)
    run_fit_pipeline(args.config, definition, stages=bivariate_re_stages(definition))
    # The pipeline promotes into a directory shaped exactly like a publishable
    # fit; say what this one is not (see `vocab_growth.models.exploratory`).
    write_exploratory_marker(
        target,
        model_label=f"VG20 sex arm '{args.arm}'",
        note=(
            "Exploratory VG20 variant (issue #295): an unregistered definition "
            "subclass fitted through the engine pipeline into a separate output "
            "root. Not a model of record and not validatable for publication."
        ),
    )
    return 0


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------


#: The population-curve columns of each outcome's stored summary. For the two
#: count outcomes the summaries lead with the subject-marginal predictive and
#: carry the population (median-child) curve as ``*_population_*``; the
#: production ratio has no marginal form and is stored as ``q_*``.
POPULATION_COLUMNS = {
    "u": ("Ey_population_median", "Ey_population_ci_lo", "Ey_population_ci_hi"),
    "s": ("Ey_population_median", "Ey_population_ci_lo", "Ey_population_ci_hi"),
    "q": ("q_median", "q_ci_lo", "q_ci_hi"),
}


def _population_columns(stem: str, df: pd.DataFrame) -> tuple[str, str, str]:
    cols = POPULATION_COLUMNS[stem]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"posterior_summary_{stem}.csv lacks {missing}; has {list(df.columns)}")
    return cols


def _summ(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    lo, hi = np.quantile(x, CI)
    return {"median": float(np.median(x)), "lo89": float(lo), "hi89": float(hi)}


class Arm:
    def __init__(self, output_root: str, arm: str, sigma: float):
        import arviz as az

        self.name = arm
        self.definition = arm_definition(arm, sigma)
        self.dir = arm_dir(output_root, arm, sigma)
        if not os.path.isfile(os.path.join(self.dir, "trace.nc")):
            raise FileNotFoundError(f"No finished fit for arm {arm!r} at {self.dir}")
        self.idata = az.from_netcdf(os.path.join(self.dir, "trace.nc"))
        self.diag = pd.read_csv(os.path.join(self.dir, "diagnostics.csv"), index_col=0)
        with open(os.path.join(self.dir, "fit_manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        # The directory name does not encode `--sigma`, so check the fitted
        # definition against the one asked for, field by field, the way the
        # pipeline's own validation does.
        diffs = _recorded_definition_differences(self.dir, self.definition)
        if diffs:
            fields = ", ".join(f"{d.field} ({d.recorded!r} fitted, {d.expected!r} asked)" for d in diffs)
            raise RuntimeError(f"Arm {arm!r} at {self.dir} was fitted under a different definition: {fields}.")
        self._frame = None

    @property
    def frame(self) -> pd.DataFrame:
        """The arm's prepared frame, rebuilt and checked against the manifest."""
        if self._frame is None:
            from vocab_growth.analysis_frames import analysis_frame_hash
            from vocab_growth.models.common_bivariate_re import (
                build_bivariate_re_analysis_frame,
            )

            frame, _ = build_bivariate_re_analysis_frame(self.definition)
            recorded = self.manifest["data"]["analysis_frame_hash"]
            if analysis_frame_hash(frame) != recorded:
                raise RuntimeError(
                    f"Arm {self.name!r}: the rebuilt frame's hash differs from the "
                    "manifest's, so rows cannot be aligned to the trace. The data or "
                    "the loader rules have moved since the fit."
                )
            self._frame = frame
        return self._frame

    def param_row(self, name: str) -> dict | None:
        if name not in self.diag.index:
            return None
        lo, hi = "eti89_lb", "eti89_ub"  # what `az.summary(..., ci_kind="eti")` writes
        r = self.diag.loc[name]
        row = {"mean": float(r["mean"]), "sd": float(r["sd"]), "lo89": float(r[lo]), "hi89": float(r[hi])}
        for k in ("r_hat", "ess_bulk", "ess_tail"):
            if k in self.diag.columns:
                row[k] = float(r[k])
        return row

    def draws(self, name: str) -> np.ndarray:
        return self.idata.posterior[name].values

    def convergence(self) -> dict:
        """The gate's verdict, not the maximum of the scalar-only `diagnostics.csv`.

        That CSV covers variables of size two or less, so its maximum R-hat
        cannot see the HSGP coefficient vectors the gate screens.
        """
        from vocab_growth.sensitivity.compare import diagnostics_gate

        gate = diagnostics_gate(self.dir)
        return {
            "divergences": int(self.idata.sample_stats["diverging"].values.sum()),
            "max_r_hat": gate.max_rhat,
            "min_ess_bulk": gate.min_ess,
            "converged": gate.converged,
            "convergence_source": gate.source,
            "caveats": "; ".join(gate.caveats),
        }


def parameters_table(arms: dict[str, Arm]) -> pd.DataFrame:
    rows = []
    for name, arm in arms.items():
        conv = arm.convergence()
        for p in ("beta_sex_u", "beta_sex_q", "rho_uq", "tau_subj_u", "tau_subj_q", "tau_u", "tau_q"):
            r = arm.param_row(p)
            if r is None:
                continue
            rows.append({"arm": name, "parameter": p, **r, **conv})
        if "beta_sex_u" in arm.idata.posterior:
            for coef, scale in (("beta_sex_u", "tau_subj_u"), ("beta_sex_q", "tau_subj_q")):
                b = arm.draws(coef)
                s = arm.draws(scale)
                rows.append({"arm": name, "parameter": f"{coef} / {scale}", **{k: v for k, v in _summ(b / s).items()}})
                rows.append({"arm": name, "parameter": f"P({coef} > 0)", "mean": float((b > 0).mean())})
            d = arm.draws("beta_sex_q") - arm.draws("beta_sex_u")
            rows.append({"arm": name, "parameter": "beta_sex_q - beta_sex_u", **_summ(d)})
            rows.append({"arm": name, "parameter": "P(beta_sex_q > beta_sex_u)", "mean": float((d > 0).mean())})
    return pd.DataFrame(rows)


def trajectories_table(arms: dict[str, Arm]) -> pd.DataFrame:
    rows = []
    for outcome, stem in (("understood", "u"), ("spoken", "s"), ("production ratio", "q")):
        per_arm = {}
        for name, arm in arms.items():
            df = pd.read_csv(os.path.join(arm.dir, f"posterior_summary_{stem}.csv"))
            per_arm[name] = df.set_index("age_months")
        control = per_arm["control"]
        med, lo, hi = _population_columns(stem, control)
        for age in CANONICAL_AGES:
            if age not in control.index:
                continue
            width = float(control.loc[age, hi] - control.loc[age, lo])
            row = {
                "outcome": outcome,
                "unit": "words" if stem in ("u", "s") else "ratio",
                "age_months": age,
                "control_median": float(control.loc[age, med]),
                "control_lo89": float(control.loc[age, lo]),
                "control_hi89": float(control.loc[age, hi]),
            }
            for name in ("sex", "full"):
                if name in per_arm and age in per_arm[name].index:
                    other = per_arm[name]
                    m, _, _ = _population_columns(stem, other)
                    row[f"{name}_median"] = float(other.loc[age, m])
                    row[f"{name}_minus_control"] = float(other.loc[age, m] - control.loc[age, med])
                    row[f"{name}_shift_as_share_of_control_width"] = (
                        row[f"{name}_minus_control"] / width if width > 0 else np.nan
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def words_table(sex: Arm) -> pd.DataFrame:
    """The fitted shift on the population curve, in words, by age."""
    post = sex.idata.posterior
    ages = np.asarray(sex.idata.constant_data["X_query"].values, dtype=float)
    pu = post["p_u_query"].values  # chain, draw, query
    q = post["q_query"].values
    bu = post["beta_sex_u"].values[..., None]
    bq = post["beta_sex_q"].values[..., None]
    lu, lq = logit(pu), logit(q)
    girls_u, boys_u = expit(lu + 0.5 * bu), expit(lu - 0.5 * bu)
    girls_s = girls_u * expit(lq + 0.5 * bq)
    boys_s = boys_u * expit(lq - 0.5 * bq)
    rows = []
    for j, age in enumerate(ages):
        if int(round(age)) not in CANONICAL_AGES:
            continue
        for outcome, g, b in (("understood", girls_u, boys_u), ("spoken", girls_s, boys_s)):
            gw, bw = N_TRIALS * g[..., j], N_TRIALS * b[..., j]
            gap = _summ(gw - bw)
            gap_logit = _summ(logit(g[..., j]) - logit(b[..., j]))
            rows.append(
                {
                    "outcome": outcome,
                    "age_months": int(round(age)),
                    "gap_logit_median": gap_logit["median"],
                    "pooled_words": float(np.median(N_TRIALS * (pu if outcome == "understood" else pu * q)[..., j])),
                    "girls_words": float(np.median(gw)),
                    "boys_words": float(np.median(bw)),
                    "gap_median": gap["median"],
                    "gap_lo89": gap["lo89"],
                    "gap_hi89": gap["hi89"],
                }
            )
    return pd.DataFrame(rows)


def _cell_rows(arm: str, outcome: str, rows_df: pd.DataFrame, y: np.ndarray, rep: np.ndarray) -> list[dict]:
    """Observed and replicated mean counts by sex within age band, plus the girl-minus-boy difference.

    ``rep`` is ``draws x n`` replicated counts aligned to ``rows_df``/``y``.
    """
    age = rows_df["age"].to_numpy(dtype=float)
    sex = rows_df["sex"].to_numpy()
    out: list[dict] = []
    bands = list(AGE_BANDS) + [(AGE_BANDS[0][0], AGE_BANDS[-1][1])]
    for lo_age, hi_age in bands:
        in_band = (age >= lo_age) & (age < hi_age)
        label = f"{lo_age}-{hi_age}" if (lo_age, hi_age) in AGE_BANDS else "all"
        cell_obs, cell_rep = {}, {}
        for s in ("F", "M"):
            cell = in_band & (sex == s)
            n = int(cell.sum())
            if n < MIN_CELL:
                continue
            t_obs = float(y[cell].mean())
            t_rep = rep[:, cell].mean(axis=1)
            cell_obs[s], cell_rep[s] = t_obs, t_rep
            out.append(_cell_row(arm, outcome, label, "girls" if s == "F" else "boys", n, t_obs, t_rep))
        if {"F", "M"} <= set(cell_obs):
            out.append(
                _cell_row(
                    arm,
                    outcome,
                    label,
                    "girls minus boys",
                    int(in_band.sum()),
                    cell_obs["F"] - cell_obs["M"],
                    cell_rep["F"] - cell_rep["M"],
                )
            )
    return out


def _cell_row(arm, outcome, band, cell, n, t_obs, t_rep) -> dict:
    sd = float(t_rep.std())
    return {
        "arm": arm,
        "outcome": outcome,
        "age_band": band,
        "cell": cell,
        "n": n,
        "observed_mean": float(t_obs),
        "predicted_mean": float(t_rep.mean()),
        "predicted_lo89": float(np.quantile(t_rep, CI[0])),
        "predicted_hi89": float(np.quantile(t_rep, CI[1])),
        "z": float((t_obs - t_rep.mean()) / sd) if sd > 0 else np.nan,
        "p_rep_ge_obs": float((t_rep >= t_obs).mean()),
    }


def _outcome_rows(arm: Arm):
    """``(outcome, frame rows, observed counts, obs index)`` for each likelihood."""
    cd = arm.idata.constant_data
    obs = arm.idata.observed_data
    for outcome, var, mask in (("understood", "y_u_obs", "obs_u_mask"), ("spoken", "y_s_obs", "obs_s_mask")):
        idx = np.flatnonzero(np.asarray(cd[mask].values))
        y = np.asarray(obs[var].values, dtype=float)
        yield outcome, var, idx, arm.frame.iloc[idx], y


def ppc_by_sex_table(arms: dict[str, Arm]) -> pd.DataFrame:
    """The stored posterior predictive by sex within age band.

    **Conditional on each child's fitted random effect**, because that is what
    the pipeline stores. A child-level covariate is largely absorbed by those
    effects, so this replicates most of the observed girl-boy difference even
    under the control and is a weak test of it; see
    :func:`marginal_ppc_by_sex_table` for the sharper one.
    """
    rows = []
    for name in ("control", "sex"):
        if name not in arms:
            continue
        arm = arms[name]
        ppc = arm.idata.posterior_predictive
        for outcome, var, idx, rows_df, y in _outcome_rows(arm):
            rep = ppc[var].values.reshape(-1, len(idx)).astype(float)
            rows.extend(_cell_rows(name, outcome, rows_df, y, rep))
    return pd.DataFrame(rows)


def marginal_ppc_by_sex_table(arms: dict[str, Arm], n_draws: int = 400, seed: int = 20260904) -> pd.DataFrame:
    """The new-child posterior predictive by sex within age band.

    Replicates every administration from the population curve at its own age
    (interpolated from the stored plot grid), its study's effect, and a **fresh**
    child effect pair drawn from the fitted between-child distribution -- one
    pair per child, so repeated administrations share it -- with the sex shift
    applied in the sex arm. Spoken counts keep the model's paired structure:
    conditional on the observed understood count where the likelihood was, on
    the 810-item inventory with mean ``p_u * q`` otherwise. Because the child
    effects are not fitted to the rows being predicted, a sex difference the
    model lacks shows up here as girls above and boys below the predictive,
    which is the check the note asked for.
    """
    rows = []
    for name in ("control", "sex"):
        if name not in arms:
            continue
        arm = arms[name]
        rng = np.random.default_rng(seed)
        pns = _predict_new_study()
        posterior = arm.idata.posterior.to_dataset()
        n_samples = posterior.sizes["chain"] * posterior.sizes["draw"]
        pick = np.sort(rng.choice(n_samples, size=n_draws, replace=False))
        # Subset before stacking: stacking the whole group materialises and
        # copies every posterior variable (about 0.5 GB at `test`, several
        # times that at `rep`) to pick a handful of them.
        needed = ["f_u_plot", "h_plot", "kappa_u_plot", "kappa_s_plot", "delta_u", "delta_q"]
        if "beta_sex_u" in posterior:
            needed += ["beta_sex_u", "beta_sex_q"]
        post = posterior[needed].stack(sample=("chain", "draw"))

        def take(var, post=post, pick=pick):
            return np.asarray(post[var].isel(sample=pick).transpose("sample", ...).values, dtype=float)

        cd = arm.idata.constant_data
        frame = arm.frame
        age = frame["age"].to_numpy(dtype=float)
        x_plot = np.asarray(cd["X_plot"].values, dtype=float)

        def on_obs(grid, age=age, x_plot=x_plot, pns=pns):  # (S, n_plot) -> (S, n_obs)
            return pns._interp_draws(x_plot, grid, age)

        f_u, h = on_obs(take("f_u_plot")), on_obs(take("h_plot"))
        kappa_u, kappa_s = on_obs(take("kappa_u_plot")), on_obs(take("kappa_s_plot"))
        study = np.asarray(cd["study_obs"].values).astype(int)
        subject = np.asarray(cd["subject_obs"].values).astype(int)
        eta_u = f_u + take("delta_u")[:, study]
        eta_q = h + take("delta_q")[:, study]
        # A fresh correlated child-effect pair per child, drawn by the helper
        # `predict_new_study.py` uses, which mirrors the engine's own
        # unseen-child construction; indices 0 and 2 are the two intercepts.
        child = pns.draw_child_params(
            posterior, arm.definition, "constant", pick, rng, (n_draws, subject.max() + 1)
        )
        eta_u = eta_u + child[..., 0][:, subject]
        eta_q = eta_q + child[..., 2][:, subject]
        if "beta_sex_u" in arm.idata.posterior:
            x_sex = np.asarray(cd["x_sex"].values, dtype=float)
            eta_u = eta_u + take("beta_sex_u")[:, None] * x_sex
            eta_q = eta_q + take("beta_sex_q")[:, None] * x_sex
        p_u, q = expit(eta_u), expit(eta_q)

        trials_s = np.asarray(cd["s_likelihood_n"].values).astype(int)
        is_cond = np.asarray(cd["s_is_conditional"].values).astype(bool)
        for outcome, _var, idx, rows_df, y in _outcome_rows(arm):
            if outcome == "understood":
                rep = pns._betabinom_draw(rng, N_TRIALS, p_u[:, idx], kappa_u[:, idx])
            else:
                mean = np.where(is_cond, q[:, idx], (p_u * q)[:, idx])
                rep = pns._betabinom_draw(rng, trials_s, mean, kappa_s[:, idx])
            rows.extend(_cell_rows(name, outcome, rows_df, y, rep.astype(float)))
    return pd.DataFrame(rows)


def loo_table(arms: dict[str, Arm]) -> pd.DataFrame:
    """Paired LOO difference, sex minus control, per outcome."""
    import arviz as az

    from vocab_growth.loo_reff import sampled_parameter_reff

    if not {"control", "sex"} <= set(arms):
        return pd.DataFrame()
    rows = []
    for outcome, var in (("understood", "y_u_obs"), ("spoken", "y_s_obs")):
        # A spoken row whose observed understood count is zero has a
        # structurally constant log-likelihood (n = 0 trials), which PSIS
        # rejects; the pipeline drops such points per fit
        # (`common.loo_dropping_degenerate`). Here the drop is applied to BOTH
        # arms from one shared mask, so the pointwise scores stay paired.
        das = {name: arms[name].idata.log_likelihood[var] for name in ("control", "sex")}
        sample_dims = [d for d in ("chain", "draw") if d in das["control"].dims]
        obs_dim = [d for d in das["control"].dims if d not in sample_dims][0]
        keep = np.ones(das["control"].sizes[obs_dim], dtype=bool)
        for da in das.values():
            keep &= (da.var(dim=sample_dims) > 1e-12).values
        n_dropped = int(keep.size - keep.sum())
        loos = {}
        reffs = {}
        for name in ("control", "sex"):
            arm = arms[name]
            # The project pins PSIS-LOO's relative efficiency to the sampled
            # parameters (`vocab_growth.loo_reff`, decision of 2026-08-23), so
            # this table is comparable with each arm's own `loo_summary.csv`.
            sampled = ((arm.manifest.get("artefacts") or {}).get("trace") or {}).get("sampled_parameters")
            reffs[name] = sampled_parameter_reff(arm.idata, names=sampled)
            source = arm.idata.copy(deep=False)
            source["log_likelihood"] = source["log_likelihood"].isel({obs_dim: keep})
            loos[name] = az.loo(source, var_name=var, pointwise=True, reff=reffs[name])
        li_c = np.asarray(loos["control"].elpd_i.values, dtype=float)
        li_s = np.asarray(loos["sex"].elpd_i.values, dtype=float)
        if li_c.shape != li_s.shape:
            raise RuntimeError(f"{var}: the two arms have different numbers of observations.")
        diff = li_s - li_c
        n = len(diff)
        ks = {name: np.asarray(l.pareto_k.values) for name, l in loos.items()}
        rows.append(
            {
                "outcome": outcome,
                "n_obs": n,
                "n_dropped_degenerate": n_dropped,
                "reff_control": float(reffs["control"]),
                "reff_sex": float(reffs["sex"]),
                "elpd_control": float(li_c.sum()),
                "elpd_sex": float(li_s.sum()),
                "elpd_diff_sex_minus_control": float(diff.sum()),
                "se_diff": float(np.sqrt(n * diff.var(ddof=1))),
                "pareto_k_gt_0.7_control": int((ks["control"] > 0.7).sum()),
                "pareto_k_gt_0.7_sex": int((ks["sex"] > 0.7).sum()),
            }
        )
    return pd.DataFrame(rows)


def compare(args) -> int:
    from vocab_growth import environment as env

    env.set_output_root(args.output_dir)
    arms: dict[str, Arm] = {}
    for arm in ("control", "sex", "full"):
        try:
            arms[arm] = Arm(args.output_dir, arm, args.sigma)
        except FileNotFoundError as exc:
            print(f"[vg20_sex_arm] skipping {arm}: {exc}")
    if "control" not in arms or "sex" not in arms:
        raise SystemExit("compare needs at least the control and sex arms fitted.")

    out_dir = os.path.join(args.output_dir, "comparisons", "sex-effect")
    os.makedirs(out_dir, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    tables = {
        "vg20_sex_arm_parameters.csv": parameters_table(arms),
        "vg20_sex_arm_trajectories.csv": trajectories_table(arms),
        "vg20_sex_arm_words.csv": words_table(arms["sex"]),
        "vg20_sex_arm_ppc_by_sex.csv": ppc_by_sex_table(arms),
        "vg20_sex_arm_ppc_marginal_by_sex.csv": marginal_ppc_by_sex_table(arms),
        "vg20_sex_arm_loo.csv": loo_table(arms),
        "vg20_sex_arm_reference_descriptive.csv": reference_table(args.output_dir),
    }
    for filename, table in tables.items():
        if table.empty:
            continue
        table.to_csv(os.path.join(out_dir, filename), index=False)
        print(f"\n=== {filename}")
        print(table.round(3).to_string(index=False))
    for name, arm in arms.items():
        print(f"\n{name}: rows={arm.manifest['data']['rows']} children={arm.manifest['data']['children']} {arm.convergence()}")
    print(f"\nWritten to {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True, help="Output root; must not be the project's.")
    ap.add_argument("--sigma", type=float, default=0.5, help="Prior SD of the sex coefficients (default 0.5).")
    sub = ap.add_subparsers(dest="command", required=True)
    p_fit = sub.add_parser("fit", help="Fit one arm.")
    p_fit.add_argument("arm", choices=sorted(ARMS))
    p_fit.add_argument("--config", default="test")
    p_fit.set_defaults(func=fit)
    p_cmp = sub.add_parser("compare", help="Compare the finished arms.")
    p_cmp.set_defaults(func=compare)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
