# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Wave-forward sequential validation for the cross-lag models (issue #242 item 5).

VG16's understood PSIS-LOO is suppressed, and correctly so: the lag predictor for
a child's later wave embeds that child's *earlier* understood count, so leaving
one administration out of the likelihood does not leave it out of the model, and
Pareto-k cannot see the leak. The registered replacement is a grouped
forward-chaining score, and this is it.

What is held out, and why
-------------------------

Children are split into K folds. For fold ``k``, every row of a fold-``k`` child
**after their first administration wave** is removed from the likelihood; their
first wave stays in training, as does everything belonging to every other child.
The held-out later waves are then scored.

That conditioning is the point. The lag predictor for a held-out wave is built
from the child's earlier wave, which is training data, so the score is an honest
one-step-ahead prediction: *given what this child was at their first visit and
what other children did, how well is their next visit predicted?* Holding out the
whole child instead (``--holdout-unit child``) makes the lag source a row the
model never saw, which is a different and stricter question -- the child's random
effect is then drawn from its prior -- and it is offered because #289 task 3.8
asks for the choice to be made explicitly rather than assumed.

**First waves are never scored, under either unit.** A first wave carries
``has_lag = 0``, so ``x_lag`` is zero and the cross-lag term vanishes from its
likelihood: the two models being compared are numerically identical there, and
including those rows would dilute the comparison with rows that cannot inform it.
Excluding them is therefore not only leak avoidance, it is the only informative
scored set.

What is compared
----------------

VG16 against **itself with the coefficient removed** (``use_cross_lag=False``),
on the identical prepared frame and the identical folds. Comparing against VG10
would confound the coefficient with every other field the two definitions do not
share; comparing against a definition that differs in one boolean does not. The
control is fitted in a scratch directory and is never a model of record.

The headline is the **spoken** elpd difference. The lag term enters the graph only
through the production-ratio logit, so a held-out row's understood density is the
same under both models up to sampling noise; reporting it as part of the total
would bury the signal under variation the comparison is not about. Both are
written.

Every fold fit is screened by the canonical diagnostics scan, and a model with
any failed fold is flagged rather than dropped. The fit itself is
``vocab_growth.fold_fits.fit_holdout_fold``, shared with ``kfold_loso.py``: the
two scripts hold different things out and score different units, but the fit
between those decisions is the same one, and the hand copy this script started
with read the energy verdict from the wrong key.

Outputs
-------

- ``output/comparisons/wave_forward_row_elpds.csv`` -- per scored row, per model
- ``output/comparisons/wave_forward_summary.csv`` -- per-model totals
- ``output/comparisons/wave_forward_compare.csv`` -- the paired difference with
  its standard error
- ``output/comparisons/wave_forward_fits.csv`` -- per (model, fold) convergence

Usage
-----

    uv run python scripts/wave_forward_score.py [--model vg16] [--folds 5]
        [--config test] [--holdout-unit later-waves|child] [--suffix ...]

Like ``kfold_loso.py`` this fits its own folds rather than reading a model of
record, so its comparison-manifest entry records the raw-data fingerprint rather
than a contributing fit.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import time
from dataclasses import dataclass

import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import betabinom

from vocab_growth import environment as env
from vocab_growth.comparisons_provenance import (
    ComparisonOutputs,
    write_comparison_manifest,
)
from vocab_growth.fit_artifacts import source_data_hash
from vocab_growth.fold_fits import fit_holdout_fold, fold_gate_fields
from vocab_growth.models.common_bivariate_re import (
    build_bivariate_re_analysis_frame,
)
from vocab_growth.models.cross_lag import (
    iter_subject_age_waves,
    prev_wave_lag_for_frame,
)
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.models.likelihood_utils import (
    SPOKEN_FALLBACK_PRODUCT,
    nested_outcome_spec,
)

OUT_DIR = env.comparisons_output_dir()
TMP_DIR = os.path.join(env.output_root(), "wave_forward_tmp")

#: Models this script can score. The requirement is a cross-lag: without one
#: there is no coefficient to remove and the two arms are the same model.
CROSS_LAG_MODELS = tuple(
    key for key, d in MODEL_REGISTRY.items() if getattr(d, "use_cross_lag", False)
)

CONTROL_SUFFIX = "-nolag"


# ============================================================
# Waves and folds
# ============================================================


def wave_index(analysis_df: pd.DataFrame) -> np.ndarray:
    """0 for each child's first administration wave, 1 for the next, and so on.

    Built from :func:`vocab_growth.models.cross_lag.iter_subject_age_waves`, the
    same grouping the lag itself uses, so "a later wave" here and "a row with a
    prior-wave source" there cannot drift apart. Every row at one recorded age
    takes the same index: a child measured on two forms on one day has one wave,
    not two, which is the wave definition issue #242 settled.
    """
    subject = np.asarray(analysis_df["subject_code"], dtype=int)
    age = np.asarray(analysis_df["age"], dtype=float)
    index = np.zeros(len(analysis_df), dtype=int)
    current_subject: int | None = None
    counter = 0
    for rows in iter_subject_age_waves(subject, age):
        s = int(subject[rows[0]])
        if s != current_subject:
            current_subject = s
            counter = 0
        index[rows] = counter
        counter += 1
    return index


def stratified_subject_folds(
    analysis_df: pd.DataFrame, K: int, seed: int = 47
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Assign each subject to a fold, stratified by study and wave count.

    Stratifying on the number of *waves* rather than the number of rows is what
    this script needs and ``kfold_loso.py`` does not: a child with one wave
    contributes nothing to score however many forms they carry at it, so folds
    balanced on row count could still put most of the scorable children in one
    fold.
    """
    waves = wave_index(analysis_df)
    subj = (
        analysis_df.assign(_wave=waves)
        .groupby("subject_code")
        .agg(study_code=("study_code", "first"), n_waves=("_wave", "max"))
        .reset_index()
    )
    subj["n_waves"] = subj["n_waves"] + 1

    def _bin(n: int) -> str:
        return str(n) if n < 4 else "4+"

    subj["stratum"] = (
        subj["study_code"].astype(str) + "_" + subj["n_waves"].apply(_bin)
    )
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(subj), dtype=int)
    for _stratum, group in subj.groupby("stratum"):
        idxs = np.asarray(group.index.to_numpy(), dtype=np.int64).copy()
        rng.shuffle(idxs)
        for k, i in enumerate(idxs):
            fold_of[i] = k % K
    subj["fold"] = fold_of
    folds = [subj.loc[subj["fold"] == k, "subject_code"].to_numpy() for k in range(K)]
    return folds, subj


def holdout_mask(
    analysis_df: pd.DataFrame,
    fold_subjects: np.ndarray,
    *,
    unit: str,
) -> np.ndarray:
    """Which rows leave the likelihood for this fold.

    ``later-waves`` keeps each fold child's first wave in training, so the lag
    source for every scored row is data the model saw. ``child`` removes the
    whole child, which makes the lag source a row the model never saw and the
    child's random effect a prior draw.
    """
    in_fold = analysis_df["subject_code"].isin(fold_subjects).to_numpy()
    if unit == "child":
        return in_fold
    if unit != "later-waves":
        raise ValueError(f"unknown holdout unit {unit!r}")
    return in_fold & (wave_index(analysis_df) > 0)


def scored_rows(analysis_df: pd.DataFrame, fold_subjects: np.ndarray) -> np.ndarray:
    """The rows this fold contributes to the comparison, as positional indices.

    A fold child's later waves, and only those. First waves are excluded under
    both holdout units because ``x_lag`` is zero there, so the two arms assign
    them the identical density and scoring them adds noise without signal.
    """
    in_fold = analysis_df["subject_code"].isin(fold_subjects).to_numpy()
    return np.flatnonzero(in_fold & (wave_index(analysis_df) > 0))


def lag_source(analysis_df: pd.DataFrame, definition) -> tuple[np.ndarray, np.ndarray]:
    """``(has_lag, source_row)`` from the model's own lag rule.

    A later wave is not automatically a lagged wave: if every earlier wave of
    that child lacks a comprehension count -- and 391 of the pool's 448
    comprehension-less rows sit on three forms that never record one -- then
    ``has_lag`` is 0 and the cross-lag term drops out of that row's likelihood.
    The two arms then assign the row the identical density, so it contributes
    exactly nothing to the elpd difference while still changing the paired
    standard error.

    Read from ``prev_wave_lag_for_frame`` rather than reconstructed, so a variant
    that moves the gap ceiling, the zero handling or the same-form restriction
    moves this too -- and so ``source_row`` is the row the model actually reads,
    which is what decides whether the predictor stands on training data.
    """
    prev_idx, has_lag_f, _logit = prev_wave_lag_for_frame(
        analysis_df, definition.n_trials, definition
    )
    return np.asarray(has_lag_f, dtype=float) > 0, np.asarray(prev_idx, dtype=int)


def source_in_training(
    has_lag_flags: np.ndarray, source_row: np.ndarray, holdout: np.ndarray
) -> np.ndarray:
    """Whether each row's lag source is a row the likelihood saw.

    Under ``later-waves`` a second wave's source is the child's first, which
    stays in training; a *third* wave's source is the second, which does not. So
    the scored set splits in two, and the split matters for what may be claimed.
    A row whose source is in training is predicted from data the model was fitted
    to, which is the ordinary out-of-sample statement. A row whose source is held
    out is still an honest forward-chaining prediction -- its own outcome appears
    nowhere in its own predictor -- but it conditions on a row the likelihood
    never saw, so it is not "trained on everything except the scored rows".

    Reported as a column rather than settled in prose, and the headline
    restriction takes the strict side.
    """
    return has_lag_flags & ~np.asarray(holdout, dtype=bool)[source_row]


# ============================================================
# Per-fold fit
# ============================================================


@dataclass
class FoldFitRecord:
    arm: str
    fold: int
    n_holdout_rows: int
    n_scored_rows: int
    wall_seconds: float
    passed: bool
    max_rhat: float
    min_ess: float
    divergences: int
    bfmi_ok: bool


# ============================================================
# Row-level held-out predictive density
# ============================================================


def row_elpds(
    frame: pd.DataFrame,
    trace: xr.DataTree,
    rows: np.ndarray,
    n_trials: int,
    lagged: np.ndarray,
    clean: np.ndarray,
) -> pd.DataFrame:
    """Marginal predictive log-density of each scored row, by outcome.

    One row at a time rather than one child at a time (``kfold_loso``'s unit),
    because the scored set here is a subset of a child's rows and the paired
    comparison needs the two arms aligned on the same rows.

    The understood and spoken densities are kept apart. The cross-lag enters only
    the production-ratio logit, so the understood column is a control that should
    show no difference between the arms; summing them would hide that.
    """
    p_u_obs = trace.posterior["p_u_obs"].values
    p_s_obs = trace.posterior["p_s_obs"].values
    q_obs = trace.posterior["q_obs"].values
    kappa_u_obs = trace.posterior["kappa_u_obs"].values
    kappa_s_obs = trace.posterior["kappa_s_obs"].values
    n_chain, n_draw, _ = p_u_obs.shape
    log_NK = math.log(n_chain * n_draw)

    spec = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=n_trials
    )
    spoken_observed = np.full(len(frame), -1, dtype=int)
    spoken_trials = np.full(len(frame), n_trials, dtype=int)
    spoken_is_conditional = np.zeros(len(frame), dtype=bool)
    spoken_observed[spec.indices] = spec.observed
    spoken_trials[spec.indices] = spec.trials
    spoken_is_conditional[spec.indices] = spec.is_conditional

    records = []
    for idx in rows:
        row = frame.iloc[idx]
        elpd_u = float("nan")
        elpd_s = float("nan")
        if pd.notna(row["understood"]):
            y = int(row["understood"])
            p = np.clip(p_u_obs[:, :, idx], 1e-12, 1 - 1e-12)
            k = kappa_u_obs[:, :, idx]
            ll = betabinom.logpmf(y, n_trials, p * k, (1 - p) * k)
            elpd_u = float(logsumexp(ll.ravel()) - log_NK)
        if pd.notna(row["spoken"]):
            y = spoken_observed[idx]
            if spoken_is_conditional[idx]:
                p = np.clip(q_obs[:, :, idx], 1e-12, 1 - 1e-12)
            else:
                p = np.clip(p_s_obs[:, :, idx], 1e-12, 1 - 1e-12)
            k = kappa_s_obs[:, :, idx]
            ll = betabinom.logpmf(y, spoken_trials[idx], p * k, (1 - p) * k)
            elpd_s = float(logsumexp(ll.ravel()) - log_NK)
        records.append(
            {
                "row": int(idx),
                "subject_code": int(row["subject_code"]),
                "age_months": float(row["age"]),
                "has_lag": bool(lagged[idx]),
                "source_in_training": bool(clean[idx]),
                "elpd_understood": elpd_u,
                "elpd_spoken": elpd_s,
                "spoken_branch": (
                    ""
                    if pd.isna(row["spoken"])
                    else ("conditional" if spoken_is_conditional[idx] else "marginal")
                ),
            }
        )
    return pd.DataFrame(records)


# ============================================================
# Comparison
# ============================================================


#: The three row sets the comparison is reported on, strictest first.
#:
#: ``lagged-from-training`` is the headline: the coefficient can move these rows,
#: and their predictor stands entirely on data the likelihood saw. ``lagged``
#: adds the rows whose lag source was itself held out -- still forward chaining,
#: but conditioning on an unseen row. ``all-later-waves`` adds the rows with no
#: lag at all, which enter both arms identically: they leave the total untouched
#: and shrink the per-row spread the standard error is built from, so quoting
#: that standard error would make the comparison look more precise than its
#: evidence.
RESTRICTIONS = ("lagged-from-training", "lagged", "all-later-waves")


def _restriction_mask(wide: pd.DataFrame, restriction: str) -> np.ndarray:
    if restriction == "lagged-from-training":
        return wide["source_in_training"].to_numpy(dtype=bool)
    if restriction == "lagged":
        return wide["has_lag"].to_numpy(dtype=bool)
    if restriction == "all-later-waves":
        return np.ones(len(wide), dtype=bool)
    raise ValueError(f"unknown restriction {restriction!r}")


def paired_difference(wide: pd.DataFrame, column: str, *, restriction: str) -> dict:
    """The cross-lag arm minus the control, paired on the scored rows.

    Paired because both arms score the identical rows: differencing two
    independent totals throws away the correlation between them and inflates the
    standard error, which is the defect #289 task 3.2 records for the VG20/VG22
    comparison. Rows either arm could not score are dropped from both.

    See :data:`RESTRICTIONS` for what each row set contains and why the strictest
    is the one to quote.
    """
    keep = _restriction_mask(wide, restriction)
    lag = wide[f"{column}_lag"].to_numpy()
    ctl = wide[f"{column}_control"].to_numpy()
    usable = keep & np.isfinite(lag) & np.isfinite(ctl)
    diff = lag[usable] - ctl[usable]
    n = int(usable.sum())
    row = {"outcome": column, "rows_scored": restriction, "n_rows": n}
    if n < 2:
        return {**row, "elpd_diff": float("nan"), "se": float("nan"),
                "mean_per_row": float("nan")}
    return {
        **row,
        "elpd_diff": float(diff.sum()),
        "se": float(np.sqrt(n) * np.std(diff, ddof=1)),
        "mean_per_row": float(diff.mean()),
    }


# ============================================================
# Driver
# ============================================================


def control_definition(definition):
    """``definition`` with the cross-lag removed and nothing else changed."""
    control = dataclasses.replace(
        definition,
        use_cross_lag=False,
        config_name=f"{definition.config_name}{CONTROL_SUFFIX}",
    )
    changed = {
        f.name
        for f in dataclasses.fields(definition)
        if getattr(definition, f.name) != getattr(control, f.name)
    }
    # The control exists to isolate one coefficient. If `replace` ever starts
    # moving anything else -- a derived field, a renamed flag -- the comparison
    # stops being about the cross-lag and this is where it should stop.
    assert changed == {"use_cross_lag", "config_name"}, changed
    return control


def main(
    model_key: str,
    K: int,
    sampling_config_name: str,
    unit: str,
    suffix: str,
) -> None:
    definition = MODEL_REGISTRY[model_key]
    if not getattr(definition, "use_cross_lag", False):
        raise SystemExit(
            f"{model_key} carries no cross-lag; there is no coefficient to "
            f"remove. Models with one: {', '.join(CROSS_LAG_MODELS)}."
        )
    if definition.spoken_fallback != SPOKEN_FALLBACK_PRODUCT:
        raise NotImplementedError(
            "row_elpds implements only the "
            f"{SPOKEN_FALLBACK_PRODUCT!r} spoken fallback; {model_key} uses "
            f"{definition.spoken_fallback!r} and would be scored under the "
            "wrong likelihood."
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    written = ComparisonOutputs(OUT_DIR)

    frame, _meta = build_bivariate_re_analysis_frame(definition)
    waves = wave_index(frame)
    n_scorable = int((waves > 0).sum())
    print(
        f"{model_key.upper()}: {len(frame)} rows / "
        f"{frame['subject_code'].nunique()} children; "
        f"{n_scorable} rows in a later wave, "
        f"{frame.loc[waves > 0, 'subject_code'].nunique()} children contribute one"
    )
    if n_scorable == 0:
        raise SystemExit("No child has a second wave; there is nothing to score.")

    lagged, source_row = lag_source(frame, definition)
    print(
        f"  of the later-wave rows, {int((lagged & (waves > 0)).sum())} carry a "
        "non-zero lag; the rest enter both arms identically"
    )

    arms = {"lag": definition, "control": control_definition(definition)}
    folds, _subj = stratified_subject_folds(frame, K=K)
    sampling_cfg = sampling.get_sampling_configuration(sampling_config_name)

    per_arm: dict[str, list[pd.DataFrame]] = {name: [] for name in arms}
    fit_records: list[FoldFitRecord] = []

    for k, fold_subjects in enumerate(folds):
        rows = scored_rows(frame, fold_subjects)
        mask = holdout_mask(frame, fold_subjects, unit=unit)
        print(
            f"\n=== Fold {k}/{K} — {len(fold_subjects)} children, "
            f"{int(mask.sum())} rows out of the likelihood, "
            f"{len(rows)} scored ==="
        )
        if len(rows) == 0:
            print("  no later waves in this fold; skipped")
            continue
        for arm, arm_definition in arms.items():
            label = f"{model_key}_{arm}_fold{k}"
            started = time.perf_counter()
            marked = frame.copy()
            marked["holdout"] = mask
            clean = source_in_training(lagged, source_row, mask)
            trace, gate = fit_holdout_fold(
                arm_definition,
                marked,
                sampling_cfg,
                label=label,
                tmp_root=TMP_DIR,
                name_prefix="WAVEFWD",
            )
            scores = row_elpds(
                marked, trace, rows, definition.n_trials, lagged, clean
            )
            scores.insert(0, "fold", k)
            per_arm[arm].append(scores)
            elapsed = time.perf_counter() - started
            record = FoldFitRecord(
                arm=arm,
                fold=k,
                n_holdout_rows=int(mask.sum()),
                n_scored_rows=len(rows),
                wall_seconds=elapsed,
                **fold_gate_fields(gate),
            )
            fit_records.append(record)
            print(
                f"  {arm:8s} {elapsed:6.1f}s  gate "
                f"{'PASS' if record.passed else 'FAIL'}  "
                f"max R-hat {record.max_rhat:.4f}  div {record.divergences}"
            )

    if not fit_records:
        raise SystemExit("No fold produced a scorable row.")

    long = pd.concat(
        [df.assign(arm=arm) for arm, frames in per_arm.items() for df in frames],
        ignore_index=True,
    )
    wide = long.pivot_table(
        index=[
            "fold",
            "row",
            "subject_code",
            "age_months",
            "has_lag",
            "source_in_training",
            "spoken_branch",
        ],
        columns="arm",
        values=["elpd_understood", "elpd_spoken"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    fits = pd.DataFrame([dataclasses.asdict(r) for r in fit_records])
    converged = bool(fits["passed"].all())
    if not converged:
        print("\n" + "!" * 74)
        print("!!! CONVERGENCE WARNING: at least one fold fit failed the gate.")
        print("!!! The elpd difference below must not be interpreted.")
        print("!" * 74)

    comparison = pd.DataFrame(
        [
            paired_difference(wide, column, restriction=restriction)
            for restriction in RESTRICTIONS
            for column in ("elpd_spoken", "elpd_understood")
        ]
    )
    comparison["all_folds_converged"] = converged
    comparison["holdout_unit"] = unit
    comparison["config"] = sampling_config_name

    summary = (
        long.groupby("arm")[["elpd_understood", "elpd_spoken"]]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(part for part in col if part).strip("_") for col in summary.columns
    ]
    summary["all_folds_converged"] = converged

    paths = {
        f"wave_forward_row_elpds{suffix}.csv": wide,
        f"wave_forward_summary{suffix}.csv": summary,
        f"wave_forward_compare{suffix}.csv": comparison,
        f"wave_forward_fits{suffix}.csv": fits,
    }
    for name, table in paths.items():
        table.to_csv(os.path.join(OUT_DIR, name), index=False)

    spoken = comparison.iloc[0]
    print(
        f"\nSpoken elpd on {spoken['rows_scored']} rows, cross-lag minus "
        f"control: {spoken['elpd_diff']:+.2f} (SE {spoken['se']:.2f}) "
        f"over {int(spoken['n_rows'])} rows"
    )
    print(f"Written to {OUT_DIR}")

    # This script fits its own folds and reads no model of record, so its
    # provenance is the raw data it prepared them from -- the same claim
    # `kfold_loso.py` records.
    write_comparison_manifest(
        OUT_DIR,
        script="wave_forward_score.py",
        contributing={},
        outputs=written.written(),
        source_data_hash=source_data_hash(env.DATA_DIR),
        arguments=[
            model_key,
            f"K={K}",
            f"config={sampling_config_name}",
            f"holdout_unit={unit}",
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vg16", choices=sorted(CROSS_LAG_MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--config", default="test")
    parser.add_argument(
        "--holdout-unit",
        default="later-waves",
        choices=("later-waves", "child"),
        help=(
            "later-waves keeps each fold child's first administration in "
            "training, so the lag source for every scored row is data the model "
            "saw; child removes the whole child, making the lag source unseen "
            "and the child effect a prior draw."
        ),
    )
    parser.add_argument("--suffix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        model_key=args.model,
        K=args.folds,
        sampling_config_name=args.config,
        unit=args.holdout_unit,
        suffix=args.suffix,
    )
