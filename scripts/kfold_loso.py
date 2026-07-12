# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
K-fold leave-one-subject-out (LOSO) gold-standard comparison of VG07, VG08, VG09.

Splits the 510 unique DS subjects into K folds (stratified by study and
observation count), then for each (model, fold) pair refits the model with
the fold's subjects' observations excluded from the likelihood (but kept
in `obs_id` space so f_u_obs, h_obs and so on are computed at their ages).
Subject REs for the held-out subjects are then drawn from their priors
during MCMC, which means the trace's `p_u_obs` and `p_s_obs` at held-out
observations are already the marginal posterior predictive probabilities
— no extra Monte Carlo integration is needed.

For each held-out subject the predictive log-density of their observed
counts is

    log p(y_subject | training_data, model)
        = logsumexp_{c, d} log p(y_subject | params_{c,d}, RE_{c,d})
          - log(n_chain * n_draw)

where the inner log p is the Beta-Binomial log-pmf summed over the
subject's held-out (understood, spoken) observations, and the outer
logsumexp marginalises over the joint posterior over hyperparameters and
held-out REs.

Outputs:

- `output/comparisons/kfold_loso_subject_elpds.csv`
- `output/comparisons/kfold_loso_summary.csv`
- `output/comparisons/kfold_loso_compare.csv`
"""

from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass

import dse_research_utils.statistics.models.data as model_data
import dse_research_utils.statistics.models.reporting as reporting
import dse_research_utils.statistics.models.sampling as sampling
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import betabinom

import vocab_growth.data_utils as data_utils
from vocab_growth import environment as env
from vocab_growth.models.common import ModelFitContext
from vocab_growth.models.common_bivariate import (
    configure_bivariate_priors,
    sample,
)
from vocab_growth.models.common_bivariate_re import build_model_re
from vocab_growth.models.definitions import VG07, VG08, VG09, BivariateModelDefinition

# Derive the Beta-Binomial trial count from the model definitions rather than a
# literal (issue #131). All three compared models share the common 810-item
# reference scale; assert they agree so a future divergence surfaces here.
_n_trials_set = {VG07.n_trials, VG08.n_trials, VG09.n_trials}
assert len(_n_trials_set) == 1, (
    f"VG07/VG08/VG09 disagree on n_trials ({sorted(_n_trials_set)}); "
    "kfold_loso assumes a single common trial count."
)
N_TRIALS = _n_trials_set.pop()
OUT_DIR = env.comparisons_output_dir()
KFOLD_TMP_DIR = os.path.join(env.output_root(), "kfold_tmp")


@dataclass(frozen=True)
class FoldFitRecord:
    model_short: str
    fold: int
    n_holdout_subjects: int
    n_holdout_obs_u: int
    n_holdout_obs_s: int
    wall_seconds: float


# ============================================================
# Data + folds
# ============================================================


def load_analysis_frame() -> pd.DataFrame:
    df = data_utils.load_combined_data()
    analysis_df = df[["age", "understood", "spoken", "study", "subject_id"]].copy()
    analysis_df = analysis_df.dropna(subset=["age"])
    has_u = analysis_df["understood"].notna()
    has_s = analysis_df["spoken"].notna()
    analysis_df = analysis_df[has_u | has_s].reset_index(drop=True)

    unique_studies = sorted(analysis_df["study"].unique())
    study_map = {s: i for i, s in enumerate(unique_studies)}
    analysis_df["study_code"] = analysis_df["study"].map(study_map).astype(int)

    subj_keys = (
        analysis_df["study"].astype(str) + "::" + analysis_df["subject_id"].astype(str)
    )
    unique_subjects = sorted(subj_keys.unique())
    subject_map = {s: i for i, s in enumerate(unique_subjects)}
    analysis_df["subject_code"] = subj_keys.map(subject_map).astype(int)
    return analysis_df


def stratified_subject_folds(
    analysis_df: pd.DataFrame, K: int = 5, seed: int = 47
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Assign each subject to one of K folds, stratified by (study, n_obs_bin)."""
    subj = analysis_df.groupby("subject_code").agg(
        study_code=("study_code", "first"),
        n_obs=("age", "size"),
    ).reset_index()
    # Bin observation counts.
    def _bin(n):
        if n == 1:
            return "1"
        if n == 2:
            return "2"
        if n == 3:
            return "3"
        return "4+"

    subj["n_obs_bin"] = subj["n_obs"].apply(_bin)
    subj["stratum"] = (
        subj["study_code"].astype(str) + "_" + subj["n_obs_bin"]
    )

    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(subj), dtype=int)
    for _stratum, group in subj.groupby("stratum"):
        idxs = np.asarray(group.index.to_numpy(), dtype=np.int64).copy()
        rng.shuffle(idxs)
        for k, i in enumerate(idxs):
            fold_of[i] = k % K
    subj["fold"] = fold_of
    folds = [
        subj.loc[subj["fold"] == k, "subject_code"].to_numpy() for k in range(K)
    ]
    return folds, subj


# ============================================================
# Per-fold fit (minimal pipeline)
# ============================================================


def fit_fold(
    definition: BivariateModelDefinition,
    analysis_df_with_holdout: pd.DataFrame,
    sampling_cfg: sampling.SamplingConfiguration,
    label: str,
) -> tuple[xr.DataTree, int]:
    """Run prepare → priors → build → sample on a holdout-marked analysis frame."""
    n = len(analysis_df_with_holdout)
    has_u = analysis_df_with_holdout["understood"].notna().values
    bmd = model_data.BinomialModelData(
        X_obs=np.asarray(analysis_df_with_holdout["age"], dtype=float).reshape(-1, 1),
        y_obs=np.where(
            has_u,
            analysis_df_with_holdout["understood"].fillna(0).astype(int),
            0,
        ).astype(int),
        n_trials=definition.n_trials,
    )

    reporting_cfg = reporting.ReportingConfiguration(
        model_name=f"KFOLD-{label}",
        config_name=definition.config_name,
        output_root_dir=KFOLD_TMP_DIR,
        ci_prob=0.90,
        interval_kind="hdi",
    )
    if os.path.exists(reporting_cfg.output_dir):
        shutil.rmtree(reporting_cfg.output_dir)
    os.makedirs(reporting_cfg.output_dir, exist_ok=True)

    context: ModelFitContext = ModelFitContext(
        reporting=reporting_cfg,
        sampling=sampling_cfg,
    )
    context.set_model_data(bmd, analysis_df_with_holdout)
    configure_bivariate_priors(context, definition)
    build_model_re(context, definition)
    sample(context)

    return context.trace, n


# ============================================================
# Held-out predictive log-density
# ============================================================


def holdout_subject_elpds(
    analysis_df: pd.DataFrame,
    trace: xr.DataTree,
    holdout_subject_codes: np.ndarray,
) -> dict[int, float]:
    """Marginal predictive log-density per held-out subject."""
    p_u_obs = trace.posterior["p_u_obs"].values
    p_s_obs = trace.posterior["p_s_obs"].values
    kappa_u_obs = trace.posterior["kappa_u_obs"].values
    kappa_s_obs = trace.posterior["kappa_s_obs"].values

    n_chain, n_draw, _ = p_u_obs.shape
    log_NK = math.log(n_chain * n_draw)
    elpd: dict[int, float] = {}

    holdout_set = set(int(s) for s in holdout_subject_codes)
    for s_code in holdout_set:
        rows = analysis_df[analysis_df["subject_code"] == s_code]
        log_lik = np.zeros((n_chain, n_draw), dtype=np.float64)
        for idx, row in rows.iterrows():
            if pd.notna(row["understood"]):
                y = int(row["understood"])
                p = np.clip(p_u_obs[:, :, idx], 1e-12, 1 - 1e-12)
                k = kappa_u_obs[:, :, idx]
                log_lik += betabinom.logpmf(y, N_TRIALS, p * k, (1 - p) * k)
            if pd.notna(row["spoken"]):
                y = int(row["spoken"])
                p = np.clip(p_s_obs[:, :, idx], 1e-12, 1 - 1e-12)
                k = kappa_s_obs[:, :, idx]
                log_lik += betabinom.logpmf(y, N_TRIALS, p * k, (1 - p) * k)
        elpd[s_code] = float(logsumexp(log_lik.ravel()) - log_NK)
    return elpd


# ============================================================
# Driver
# ============================================================


SPECS = [("VG07", VG07), ("VG08", VG08), ("VG09", VG09)]


def main(K: int = 5, sampling_config_name: str = "test") -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reloading DS analysis frame …", flush=True)
    analysis_df = load_analysis_frame()
    print(
        f"  {len(analysis_df)} observations / "
        f"{analysis_df['subject_code'].nunique()} subjects"
    )

    print(f"\nBuilding {K} stratified folds …", flush=True)
    folds, _subj_table = stratified_subject_folds(analysis_df, K=K)
    for k, fold in enumerate(folds):
        n_obs_in = analysis_df["subject_code"].isin(fold).sum()
        print(f"  fold {k}: {len(fold)} subjects, {n_obs_in} observations")

    sampling_cfg = sampling.get_sampling_configuration(sampling_config_name)

    # Per-model, per-subject elpd accumulator.
    elpd_per_model: dict[str, dict[int, float]] = {
        short: {} for short, _ in SPECS
    }
    fit_records: list[FoldFitRecord] = []

    for k, fold_subjects in enumerate(folds):
        print(f"\n=== Fold {k}/{K} — {len(fold_subjects)} held-out subjects ===")
        for short, definition in SPECS:
            label = f"{short}_fold{k}"
            print(f"  fitting {short} …", flush=True)
            started = time.perf_counter()

            df_with_holdout = analysis_df.copy()
            df_with_holdout["holdout"] = (
                df_with_holdout["subject_code"].isin(fold_subjects)
            )
            trace, _ = fit_fold(definition, df_with_holdout, sampling_cfg, label)
            elpds = holdout_subject_elpds(df_with_holdout, trace, fold_subjects)
            elpd_per_model[short].update(elpds)

            holdout_mask = df_with_holdout["holdout"].to_numpy()
            n_u_holdout = int(
                ((df_with_holdout["understood"].notna()) & holdout_mask).sum()
            )
            n_s_holdout = int(
                ((df_with_holdout["spoken"].notna()) & holdout_mask).sum()
            )
            elapsed = time.perf_counter() - started
            fit_records.append(
                FoldFitRecord(
                    model_short=short,
                    fold=k,
                    n_holdout_subjects=len(fold_subjects),
                    n_holdout_obs_u=n_u_holdout,
                    n_holdout_obs_s=n_s_holdout,
                    wall_seconds=elapsed,
                )
            )
            print(
                f"    fold {k} / {short} done in {elapsed:.1f}s — "
                f"{len(elpds)} holdout subjects evaluated"
            )

    # Build a per-subject elpd table.
    rows = []
    for short in elpd_per_model:
        for s_code, e in elpd_per_model[short].items():
            rows.append(
                {"model": short, "subject_code": s_code, "elpd": e}
            )
    elpd_df = pd.DataFrame(rows).pivot_table(
        index="subject_code", columns="model", values="elpd"
    )
    elpd_df.to_csv(os.path.join(OUT_DIR, "kfold_loso_subject_elpds.csv"))

    # Per-model totals + paired SE.
    print("\n=== K-fold LOSO summary ===")
    summary_rows = []
    for short, _ in SPECS:
        e = elpd_df[short].dropna().to_numpy()
        n = len(e)
        # SE of total elpd = sqrt(n * var of per-subject elpd).
        se = float(np.sqrt(n) * np.std(e, ddof=1))
        summary_rows.append(
            {
                "model": short,
                "elpd_loso": float(e.sum()),
                "se": se,
                "n_subjects": n,
                "mean_elpd_per_subject": float(e.mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(OUT_DIR, "kfold_loso_summary.csv"), index=False
    )
    print(summary_df.to_string(index=False))

    # Pairwise comparison with paired SE.
    print("\n=== Pairwise comparisons (paired-difference SE) ===")
    pair_rows = []
    for i, (sa, _) in enumerate(SPECS):
        for j, (sb, _) in enumerate(SPECS):
            if i >= j:
                continue
            common = elpd_df[[sa, sb]].dropna()
            diff = common[sb] - common[sa]
            elpd_diff = float(diff.sum())
            dse = float(np.sqrt(len(diff)) * np.std(diff, ddof=1))
            pair_rows.append(
                {
                    "model_a": sa,
                    "model_b": sb,
                    "elpd_diff_b_minus_a": elpd_diff,
                    "dse_paired": dse,
                    "diff_over_dse": elpd_diff / dse if dse > 0 else float("nan"),
                    "n_subjects": len(diff),
                }
            )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(
        os.path.join(OUT_DIR, "kfold_loso_compare.csv"), index=False
    )
    print(pair_df.to_string(index=False))

    fit_df = pd.DataFrame([r.__dict__ for r in fit_records])
    fit_df.to_csv(os.path.join(OUT_DIR, "kfold_loso_fits.csv"), index=False)
    print("\n=== Fit timings ===")
    print(fit_df.to_string(index=False))

    total_wall = fit_df["wall_seconds"].sum()
    print(f"\nTotal fit wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")


if __name__ == "__main__":
    main()
