> [!NOTE]
> Drafted with assistance from Claude Code/Opus 5.5.

# Held-out validation harness for the typically developing models

**Date:** 2026-10-01. **Issue:** [#240](https://github.com/dseinternational/vocabulary-growth/issues/240), the item "Replace leave-one-administration-out PSIS-LOO as the principal generalisation check with leave-one-child-out and leave-one-study-out validation that integrates held-out effects". **Plan it implements:** [202609131214](202609131214-held-out-validation-for-the-td-models.md), which found PSIS-LOO structurally unusable on VG11, VG12, VG21 and VG23 (37% to 59% of rows at Pareto k ≥ 0.7) and listed three missing pieces in `scripts/kfold_loso.py`: a typically developing frame path, a univariate scorer and a study holdout unit. All three now exist. Nothing was fitted at a reporting tier and no result here is a finding: the smoke runs below show only that the harness runs end to end.

## What was built

`scripts/kfold_loso.py --models` now also accepts VG11, VG12 (univariate) and VG21, VG23, VG26 (bivariate). The Down syndrome path is unchanged. A run takes one population, and every model in a run must share one prepared frame (checked by `analysis_frame_hash`), so a paired difference is always on identical children: VG21 with VG26 is allowed, VG21 with VG23 is not.

**Frame.** Rebuilt from the definition through `analysis_frames.build_analysis_frame`, so each model is scored on exactly the frame its engine fits, as `wave_forward_score.py` already does.

**Holdout units.**

- `--holdout-unit subject`: grouped K-fold by child, stratified by study and administration count with the existing `stratified_subject_folds`. A fold's children leave the likelihood but stay in observation space.
- `--holdout-unit study`: leave one study out, one fold per retained study (six for VG12, VG21, VG23 and VG26; ten for VG11). The held-out study is **removed from the zero-sum study block**, whose remaining studies are recoded `0 .. K-2`. Leaving it in would have fixed its offset at minus the sum of the others, which is a fitted value rather than a new study. Its rows stay in observation space, under a placeholder study code that the scorer removes, so the population curve and dispersion are evaluated at their exact ages.

**Integration.** The Down syndrome scorer reads the held-out rows' probabilities off the trace, where each held-out child's effect is one prior draw per posterior draw. The typically developing scorer integrates instead. For each posterior draw it removes the child effect (and, under `study`, the placeholder offset) from the row's logit, then integrates over the child's effects by adaptive Gauss-Hermite quadrature: nodes placed at the mode and curvature of the child's own likelihood, found by a damped Newton search on finite differences, as the engine's singleton marginalisation does. Under `study` the new study's offset is `Normal(0, tau)` at the fitted between-study scale, the convention `predict_new_study.py` uses; it is Gaussian and independent of the child effect, so the two add into one Gaussian and the integral stays one- or two-dimensional. The draws are then averaged on the probability scale (log-mean-exp).

| Setting                  | Default                      | Why                                                                                                                                               |
| ------------------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Posterior draws scored   | 2,000, evenly thinned        | The cap the [2026-09-23 corrections](202609231200-statistical-review-corrections.md) use for new-child predictive distributions. `--score-draws`. |
| Quadrature nodes         | 11 per dimension (121 in 2D) | Worst error 1.2 × 10⁻⁴ nats per child on a deliberately extreme stress set, against 7 × 10⁻⁴ at 7 nodes. `--quadrature-nodes`.                    |
| Child effects integrated | 1 (VG11, VG12), 2 (others)   | VG23 and VG26 carry `rho_uq`; the Cholesky factor reproduces the model's covariance exactly (tested).                                             |

**Likelihood.** The model's own: Beta-Binomial with the age-varying `kappa` at each row, the variance-partition child scale for VG11 and VG12, the sex contrast, and for the bivariate models comprehension on the 810-item scale with speech nested in it by the engine's paired-count rule (`nested_outcome_spec` on the whole fold frame, as the 2026-09-23 correction requires), falling back to the `product_marginal` density where it does. The fixed part of each logit is checked against the trace's own per-row probabilities and against an age-only invariant, so a graph term the scorer does not model raises rather than changing the density.

**Quantities.** Every model writes `elpd`, the log predictive density of all of a held-out child's counts. The bivariate models also write `elpd_understood` and `elpd_spoken_given_understood`, their difference, which is speech conditional on the child's observed comprehension with the mixture reweighted by it (finding 39), not merely given it as a denominator.

**Standard errors.** Over clusters: children under `subject`; studies under `study`, because children of one study share its offset. With six to ten studies the study-unit standard error is itself rough. The study unit scores each child as a new child in a new study, which is the per-child predictive the reports' bands describe; it is not the joint density of the whole study, which would need the study offset integrated once across all its children.

**Refusals**, before any fit, through `td_scoring_refusal`: an age-varying child scale (the A1 arms), child slopes or factors, singleton marginalisation, the single-administration frame, any cross-lag, any spoken fallback other than `product_marginal`, per-study age slopes under the study unit, and the `later-waves` unit. Mixing populations, `--holdout-unit study` on a Down syndrome model, and models on different frames are refused by the driver.

**Outputs** go to `<output-root>/comparisons/kfold_loso_td_<unit>_{child_elpds,by_study,summary,compare,fits}<suffix>.csv`, under their own manifest label `kfold_loso.py (td <unit>: <models><suffix>)`, so they never overwrite the Down syndrome tables. The manifest records the raw-data fingerprint, the prepared-frame hash and every scoring setting.

## Package changes

The work is in the script, but three package changes were needed. Each moves the executable-code signature, which the pending refit absorbs:

- `common_univariate_re.build_model_graph` honours a `holdout` column, as the bivariate engine has since the LOSO script was written. Ordinary fits have no such column and their graph is unchanged.
- `fold_fits.fit_holdout_fold` reads a univariate definition's own outcome column, and its prior and build stages are split out as `build_holdout_fold_context` so a test can inspect a fold's graph without sampling.
- `report_cells.HELD_OUT_CHECKS` lists the five models under `kfold_loso.py`, as `tests/test_report_cells.py` requires, so their pages now name the check.

## Tests

`tests/test_kfold_loso_td.py` (fast, no sampling): each frame's hash equals the engine's expected hash; child folds partition the children and no held-out child or study has a likelihood row; a study fold leaves the zero-sum block; the NumPy log density equals the PyMC model's pointwise `logp` at a jittered parameter point for VG12, VG21 and VG23, row by row, to 10⁻⁹; the study-unit decomposition and integrated variance; a tampered trace fails loudly; quadrature against brute-force integration in one and two dimensions, including a case where a Newton search without backtracking cycled; summaries and paired differences clustered correctly; and each refusal.

## Smoke runs

All at `--config dev` (2 chains of 500 draws), one fold each, written to a scratch output root inside the worktree. Every fold failed the convergence gate, as `dev` folds are expected to (worst R-hat 1.08 to 1.12, minimum ESS 14 to 18), so **none of these numbers may be interpreted**; they show that the harness runs and that its pieces agree.

| Run                                              | Fold                 | Held out                   | Fit + score | Scoring | elpd (all counts) |
| ------------------------------------------------ | -------------------- | -------------------------- | ----------: | ------: | ----------------: |
| `--models VG12 --holdout-unit subject --folds 5` | fold 0               | 1,169 children, 1,419 rows |     6.0 min |       — |          −7,918.7 |
| `--models VG21,VG26 --holdout-unit study`        | ByersHeinlein (VG21) | 507 children, 568 rows     |    11.3 min |   102 s |          −4,464.7 |
| same run                                         | ByersHeinlein (VG26) | same                       |     9.8 min |    95 s |          −4,474.1 |

The second run also wrote the three bivariate quantities and the paired VG26 − VG21 table; with one study its standard error is undefined, and the CSVs say so rather than inventing one.

Two checks on a further VG12 child fold (fold 1, 1,000 draws):

- **Against the Down syndrome path's estimator.** Reading the trace's `p_obs`, where each held-out child has one prior draw of their effect per posterior draw, gives a fold total of −7,848.42 against the integrated scorer's −7,846.54. The per-child differences have mean 0.0016 and SD 0.068 nats: the same quantity, with the one-draw estimator noisier and, through Jensen's inequality, lower.
- **Quadrature.** 11 against 21 nodes moves no child by more than 8 × 10⁻⁵ nats and the fold total by 0.0024.

Scoring cost about 50 s for 1,169 univariate children and 100 s for 507 bivariate children at 1,000 draws, scaling linearly in draws and children.

## Commands for the real runs

Not run. The note of 13 September recommends settling #240's form-scale item first, because folds scored before that would describe graphs that may no longer be registered. Use a scratch output root only when testing; the real runs belong in the shared root.

```bash
# Child folds (K = 5) and leave-one-study-out, one model per run
uv run python scripts/kfold_loso.py --models VG12 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg12
uv run python scripts/kfold_loso.py --models VG12 --holdout-unit study --config rep-lite --suffix _vg12
uv run python scripts/kfold_loso.py --models VG11 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg11
uv run python scripts/kfold_loso.py --models VG11 --holdout-unit study --config rep-lite --suffix _vg11
uv run python scripts/kfold_loso.py --models VG23 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg23
uv run python scripts/kfold_loso.py --models VG23 --holdout-unit study --config rep-lite --suffix _vg23
# VG21 and VG26 share a frame, so one run gives the paired VG26 - VG21 difference
uv run python scripts/kfold_loso.py --models VG21,VG26 --holdout-unit subject --folds 5 --config rep-lite --suffix _vg21_vg26
uv run python scripts/kfold_loso.py --models VG21,VG26 --holdout-unit study --config rep-lite --suffix _vg21_vg26
```

**Estimated cost.** A fold fit is roughly a full fit. From the 2026-09-08 `rep` spans (VG11 4 h 39 m, VG12 55 m, VG21 1 h 46 m, VG23 1 h 36 m; VG26 taken as VG21), five child folds plus one fold per study is 15 fits for VG11 and 11 for each other model, about 140 hours at `rep`. `rep-lite` runs two-thirds the iterations per chain on four chains rather than six, so perhaps 90 to 100 hours; this is an expectation, not a measurement, and the machine's hybrid cores make wall times unreliable. Scoring at 2,000 draws adds roughly 4 minutes to a VG11 child fold and 8 to a bivariate one, scaled from the smoke runs; under an hour per model. Fold traces keep the observation-level deterministics, so a VG11 fold at `rep-lite` holds roughly 16,000 draws by 18,500 rows for each of `f_obs`, `p_obs`, `kappa_obs` and `z_obs` (about 2.4 GB each) plus 14,553 child effects; run folds serially. As with the Down syndrome folds, a fold below the reporting tier may fail the convergence gate, and a model with any failed fold is flagged rather than dropped.
