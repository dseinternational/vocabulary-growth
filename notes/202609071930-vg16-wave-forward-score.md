# VG16's sequential validation target: what is held out, and what may be claimed from it

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-07. Implements [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) item 5's second half and [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 3.8. The harness is `scripts/wave_forward_score.py`; **no reporting-quality run has been made**, and the numbers it will produce belong in the VM refit window alongside task 3.7's arms.

## The gap it fills

VG16's understood PSIS-LOO is suppressed, correctly: the lag predictor for a child's later wave embeds that child's _earlier_ understood count, so leaving one administration out of the likelihood does not leave it out of the model, and Pareto-k cannot see the leak because nothing about the pointwise density looks wrong. The registered replacement — a grouped forward-chaining or held-out-child sequential score — was never built, which left VG16 with no generalisation check at all rather than a suppressed one.

## The decision #289 asked for: which unit is held out

Task 3.8 asks for the unit to be decided rather than assumed, between "a child's later waves given their earlier ones" and "whole children". Both are implemented, and **later waves is the default**, for a reason that is about what each measures rather than about strictness.

Under `--holdout-unit later-waves` a fold child's first administration wave stays in training and everything after it leaves the likelihood. The child's random effect is therefore informed by their first visit, exactly as it would be for a child the model has met once. The question the score then answers is the one the coefficient is contested on: _given that the child effect already knows what this child was at their first visit, does the cross-lag term improve the prediction of their next one?_ That is the sharp form of the review's own worry that `beta_lag` may be proxying omitted persistent covariance — if it is, the child effect will already have absorbed it and the term will add nothing.

Under `--holdout-unit child` the whole child leaves, so their random effect is a prior draw. That is a harder prediction and a different question — unseen-child generalisation rather than next-visit prediction — and it is offered rather than chosen because both are legitimate and they are not interchangeable.

**First waves are never scored under either unit.** A first wave carries `has_lag = 0`, so `x_lag` is zero and the cross-lag term drops out of its likelihood entirely: the two arms assign it the identical density. Excluding those rows is therefore not merely leak avoidance, it is the only informative scored set — 975 of the frame's 1,708 rows are first waves.

## The comparator is VG16 without its coefficient, not VG10

`dataclasses.replace(VG16, use_cross_lag=False)`, on the identical prepared frame and the identical folds. VG10 would have been the obvious control and is the wrong one: the two definitions differ in more than the cross-lag, so a difference in score could not be attributed to the coefficient. The control's construction asserts that exactly two fields moved — `use_cross_lag` and the `config_name` that keeps its scratch output apart — so a future field that `replace` starts carrying along stops the run rather than quietly widening the contrast.

The headline outcome is **spoken**. The lag term enters the graph only through the production-ratio logit, so a held-out row's understood density is the same under both arms up to sampling noise; the understood column is reported as a control that should show nothing, and summing the two would bury the signal in it.

## Three row sets, and why the strictest is the one to quote

The scored rows are not homogeneous, and the differences change what may be claimed. All three are written to `wave_forward_compare.csv`:

- **`lagged-from-training`** — the row carries a non-zero lag _and_ its lag source is a row the likelihood saw. This is the headline. A second wave qualifies under `later-waves` because its source is the child's first wave, which stayed in training.
- **`lagged`** — adds the rows whose lag source was itself held out. A _third_ wave's source is the second, which `later-waves` removes, so its predictor conditions on a row the model never fitted to. That is still honest forward chaining — the row's own outcome appears nowhere in its own predictor — but it is not "trained on everything except the scored rows", and the two should not be quoted as though they were the same claim.
- **`all-later-waves`** — adds the rows with no usable lag at all. These enter both arms identically, so they leave the elpd difference untouched while shrinking the per-row spread the paired standard error is built from. Quoting that standard error would make the comparison look more precise than its evidence, which is why the distinction is a column in the output rather than a sentence here.

The difference is **paired** on the scored rows, with `SE = sqrt(n) * sd(per-row difference)`. Differencing two independent totals instead discards the correlation between arms scoring the same rows and inflates the error — the defect #289 task 3.2 records for the hand-differenced VG20-versus-VG22 comparison.

## Two cross-checks that the scored set is the right one

The wave grouping here is a second implementation of "a child's administration waves in age order": `wave_index` walks the frame, `prev_wave_lag_for_frame` walks it inside the engine. A disagreement would be invisible in the output and fatal to the meaning, so it is checked against numbers derived independently:

| quantity                            | this harness | independent source                                                                     |
| ----------------------------------- | -----------: | -------------------------------------------------------------------------------------- |
| first-wave rows                     |          975 | 975 — [202609071000](202609071000-vg16-available-case-audit.md)'s available-case audit |
| later-wave rows with no usable lag  |          153 | 153 — the same audit                                                                   |
| lagged rows carrying a spoken count |          473 | 473 — #242's support count for the coefficient                                         |

All three agree exactly, and no first wave carries a lag. The last two are pinned as tests.

## What has and has not been run

The pipeline has been exercised end to end at `--config dev --folds 2` into an isolated output root, which proves the plumbing: the folds build, both arms fit, the diagnostics gate is captured per fold, and the four tables are written with a comparison-manifest entry. The tables were read rather than merely produced: six comparison rows (three restrictions x two outcomes), the `lagged` set at 473 rows matching the audit in the live run, the understood control moving two orders of magnitude less than spoken as its role requires, and a manifest entry carrying an empty `contributing_fits` beside a raw-data fingerprint. **A dev-tier two-fold run is not evidence about the coefficient** and its numbers are not recorded here: every one of its four fold fits failed the convergence gate, one at maximum R-hat 1.23 with a minimum effective sample size of 7, and the script says so in capitals before printing them. The reporting-tier run — five folds, both arms, `--config rep` — is ten VG16 fits and belongs in the VM window; it is what #289 task 3.10 needs before VG16's headline can be reinstated or recorded as withdrawn.

Like `kfold_loso.py`, this script fits its own folds and reads no model of record, so its comparison-manifest entry records the raw-data fingerprint rather than a contributing fit.

## One thing the smoke run caught, and the refactor it forced

The fold fit itself — validate the counts, build on a frame carrying a `holdout` column, sample with the observation-level deterministics stored, run the canonical diagnostics scan — is the same in both scripts, and the first version of this one had it copied. The copy was wrong within the hour: `fold_gate_fields` reads the energy verdict from `gate["checks"]["bfmi"]`, and the copy read `gate["bfmi_ok"]`, which does not exist, so every fold reported a passing BFMI whatever the sampler found. Nothing in the output would have shown it — the column would simply have been `True` throughout.

It now lives in `vocab_growth.fold_fits`, and `kfold_loso.py` calls the same function. What did **not** move is policy: which rows are held out, which are scored, and whether a failed fold aborts or is recorded and flagged differ between the two scripts and belong with the question each is asking.

The `--config dev --folds 2` run also caught a plain error — the manifest call used a keyword the function does not take — which is the ordinary reason to run the thing end to end before committing it.

## What it does not do

It does not address [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) item 6, the wave-sequential _recovery_ harness — that needs a simulator for a model whose predictor is a function of its outcome, and is the piece [#297](https://github.com/dseinternational/vocabulary-growth/issues/297) also waits on. This is real-data validation and needs no simulator, which is why it could be built first.
