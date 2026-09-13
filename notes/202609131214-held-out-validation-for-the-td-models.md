# Held-out validation for the typically-developing models, and why it is not pre-refit work

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-13. **Question:** should `scripts/kfold_loso.py` be extended to the typically-developing reference models (VG11, VG12, VG21, VG23), and should that happen before the refit? **Evidence:** the `loo_summary.csv` and `diagnostics.csv` of each TD fit of record (the 2026-09-08 `rep` fits), their prepared frames rebuilt through `vocab_growth.analysis_frames`, the memory-watch log those fits ran under, their rendered report pages, and the code of `kfold_loso.py`, `loso_compare.py` and `report_cells.render_loo_section`. **Answer:** yes, eventually. Nothing else can give these models a usable out-of-sample check, and [#240](https://github.com/dseinternational/vocabulary-growth/issues/240) cannot close without one. But it decides nothing, it costs tens of hours of fits, and #240's own open graph question could overtake it, so it is not pre-refit work. One small thing found on the way _was_ pre-refit work, and is fixed in the same change as this note: the report pages pointed readers at checks that cannot run on them (§2).

## 1. PSIS-LOO is unusable on every TD reference model

| model |   rows | children | children with one administration | rows from those children | Pareto k ≥ 0.7 or non-finite | `p_loo` |
| ----- | -----: | -------: | -------------------------------: | -----------------------: | ---------------------------: | ------: |
| VG11  | 18,500 |   14,553 |                            86.7% |                    68.2% |                  8,758 (47%) |  10,611 |
| VG12  |  7,049 |    5,819 |                            82.8% |                    68.4% |                  2,630 (37%) |   3,844 |
| VG21  |  6,783 |    5,707 |                            83.5% |                    70.2% |                  3,796 (56%) |   7,555 |
| VG23  |  6,356 |    5,496 |                            84.9% |                    73.4% |                  3,726 (59%) |   7,317 |

VG21's and VG23's Pareto and `p_loo` figures are the administration-level score; their per-outcome scores are less bad and still unusable, 40% to 48% of rows. For contrast, the Down syndrome frame VG20 is fitted to has 943 children, 46.6% of them measured more than once, carrying 70.5% of its rows.

This is structural, not a sampling shortfall, and a better-tuned fit will not change it. Most TD children have a single administration, so each one's intercept is informed by one row; removing that row moves the posterior a long way, which is exactly the case importance sampling cannot approximate. `p_loo` at 57% of VG11's rows is the same signature. Refitting per fold is the remedy, as it was for VG20 and VG22 (`notes/202609091200-vg20-vg22-gate-resolved.md`).

## 2. The pages send the reader to checks that cannot run on them

`render_loo_section` adds a paragraph to every fit that samples `tau_subject`, `tau_subj_u`, `tau_subj_q` or `tau_subj_sign`, saying the generalisation question "is better put to" `scripts/kfold_loso.py` and `scripts/loso_compare.py`, which "hold out whole studies or whole children rather than single rows". All four TD pages of record carry it (checked in each rendered `index.html`), and it is wrong for them twice over:

- **Neither script accepts a TD model.** `kfold_loso.py`'s `AVAILABLE` is VG07–VG10, VG19, VG20 and VG22, and it loads the Down syndrome pool through `load_combined_data()`. `loso_compare.py` covers VG07–VG09, read from hard-coded output directories.
- **Neither holds out a study, for any model.** `kfold_loso.py`'s holdout units are `subject` and `later-waves`; `loso_compare.py` integrates over held-out subjects. In both, LOSO means leave-one-_subject_-out, and `loso_compare.py`'s own `require_full_trace` purpose string reads it as "Leave-one-study-out", which is likely where the pointer's "whole studies" came from.

The same paragraph fires for VG15, VG24 and VG25 through `tau_subj_sign`, and neither script supports those either. It is the pattern of `notes/202609121430-the-forward-score-vg25-was-sent-to.md` again: a page naming a remedy that had never been run on it.

Rewording it is cheap but not free. The text is a string inside a `print` call, so it is in the executable-code signature's hashed AST and moves every fit's signature. Before the refit that costs nothing extra, because the refit replaces those fits anyway; after it, the same edit would stale the refit's output. So it belongs with the other signature-moving changes that land first.

**Fixed in the same change.** `report_cells.HELD_OUT_CHECKS` now declares each held-out check with the registered models it scores — `kfold_loso.py` VG07–VG10, VG19, VG20, VG22; `loso_compare.py` VG07–VG09; `wave_forward_score.py` VG16, VG25 — and the paragraph reads the page's `model_id` from its fit manifest and names only those, each with what it actually holds out. A page no check covers says so; a sensitivity arm is told that the checks score the registered model rather than it; a page without a manifest says it cannot tell. "Whole studies" is gone. `tests/test_report_cells.py` pins each entry against the script's own model list (`AVAILABLE`, `SPECS`, `CROSS_LAG_MODELS`), so extending a script without updating the declaration fails a test. Rendered against the fits on disk: VG11 and VG21 now say no check covers them, VG20 names `kfold_loso.py`, and VG25's smoke fit names `wave_forward_score.py`.

## 3. What held-out validation would add

- **Whether the predictive bands hold for a child the model has not seen.** VG21's and VG23's captions say the band approximates "where one more child's counts would fall". #240's first open finding is that the fixed 810-word scale may route form differences into `kappa(age)`; if it does, the result is miscalibrated bands for new children. The in-sample calibration cannot show that, because it conditions on the fitted child effects.
- **Whether the reference curve transfers to a dataset it has not seen.** Each TD curve is "the average study" over 10 datasets (VG11) or 6, with language inseparable from dataset. Holding out a study, its intercept drawn from the prior as a held-out child's is, is the only test of that. With 6 to 10 studies it is also only 6 to 10 fits per model.
- **#240 itself.** Its open item "Replace leave-one-administration-out PSIS-LOO as the principal generalisation check with leave-one-child-out and leave-one-study-out validation that integrates held-out effects", and its completion criterion "New-child and new-study performance is evaluated at the appropriate grouping level".

## 4. Why it is not pre-refit work

- **It decides nothing.** Model roles were settled on 2026-09-09 (`notes/202609091600-model-roles-settled.md`), and no choice between TD models is waiting on a predictive comparison. What it produces is a calibration and transfer statement for four reference pages.
- **There is no refit window to miss.** It had been listed with the pre-refit work, on the ground that the refit window would otherwise have nothing to run for these models. That ground does not hold: `kfold_loso.py` fits its own folds and records the raw-data fingerprint rather than contributing fits, so the fits of record neither feed it nor are affected by it. It can run at any time.
- **It could be overtaken.** #240's first open item — the form-scale compression, an age- or form-varying child loading, possibly a measurement model — may change the TD graphs. Folds scored before that settles would describe graphs that are no longer registered.
- **It is expensive.** The 2026-09-08 `rep` fits' spans in `td-rep-memory.log` (UTC, matching each manifest's `created_at_utc`) were VG11 4 h 39 m, VG12 55 m, VG21 1 h 46 m and VG23 1 h 36 m, with VG11 and VG12 overlapping for 40 minutes, so they are indicative only. A fold fit is the whole graph with a fifth of its rows out of the likelihood, so roughly a full fit: five child folds per model comes to about 45 hours of fitting before any study holdout. A cheaper tier is unlikely to be scorable: the Down syndrome criterion-3 folds at `test` failed the element-wise gate in 9 of 10 with 943 children (worst ESS 265), and these models have 5,496 to 14,553. Plan for `rep-lite` or `rep` — an expectation, not a measurement.
- **It is script work, not configuration.** The frame has to come from the definition (`analysis_frames.build_analysis_frame`, as `wave_forward_score.py` now does) rather than from the Down syndrome pool. `holdout_subject_elpds` scores the bivariate nested likelihood, so VG21 and VG23 (`bivariate_re`) are nearest, and VG11 and VG12 (`univariate_re`) need a univariate scorer. A `study` holdout unit is new for every model. Fitting the folds is already engine-general: `fold_fits.fit_holdout_fold` has dispatched on the definition's engine since #341.

## 5. Recommendation

Not yet decided by the study owner.

1. **Before the refit:** reword the pointer in `render_loo_section` so that it names only checks that exist for the model on the page, batched with the other signature-moving changes. _Done in the same change (§2)._
2. **After #240's form-scale question is settled:** extend `kfold_loso.py` — VG21 and VG23 first, with child folds and a study holdout, at `rep-lite` or above. VG11 and VG12 follow once a univariate scorer exists.
