# Held-out validation for the typically-developing models, and why it is not pre-refit work

> [!NOTE]
> Documentation review, 7 October 2026, with assistance from OpenAI Codex/GPT-6. The [1 October harness](202610011300-td-held-out-validation-harness.md) implements child- and study-held-out validation for VG11, VG12, VG21, VG23 and VG26. The missing-tool statements below describe the September code. Harness and smoke-test completion do not establish that reporting-quality validation has been run.

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

## 4. Outcome

The September review deferred the full validation work while correcting report pointers that named unsupported checks. The [October implementation](202610011300-td-held-out-validation-harness.md) now provides both subject and study holdouts, including a univariate scorer and integration over held-out effects. It records smoke runs and commands for reporting runs. Use that record for execution; the earlier cost estimates and proposed implementation sequence are superseded.
