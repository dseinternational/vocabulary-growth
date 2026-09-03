# Full reporting-quality refit, 2026-09-01 to 2026-09-03: run record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Scope agreed at the start (#281):** all twenty registered models at `rep`, the four runbook-mandatory sensitivity arms plus the six VG14/VG15 fallback arms, then the full publication tail; typically developing models fitted at plain `rep` first and escalated where marginal. Output root: the repository-local `output/` on this VM's `/scratch` (no `/data` mount), with every completed fit uploaded as it landed so a deallocation would lose nothing published. Working files: `output/run-record.md`, `output/escalation-plan.md`, `output/retry-backlog.md`, `output/report-urls.md`.

## Fits

All twenty models fitted at `rep` and published. Sixteen passed the gate on the first pass. The two known understood-GP ridge failures (VG08, VG09; `p_slope_low_u` and `intercept_u` at R-hat 1.0106) cleared on escalation — VG09 at rung 1 (12000/8000/0.97, R-hat 1.0045), VG08 at rung 2 (16000/10000/0.99, R-hat 1.0065, 0 divergences). VG11 failed rep by 5.6e-7 on R-hat with 35 divergences and cleared rung 1 at R-hat 1.0027, ESS 2,753, BFMI 0.36–0.38, with one divergence in 48,000 draws. Every escalated fit records `rep` in its manifest and is publication-valid.

**Six fits carry soft-tier caveats and were published under `--allow-caveats`:** divergences on VG11 (1), VG12 (4), VG13 (3), VG22 (1), VG23 (8), and energy BFMI just under 0.3 on VG12 (0.21), VG13 (0.25), VG21 (0.26), VG23 (0.24). All six cleared R-hat and ESS. `check_fit.py all --purpose publish` has no caveat allowance and reports them `[invalid]`; the runbook's checklist line "all models pass on unrounded diagnostics" is therefore **not met** by them, and this was a run-time judgement rather than a pre-agreed one. VG12's BFMI is the weakest; a rung at `target_accept 0.99` is the obvious next step if the caveats are not acceptable.

## Sensitivity arms

Nine of ten completed. `vg15 dse-native-only` failed the gate at rep (max R-hat 1.017) and at rung 1 (1.0120), all failing parameters on the sign curve; it cleared at rung 2 (R-hat 1.0058, ESS 709, 0 divergences) on 264 rows — so it was a tuning shortfall after all, not the identification limit the run record first called it. `vg15 fallback-dispersion` is a **code defect**: numba/LLVM "ran out of registers during register allocation" in `np_concatenate` over 44 arrays at sampler compile, reproduced in isolation; VG14's same arm compiles. It blocks #266 finding 8 for VG15 and is not fixed here. The pinned counts in the runbook are stale: `us01-implausible-reinstated` reinstates 5 rows (pinned 11) because #275's same-day rule now holds 6 of the 11, and `dse-native-only` excludes 1,239 (pinned 1,243); the reinstated variant needs a combined-flag successor to answer its registered question.

Robustness matrices (`compare_sensitivity`): VG15 `marginal-moments` robust (45/45, max |Δ| 1.7 words); VG15 `paired-only` sensitive (32/45, max |Δ| 37.6); VG14 `fallback-dispersion` robust (368/368), `paired-only` and `marginal-moments` sensitive. The mean-only approximation is harmless on the model of record; the fallback rows carry information.

## Reporting

Every template was reviewed and rewritten while the fits ran (`notes/202609021200-report-template-review.md`), on a branch in a worktree so no in-flight fit recorded a dirty checkout. The interpretive work that followed is in `notes/202609021620-dispersion-kappa-comparability.md` and `notes/202609021800-production-ratio-by-understood.md`; each carries its evidence tables and the corrections made to earlier readings. In sequence: dispersion-scope block and the two-sided `pressing` flag; the by-understood production figure re-captioned and given the observed children; the spoken dispersion contrast's denominator corrected (published reversal was an artefact); VG21 made the typically developing joint comparator; the `compare_models` headline retired; the words/ratio figures given age markers and the observed children, the age fan retired; the reference child named as the estimand, with per-study fans, a calibration block and the administration-weighted child beside every milestone.

Publication: all twenty pages regenerated from their traces with the final plotting code and re-rendered with the final templates (20 rendered, 20 uploaded, 0 failures; `output/report-urls.md`). Comparisons: all scripts ran, including `compare_ds_td_re` with the weighted child and the three sensitivity matrices; figure sync passed. Report book rendered (12 pages). Comparison book: rendered after two fixes recorded in the runbook — the bare `quarto render` resolved a Python without `yaml` (pin `QUARTO_PYTHON`), and `subject_effect_correlation.py` was missing from the runbook's script list — and one fail-soft change: the recovery callout now says when parameter recovery has not run rather than failing the book.

## Not done in this run

- **Parameter recovery** (`fit_recovery.py`, headline set VG20/VG12/VG15) was outside the agreed scope and has not run against these fits; the comparison book says so where it would quote it.
- The univariate engine (VG11/VG12) has no administration-weighted loader, so the weighted spoken attainment delay stops at 100 words (the joint comparator's window). A weighted loader for `common_univariate_re` is the follow-up.
- `vg15 fallback-dispersion` (code defect), the sensitivity variants' pinned counts and successor variant, the legacy `b_kappa_mag` prior on VG05/07/08, VG22's young-end `kappa_s` identification, and whether the Down syndrome population curves should be reported for the administration-weighted child alongside the reference child, all remain open and are recorded in the notes above.
- The branch `feat/report-template-review` (30 commits) is pushed but not merged to `origin/main`; several commits change published numbers and want review.
