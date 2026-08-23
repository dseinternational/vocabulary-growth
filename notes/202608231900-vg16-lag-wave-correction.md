# VG16 lag correction: the administration wave, the source rule, and what was rebuilt on them

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!IMPORTANT]
> This is the Gate A decision record and partial Gate B implementation for [#242](https://github.com/dseinternational/vocabulary-growth/issues/242), following the review in [`202608231714`](202608231714-vg16-statistical-model-review.md). It corrects the code and the claims; it does **not** produce a quotable estimate. VG16's headline remains withdrawn until the corrected definition passes a `rep` fit's convergence and provenance gates and the remaining #242 validation and sensitivity work is done.

## 1. The estimand's data unit: an administration wave

The wave is defined operationally as **every row a child carries at one recorded age in months** — a complete `(subject, age)` group. A child given two checklist forms at the same recorded age contributes one wave with two rows. The stored data establish the same recorded age, not necessarily simultaneous administrations; the likelihood continues to treat two same-wave forms as conditionally independent measurements, which is an assumption the review's P2 finding (§3.2) leaves open, not a finding this correction settles. Modelling the paired forms jointly with their overlap remains available as future work; nothing in this correction forecloses it.

## 2. The lag rule, stated completely

1. **Retain all rows.** Every row of a wave stays a separate likelihood row, as before; no row is deleted or collapsed. Same-wave forms may carry distinct item information, so deletion is not obviously correct, and collapsing would need a measurement model the project has not built.
2. **One source per wave.** Every row in a wave receives the same lag source: the child's most recent strictly earlier wave with at least one usable understood count, skipping earlier waves without one. The source state advances only after the whole wave is assigned, so no row can receive a same-age source and the construction is invariant to input row order.
3. **Largest count at a multi-measurement source wave.** Where a source wave carries understood counts from two forms, the larger count is selected. Rationale: every count is scored against the same 810-item inventory under the project's difficulty-ordering harmonisation, and a shorter form right-truncates it, so the largest observed count is the least-truncated measurement available; it is also the only rule reproducible from the arrays stored in the trace (`X_obs`, `subject_obs`, `y_u_obs`), which keeps the experiment scripts exact. Ties are statistically inert — same child, study and recorded age, so both rows imply the identical likelihood contribution. Alternatives (form-priority by ceiling, averaging) were considered and rejected: ceiling-priority needs a column the trace does not store, and averaging mixes a censored measurement in.

**The selection rule currently never fires.** Measured on the reconstructed default frame: of the source waves under the corrected assignment, **zero** carry more than one understood measurement (the frame's 10 dual-understood child-age groups never serve as sources). The rule is registered ahead of need, and `cross_lag_audit.csv` counts its bite on every future fit.

## 3. What the correction changes, measured

Reconstructed default frame (1,431 rows, 987 understood, 1,428 spoken, 767 children, 14 studies — matching the review §8):

| quantity                                             | defective row-order walk | corrected wave-grouped rule |
| ---------------------------------------------------- | -----------------------: | --------------------------: |
| rows with a prior understood source                  |                      412 |                         478 |
| rows with a source and a current spoken likelihood   |                      409 |                         475 |
| children contributing a supporting row               |                      248 |                         249 |
| rows whose lag flips under 5 random row permutations |                    29–38 |                       **0** |

The 66 recovered spoken rows come from 46 children (65 `us_01`, 1 `uk_02`), exactly the review's grouped retain-all candidate set. No row loses a lag. Supporting-row gaps run 1–28 months (median 6), and 7 supporting sources have an understood count of zero (clipped logit) — the boundary-treatment sensitivity remains registered in #242.

## 4. What was rebuilt

- **Engine** (`src/vocab_growth/models/common_bivariate_re.py`): `compute_prev_wave_lag()` is the corrected, public, array-based construction built on `iter_subject_age_waves()`; `_compute_prev_wave_lag()` is now a DataFrame adapter over it. `cross_lag_audit_frame()` persists the coefficient's support to `cross_lag_audit.csv` on every cross-lag fit — source rows, gaps, zero-count sources, multi-measurement source waves, spoken-likelihood branch, and (the frame now loads `survey_vocab_max` for cross-lag models) checklist-form ceilings and transitions.
- **Experiment scripts**: `vg16_crosslag_quantification.py` and `vg16_within_lag_bias.py` now import the engine's construction instead of carrying copies of the defective walk; the bias script's sequential simulator walks the same wave groups (`vg16_within_ridge_arm.py` inherits it by module load).
- **Tests** (`tests/test_cross_lag.py`): the review's three named regression cases — the `[12, 24, 24]` parallel-form pattern in both tie orders, row-permutation invariance on a frame combining every state-tripping structure, and multiple source-form rows under shuffles — plus audit-frame coverage and array/DataFrame API agreement.
- **Diagnostics** (`common_bivariate.diagnostics`, issue #242): for cross-lag definitions, understood PSIS-LOO is suppressed (the lag predictor embeds the held-out counts, and Pareto-k cannot see the leak), spoken LOO is labelled as conditional on the observed understood history, and the pair plot is reordered so `beta_lag` and the child/study scales fill the capped grid — previously `beta_lag` fell off the end of model order and the report described a figure that could not show it.
- **Prior artefact**: `beta_lag_dist.png` is now emitted beside the other prior figures, so the report can put the symmetric prior against the coefficient's posterior. The fuller effect-scale prior predictive and prior-scale sensitivity remain #242 follow-ups.
- **Definition and docs**: the stale "bias-robust" banner and the withdrawn within-child-bias rationale are gone from `definitions.py`; the VG16 report now defines the wave, reads its support counts from the audit artefact, labels every prediction grid as a zero-lag reference, withdraws the temporal-direction / proven-floor / negligible-within-child claims, and carries the LOO caveat; `docs/models/README.md` records the withdrawal, and the book chapters no longer quote the disattenuated +0.24 as a live check or tabulate VG16's LOO beside unconditional scores.

## 5. What this note does not claim

No sampling run was performed. This note asserts nothing about the corrected coefficient's sign, size or interval, and the pre-correction `+0.203` (89% ETI [0.093, 0.316], `rep`, commit `d041e7f`) must not be used to predict it. Still open in #242 before any estimate is current: the wave-sequential positive-truth and correlated-trait recovery matrix, a grouped forward or held-out-child validation score, the correlated-random-effects-plus-lag comparison, the gap / leave-one-study-out / same-form / conditional-only / zero-count sensitivities, the available-case audit, the effect-scale prior predictive, and the `rep` refit with rerendered artefacts.
