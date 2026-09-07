# VG16's available-case assumption, audited

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-07 · **Issue:** [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) · **Reproduced by:** `scripts/experiments/vg16_available_case_audit.py` · **Frame:** VG16's own, 1,708 rows / 943 children / 15 studies, the frame the current `rep` fit of record was made on (manifest hash `sha256:b27f64ea…`)

## What was asked

The VG16 statistical review ([202608231714](202608231714-vg16-statistical-model-review.md) §4.5) recorded that the cross-lag "is active only where an earlier usable understood count and a later spoken count are observed", that "understood missingness is partly structural because forms measure different outcomes", and that "VG16 conditions on these available observations and has no model for outcome-observation or form-assignment processes". It asked that support counts distinguish structural form missingness from other missingness and that complete-pair and form-restricted analyses be added.

None of that had been counted. The word _partly_ was carrying the argument in both directions at once: benign enough to leave the estimate standing, serious enough to withhold it.

## 1. The missingness is overwhelmingly structural

A checklist form counts as comprehension-measuring if it records an understood count _anywhere_ in the pool. That is a property of the form, which is what "structural" has to mean if it is to mean anything at all — the alternative, classifying row by row, cannot separate a form that never asks from a form that asked and got no answer.

**Understood is missing on 448 of 1,708 rows. 391 of them (87%) are structural**: three forms — ceilings 670 (`it_01`), 675 (`nz_01`) and 680 (`uk_01`, `us_01`) — never record a comprehension count anywhere, in any study. Only the 680 is identified in the data documentation, as the CDI: Words & Sentences checklist ([`data/vocab_data_uk_01.md`](../data/vocab_data_uk_01.md), verified 2026-08-31); the other two are named here by their ceiling rather than by an instrument, because `data/` does not record which form they are. What is measured, and all that the classification needs, is that each of the three records production and never comprehension. The remaining 57 are on forms that do measure comprehension, and they are scattered: `uk_01` 42, `ie_01` 7, `uk_02` 6, `it_01` 1, `us_02` 1.

**Spoken is missing on 287 rows. 284 of them are `us_03`**, which records no production count at all — structural at the _study_ level rather than the form level, which is a distinction the review did not anticipate and which matters concretely: `us_03` is the largest single study in the comprehension frame (284 rows, 16.6%, 180 children of whom 104 are seen twice), and its second waves duly receive a valid prior-wave source — 104 of them — which can never inform the coefficient, because the row has no spoken count to predict. Every form in the pool records production somewhere; only this study does not. The other three spoken-missing rows are `us_01` 2 and `uk_02` 1.

No row is missing both outcomes.

## 2. Every row without a lag, accounted for

580 rows carry a prior-wave understood source; **473 of them enter the spoken likelihood**, from **248 children in 8 of the 15 studies** — 361 on the conditional $S \mid U$ branch and 112 on the marginal fallback, 102 of those 112 being rows whose form never measures comprehension at all.

Of the 1,128 rows with no lag, **975 are at the child's first wave**, where no earlier wave exists to be a source. That is not missingness; it is what a lag means. Only **153** are later rows whose every earlier wave lacked comprehension, and they concentrate exactly where §1 predicts: `nz_01` 78 (its only form is the 675, which never measures comprehension), `uk_01` 58, `it_01` 13, `uk_02` 3, `us_01` 1.

Of the 943 children, **518 are seen once** and so can never carry a lag — the panel is short and irregular, which the report already says. Of the 425 seen more than once, 248 support the coefficient and 177 do not: 69 never have any understood count, 104 never have any spoken count, and 4 have both but never in the order the lag needs.

**Seven studies contribute nothing to the coefficient**: `es_01`, `ie_01`, `nz_01`, `uk_03`, `uk_06`, `us_02`, `us_03`. Of the eight that do, `us_01` (28.1%) and `it_01` (22.4%) supply half the supporting rows between them.

The gap distribution is tighter than the review's "1 to 28 months" suggests: **median 6 months, IQR 5–7**, with 42 of 473 rows (8.9%) above 12 months, 10 above 18 and 4 above 24. Target ages are **median 33 months, IQR 24–47** against a model reported to 115. So the coefficient is estimated mostly from six-month intervals in the third and fourth years and applied across the whole reported range.

## 3. The assignment of forms is not a function of age alone

This is the part that does not come out benign, and it is the part the review's ignorability condition actually turns on. Under an available-case likelihood the observation process must be ignorable given what the model carries — study, age and the child effects. Three studies use both a comprehension-measuring and a comprehension-less form (`it_01`, `uk_01`, `us_01`), so within those three the assignment can be examined directly.

**The parallel-form artefact is measured first, and it is small.** 88 child–age waves carry both form classes; on the 87 with a spoken count on each, the longer form records a median of **+1 word** more, and is higher on 54 of 87 pairs. The item-pool difference is real and negligible.

**With those waves removed, the association is large.** Within a study and an age band, the rank correlation between being on the comprehension-less form and the spoken count is:

| Study   | 36–48 mo       | 48–60 mo       | 60+ mo          |
| ------- | -------------- | -------------- | --------------- |
| `it_01` | +0.39 (n = 55) | +0.73 (n = 42) | +0.55 (n = 23)  |
| `uk_01` | −0.02 (n = 38) | +0.42 (n = 25) | +0.37 (n = 111) |

with `us_01` at +0.18 (n = 44) in its only qualifying band, 0–24 months. In `it_01` at 48–60 months the median spoken count is **344 on the comprehension-less form against 46 on the comprehension-measuring one**. A +1-word item-pool difference cannot produce that.

**Matching age exactly is the version with no residual gradient, and it is far too thin to settle the question.** Eight strata qualify (same study, same recorded month, at least two rows on each form class): the comprehension-less form is higher in six and lower in two, median within-stratum difference +163 words, Wilcoxon signed-rank _p_ = 0.078. `it_01` is 5 of 5 in the same direction; `uk_01` is 1 of 3.

**Reading.** The form is chosen partly on the child's developmental level, not only on their age — which is exactly what a clinician or researcher should do, and exactly what makes the missingness non-ignorable given age and study alone. It is _not_ obviously non-ignorable given the child effects: VG16 carries `tau_subj_u` and `tau_subj_q`, so a child whose form was chosen on their standing has that standing partly represented. But for the 69 multi-wave children who never carry an understood count, that representation is thin: their comprehension child effect reaches the likelihood only through the marginal fallback branch, where spoken is modelled as $\text{BB}(N, p_U \cdot q, \kappa_S)$ and the data see only the _product_, so `delta_subj_u` and `delta_subj_q` are identified for them only up to it. They contribute nothing to the lag either way, because the lag needs an observed earlier comprehension count and they have none. The direction of any resulting bias is not established here and this note does not claim one; what it establishes is that the assumption is not free, and the evidence in three of fifteen studies is directional rather than decisive.

## 4. What each registered restriction has to work with

Each variant's frame is rebuilt through its own definition, because a data restriction changes which waves exist and therefore changes the lag — subsetting the baseline's supporting rows would not show that.

| Variant            | Frame rows | Supporting rows | Children | Studies |
| ------------------ | ---------- | --------------- | -------- | ------- |
| baseline           | 1,708      | 473             | 248      | 8       |
| `conditional-only` | 1,708      | 361             | 238      | 8       |
| `lag-gap-12`       | 1,708      | 431             | 220      | 8       |
| `lag-continuity`   | 1,708      | 473             | 248      | 8       |
| `no-us01`          | 1,497      | 340             | 202      | 7       |
| `dse-native-only`  | **264**    | **80**          | **74**   | **2**   |
| `lag-same-form`    | 1,708      | 342             | 226      | 8       |

`conditional-only` is the **complete-pair check** the review asked for, and it is well powered: it keeps 361 of 473 supporting rows in all eight studies, dropping exactly the rows whose spoken count has no observed comprehension parent.

The **form-restricted check** was the problem. `dse-native-only` was the only one registered, and it leaves **80 supporting rows from 74 children in two studies** — whatever it returns is as much a statement about `uk_02` and `ie_02` as about the measurement scale, and the review said as much when it estimated the restriction would leave "only 81 active lag rows from 75 children in two studies" on the pre-correction frame, a figure the corrected frame reproduces almost exactly. So `("vg16", "lag-same-form")` was registered in the same change as this note, with the field `lag_same_form_only` it needs: it keeps only lags whose source and target waves were scored against the same checklist, which is **342 rows from 226 children in all eight contributing studies**, and drops exactly the 131 supporting rows (28%) that cross a form ceiling. Like `lag_max_gap_months` it drops the lag, not the row, so the measurement question is not confounded with a sample-size change; and it is applied after the source wave is chosen, so a row whose source used a different form loses its lag rather than falling back to an earlier same-form wave and silently acquiring a longer gap.

Neither form-restricted arm identifies a scale effect on its own. Together they bracket it: `dse-native-only` removes the harmonisation entirely on a narrow, study-confounded slice, `lag-same-form` removes only the _transitions_ on a wide one.

## What this changes

- #242's available-case item is answered rather than open: the missingness is classified, every non-supporting row is accounted for, and the ignorability condition has been probed rather than assumed.
- Two registered checks were added. `lag-same-form`, with the field it needs, because the audit showed the existing form-restricted arm could not carry the question; and the `beta-tight`/`beta-wide` prior-scale pair on the existing `beta_lag_sigma`, which is the last item on the review's own sensitivity list never to have been registered.
- Nothing here moves an estimate. `beta_lag` is unchanged at **+0.146, 89% ETI [0.041, 0.251]** from the 2026-09-06 `rep` fit, and remains withdrawn: the five — now eight — registered arms are still unfitted, understood PSIS-LOO is suppressed with no sequential replacement, and no wave-sequential recovery exists ([#289](https://github.com/dseinternational/vocabulary-growth/issues/289) tasks 3.7–3.9).

The generalisability statement the audit supports, and which VG16's report now carries, is narrower than "children with Down syndrome": the coefficient is estimated from 473 administrations of 248 children in 8 of 15 studies, over intervals of typically five to seven months, at target ages typically between two and four years, and from children whose comprehension was measured — which, in the three studies that switch forms, means disproportionately the less advanced ones.
