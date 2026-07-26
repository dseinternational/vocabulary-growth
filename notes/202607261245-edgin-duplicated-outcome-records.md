# Duplicated outcome columns in six us_01 infant administrations

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Analysis and implementation note, 2026-07-26. The rule described here is implemented and active by default. It masks 12 values (6 administrations × 2 outcomes) and changes the analysis frame, so the current fits are stale in substance and every model of record needs refitting before its numbers are quoted again.

## Summary

Six `us_01` (Edgin/Wordbank) Words & Gestures administrations between 11 and 18 months record production within 1–9% of comprehension — between 173 and 396 words *said* by an infant with Down syndrome. Three independent lines of evidence identify these as a data-preparation artefact in which one outcome column was written over the other, not as exceptional children. Both counts are now masked by default, with the row retained and a flag to reinstate them for sensitivity.

The rule is **conditioned on age, not on source study**. The same production-to-comprehension ratio above 18 months describes a child who says most of what they understand, which is ordinary; 21 of the 27 paired records in the pool meeting the ratio and count conditions are of that kind and are untouched.

Deliberately **not** caught: two administrations with a high infant comprehension count but a normal production gap. These are retained on the study owner's judgement that they are clinically unusual but should not be excluded, and are now a registered sensitivity target rather than a defect.

## 1. Why not a threshold on extreme values

The proposal that prompted this analysis was to censor implausibly high infant counts — for example, anything above 200 words understood or spoken. That rule is worse than the one adopted, for three reasons.

It is **selection on the outcome**, which is precisely the objection that removed this dataset's previous filter. `methods-data.qmd` records that the former `us_01` rule retaining only production counts at or below 100 words "had no documented measurement or sampling rationale", excluded 8 of 87 Words & Gestures and 24 of 109 Words & Sentences records, and conflicted with the source study's own reported subgroup mean of 275.5 words (SD 198.2). Reintroducing a cap at a higher number would repeat that error rather than correct it.

It **mis-targets**. Benchmarking every `us_01` administration above 200 words against typically-developing children at the same age on the same form, a `> 200` rule would have caught 31 administrations: the 6 genuine defects, 2 records that sit at the 48th–50th typically-developing percentile with an entirely normal production gap, and roughly 20 Words & Sentences records at the 680-item form ceiling that are right-censoring rather than error and are already handled by the registered `us01-ceiling-excluded` sensitivity.

And it has **no mechanism**. A threshold asserts that a number is too big; it does not say what went wrong. The signature rule names a specific failure — two columns collapsing onto one value — which is falsifiable, has a measurable false-positive rate, and can be confirmed or refuted outright from item-level responses.

## 2. The six records

All are `us_01`, form WG (396 items). Percentiles are against typically-developing American-English children on the same form within ±1 month of the same age (reference *n* per cell 649–1,923).

| child | age (mo) | understood | spoken | ratio | understood TD %ile | spoken TD %ile |
| ----- | -------- | ---------- | ------ | ----- | ------------------ | -------------- |
| 81124 | 11       | 319        | 313    | 0.981 | 96.6              | 99.7           |
| 81131 | 12       | 190        | 173    | 0.911 | 88.0              | 99.0           |
| 81091 | 12       | 386        | 385    | 0.997 | 98.1              | 100.0          |
| 81122 | 14       | 350        | 348    | 0.994 | 97.8              | 99.7           |
| 81132 | 17       | 235        | 220    | 0.936 | 61.5              | 96.9           |
| 81114 | 18       | 396        | 396    | 1.000 | 98.8              | 100.0          |

Note the shape of the last row: 396 is the entire form, both columns at ceiling. And note child 81132, whose comprehension is unremarkable at the 61st typically-developing percentile while its production sits at the 97th — the production value is the anomalous one there.

## 3. Three independent lines of evidence

**The pattern is rare where it can be checked at scale.** Among 2,480 typically-developing WG administrations with comprehension of at least 100 words, only **17 (0.69%)** have production at or above 0.9 of comprehension. This is the rule's false-positive rate on a large reference sample, and it is the reason the ratio threshold is defensible rather than arbitrary. Six such records among 87 Down syndrome WG administrations is a rate of 6.9%, an order of magnitude higher — in the population where comprehension leading production is *most* strongly expected.

**The production levels are impossible against an external benchmark.** Berglund et al. (2001), 330 children with Down syndrome on a 710-item Swedish CDI and independent of this training data, give median spoken vocabulary of approximately zero words at 12 months and about 10 words at 24 months, with 53% passing a 10-word threshold only by 24 months (`docs/models/PRIORS.md`). The flagged records claim 173, 220, 313, 348, 385 and 396 words spoken between 11 and 18 months. This is not a tail of a distribution; it is off the map, and the evidence is external rather than in-sample.

**Every affected child with a second administration contradicts the flagged one.** Three of the six have another record:

| child | flagged administration | other administration | flagged ratio | other ratio |
| ----- | ---------------------- | -------------------- | ------------- | ----------- |
| 81124 | 11 mo: U 319, S 313    | 17 mo: U 110, S 9    | 0.98          | 0.08        |
| 81091 | 12 mo: U 386, S 385    | 18 mo: U 31, S 4     | 1.00          | 0.13        |
| 81131 | 12 mo: U 190, S 173    | 18 mo: U 166, S 15   | 0.91          | 0.09        |

In each case the *other* record shows an ordinary comprehension–production gap. For 81124 and 81091 comprehension also falls implausibly with age — by 209 and 355 words over six months — so in those two records neither column is defensible. For 81131 comprehension is roughly stable (190 → 166) while production collapses (173 → 15), which points specifically at the production column having been overwritten.

Since the direction of the overwrite differs across records and cannot be recovered from aggregate totals, **both counts are masked** rather than one repaired. Half-repairing on a guess would reintroduce exactly the outcome-dependent editing that the `GREATEST` fix in #182 removed from `ie_01`.

## 4. Why the rule is age-conditioned

Applying the ratio and count conditions across the whole pooled Down syndrome sample matches 27 of 679 paired records:

| age band (months) | matching records | all paired records |
| ----------------- | ---------------- | ------------------ |
| ≤ 18              | 6                | 148                |
| 19–36             | 1                | 289                |
| 37–60             | 12               | 196                |
| 60+               | 8                | 46                 |

The 21 records at 37 months and above are legitimate: an older child with a large vocabulary who says most of what they understand is the expected end state of the developmental process these models describe. Masking them would delete real data and bias the production trajectory downward at exactly the ages where the `q` trajectory is most informative.

The boundary is clean. The nearest match above the cut is a `uk_02` record at 35 months (understood 407, spoken 367) — entirely plausible. Between 19 and 24 months the maximum spoken count in the whole pool is 100 words and the maximum ratio is 0.49, so nothing approaches the signature just above the threshold. Conditioning on age rather than on study identity is also better science: it states a developmental fact about when the pattern is impossible, rather than asserting that one dataset is untrustworthy.

## 5. What is retained, and why

Two `us_01` administrations at 18 months have high comprehension for an infant with Down syndrome but an entirely normal production gap:

| child | age (mo) | understood | spoken | understood TD %ile | ratio |
| ----- | -------- | ---------- | ------ | ------------------ | ----- |
| 81092 | 18       | 213        | 31     | 48.2              | 0.15  |
| 81106 | 18       | 217        | 22     | 49.8              | 0.10  |

Their comprehension sits at the median for typically-developing children of the same age, their production gap is ordinary, and 81106 has a coherent trajectory across two visits (comprehension 65 at 13 months rising to 217 at 18). Statistically there is no case for excluding them, and a `> 200` threshold would have removed the best-behaved young comprehension records in the Down syndrome pool.

The study owner's assessment is that these are **highly unusual and clinically unlikely, but should not be ruled out**. That judgement is recorded here rather than resolved silently, and it is now actionable in two ways: §8.8 of the Route 1 pre-specification refits without them, and a `us01-young-comprehension-excluded` sensitivity should be registered for VG10 and VG15 when the sensitivity suite is next run, so any young-comprehension conclusion can be shown not to hinge on two records.

Also retained, and unrelated to this rule: the roughly 20 Words & Sentences administrations at the 680-item form ceiling. These are genuine right-censoring, not error, and remain covered by the existing `us01-ceiling-excluded` sensitivity.

## 6. Implementation

`data_utils.mask_duplicated_outcome_administrations` masks `understood`, `spoken` and `produced` where all three conditions hold:

```text
age      <= DUPLICATED_OUTCOME_MAX_AGE_MONTHS  (18)
understood >= DUPLICATED_OUTCOME_MIN_UNDERSTOOD  (100)
spoken   >= DUPLICATED_OUTCOME_RATIO * understood  (0.9)
```

Applied in `load_combined_data` alongside the incomplete-administration mask, so every consumer of the Down syndrome pool sees the same frame; `include_duplicated_outcomes=True` reinstates the records for sensitivity. Rows are retained with missing outcomes so age coverage and provenance stay auditable, and the per-study masked-value counts are returned for the fit log. Effect on the pool: understood observations 680 → 674, spoken 1,145 → 1,139, all 12 masked values in `us_01`.

Six regression tests pin the behaviour: both counts masked and the row retained; the flag restoring exactly; the age condition (the identical ratio at 40 months is kept); the understood floor (an infant with 8 understood and 8 spoken is kept); the retained Group 2 shape; and the end-to-end count against the real database.

## 7. Consequences and follow-ups

**Refit required.** The analysis frame has changed, so all fits are stale in substance. The six records sit where the Down syndrome data are thinnest and most influential — before this change, the maximum understood count below 15 months across the *entire* pooled sample was 386, one of these records, so they were disproportionately setting the young comprehension level. Their removal lowers the maximum understood at 8–14 months to 169 and the maximum spoken to 152.

**It matters more for Route 1 than for the aggregate models.** The aggregate likelihoods see totals; an item response theory model sees response *patterns*, and a duplicated column is a systematically wrong pattern that would bias the Down syndrome difficulty estimates directly. Six of 53 WG contributors is over a tenth of the comprehension sample. Fixing this before the item-level pull is why the rule is implemented now rather than after.

**The mechanism can be confirmed directly, and the finding is falsifiable.** Item-level responses settle it outright: a duplicated column appears as two identical response vectors. §8.7 of the Route 1 pre-specification requires reporting, for each of the six, whether the vectors are identical, near-identical or unrelated, and logging the result as confirming or refuting this signature. A refutation reinstates the records.

**Open follow-up.** Masking the six leaves a maximum spoken count of 152 words at 8–14 months, which remains high against the Berglund benchmark but does not match this signature (its production-to-comprehension ratio is below the threshold). Whether that reflects a different defect, genuine cohort variation, or the pooled sample's selection toward intervention-exposed families is not addressed here.
