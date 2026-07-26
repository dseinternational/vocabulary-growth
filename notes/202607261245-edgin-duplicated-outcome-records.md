# Data defects in the us_01 (Edgin) subset

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Analysis and implementation note, 2026-07-26. The rules described here are implemented and active by default. They change the analysis frame, so the current fits are stale in substance and every model of record needs refitting before its numbers are quoted again.

> [!IMPORTANT]
> Extended the same day. The note began as an analysis of six Words & Gestures administrations with duplicated outcome columns. Following that up — the residual maximum of 152 spoken words at 8–14 months did not look survivable either — led to a systematic audit of the whole subset ([`scripts/audit_edgin_subset.py`](../scripts/audit_edgin_subset.py)), which found **four** defect classes affecting **32 of 196 administrations (16%)** and 31 of 119 children. §§1–7 record the original analysis; §§8–11 record the audit, the widened rules, and one earlier conclusion of mine that it overturned. The registered `us01-ceiling-excluded` sensitivities are retired as a consequence (§10).

> [!IMPORTANT]
> Extended again, 2026-07-26, after the study owner established with the source author that **the original data files no longer exist**. That closes repair at source, and prompted a provenance check which found that `us_01` is not a study file at all but a subset of the public Wordbank export — so the cohort's other condition groups are available as internal controls. §13 records what that showed: the ceiling-saturation finding is far stronger than §9 claimed, but **§9's characterisation of it was wrong in three specifics**, and two reference figures quoted in §9 and in `methods-data.qmd` were wrong. It also records a contamination §§1–12 missed entirely — two Edgin rows were in the typically-developing reference pool the exclusions are benchmarked against. §14 then registers the reinstatement sensitivity, which is now the only published check on an exclusion that can never be confirmed at source.

## Summary

A systematic audit of the `us_01` (Edgin/Wordbank) Down syndrome subset finds **four independent defect classes affecting 32 of 196 administrations (16%)** and 31 of 119 children: outcome-column duplication (8), saturation at the form maximum (19, of which 13 fall inside a run of 21 consecutive ceiling records in identifier order — a run that crosses out of the subset entirely, §13), longitudinal collapse (12 within the rule's floor and age scope, overlapping the others and contributing 1 uniquely), and one administration recorded twice. Two further checks came back clean — no count exceeds its form's ceiling and no form was administered outside its age band.

Three rules now mask these by default, each with an `include_*` flag for sensitivity. Every exclusion falls in `us_01`; the pool goes from 1,219 to 1,218 rows, understood observations from 680 to 671, and spoken from 1,145 to 1,114. §11 tabulates the rules.

The rules are **conditioned on age, not on source study**. A production count close to comprehension, or close to the form ceiling, is ordinary in an older child with a large vocabulary and impossible in an infant; conditioning on age states the developmental fact rather than asserting that one dataset is untrustworthy, and it leaves the legitimate older records untouched.

Deliberately **not** caught, and now flagged rather than silently included: two administrations with a high infant comprehension count but a normal production gap, retained on the study owner's judgement that they are clinically unusual but should not be excluded; and one Words & Sentences record of 406 words at 23 months with no later administration to contradict it. All three are sensitivity targets rather than defects.

> [!NOTE]
> §§1–7 below are the original six-record analysis, kept as written because the reasoning about method — signature rather than threshold, age-conditioned rather than study-scoped — is what the audit went on to generalise. Where they are superseded, §§8–14 say so: the rule now catches eight rather than six Words & Gestures records (§8), and the claim in §1 that the Words & Sentences ceiling block is right-censoring is **wrong and corrected in §9**, on evidence restated in §13.

## 1. Why not a threshold on extreme values

The proposal that prompted this analysis was to censor implausibly high infant counts — for example, anything above 200 words understood or spoken. That rule is worse than the one adopted, for three reasons.

It is **selection on the outcome**, which is precisely the objection that removed this dataset's previous filter. `methods-data.qmd` records that the former `us_01` rule retaining only production counts at or below 100 words "had no documented measurement or sampling rationale", excluded 8 of 87 Words & Gestures and 24 of 109 Words & Sentences records, and conflicted with the source study's own reported subgroup mean of 275.5 words (SD 198.2). Reintroducing a cap at a higher number would repeat that error rather than correct it.

It **mis-targets**. Benchmarking every `us_01` administration above 200 words against typically-developing children at the same age on the same form, a `> 200` rule would have caught 31 administrations: the 6 genuine defects, 2 records that sit at the 48th–50th typically-developing percentile with an entirely normal production gap, and roughly 20 Words & Sentences records at the 680-item form ceiling which I then took to be right-censoring rather than error. **That reading was wrong — see §9**; the mis-targeting argument against a bare threshold stands regardless, since a threshold would still have caught the two valid records.

And it has **no mechanism**. A threshold asserts that a number is too big; it does not say what went wrong. The signature rule names a specific failure — two columns collapsing onto one value — which is falsifiable, has a measurable false-positive rate, and can be confirmed or refuted outright from item-level responses.

## 2. The six records

All are `us_01`, form WG (396 items). Percentiles are against typically-developing American-English children on the same form within ±1 month of the same age (reference _n_ per cell 649–1,923).

| child | age (mo) | understood | spoken | ratio | understood TD %ile | spoken TD %ile |
| ----- | -------- | ---------- | ------ | ----- | ------------------ | -------------- |
| 81124 | 11       | 319        | 313    | 0.981 | 96.6               | 99.7           |
| 81131 | 12       | 190        | 173    | 0.911 | 88.0               | 99.0           |
| 81091 | 12       | 386        | 385    | 0.997 | 98.1               | 100.0          |
| 81122 | 14       | 350        | 348    | 0.994 | 97.8               | 99.7           |
| 81132 | 17       | 235        | 220    | 0.936 | 61.5               | 96.9           |
| 81114 | 18       | 396        | 396    | 1.000 | 98.8               | 100.0          |

Note the shape of the last row: 396 is the entire form, both columns at ceiling. And note child 81132, whose comprehension is unremarkable at the 61st typically-developing percentile while its production sits at the 97th — the production value is the anomalous one there.

## 3. Three independent lines of evidence

**The pattern is rare where it can be checked at scale.** Among 2,480 typically-developing WG administrations with comprehension of at least 100 words, only **17 (0.69%)** have production at or above 0.9 of comprehension. This is the rule's false-positive rate on a large reference sample, and it is the reason the ratio threshold is defensible rather than arbitrary. Six such records among 87 Down syndrome WG administrations is a rate of 6.9%, an order of magnitude higher — in the population where comprehension leading production is _most_ strongly expected.

**The production levels are impossible against an external benchmark.** Berglund et al. (2001), 330 children with Down syndrome on a 710-item Swedish CDI and independent of this training data, give median spoken vocabulary of approximately zero words at 12 months and about 10 words at 24 months, with 53% passing a 10-word threshold only by 24 months (`docs/models/PRIORS.md`). The flagged records claim 173, 220, 313, 348, 385 and 396 words spoken between 11 and 18 months. This is not a tail of a distribution; it is off the map, and the evidence is external rather than in-sample.

**Every affected child with a second administration contradicts the flagged one.** Three of the six have another record:

| child | flagged administration | other administration | flagged ratio | other ratio |
| ----- | ---------------------- | -------------------- | ------------- | ----------- |
| 81124 | 11 mo: U 319, S 313    | 17 mo: U 110, S 9    | 0.98          | 0.08        |
| 81091 | 12 mo: U 386, S 385    | 18 mo: U 31, S 4     | 1.00          | 0.13        |
| 81131 | 12 mo: U 190, S 173    | 18 mo: U 166, S 15   | 0.91          | 0.09        |

In each case the _other_ record shows an ordinary comprehension–production gap. For 81124 and 81091 comprehension also falls implausibly with age — by 209 and 355 words over six months — so in those two records neither column is defensible. For 81131 comprehension is roughly stable (190 → 166) while production collapses (173 → 15), which points specifically at the production column having been overwritten.

Since the direction of the overwrite differs across records and cannot be recovered from aggregate totals, **both counts are masked** rather than one repaired. Half-repairing on a guess would reintroduce exactly the outcome-dependent editing that the `GREATEST` fix in #182 removed from `ie_01`.

## 4. Why the rule is age-conditioned

Applying the ratio and count conditions — at the final 0.75 ratio — across the whole pooled Down syndrome sample matches 52 of 678 paired records:

| age band (months) | matching records | all paired records |
| ----------------- | ---------------- | ------------------ |
| ≤ 18              | 8                | 147                |
| 19–36             | 3                | 289                |
| 37–60             | 26               | 196                |
| 60+               | 15               | 46                 |

The 41 records at 37 months and above are legitimate: an older child with a large vocabulary who says most of what they understand is the expected end state of the developmental process these models describe. Masking them would delete real data — a fifth of all paired records above 37 months — and bias the production trajectory downward at exactly the ages where the `q` trajectory is most informative.

The boundary is clean. The nearest match above the cut is a `uk_02` record at 35 months (understood 407, spoken 367) — entirely plausible. Between 19 and 24 months the maximum spoken count in the whole pool is 100 words and the maximum ratio is 0.49, so nothing approaches the signature just above the threshold. Conditioning on age rather than on study identity is also better science: it states a developmental fact about when the pattern is impossible, rather than asserting that one dataset is untrustworthy.

## 5. What is retained, and why

Two `us_01` administrations at 18 months have high comprehension for an infant with Down syndrome but an entirely normal production gap:

| child | age (mo) | understood | spoken | understood TD %ile | ratio |
| ----- | -------- | ---------- | ------ | ------------------ | ----- |
| 81092 | 18       | 213        | 31     | 48.2               | 0.15  |
| 81106 | 18       | 217        | 22     | 49.8               | 0.10  |

Their comprehension sits at the median for typically-developing children of the same age, their production gap is ordinary, and 81106 has a coherent trajectory across two visits (comprehension 65 at 13 months rising to 217 at 18). Statistically there is no case for excluding them, and a `> 200` threshold would have removed the best-behaved young comprehension records in the Down syndrome pool.

The study owner's assessment is that these are **highly unusual and clinically unlikely, but should not be ruled out**. That judgement is recorded here rather than resolved silently, and it is now actionable in two ways: §8.8 of the Route 1 pre-specification refits without them, and a `us01-young-comprehension-excluded` sensitivity should be registered for VG10 and VG15 when the sensitivity suite is next run, so any young-comprehension conclusion can be shown not to hinge on two records.

Also retained, and unrelated to this rule: the roughly 20 Words & Sentences administrations at the 680-item form ceiling. These are genuine right-censoring, not error, and remain covered by the existing `us01-ceiling-excluded` sensitivity.

> [!CAUTION]
> Both sentences above are **wrong**, and are kept only because §§1–7 are preserved as written. Those records are not right-censoring but invalid values (§9, restated on firmer evidence in §13), they are now masked by default rather than retained, and the `us01-ceiling-excluded` sensitivity no longer exists (§10).

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

**Refit required.** The analysis frame has changed, so all fits are stale in substance. The six records sit where the Down syndrome data are thinnest and most influential — before this change, the maximum understood count below 15 months across the _entire_ pooled sample was 386, one of these records, so they were disproportionately setting the young comprehension level. Their removal lowers the maximum understood at 8–14 months to 169 and the maximum spoken to 152.

**It matters more for Route 1 than for the aggregate models.** The aggregate likelihoods see totals; an item response theory model sees response _patterns_, and a duplicated column is a systematically wrong pattern that would bias the Down syndrome difficulty estimates directly. Six of 53 WG contributors is over a tenth of the comprehension sample. Fixing this before the item-level pull is why the rule is implemented now rather than after.

**The mechanism can be confirmed directly, and the finding is falsifiable.** Item-level responses settle it outright: a duplicated column appears as two identical response vectors. §8.7 of the Route 1 pre-specification requires reporting, for each of the six, whether the vectors are identical, near-identical or unrelated, and logging the result as confirming or refuting this signature. A refutation reinstates the records.

**Open follow-up.** Masking the six leaves a maximum spoken count of 152 words at 8–14 months, which remains high against the Berglund benchmark but does not match this signature (its production-to-comprehension ratio is below the threshold). Whether that reflects a different defect, genuine cohort variation, or the pooled sample's selection toward intervention-exposed families is not addressed here.

---

## 8. The systematic audit

Chasing one residual — after masking the six, the maximum spoken count at 8–14 months was still 152 words, against a Berglund median near zero at 12 months — showed that hand-finding defects one at a time was the wrong method. [`scripts/audit_edgin_subset.py`](../scripts/audit_edgin_subset.py) instead enumerates every defect class the aggregate data can expose, and is committed so any later change can be re-checked by re-running it. It is descriptive: it reports and exits zero, whatever it finds.

Findings, over 196 administrations from 119 children:

| check                                   | flagged |
| --------------------------------------- | ------- |
| values impossible for their form        | 0       |
| form administered outside its age range | 0       |
| duplicate administrations               | 2       |
| outcome-column duplication              | 8       |
| saturation at the form maximum          | 19      |
| contiguous child-id batch               | 13      |
| longitudinal collapse                   | 12      |

The union is **32 administrations (16%)** affecting **31 children (26%)**. The implemented rules mask 30 of them and drop one of the two duplicate copies, so 31 records change. Two checks came back clean, which is worth recording: no count exceeds its form's ceiling, none is non-integer, and no form was administered outside its intended age band.

Four things the audit established that the hand analysis had missed or got wrong.

**My ratio threshold was in the wrong place.** The audit computes the largest gap in the ratio distribution rather than taking a threshold on faith. Among Words & Gestures administrations with comprehension ≥ 100 the ratios descend 1.00, 1.00, 0.99, 0.98, 0.94, 0.91, 0.90, 0.86 and then fall to 0.55 — a gap of 0.306, the largest in the distribution. My original 0.90 cut ran through the middle of that cluster and missed two records; any cut inside the gap separates it identically. The rule now uses **0.75**, and the two recovered records are exactly the ones that prompted this section:

```
child 81127:  11mo C=169 P=152  (ratio 0.899)  ->  17mo C=105 P=2
child 81119:  12mo C=2   P=0                   ->  18mo C=156 P=134  (ratio 0.859)
```

The 152 words at 11 months was this defect, sitting one thousandth below my cut. Note 81119's shape: the _later_ record is the anomalous one, production rising 0 → 134 by 18 months.

**An exact duplicate row.** One administration is recorded twice, identically — 60 words understood and 1 spoken at 11 months. A repeated row double-weights that observation in every likelihood and, in the random-effect models, makes a single-visit child look like a repeated-measures one. It is now de-duplicated on study, subject, age and every outcome, so genuine repeat visits, which differ in age, are untouched.

**A longitudinal-collapse signature that generalises.** Vocabulary does not shrink, so a count exceeding the same child's later count several-fold is unambiguous. This catches most of the other defects independently, and one record nothing else caught: 454 words at 18 months against 35 at 24. Two calibrations matter. A **floor** is needed — without one the rule fires on trivial pairs such as five understood words falling to one, which is noise. And an **age scope** is needed: at older ages an apparent decline can arise from a form change or from noise in large counts, and the audit found two such records outside `us_01` (a uk_01 record at 76 months, an ie_02 record at 45) which are left for separate investigation rather than masked by a rule whose justification is developmental.

**The Words & Sentences ceiling block is not right-censoring.** This overturns a conclusion I stated earlier in the review, and §9 sets it out.

## 9. Correction: the ceiling block is invalid, not censored

I previously characterised the ~20 Words & Sentences administrations at the 680-item ceiling as "genuine right-censoring, not error", already covered by the registered `us01-ceiling-excluded` sensitivity. That was wrong on both counts, and it is worth stating plainly because the reading had consequences: right-censoring means the child knows _at least_ 680 words, which is a real if incomplete measurement, whereas an invalid value is not a measurement at all — and a _sensitivity_ leaves the records in the primary analysis by default.

The block is two distinct defects.

**Eight young records, ages 17–19, production 641–680.** Seven of the eight have a later administration and every one collapses:

| child | flagged    | later     | factor |
| ----- | ---------- | --------- | ------ |
| 81322 | 17 mo: 656 | 23 mo: 12 | 55×    |
| 81344 | 17 mo: 680 | 23 mo: 4  | 170×   |
| 81345 | 17 mo: 641 | 24 mo: 4  | 160×   |
| 81370 | 17 mo: 680 | 23 mo: 5  | 136×   |
| 81354 | 18 mo: 668 | 23 mo: 3  | 223×   |
| 81359 | 18 mo: 680 | 24 mo: 25 | 27×    |
| 81342 | 19 mo: 680 | 24 mo: 13 | 52×    |

Seven for seven. For scale, **no typically-developing child** of the 2,489 Words & Sentences administrations at 16–19 months reaches the ceiling; their maximum is 661.

> [!NOTE]
> That sentence originally read "no typically-developing child of 1,469 aged 16–19 months … their maximum is 643". Both figures were wrong: 1,469 is the audit's **Words & Gestures** reference count at those ages, which I attached to a Words & Sentences claim, and 643 appears nowhere in the audit output. The claim itself — that no typically-developing child reaches the Words & Sentences ceiling at 16–19 months — holds, at 0.00% of 2,489 with a maximum of 661.

**Thirteen records at exactly 680, adjacent in the dataset's identifier ordering**, ids 81207–81241, ages 24–30. No child among them has any other administration anywhere in the dataset. Uniformity at the form maximum across a run of adjacent records, with no corroborating data, is a preparation-batch signature rather than thirteen exceptionally able children. Berglund's most able child of 330 reached 668 words at _48_ months; this block asks us to accept thirteen children reaching 680 by 24–30 months, with nothing else recorded about any of them.

> [!CAUTION]
> This paragraph originally described "a contiguous child-id block … a disjoint run of consecutive ids" with "an 81-id gap to the next child present". **Two of those three claims are false**, and the third is understated. The identifiers are _not_ consecutive integers — 22 integers inside 81207–81241 are absent from the Wordbank export altogether — and the "81-id gap" was an artefact of looking only at the Down syndrome subset: the next identifier present in the cohort is 81246, an autism-group child, also at exactly 680. What is true, and stronger, is that the records are adjacent in _identifier order_, and the run does not stop where the Down syndrome subset does. §13 sets this out with an exact probability.

## 10. The retired sensitivity, and the inverse that replaces it

Because those records are now masked by default, the registered `us01-ceiling-excluded` variants for VG10 and VG15 could no longer do anything: they excluded records already excluded. A registered check that cannot fail is worse than no check, so both entries are removed from [`sensitivity/registry.py`](../src/vocab_growth/sensitivity/registry.py) with the reasoning recorded in place.

The live question is the inverse — what changes if the exclusion is _wrong_ — and it is now registered as `us01-implausible-reinstated` for both models (§14). The `exclude_us01_spoken_ceiling` flag and its helper are retained, and remain functional on a reinstated frame, so the decision can be reverted without re-deriving anything.

## 11. The rules as implemented

Three rules, each independently defensible, applied in [`load_combined_data`](../src/vocab_growth/data_utils.py) with de-duplication first so a repeated row cannot affect the within-child comparisons the later rules make:

| rule                                          | signature                                                                                                               | effect                              |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| `drop_duplicate_administrations`              | identical study, subject, age and outcomes                                                                              | 1 row removed                       |
| `mask_duplicated_outcome_administrations`     | `spoken >= 0.75 * understood`, `understood >= 100`, `age <= 18`                                                         | 8 administrations, both counts      |
| `mask_implausible_production_administrations` | `spoken >= 0.9 * form ceiling` at `age <= 30`, **or** a ≥5× collapse against a later count of the same child (floor 50) | 30 administrations, production only |

Effect on the pool: 1,219 → 1,218 rows; understood observations 680 → 671; spoken 1,145 → 1,114. Every exclusion falls in `us_01`, which retains 165 of its 195 spoken and 78 of its 86 understood observations. Each rule has an `include_*` flag for sensitivity, and eighteen regression tests pin the behaviour — including the age scope, the ratio gap, the collapse floor, and that the two retained borderline cases survive.

**Retained, and now flagged rather than silently included:** the two 18-month records with high comprehension and a normal production gap (§5), and a Words & Sentences record of 406 words at 23 months with no later administration to contradict it. All three are extreme against the external benchmark but carry no positive defect signature. Treating them as sensitivity targets rather than exclusions is the same judgement the study owner applied to the first pair.

## 12. Where this leaves the subset, and the open question

Thirty-two of 196 `us_01` administrations (16%) carry a defect, across four independent classes, in a subset that has already been patched three times — the removed `production <= 100` inclusion rule, the Words & Sentences comprehension proxy, and now these. The defects are not random noise: two of the four classes have batch or column-level structure, which points at data preparation rather than at parent report.

That pattern is itself the finding. It would be worth putting the audit output to the source team: the contiguous-id block in particular looks like something they could confirm or explain quickly, and the item-level pull would settle the duplication mechanism outright, since a duplicated or auto-filled column appears as two identical response vectors. Inferring these rules from aggregate data is the second-best route and is only necessary while that confirmation is unavailable.

> [!IMPORTANT]
> **This route is now closed.** The study owner raised the audit with the source author, who no longer has access to the original data files. The rules in §11 are therefore permanent rather than provisional: nothing further will arrive to confirm or refute them at source. §13 records what follows from that — including two evidential routes that do _not_ depend on the source team, one of which has already strengthened the case considerably.

Residual after all three rules, and not explained: the maximum spoken count at 15–18 months is 68 words and at 19–24 months 406, both still high against a Berglund median of about 10 at 24 months. The 406 is the retained borderline record. Whether the remaining elevation reflects genuine cohort variation, this pooled sample's selection toward intervention-exposed families, or a further defect class the aggregate data cannot expose is not resolved here.

---

## 13. After the source route closed: a provenance check, a stronger finding, and a contamination

The source author no longer holds the original files (§12). Before accepting that the rules can never be checked, it was worth asking precisely what `us_01` _is_ — and the answer changes the picture, because the premise of §12's recommendation was itself partly wrong.

### 13.1 `us_01` is not a study file

There is no `data/vocab_data_us_01.csv` supplied by the source team — every other study has one, with a companion `.md` codebook, and `us_01` has neither. It is built in [`vocab_combined_view_sql`](../src/vocab_growth/data_utils.py) by filtering `data/wordbank_administration_data.csv` — the public Wordbank by-child export, downloaded 15 June 2026 — to `dataset_name = 'Edgin'`, American English, `health_conditions = 'down syndrome'`. The numeric identifiers throughout this note are Wordbank's, not the study's.

> [!CAUTION]
> The first clause is **wrong**, and was asserted without checking `git ls-files`. `data/vocab_data_us_01.csv` does exist and is tracked: 66 rows, 42 children, Words & Gestures only, ages 11–18, comprehension at most 96 and production at most 12, with hashed subject identifiers rather than Wordbank integers. Its only commit is `7eafe97`.
>
> What is true is that **nothing reads it**. It is absent from `prepare_data.py`'s `DATASETS` registry and unreferenced anywhere else in the repository, and it has no companion `.md` codebook. It is an orphaned early extract, superseded by the Wordbank derivation described above — 196 administrations from 119 children, ages 11–30, across both the 396- and 680-word forms — and it cannot even be reconciled with it, because the two use different identifier schemes.
>
> The provenance conclusion is unaffected: the `us_01` that §§8–11 analyse is the Wordbank derivation, and the numeric identifiers throughout this note are Wordbank's. But the file leaves a live trap — a tracked CSV named exactly like every other study's source file, which anyone could add to the registry and thereby double-count `us_01` under two incompatible identifier schemes.

Two consequences follow, and they point in opposite directions.

The **loss is narrower than it looked**: we were never analysing the source team's files, so what has become unavailable is the ability to _explain_ and _repair_ the defect, not the ability to _detect_ it. Everything in §§8–11 was derived from a public file that is still there and still re-checkable.

The **available evidence is wider than §§8–11 used**. The Edgin dataset in Wordbank is not only the Down syndrome group. It holds 251 children in four condition groups — Down syndrome 119, unlabelled 108, pre-term 17, autism 7 — and `us_01` is one of the four. The other three are not in any analysis pool, but they passed through the _same_ preparation. A defect introduced in preparation has no reason to respect the condition label, so those groups are internal controls that the `us_01` filter had been hiding. The audit now runs its batch check over the whole cohort for exactly this reason.

### 13.2 The ceiling saturation, restated correctly

Ordering all 235 Edgin Words & Sentences records by identifier and marking those at exactly 680:

| statement                | §9 claimed           | actually                                                                      |
| ------------------------ | -------------------- | ----------------------------------------------------------------------------- |
| identifiers in the block | consecutive integers | **not** consecutive — 22 integers inside 81207–81241 are absent from Wordbank |
| what follows the block   | an 81-identifier gap | identifier 81246, an autism-group child, **also at exactly 680**              |
| length of the run        | 13                   | **21**, ids 81207–81299                                                       |
| groups spanned           | Down syndrome only   | Down syndrome, autism **and** unlabelled                                      |

The corrected statement: **21 consecutive records in identifier order, every one at exactly the 680-word form ceiling, ages 24–30, crossing two condition-group boundaries, and not one of the 21 children has any other administration anywhere in the dataset.**

That run is not a near-miss on chance. Placing the 27 ceiling records at random among the 235 Words & Sentences records, the probability of _some_ run of 21 or longer is **1.3 × 10⁻²²** — computed exactly rather than simulated, by counting the arrangements in which every gap between unflagged records holds at most 20 flags. Even the truncated Down-syndrome-only view that §9 saw gives 3.4 × 10⁻¹¹. Because this figure is published and can no longer be corroborated at source, the `_run_probability` helper is pinned by [`tests/test_audit_edgin_subset.py`](../tests/test_audit_edgin_subset.py) against five hand-countable cases _and_ against brute-force enumeration of every arrangement for all layouts up to ten records.

The reference comparison also has to be made at the right ages, which §9 and `methods-data.qmd` both failed to do — they quoted the 16–19 month reference against a block aged 24–30. Making it correctly does not weaken the case:

| group                                 | Words & Sentences records, 24–30 months | at exactly 680 | median production |
| ------------------------------------- | --------------------------------------- | -------------- | ----------------- |
| Edgin Down syndrome (`us_01`)         | 37                                      | **13 (35.1%)** | 35                |
| typically developing (Edgin excluded) | 5,135                                   | 24 (0.47%)     | 399               |

A third of the Down syndrome records at these ages sit at the absolute ceiling of the instrument, while the median of the same 37 records is 35 words. No developmental process produces that shape; a preparation step that writes the form maximum into a block of rows does.

So the masking decision in §11 stands on materially better evidence than the reasoning that produced it — which is the outcome to hope for from a falsification attempt, but not the one to assume.

### 13.3 A contamination §§1–12 missed

The typically-developing pool is built by `load_data` from the same Wordbank export, requiring `typically_developing` true and no health condition recorded. **Two Edgin rows satisfy that filter**, and one of them is child 81299 — a Words & Sentences record at exactly 680, at 29 months, the last member of the run of 21.

Edgin is the only clinical cohort in the export that leaks this way: it contributes 2 of its 435 rows (0.5%) to the reference pool, where every other contributing dataset gives at least 10% of its rows. So excluding it is not an ad-hoc carve-out for an inconvenient dataset; it is the one instance of a clinical cohort whose non-affected children slip through a filter designed for norming samples.

`TD_POOL_EXCLUDED_DATASETS` in [`data_utils.py`](../src/vocab_growth/data_utils.py) now bars it, removing 1 row from the bivariate pool and 2 from the spoken-inclusive pool, out of 15,379. **No estimate moves.** The reason to do it anyway is that the reference pool is the benchmark the Down syndrome exclusions are justified against, and a record we exclude as invalid on one side of that comparison cannot sit inside the other side. A regression test pins it, and the audit script's reference pool excludes Edgin too — its previously reported reference counts were inflated by these two rows.

### 13.4 What can still be established, and what cannot

**Cannot, now:** why the defect exists, whether the 680 block was a merge artefact or a coding convention for "completed the form", and which column was overwritten in the eight duplicated-outcome records. These were only ever answerable at source.

**Can, without the source team:** the item-level check. Wordbank publishes item-level ("instrument data") exports alongside the by-child summaries, and the Route 1 pull is already specified to take them from Wordbank rather than from any study team (pre-specification §3). If the Edgin dataset has an item-level export, a duplicated or auto-filled column appears as two identical response vectors, and a form written to its maximum appears as 680 items all marked known — both decisive, both from a public file. Whether that export exists for this dataset is unknown and is a checkable ingest question, not an assumption; §8.7 of the pre-specification should record "no item-level export available" as a possible and reportable outcome rather than presuming the check can be run.

**Also can, and cheaply:** the internal controls. The unlabelled and pre-term groups are in the same file and the same preparation. Pre-term shows no ceiling saturation at all (maximum production 321 across 17 records), which is informative — the defect is not uniform across the cohort, so it is not a whole-file transformation.

The honest summary for publication is that four defect classes in 32 of 196 administrations were identified from aggregate data, that the strongest of them is established at a probability of 10⁻²² against chance, and that the source of the error cannot now be determined. The remaining `include_*` flags are what make that reviewable: they let a reader see what the conclusions would have been had we judged wrongly. §14 turns the most consequential of them into a registered sensitivity, so that this is a reported result rather than an available option.

---

## 14. The reinstatement sensitivity, registered

A flag a reader could set is not evidence a reader has been given. Because the exclusion can no longer be checked at source, the inverse of the retired variant is now registered and runs in the refit: **`us01-implausible-reinstated`** for VG10 and VG15, setting `include_implausible_production=True`.

|                      |                                                                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| what it does         | reinstates the production counts masked by `mask_implausible_production_administrations` and refits |
| effect on the frame  | spoken observations 1,114 → 1,136 (VG10); marginal spoken 947 → 969 (VG15)                          |
| why these two models | they carry the headline joint trajectories, and mirror the footprint of the variants retired in §10 |

**Why 22 and not 30.** The rule masks 30 administrations, but 8 of those are also caught by the duplicated-outcome rule (§8), which stays active and has its own separate flag. Reinstating one rule's exclusions does not reinstate another's, and it should not — the two rest on independent evidence.

**Three implementation points worth recording**, because each was a way to get a plausible-looking but useless sensitivity:

The flag has to reach the frame. `load_data` previously had no way to pass the `include_*` flags through to `load_combined_data`, so a `ModelDefinition` field alone would have produced a variant that changed the config name and nothing else — the retired variants' fault in a new costume. `load_data` now forwards all three, and **raises** if one is passed for the typically-developing population, where the flags name defect classes that do not exist.

The count has to be reported. Each engine prints `us_01 implausible production reinstated`, derived by differencing the two loader paths rather than reimplementing the signature, so it cannot drift from the rule it describes. A run where that line reads 0 is a failure to investigate, not a robustness result — which is the check the retired variants lacked and could not have passed.

The count has to match the frame the engine actually loaded. The first implementation read `definition.max_age_months`, which `JointModelDefinition` does not define, so VG15 raised `AttributeError` the moment the engine ran — invisible to every unit test until [`tests/test_implausible_production_sensitivity.py`](../tests/test_implausible_production_sensitivity.py) exercised both engines' preparation. Two of the three engines pass no age bound to `load_data`, so the count is now taken over the same unbounded frame; reporting a bounded count against an unbounded fit would have understated what was reinstated.
