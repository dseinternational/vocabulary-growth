# The Edgin subset is age-truncated, and what the missing administrations turn out to be

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Date: 2026-08-03.

## Summary

`data/wordbank_administration_data.csv` gave us 196 Edgin Down syndrome administrations. The Wordbank database holds 347. The gap is an artefact of Wordbank's download page, not of the database.

Four things follow, and the order they were settled in matters — see §7, which records a reversal.

1. **Administrations above a form's age window are admitted.** For a Down syndrome cohort, giving an early-vocabulary form to a chronologically older child is developmentally appropriate rather than an error, and the age window governs whether Wordbank's _percentile norms_ apply — which this project does not use. These 50 Words & Gestures administrations at 19–27 months are `us_01`'s **only** comprehension observations in that band; before them the study contributed none, because everything it had at those ages was Words & Sentences, whose comprehension is a production proxy.
2. **The genuinely defective administrations are identified on their provenance, not their age.** 64 children whose _every_ record sits at the form ceiling are removed wholesale — the batch signature `202607261245` §13 first described. Age and count together cannot do this job: removing the near-ceiling rule's age bound would mask 19 apparently legitimate records in six other studies.
3. **Administrations below a form's floor are dropped.** 16 rows at 5–7 months, three of them reporting 236–368 words _spoken_, which no 6-month-old in any population produces.
4. **Wordbank issues a separate `child_id` per form.** The 119 apparent children were 53 Words & Gestures records plus 66 Words & Sentences records with no child linked across forms. They are 71 children, 46 of whom took both.

## 1. The truncation mechanism

Verified in source rather than inferred:

- `wordbankr::get_administration_data()` defaults to `filter_age = TRUE`, which filters `age >= age_min AND age <= age_max` from the instrument record (`langcog/wordbankr/R/wordbankr.R`, near line 404).
- The By-Child Summary Data app calls it **without** `filter_age = FALSE` (`langcog/wordbank-shiny/apps/admin_data/server.R`, line 11), and then builds its age slider from `min(admins$age)`/`max(admins$age)` of the already-filtered frame. The truncation therefore happens _before_ the slider exists: **no setting in the web UI can recover these rows.**
- `instruments/import_dataset.py` applies no age filter, so the rows are in the database.

The counts reproduce exactly from `langcog/wordbank/raw_data/English_American_{WG,WS}/English{WG,WS}_Edgin_data.csv`, restricting `DevStatus`/`DevelopmentalDiagnosis` = 1 to each form's window: 87 Words & Gestures + 109 Words & Sentences = 196, matching the export cell for cell.

For the typically-developing pool the same truncation is _appropriate_ — the forms are used within the range they were normed for — so `wordbank_administration_data.csv` is kept as-is and remains the TD source.

## 2. What the out-of-window administrations are, and how they are separated

149 out-of-window Down syndrome administrations (151 before the two empty forms are dropped).

|                                           | Words & Gestures      | Words & Sentences |
| ----------------------------------------- | --------------------- | ----------------- |
| Out-of-window administrations             | 89                    | 62                |
| Ages                                      | 5–7 and 19–173 months | 31–88 months      |
| At or above 0.9 × form ceiling            | 25                    | **62 (all)**      |
| At exactly the ceiling                    | 23                    | 61                |
| Caught by the near-ceiling rule as scoped | 2                     | **0**             |

The last row is what makes an age-based rule tempting and a count-based rule insufficient. `IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS` scopes the near-ceiling rule to `age <= 30`, so of the 87 ceiling-saturated administrations it catches two.

**But that age scope is correct, and removing it is worse.** Above 30 months a near-ceiling count is ordinary rather than suspect: an eight-year-old with Down syndrome knowing 658 of 690 words is expected. Dropping the bound would newly mask 19 apparently legitimate records across six studies — uk_01 at 115 months with 658 of 690, ie_01 at 69 with 741 of 810, es_01 at 54 with 637 of 651, it_01 at 94 with 654 of 670, us_02 at 75 with 381 of 396. Age and count together cannot separate those from the Edgin batch.

**What does separate them is provenance.** The batch children have no non-ceiling record of their own — a fact about how the data were prepared, not about the values, so it is not selection on the outcome. `exclude_ceiling_only_children` removes 64 children and 98 administrations on that criterion: 23 Words & Gestures at 39–173 months all at exactly 396 spoken, 62 Words & Sentences at 31–88 months with 61 at exactly 680, and 13 Words & Sentences at 24–30 months whose counts every other rule already masked. Four administrations at exactly 680 between 31 and 64 months belong to children who _do_ have other records; those children are kept and the near-ceiling rule handles the counts.

Stated plainly, the rule is not free: **23 of the removed administrations carry a live comprehension value**, all at exactly 396 on the 396-item form between 39 and 173 months. A child recorded as understanding every word _and_ saying every word at 173 months is the artefact, so they go with the rest of their record — but that is a real loss, not bookkeeping.

Because the criterion is applied to raw source counts before any masking, it also removes 14 children who were in the pool only as **phantoms**: every one of their counts was already masked, so they contributed a subject random effect informed by nothing but its prior. Of the 71 children the previous `us_01` pool reported, 57 had at least one live observation; the figure now is 58.

## 2a. Why the floor is enforced when the ceiling is not

The 16 below-floor administrations at 5–7 months are a different case from the above-window ones. Three report 236, 364 and 368 words _spoken_ at 6 months. Two of the same children show comprehension collapsing from 247–371 words at 6 months to 5–19 by 11–12 months. The block is unreliable — most likely mis-keyed ages — and the remaining rows in it are near-zero counts carrying almost no information. `exclude_below_form_floor` drops them, with `include_below_form_floor=True` to reinstate.

## 3. The child-linkage defect (this one moves estimates)

In the by-child export, for the Edgin Down syndrome subset:

```
distinct child_ids       119
  ... on Words & Gestures  53
  ... on Words & Sentences 66
  ... on both forms         0        <-- 53 + 66 = 119
```

Keyed on the study's own subject identifier (`SubID` / `ID`, the same identifier space in both files — 162 of 176/232 ids overlap):

```
distinct subjects         71
  ... on Words & Gestures  51
  ... on Words & Sentences 66
  ... on both forms        46        <-- 51 + 66 - 46 = 71
```

So the pipeline was treating one child assessed on Words & Gestures at 14 months and Words & Sentences at 24 months as two unrelated children. Consequences:

- Child random intercepts in every model carrying them understated within-child correlation and overstated the effective sample size for `us_01`.
- **VG16's within-child cross-lag had no Edgin cross-form links at all.** The Route 1 pre-specification note states that "many children link the two forms within-child" — an expectation the pipeline did not meet.
- The duplicated-outcome and longitudinal-collapse rules compare a child's records against each other, so they were blind to Words & Gestures → Words & Sentences pairs, and the collapse signature could never see a cross-form decline. The linkage fix makes one further such defect visible.

## 4. Empty administrations scored as zeros

Three Words & Gestures and one Words & Sentences source row have _every_ word item blank; there are no partially-blank administrations. Wordbank scores them as zero. Two of the four are Down syndrome rows inside the age window, and they cannot be separated in the export: at 12 months it holds two `(0, 0)` Down syndrome rows, of which only one is the empty form. They are excluded when `data/vocab_data_us_01.csv` is built — the only point at which they are still identifiable — and listed in `data/vocab_data_us_01_manifest.json`.

## 5. Net effect on the pool

Loader to loader, default flags. **Only `us_01` changed**; every other study is identical row for row and value for value.

|                                       | Before                          | After              |
| ------------------------------------- | ------------------------------- | ------------------ |
| `us_01` rows                          | 195                             | 230                |
| `us_01` children                      | 119 nominal (57 with live data) | 58                 |
| `us_01` age range                     | 11–30 months                    | 11–27 months       |
| `us_01` understood present            | 78                              | **126**            |
| `us_01` spoken present                | 165                             | **211**            |
| `us_01` comprehension at 19–27 months | **0**                           | **50**             |
| DS pool rows                          | 1,404                           | 1,439              |
| DS pool children                      | 812                             | 751                |
| Rows at or near a form ceiling        | —                               | 0 (max spoken 406) |

The production rules now mask 19 `us_01` spoken counts where they masked 30. That is not less scrutiny but earlier removal: the ceiling batch leaves as whole children before the count-level rules see it.

**Every Down syndrome fit is stale.** The pool has changed and, more consequentially, the subject index has changed, so the child random effects are not comparable with the previous traces.

## 6. Wordbank has no non-English Down syndrome data

Checked the same day, since it bore on whether to widen the pool at all. The contributor listing (42 languages, 128 dataset entries) names exactly one Down syndrome dataset, Edgin; and all 147 `raw_data/*/*_fields.csv` field-mapping files in `langcog/wordbank` were checked for a mapped condition column, with hits only in the two Edgin files. Site totals are byte-identical across Wayback snapshots from January to August 2026 and the repo has had no commits since 2026-05-15, so the 15 June export is current.

What that does not establish: `health_conditions` is populated only where a contributor supplied and mapped a condition column, so a non-English clinical cohort would be invisible. The negative result rests on the contributor listing, not on the condition field.

## 7. A reversal, and its ordering

The first version of this work held back **all** out-of-window administrations behind a single age-window rule, on the reasoning that 87 of 149 were ceiling-saturated and the near-ceiling rule would catch almost none of them. That was wrong in a way worth recording, because the error is instructive.

**What was wrong.** The rule used age as a proxy for a defect criterion. It therefore also discarded 50 administrations with ordinary counts — and those turned out to be the study's only comprehension observations between 19 and 27 months, from 47 children all already in the pool, so they carry within-child information rather than merely adding rows. Worse, the exclusion was not neutral: a child still on Words & Gestures at 25 months is plausibly lower-ability than one who had moved to Words & Sentences, so dropping the whole block removed observations non-randomly with respect to ability. The rule was more biased than the thing it was guarding against.

**What changed the answer.** Checking, rather than reasoning from the principle. Three facts settled it: `us_01` contributed zero comprehension observations at 19–27 months; all 47 contributing children were already in the pool; and the ceiling-saturated children form a disjoint set with no non-ceiling record, so a provenance criterion separates them cleanly where age cannot.

**Ordering, stated honestly.** The provenance criterion was chosen after the counts had been seen. The criterion itself is not selection on the outcome — it keys on whether a child has any non-ceiling record, not on whether a value is extreme — which is the part that matters for `202607261210-route1-dif-prespecification.md` §5. But the sequence is not ideal and is recorded here as a deviation rather than presented as pre-specification. Two claims made along the way were also wrong and are withdrawn: that the ceiling-saturated children were entirely disjoint from the pool (true only of the Words & Gestures subset), and that four residual ceiling rows would need a study-scoped age-unbounded near-ceiling rule (they do not; the provenance rule reaches them, and such a rule caught zero rows — a check that cannot fire).

**A rule that was proposed and is not needed.** Extending the longitudinal-collapse signature to comprehension. It looked necessary: 4 of 40 children showed comprehension decreasing across the window boundary, one by 387 words. On inspection all four were artefacts of the comparison — 71 apparent "collapses" in the raw source are Words & Gestures comprehension against Words & Sentences comprehension, which is a production proxy the bivariate guard discards. On the guarded frame there are **zero** genuine comprehension collapses, and the one large case (396 → 9) is already masked by the duplicated-outcome rule. The remaining three are 1.8× to 2.1×, well inside what parent report produces and well below the factor-5 threshold that exists to avoid firing on them.

## 8. What was not done

- `IMPLAUSIBLE_PRODUCTION_MAX_AGE_MONTHS` is left at 30, and deliberately so: widening it would mask 19 apparently legitimate records in six other studies. The provenance rule reaches the Edgin batch without touching it.
- No comprehension-collapse rule was added; §7 records why it is not needed.
- The 4 children whose comprehension decreases modestly across the window boundary (1.8× to 2.1×) are retained. They are a sensitivity target, not a defect.
- The Edgin comparison group (321 administrations, `dev_status = 'comparison'`) is carried in `vocab_us_01` and used nowhere. Whether it is a typically-developing comparison sample is a strong inference from its source coding and its vocabulary profile, but no codebook states it.
