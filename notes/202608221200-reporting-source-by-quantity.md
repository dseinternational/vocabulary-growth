# Which model each reported quantity comes from, after the child-slope work

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Reporting decision record, 2026-08-22, written after VG19 completed all five gates and the k-fold that both LOO claims needed. **VG20 remains the model of record for every Down syndrome joint estimand, and is now supported out-of-sample as well as interpretively.** VG19 contributes a finding, not a number. What changes is the **reporting range**: three quantities move materially with the child-effect structure, all above 60 months, and on this evidence the study owner returned the comprehension reporting cap from 84 to 72 months (§4). Reproduced by `scripts/experiments/model_dependence_of_reported_quantities.py`.

## 1. What made this an open question

VG19 beat both VG10 and VG20 on PSIS-LOO by margins that looked decisive — +7.69 and +6.86 standard errors on spoken words. If that had held, the model of record would have been in play, and with it the source of every published Down syndrome trajectory. [202608212000](202608212000-vg19-gates-g2-g4-g5.md) §4c argued the margin was an artefact of the hold-out unit rather than a property of the models, and deferred to k-fold.

## 2. K-fold settles the model of record, and not in the direction LOO suggested

| pair         | PSIS-LOO, spoken |  k-fold LOSO |
| ------------ | ---------------: | -----------: |
| VG19 vs VG10 |         +7.69 SE |     +2.10 SE |
| VG20 vs VG10 |         +1.75 SE | **+3.09 SE** |
| VG19 vs VG20 |         +6.86 SE | **+0.93 SE** |

Full result in [202608212000](202608212000-vg19-gates-g2-g4-g5.md) §4e. Two consequences for reporting.

**VG19 has no out-of-sample case for replacing VG20.** At +0.93 SE the two are indistinguishable at predicting a child nobody has seen, which is the criterion that matches what the study publishes.

**VG20's promotion is now a predictive result and not only an interpretive one.** Its margin over VG10 _strengthens_ from +1.75 to +3.09 SE when the held-out unit becomes a whole child, because correlating two child-level effects helps most when neither has been observed. [202608212000](202608212000-vg19-gates-g2-g4-g5.md) §4c's statement that the promotion "should not be cited as a predictive result" is withdrawn there and should not be quoted from earlier drafts.

## 3. Quantity by quantity

VG10, VG19 and VG20 differ **only** in how a child departs from the population trajectory; every other part of the three graphs is identical. The gap between their fitted curves is therefore a clean measurement of how much a published number depends on a modelling choice the data does not settle — a different question from how wide any one model's interval is. Gaps below are expressed as a fraction of VG20's own 89% ETI width at the same age, because a difference matters to a reader only in proportion to the uncertainty already shown.

### Robust — report from VG20 without qualification

| quantity                       | worst VG19-VG20 gap | worst VG10-VG20 gap |
| ------------------------------ | ------------------: | ------------------: |
| understood, population curve   |    0.14 (5.9 words) |    0.15 (4.8 words) |
| spoken, subject-marginal curve |    0.02 (7.2 words) |   0.03 (17.0 words) |
| `q`, 36 to 48 months           |                0.14 |                0.15 |

The understood trajectory and the spoken subject-marginal trajectory are the two most-reported curves in the study, and neither is hostage to the child-structure choice at any reported age. That is the main practical finding of this note and it is a reassuring one.

### Model-dependent — report from VG20, but say so

| quantity                        | VG20 | VG19 |       gap |
| ------------------------------- | ---: | ---: | --------: |
| `q` at 84 months                | 0.83 | 0.94 |      0.93 |
| `q` at 72 months                | 0.75 | 0.85 |      0.89 |
| spoken, population curve, 84 mo |  436 |  491 |      0.65 |
| spoken, population curve, 72 mo |  346 |  384 |      0.58 |
| `q` at 18 to 24 months          | 0.03 | 0.04 | 0.58-0.69 |

Between-child spread belongs in this class at **every** age rather than only above 60 months, because VG20 holds it constant and VG19 grows it, and [202608220748](202608220748-vg19-individual-trajectories.md) §4 shows the two cross at about 48 months — so neither is the conservative choice across the range.

The young-age `q` rows are listed for completeness and are the least important: 0.03 against 0.04 is a large fraction of a very narrow interval and a negligible difference in substance.

Note that **VG10 is not the model that disagrees.** Its gaps from VG20 never exceed 0.15 anywhere in the table. The divergence is specifically the child _slope_, not the child _correlation_, which is the same asymmetry §4e found in the k-fold standard errors.

## 4. The divergence sits where the data run out, and the cap was just extended into it

| band         | rows | children | understood | spoken |
| ------------ | ---: | -------: | ---------: | -----: |
| 8-24 mo      |  361 |      215 |        273 |    342 |
| 24-36 mo     |  350 |      254 |        297 |    341 |
| 36-48 mo     |  299 |      227 |        213 |    286 |
| 48-60 mo     |  214 |      164 |        100 |    189 |
| 60-72 mo     |  165 |      132 |         66 |    143 |
| **72-84 mo** |   73 |       62 |     **25** |     68 |
| 84-120 mo    |   59 |       50 |         13 |     59 |

`q` is a ratio of spoken to understood, so it is the **understood** column that limits it, and the 72-84 month band holds **25 comprehension observations across 62 children**. That is why the models diverge there and agree everywhere else: below 60 months the data pin the trajectory and the child structure cannot move it; above 72 months almost nothing pins it and the child structure supplies the answer.

This band is where `df91f80` raised the comprehension reporting cap from 72 to 84 months on 2026-08-13. **The study owner's decision, 2026-08-22: return the cap to 72, and revisit if data on older children arrives.** Implemented in the same change as this note.

### Why 72, when the 2026-08-13 raise was correct on its own terms

The raise asked whether the 72-84 band is **observed rather than extrapolated**, and answered it correctly — 25 understood rows from 20 children across five studies (`ie_01`, `uk_01`, `uk_06`, `uk_07`, `us_02`) is a real, if thin, band of data, and its reasoning explicitly matched the threshold the study owner had applied to `report_max_age_signed` in [#212](https://github.com/dseinternational/vocabulary-growth/issues/212). Nothing in that argument was wrong and nothing here contradicts it.

The cap moves because a **second and stricter test** applies, one the raise did not ask: is the number in that band determined by the **data**, or by the **model**? It is not determined by the data. VG19 and VG20 differ only in the child-effect structure, are indistinguishable out of sample at +0.93 SE over 767 children, and still disagree about `q` by 0.89 and 0.93 of VG20's own 89% interval width at 72 and 84 months. Twenty-five comprehension observations cannot separate the two structures. Below 60 months the same comparison never exceeds 0.15.

So the two tests give different answers, and the difference is exactly the point: **the band is observed but not determined.** A reader shown "at 6 years a child with Down syndrome typically says about 75% of what they understand" would reasonably take that as a measurement. It is a measurement to within a wide interval _and_ a modelling choice worth most of that interval again, and only the first of those reaches them. Stopping at 72 is the only one of the three options that does not require the reader to hold that distinction.

The two alternatives are recorded because they were real and may be reopened: keeping 84 with an explicit model-dependence note on the 60-84 band, which is honest but asks a general reader to carry two numbers for one quantity; and keeping 84 unqualified on the grounds that VG20's interval already covers VG19's estimate, which is defensible arithmetic and was the behaviour in place, but presents as settled a number the model comparison does not settle.

### What it costs, and what it does not

**`understood` is trimmed although it did not need to be.** The three models agree on the understood curve to within 0.15 interval widths at every age to 84, so understood alone would still support 84. It comes down with `q` because both ride the single `report_max_age_understood` field and `q` is conditioned on understood (`reporting_ages.ReportedQuantity.RATIO_OF_UNDERSTOOD`). Giving `q` its own field would invalidate all seventeen model definitions to express a cap this field already expresses correctly, if conservatively. The policy is per quantity; the mechanism is not, and this is the one place that costs something.

**`spoken` is untouched** — it keeps the full grid to 90, and the three models agree on its subject-marginal curve to within 0.03 interval widths at every age. **`signed` is untouched** at 84, on its own field since [#212](https://github.com/dseinternational/vocabulary-growth/issues/212) and deliberately independent since `4ff48e5`. The signed _share_ `r` and `p_any` do come down to 72, because both are conditioned on understood — and `p_any` is the case `reporting_ages` anticipated in writing, where the comprehension and signing caps diverge and the tighter binds.

**Eleven fits go stale.** `report_max_age_understood` is a model-definition field and `validate_fit_output` compares the definition whole, so VG02, VG05, VG07, VG08, VG09, VG10, VG14, VG15, VG16, VG19 and VG20 all need refitting at `rep` before anything can be published. That is the deliberate design recorded in `reporting_ages`: "change it and every affected model of record is correctly marked stale." It cannot move a posterior — every refit will reproduce its own trace up to Monte Carlo error — but the gate does not and should not know that.

### When to raise it again

Not when the band is merely populated. **When it can distinguish the child structures** — which is answered by rerunning `scripts/experiments/model_dependence_of_reported_quantities.py` on the enlarged pool and checking whether the VG19-VG20 gap at 72 and 84 months has fallen to the 0.15 that holds below 60, not by recounting rows. That is a different and harder test than the one the 2026-08-13 raise passed, and stating it here is the point of recording the decision at all.

This should be attached to [#228](https://github.com/dseinternational/vocabulary-growth/issues/228), whose subject is scope rules justified by properties their outcome does not have. This is the same failure mode one level up: a scope rule justified by the presence of data, where the property that matters is the data's ability to identify the quantity.

## 5. What VG19 contributes, given that it contributes no numbers

The dissociation. Comprehension trajectories run close to parallel and production-ratio trajectories genuinely diverge — [202608220748](202608220748-vg19-individual-trajectories.md) §2 and §3. That passed all five gates, its direction survives the recovery bias in `tau_subj_q_1`, and its comprehension half is corroborated by a tracking ICC measured from raw residuals with no model at all. It belongs in the report as a structural result about **how children differ from one another**, stated qualitatively, and explicitly not as a source of published intervals.

The negative result belongs there too: the child slope does **not** explain `kappa_u`'s decline ([202608212000](202608212000-vg19-gates-g2-g4-g5.md) §4), which removes the leading candidate and is worth a sentence in any methods discussion of the dispersion model.

## 6. Unchanged

VG14 and VG15 remain the source for signing and total expressive vocabulary; VG03, VG04, VG11, VG12 and VG13 for the typically-developing reference; VG16 for the cross-lag. Nothing in the child-slope work touches them, and none of them was refitted for it.

## 7. Caveats

1. **The k-fold fits are `test` tier** (4 chains x 2000 draws), the tier `kfold_loso.py` has always used. The differences are large relative to the resulting MCMC noise, but +0.93 SE should be read as "no detectable difference", not as a precise zero.
2. **The gap fractions compare posterior medians**, so they say nothing about whether the two models' intervals overlap -- they do, everywhere. The claim is about the point estimate a reader is shown, which is the number that gets quoted.
3. **VG19 is not a candidate for promotion and this note does not treat it as one.** The comparison is used as an instrument for measuring structural sensitivity, which is a legitimate use of a well-gated model that is not a model of record.
4. **The successor that would carry both mechanisms does not exist**, and by [202608221000](202608221000-four-by-four-gate1.md) it should be a low-rank factor form rather than the free 4x4 that was originally proposed. Until it is registered and fitted, the model-dependence recorded here cannot be resolved -- only disclosed.
