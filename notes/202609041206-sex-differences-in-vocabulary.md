# Sex differences in vocabulary: what the literature and this repository's data say, and whether sex belongs in the models

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

**Date:** 2026-09-04. **Scripts:** `scripts/experiments/sex_effect_by_study.py` (descriptive estimates and forest plot) and `scripts/experiments/sex_shift_predictive.py` (what a sex shift does to VG20's predictive). **Output:** `output/comparisons/sex-effect/`. **Sources:** the merged Down syndrome analysis view, the Wordbank English (American) typically developing export, the `es_01` matched pairs, and VG20's stored posterior summaries.

Prompted by the question of whether sex should be a predictor in the models. The study owner confirmed ie_02's sex coding (1 = male, 2 = female) during this work and it is now decoded by the loader (#292); the estimates below were re-run after the source CSVs were tidied to a consistent sex coding and the database regenerated, and did not move.

## Headline

**Girls are ahead of boys on CDI vocabulary counts at the same age, in Down syndrome as in typical development, by about 0.2 logits on words understood and 0.35 on words spoken, with the two populations agreeing on both sizes.** That is a real and cross-linguistically replicated effect, and a small one against the between-child spread: it explains about 1% of the between-child variance the models carry in their child scales. For a new child of known age and sex it moves the centre of VG20's predictive by 4 to 8% of the 89% interval's width and leaves the width essentially unchanged.

**Recommendation: not in the model of record; one exploratory variant of VG20 on the sex-known subset, reported in the discussion as heterogeneity.** The reasons are in the last two sections. _Update, same day:_ the variant has been fitted under [#295](https://github.com/dseinternational/vocabulary-growth/issues/295); see `notes/202609041530-vg20-sex-shift-arm.md` for what it returned.

## What the literature says

**Typically developing children.** The evidence is unambiguous. Eriksson and colleagues' 2012 synthesis of 13,783 children in ten language communities found girls slightly ahead of boys on communicative gestures, productive vocabulary and word combinations, with the gap growing across the CDI age window and boys no more variable than girls ([British Journal of Developmental Psychology](https://bpspsychub.onlinelibrary.wiley.com/doi/abs/10.1111/j.2044-835X.2011.02042.x)). The Wordbank book's demographics chapter reports a female advantage on Words and Gestures comprehension in 16 of 22 languages with a small median effect, and on Words and Sentences production in 25 of 26 languages with a median effect about five times larger ([Frank, Braginsky, Yurovsky and Marchman](https://langcog.github.io/wordbank-book/demographics.html)). This is why the CDI's own norms are published separately for girls and boys.

**Children with Down syndrome.** The direction agrees but the evidence base is thin and mostly small samples.

- Berglund, Eriksson and Johansson's 2001 Swedish CDI study of 330 children aged 1 to 5 examined sex differences, and is cited by later authors for girls having richer vocabulary and syntax and longer utterances from a very young age ([Journal of Speech, Language, and Hearing Research](https://pubs.asha.org/doi/10.1044/1092-4388%282001/016%29)).
- A 2018 Frontiers in Genetics comparison of youth with Down syndrome and fragile X syndrome found large female advantages on expressive vocabulary, syntax and lexical diversity, with no sex difference on receptive vocabulary or nonverbal cognition ([Frontiers](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2018.00424/full)).
- Udhnani and Lee's 2025 study of 37 youth with Down syndrome found a female advantage in parent-reported structural and pragmatic language that was absent in the autistic and typically developing groups ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39874988/)); Lee and colleagues' 2017 pragmatics study found sex-specific patterns on direct assessment ([PubMed](https://pubmed.ncbi.nlm.nih.gov/28654411/)).
- Against those, Næss and colleagues' 2021 predictor study, already in the report's bibliography, entered sex as a covariate and found no association with expressive vocabulary in either group ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7998706/)), and the 2020 lifespan study of health and cognition in Down syndrome found males below females on receptive language across ages with a partial eta squared of 0.01, not significant within any age band ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6979347/)).

The consistent pattern is that any sex difference in Down syndrome sits on production rather than comprehension, which is the same shape as the typically developing effect.

## Coverage in this repository

Sex is recorded for 624 of the 841 children in the merged Down syndrome view, from eight studies, and for 97% of Wordbank's typically developing administrations. It is absent from ie_01, it_01, nz_01, uk_03, uk_04 and us_02. The pool with known sex is 54% male against 50% in Wordbank, which at an effect of 0.33 logits shifts the pooled DS-versus-TD contrast by about 0.015 logits, so composition is not a source of bias.

| Study | Male rows | Female rows | Children |
| ----- | --------: | ----------: | -------: |
| es_01 |        98 |          88 |      186 |
| ie_02 |        51 |          60 |       65 |
| uk_01 |       116 |          99 |      129 |
| uk_02 |        73 |          63 |       65 |
| uk_05 |        27 |          21 |       16 |
| uk_06 |         6 |           5 |       11 |
| uk_07 |        51 |          31 |       30 |
| us_01 |       203 |         142 |      122 |

One inconsistency surfaced and predates the tidy: raw us_01 subject 151210118 carries a female Words and Gestures row and a male Words and Sentences row on the same day at 43 months, and comparison-group subject 151120229 shows the same pattern at 31 months. Both are in the ceiling-saturated preparation batch, every count at its form ceiling and outside the norming window, which the loader already drops on provenance grounds; neither reaches the VG20 analysis frame. The inconsistent sex is one more sign that batch is not real data.

## Descriptive estimates

`sex_effect_by_study.py` fits an empirical-logit OLS of each count against a cubic in age and a female indicator, with study dummies for the pooled rows and HC0 robust standard errors, on the merged view before the loader's masking rules, restricted to the models' age domain of 8 to 115 months. Intervals are 89%. This is descriptive, not a model fit, and the rows include administrations the models mask.

| Row                                 |           Understood |               Spoken |     Production ratio |
| ----------------------------------- | -------------------: | -------------------: | -------------------: |
| es_01                               | +0.17 [−0.12, +0.46] | +0.36 [+0.03, +0.69] | +0.26 [−0.05, +0.56] |
| uk_01                               | +0.17 [−0.20, +0.54] | +0.61 [+0.31, +0.91] | +0.80 [−0.09, +1.70] |
| uk_02                               | +0.24 [−0.17, +0.66] | +0.45 [−0.06, +0.96] | +0.26 [−0.25, +0.76] |
| uk_05                               | +0.06 [−0.40, +0.52] | −0.07 [−0.79, +0.65] | −0.45 [−1.35, +0.45] |
| uk_07                               | +0.40 [+0.04, +0.75] | +0.41 [−0.10, +0.92] | +0.09 [−0.49, +0.67] |
| us_01                               | +0.05 [−0.41, +0.50] | +0.13 [−0.42, +0.68] | +0.48 [−0.06, +1.03] |
| ie_02                               | +1.09 [+0.62, +1.56] | +0.51 [+0.08, +0.95] | −0.18 [−0.66, +0.30] |
| **DS pooled, study-adjusted**       | +0.33 [+0.15, +0.51] | +0.35 [+0.12, +0.58] | +0.23 [+0.01, +0.45] |
| **TD Wordbank, English (American)** | +0.20 [+0.13, +0.28] | +0.33 [+0.29, +0.37] | +0.14 [+0.08, +0.20] |
| **TD es_01 matches, same CDI-Down** | +0.27 [+0.02, +0.53] | +0.48 [+0.20, +0.77] | +0.35 [+0.07, +0.63] |

Four things stand out. **Words spoken is the consistent panel**: six of seven Down syndrome studies sit to the right of zero, and the two typically developing references land inside the pooled band. **Words understood is driven by one study**: without ie_02 the Down syndrome estimates cluster near +0.2, matching Wordbank; ie_02's +1.09 sits outside every other interval in the panel and is not a recruitment artefact (sex is balanced across its two groups and both time points, and the effect holds within each group), but it is 65 children and the widest single-study estimate in the pool, so it widens the range rather than settling the size. **The production ratio is the weakest and widest**: girls' spoken advantage is mostly the comprehension advantage carried through. **The effect survives mental-age matching**: in es_01 the female effect on spoken words is +0.30 (se 0.18) with mental age in place of chronological age, so it is not a mental-age composition artefact. Whether the effect changes with age is taken up in its own section below: on the logit scale no age-by-sex interaction is detectable in either population.

In words, the spoken shift is about 30 words at 36 to 48 months and about 60 at 48 to 60 months. Against the models' own child scales it explains about 1% of between-child variance: the shift is a quarter of a child standard deviation on comprehension and a tenth on the production ratio.

## What a sex predictor would do to predictions

`sex_shift_predictive.py` rebuilds VG20's new-child predictive from the fit's stored posterior medians, with the paired spoken-given-understood structure, and reproduces the model's own stored predictive tables to within a word at every canonical age and every threshold probability. It then applies the sex effect symmetrically, girls up by half the difference and boys down by half. The base case uses the shifts most of the studies agree on, 0.20 on understood and 0.15 on the production ratio, about 0.35 on spoken words. Words spoken, median and 89% interval:

| Age | Pooled        | Girls         | Boys          | Median gap | Gap as share of 89% width |
| --: | ------------- | ------------- | ------------- | ---------: | ------------------------: |
|  36 | 32 [0, 223]   | 36 [0, 242]   | 27 [0, 204]   |         +9 |                      0.04 |
|  48 | 127 [10, 429] | 140 [12, 451] | 114 [8, 407]  |        +26 |                      0.06 |
|  60 | 243 [35, 563] | 263 [40, 582] | 225 [30, 545] |        +38 |                      0.07 |
|  72 | 336 [69, 636] | 357 [79, 652] | 316 [61, 622] |        +41 |                      0.07 |

Words understood behaves the same way: a gap of 30 to 40 words from 36 months on, about 8% of the interval width. In both outcomes a boy at the boys' median sits at roughly the 42nd percentile of girls. Under the larger shifts the pooled data give once ie_02 is included, 0.33 and 0.22, the gaps grow by about half and the boys' median moves to the 35th to 38th percentile.

The threshold questions a practitioner actually asks move by a few points:

| Question                                 | Pooled | Girls | Boys |
| ---------------------------------------- | -----: | ----: | ---: |
| P(10 or fewer spoken words at 36 months) |   0.28 |  0.25 | 0.31 |
| P(50 or more at 36 months)               |   0.39 |  0.42 | 0.35 |
| P(50 or fewer at 48 months)              |   0.24 |  0.21 | 0.27 |
| P(100 or more at 48 months)              |   0.58 |  0.62 | 0.54 |
| P(100 or fewer at 60 months)             |   0.19 |  0.17 | 0.22 |
| P(300 or more at 60 months)              |   0.39 |  0.43 | 0.35 |

So in practice: expectations quoted to a family would read about 114 words for a 4-year-old boy and 140 for a girl rather than 127 for both, a visible difference on a page and the reason the typically developing CDI norms are split by sex, but the range quoted alongside it, 10 to 429 words, is untouched. A fixed effect explaining about 1% of between-child variance narrows predictive intervals by less than 1%. A cut-off such as fewer than 50 words at 48 months catches 27% of boys and 21% of girls under the pooled curve, so sex-specific curves would reclassify the few children within about 15 words of the cut-off and no one else. And the child scale on the production ratio is 1.35 logits against a sex shift of 0.15: one earlier administration from the same child shifts the child-specific predictive several times further than sex does, which is where the real gains in individual prediction sit.

## Should the effect be a function of age?

No. Enter sex as a constant shift on the logit scale and let age do the rest. The question has two readings, and the answer is different for each.

**On the count scale, and in months, the effect is already a function of age.** A fixed shift on a rising logistic curve opens up in words and in months as the curve climbs. `sex_shift_predictive.py` reads the base-case shift along VG20's population curve (`sex_shift_words_months.csv`):

| Age (months)                    |  24 |  30 |  36 |  42 |  48 |  54 |  60 |  66 |  72 |
| ------------------------------- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Spoken, gap in words (0.35)     |   2 |   5 |  12 |  25 |  41 |  54 |  63 |  67 |  70 |
| Spoken, gap in months           | 2.0 | 2.0 | 2.3 | 2.8 | 3.6 | 4.9 | 6.6 | 8.4 | 9.2 |
| Understood, gap in words (0.20) |  19 |  27 |  32 |  35 |  38 |  40 |  40 |  40 |  39 |
| Understood, gap in months       | 2.2 | 2.9 | 4.4 | 4.6 | 4.3 | 4.8 | 6.2 | 7.2 | 7.2 |

Eriksson's finding that the girl-boy gap grows across the CDI window is largely this: a constant logit difference read on the count scale. The forest plot in this note is on the logit scale precisely because that is where the effect is closest to a single number. The months figure is the shift divided by the population's logit gain per year, so it grows as the curve flattens rather than because girls pull further ahead; at 72 months a fixed 0.35 logits reads as nine months because the population gains only about half a logit a year by then. The report should quote the gap in words at fixed ages, which is what a parent or practitioner would compare against.

**On the logit scale there is no detectable interaction.** `sex_effect_by_study.py` adds an age-by-female term to the study-adjusted regressions (`sex_effect_age_interaction.csv`):

| Data                         | Female at centre age | Age × female per year | Implied change, 24 to 72 months |
| ---------------------------- | -------------------: | --------------------: | ------------------------------: |
| DS understood                |         +0.33 (0.12) |         −0.003 (0.08) |                    −0.01 ± 0.32 |
| DS spoken                    |         +0.35 (0.15) |         +0.027 (0.07) |                    +0.11 ± 0.29 |
| DS production ratio          |         +0.24 (0.14) |         +0.045 (0.10) |                    +0.18 ± 0.38 |
| TD Wordbank WG comprehension |         +0.20 (0.05) |         +0.209 (0.21) |                                 |
| TD Wordbank WS production    |         +0.39 (0.04) |         +0.021 (0.09) |                                 |

The Down syndrome data rule out a change over four years much larger than 0.3 to 0.4 logits, which is the size of the whole effect; anything age-varying would have to be as big as the effect itself, and the point estimates sit at zero. Within each Wordbank form the slope is small and not distinguishable from zero, and the Wordbank book reports the same across languages: roughly constant with age, if anything a slight decline near the form ceiling. Estimates by age band in the Down syndrome pool jump around, +1.0 at 30 to 42 months and −0.04 at 42 to 54 on spoken words, but with standard errors of 0.2 to 0.3 and a study mix that changes band to band that is noise, not a trend.

**So the variant should carry one `beta_sex` on the understood predictor and one on the production-ratio predictor, constant in age, with VG20 nested exactly at zero.** That keeps the one-parameter discipline the repository used for `rho_uq` and `tau1`, and it is what the data can support. A sex-specific trajectory, meaning separate GP departures by sex, would halve the data behind each curve, need a third curve or marginalisation for the quarter of children with no recorded sex, and would be estimating a shape the interaction test says is flat. The assumption should be checked after the fit rather than built in: a posterior predictive check of residual means by sex within age band is the right test, and if it fails the next step is a linear age-by-sex term, not a sex-specific curve.

## Why not the model of record

- **It does not change what the report reports.** The estimands are population trajectories and the production ratio by age. A sex fixed effect leaves those unmoved and narrows child-level predictive intervals by less than 1%.
- **A quarter of the children have no sex.** A fixed effect means either dropping six studies, including the Italian and Irish sources that supply the older comprehension tail, or marginalising a missing indicator inside the likelihood. Both are real costs for a covariate that explains about 1% of the spread.
- **The sex mix does not bias the comparison.** 54% male against 50% moves the population contrast by 0.015 logits.
- **Refit cost.** A new definition field invalidates every fit of that dataclass unless it goes on a sibling subclass or through the backfill registry, so this has to be a separate registered model, not a flag on VG20.
- **Practice implications cut both ways.** Sex-split expectations for children with Down syndrome would rest on one large study and a few small ones. The typically developing norms carry that split because their base is tens of thousands of children.

## Why it is still worth one fit

The production-ratio result is the interesting one. If the female advantage in Down syndrome sits on spoken words given comprehension, that is a statement about the study's own estimand and matches the expressive-not-receptive pattern in the literature. The recommended path:

1. **Recover sex** where it is cheap. ie_02 is done. The uk_03 and uk_04 source files and the nz_01 and it_01 contributors are worth one email each; uk_07's UK Data Service deposit also holds Mullen age equivalents that were never extracted (`202608121030`).
2. **Add a by-sex panel** to the descriptive report, which needs no model and puts the table above on record.
3. **Register one exploratory variant** derived from VG20 on a sibling subclass, with a Normal(0, 0.5) sex shift on the understood and production-ratio predictors, constant in age for the reasons above, fitted at the `test` tier on the sex-known subset and compared against VG20 restricted to the same rows, with a posterior predictive check of residuals by sex within age band. Report it in the discussion as heterogeneity, alongside the existing caveat in `docs/report/_caveats-ds.qmd` that sex, health and cognitive level are not modelled. That caveat should stay until step 3 has a fit behind it.
4. **Revisit the model of record only if** the variant moves the child-level scales or the production ratio by more than their interval widths, which the descriptive check suggests it will not.

## Caveats

The estimates are a descriptive regression on parent-report counts, not fitted quantities; a proper variant fit could return values somewhat inside or outside the two cases used here. The predictive reconstruction drops parameter uncertainty by using posterior medians, which is small next to the child spread and count noise that dominate these intervals. The shifts are applied symmetrically about the pooled curve, which is right for a pool that is close to sex-balanced. Sex effects on parent-report instruments may include reporter expectation as well as real difference; the cross-linguistic consistency Frank and colleagues describe argues for the latter, but the same reporter confound applies in the Down syndrome sources. The descriptive regressions are restricted to the models' age domain, 8 to 115 months, which is the range VG20's own frame spans; a single us_01 administration at 173 months that the loader drops moved us_01's own estimates by up to 0.2 logits when it was left in, which is a warning about a cubic in age on a thin tail, and capping at 90 months instead moves the pooled spoken estimate from +0.35 to +0.32 and nothing else.
