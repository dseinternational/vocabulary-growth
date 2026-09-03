# Why matching on mental age and matching on comprehension give opposite answers

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-03. **Script:** `scripts/compare_matched_designs.py`. **Output:** `output/comparisons/matched_design_*.csv`. **Source:** `data/vocab_data_es_01.csv` (Galeote et al. 2011), the only source in this repository with a typically developing comparison group and a recorded mental age.

Prompted by an objection worth taking seriously: studies matching children with Down syndrome to typically developing children on **mental age** often report similar or better vocabulary production for the Down syndrome group, and this repository reports the opposite — a substantially lower share of understood words spoken — when matching on **comprehension**. If those two findings conflicted, one of them would be wrong.

They do not conflict. Both are reproduced here in the same 186 children, at the same time, on one instrument, and the arithmetic that reconciles them is exact.

## Headline

**At matched mental age children with Down syndrome understand about a fifth more words and say about a quarter less of what they understand. The two effects are opposite in sign and comparable in size, so they cancel in the level of production, which comes out indistinguishable from parity.** Which finding a study reports is therefore decided by its matching variable, and the two designs are answering different questions rather than disagreeing about one.

The matched-mental-age design asks: _how many words does this child say, given their cognitive level?_ The comprehension-matched design asks: _given that the child knows the word, do they say it?_ Comprehension vocabulary lies on the path between cognitive level and production **and is elevated in Down syndrome relative to mental age**, so conditioning on it removes the group's advantage by construction and isolates the step where the deficit lives.

## The source, and why it settles this

`es_01` supplied 186 children with Down syndrome, each matched pairwise on mental age and sex to a typically developing child, all assessed on the same 651-item CDI-Down. The matching is tight: the Brunet-Lézine total developmental age is within 0.30 months on every one of the 186 pairs and the CDI-Down matching band is identical on all 186. Chronological age differs sharply, as the design requires — Down syndrome median 32 months (range 11–71) against 19 months (6–33).

Two features make this the right source and no other in the repository can substitute. It carries the matching variable itself, so the matched-mental-age design does not have to be approximated. And both groups sit on **one instrument**, so the comparison is free of the form-harmonisation assumption every cross-study contrast in this project carries.

One point of independence. `es_01`'s Down syndrome children are all in the Down syndrome analysis pool, so nothing here is independent evidence _of_ the conversion shortfall. Its typically developing children are **not** in the Wordbank-scoped reference pool — they are a Spanish-normed sample on a different instrument, deliberately excluded from it — so the matched comparison itself is new.

## At matched mental age

From `matched_design_mental_age.csv`, Wilcoxon signed-rank over the 186 pairs:

| Measure                     | DS median | TD median | DS/TD | Paired median difference | Pairs DS ≥ TD | _p_      |
| --------------------------- | --------- | --------- | ----- | ------------------------ | ------------- | -------- |
| Words understood            | 266.5     | 210.0     | 1.27  | +18.5                    | 116 of 186    | 0.0099   |
| Words spoken                | 26.0      | 31.0      | 0.84  | -1.0                     | 92 of 186     | **0.31** |
| Words gestured              | 30.5      | 17.0      | 1.79  | +11.0                    | 138 of 186    | 3.3e-12  |
| Produced, spoken ∪ gestured | 73.0      | 60.5      | 1.21  | +4.0                     | 112 of 186    | 0.066    |
| Ratio spoken / understood   | 0.143     | 0.186     | 0.77  | -0.02                    | 86 of 186     | 0.0064   |

**Spoken production is at parity** — a paired median difference of one word against it, 92 of 186 pairs at or above the match, and no detectable difference. On total lexical production including symbolic gesture the Down syndrome children are **ahead**. That is the literature finding, in this repository's own data.

And in the same children at the same moment, comprehension is higher and the conversion ratio is lower. Nothing about the first result is in tension with the second.

## The decomposition, which is exact

Words spoken is words understood times the share spoken, so on logs the group difference splits additively with no residual. Differencing within pairs (`matched_design_decomposition.csv`, 155 pairs where both children have a non-zero count on both measures):

| Term                                      | Paired mean of the log difference | Multiplier |
| ----------------------------------------- | --------------------------------- | ---------- |
| Comprehension advantage, log(U_DS / U_TD) | +0.177 (se 0.058, _t_ = +3.03)    | **×1.19**  |
| Conversion shortfall, log(r_DS / r_TD)    | -0.319 (se 0.095, _t_ = -3.37)    | **×0.73**  |
| Net production, log(S_DS / S_TD)          | -0.142 (se 0.103, _t_ = -1.37)    | ×0.87      |

The identity holds to 1.3e-15, so this is a decomposition rather than a model: **+0.177 and -0.319 sum to -0.142 exactly.** A comprehension advantage of 19% and a conversion shortfall of 27% leave production 13% low, which at this sample size is not distinguishable from parity.

Both component terms are individually well determined and the net is not. That is the whole phenomenon: two real effects of opposite sign, each about three standard errors from zero, cancelling into a null.

## The three matching choices, ranked

The same 372 children, grouped three ways (`matched_design_bands.csv`). Ratio of median conversion ratios, Down syndrome over typically developing:

| Matched on            | Ratio of ratios, by band                                                                                    | What the design does to the comparison                                                                          |
| --------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Mental age**        | spoken production at parity (paired log mean ×0.87, _p_ = 0.31); ×1.21 on median produced including gesture | Lets the comprehension advantage operate, so it carries production up with it                                   |
| **Comprehension**     | 0.56, 0.50, 0.97, 0.33, 0.89 across bands                                                                   | Removes the comprehension advantage by construction, leaving only the conversion step                           |
| **Chronological age** | 0.21, 0.33, 0.15 across the 11–33 month overlap                                                             | Removes nothing and adds the delay: at 24–33 months the DS children understand a median 238.5 words against 489 |

Chronological-age matching is the harshest because it stacks both effects — the Down syndrome children are behind on comprehension _and_ convert less of it. Mental-age matching is the most favourable because it is matching on the variable that carries the group's relative strength. Comprehension-matching sits between them, and is the only one of the three that isolates a single step.

Two bands must not be read as evidence in either direction. In the lowest mental-age band and the lowest comprehension band the typically developing median spoken count is **zero**, so the ratio of ratios is undefined rather than large; the script writes `NaN` there rather than a number. The highest bands are at the instrument's ceiling, where both groups converge (0.73 against 0.80 at mental-age band 7).

## Separating the three variables in one model

Words spoken out of words understood, binomial with HC0 standard errors (`matched_design_regression.csv`, n = 372):

| Specification                                          | DS effect (logit) | HC0 se | _z_   | Odds multiplier |
| ------------------------------------------------------ | ----------------- | ------ | ----- | --------------- |
| Group only                                             | -0.349            | 0.170  | -2.05 | 0.71            |
| Group + comprehension                                  | -0.471            | 0.154  | -3.06 | 0.62            |
| Group + mental age                                     | -0.338            | 0.128  | -2.65 | 0.71            |
| Group + comprehension + mental age                     | -0.359            | 0.126  | -2.84 | 0.70            |
| Group + comprehension + mental age + chronological age | -0.917            | 0.239  | -3.84 | 0.40            |

Three readings. **Conditioning on comprehension makes the group effect worse** (0.71 → 0.62), which is the mechanism above in one line: comprehension is the group's strength, and holding it fixed takes the advantage away. **Conditioning on mental age instead leaves it unchanged** (0.71 → 0.71), so mental age is not doing the work comprehension does. And **adding chronological age more than doubles it** (0.70 → 0.40), which is the finding already recorded in `notes/202609021800-production-ratio-by-understood.md`: at equal vocabulary knowledge, elapsed months buy production, and the Down syndrome children have had many more of them.

**The standard errors need saying plainly.** A binomial likelihood on 651 words treats items within a child as independent. The count-scale Pearson dispersion here is 46 to 106, so the model-based standard errors are eight to thirteen times too small and would report _z_ between -22 and -33. Those are meaningless; the HC0 sandwich estimates above are the honest ones, and they still place every group effect at two to four standard errors from zero. This is the same overdispersion the fitted models in this repository carry a Beta-Binomial likelihood for.

## What this settles that the earlier note said it could not

`notes/202609021800-production-ratio-by-understood.md` closed with a residual worry it could not test: "comprehension may not fully index developmental level, and if the Down syndrome children at matched comprehension are behind on a broader cognitive dimension, part of the age-matched gap is that rather than speech. These data cannot settle it."

`es_01` carries mental age, so it can be tested here, and the worry does not survive:

| Comprehension band | DS mental age | TD mental age |
| ------------------ | ------------- | ------------- |
| 50–100             | 12.3          | 13.9          |
| 100–200            | 15.3          | 17.9          |
| 200–300            | 20.8          | 19.2          |
| 300–450            | 21.8          | 23.2          |
| 450–651            | 26.4          | 25.8          |

At matched comprehension the Down syndrome children's mental age runs **at or slightly below** the typically developing children's, by 1.4 to 2.6 months in three of five bands. So comprehension-matching does not hand them a cognitive advantage that would flatter the comparison; if anything it slightly disadvantages them, which would make the observed conversion shortfall marginally conservative rather than inflated. The regression agrees from the other side: adding mental age alongside comprehension moves the group effect from 0.62 to 0.70, a small move in the direction of a smaller deficit, not the collapse the worry would predict.

That worry can now be recorded as tested on one sample rather than untestable.

## What this means for the report

The comparison chapter currently reports the conversion shortfall without saying which matching choice produces it, and the shortfall is the number a reader will carry away. Three things follow.

1. **Name the estimand wherever the shortfall is reported.** "Children with Down syndrome say a smaller share of the words they understand" is a statement about conversion at matched comprehension. It is not, and should not be allowed to read as, a statement that they speak less than cognitively comparable peers.
2. **Report the comprehension advantage beside it.** It is the other half of the same decomposition and it is what makes the production result come out at parity. Reporting one without the other is what generates the apparent conflict with the literature.
3. **The intervention reading is the one that changes.** A conversion shortfall at matched comprehension, with production at parity at matched mental age, points at the step from understanding a word to saying it — which is where speech and oral-motor support act — rather than at a general vocabulary deficit. The gesture result points the same way: at matched mental age these children have a _larger_ expressive lexicon than their matches once the non-vocal modality is counted, which is an argument about modality rather than about lexical knowledge.

## Caveats

- **One sample, one language, one instrument, cross-sectional.** 186 pairs, Spanish, CDI-Down, one observation per child. Everything above should be treated as a demonstration that the two designs diverge and by roughly how much, not as a population estimate of either term.
- **The matching variable is not a nonverbal mental age.** It is the Brunet-Lézine Psychomotor Development Scale-Revised **total** developmental age, which includes a language component. Matching on it therefore partly equates language, which biases _against_ finding a Down syndrome comprehension advantage. The ×1.19 is on that account conservative, and a nonverbal-mental-age match would be expected to show a larger comprehension advantage and a firmer production parity. This cuts in a known direction, which is why it is worth stating rather than merely listing.
- **The gesture column has an open question with the author.** Galeote et al. report eliminating 11 of the instrument's 21 categories from the gesture data; whether this file carries that treatment is unconfirmed (see `data/vocab_data_es_01.md` and `notes/202608271551-es01-gesture-construct.md`). If it does, `gestured` is measured over roughly half the item universe of `understood` and `spoken`, and the gesture advantage above is understated. The spoken and comprehension columns are unaffected — two children reach the full 651 on comprehension.
- **The gesture construct is broader than taught signing.** es_01's non-vocal modality is a symbolic-gesture repertoire, "properly taught or spontaneously learnt", not the taught key-word signing `uk_02`, `uk_07` and `nz_01` record. The union measure is not comparable with the signing sources' `produced`.
- **Two Down syndrome children are at the comprehension ceiling** (651 of 651, aged 59 and 60 months), so their receptive vocabulary is censored and the comprehension advantage is very slightly understated.
- **The decomposition drops 31 of 186 pairs** where either child has a zero count on either measure, and those pairs are concentrated at the youngest mental ages. The matched-mental-age table above uses all 186 and agrees in direction, so the restriction is not what produces the result.
- **The percentile asymmetry recorded in the earlier note still applies** to the comprehension-matched comparison drawn from the pooled sources; it does not apply to the paired analysis here, which is matched by design rather than by band.
