# VG16 quantified: what the cross-lag says about receptive vocabulary predicting expressive vocabulary

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Analysis of the fitted `rep` model of record (commit `d041e7f`, fitted 2026-08-14, clean provenance, `compact` trace). Every figure below is reproduced by `scripts/experiments/vg16_crosslag_quantification.py`. Companion note: [202608151140](202608151140-cross-lag-not-for-models-of-record.md) answers whether the cross-lag should enter the other models of record (it should not).

## 1. The question and the fit

VG16 exists to report one estimand: does a child's earlier receptive standing predict how much of what they understand they later _say_? Its single addition to the VG09/VG10 structure is `beta_lag`, which shifts the logit of the current production ratio `q` by the child's prior-wave understood count relative to the population + study expectation at that age (`lag_baseline="population"`; the child's stable production level is separately controlled by their `q` subject intercept).

The fit is the cleanest-sampling model in the family: 0 divergences, max R-hat 1.0048, min ESS 1,599, BFMI 0.46–0.50 on every chain, and `beta_lag` itself at ESS 13,270, R-hat 1.000.

**Headline: `beta_lag` = +0.203, 89% ETI [0.093, 0.316], P(>0) = 0.999.** Positive, reliably so, and modest. The rest of this note converts that logit-scale number into units a reader can weigh, and measures what the term does and does not do.

## 2. What identifies it

The cross-lag is identified only by observations with a prior-wave comprehension source:

|                                                  |                                                |
| ------------------------------------------------ | ---------------------------------------------: |
| observations with a prior-wave understood source |                         **412 of 1,431** (29%) |
| children contributing at least one               |                                 **250 of 767** |
| gap to the lag source                            | median **6.0 months**, IQR 5.0–7.2, range 1–28 |
| age at the current wave                          |                median 36.5 months, range 12–95 |

So the estimate rests on a quarter of the frame, and on gaps of about half a year. Both facts matter below: the gaps sit exactly in the window where receptive standing tracks best (§6), and the 71% of observations for which the term is identically zero is one reason it cannot restructure the model (§5).

## 3. The effect in interpretable units

The natural unit for "ahead receptively" is `tau_subj_u` = 0.797 [0.748, 0.848], the between-child SD of persistent receptive standing. It is a large gap in substantive terms: at 24 months it separates 112 from 212 words understood (+11.6 months of comprehension age-equivalent [9.4, 14.5]); at 36 months, 215 from 360 (+17.9 [15.1, 21.7]); at 48 months, 307 from 465 (+24.1 [17.5, 33.5]).

The cross-lag effect of one SD of receptive standing, expressed three ways:

- **On the logit of `q`:** `beta_lag × tau_subj_u` = **+0.161** [0.074, 0.252].
- **Relative to the between-child spread in conversion:** **0.125** [0.058, 0.198] of `tau_subj_q` (= 1.283 [1.197, 1.373]).
- **As months of the population `q` trajectory:** +1.8 at 24 months, +1.2 at 36, +2.0 at 48. Beyond about 54 months `q` saturates and the age-equivalence is unstable; it should not be quoted there.

Scaled instead to one SD of the predictor as the model sees it — SD(`x_lag`) = 1.097, the observed residual carrying occasion movement and sampling noise on top of persistent standing — the shift is +0.222 [0.099, 0.351] logits.

## 4. The two-channel decomposition, and the number that survives every framing

Because the joint models build spoken vocabulary multiplicatively — words spoken = words understood × conversion rate `q` — a spoken-vocabulary gap between two children splits exactly into two channels: having more words available (**direct**), and converting a higher fraction of them (**cross-lag**). The split is computed by a counterfactual step: pair the upper child's comprehension with the lower child's conversion rate. That hybrid corresponds to no real child; it is a decomposition device, not a prediction.

Worked at 36 months for the interquartile contrast (children 0.674 SD either side of the median): the Q1 child understands 141 words and converts 16.0% of them — 23 spoken; the Q3 child understands 309 and converts 19.1% — 59 spoken. At the Q1 child's rate, 309 words would yield 49 spoken. So of the 36-word gap, 26 come from the bigger store and 10 from the better rate: a **share of 27%** [13, 37] carried by conversion.

The full table for that contrast:

| age (mo) | understood Q1 → Q3 | compr. gap | spoken Q1 → direct → Q3 | cross-lag adds | share of gap |
| -------: | ------------------ | ---------: | ----------------------- | -------------: | -----------: |
|       24 | 69 → 174           |    10.1 mo | 3 → 8 → 10              |      +2 [1, 3] | 28% [14, 39] |
|       30 | 111 → 256          |    18.4 mo | 9 → 21 → 25             |      +5 [2, 7] | 28% [14, 39] |
|       36 | 141 → 309          |    21.7 mo | 23 → 49 → 59            |    +10 [4, 15] | 27% [13, 37] |
|       42 | 171 → 356          |    23.7 mo | 50 → 104 → 120          |    +17 [8, 26] | 24% [12, 34] |
|       48 | 213 → 413          |    26.7 mo | 92 → 178 → 201          |   +22 [10, 35] | 20% [10, 30] |
|       54 | 259 → 469          |    30.4 mo | 139 → 253 → 278         |   +25 [11, 39] |  18% [9, 27] |
|       60 | 296 → 508          |    33.4 mo | 183 → 315 → 340         |   +25 [11, 40] |  16% [8, 24] |
|       72 | 356 → 565          |    38.6 mo | 257 → 408 → 431         |   +23 [10, 38] |  13% [6, 21] |

**The share is nearly invariant to the contrast chosen** — at 36 months it is 22% for a +0.5 SD contrast, 23% for Q3-vs-median, 26% for +1 SD and 27% for Q3-vs-Q1 — while the absolute word counts scale with it (a +0.5 SD contrast contributes only 1–9 words, rounding to +1 at 24 months). The framing-robust statement, and the one to lead with, is therefore:

> **Roughly a quarter of the spoken-vocabulary gap between children who differ in comprehension comes from converting a higher fraction of what they understand, not from having more words to convert** — declining gently to about 13–16% by 60–72 months as `q` saturates and the rate channel runs out of headroom.

Under every model before VG16 this share is exactly zero by construction: VG05–VG10 make `q` a function of age alone, so two same-age children necessarily convert at the same rate. `beta_lag = 0` and "share = 0%" are the same statement; the share _is_ the cross-lag on a readable scale.

Two illustration choices follow. **Q3-vs-Q1 is the best paired illustration**: it describes two real children in the vocabulary the tracking note already uses, and the effect stays legible in whole words at every age. **Q3-vs-median suits a single-child narrative.** A validity anchor for either: the model puts Q1–Q3 spoken at 24 months at 3–10 words against the observed median 9, IQR 4–21 at 22–26 months ([202608141600](202608141600-rank-stability-tracking.md) §5) — narrower, as quartiles of the persistent child effect with occasion and sampling variation removed should be.

## 5. What the term does not do

Three measurements bound its importance.

**It carries ~2.9% of the variance.** `var(beta_lag · x_lag)` = 0.050 against `tau_subj_q²` = 1.65: **2.9%** [0.6, 6.9] of the combined lag-plus-between-child variance in the `q` logit.

**It absorbs none of the between-child spread.** `tau_subj_q` is 1.2855 [1.2006, 1.3775] in VG10 and 1.2826 [1.1975, 1.3733] in VG16 — a 0.2% shift. `tau_subj_u` is likewise unmoved (0.7970 vs 0.7966). The cross-lag sits alongside the persistent structure; it does not explain any of it.

**A persistent receptive–expressive link remains that the model cannot express.** With the cross-lag already fitted, the correlation between the realised understood and `q` subject intercepts across the 767 children is **+0.135** [0.087, 0.180]. The two random-effect blocks have independent priors and no correlation parameter, so a leftover correlation in the realised effects is the only place the model can put that association — direct evidence that the structure wants a correlated random-effect block (see [202608151140](202608151140-cross-lag-not-for-models-of-record.md) §4 and the VG20 proposal).

**And the fitted coefficient is a floor, not an estimate, of the effect of true standing.** `x_lag` is an observed residual: persistent standing plus occasion movement plus sampling noise. Its reliability as a measure of persistent standing is `tau_subj_u²/var(x_lag)` = **0.526** [0.465, 0.598]; under classical errors-in-variables attenuation the disattenuated coefficient is **0.383** [0.173, 0.610]. Quote +0.20 as the fitted association and 0.53 as the reliability that attenuates it.

## 6. How it sits with the tracking analysis

Four connections to [202608141600](202608141600-rank-stability-tracking.md), each strengthening the other's reading.

**The lag gaps sit exactly in the well-tracked window.** The median lag is 6 months, IQR 5–7.2. DS comprehension correlates 0.701 (1–6 months) and 0.780 (6–12), disattenuating to roughly 0.83/0.90. Essentially all of `beta_lag`'s information comes from the interval where receptive standing is most stable — which is why the coefficient is estimable, and why the collapse to ρ = 0.238 beyond two years does not contaminate it.

**The AR(1) rejection and the attenuation are the same fact seen twice.** §10.4 of the tracking note rejected the latent AR(1) on both outcomes (`ell → 0`): the within-child deviation has no memory beyond the occasion. If the non-persistent part of `x_lag` has no memory, it is pure noise for predicting current `q`, and only the persistent component carries signal — precisely the errors-in-variables structure of §5, whose predicted attenuation factor `var(persistent)/var(x_lag)` is the 0.526 measured there. The two analyses were done independently and agree on the mechanism.

**The small variance share mirrors the comprehension-slope power limit.** The tracking note found comprehension's random slope worth 27.09 across all 610 children but 0.82 on the 253 with repeats — its widening is cross-sectional, not measurable within-child drift. VG16 measures the mirror image on production: a reliably positive but small within-child lead on top of a large persistent spread it does not touch. Both say the same thing about this design: the repeated measures establish direction, not magnitude.

**The honest reading is between-child, not within-child.** Combining the two points above: if the occasion component carries no memory, `beta_lag` is not really measuring a _lag_. It is measuring a **between-child association between persistent receptive standing and persistent conversion efficiency**, read through a noisy proxy — and the +0.135 residual correlation in §5 is the part of that association the proxy missed. The clean estimator of the association is a correlated random-effect block on `(delta_subj_u, delta_subj_q)`, which is the follow-up model proposed as VG20 in [#224](https://github.com/dseinternational/vocabulary-growth/issues/224) (VG19 being reserved by the child-slope plan, [202608141900](202608141900-child-slope-implementation-plan.md)). VG16's report framing should present `beta_lag` as directional evidence with a credible floor on its size, described as between-child rather than within-child.

## 7. History of the estimate, and two stale documentation claims

`beta_lag` has never been null at any tier above `dev`:

| fit                                | `beta_lag` | 89% ETI            | note                                                                                                                                    |
| ---------------------------------- | ---------: | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `dev`, at introduction             |    ≈ +0.05 | spans 0            | source of the "≈ null" framing                                                                                                          |
| `test`, anchored                   |      0.308 | [0.182, 0.438]     | [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §16                                                                 |
| `test`, post-Edgin linkage rebuild |      0.167 | [0.053, 0.283]     | [202608031341](202608031341-test-refit-after-data-and-prior-changes.md) §4.7: re-linking 46 children across CDI forms roughly halved it |
| **`rep`, model of record**         |  **0.203** | **[0.093, 0.316]** | this note                                                                                                                               |

Two published statements still carry the `dev`-tier reading and are wrong against the model of record: the VG16 row of `docs/models/README.md` ("Population-relative headline (≈ null)") and the callout in `docs/models/vg16/index.qmd` ("essentially null at dev-tier (β ≈ +0.05, interval spanning 0)") — the latter's within-child-contrast caution remains accurate; it is the headline figure that is stale. [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §23 already flagged the discrepancy for checking before either figure was quoted. Both corrections are pending as of this note.

## 8. Caveats

- **Not causal.** The association is observed across children, not manipulated. Nothing here says raising comprehension would produce the conversion gain.
- **The decomposition's "direct" column is a counterfactual hybrid**, not a prediction for any child; present it only as a decomposition step.
- **All illustrated contrasts hold the child's advantage as noise-free persistent standing** — the attenuated reading. The effect of true receptive standing is larger (§5).
- **"Better conversion" names the arithmetic, not a mechanism.** Articulation, expressive-language ability, exposure feeding both sides, or reporting tendency are all compatible with it.
- **The share's decline with age is a ceiling effect** — the same `beta_lag` yields a smaller slice as `q` approaches saturation — not evidence the association weakens.
- The per-draw loops in the reproduction script thin the posterior (default step 8, 4,500 of 36,000 draws); third-decimal wobble against a full-posterior pass is expected.
