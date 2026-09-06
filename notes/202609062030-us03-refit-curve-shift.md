# The comprehension curve moved, and `us_03`'s departure went with it (#289 task 2.1, preliminary)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-06, measured on the **refitted VG10** while VG19, VG20, VG22 and VG24 were still sampling. VG20 is the Down syndrome model of record and is what task 2.1 has to be decided on, so **everything here is preliminary** and must be repeated on VG20 before 2.2 is accepted or rejected. It is recorded now because the VG10 result is unambiguous enough to change what 2.2 should expect.

## The question

[#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 2.1 sets a two-branch test. If the comprehension curve drops and `us_03`'s band-to-band spread collapses, the pre-ingestion curve was too steep and there is nothing structural to fix. If `us_03` keeps an age-varying offset _against a curve that has moved to accommodate it_, study effects are genuinely age-varying and task 2.2 — a study-level term whose offset varies with age — is required.

Both halves have to be measured. Only the first is a curve comparison.

## Half one: the curve moved, and it flattened

VG10's understood curve against the pre-ingestion figure cache, at the reference child:

| age | published | refitted | change |
| --- | --------: | -------: | -----: |
| 18  |    0.0634 |   0.0666 |  +4.9% |
| 24  |    0.1423 |   0.1369 |  −3.8% |
| 30  |    0.2198 |   0.2057 |  −6.4% |
| 36  |    0.2727 |   0.2590 |  −5.0% |

It is not a level shift: the young end went **up** and 24–36 months came **down**. That is the shape #288 predicted from the other side — `us_03` sitting on the fitted curve at 18–19 months and below it from 20 — and the refit has absorbed it by flattening rather than by translating.

**Production moved only because comprehension did.** The conversion rate `q` is unchanged (+0.2%, +0.7%, +0.9%, +0.6% at 18/24/30/36 months), and the spoken curve moved by almost exactly comprehension's percentages (+5.3%, −3.0%, −5.6%, −4.4%). Since `S = p_U · q`, the whole spoken shift is inherited. This is not evidence that production is robust to the new cohort: **`us_03` contributes no production data at all.** Its expressive cell is a produced union — "understands and says _or signs_", the study authors' wording — so `spoken` is NULL by design and its 284 rows enter as ordinary understood-without-spoken observations. VG01, whose outcome is spoken, gains **zero** `us_03` rows and its curve moves 0.1%.

`us_03` is now the largest single study in the comprehension frame: **284 of VG02's 1,260 rows (22.5%)**, and 284 of 1,708 (16.6%) in the joint frames.

## Half two: the departure did **not** survive the refit

Residual = `logit(observed count / that row's form length) − logit(study's own fitted curve / n_trials)`, where the fitted curve is `study_fans.csv`, which is `sigmoid(f_u + d_u[study])` — the population trajectory plus that study's own constant offset, at a zero child effect. So this measures whether a **constant** study offset is enough, which is exactly what 2.2 asks.

> [!WARNING]
> `study_fans.csv` is written on the model's reporting `n_trials` (810), **not** on each study's own form length. Dividing it by `survey_vocab_max` overstates expected `p` by 810/396 for `us_03` and produced a spurious −0.48 mean residual on the first pass. Within-study differences (the slope and the swing) survived the correction almost unchanged, but the level did not.

OLS slope of the residual on age across 17–25 months — `us_03`'s first-visit window, applied identically to every study with at least 20 observations in it:

| study   | obs | children |   slope |     se |     t |
| ------- | --: | -------: | ------: | -----: | ----: |
| `ie_02` |  32 |       21 | −0.2236 | 0.1156 | −1.94 |
| `us_03` | 180 |      180 | −0.1422 | 0.1225 | −1.16 |
| `us_01` |  93 |       55 | +0.0218 | 0.0388 | +0.56 |
| `uk_05` |  20 |       16 | +0.0447 | 0.0828 | +0.54 |
| `es_01` |  44 |       44 | +0.0859 | 0.0626 | +1.37 |

**`us_03`'s post-refit age slope is not distinguishable from zero, and it is not even the steepest in the pool** — `ie_02`'s is steeper. `us_03`'s standard error is the trustworthy one of the two: it has exactly **one row per child** in this window (180 rows, 180 children), where `ie_02` has 32 rows from 21 children and `us_01` 93 from 55, so their errors are optimistic and their `t` values overstated.

On the current evidence the answer is **branch one**: the pre-ingestion curve was too steep, the refit has absorbed the departure, and **task 2.2 is not justified**.

## What is left is a data question, not a structural one

The one feature that does not fit a smooth story is a single month:

| month    |  17 |    18 |    19 |        20 |    21 |    22 |    23 |    24 |
| -------- | --: | ----: | ----: | --------: | ----: | ----: | ----: | ----: |
| n        |   1 |    18 |    46 |        38 |    32 |    26 |    13 |     5 |
| residual |   — | +0.49 | +0.97 | **−0.71** | +0.66 | −0.13 | −0.34 | +0.34 |

Month 20 sits 1.4–1.7 logits below the months either side of it. That is the group [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 0.3 already asks the data providers about — 38 children with a median of 28 words understood, between a 19-month group at 52 and a 21-month group at 60.

It is also most of the band contrast. The 17–19 versus 20–22 swing is **+0.955 (se 0.414, z 2.31)** with month 20 in and **+0.554 (se 0.431, z 1.29)** with it out. So the one statistic that looked like an age-varying study effect is substantially one anomalous month, and task 0.3's answer bears on it more than task 2.2's design does.

## What this does not establish

VG10 is not the model of record. The measurement is against each study's own fitted curve, so it tests the **shape** of a study's departure and not its level — the level is absorbed by the fitted offset and by shrinkage, and `us_03`'s overall mean residual is +0.286. The residuals also carry each child's own effect, which the fan excludes; within `us_03`'s first-visit window those are one per child and independent, but they are not zero. And a slope that cannot be distinguished from zero on 180 children is not the same as a slope that is zero: this rules out an effect of the size #288 saw pre-refit, not every effect.

Repeat on VG20 before closing 2.1.
