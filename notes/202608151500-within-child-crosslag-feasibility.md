# Can the within-child comprehension-to-production lead be estimated at all? A feasibility scope

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **Scoping note, not an analysis and not a plan to execute as written.** Its conclusion is that the analysis it scopes is very likely underpowered, and that the cheap decisive test should be run before any of the rest. Feasibility figures computed from VG16's analysis frame (`scripts/experiments/vg16_crosslag_quantification.py` shares the wave-reconstruction logic). Companion notes: [202608151120](202608151120-vg16-cross-lag-quantified.md), [202608151140](202608151140-cross-lag-not-for-models-of-record.md).

## 1. Why this question came up

VG16 finds that children further ahead in comprehension convert a higher fraction of it to speech ([202608151120](202608151120-vg16-cross-lag-quantified.md)). The natural practice question is whether that licenses a hypothesis about **intervention**: would teaching vocabulary raise production beyond the words directly taught?

It does not, for a reason that is structural rather than a matter of caution. The association is **between children**; an intervention acts **within** a child. The between-child pattern is equally consistent with a stable child-level characteristic — cognitive profile, hearing, oral-motor capacity — driving both comprehension and conversion, under which moving one child's comprehension would change conversion not at all. Distinguishing those requires a _within-child_ estimate: does a child who gains comprehension faster than expected subsequently convert more?

There is a second, subtler obstacle worth recording because it survives any amount of extra data. `q` is spoken ÷ understood, so teaching words raises the **denominator** immediately. VG16's association describes children who reached their comprehension level developmentally, with production having had time to follow — an equilibrium relationship among children. An intervention is a perturbation of one input, and nothing guarantees it lands on the same curve. **Even a clean within-child estimate would not by itself answer the intervention question.**

## 2. What is already known, and why the current estimate is unusable

VG16 fits the population-relative baseline as its headline precisely because it is bias-robust. The pure within-child (RI-CLPM) baseline — which additionally subtracts the child's _own_ understood intercept — was fitted as a diagnostic and is **strongly and spuriously negative**: `beta` ≈ −0.60, 89% interval [−0.85, −0.35] at `dev` tier, recorded in [`docs/models/vg16/index.qmd`](../docs/models/vg16/index.qmd) and attributed there to a short-T (Nickell-type) / errors-in-variables artefact of regressing on a lag tied to the child's own random intercept.

**The attributed mechanism has not been demonstrated, only reasoned.** That matters, because the remedy depends on it, and §4 makes reproducing it the first stop-gate rather than assuming it.

## 3. The feasibility census, and it is thin

The within-child estimator needs children with **three or more comprehension waves**: two waves give one lagged observation, which vanishes entirely under within-child demeaning.

| comprehension waves | children |
| ------------------: | -------: |
|                   1 |      366 |
|                   2 |      144 |
|                   3 |   **83** |
|                   4 |       11 |
|                   5 |        6 |
|       **total ≥ 3** |  **100** |

So **100 children**, not the 182 quoted in conversation on 2026-08-15 — that figure was children with three or more observations of _any_ kind, from [202608141600](202608141600-rank-stability-tracking.md) §10.6, and is the wrong census for this question. Of the 100, **83 have exactly three waves**, which matters because short-T bias falls as O(1/T): three waves reduce it by roughly a third against two, not to zero.

What those children contribute:

|                                     |                                               |
| ----------------------------------- | --------------------------------------------: |
| observations                        |              384 (324 understood, 382 spoken) |
| lagged observations                 |          **240** of the 412 in the full frame |
| lagged observations per child       |          2 for 77 children, 3 for 6, 4 for 17 |
| **within-child degrees of freedom** |                                       **140** |
| age range                           |                    11–90 months (median 30.5) |
| comprehension span per child        |       median **13 months**, IQR 12–15, max 28 |
| studies represented                 | 6 (113 / 97 / 75 / 45 / 30 / 24 observations) |

Two features to note. The subset spans only six of the fourteen sources and is dominated by two of them, so anything estimated here is **partly a statement about those studies**. And the median within-child comprehension window is about a year — short relative to the developmental change the question is about.

## 4. The decisive number: within-child variation is 19% of the total

Power for a within estimator comes from variation in the predictor _after removing each child's own mean_, and that is where this collapses:

| quantity                                         |     value |
| ------------------------------------------------ | --------: |
| SD of `x_lag`, all 412 lagged observations       |     1.096 |
| SD of `x_lag`, the 240 from ≥3-wave children     |     0.810 |
| **SD after within-child demeaning**              | **0.353** |
| within variance as a share of the subset's total |   **19%** |
| Σ squared within-child deviations                |     29.96 |

At the fitted effect size (`beta` = 0.203) the within-child signal is 0.203 × 0.353 ≈ **0.07 logits**, against a residual on the `q` logit whose occasion component alone is about 0.58 ([202608141600](202608141600-rank-stability-tracking.md) §10.3) before observation-level sampling noise. The implied precision:

| assumed residual SD on the `q` logit | SE(`beta`) | 89% half-width | power at `beta` = 0.203 |
| ------------------------------------ | ---------: | -------------: | ----------------------: |
| 0.58 — occasion term only            |      0.106 |          0.169 |                 **62%** |
| 0.80 — plus some sampling noise      |      0.146 |          0.234 |                 **42%** |
| 1.00 — realistic observation level   |      0.183 |          0.292 |                 **31%** |

**Every row ignores the short-T bias entirely, so each is an upper bound.** The realistic reading is that a clean within-child estimate would have somewhere between a third and two-thirds chance of excluding zero even if VG16's effect is exactly right and the bias were fully solved — and a wide interval either way. A null result would be uninformative, which is the specific failure mode worth avoiding: it would be read as evidence against the effect when it is evidence of nothing.

This corrects the "a day's analysis" framing offered in conversation. The _work_ is about that size; what it would deliver is not.

## 5. Proposed design, with the cheap test first

**Stage 1 — reproduce the bias by simulation. This is the stop-gate.** Simulate outcomes from VG16's own posterior on the _actual observed wave structure_, at known `beta_lag`, including `beta_lag = 0`. Fit the within-child estimator to the simulated data. Three things are then established at once, and none needs new data:

- whether the −0.60 is reproduced when the truth is zero, confirming or refuting the attributed mechanism;
- how much of it survives restriction to the ≥3-wave children, which is the entire premise of this exercise;
- the realised SE, replacing §4's assumed-`sigma` bracket with a measured one.

The project's parameter-recovery machinery does exactly this shape of job (`scripts/fit_recovery.py`, `docs/runbooks/parameter-recovery.md`), and simulating at `beta_lag = 0` is the honest null.

**Gate: proceed only if the bias at T ≥ 3 is small relative to the effect, and the realised SE gives better than ~60% power.** On §4's arithmetic the expected outcome is that it does not, and the correct action is then to stop and record the negative feasibility result — which is a publishable limitation, not a wasted day.

**Stage 2 — if and only if Stage 1 passes.** Estimate on the subset. Options, cheapest first: the within baseline VG16 already implements, restricted to ≥3-wave children; a bias-corrected variant if Stage 1 identifies the mechanism precisely; or joint estimation of the child's understood trajectory with the lag, so the intercept is not a plug-in. Registered as a VG16 sensitivity variant, never a model of record.

**Stage 3 — interpretation limits, fixed in advance.** Write the caveats before seeing the estimate: six studies dominated by two; a one-year median window; and §1's equilibrium-versus-perturbation point, which no within-child estimate resolves.

## 6. Recommendation

**Run Stage 1. Expect it to fail, and treat the failure as the deliverable.** It is a day at most, it needs no new data and no VM time, and it converts "the within-child estimate is biased, we think we know why" into a demonstrated and quantified statement — which is what the report's limitations section should say either way.

**Do not schedule Stage 2 in advance.** On present arithmetic it is unlikely to be reached, and committing to it invites reporting an underpowered estimate because the work was budgeted.

Two things are better uses of the same effort. **[#224](https://github.com/dseinternational/vocabulary-growth/issues/224) (VG20, correlated subject random effects)** answers the _between_-child question properly, on all 767 children rather than 100, and is the estimate VG16's cross-lag is a noisy proxy for. And the intervention question itself is answerable only by an intervention study; this cohort cannot reach it, and §1's second obstacle means it could not even in principle with more waves.

## 7. Caveats on this note

- **The power arithmetic is a normal-approximation sketch**, not a simulation: a fixed-effects within estimator with a plug-in residual SD, ignoring the beta-binomial likelihood, the hierarchical shrinkage that would partially pool the 100 children, and the bias. Stage 1 replaces it with a measured figure; treat the 31–62% bracket as an order of magnitude.
- **Partial pooling could do better than the fixed-effects sketch**, since a Bayesian hierarchical fit borrows strength from the 2-wave and singleton children for the nuisance parameters. That is a reason to measure the SE in Stage 1 rather than to assume §4 is the last word — but it does not create within-child variation where there is none.
- **The wave census counts distinct ages with an understood measure** in VG16's analysis frame, after the DS pool's default masking. Reinstating any masked defect class would change it.
