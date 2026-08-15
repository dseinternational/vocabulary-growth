# Can the within-child comprehension-to-production lead be estimated? Scope, and the Stage 1 result

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **This note's original conclusion was wrong, and §7 is the retraction.** It was written as a scope and concluded the analysis was very likely underpowered. Stage 1 was then run, and it refutes both the note's own power arithmetic and the mechanism the VG16 report attributes the problem to. The scoping sections are kept as written so the correction is legible. Reproduced by `scripts/experiments/vg16_within_lag_bias.py`; figures are the 2026-08-15 run at 150 replicates.

## 1. Why this question came up

VG16 finds that children further ahead in comprehension convert a higher fraction of it to speech ([202608151120](202608151120-vg16-cross-lag-quantified.md)). The natural practice question is whether that licenses a hypothesis about **intervention**: would teaching vocabulary raise production beyond the words directly taught?

It does not, for a reason that is structural rather than a matter of caution. The association is **between children**; an intervention acts **within** a child. The between-child pattern is equally consistent with a stable child-level characteristic — cognitive profile, hearing, oral-motor capacity — driving both comprehension and conversion, under which moving one child's comprehension would change conversion not at all. Distinguishing those requires a _within-child_ estimate: does a child who gains comprehension faster than expected subsequently convert more?

There is a second obstacle worth recording because it survives any amount of extra data, and **nothing below softens it**. `q` is spoken ÷ understood, so teaching words raises the **denominator** immediately. VG16's association describes children who reached their comprehension level developmentally, with production having had time to follow — an equilibrium relationship among children. An intervention is a perturbation of one input, and nothing guarantees it lands on the same curve. **Even a perfectly estimated within-child lead would not answer the intervention question.**

## 2. What was already known, and what was only assumed

VG16 reports the population-relative baseline. Its within-child (RI-CLPM) alternative — which additionally subtracts the child's _own_ understood intercept — was fitted as a diagnostic and came out **strongly negative**: `beta` ≈ −0.60, 89% interval [−0.85, −0.35] at `dev` tier, recorded in [`docs/models/vg16/index.qmd`](../docs/models/vg16/index.qmd) and attributed there to a short-T (Nickell-type) / errors-in-variables artefact of regressing on a lag tied to the child's own random intercept.

**That attribution had been reasoned, never demonstrated.** Making it the first thing to test, rather than the premise to build on, is the one methodological choice in this note that survived contact with the results.

## 3. The feasibility census

The census below counts children by distinct comprehension waves, on the reasoning that a within-child estimator needs three or more because two waves give one lagged observation, which vanishes under within-child demeaning. **That reasoning is wrong for the estimator VG16 actually uses** — see §6.3 — but the census is still the right description of the data.

| comprehension waves | children |
| ------------------: | -------: |
|                   1 |      366 |
|                   2 |      144 |
|                   3 |   **83** |
|                   4 |       11 |
|                   5 |        6 |
|       **total ≥ 3** |  **100** |

**100 children** have three or more comprehension waves, not the 182 quoted in conversation on 2026-08-15 — that figure was children with three or more observations of _any_ kind, from [202608141600](202608141600-rank-stability-tracking.md) §10.6, and is the wrong census for this question. Of the 100, 83 have exactly three. They contribute 240 of the frame's 412 lagged observations, span 11–90 months, come from six of the fourteen sources with two dominating, and carry a median within-child comprehension window of 13 months (IQR 12–15).

## 4. The original power sketch — superseded, and wrong

This section argued the analysis was underpowered. Within-child demeaning of the ≥3-wave subset leaves an `x_lag` SD of 0.353 — 19% of that subset's variance — and 140 degrees of freedom, from which a normal-approximation sketch gave SE(`beta`) between 0.106 and 0.183 and power of 31–62% at the fitted effect size.

**Every number there is defensible; the sketch is nonetheless the wrong calculation**, because it prices a fixed-effects demeaning estimator that no one proposes to use. §6 measures the realised SE directly and gets 0.10 — at the optimistic end of the sketch's range, and achieved on more data than the sketch assumed available.

## 5. Stage 1 as designed

Simulate outcomes from VG16's own posterior on the **real observed wave structure** at a known `beta_lag`, including `beta_lag = 0`, and apply both baselines to the same simulated data. The two estimators differ only in whether the child's own intercept is subtracted, so any divergence between them is the mechanism under test rather than a simulation artefact.

A bespoke simulator was needed because VG16 is deliberately excluded from `scripts/fit_recovery.py` — its cross-lag predictor is a function of the outcome — so simulation must walk each child's waves in age order, deriving `x_lag` at wave t from the _already simulated_ understood count at wave t−1. Estimation is by marginal likelihood with the child effects integrated out by Gauss-Hermite quadrature, holding the population trajectory, study effects and dispersion at their fitted values.

A third estimator was added after the first results: **within-oracle**, identical to the within estimator but given each child's _true_ simulated intercept. If the plug-in and oracle variants agree, estimation error in that intercept — the errors-in-variables half of the attributed mechanism — is not producing the bias.

## 6. Stage 1 results

150 replicates per cell. Columns are the estimator applied; rows are the baseline the data were generated under.

| truth | generated under | population: mean (bias) | within: mean (bias) | within-oracle: mean (bias) |
| ----: | --------------- | ----------------------: | ------------------: | -------------------------: |
| 0.000 | population      |     −0.004 (**−0.004**) |     −0.014 (−0.014) |            −0.013 (−0.013) |
| 0.203 | population      |      0.202 (**−0.001**) |      0.222 (+0.019) |             0.204 (+0.001) |
| 0.400 | population      |      0.402 (**+0.002**) |      0.454 (+0.054) |             0.416 (+0.016) |
| 0.000 | within          |         −0.001 (−0.001) | −0.001 (**−0.001**) |            −0.001 (−0.001) |
| 0.203 | within          |          0.120 (−0.083) |  0.197 (**−0.006**) |             0.199 (−0.004) |
| 0.400 | within          |          0.224 (−0.176) |  0.392 (**−0.008**) |             0.395 (−0.005) |

Estimator SD is 0.07–0.08 (population) and 0.10–0.11 (within and oracle) throughout.

### 6.1 The −0.60 does not reproduce, and the attributed mechanism is refuted

**At `beta_lag = 0` every estimator returns zero** — the largest deviation anywhere is −0.014, against the −0.60 to be explained. The short-T / errors-in-variables mechanism as attributed does not produce a large negative bias in this design, at this wave structure, at any truth tested.

The oracle column closes off the remaining escape. Plug-in and oracle agree to within 0.02 in every row, and to within 0.005 wherever the estimator matches the generating baseline. **Estimation error in the child's understood intercept contributes essentially nothing.** That is precisely the quantity the errors-in-variables story blames.

So the −0.60 needs a different explanation. Three candidates remain, and this note cannot separate them:

- **The joint-estimation ridge**, which is the leading candidate. In the real model `delta_subj_u` is estimated _jointly_ with `beta_lag`, so the spoken likelihood feeds back onto the understood intercept through `x_lag`. The two-step estimator here deliberately severs that feedback — it estimates the intercept from the understood data alone — so this simulation is blind to it by construction.
- **Non-convergence.** The −0.60 is a `dev`-tier figure, and the project's own documentation holds that `dev` under-converges the hierarchical models.
- **Misspecification.** Simulating from the model cannot reveal a way the model is wrong about the real data.

### 6.2 Each baseline is unbiased for its own generating process, and mixing them costs

Where estimator and generating baseline agree, bias is ≤ 0.008. Where they differ it is orderly: **the population estimator applied to within-generated data attenuates by about 45%** (0.120 for a truth of 0.203; 0.224 for 0.400), while the within estimator applied to population-generated data inflates mildly (+0.019, +0.054). The asymmetry is worth carrying: if the truth were genuinely within-child, VG16's reported +0.203 would be an underestimate of it by roughly a factor of two.

### 6.3 Two-wave children are not excluded, which is where §4 went wrong

The within estimator uses **361 observations from 240 children — 141 of those observations from 2-wave children.** They contribute because VG16's construction subtracts a _partially pooled, shrunk_ intercept, not a within-child mean: a 2-wave child's lagged observation retains its deviation from that shrunk average instead of being annihilated. §3's premise and §4's degrees-of-freedom count both assumed a fixed-effects wipe, and both are therefore too pessimistic about the data available.

## 7. Revised recommendation

**The original recommendation — run Stage 1, expect it to fail, treat the failure as the deliverable — is withdrawn.** Stage 1 ran and did not fail; it refuted the reason for expecting failure.

1. **Correct the VG16 report.** The `dev`-tier within-child figure is described there as a short-T / errors-in-variables artefact. That attribution is not supported: a simulation reproducing exactly that structure, on exactly this wave design, shows no such bias, and an oracle-intercept variant rules out the errors-in-variables half specifically. The honest statement is that the within-child estimate is anomalous and its cause is not yet established.
2. **Stage 2 is now worth doing, on the evidence rather than despite it.** A two-step or partially decoupled estimator of the within-child lead is unbiased in simulation with SE ≈ 0.10 against an effect of about 0.20. That is a usable estimate, not a foregone null, and it needs no restriction to ≥3-wave children.
3. **Test the joint-estimation ridge directly**, since it is now the leading explanation and the cheapest decisive check: refit VG16 with `lag_baseline="within"` at `test` rather than `dev`, on simulated data with a known `beta_lag`. If the −0.60 reappears where the two-step estimator is unbiased, the ridge is confirmed and the two-step estimator is the remedy.
4. **§1's second obstacle still stands.** None of this makes the within-child lead an answer to the intervention question, and the report should not present it as one.

[#224](https://github.com/dseinternational/vocabulary-growth/issues/224) (VG20, correlated subject random effects) remains the better instrument for the _between_-child question, on all 767 children. It is unaffected by any of this.

## 8. Caveats, and corrections to this note

- **An earlier version of the simulator got the sign wrong.** Regressing `logit(y_s / y_u)` on the lag returned _positive_ bias under every truth including zero. 159 of 973 conditional rows (16%) have zero spoken words, and `logit(0)` clips to −9.21 — a large negative outlier arising for exactly the small-vocabulary children who also have a low lag value, manufacturing positive correlation. Any estimator that reduces the outcome to a ratio inherits this; the beta-binomial likelihood does not. Had that version been reported it would have been confidently wrong in the opposite direction to the finding it was testing.
- **The estimators are not VG16's estimator.** They are two-step marginal likelihood with the trajectory, study effects and dispersion held at fitted values. This is deliberate — it is what isolates the mechanism — but it means the realised SE of 0.10 is optimistic for a real analysis that must estimate those too, and that §6.1's leading candidate is untestable here.
- **Simulating from the model cannot detect misspecification**, so "unbiased" throughout means "unbiased if VG16's structure is right".
- **The wave census counts distinct ages with an understood measure** in VG16's analysis frame after the DS pool's default masking; reinstating any masked defect class would change it.
- The power sketch in §4 is retained as written and superseded by §6; the child count in §3 corrects a figure quoted in conversation.
