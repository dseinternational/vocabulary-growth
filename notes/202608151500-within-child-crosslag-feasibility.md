# The within-child cross-lag: scope, and what the anomaly turned out to be

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **This note began as a scope, concluded the analysis was underpowered, and was wrong on both its own arithmetic and the mechanism it assumed.** §§3–4 are the superseded scoping; §6 refutes the attributed mechanism by simulation; §7 identifies the actual cause. The scoping sections are kept as written so the corrections are legible. Reproduced by `scripts/experiments/vg16_within_lag_bias.py` (§6) and `scripts/experiments/vg16_within_ridge_arm.py` (§7); figures are the 2026-08-15 runs.

## 1. Why this question came up

VG16 finds that children further ahead in comprehension convert a higher fraction of it to speech ([202608151120](202608151120-vg16-cross-lag-quantified.md)). The natural practice question is whether that licenses a hypothesis about **intervention**: would teaching vocabulary raise production beyond the words directly taught?

It does not, for a reason that is structural rather than a matter of caution. The association is **between children**; an intervention acts **within** a child. The between-child pattern is equally consistent with a stable child-level characteristic — cognitive profile, hearing, oral-motor capacity — driving both comprehension and conversion, under which moving one child's comprehension would change conversion not at all. Distinguishing those requires a _within-child_ estimate.

A second obstacle survives any amount of extra data, and **nothing below softens it**. `q` is spoken ÷ understood, so teaching words raises the **denominator** immediately. VG16's association describes children who reached their comprehension level developmentally, with production having had time to follow — an equilibrium relationship among children. An intervention perturbs one input, and nothing guarantees it lands on the same curve. **Even a perfectly estimated within-child lead would not answer the intervention question.**

## 2. What was assumed, and why it was worth testing

VG16 reports the population-relative baseline. Its within-child (RI-CLPM) alternative — which additionally subtracts the child's _own_ understood intercept — was fitted as a diagnostic and came out **strongly negative**: `beta` ≈ −0.60, 89% interval [−0.85, −0.35] at `dev` tier, attributed in the VG16 report to a short-T (Nickell-type) / errors-in-variables artefact.

**That attribution had been reasoned, never demonstrated.** Making it the first thing to test, rather than the premise to build on, is the one methodological choice here that survived contact with the results.

## 3. The feasibility census

The census counts children by distinct comprehension waves, on the reasoning that a within-child estimator needs three or more because two waves give one lagged observation, which vanishes under within-child demeaning. **That reasoning is wrong for the estimator VG16 actually uses** (§6.3), but the census still describes the data.

| comprehension waves | children |
| ------------------: | -------: |
|                   1 |      366 |
|                   2 |      144 |
|                   3 |   **83** |
|                   4 |       11 |
|                   5 |        6 |
|       **total ≥ 3** |  **100** |

**100 children** have three or more comprehension waves, not the 182 quoted in conversation on 2026-08-15 — that figure counted three or more observations of _any_ kind, from [202608141600](202608141600-rank-stability-tracking.md) §10.6, and is the wrong census. They contribute 240 of the frame's 412 lagged observations, span 11–90 months, come from six of fourteen sources with two dominating, and carry a median within-child comprehension window of 13 months.

## 4. The original power sketch — superseded, and wrong

This section argued the analysis was underpowered: within-child demeaning of the ≥3-wave subset leaves an `x_lag` SD of 0.353 (19% of that subset's variance) and 140 degrees of freedom, implying SE(`beta`) of 0.106–0.183 and power of 31–62%.

**The numbers are defensible; the calculation is the wrong one**, because it prices a fixed-effects demeaning estimator nobody proposed using. §6 measures the realised SE at 0.10, and the real-data fit in §7 achieves a posterior SD of 0.119 while using far more data than the sketch assumed available.

## 5. Stage 1 as designed

Simulate outcomes from VG16's own posterior on the **real observed wave structure** at a known `beta_lag`, including zero, and apply both baselines to the same simulated data. A bespoke simulator was needed because VG16 is excluded from `scripts/fit_recovery.py` — its predictor is a function of the outcome — so simulation walks each child's waves in age order, deriving `x_lag` at wave t from the already-simulated understood count at t−1. Estimation is by marginal likelihood with child effects integrated out by Gauss-Hermite quadrature.

A **within-oracle** variant was added after the first results: identical to the within estimator but given each child's _true_ intercept. If plug-in and oracle agree, estimation error in that intercept — the errors-in-variables half of the attributed mechanism — is not the cause.

## 6. Stage 1: the attributed mechanism is refuted

150 replicates per cell. Columns are the estimator applied; rows the generating baseline.

| truth | generated under | population | within | within-oracle |
| ----: | --------------- | ---------: | -----: | ------------: |
| 0.000 | population      |     −0.004 | −0.014 |        −0.013 |
| 0.203 | population      |      0.202 |  0.222 |         0.204 |
| 0.400 | population      |      0.402 |  0.454 |         0.416 |
| 0.000 | within          |     −0.001 | −0.001 |        −0.001 |
| 0.203 | within          |      0.120 |  0.197 |         0.199 |
| 0.400 | within          |      0.224 |  0.392 |         0.395 |

Estimator SD is 0.07–0.08 (population) and 0.10–0.11 (within, oracle).

**At `beta_lag = 0` every estimator returns zero** — largest deviation 0.014, against the −0.60 to be explained. Plug-in and oracle agree to within 0.02 everywhere, so **estimation error in the child's understood intercept contributes essentially nothing** — precisely the quantity the errors-in-variables story blames.

Two by-products. Each baseline is unbiased for its own generating process (bias ≤ 0.008). And **the population estimator attenuates within-generated truth by about 45%**, while the within estimator inflates population-generated truth mildly.

### 6.3 Two-wave children are not excluded

The within estimator uses **361 observations from 240 children — 141 of those from 2-wave children.** They contribute because VG16 subtracts a _partially pooled, shrunk_ intercept, not a within-child mean, so a 2-wave child's lagged observation retains its deviation from that shrunk average. §3's premise and §4's degrees-of-freedom count both assumed a fixed-effects wipe.

## 7. Stage 2: fitting the real model, and the actual cause

§6 severs one thing the real model has — in VG16 `delta_subj_u` is estimated **jointly** with `beta_lag`, so the spoken likelihood feeds back onto the intercept through `x_lag`. Four arms fit the _actual_ PyMC model at `test` (not `dev`, which could not separate a ridge from non-convergence):

| arm        | data                      | fitted baseline | `beta_lag` | 89% ETI         | fit diagnostics                         |
| ---------- | ------------------------- | --------------- | ---------: | --------------- | --------------------------------------- |
| truth-zero | simulated, `beta` = 0     | within          | **+0.063** | [−0.091, 0.222] | 1 div, max R-hat 1.041, min ESS 265     |
| truth-plus | simulated, `beta` = 0.203 | within          | **+0.380** | [0.216, 0.546]  | 1 div, max R-hat 1.015, min ESS 362     |
| **real**   | **real data**             | within          | **+0.103** | [−0.085, 0.294] | **0 div**, max R-hat 1.014, min ESS 400 |
| pop-null   | simulated, `beta` = 0     | population      | **+0.038** | [−0.054, 0.131] | 1 div, max R-hat 1.012, min ESS 353     |

`beta_lag` is among the best-sampled parameters in every arm (ESS 2,063–5,667, R-hat ≤ 1.002); where an arm misses the convergence gate it does so on the understood GP length-scale and amplitude, the known GP/linear ridge in this family, unrelated to the cross-lag.

### 7.1 The −0.60 was `dev`-tier non-convergence

**The real-data arm settles it.** The same within baseline, on the same real data, at `test` instead of `dev`, gives **+0.103** — positive, containing zero, with zero divergences. The −0.60 does not survive an adequate sampling budget, and no simulated arm reproduces a negative value at any truth.

### 7.2 The ridge exists, but pushes the other way

The joint fit is not unbiased: at a within-generated truth of 0.203 it returns 0.380, against 0.197 from the two-step estimator on the same generating process — an inflation of roughly 1.9×. So joint estimation does bias the within baseline, **upward**. That is the opposite direction to the anomaly, and it eliminates the ridge as an explanation while establishing it as a real effect to correct for.

### 7.3 The headline estimator null-calibrates on the real machinery

The pop-null arm returns **+0.038** [−0.054, 0.131] from null data, confirming on the actual pipeline what §6 showed for the two-step approximation: **the +0.203 headline is not manufactured by its estimator.**

### 7.4 The two baselines now agree, and say something

The apparent contradiction between +0.203 and −0.60 was entirely an artefact. Both baselines are positive, and their _difference_ is informative: the population baseline carries between-child and within-child association together, while the within baseline strips the persistent part.

|                                                         |                   estimate |
| ------------------------------------------------------- | -------------------------: |
| population baseline (real data, `rep`, model of record) |  **+0.203** [0.093, 0.316] |
| within baseline (real data, `test`)                     | **+0.103** [−0.085, 0.294] |

The within-child component is roughly half the mixed estimate before any correction, and smaller still after §7.2's inflation — and it does not exclude zero. **The association is predominantly between-child.** That is the reading [202608151120](202608151120-vg16-cross-lag-quantified.md) §6 reached on entirely separate grounds (the AR(1) rejection, and the attenuation implied by treating `x_lag` as a proxy for persistent standing). Two unrelated lines of evidence agree.

## 8. Consequences

1. **The VG16 report's short-T attribution is withdrawn and replaced.** The cause is `dev`-tier non-convergence, demonstrated, not a structural artefact of the estimator.
2. **The +0.203 headline stands, with its between-child reading strengthened** rather than merely unchallenged.
3. **A within-child estimate is feasible** — the original conclusion of this note is withdrawn — but it is small, uncertain, and needs the §7.2 inflation calibrated before it is quoted. It is not currently worth a registered model.
4. **§1's second obstacle still stands.** None of this makes the within-child lead an answer to the intervention question.
5. [#224](https://github.com/dseinternational/vocabulary-growth/issues/224) (VG20, correlated subject random effects) remains the better instrument for the between-child question, on all 767 children, and is unaffected.

## 9. Caveats, and corrections to this note

- **An earlier version of the simulator got the sign wrong.** Regressing `logit(y_s / y_u)` on the lag returned _positive_ bias at every truth including zero: 159 of 973 conditional rows (16%) have zero spoken words, and `logit(0)` clips to −9.21 — a large negative outlier arising for exactly the small-vocabulary children who also have a low lag value. Any estimator reducing the outcome to a ratio inherits this; the beta-binomial likelihood does not. Reported as-is it would have been confidently wrong in the opposite direction to the finding it was testing.
- **The 1.9× inflation is calibrated at one truth under one generating baseline** and should be treated as indicative, not applied as a correction factor.
- **The `test`-tier arms are diagnostics, not models of record.** Three of four miss the hard convergence gate, on parameters unrelated to the cross-lag; the real-data arm has zero divergences but min ESS exactly at the 400 floor. A publication-grade within-child estimate would need a `rep` fit.
- **Simulating from the model cannot detect misspecification**, so every "unbiased" claim is conditional on VG16's structure being right.
- **The wave census counts distinct ages with an understood measure** after the DS pool's default masking.
- §§3–4 are retained as written and superseded by §§6–7; the child count in §3 corrects a figure quoted in conversation.
