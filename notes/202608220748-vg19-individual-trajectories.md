# What VG19 says about individual trajectories

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Interpretation of the fitted VG19 `rep` trace of 2026-08-21, written 2026-08-22. **VG19 is not the model of record** and G3 (recovery) has not returned, so nothing here is publishable yet. Everything is a deterministic function of six fitted scalars propagated over posterior draws; reproduced by `scripts/experiments/vg19_individual_trajectories.py`. The headline: comprehension trajectories run close to parallel, production-ratio trajectories genuinely diverge, and that dissociation is the substantive content of the child slope.

## 1. Why the question could not be asked before

VG08-VG10 and VG20 give each child one number — a constant random intercept per outcome. Under those models every child's curve is the population curve shifted vertically, tracking is perfect by construction, and no child can change standing relative to peers. The empirical tracking analysis in [202608141600](202608141600-rank-stability-tracking.md) made exactly this point: the repeated measures can test tracking and the fitted models could not, because they assume it.

VG19 gives each child an intercept **and** a rate per outcome, drawn from a 2x2 covariance, so the child effect at age `a` is `b0 + b1 * D(a)` with `D(a) = (a - 36) / 12`. Everything below follows from

    Cov(a, b) = tau0^2 + rho01 * tau0 * tau1 * (D(a) + D(b)) + tau1^2 * D(a) * D(b)

evaluated over posterior draws. `P(swap)` is the bivariate-normal orthant probability `arccos(corr) / pi` for the difference between two independently drawn children. No refitting and no simulation is involved except in §4, where the quantity has no closed form.

## 2. Comprehension: children run close to parallel

`tau0_u` = 0.751, `tau1_u` = 0.176 [0.092, 0.242], `rho01_u` = −0.219 [−0.423, −0.008]. The rate spread is 23.5% of the level spread, and the between-child SD is nearly flat in age (0.851 at 18 months, a minimum of 0.734 around 48, 0.921 at 84).

Correlation between a child's own comprehension effect at one age and another:

|        |   18 |   24 |   36 |   48 |   60 |   72 |   84 |
| ------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **18** | 1.00 | 1.00 | 0.95 | 0.85 | 0.70 | 0.54 | 0.40 |
| **24** | 1.00 | 1.00 | 0.98 | 0.89 | 0.76 | 0.62 | 0.48 |
| **36** | 0.95 | 0.98 | 1.00 | 0.97 | 0.88 | 0.77 | 0.66 |
| **48** | 0.85 | 0.89 | 0.97 | 1.00 | 0.97 | 0.90 | 0.82 |
| **60** | 0.70 | 0.76 | 0.88 | 0.97 | 1.00 | 0.98 | 0.93 |
| **72** | 0.54 | 0.62 | 0.77 | 0.90 | 0.98 | 1.00 | 0.99 |
| **84** | 0.40 | 0.48 | 0.66 | 0.82 | 0.93 | 0.99 | 1.00 |

A child one SD above the mean at 24 months is expected at **+0.76 SD** at 60 months, and two randomly chosen children swap order between those ages **21.6%** of the time. Comprehension standing established early largely holds.

This is the one number in the note with an external check. [202608141600](202608141600-rank-stability-tracking.md) measured a tracking ICC of **0.786** for understood, from raw within-child residuals with study fixed effects, fitting no model at all. VG19's model-based correlation over a comparable span (0.76 at 24-60 months, 0.89 at 24-48) brackets it. Two methods with nothing in common but the data agree, which is the strongest evidence in this note.

## 3. Production ratio: trajectories genuinely diverge

`tau0_q` = 1.207, `tau1_q` = 0.640 [0.538, 0.743], `rho01_q` = +0.469 [0.338, 0.591]. The rate spread is 53.1% of the level spread and it compounds, so the between-child SD grows from a minimum of 1.069 at 25.3 months to 3.303 at 84 — a 2.6-fold widening.

|        |    18 |   24 |   36 |   48 |   60 |   72 |    84 |
| ------ | ----: | ---: | ---: | ---: | ---: | ---: | ----: |
| **18** |  1.00 | 0.96 | 0.67 | 0.36 | 0.17 | 0.06 | −0.02 |
| **24** |  0.96 | 1.00 | 0.85 | 0.61 | 0.44 | 0.33 |  0.26 |
| **36** |  0.67 | 0.85 | 1.00 | 0.94 | 0.85 | 0.78 |  0.73 |
| **48** |  0.36 | 0.61 | 0.94 | 1.00 | 0.98 | 0.95 |  0.92 |
| **60** |  0.17 | 0.44 | 0.85 | 0.98 | 1.00 | 0.99 |  0.98 |
| **72** |  0.06 | 0.33 | 0.78 | 0.95 | 0.99 | 1.00 |  1.00 |
| **84** | −0.02 | 0.26 | 0.73 | 0.92 | 0.98 | 1.00 |  1.00 |

A child one SD above the mean at 24 months is expected at only **+0.44 SD** at 60 — not because they fall back, but because the spread outgrows the persistence: the regression coefficient is 0.88 in logit units while the SD doubles from 1.07 to 2.13. The unpredictable residual at 60 months is **89%** of that age's entire between-child SD, and **35.4%** of pairs swap rank between 24 and 60 months.

On the scale a family or a teacher would recognise — the proportion of understood words a child says, holding the population trajectory at its median:

| age   | median child | 10th percentile | 90th percentile | between-child SD (logit) |
| ----- | -----------: | --------------: | --------------: | -----------------------: |
| 24 mo |         0.06 |            0.02 |            0.20 |                     1.07 |
| 36 mo |         0.18 |            0.04 |            0.50 |                     1.21 |
| 48 mo |         0.44 |            0.09 |            0.86 |                     1.61 |
| 60 mo |         0.68 |            0.12 |            0.97 |                     2.13 |
| 72 mo |         0.85 |            0.15 |            0.99 |                     2.70 |
| 84 mo |         0.94 |            0.18 |            1.00 |                     3.30 |

The middle 80% of children go from spanning 0.02-0.20 at two years to spanning 0.12-0.97 at five. VG20 says that band is the same width throughout.

## 4. Spoken words: the two models cross over, they do not merely differ

Spoken is `u * q`, so a child's spoken position depends on both random effects at once, and here the comparison with VG20 reverses with age rather than pointing one way. VG20 **correlates** the two effects (`rho_uq` = +0.368) but holds each constant in age; VG19 lets each **grow** with age but forces the correlation to zero. Simulated, because `log(expit(x) * expit(y))` has no closed-form variance — 400 posterior draws x 4000 synthetic children at each model's own population median curve:

| age   | VG19 sd(log p) | VG19 10th-90th pct (words) | VG20 sd(log p) | VG20 10th-90th pct (words) |
| ----- | -------------: | -------------------------: | -------------: | -------------------------: |
| 24 mo |           1.19 |                       1-27 |       **1.55** |                       1-35 |
| 36 mo |           1.09 |                      8-125 |       **1.32** |                      5-156 |
| 48 mo |           1.03 |                     24-311 |           1.01 |                     27-340 |
| 60 mo |       **1.04** |                     42-466 |           0.79 |                     69-471 |
| 72 mo |       **1.07** |                     58-560 |           0.65 |                    118-547 |
| 84 mo |       **1.10** |                     82-645 |           0.52 |                    183-613 |

**Below about 48 months VG20 gives the wider individual band; above it, VG19 does.** The compounding wins early and the growth wins late, and they happen to cross in the middle of the reported range. VG20's spread also _narrows_ with age on this scale (1.55 to 0.52) purely through ceiling compression — its logit-scale spread is constant and `p_s` approaches 1 — whereas VG19's stays near 1.0-1.1 throughout. This is the concrete form of the claim in §5 of [202608212000](202608212000-vg19-gates-g2-g4-g5.md) that neither model is a superset of the other: adopting either one alone would narrow the published individual band at one end of the age range.

## 5. The dissociation is the finding

Comprehension trajectories are near-parallel; **whether a child says what they understand** is where children diverge, and it diverges most across exactly the ages at which intervention decisions are made. Early comprehension standing is a reasonable guide to later comprehension standing. Early production-ratio standing is a poor guide to later production-ratio standing — 44% correlation from 24 to 60 months, effectively zero from 18 to 84 — and that cuts both ways, which is the more useful half of the result for anyone advising a family at age two.

Note that this is a statement about a _ratio_, not about speech in absolute terms: a child low on `q` at five is saying a small fraction of a comprehension vocabulary that has itself been growing. §4 is the version of the claim in words spoken.

## 6. What this rests on

1. **The child slope is forced linear in age.** Any real curvature in a child's departure from the population curve is read as rate variance, and the near-zero long-lag correlations in §3 partly follow from the functional form rather than from the data. A child effect that plateaus would produce the same fitted `tau1` and a very different correlation table.
2. **`q` is near its floor at 18-24 months (median 0.06) and near its ceiling by 84 (median 0.94).** Both ends of the §3 table are weakly informed, and the 90th-percentile figure of 1.00 at 84 months is a ceiling artefact, not a measurement.
3. **`tau_subj_u_1` is the worst-sampled parameter in the fit** (ESS 862, R-hat 1.008, against a minimum of 2252 elsewhere). §2's conclusions are the least well estimated in the note, though they are also the ones with the external check.
4. **VG20, the model of record, contradicts this picture** — flat between-child spread on both outcomes — and §4 shows the disagreement is not uniformly in VG19's favour.
5. **G3 has not returned.** These are exactly the parameters recovery is meant to validate, and the `test` tier left all three replicates unassessable with `tau_subj_u_1` the sole failing parameter. For a claim this specific that is the gate that matters most, and until it passes the correlation tables are a description of a posterior rather than a finding about children.
6. **These are model-based random effects, not observed trajectories.** Shrinkage means the spread of fitted per-child effects is narrower than the spread implied here, which is the population distribution the effects are drawn from.
