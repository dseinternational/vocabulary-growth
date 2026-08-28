# The sign–speech association: between-study heterogeneity, and why it is not age

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Investigation record, 12 August 2026. Every number here is reproduced by [`scripts/psi_heterogeneity_audit.py`](../scripts/psi_heterogeneity_audit.py), committed so it can be re-run after any later data or loader change. Two decisions came out of it, both implemented: $\psi$ gains a **study-level term**, and it gains **no age term**.

## 1. Where this started

Integrating `uk_07` ([202608120030](202608120030-uk07-pactds-integration-and-ds-refit.md)) moved $\psi$ from 1.797 to 2.495. A headline parameter shifting by 39% on one new source is a fragility worth understanding, and the obvious next question — could `es_01` also supply cross-tab cells? — turned out to make it unavoidable.

## 2. es_01 does supply them, and the source says so

The Galeote table's four columns are labelled **TOTAL COMPREHENSIÓN**, **TOTAL PRODUCTION**, **TOTAL GESTURES**, **WORD PRODUCED + GESTURES ONLY**, the last being what Galeote et al. (2011) describe as "total lexical production combining the two modalities (oral + gestural production)". So the third column is a _total_ and the fourth a de-duplicated union, and the four cells follow by subtraction:

```
understood_only = understood − union
spoken_only     = union      − gestured
gestured_only   = union      − spoken
both            = spoken + gestured − union
```

**185 of 186 rows** yield a valid partition summing to `understood` exactly, spanning 11–71 months. The exception is the known defective row (`pair_id` 148: 1 spoken, 15 gestured, union 11), whose `spoken_only` is −4.

Two independent checks, because a mislabelled column would be expensive here:

- **The disjoint reading is impossible.** If `spoken` and `gestured` were exclusive cells — the `nz_01`/`uk_07` convention — then `union = spoken + gestured` would hold identically. It holds on 52 of 186 DS rows; the other 134 have a union strictly smaller, by a median of 7 words and up to 212. The 52 are not a rival explanation: 17 have zero spoken words (overlap arithmetically forced to zero), and their median spoken vocabulary is 2 words against 58 among the rest.
- **The intersection parse is impossible.** "WORD PRODUCED + GESTURES ONLY" could be misread as "produced by word _and_ gesture only". Column 4 exceeds `min(spoken, gestured)` on all 186 rows.

### A framing that was wrong

An earlier draft of this work treated es_01's gestures as a _different construct_ from taught signs, and set `include_es01_cells = False` on that basis. That was corrected by the study owner: the CDI-Down's third column scores "gestures representing specific lexical items", each tied to one of the 651 checklist words — a per-word lexical marker on an adapted CDI, structurally the same coding `uk_02`, `uk_07` and `nz_01` apply to signs. The terminology differs; the measurement does not. The heterogeneity below is therefore a real difference between samples, not an artefact of what was being counted.

> **Correction (2026-08-28, Claude Code/Fable 5, re [202608271551](202608271551-es01-gesture-construct.md)):** this subsection over-corrected. The earlier draft was right about the coding and wrong about the construct, and this note then asserted the converse. The CDI-Down's third column does carry the same per-word lexical coding as the sign sources, but what it scores is the child's **symbolic-gesture repertoire** — gestures "properly taught or spontaneously learnt", undifferentiated — a related but broader construct than the taught signing the UK and New Zealand sources record, which the TD group's own totals prove: 23.6 gestured words on average with no sign instruction reported and none plausible at scale. The final sentence above therefore no longer stands in full: the es_01-versus-sign-sources gap is plausibly first-order a difference in **what was counted**, with the residual spread among the three sign sources (6–15) the part that is about samples and context, and the two cannot be apportioned from these data. Everything else here stands — §4's rejection of an age term, the study-level structure of §6 (which is precisely what makes pooling a gesture measure with sign measures defensible; the verification is [202608281147](202608281147-study-term-pooling-licence.md)), and §8's warning against a teaching interpretation of the residual spread.

## 3. The sources disagree by an order of magnitude

Mantel-Haenszel odds ratios over the same four cells, each child its own stratum:

| source  | rows | MH odds ratio | reference set     | per-child OR < 1 | non-vocal words also spoken |
| ------- | ---- | ------------- | ----------------- | ---------------- | --------------------------- |
| `uk_02` | 56   | 6.09          | within understood | 4%               | 50.4%                       |
| `uk_07` | 82   | 13.90         | within understood | 11%              | 72.2%                       |
| `nz_01` | 111  | 14.63         | all 675 items     | 4%               | 44.8%                       |
| `es_01` | 185  | **0.90**      | within understood | **45%**          | 30.8%                       |

Two caveats travel with that table and are printed by the audit script. Mantel-Haenszel is a descriptive statistic on observed cells, **not** $\psi$, which is population-conditioned against the model's own $r$ and $q$. And `nz_01` has no comprehension total, so its "neither" spans all unproduced items — the same `uk_07` rows read **13.90** within understood and **40.72** over all 674 items. Magnitudes compare only within a reference set; the per-child sign and the share-also-spoken column need no "neither" cell and compare throughout.

The gap survives every control available: by age band `es_01` runs 0.30–1.12 against 4.4–41.6 for `uk_02` and 4.4–18.1 for `uk_07`, with no overlap in any band; matched on expressive vocabulary (30–300 words) it is 1.05 against 4.80 and 9.68.

### The cleanest internal control

`es_01` carries its own mental-age-matched typically developing group **on the identical instrument**: DS **0.90**, TD **2.13**. Same country, same form, same question — positive association in TD children, independence in DS children. The instrument can plainly detect coincidence, so the DS result is about how the non-vocal lexicon functions, not about the wording.

By the study's own matching stratum, both groups start in strong substitution and diverge:

| MA level | DS   | TD   |
| -------- | ---- | ---- |
| 1        | 0.08 | 0.05 |
| 2        | 0.17 | 0.87 |
| 3        | 1.01 | 0.99 |
| 4        | 1.26 | 2.32 |
| 5        | 1.04 | 2.63 |
| 6        | 0.88 | 3.53 |
| 7        | 0.80 | 1.48 |

TD climbs to 2.6–3.5; DS plateaus near 1 and never converges. (Sparse 2×2 tables bias odds ratios toward 0, so the lowest levels overstate the floor.) One reading is that TD children gesture words they are also learning to say, while DS children — who have a specific expressive-speech deficit relative to comprehension — gesture what they cannot yet say; another is that where signing is taught as sign-and-say pairs both modalities land on the same words by construction. **Neither is evidenced here.** No source in this pool records whether its children were taught to sign, so the teaching contrast is background assumption rather than measurement, and the pooled es_01 figure (0.90, with 45% of children individually below 1) is independence rather than the substitution the first reading implies. What is measured is that es_01's DS and TD groups differ on the identical instrument — which excludes instrument, language and country, but does not identify what differs.

### It is not uk_07's intervention

`uk_07` is an RCT, so the obvious alternative is that its programme creates the association. It does not:

| arm          | t1    | t2    | t3    | pooled |
| ------------ | ----- | ----- | ----- | ------ |
| control      | 23.96 | 19.03 | 13.68 | 17.93  |
| intervention | 8.41  | 13.35 | 16.04 | 11.65  |

The **control** arm is higher, and the gap is present at t1 before any intervention — a baseline imbalance at 15 per arm, not an effect. Both arms sit far above `es_01` regardless, so the association is a property of uk_07's context rather than its trial. (Weakly suggestive and not leaned on: the intervention arm rises across the three assessment points while control falls, which is equally consistent with regression to the mean.)

## 4. Age: investigated, and rejected

### The first analysis was under-specified

The initial test regressed per-child log odds ratio on age **with study fixed effects**, and found nothing: +0.009 per year (SE 0.062, z = +0.14), with no individual source significant either (`uk_02` z = −0.30, `uk_07` z = +1.61, `es_01` z = −0.85).

That result was correct but did not answer the question asked. Study fixed effects absorb between-study age differences **by construction**, so it could only ever speak to _within_-study age variation. The study owner caught this: `uk_07`'s children are markedly older than the rest, so its high association could be an age effect that the study dummy was silently swallowing.

| source  | median age of ψ-informing rows | IQR   |
| ------- | ------------------------------ | ----- |
| `es_01` | 32 mo                          | 22–44 |
| `uk_02` | 38 mo                          | 34–46 |
| `uk_07` | **60 mo**                      | 51–73 |

The concern is well-founded on its face. Fitted **without** a study term, age looks like a strong driver: **+0.381 per year (SE 0.066, z = +5.78)**, i.e. $\psi \times 1.46$ per year.

### Two tests settle it

**Model comparison** (weighted SSR, lower is better):

| model                    | weighted SSR |
| ------------------------ | ------------ |
| age only (no study term) | 2688         |
| study only               | **1640**     |
| study + age              | **1640**     |

Study alone fits far better than age alone, and adding age on top of study improves the fit by _nothing_. Once the study is known, age is uninformative; once age is known, study still carries an order of magnitude.

**The age-matched contrast** — restricting to the window where all three within-understood sources overlap:

| 34–56 months | MH $\psi$ | rows | median age |
| ------------ | --------- | ---- | ---------- |
| `es_01`      | 1.00      | 61   | 41         |
| `uk_02`      | 5.94      | 41   | 41         |
| `uk_07`      | 11.89     | 34   | 47         |

`es_01` and `uk_02` are matched at exactly 41 months and still differ six-fold. The 30–60 window gives 1.03 / 5.86 / 10.96. The apparent age effect is entirely `uk_07` being simultaneously the oldest sample and the highest-association one.

### Coverage would not have supported it anyway

Three-way overlap exists only at 30–60 months. Below 20 months is `es_01` alone; above 75 is `uk_07` alone. Any curvature an age term drew in those tails would be that single study's intercept re-labelled as a trend.

**Decision: no age term on $\psi$.** The study-level term is the correct structure and age adds nothing to it.

## 5. The level gradient, and why half of it is unreadable

Against expressive vocabulary the study-specific slopes are large, individually significant, and **opposite in sign** — which is why the pooled study-adjusted slope is ~0 (−0.030, z = −0.36):

| scope   | slope on log produced vocabulary | z     |
| ------- | -------------------------------- | ----- |
| `uk_02` | +0.555                           | +2.92 |
| `uk_07` | +0.731                           | +3.13 |
| `es_01` | −0.348                           | −3.63 |

**But produced vocabulary is a circular covariate**: `produced = sign_only + speak_only + both`, three of the four cells that define $\psi$. Raising it at fixed comprehension shrinks `neither` and mechanically lowers the ratio, so the induced bias is **negative**. The two positive slopes run against that bias and are conservative; `es_01`'s negative slope runs with it and is not established.

`es_01` is the only source carrying a developmental measure external to the cells, and against it the gradient vanishes:

| es_01 group | slope on mental age (per 6 mo) | z     |
| ----------- | ------------------------------ | ----- |
| DS          | −0.096                         | −0.80 |
| TD          | −0.109                         | −0.94 |

So `es_01`'s apparent fall was the circularity. The high-association sources cannot be tested this way — neither `uk_02` nor `uk_07` carries an external developmental measure in its CSV.

**This is the open question, and it is cheap to close.** `uk_07`'s UK Data Service deposit holds Mullen Scales visual-reception and fine-motor age equivalents at all three assessment points (`T{1,2,3}MullenCombinedAE`); they were simply not extracted, since only the CDI, age, sex and arm were carried across. Extracting them and re-running §5 would establish whether uk_07's +0.731 survives a covariate external to the cells. If it does, a level-varying $\psi$ with **study-specific slopes** is worth building — not age, and not a shared trend. If it does not, the study-varying scalar is the right stopping point.

## 6. What was implemented

`delta_psi` is a zero-sum study random intercept over the $\psi$-informed studies, following `delta_sign`'s construction: studies with no cross-tab never enter a $\psi$ term and are pinned to 0 rather than allowed to counterbalance the informed ones. Both Dirichlet-Multinomial likelihoods take the per-study value; the reported composition and `p_any` keep the population one, matching `p_u_plot` / `q_plot` / `r_plot`. `tau_psi ~ HalfNormal(1.0)`, wider than the trajectory scales (0.5) because the measured spread is about 2.8 on the log scale.

With the heterogeneity modelled rather than averaged away, `include_es01_cells` defaults **True**. $\psi$ is now identified by four sources and 434 rows (56 `uk_02` + 82 `uk_07` + 185 `es_01` four-cell, plus 111 `nz_01` produced-cell), against 138 rows and three sources before.

Dev-config fit (provisional — `dev` never converges the GP parameters; 0 divergences, healthy BFMI, and `tau_psi`/`z_psi` absent from both convergence failure lists):

| study      | $\psi$ median | 90% HDI      | P($\psi$>1) |
| ---------- | ------------- | ------------ | ----------- |
| `es_01`    | 1.08          | [0.87, 1.35] | 0.71        |
| `uk_02`    | 2.31          | [1.56, 3.18] | 1.00        |
| `nz_01`    | 3.38          | [1.70, 5.38] | 1.00        |
| `uk_07`    | 3.67          | [2.56, 4.71] | 1.00        |
| population | 2.36          | [1.90, 2.84] | —           |

`tau_psi` 0.654 [0.233, 1.176] — bounded away from zero, so the spread is estimated rather than assumed. The per-study ordering reproduces the crude Mantel-Haenszel diagnostic exactly, which is independent confirmation the model is picking up the measured pattern. And the population value did **not** collapse toward independence despite `es_01` being the largest single source, which is the fix working.

## 7. How to read $\psi$ now

With four groups `tau_psi` is weakly identified and the prior does real work. **The per-study values are the primary read; the population $\psi$ is a shrunk centre, not a consensus.** Reporting it as a single number without the spread would repeat, in a subtler form, the problem this change was made to fix.

## 8. Outstanding

- [ ] Extract `uk_07`'s Mullen age equivalents upstream and re-run §5 (the deciding evidence for a level-varying $\psi$).
- [ ] Refit VG15 at `rep` — the graph changed; no other model reads this engine.
- [ ] `docs/models/vg15/index.qmd` still says $\psi$ is identified by "~60 uk_02 rows" and calls it a single scalar. Both are now wrong.
- [ ] Decide how the between-source spread is described in the report's discussion. The tempting reading — that $\psi$ marks whether the non-vocal lexicon was _taught alongside_ speech or arose _as compensation for its absence_ — is **not supported by anything measured here** and should not be stated as a finding. No source records its children's signing instruction, so the teaching contrast rests on assumed background practice; and `uk_07`, the one source with experimental variation in instruction, has the **higher** association in its _control_ arm (17.93 against 11.65, §3), which cuts against the mechanism rather than for it. If the reading appears at all it must be labelled a hypothesis, with the arm contrast disclosed beside it. Closing §5's Mullen extraction is the cheapest route to evidence that would bear on it.
