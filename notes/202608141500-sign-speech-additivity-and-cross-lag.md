# Is signing additive? Between-child association and a sign→speech cross-lag

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Investigation record, 14 August 2026. Descriptive analysis only — no model, prior or data changes, and no refit. Every number is reproduced by [`scripts/sign_speech_association_audit.py`](../scripts/sign_speech_association_audit.py). Nothing here is causal: no source in the pool records whether its children were taught to sign, which is the same limitation that forced the withdrawal in [202608121030](202608121030-psi-heterogeneity-and-age-invariance.md).

## 1. The question, and why $\psi$ does not answer it

Asked on 2026-08-14: among children who sign, does more signing mean more total (signed + spoken) production?

The plain-language account of $\psi$ is [202608141141](202608141141-psi-plain-language-overview.md); this note asks the adjacent question it does not cover. VG15's $\psi$ does not answer this. It is a _within_-child, item-level overlap parameter — given a child's understood words, are the ones they sign also the ones they say — carried with a study-level term. The question above is _between_ children, and no fitted model in the family estimates it: VG15's subject random effects on understood, $q$ and signed are independent scales with no correlation structure, and VG16's cross-lag runs understood → $q$, not sign → speech.

Three sources partition the words a child **understands** (`uk_02`, `uk_07`, `es_01`; 323 rows, 243 children). `nz_01` partitions only what the child **produces**, so it is never pooled with them. Of these, `uk_02` (28 children × 2 waves), `uk_07` (30 children, 25 with 3 waves) and `nz_01` (33 children, ~3.4 waves) are longitudinal — 83 children with repeated cross-tabs, which is more longitudinal signing data than the earlier scoping assumed.

## 2. How much does counting signs add at all?

Share of produced vocabulary that is signed but **not** spoken — what a speech-only count misses. Three denominators give three different numbers, and none of them is "the" share:

| source  | pooled (word-weighted) | median child (first wave) | median observation | observation IQR |
| ------- | ---------------------: | ------------------------: | -----------------: | --------------: |
| `uk_02` |                  36.1% |                     74.7% |              72.4% |         16%–86% |
| `nz_01` |                  23.5% |                     54.1% |              18.3% |          2%–67% |
| `es_01` |                  21.2% |                     57.1% |              57.1% |         17%–77% |
| `uk_07` |                  13.5% |                     27.6% |               4.3% |          0%–54% |

**No single figure should be quoted as "signing adds X%".** The pooled column weights children by vocabulary size, so `uk_07`'s older children carry most of the denominator at 13.5% while `uk_02`'s younger children sit at 74.7% per child. The observation median counts a child once per wave, so for the two most heavily repeated sources it is pulled 3- to 6-fold toward those children's later, lower-signing waves — `nz_01` 54.1% per child against 18.3% per observation, `uk_07` 27.6% against 4.3%. `es_01` is identical across the two because it is one row per child. The IQRs (0%–86% across the pool) say the same thing from the other side: this is not a stable per-child quantity. **The gradient is the result, not any level.**

### Signs fully absorbed into speech

The cleanest view needs no denominator choice and no modelling at all — the share of observations where the child signs nothing they do not also say, ordered by how much speech the sample has:

| source  | median age | median spoken | observations fully duplicated | among observations with any signs |
| ------- | ---------: | ------------: | ----------------------------: | --------------------------------: |
| `es_01` |         32 |            26 |                          3.2% |                              3.2% |
| `uk_02` |         38 |            40 |                          1.8% |                              1.8% |
| `nz_01` |         48 |            90 |                         16.2% |                             13.1% |
| `uk_07` |         60 |           213 |                         28.0% |                             26.2% |

From ~2% to 28%: in the oldest sample more than a quarter of observations show signing adding no vocabulary whatever. But this is a contrast between four studies differing in age, country, instrument and recruitment, so it confounds development with everything else. §6 tests the same thing within children, where it survives.

## 3. Between-child: signing more does not go with speaking more

Standardised slope of spoken vocabulary on signing, controlling comprehension and age, cluster-robust on child. Two specifications bracket the mechanical bias, because neither is clean on its own — `sign_only` is disjoint from `spoken` but competes with it inside a fixed comprehension budget (biased **negative**), while `signed` shares the `both` cell with `spoken` (biased **positive**):

| source            | `sign_only` (lower bracket) |      z | `signed` (upper bracket) |     z |
| ----------------- | --------------------------: | -----: | -----------------------: | ----: |
| `uk_02`           |              −0.558 (0.051) | −11.02 |           −0.225 (0.156) | −1.44 |
| `uk_07`           |              −0.482 (0.060) |  −8.00 |           −0.016 (0.130) | −0.12 |
| `es_01`           |              −0.394 (0.042) |  −9.45 |           −0.150 (0.092) | −1.63 |
| pooled (study FE) |              −0.440 (0.031) | −14.13 |           −0.051 (0.076) | −0.67 |

The two specifications carry opposing _dominant_ biases and both land negative, putting the association in the region **[−0.44, −0.05]**. Read that as a bracket, not a strict bound: the `signed` specification carries the compositional bias too, so its positive shared-cell inflation is only the larger of two opposing terms, not the only one. What survives either way is the direction — at matched comprehension and age, children who sign more do not speak more, and if anything speak slightly less, in all three sources.

One qualification that weakens it. `es_01` is the only source carrying a developmental measure external to the cells, and substituting mental age for comprehension as the control gives −0.251 (z = −4.78) and +0.043 (z = 0.48) — a bracket of **[−0.25, +0.04]**, which includes zero. So the negative finding is partly a property of conditioning on comprehension specifically.

## 4. The total: additive at low comprehension, not at high

Total production is an identity, `produced = spoken + sign_only`, so the slope of the total on `sign_only` is exactly one plus the slope of spoken on `sign_only`, and the whole question is where it lies between 1.0 (purely additive — every sign is a word speech would have missed) and 0.0 (fully displaced — the total does not move).

**The pooled answer is −0.66 [−0.89, −0.43], and it is the wrong summary.** A model-free check — one row per child, comprehension quartiles, then a within-quartile split on sign-only vocabulary — shows the total is _higher_ for high signers in the middle bands and collapses only in the top one. A single linear slope averages a sign change. Refitting within band:

| comprehension band | understood | spoken per sign-only word | **total** per sign-only word |        95% CI |
| ------------------ | ---------: | ------------------------: | ---------------------------: | ------------: |
| q1                 |      7–122 |                     −0.20 |                    **+0.80** |  [0.39, 1.22] |
| q2                 |    128–275 |                     −0.34 |                    **+0.66** |  [0.37, 0.96] |
| q3                 |    276–401 |                     −0.63 |                    **+0.37** | [−0.01, 0.74] |
| q4                 |    403–651 |                     −1.53 |                    **−0.53** | [−1.36, 0.30] |

So signing is close to **purely additive early** and progressively less so as comprehension grows, reaching zero-to-negative at the top of the range. The low-comprehension end is the conservative one: the compositional constraint biases this slope downward, so a measured +0.80 is a floor on how additive signing is there, not a ceiling. That is developmentally coherent: early signs are words the child would otherwise not produce at all, while at high comprehension a large sign-only vocabulary marks a child whose speech is lagging.

The gradient is not `es_01`'s alone, which matters because `es_01` is 185 of the 243 children and the one source whose within-child association is ≈1 where the others are 6–15. Splitting each source at its own comprehension median:

| source  |          lower half |           upper half |
| ------- | ------------------: | -------------------: |
| `es_01` |  +0.63 [0.05, 1.21] | −0.95 [−1.54, −0.37] |
| `uk_02` | +0.43 [−0.62, 1.49] |  −0.11 [−0.55, 0.32] |
| `uk_07` | −0.06 [−0.86, 0.74] | −1.13 [−2.23, −0.03] |

All three decline in the same direction. Only `es_01` has the sample size to establish either end on its own (14–15 children per half elsewhere).

## 5. Cross-lag: no evidence signing precedes speech, and the design cannot rule it out

Consecutive within-child wave pairs, rates over understood: 80 pairs from 55 children (`uk_02` 28 pairs, `uk_07` 52 pairs), median 10 months between waves. Both directions fitted, because a result appearing in both is shared growth rather than a lead.

| direction                     |         β (SE) |     z |
| ----------------------------- | -------------: | ----: |
| sign-only(t) → gain in spoken | −0.097 (0.157) | −0.62 |
| spoken(t) → gain in signed    | −0.418 (0.115) | −3.63 |

`nz_01`, on a produced-vocabulary denominator: forward +0.040 (0.079), z = 0.50; reverse −1.088 (0.267), z = −4.08. Its reverse figure is partly arithmetic, since spoken and sign-only shares of production are compositional.

Two readings, and only the second is safe:

- **Forward is null but uninformative.** With 55 clustered children the standard-error floor on a standardised slope is ≈0.135, so the smallest reliably detectable effect is ≈0.38. Anything short of a large effect would be invisible. This is a non-result, not evidence of absence.
- **Reverse is the consistent finding**: children with more speech at one wave subsequently gain less signing, in both source groups. That fits signing as a transitional channel that recedes as speech arrives — though regression to the mean is not fully excluded.

## 6. Within child: signing is absorbed as speech grows

The within-child test in this section was proposed by the study owner, against the cross-study gradient in §2. Everything above compares children, so it confounds development with everything else that differs between four studies and between families. The 83 children measured more than once (`uk_02` 28, `uk_07` 27, `nz_01` 28 — 8 children seen once contribute nothing and are dropped) let the child be their own control, which removes every time-invariant confound at once — severity, family, study, instrument, and whether the child was taught to sign.

**C1. First wave against last.** The sign-only share of production falls for 66 children, rises for 10 and ties for 7 (sign test p = 3.0 × 10⁻¹¹), in every source separately:

| source  | children | share, first | share, last | spoken, first | spoken, last | fell | rose | tied |      p |
| ------- | -------: | -----------: | ----------: | ------------: | -----------: | ---: | ---: | ---: | -----: |
| `uk_02` |       28 |        74.7% |       58.0% |            24 |           78 |   25 |    2 |    1 | 0.0000 |
| `nz_01` |       28 |        57.0% |        4.3% |            28 |          202 |   23 |    3 |    2 | 0.0001 |
| `uk_07` |       27 |        27.0% |        0.7% |           111 |          272 |   18 |    5 |    4 | 0.0106 |

**C2. The same thing as a slope, with child fixed effects** (SEs still clustered, since waves are not independent):

| source  | children | d(sign-only)/d(spoken) |    SE |     z | implied d(total)/d(spoken) |
| ------- | -------: | ---------------------: | ----: | ----: | -------------------------: |
| `uk_02` |       28 |                 −0.389 | 0.069 | −5.61 |                       0.61 |
| `nz_01` |       28 |                 −0.275 | 0.075 | −3.67 |                       0.73 |
| `uk_07` |       27 |                 −0.163 | 0.054 | −3.02 |                       0.84 |

As a child gains a spoken word, their sign-only vocabulary falls by 0.16–0.39 words, so total production gains **0.6–0.8 words rather than one**. Signs are progressively absorbed, and the total still rises throughout.

**C3. Transitions.** 14 children moved into signing nothing they do not also say; 1 moved the other way.

This is the within-child test §4's cross-study gradient could only hint at, and it survives. It is also the strongest result in the investigation — but it remains directional, not causal: signing receding as speech arrives is what a transitional channel looks like, and equally what discontinuing an intervention looks like once it is judged no longer needed.

## 7. What binds all of it

**The dominant confound is indication.** If signing is adopted _because_ a child's speech is hard, every result above follows with no causal role for signing whatever: the negative between-child slope at high comprehension, the additivity at low comprehension, and the receding of signing as speech arrives. Nothing here distinguishes that from an effect of signing, and it is the more parsimonious reading. `uk_07` is an RCT, but its arm contrast points the wrong way for a signing-programme mechanism (control 17.93 against intervention 11.65, with the gap present at t1).

Everything else is secondary: these are observational cross-tabs on parent report; the between-child section is cross-sectional; the analysis conditions on comprehension, which §3 shows is doing real work; and `es_01` dominates the pooled sample.

## 8. Outstanding

1. **The §3 conditioning question is answerable and cheap.** `uk_07`'s UK Data Service deposit holds Mullen age equivalents at all three assessment points (`T{1,2,3}MullenCombinedAE`), already flagged as unextracted in [202608121030](202608121030-psi-heterogeneity-and-age-invariance.md) §5. Extracting them would let §3 and §4 be refitted on a developmental control external to the cells in a second source, not just `es_01`.
2. **Whether §4's gradient warrants a model change.** A level-varying $\psi$ with study-specific slopes was already left open by the $\psi$ note pending exactly that covariate. §4 is the between-child analogue and points the same way. Neither is a case for a graph change yet.
3. **The `ds_td_expressive_delay_by_age` panel plots without a coverage filter** (`_band` defaults `cov=0.0` where the level-indexed panels pass 0.80), so its right-hand third is drawn from a vanishing fraction of draws — at 40 months the Δ_exp interval is degenerate (lower = median = upper) on a coverage of 2.8 × 10⁻⁵, which is one draw. Unrelated to this investigation, found alongside it.
