# Challenging item exchangeability

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Working note, 2026-07-26. The existence and direction of the finding are established; several magnitudes are provisional and are flagged as such. Nothing here has been actioned in the models. No refit is proposed.

> [!IMPORTANT]
> Revised the same day, before merge, after an independent verification pass that recomputed every checkable number from the raw CSVs and the fitted output. The §3B dispersion table is corrected (an age-standardisation error, recorded in §8 — the correction _strengthens_ the conclusion), §3B now reads `tau_subj_u` alongside `kappa`, and the IRT-feasibility conclusion in §7 is superseded by the companion note [`202607261048-incorporating-item-difficulty.md`](202607261048-incorporating-item-difficulty.md).

## Summary

Every model in the family assumes that, conditional on a child's latent ability, the 810 words of the reference inventory are equally likely to be known. That assumption is false, decisively and in a consistent direction. But three quite different concerns have been travelling under the label "exchangeability", and separating them is most of the value: one is negligible, one costs a definition rather than a number, and the most consequential of the three is **not an exchangeability problem at all** — it is that the Beta-Binomial concentration on a bounded proportion scale cannot be read as latent between-child heterogeneity, which would be true even if every word were equally hard.

Nothing in this note requires a reported mean trajectory or expected word count to change. What it requires is that three passages in the technical report be rewritten, and that one interpretive claim be retired.

## 1. Where the assumption lives

Every likelihood is `y ~ BetaBinomial(810, alpha = p * kappa, beta = (1 - p) * kappa)`. Its generative story has two steps: first draw a child-specific probability `pi ~ Beta(p * kappa, (1 - p) * kappa)`, then draw `y ~ Binomial(810, pi)`.

The first step is the one the report describes: children differ, and `kappa` governs how much. The second step is the one nothing in the documentation mentions. It says that _given_ a child's `pi`, each of the 810 words is an independent Bernoulli trial with the **same** `pi` — conditional on ability, every word is equally likely to be known, and the word list could be permuted without changing anything. That is item exchangeability, and it is logically independent of the between-child homogeneity the Beta step relaxes.

[`methods-models.qmd`](../docs/report/methods-models.qmd) at `§sec-betabinomial` (line 26) conflates the two: it says a Binomial "would assume exchangeable Bernoulli trials within each observation **and** no unmodelled heterogeneity between children", then presents the Beta-Binomial as the remedy. The Beta-Binomial remedies only the second. The first survives untouched into every model of record.

## 2. The assumption is violated, on a test that fits no model

Under item exchangeability, a child who knows `T` of the 810 words has a stratum-`k` count distributed `Hypergeometric(810, n_k, T)`. This is pure combinatorics from the known checklist sizes — no model, no fitted parameter, no distributional assumption beyond the one under test. Standardising ie_01's observed follow-up-wave checklist counts against it (n = 44 children with `0 < T < 810`):

| statistic                                    | observed | exchangeability predicts |
| -------------------------------------------- | -------- | ------------------------ |
| RMS z, all strata                            | 9.61     | 1.0                      |
| mean z, Checklist 1                          | +8.40    | 0                        |
| mean z, Checklist 2                          | +3.00    | 0                        |
| mean z, Checklist 3                          | −9.01    | 0                        |
| children with a positive Checklist 1 excess  | 37 / 44  | ~22 / 44                 |
| children with a negative Checklist 3 deficit | 42 / 44  | ~22 / 44                 |
| children with `p_C1 >= p_C2 >= p_C3`         | 79.5%    | 19.2%                    |

The sign tests give p = 5.3e-06 (Checklist 1) and p = 1.1e-10 (Checklist 3). For the monotonicity rate, none of 2000 simulations drawn under exchangeability reached the observed 79.5% (simulated maximum 40.9%). Pooled proportion understood, over the 46 complete follow-up records, is 0.671 / 0.452 / 0.253 across Checklists 1 / 2 / 3, where exchangeability requires all three to equal the overall 0.398 (per-child means are slightly higher, 0.701 / 0.473 / 0.264 over the 44 tested).

Worth stating plainly: the DSE checklists were **constructed** as a difficulty ladder, so finding `C1 > C2 > C3` is partly instrument validation. The corollary is the point — the instrument's own design contradicts within-observation exchangeability, so the assumption was never plausible for this inventory.

## 3. Three channels, which differ by two orders of magnitude

### 3A. The count-variance channel — negligible, and the intuition is backwards

For independent Bernoulli trials there is an exact identity, `sum_i p_i (1 - p_i) = N * pbar * (1 - pbar) - N * Var_items(p)`, verified here to machine precision. Item-difficulty heterogeneity therefore **reduces** within-child count variance; the natural intuition that "items differ, so there is extra noise" is the wrong way round. At the measured difficulty spread (~1.8 logits between the outer checklists) the reduction reaches 9% of within-child variance.

That does not propagate, because the between-child term carries `N^2 = 810^2`. Within-child variance is only 0.6%–4.6% of total count variance across the plausible range of ability spreads and levels, so the effect on a recovered `kappa` is about 1%. Confirmed by simulating through the project's own likelihood: `kappa` moves by under 1.1%.

This channel can be set aside.

### 3B. The level-scale channel — the consequential one, and not about exchangeability

Children's abilities live on a latent logit scale, on which the proportion scale compresses near 0 and 1 and stretches near 0.5. With a **constant** latent standard deviation `sigma`, the delta method gives `Var(p) ~= [p (1 - p)]^2 * sigma^2`, while the Beta-Binomial encodes between-child variance as `p (1 - p) / (kappa + 1)`. Equating the two:

```text
kappa + 1  ~=  1 / [ p (1 - p) * sigma^2 ]
```

So `kappa` **must** fall as `p` rises toward 0.5, with no change whatever in latent spread — a 6.7-fold decline from `p = 0.05` to `p = 0.5` at `sigma = 1` by this first-order formula (exact integration of a logit-normal ability distribution gives 16.3 → 4.8, still a 3.4-fold decline) — and must rise again above 0.5. The relationship is U-shaped, minimised at `p = 0.5`. The fitted form `kappa(z) = kappa_min + exp(a_kappa - b_mag * z)` with `b_mag > 0` imposed is monotone, so it cannot represent that shape; it fits a one-way trend through it.

This effect is present under _perfect_ exchangeability. It is a bounded-inventory scale-and-link property, surfaced by the exchangeability investigation rather than caused by the thing under investigation. Item heterogeneity's own contribution to the `kappa` trend runs the other way and is small: at matched level, heterogeneity inflates `kappa` by a fairly uniform 10%–13% and _flattens_ the level-driven decline by about 2.5%–3% (exact integration at `sigma = 1`; the companion note's §1 table is the same computation).

**What this means for the report.** `§sec-kappa` (line 47) is careful to call the monotone decline an **assumption** rather than an inference, and that framing is correct and should be kept. The problem is the parenthetical defending it, which states that because the outcome is bounded the decline "allows greater heterogeneity around the mean trajectory on the latent scale". That specific reassurance is backwards: `kappa` is not a latent-scale heterogeneity parameter, and constant latent heterogeneity already implies a declining `kappa`. Two facts from VG10's own output sharpen this. First, VG10 already carries an explicit between-child latent spread — the subject random intercept scale `tau_subj_u`, posterior mean 0.754, constant by construction — _underneath_ the Beta step, so `kappa` there is residual, occasion-level dispersion, which is one more reason it cannot be read as "children fanning out". Second, reading the model's own fitted dispersion (`posterior_kappa_u.csv` medians; the parameter posteriors are `a_kappa_u = 2.837`, `b_kappa_mag_u = 0.557`, `kappa_min_u = 2.996`) over the ages where DS comprehension data exist:

| age (months) | fitted `p` | fitted `kappa` | implied residual latent SD | total, with `tau_subj_u` |
| ------------ | ---------- | -------------- | -------------------------- | ------------------------ |
| 12           | 0.028      | 39.77          | 0.956                      | 1.218                    |
| 24           | 0.146      | 29.64          | 0.512                      | 0.911                    |
| 30           | 0.227      | 25.66          | 0.462                      | 0.884                    |
| 48           | 0.379      | 16.97          | 0.486                      | 0.897                    |
| 66           | 0.526      | 11.68          | 0.562                      | 0.941                    |

The fitted `kappa` declines by 1.225 log units over 12–66 months where a constant latent spread alone predicts 2.230 — a ratio of 0.55. The fitted parameters are therefore consistent with residual latent spread that **falls** steeply to about 30 months and then rises modestly — and with total between-child spread (subject intercept plus residual, last column) that falls to about 30 months and is nearly flat thereafter — not with spread that grows throughout. Two honest qualifications: the profile is not monotone, and the 12-month end sits at `p = 0.028`, where the delta-method reading is least reliable and the data thinnest — which is itself the point, because the direction one infers depends on where the range is taken to start. And because `b_mag > 0` is imposed, the exponential-decline family cannot express the U-shaped level effect at all, so a positive fitted `b_mag` is by itself uninformative about latent divergence; what carries information is its magnitude read against the level-effect benchmark, which is the ratio of 0.55 above.

Mitigating, and worth recording: `discussion.qmd`, `summary.qmd` and the signed-vocabulary results chapter are TODO stubs, and the one substantive results chapter — [`results-words-understood-spoken.qmd`](../docs/report/results-words-understood-spoken.qmd), filled in #182 — contains no dispersion or fan-out prose, so no published prose currently draws the "children fan out with age" inference. It appears as a recommendation in [`202607121200-statistical-model-review.md`](202607121200-statistical-model-review.md) §5, which proposes leading with `posterior_kappa` to communicate "the clinical 'children fan out' message". Acting on that recommendation would be the error.

### 3C. The estimand channel — the genuine exchangeability bite

`q(a) = P(speak | understand)` applied to a child's `U` understood words treats those words as a random draw from the inventory. They are not: they are the `U` easiest words. And production propensity is itself steeply difficulty-graded. Within the same child, the log-odds gap between the outer checklists has median +2.53 — an odds ratio of about **12.6** — positive in 27 of 30 children, exact Wilcoxon p = 4.7e-07 (a +0.5 continuity correction on the stratum counts; construction in §8).

`q` is therefore a composition-weighted average over each child's own understood set, not an item-level probability. `§sec-ratios` (line 111) defines it as the latter. Three consequences follow: `q`'s **level** is not transportable to another inventory; it is not comparable between populations matched on age; and it is acutely sensitive to how it is summarised, which no item-level probability would be.

| summary of `q_k` (n = 38 coherent records) | C1    | C2    | C3    |
| ------------------------------------------ | ----- | ----- | ----- |
| ratio of sums                              | 0.667 | 0.526 | 0.521 |
| median per child                           | 0.683 | 0.364 | 0.255 |
| mean per child                             | 0.612 | 0.414 | 0.318 |

The gradient roughly halves between the first and second rows. The ratio of sums weights children by vocabulary size, and the top quartile of children hold 53% of all understood words; those children have high `q` in every stratum, which compresses the between-stratum contrast. Both summaries are defensible and they answer different questions — which is exactly why `q` needs defining as the composition-weighted quantity it is.

**The rise in `q` survives, and the composition bias is conservative in sign.** As comprehension grows, a child's understood words shift toward the harder checklists (Checklist 1 weight 0.422 → 0.210, Checklist 3 weight 0.124 → 0.312 between the low and high halves of the sample), which drags marginal `q` _down_. A Kitagawa decomposition of the low-to-high difference gives:

| component             | change in `q` | share  |
| --------------------- | ------------- | ------ |
| total                 | +0.3399       | 100%   |
| within-stratum        | +0.3950       | 116.2% |
| composition (weights) | −0.0550       | −16.2% |

So composition explains none of the rise and works against it. One caveat that matters: this is a split on **comprehension level**, whereas the models report an **age** gradient. Across the wider set of specifications tried in the supporting investigation the composition share was negative in every one, ranging −2.8% to −35.2%, but for the age gradient specifically it was only −2.8% to −9.9% with intervals spanning zero. The defensible statement is that composition explains none of `q`'s rise and its sign is conservative; that it is _materially_ conservative is not established.

## 4. The matched-comprehension DS-versus-TD contrast

Because `q` is composition-dependent, matching two populations on the _number_ of words understood controls composition only if the difficulty ordering is the same in both. The headline result reproduces exactly (peak `Delta q` = +0.063 at `N` = 175, 89% interval +0.021 to +0.095), and the one quantifiable threat — instrument non-equivalence — is bounded rather than fatal. Applying the project's own dual-form crosswalk consistently to numerator and denominator moves the peak to about +0.045 with break-even around `R` = 1.25, rising to roughly 1.78 once shared short-form exposure is allowed for (only 32.4% of DS observations in the band driving the result are on the 810-item form). Two unmodelled effects push the peak up rather than down.

What binds is not measurable here. At every credibly positive level the DS children are 8–13 months older than the TD children (29.2 versus 16.8 months at `N` = 175), so any age-dependent composition difference is fully confounded with the population contrast; and whether word-difficulty ordering is the same in the two populations — the assumption matched-`N` actually needs — cannot be tested with anything in this repository, because there is **no item-level or category-level typically-developing data** here. This must be stated as a limitation rather than bounded with a fragile number.

These crosswalk figures come from the supporting investigation and were not independently reproduced for this note; the reproduced quantity is the published `Delta q` table itself.

## 5. What is not affected

Worth recording explicitly, to forestall over-correction:

- **Mean trajectories and expected word counts.** Unaffected by construction wherever data constrain the fit: `E[y] = 810 * p(z)` whatever the composition of the words a child knows, and the HSGP mean is flexible enough to absorb link misfit in-sample. Where the fit extrapolates — past ~60 months and toward the ceiling — the link's shape does real work, which is the territory of the companion note's difficulty-mixed-link proposal.
- **The DS-versus-TD dispersion ratio.** The item-difficulty contribution to the overdispersion factor is additive and small, moving the ratio by 0.2%–1.2% against a reported difference of four to five fold.
- **The count-variance accounting inside the likelihood.** Under 1.1% on `kappa` (§3A).

## 6. Evidence base, and what limits it

Every sub-inventory number here rests on two Down syndrome studies: ie_01's follow-up wave (46 children, 27–86 months, the only source recording checklist-level counts) and uk_01's 19 comprehension categories (29 children with complete category data). There is no item-level or category-level typically-developing data in the repository. The defensible grading is: **existence and direction — strong; magnitudes — provisional, single wave, single study, n < 50.**

Three data problems bear directly on the magnitudes and must be settled before any stratum number is published:

1. **The ie_01 baseline wave is defective beyond the missing Checklist 3.** Pooled Checklist 1 comprehension _falls_ from 0.855 to 0.671 between waves while the mean understood total rises from 252 to 323 words, and Checklist 1 comprehension decreases for 22 of 46 children (minimum −124). At least one wave's Checklist 1 field is unreliable. (The missing-Checklist-3 defect is handled separately, by the masking rule added in #182.)
2. **The Checklist 1 denominator is contradicted by the data.** 27 records carry Checklist 1 counts above the nominal 120 items, to a maximum of 124. Either the checklist has 124 items or those records are miscoded; the source codebook should settle it. The exchangeability test above is insensitive to this (the Checklist 3 deficit does not depend on the Checklist 1 denominator at all), but the difficulty magnitudes are not.
3. **uk_01's `understood` appears to exclude words the child also produces.** `spoken / understood` exceeds 1 for 2 of 29 children (maximum 1.95), which an inclusive comprehension field cannot do. Scope is 29 of 680 understood observations, but the bias is one-directional and sits where `q` is least identified. Verify against the source codebook before changing the pipeline.

A screening rule covering the internally inconsistent records is a **prerequisite** for publishing any stratum table, not a parallel task: the headline proportions move materially under screening (0.710 / 0.321 / 0.095 unscreened against 0.869 / 0.327 / 0.059 on a cleaned subset in the supporting investigation), as does the implied difficulty spread.

## 7. Recommended actions

No refit is proposed, and the likelihood should not be rebuilt. A stratified three-level likelihood could be supported by 46 of 680 understood observations from one wave of one study, against a blast radius of dozens of likelihood call sites plus a reporting-quality refit and recovery pass for every model of record — and the resulting model is already known to be misspecified. A full IRT reformulation is not feasible with the aggregate counts held in this repository — but this conclusion is superseded by the companion note (§5): with the item-level data reported available for roughly 218 children, a calibrate-the-instrument-then-consume-the-calibration route becomes feasible, which is a different thing from rebuilding the aggregate likelihood.

**Now, documentation only.**

1. Settle the ie_01 screening rule and the Checklist 1 item count (§6). Everything else depends on it.
2. Rewrite `§sec-betabinomial` (line 26) to separate within-observation item exchangeability from between-observation homogeneity of `p`, state that the former is false, and name what is and is not affected.
3. Rewrite the `§sec-kappa` parenthetical (line 47). Keep the "this is an assumption" framing; remove the latent-scale defence; state that the fitted decline in `kappa` is quantitatively consistent with unchanged latent between-child spread on a bounded inventory, that the monotone form cannot represent the U-shaped level effect, and that no corrected estimate of real divergence is available from these data. The same rewrite should touch the section's opening motivation ("children making more progress diverge from those making less") and the `§sec-betabinomial` gloss "(more between-child heterogeneity at a given age)" — in the RE models `kappa` is occasion-level residual dispersion above the subject intercept, so both carry the same conflation. The fan-out message that _does_ survive is on the count scale: posterior-predictive spread widens roughly five-fold from 12 to 66 months, so the `P(Y<=k)` columns and predictive intervals — not `posterior_kappa` — are the artefacts to lead with.
4. Correct the definition of `q` at `§sec-ratios` (line 111) to the composition-weighted average it is, give the measured stratum gradient, and state that composition drift suppresses the observed rise so that the reported rise is conservative in sign — while declining to say how conservative.
5. Fill the [`_caveats-ds.qmd`](../docs/report/_caveats-ds.qmd) stub with the screened stratum table and the untestable-ordering statement.
6. Reword the claim at [`comparison.py:502`](../src/vocab_growth/comparison.py) that contrasting the overdispersion factor "isolates the pure concentration difference". It is true only in the narrow sense that the factor removes the explicit `p (1 - p)` mean dependence at fixed `kappa`; because `kappa` is itself level-driven (§3B), the cross-population contrast is not a _pure_ concentration difference. The reported ratio is robust (§5) — the docstring should claim the narrow thing.
7. Open issues, do not fix inline: the uk_01 exclusive-coding question, and the ie_01 baseline Checklist 1 defect.

**Next milestone.** Add a registered `dse-native-only` sensitivity for VG10 and VG15 (`survey_vocab_max == 810`: 259 of 680 understood observations, 194 of 626 children, four sources). This is the highest-value item after the documentation, because pooling 396-, 408-, 416-, 675- and 680-item forms onto an 810 denominator is where the difficulty-ordering assumption does silent load-bearing work, and it has never been tested inside the models.

**Only if data arrives** — and the companion note (§5) reports that it now has: item-level records for ie_01, a second UK study, and the Wordbank corpus. Per-item DSE response vectors for DS children **and** item-level Wordbank responses mapped onto the DSE item pool are what make IRT and a real differential-item-functioning test possible. Nothing short of both, because the question that matters is whether the ordering is the same across populations, and that needs typically-developing items. Meanwhile, add per-item response retention to any future DSE data-collection protocol: it is free at collection time and impossible to reconstruct wherever responses were not retained.

## 8. Reproducing the numbers

All figures in §2, §3 and §6 were computed directly from `data/vocab_data_ie_01.csv` and `data/vocab_data_uk_01.csv`, and from VG10's `diagnostics.csv`, `posterior_summary_u.csv` and `posterior_kappa_u.csv`. Three points of method worth recording so the numbers can be checked rather than trusted:

- The hypergeometric test uses stratum sizes 120 / 340 / 350 and the follow-up wave only, restricted to children with `0 < T < 810`; the exchangeability null for the monotonicity rate is simulated by drawing each child's actual `T` uniformly without replacement from 810 positions, 2000 replicates.
- `q_k` is reported on records that are internally coherent (`says_k <= understands_k` for all three strata), which is 38 of 46. Note that screening on this criterion is itself selection on the outcome; the within-child paired log-odds contrast is the robust statement and does not depend on the choice. That paired contrast adds 0.5 to all four cells (a Haldane-style continuity correction) and includes every coherent child with a non-empty understood set in both outer checklists, which is where n = 30 comes from; the p-value is the exact signed-rank distribution.
- §3B's `kappa` column is the `kappa_median` of VG10's `posterior_kappa_u.csv` at the tabulated ages — equivalently, the posterior-mean parameters pushed through `kappa_min + exp(a_kappa - b_mag * z)` with the **model frame's** age standardisation (mean 40.5 / SD 20.7 months). The implied residual latent SD is a first-order delta-method reading against the population `p`, not a refit, and the total column adds the posterior-mean `tau_subj_u = 0.754` in quadrature. A latent-scale reparameterisation would be needed to estimate any of this properly, which is one reason `kappa` should not be asked to carry the interpretation.

Two corrections to earlier working claims, recorded so they are not propagated. First, uk_01's category columns **do** reconcile exactly with the recorded totals — the 19 comprehension categories sum to `understood` for all 29 complete rows. An earlier report of a 45-word discrepancy was an artefact of a column-matching pattern that split `verb14c` from `verbs14v` and missed `adject15`, `inounv` and `iverbv`. Second, the first circulated version of this note (and the pull-request description) tabulated §3B's fitted `kappa` with age standardised on the understood subset (mean 34.6 / SD 16.6) instead of the model frame's statistics (mean 40.5 / SD 20.7). That understated `kappa` at older ages by up to a quarter (8.94 against 11.68 at 66 months) and overstated the fitted log decline, so the superseded headline figures — a 1.484 log-unit decline and a ratio of 0.67 — become 1.225 and 0.55; the correction _strengthens_ the conclusion, and the values above now come from `posterior_kappa_u.csv` directly.
