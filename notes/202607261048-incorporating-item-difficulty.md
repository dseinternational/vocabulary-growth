# Incorporating item difficulty into the likelihood

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Design proposal, 2026-07-26. Nothing here is implemented. Companion to [`202607261008-challenging-item-exchangeability.md`](202607261008-challenging-item-exchangeability.md), which establishes the findings this responds to. Costs quoted below are code-seam counts taken from the current tree, not effort estimates.

> [!IMPORTANT]
> Revised the same day, before merge, after an independent verification pass: the §1 and §3 tables are recomputed exactly with their method and anchor now stated, the seam counts are corrected against the tree, and the §6 attribution of the imitation finding is fixed. A second round, responding to review on the pull-request thread, labels Proposal C's retained nested likelihood as a moment-level approximation, adds the Rasch location constraint and the variance-share correction to the DIF design, extends Proposal B's cost to anchor re-derivation and the hard-coded logistic derivative, restates the singleton counts for the fitted frame (282 of 613), scopes difficulty transportability across forms, and reframes A1 as the first registered sensitivity.

> [!CAUTION]
> **Re-billed 2026-07-26, on the study owner's challenge.** Under a Rasch-type model the total score is a sufficient statistic for ability, so heterogeneous item difficulty costs nothing for inference from sum scores; the only question is whether the Beta-Binomial fits the total, and it does, to within ~1% of the total standard deviation ([companion note](202607261008-challenging-item-exchangeability.md) §3A′, measured against VG10's fitted `kappa`).
>
> **So there is no fit-based case for any proposal here, and none should be argued on those grounds.** What survives is narrower and should be stated plainly:
>
> - **Proposal C** and the composition-free `q` address a real defect — `q` as currently defined is a composition-weighted average, not the item-level probability the report calls it. This is the estimand channel, and sufficiency does not touch it. Indeed sufficiency is the _proof_ that sum scores cannot address it: a statistic sufficient for ability is by construction uninformative about which items, so an estimand that depends on composition needs item data.
> - **Proposals A and B** are **interpretive**, not corrective. B's difficulty-mixed link shifts the implied `kappa` by roughly 10% (§1, §3) — a change to what `kappa` _means_, not to how well anything fits, and not a shift these data could detect. Neither proposal should be sold as improving the likelihood.
> - **§6, the test of the scientific hypothesis, is the load-bearing section.** "Do children with Down syndrome learn words in roughly the same order, but later?" is a claim about item difficulties. No aggregate model of any quality can test it. That, not any deficiency in the current likelihood, is why item-level data matters.
>
> §7's sequence is unchanged and its caution is vindicated: the model changes remain downstream registered sensitivities gated on the differential-item-functioning result, and nothing here licenses a refit.

## Summary

Item difficulty can be brought into the models without item-level responses for every child, because item difficulty is treated as a property of the **instrument**, not of the children — itself an invariance assumption, and §6 is its test. A small number of children measured at item level calibrates the inventory; the calibrated inventory then improves the likelihood for every aggregate observation. That is the architectural idea behind all the proposals below.

Two things must be separated first, because they look like one problem and are not. The concentration parameter `kappa` cannot be read as between-child heterogeneity, and **item difficulty is not the cause and will not fix it**. Composition dependence of the production ratio `q` _is_ an item-difficulty problem, and is what these proposals address.

With item-level data now available for roughly 218 children with Down syndrome across two instruments, plus the Wordbank typically-developing corpus, a genuine Rasch calibration becomes feasible and the differential-item-functioning question — which the companion note recorded as untestable — becomes testable. That changes the recommendation materially: the earlier conclusion "a full IRT reformulation is not close" was conditional on having only aggregate counts, and no longer holds — conditional on the ingest audit (§5).

## 1. The two problems are separate

Substituting a difficulty-aware mean function does not repair the `kappa` interpretation. Under a constant latent between-child standard deviation (`sigma = 1`; exact Gauss–Hermite integration of the logit-normal ability distribution, with the mixed link's `d_k` set from the pooled checklist profile as in §3), the implied `kappa` at matched observed level is:

| observed `p` | plain logit link | difficulty-mixed link |
| ------------ | ---------------- | --------------------- |
| 0.05         | 16.28            | 17.94                 |
| 0.20         | 6.47             | 7.44                  |
| 0.50         | 4.76             | 5.50                  |

A three-to-four-fold level-driven decline survives the change, because the effect is inherent to any bounded, saturating link rather than to equal item difficulties; at matched level the mixed link merely shifts `kappa` up by a fairly uniform 10%–13% and flattens the level-driven log decline by only about 3%. Conversely, repairing `kappa` does nothing for `q`'s composition dependence. The two need separate fixes, and only the second is about item difficulty.

## 2. Proposal A — put the age variation on the latent scale (no item data needed)

This is the fix for `kappa`, and it exploits something already present. The models carry two dispersion-like quantities:

- `tau_subj_*` — between-child spread on the **latent** scale. This is the quantity that can answer "do children fan out with age". It is a scalar `HalfNormal` in every engine that carries subject intercepts (`common_univariate_re`, `common_bivariate_re`, `common_joint_modality` — between them, all three headline models), so it is **constant by construction**; the older non-RE engines have no subject term at all, so for them A1 means adding one, not reparameterising.
- `kappa(z)` — dispersion on the **proportion** scale, which carries all the age variation and cannot support that interpretation.

The age variation is on the wrong parameter.

**A1 (preferred).** Make `tau_subj_*` age-varying and hold `kappa` constant, or nearly so. The developmental question is then asked of the parameter that can answer it, and `kappa` reverts to being residual dispersion. No new likelihood, no item data, no change to the mean function. One structural caveat, so A1 is not oversold: scaling a single per-child offset by `tau(age)` imposes perfect rank correlation of children across age — children never cross — so A1 asks "does the spread widen" under one specific fan shape. Random slopes or a child-level longitudinal function are the natural relaxations; A1 is the first registered sensitivity among these, not a demonstrated fix, and a recovery run tests recoverability _under A1's structure_, not whether that structure is right.

**A2 (alternative).** Keep `kappa(z)` but reparameterise it from a latent standard deviation: place the prior on `sigma(z)` and derive

```text
kappa = p (1 - p) / [ (dp/df)^2 * sigma^2 ] - 1
```

so the level-driven component is built in rather than absorbed as a spurious age trend. One constraint to handle: a Beta distribution's variance is bounded above by `p (1 - p)`, so this admits only `sigma < 1 / sqrt(p (1 - p))` — about `sigma < 2` at `p = 0.5`. Not practically binding, but it needs a guard.

**Cost.** `kappa` already has a single shared seam, `build_kappa_of_z` / `make_kappa_of_z` in [`gp_utils.py`](../src/vocab_growth/models/gp_utils.py), with 31 invocations of the returned closure. A2 changes the factory signature to accept `p`; A1 does not touch the factory at all.

**Validation requirement.** In the fitted frame — after the #182 masking, which reduces ie_01 to its follow-up wave — 282 of 613 children (**46%**) contribute a single observation, so `tau_subj` and `kappa` are already partly confounded — for a singleton child the two are two names for the same deviation — and making `tau_subj` age-varying makes that worse. The 331 children with repeated observations are what identify it. (The raw age-valid view has 626 children and 235 singletons; the masking removes 13 children with no surviving outcome and turns ie_01's survivors into singletons.) This needs a parameter-recovery run, which has never been executed for any model — `output/` currently contains no recovery output at all.

## 3. Proposal B — a difficulty-mixed link

The minimal way to bring item difficulty into the models. Keep the Beta-Binomial entirely and change only the inverse link:

```text
current:   p = logistic(f)
proposed:  p = sum_k w_k * logistic(f - d_k)        w_k = n_k / 810,  d_k fixed
```

No new likelihood family, no convolution, no quadrature: the mean stays a deterministic transform of the latent trajectory. Measured consequences, with `d_k = -logit(pooled stratum proportions) = (-0.71, +0.19, +1.08)` — an anchor that puts `f = 0` at the observed profile (`p = 0.398`) and therefore carries the checklist-weighted mean difficulty of +0.44 into every absolute `f`:

| quantity                        | plain | mixed |
| ------------------------------- | ----- | ----- |
| latent `f` needed for `p = 0.5` | +0.00 | +0.45 |
| latent `f` needed for `p = 0.9` | +2.20 | +2.79 |
| `f(p = 0.9) - f(p = 0.5)`       | 2.20  | 2.34  |
| peak `dp/df`                    | 0.250 | 0.228 |

The first row is mostly the anchor convention; the anchor-invariant statement is the third row. The curve flattens and its approach to the 810-word ceiling slows — modestly by `p = 0.9` (+0.14 logits beyond the anchor shift), growing toward the ceiling as the hardest checklist comes to dominate (about +0.18 logits by `p = 0.99`). That is the developmentally plausible behaviour. Two hedges keep the claim honest: the fitted mean is a flexible GP, so in-sample both links can represent essentially the same `p(age)` — the difference concentrates in extrapolation (past the data and toward the ceiling) and in what `f` and its priors mean; and "the present link gets it wrong at the top end" is therefore a claim about prior-driven extrapolation that only a refit with observed-scale predictive checks can demonstrate, not something this fixed-`f` table shows. What the change certainly delivers is that `f` becomes interpretable as ability rather than as the logit of a composition-dependent proportion — and the top end is exactly where the sparse older-age data and the un-anchored 84-month prior already make the trajectory least trustworthy.

**Cost.** 35 `math.sigmoid` sites across seven engine modules with **no shared seam**, so this requires a mechanical refactor to introduce one first. There is good precedent: the `trend_and_gp` consolidation did exactly that, and its docstring records that it was done so each engine reproduces its previous PyMC graph byte-for-byte. The same discipline should apply here, with the identity `d_k = 0` reproducing the current graph exactly. The link swap is not the whole cost, though. The slope anchors are specified as probabilities and converted with `logit` inside `trend_and_gp`, so under a mixed link every anchor prior changes meaning and must be re-expressed through the mixed link's (numerically invertible) inverse — at least that conversion is centralised in the same seam — and the expected-learning-rate artefacts hard-code the logistic derivative `dE[Y]/dx = n * p * (1 - p) * df/dx`, which is wrong under the mixed link. Budget for anchor re-derivation and downstream revalidation, not just the 35 sites.

## 4. Proposal C — stratified means for both outcomes, and a composition-free `q`

This is the proposal that actually resolves the estimand problem:

```text
p_U(theta) = sum_k w_k * logistic(f_U - d_k)
p_S(theta) = sum_k w_k * logistic(f_U - d_k) * logistic(h - e_k)
```

with `d_k` the comprehension difficulty offsets and `e_k` the production-propensity offsets by stratum. Marginal `q = p_S / p_U` then emerges as a **derived**, composition-weighted quantity — which is what it actually is — while `h` is a **composition-free production propensity**.

That addresses all three defects the companion note identifies in `q`: the definition at `§sec-ratios`, the non-transportability of its level, and most importantly the headline Down syndrome versus typically-developing contrast, which could then be run on `h` instead of on marginal `q`. Running it on `h` removes the composition confound from the comparison — conditional on the calibration: `h` is composition-free _given_ correctly calibrated, shared `d_k` / `e_k` and the form weights, a model-standardised propensity rather than an assumption-free one.

Still no item-level responses needed at fit time. One exactness caveat, so the likelihood claim is not overstated: with heterogeneous items, `E[S | U = u]` is **not** exactly `u * p_S / p_U` — conditioning on the observed total changes which items plausibly compose it — so keeping the nested `S | U ~ BetaBinomial(U, q)` structure under stratified means is a **moment-level approximation**, exact only under exchangeability. The honest menu: state the approximation explicitly; or score the stratum counts where they exist (ie_01's follow-up wave); or model `S` marginally against the full inventory. And because `q` becomes a derived quantity, any anchor priors currently placed on `q` must be re-expressed through `h` and the difficulties.

## 5. Calibrating the difficulties — what the new item-level data changes

The companion note costed calibration against aggregate stratum counts only: 46 children at one wave of ie_01, plus 29 in uk_01. That was the binding constraint on every proposal above, and it is now lifted.

Available item-level data:

| source                 | children | coverage                                     | ages (months) |
| ---------------------- | -------: | -------------------------------------------- | ------------- |
| ie_01 (DS)             |       59 | all three DSE checklists, 810 items          | 27–86         |
| second UK study (DS)   |       40 | two DSE checklists, ~460 items               | to confirm    |
| Wordbank / Edgin (DS)  |      119 | CDI WG (87 records) and WS (109), item level | 11–30         |
| Wordbank (TD, English) |   35,025 | CDI, item level                              | 8–30          |

Roughly 218 children with Down syndrome at item level, on two instruments, with complementary age coverage: the DSE sources reach 86 months and therefore identify the hard end of the inventory, while the Edgin subset is young and floor-heavy but sits on the same instrument as the typically-developing corpus. The headline number should not blur the split: only the ~99 DSE-measured children calibrate the DSE inventory directly; the 119 CDI children calibrate CDI items and reach the DSE pool only through the Route 2 crosswalk.

Provenance: the item-level datasets themselves are outside this repository, so the ie_01, second-UK and Wordbank figures above must be confirmed at ingest. One anchor is already checkable: `us_01` in this repository _is_ the Edgin aggregate — 119 children, 87 WG records carrying comprehension and 109 WS records without — matching the table exactly.

Three consequences worth stating separately.

**It makes the difficulties properly estimable.** A Rasch calibration on ~218 children and several hundred items estimates the difficulty distribution far better than 46 aggregate records, and removes the in-sample circularity of calibrating `d_k` from the same data the models then report.

**It converts two open data defects into answerable questions.** The companion note flags that ie_01's Checklist 1 counts reach 124 against a nominal 120, and that the baseline wave's Checklist 1 field is unreliable (comprehension falls between waves for 22 of 46 children). Item-level responses settle both directly: the item count is simply the number of item columns, and a partially administered checklist is visible as a block of missing rather than zero responses. Treat the item-level ingest as a data-quality fix as well as a modelling input.

**It does not need to cover everyone — but it does not automatically cover every form.** Once estimated, the difficulties enter the aggregate likelihood as fixed quantities: directly for rows on the DSE forms, and for the shorter MacArthur-derived forms (396 / 408 / 416 / 675 / 680 items) only through linked item coverage or a form-specific calibration, which is exactly what the Route 2 crosswalk has to supply. "Instrument property" is the invariance assumption §6 tests, so the calibration serves the full 1,219 rows conditionally, not by fiat. This is still why Proposals B and C stay cheap: they consume a calibration, they do not require item-level data at fit time.

## 6. Testing the scientific hypothesis: same order, slower

The working assumption — that children with Down syndrome learn words in roughly the same order but take longer — is, in its testable cross-sectional form, a differential-item-functioning hypothesis: a shared item hierarchy plus an age-associated ability offset. (Within-child acquisition order and learning speed are longitudinal claims that cross-sectional item data cannot test directly; "same order, slower" should be read as the measurement claim, not the developmental mechanism.) With item-level data on both populations it is directly testable. It is worth testing in its own right, and not only as a modelling prerequisite: if it holds, it is a substantive finding that licenses every cross-population comparison in the report; if it fails, the pattern of failure is itself informative.

### The model

For child `i` in population `g(i)` and item `j`:

```text
P(y_ij = 1) = logistic( theta_i - d_j - delta_j * 1[g(i) = DS] )

theta_i  ~ Normal( mu_g(age_i), sigma_g )     population-specific ability trajectory
d_j      ~ Normal( mu_d, sigma_d )            item difficulties, hierarchical
delta_j  ~ ZeroSumNormal( sigma_DIF )         item-level DIF
```

The hypothesis decomposes cleanly onto two parameters. **"Slower"** is the ability offset: the age shift `Delta(age)` for which `mu_TD(age - Delta) = mu_DS(age)`. **"Same order"** is `sigma_DIF` being small relative to `sigma_d`. The headline statistic is the ratio `r = sigma_DIF / sigma_d` — an SD ratio, so the population-specific share of item-difficulty _variance_ is `r^2 / (1 + r^2)`, not `r` itself — and the estimand closest to "same order" is the implied probability of item-pair rank reversals, which `r` governs given the spacing of the `d_j` and which should be reported alongside it.

### Three design points that matter

**Identifiability requires two location constraints, not one.** A constant added to every `delta_j` is indistinguishable from a shift in `mu_DS`: "all words are harder for children with Down syndrome" and "children with Down syndrome have lower ability" are the same statement. Constraining `sum_j delta_j = 0` puts the population shift entirely into `mu_DS` — which is the "slower" parameter — and leaves `delta_j` carrying only the reordering. Separately, the global Rasch location — a constant in every `d_j` against the mean of `theta` — needs its own constraint, `sum_j d_j = 0` or an equivalent anchor, or the hierarchical prior is left holding a ridge. The codebase already uses `ZeroSumNormal` for exactly this pattern in the study random effects, including the `sqrt(K/(K-1))` rescaling that preserves the marginal prior variance, so the idiom is established.

**Estimate the spread, not the individual items.** With ~218 children, an individual `delta_j` rests on a few hundred binary responses at best and will be poorly determined. `sigma_DIF`, estimated from the ensemble of several hundred `delta_j`, should be far better identified — but that is a hypothesis to verify with recovery runs, prior-sensitivity and boundary diagnostics, not a property to assume: the DS side is small and floor-heavy, and the typically-developing corpus sharpens the shared `d_j` without adding any DS information. The Edgin subset's 196 administrations also come from 119 children, so the IRT needs child effects rather than treating administrations as independent. Design the test around `sigma_DIF`, and treat per-item DIF as exploratory.

**Test by item class for power and interpretability.** Pooling items into classes — function versus content words, semantic category, word length or frequency — gives far more power per parameter and a more interpretable answer than per-item DIF. A finding of the supporting investigation — not reproduced in the companion note — that the production gradient sits mainly in `P(say | imitate)` rather than `P(imitate | understand)` suggests a specific prediction worth testing directly: that DIF, if present, is concentrated on the **production** side and on phonologically demanding items, rather than on comprehension. (A quick pooled check of ie_01's follow-up wave is directionally consistent — `P(say | imitate)` falls 0.85 / 0.78 / 0.63 across the checklists while `P(imitate | understand)` is flat at 0.68 / 0.56 / 0.63 — but the within-child paired version is not clean at these `n`, so treat this as a hypothesis for the item-level data, not an established fact.) That is a sharper and more falsifiable hypothesis than a global DIF test.

### Two linking routes, in cost order

**Route 1 — crosswalk-free, do this first.** The Wordbank Down syndrome subset (119 children, Edgin) sits on the _same_ CDI forms as the typically-developing corpus, so a DIF test between them needs **no item crosswalk at all**. This is the cheapest possible first test of the hypothesis and it is well powered on the typically-developing side. Its limitation is age: 11–30 months, so the sample is floor-heavy and the hard end of the inventory will be weakly identified for the Down syndrome group.

**Route 2 — the DSE data, via a crosswalk.** ie_01 (59 children, all three checklists, 27–86 months) and the second UK study (40 children, two checklists) cover the older and harder range that Route 1 cannot reach, but linking them to the typically-developing corpus needs a DSE-to-CDI item crosswalk. Some crosswalk machinery exists in [`crosswalk_dse_oxford.py`](../scripts/crosswalk_dse_oxford.py), though at form rather than item level. Items appearing in both inventories are the linking set; a Rasch model fitted jointly with common-item linking is the standard approach.

The two routes are complementary rather than alternatives: Route 1 tests the hypothesis cheaply at young ages on one instrument, Route 2 extends it across the full developmental range and yields the DSE difficulty calibration that Proposals B and C consume.

### What a negative result would mean

If `sigma_DIF` is small, the report gains a tested foundation for its matched-comprehension comparisons and the calibrated difficulties can be shared across populations. If it is not small, the matched-`N` Down syndrome versus typically-developing contrast is confounded in a way no reweighting fixes, and the honest response is to report the comparison on ability (`theta`) — noting that with material DIF there is no assumption-free common scale, so `theta` comparisons then need an explicit linking convention: anchor items argued invariant, or partial-invariance modelling. Either outcome is publishable and either resolves a question currently carried as an untestable caveat.

## 7. Recommended sequence

1. **Proposal A1 as the first registered sensitivity**, now. Independent of all item-level work, and the code seam already exists. The documentation fix (companion note §7) is what retires the misread conclusion; A1 then asks the developmental question of a parameter that can answer it, under A1's fan shape (§2). Pair it with the first parameter-recovery run, and keep the random-slope / longitudinal-function variants on the table.
2. **Route 1 DIF test**, next and in parallel. Needs only an item-level Wordbank pull, no crosswalk, and it tests the central scientific assumption at low cost.
3. **Ingest the DSE item-level data** for the 99 Down syndrome children, and use it to settle the Checklist 1 item count and the ie_01 baseline-wave question before anything downstream depends on those numbers.
4. **Rasch calibration on the DSE inventory**, then Route 2 DIF with an item crosswalk.
5. **Proposals B and C as a registered sensitivity variant** using the calibrated difficulties — not as models of record until the DIF result is known.
6. **Promote B and C, and re-express the cross-population contrast on `h` or on `theta`**, if and only if the DIF test supports shared difficulties.

Two things not to do. Do not change the likelihood family: there are 27 `pm.BetaBinomial` construction sites across seven engine modules, and none of the proposals above requires it. Do not fit the item-level IRT and the aggregate trajectory models as one joint model on the first pass — calibrate the instrument first, then consume the calibration, so that a problem in either half is diagnosable.

## 8. What none of this fixes

Item difficulty in the likelihood does not resolve the age confound in the headline comparison. At every comprehension level where `Delta q` is credibly positive, the Down syndrome children are 8–13 months older than the typically-developing children (29.2 against 16.8 months at 175 understood words). Conditioning on ability rather than on raw counts addresses the composition confound but not the age confound, which is a design property of the pooled sample rather than a modelling choice.
