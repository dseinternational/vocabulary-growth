# Incorporating item difficulty into the likelihood

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Design proposal, 2026-07-26. Nothing here is implemented. Companion to [`202607261008-challenging-item-exchangeability.md`](202607261008-challenging-item-exchangeability.md), which establishes the findings this responds to. Costs quoted below are code-seam counts taken from the current tree, not effort estimates.

## Summary

Item difficulty can be brought into the models without item-level responses for every child, because item difficulty is a property of the **instrument**, not of the children. A small number of children measured at item level calibrates the inventory; the calibrated inventory then improves the likelihood for every aggregate observation. That is the architectural idea behind all the proposals below.

Two things must be separated first, because they look like one problem and are not. The concentration parameter `kappa` cannot be read as between-child heterogeneity, and **item difficulty is not the cause and will not fix it**. Composition dependence of the production ratio `q` *is* an item-difficulty problem, and is what these proposals address.

With item-level data now available for roughly 218 children with Down syndrome across two instruments, plus the Wordbank typically-developing corpus, a genuine Rasch calibration becomes feasible and the differential-item-functioning question — which the companion note recorded as untestable — becomes testable. That changes the recommendation materially: the earlier conclusion "a full IRT reformulation is not close" was conditional on having only aggregate counts, and no longer holds.

## 1. The two problems are separate

Substituting a difficulty-aware mean function does not repair the `kappa` interpretation. Under a constant latent between-child standard deviation, the implied `kappa` is:

| observed `p` | plain logit link | difficulty-mixed link |
| ------------ | ---------------- | --------------------- |
| 0.05         | 18.92            | 20.13                 |
| 0.20         | 5.33             | 6.26                  |
| 0.50         | 3.17             | 4.00                  |

A five-fold level-driven decline survives the change, because the effect is inherent to any bounded, saturating link rather than to equal item difficulties. Conversely, repairing `kappa` does nothing for `q`'s composition dependence. The two need separate fixes, and only the second is about item difficulty.

## 2. Proposal A — put the age variation on the latent scale (no item data needed)

This is the fix for `kappa`, and it exploits something already present. The models carry two dispersion-like quantities:

- `tau_subj_*` — between-child spread on the **latent** scale. This is the quantity that can answer "do children fan out with age". It is a scalar `HalfNormal` in every engine, so it is **constant by construction**.
- `kappa(z)` — dispersion on the **proportion** scale, which carries all the age variation and cannot support that interpretation.

The age variation is on the wrong parameter.

**A1 (preferred).** Make `tau_subj_*` age-varying and hold `kappa` constant, or nearly so. The developmental question is then asked of the parameter that can answer it, and `kappa` reverts to being residual dispersion. No new likelihood, no item data, no change to the mean function.

**A2 (alternative).** Keep `kappa(z)` but reparameterise it from a latent standard deviation: place the prior on `sigma(z)` and derive

```text
kappa = p (1 - p) / [ (dp/df)^2 * sigma^2 ] - 1
```

so the level-driven component is built in rather than absorbed as a spurious age trend. One constraint to handle: a Beta distribution's variance is bounded above by `p (1 - p)`, so this admits only `sigma < 1 / sqrt(p (1 - p))` — about `sigma < 2` at `p = 0.5`. Not practically binding, but it needs a guard.

**Cost.** `kappa` already has a single shared seam, `build_kappa_of_z` / `make_kappa_of_z` in [`gp_utils.py`](../src/vocab_growth/models/gp_utils.py), with 24 invocations of the returned closure. A2 changes the factory signature to accept `p`; A1 does not touch the factory at all.

**Validation requirement.** 235 of 626 children (38%) contribute a single observation, so `tau_subj` and `kappa` are already partly confounded — for a singleton child the two are two names for the same deviation — and making `tau_subj` age-varying makes that worse. The 391 children with repeated observations are what identify it. This needs a parameter-recovery run, which has never been executed for any model — `output/` currently contains no recovery output at all.

## 3. Proposal B — a difficulty-mixed link

The minimal way to bring item difficulty into the models. Keep the Beta-Binomial entirely and change only the inverse link:

```text
current:   p = logistic(f)
proposed:  p = sum_k w_k * logistic(f - d_k)        w_k = n_k / 810,  d_k fixed
```

No new likelihood family, no convolution, no quadrature: the mean stays a deterministic transform of the latent trajectory. Measured consequences, using stratum difficulties calibrated to the observed checklist profile:

| quantity                       | plain | mixed |
| ------------------------------ | ----- | ----- |
| latent `f` needed for `p = 0.5` | +0.00 | +0.45 |
| latent `f` needed for `p = 0.9` | +2.22 | +2.81 |
| peak `dp/df`                    | 0.245 | 0.225 |

The curve flattens and its approach to the 810-word ceiling slows markedly. That is the developmentally realistic behaviour, and it is what the present link gets wrong at the top end — which is also where the sparse older-age data and the un-anchored 84-month prior already make the trajectory least trustworthy. It also makes `f` interpretable as ability rather than as the logit of a composition-dependent proportion.

**Cost.** 35 `math.sigmoid` sites across seven engine modules with **no shared seam**, so this requires a mechanical refactor to introduce one first. There is good precedent: the `trend_and_gp` consolidation did exactly that, and its docstring records that it was done so each engine reproduces its previous PyMC graph byte-for-byte. The same discipline should apply here, with the identity `d_k = 0` reproducing the current graph exactly.

## 4. Proposal C — stratified means for both outcomes, and a composition-free `q`

This is the proposal that actually resolves the estimand problem:

```text
p_U(theta) = sum_k w_k * logistic(f_U - d_k)
p_S(theta) = sum_k w_k * logistic(f_U - d_k) * logistic(h - e_k)
```

with `d_k` the comprehension difficulty offsets and `e_k` the production-propensity offsets by stratum. Marginal `q = p_S / p_U` then emerges as a **derived**, composition-weighted quantity — which is what it actually is — while `h` is a **composition-free production propensity**.

That fixes all three defects the companion note identifies in `q`: the definition at `§sec-ratios`, the non-transportability of its level, and most importantly the headline Down syndrome versus typically-developing contrast, which could then be run on `h` instead of on marginal `q`. Running it on `h` removes the composition confound from the comparison entirely, which is the one threat to that result that could not otherwise be bounded.

Still a Beta-Binomial on totals. Still no item-level responses needed at fit time.

## 5. Calibrating the difficulties — what the new item-level data changes

The companion note costed calibration against aggregate stratum counts only: 46 children at one wave of ie_01, plus 29 in uk_01. That was the binding constraint on every proposal above, and it is now lifted.

Available item-level data:

| source                      | children | coverage                                        | ages (months) |
| --------------------------- | -------: | ----------------------------------------------- | ------------- |
| ie_01 (DS)                  |       59 | all three DSE checklists, 810 items             | 27–86         |
| second UK study (DS)        |       40 | two DSE checklists, ~460 items                  | to confirm    |
| Wordbank / Edgin (DS)       |      119 | CDI WG (87 records) and WS (109), item level    | 11–30         |
| Wordbank (TD, English)      |   35,025 | CDI, item level                                 | 8–30          |

Roughly 218 children with Down syndrome at item level, on two instruments, with complementary age coverage: the DSE sources reach 86 months and therefore identify the hard end of the inventory, while the Edgin subset is young and floor-heavy but sits on the same instrument as the typically-developing corpus.

Three consequences worth stating separately.

**It makes the difficulties properly estimable.** A Rasch calibration on ~218 children and several hundred items estimates the difficulty distribution far better than 46 aggregate records, and removes the in-sample circularity of calibrating `d_k` from the same data the models then report.

**It converts two open data defects into answerable questions.** The companion note flags that ie_01's Checklist 1 counts reach 124 against a nominal 120, and that the baseline wave's Checklist 1 field is unreliable (comprehension falls between waves for 22 of 46 children). Item-level responses settle both directly: the item count is simply the number of item columns, and a partially administered checklist is visible as a block of missing rather than zero responses. Treat the item-level ingest as a data-quality fix as well as a modelling input.

**It does not need to cover everyone.** The calibrated difficulties are instrument properties. Once estimated, they enter the aggregate likelihood for all 1,219 observations as fixed quantities. This is why Proposals B and C stay cheap: they consume a calibration, they do not require item-level data at fit time.

## 6. Testing the scientific hypothesis: same order, slower

The working assumption — that children with Down syndrome learn words in roughly the same order but take longer — is exactly a differential-item-functioning hypothesis, and with item-level data on both populations it is directly testable. It is worth testing in its own right, and not only as a modelling prerequisite: if it holds, it is a substantive finding that licenses every cross-population comparison in the report; if it fails, the pattern of failure is itself informative.

### The model

For child `i` in population `g(i)` and item `j`:

```text
P(y_ij = 1) = logistic( theta_i - d_j - delta_j * 1[g(i) = DS] )

theta_i  ~ Normal( mu_g(age_i), sigma_g )     population-specific ability trajectory
d_j      ~ Normal( mu_d, sigma_d )            item difficulties, hierarchical
delta_j  ~ ZeroSumNormal( sigma_DIF )         item-level DIF
```

The hypothesis decomposes cleanly onto two parameters. **"Slower"** is the ability offset: the age shift `Delta(age)` for which `mu_TD(age - Delta) = mu_DS(age)`. **"Same order"** is `sigma_DIF` being small relative to `sigma_d` — the headline statistic is the ratio `sigma_DIF / sigma_d`, the fraction of difficulty variation that is population-specific rather than shared.

### Three design points that matter

**Identifiability requires the zero-sum constraint.** A constant added to every `delta_j` is indistinguishable from a shift in `mu_DS`: "all words are harder for children with Down syndrome" and "children with Down syndrome have lower ability" are the same statement. Constraining `sum_j delta_j = 0` puts the population shift entirely into `mu_DS` — which is the "slower" parameter — and leaves `delta_j` carrying only the reordering. The codebase already uses `ZeroSumNormal` for exactly this pattern in the study random effects, including the `sqrt(K/(K-1))` rescaling that preserves the marginal prior variance, so the idiom is established.

**Estimate the spread, not the individual items.** With ~218 children, an individual `delta_j` rests on a few hundred binary responses at best and will be poorly determined. But `sigma_DIF` is estimated from the ensemble of several hundred `delta_j` and is well identified even when none of its components is. Design the test around `sigma_DIF`, and treat per-item DIF as exploratory.

**Test by item class for power and interpretability.** Pooling items into classes — function versus content words, semantic category, word length or frequency — gives far more power per parameter and a more interpretable answer than per-item DIF. The companion note's finding that the production gradient sits in `P(say | imitate)` while `P(imitate | understand)` is flat suggests a specific prediction worth testing directly: that DIF, if present, is concentrated on the **production** side and on phonologically demanding items, rather than on comprehension. That is a sharper and more falsifiable hypothesis than a global DIF test.

### Two linking routes, in cost order

**Route 1 — crosswalk-free, do this first.** The Wordbank Down syndrome subset (119 children, Edgin) sits on the *same* CDI forms as the typically-developing corpus, so a DIF test between them needs **no item crosswalk at all**. This is the cheapest possible first test of the hypothesis and it is well powered on the typically-developing side. Its limitation is age: 11–30 months, so the sample is floor-heavy and the hard end of the inventory will be weakly identified for the Down syndrome group.

**Route 2 — the DSE data, via a crosswalk.** ie_01 (59 children, all three checklists, 27–86 months) and the second UK study (40 children, two checklists) cover the older and harder range that Route 1 cannot reach, but linking them to the typically-developing corpus needs a DSE-to-CDI item crosswalk. Some crosswalk machinery exists in [`crosswalk_dse_oxford.py`](../scripts/crosswalk_dse_oxford.py), though at form rather than item level. Items appearing in both inventories are the linking set; a Rasch model fitted jointly with common-item linking is the standard approach.

The two routes are complementary rather than alternatives: Route 1 tests the hypothesis cheaply at young ages on one instrument, Route 2 extends it across the full developmental range and yields the DSE difficulty calibration that Proposals B and C consume.

### What a negative result would mean

If `sigma_DIF` is small, the report gains a tested foundation for its matched-comprehension comparisons and the calibrated difficulties can be shared across populations. If it is not small, the matched-`N` Down syndrome versus typically-developing contrast is confounded in a way no reweighting fixes, and the honest response is to report the comparison on ability (`theta`) rather than on word counts. Either outcome is publishable and either resolves a question currently carried as an untestable caveat.

## 7. Recommended sequence

1. **Proposal A1**, now. Independent of all item-level work, fixes the one reported conclusion that does not survive, and the code seam already exists. Pair it with the first parameter-recovery run.
2. **Route 1 DIF test**, next and in parallel. Needs only an item-level Wordbank pull, no crosswalk, and it tests the central scientific assumption at low cost.
3. **Ingest the DSE item-level data** for the 99 Down syndrome children, and use it to settle the Checklist 1 item count and the ie_01 baseline-wave question before anything downstream depends on those numbers.
4. **Rasch calibration on the DSE inventory**, then Route 2 DIF with an item crosswalk.
5. **Proposals B and C as a registered sensitivity variant** using the calibrated difficulties — not as models of record until the DIF result is known.
6. **Promote B and C, and re-express the cross-population contrast on `h` or on `theta`**, if and only if the DIF test supports shared difficulties.

Two things not to do. Do not change the likelihood family: there are 30 `BetaBinomial` call sites across seven modules, and none of the proposals above requires it. Do not fit the item-level IRT and the aggregate trajectory models as one joint model on the first pass — calibrate the instrument first, then consume the calibration, so that a problem in either half is diagnosable.

## 8. What none of this fixes

Item difficulty in the likelihood does not resolve the age confound in the headline comparison. At every comprehension level where `Delta q` is credibly positive, the Down syndrome children are 8–13 months older than the typically-developing children (29.2 against 16.8 months at 175 understood words). Conditioning on ability rather than on raw counts addresses the composition confound but not the age confound, which is a design property of the pooled sample rather than a modelling choice.
