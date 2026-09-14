# Total spread: the between-child contrast adopted for publication, in words, at matched age and level

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-14. **Decisions, by the study owner, the same day:** adopt [#229](https://github.com/dseinternational/vocabulary-growth/issues/229) option 4, so that the Down syndrome / typically developing between-child contrast is the total spread of children's counts rather than the child scale τ ([decision](https://github.com/dseinternational/vocabulary-growth/issues/229#issuecomment-5662636576)); report it **in words**, on no transformed scale; and compare the two populations **at matched age and at matched level of the same outcome**. **What this note is:** the measurements behind the second and third decisions, and the record of what [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.12 built. **Nothing here is fitted for record.** Every number is read from the 2026-09-08 fits, which predate the sex covariate and the `uk_02` mask change, so the numbers show direction and size only.

## 1. The estimand

The total spread at age _a_ is the SD, in words, of one administration of one **new** child: a fresh child effect, zero study effect and sex contrast zero. That is the child every engine's own `y_query` draws, so the population curves, `subject_heterogeneity` and the new-child predictive all describe the same child. It carries the child effect and the administration noise together and is not split between them.

It is exact, by the law of total variance over the child effect:

```
Var(Y) = n^2 Var_child(E[theta]) + E_child[ n m (1 - m) + n (n - 1) Var(theta) ]
```

with `theta ~ Beta(p kappa, (1 - p) kappa)` the administration's proportion, `m` its mean and `n = 810`. The nested spoken count needs no approximation either. The engines draw `U ~ BetaBinomial(n, ...)` and then `S | U ~ BetaBinomial(U, q, kappa_s)`, and binomial thinning makes `S | theta_U, theta_S ~ Binomial(n, theta_U theta_S)` exactly. Only the product's first two moments are needed, `E[theta_U^2] E[theta_S^2]` included, and those are elementary. The child effects enter by Gauss-Hermite quadrature: one dimension, or two for spoken, with VG20's `rho_uq`, A1's age-varying scale and VG19's child slope carried as each engine draws them. `tests/test_comparison.py` checks both forms against counts simulated the way the engines draw them, at mid-range and at the floor.

**Matched level** reads each population's total spread at the age its reference-child curve first reaches _v_ words of the same outcome. That is the curve the attainment delay D(v) crosses, so both contrasts put the populations at the same level.

## 2. Why not a logit scale

When option 4 was adopted, the recommendation was a logit-scale contrast, described as "τ² plus the dispersion converted to logits", with words alongside. Three logit-scale versions were computed on the 2026-09-08 fits beside the SD in words (600 posterior draws per fit; the exact log-odds column on 150 draws for VG11 and VG12 and 100 for VG20):

- **latent sum** — `Var_child(logit p) + E_child[1 / (p (1 - p) (kappa + 1))]`, the version recommended when option 4 was adopted;
- **first-order logit of the total** — `SD_words / (n pbar (1 - pbar))`;
- **exact log-odds SD** — the SD of `log((Y + 1/2) / (n - Y + 1/2))`, summed over the count distribution. For Down syndrome spoken it uses a moment-matched Beta-Binomial, which is an approximation.

| outcome, age                | mean words | SD words | latent sum | first-order | exact log-odds |
| --------------------------- | ---------: | -------: | ---------: | ----------: | -------------: |
| TD spoken (VG11), 12 mo     |        6.9 |     10.3 |       1.50 |        1.52 |           1.33 |
| DS spoken (VG20), 12 mo     |        1.5 |      5.2 |       3.83 |        3.44 |       **1.00** |
| DS spoken (VG20), 24 mo     |       13.8 |     29.2 |       2.42 |        2.18 |           1.73 |
| TD spoken (VG11), 30 mo     |        426 |      205 |       1.25 |        1.01 |           1.47 |
| TD understood (VG12), 24 mo |        351 |      131 |       0.72 |        0.66 |           0.74 |
| DS understood (VG20), 24 mo |        129 |     98.6 |       0.99 |        0.91 |           1.12 |

**They disagree about the direction of the spoken contrast at the floor.** The two versions that convert dispersion to logits put Down syndrome children at about 2.3–2.6 times the typically developing spread at 12 months. The exact log-odds SD puts them below it, as the SD in words does. The first two are dominated by the tiny means there. The third depends on the `+1/2` continuity correction exactly where the contrast is decided, and for Down syndrome spoken it can only be approximated. Away from the floor the versions still differ by as much as 45% (TD spoken, 30 months: 1.01 against 1.47).

**The SD in words has none of these problems but moves with the mean.** Comparing at matched level removes that dependence directly, without a transform, which is why the level-matched view was built beside the age-matched one.

## 3. The total is not automatically free of the split

Option 4 rests on the total being better identified than the split. Across VG12's posterior draws, the correlation of each quantity with `subject_variance_share` was:

| age   | SD words | first-order | latent sum | exact log-odds |     τ |
| ----- | -------: | ----------: | ---------: | -------------: | ----: |
| 12 mo |    +0.15 |       +0.22 |      +0.09 |          −0.25 | +0.76 |
| 15 mo |    +0.57 |       +0.61 |      +0.61 |          +0.50 | +0.76 |
| 21 mo |    +0.71 |       +0.74 |      +0.75 |          +0.77 | +0.76 |
| 24 mo |    +0.74 |       +0.76 |      +0.76 |          +0.79 | +0.76 |

Near the young dispersion anchor (12 months), trading τ against κ leaves the total almost unchanged. By 21–24 months the total follows the share as closely as τ does. That is consistent with the partition tying τ only to the dispersion at the young anchor, with the older-age dispersion free of it. This is why the decision made scoring the total in recovery a condition: whether the total at the reporting ages comes back right has to be measured, not inferred from `v_total`.

## 4. What the 2026-09-08 fits say, for direction

VG20 against VG12 (understood) and VG11 (spoken), 36,000 paired draws; ratio TD/DS with its 89% interval.

| outcome    | at matched age                                                                     | at matched level                                                                                            |
| ---------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| understood | 1.94 [1.66, 2.28] at 12 mo; 1.27 [1.18, 1.36] at 25 mo                             | 0.87 [0.82, 0.93] at 50 words; 0.84 [0.80, 0.89] at 100; 0.83 [0.80, 0.87] at 200                           |
| spoken     | 1.96 [1.48, 2.63] at 12 mo; 5.84 [5.01, 6.81] at 24 mo; 3.93 [3.48, 4.44] at 30 mo | 0.64 [0.59, 0.69] at 25 words; 0.91 [0.87, 0.96] at 100; 1.01 [0.97, 1.05] at 200; 1.10 [1.06, 1.15] at 400 |

**The two bases answer differently, and that is the substance of [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.13's open question.**

- **At matched age**, typically developing children are more spread out on both outcomes, because they know far more words: the new child's mean is 351 against 129 words understood at 24 months, and 260 against 14 spoken.
- **At matched level**, Down syndrome children are more spread out on comprehension at every level the two populations both reach reliably, 30 to 300 words. The ratio is 0.83–0.93 and every 89% interval excludes 1. On production they are more spread out up to 100 words (0.48 at 10 words), the intervals include 1 at 150 and 200, and from 300 words the typically developing children spread further (1.06 [1.03, 1.10] at 300).

The book's current τ contrast has Down syndrome children more variable on the logit scale. The level-matched view agrees with it in direction on comprehension throughout, and on production only below about 125 words.

## 5. What was built

- **`vocab_growth.comparison`**:
  - `total_spread_single` and `total_spread_product` compute the exact mean and SD;
  - `total_spread_plan` resolves which trace variables an outcome reads, from `subject_effects.resolve` and never from what a trace contains;
  - `total_spread_from_values` does the computation;
  - `total_spread` reads a fitted trace;
  - `value_at_level` reads a per-draw value at each draw's crossing age.

  `child_spread_product` now shares its quadrature (`_product_nodes`) with the spoken total, so the two integrate over the same children. The joint engine and VG22's low-rank factor are refused, not approximated.

- **`scripts/compare_ds_td_re.py`** writes `ds_td_<outcome>_re_total_spread.csv` (by age, with the new child's mean) and `ds_td_<outcome>_re_total_spread_at_level.csv` (on the attainment-delay level grid), with a figure each. On the 2026-09-08 traces a full run took 27 s for understood and 328 s for spoken, most of the latter in the two-dimensional quadrature. `--verify` checks both forms against simulated children. The τ and dispersion tables are still written; whether they stay is task 4.13's decision.
- **Parameter recovery** (`recovery.compare.with_total_spread`) derives the total spread at the query ages on the posterior and on the truth draw alike, by the same function, and scores it with the trajectory quantities:
  - variables: `total_spread_words_query` on VG11 and VG12, `total_spread_words_u_query` and `total_spread_words_s_query` on the bivariate models;
  - probability-scale inputs, because a truth draw does not carry the logits.

  Checked read-only on the three 2026-09-11 VG16 `test` replicates, it adds 28 targets to each, taking the target count from 133 to 161. On the one converged replicate:
  - comprehension: all 14 ages inside their 89% intervals;
  - production: 11 of 14 inside; the misses are high at 18 and 24 months (z 1.9, 2.0) and low at 54 (z −1.7).

  That is a code-path check on an older frame, not a result for VG16, and VG16 is not one of the three models the decision names.

  Each side is derived on its own definition's query ages, and only when the two grids match. A cross-definition run whose truth sits on a different grid leaves the quantity out rather than comparing spreads at different ages.

- **Against the engines' own new child.** The analytic SD was checked against the `y_*_query` posterior-predictive draws stored in the 2026-09-08 traces. The comparison is between the predicted marginal SD across the posterior, `sqrt(E[SD^2] + Var(mean))`, and the empirical SD of the stored draws, at every query age:

  | model and outcome | draws | analytic / empirical SD |
  | ----------------- | ----: | ----------------------- |
  | VG20 understood   | 9,000 | 0.993–1.007             |
  | VG20 spoken       | 9,000 | 0.965–1.028             |
  | VG11 spoken       | 9,000 | 0.984–1.002             |
  | VG19 spoken       | 9,000 | 0.961–1.007             |

  An independent review using all 36,000 draws found VG20 spoken at 1.000–1.022, apart from 1.050 at 18 months. The spread across ages is the Monte Carlo error of an SD of heavy-tailed counts, and the sign of the small departures changes between the two draw sets.

- **Reviewed.** An adversarial review of the change found no defect in the maths, the variables each structure reads, the recovery plumbing or the refactor. `child_spread_product` is bit-identical to its previous output. What it did find is fixed:
  - the truth-grid rule above;
  - two tests whose crossings all fell on grid nodes, so the interpolation inside an interval was never run;
  - a test whose name claimed exactness its tolerance did not test;
  - an undocumented effect on `coverage_ci89` (below).

## 6. What this does not do

- **It runs no recovery for record.** The condition is scored for VG20, VG11 and VG12 in task 4.8's reruns after the refit.
- **It changes no book or report text.** That is task 4.13: which basis is the headline, whether τ stays as a secondary quantity, and the variance-shares TODO in `docs/report/methods-models.qmd`.
- **It provides no administration-weighted version.** The total spread is the reference study's (study effect zero), as every reference-child curve here is.
- **It changes a replicate's target count** for every univariate and bivariate model scored from now on, and so its `coverage_ci89`. The new rows are a function of quantities already scored, so they are correlated with them and make the pooled coverage lean further on the trajectory block. Read the total-spread rows on their own for the #229 condition. Matrices scored before and after are not comparable row for row (`docs/runbooks/parameter-recovery.md`).
