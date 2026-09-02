# Dispersion $\kappa$ is not comparable across outcomes or across models

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-02, during the reporting-quality refit of #281. **Prompted by:** a reader comparing the "Dispersion $\kappa$ — understood" and "Dispersion $\kappa$ — spoken" figures across VG21 (typically developing) and VG22 (Down syndrome), finding all four dissimilar, and asking whether that is an artefact of the models or of typically developing children reaching closer to the instrument's ceiling. It is neither, and the page as it stood invited exactly that reading. This records what the four curves actually are, what separates them, and what changed.

## The ceiling hypothesis is refuted by the data

Proportion of the 810-item reference inventory understood, from each model's own analysis frame:

| Cohort and band         | Median $p_u$ | 90th percentile | Rows at $\ge$ 95% of items |
| ----------------------- | -----------: | --------------: | -------------------------: |
| VG21 (TD), 20–22 months |        0.328 |           0.461 |                       0.0% |
| VG22 (DS), 60–72 months |        0.525 |           0.814 |                       2.1% |
| VG22 (DS), 72–90 months |        0.678 |           0.807 |                       0.0% |

The typically developing window peaks at a third of the pool and contains no administration within 5% of it. The Down syndrome cohort runs closer to the ceiling than the typically developing one does, at every level. If a ceiling artefact were operating it would be in VG22 at old ages, not in VG21 at all.

The edge effect that _is_ present is the mirror image: a **floor** in Down syndrome production. Median $q$ is 0.000 through 18 months, and the model-free overdispersion of spoken counts at $\bar{p} \approx 0.005$ is 14.3 in the Down syndrome pool against 79 in the typically developing pool. Those children say almost nothing at that age, so there is no variance for the model to attribute.

## What separates the four curves

**1. $\kappa_u$ and $\kappa_s$ sit on different denominators, within a single model.** Both models run the `bivariate_re` engine, so this is not a difference between them: `nested_outcome_alpha_beta` (`src/vocab_growth/models/likelihood_utils.py`) makes $\kappa_s$ the dispersion of the _conditional_ ratio $q = $ spoken among understood, measured on the child's own understood count, while $\kappa_u$ is marginal on the 810-item pool. Setting one level beside the other compares a conditional concentration with a marginal one. Nothing on either page said so, and the shared body's spoken section went further and drew the comparison explicitly ("starts from a more dispersed position").

**2. The models give their child effects different jobs.** The between-child scales are near-identical — $\tau_{\text{subj},u}$ 0.706 (VG21) against 0.757 (VG22) — so $\kappa$ is not simply absorbing what the random effects missed. But VG22 carries child _slopes_ as well as intercepts on both trajectories through its rank-3 factor ($\tau_{\text{subj},u,1}$ 0.094, $\tau_{\text{subj},q,1}$ 0.718); VG21 carries intercepts alone. Whatever a child slope absorbs in VG22 remains in $\kappa$ for VG21. VG22 can identify those slopes because its pool has real longitudinal replication: 30.1% of its rows come from children seen once, against **70.2%** in VG21, where 83.5% of children appear a single time.

That is why the curves cross rather than sitting apart:

| Age (months) | $\kappa_u$ VG21 | $\kappa_u$ VG22 | $\kappa_s$ VG21 | $\kappa_s$ VG22 |
| -----------: | --------------: | --------------: | --------------: | --------------: |
|            8 |            27.0 |           120.5 |            55.3 |           171.5 |
|           16 |            59.9 |            89.8 |            26.3 |            89.0 |
|           22 |           143.3 |            72.3 |            15.6 |            61.1 |

$\kappa_u$ crosses at about 17 months. At matched _level_ rather than matched age the model-free overdispersion is close: variance inflation 55 for the typically developing pool at $\bar{p} = 0.248$ against 69 for the Down syndrome pool at the same $\bar{p}$.

**3. Part of VG22's $\kappa_s$ curve is prior, not data.** Its `kappa_excess_young_s` has contraction **-0.229** — the posterior is _wider_ than the prior — and `kappa_min_s` sits at prior CDF 0.96. The 89% interval on VG22's $\kappa_s$ at 8 months is [37.8, 384.8], a factor of ten, against [41.8, 70.2] for VG21. So VG22's striking $\kappa_s \approx 171$ at the young end is the prior showing through, not a finding. A third of its production rows (449 of 1,421) carry no usable understood count and enter through the `product_marginal` fallback, where the concentration is derived rather than being $\kappa_s$; VG21 has none.

**4. The identification loss is specific to VG22, not to the Down syndrome data.** VG20 fits the same pool under the same $\kappa$ priors and informs the same parameter: `kappa_excess_young_s` contracts 0.647 and `kappa_min_s` sits at prior CDF 0.858. The difference is the rank-3 factor. VG22's extra child flexibility competes with the young-end dispersion for the same information and loses it. This belongs with the open VG22 questions rather than being read as a property of the cohort.

## A defect found on the way

`scripts/prior_vs_posterior.py` flagged a parameter as `pressing` on `cdf >= CONFLICT_CDF` alone — a prior acting as a **ceiling**, never one acting as a **floor**. VG14 is entirely the second kind: five of its nine $\kappa$ parameters sit below prior CDF 0.14 (`kappa_min_u` posterior 2.74 against a prior median of 7.8, CDF 0.096; `kappa_excess_young_u` 8.7 against 84.8, CDF 0.011), and none carried a flag. `report_cells.render_prior_posterior_contraction` had tested both tails since it was written, so the console report and the page it feeds disagreed. The script's test is now two-sided. The per-fit `prior_posterior_contraction.csv` files written before this change carry the one-sided flag, so the new report block applies the test to the numbers rather than to the `flags` column and is correct against a table from either side of the fix; the sweep should be re-run before publication so the CSVs agree.

## What changed in the templates

A new shared block, `report_cells.render_dispersion_scope`, renders immediately above the first $\kappa$ figure on all sixteen templates that carry one (VG10, VG19, VG20 and VG22 receive it through `docs/models/_bivariate_re_body.qmd`). It states, from the fit's own manifest, diagnostics and contraction table:

- what each curve's dispersion is measured on, as a table, naming the item pool for $\kappa_u$ and the child's own understood count for the conditional outcomes;
- that the curves sit on different denominators and their levels are not comparable with each other;
- the conditional/fallback split of the production rows, where the definition declares a fallback, from the analysis frame rebuilt and verified against the fit's recorded hash;
- which _end_ of which curve the data did not inform, resolving a flagged parameter to the reference age its prior is anchored at ("the curve at and below 18 months") rather than naming the parameter alone;
- that $\kappa$ is residual after each model's own child structure and so is never comparable across models.

Every claim is read from the fit, so a page cannot assert a scope its own fit contradicts, and `_verified_frame` is now shared with `render_frame_composition` so the two cannot apply the hash guard differently. The shared body's cross-outcome comparison was replaced.

## Open

- The `us01-implausible-reinstated` and `dse-native-only` sensitivity arms still need the successor variants recorded during this run; nothing here changes that.
- VG22's young-end $\kappa_s$ identification loss is a candidate for the VG22 open questions: whether the rank-3 factor should carry a tighter prior at the young end, or whether the model should be reported only above the younger anchor.
- Whether the two-sided `pressing` flag changes any other model's published caveats is not yet known — it needs the sweep re-run across all twenty fits.
