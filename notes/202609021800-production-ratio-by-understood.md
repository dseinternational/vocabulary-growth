# The production ratio at a comprehension milestone: the curve is not the children

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-02, during the reporting-quality refit of #281. **Prompted by:** a reader setting VG21 (typically developing) beside VG22 (Down syndrome) and noting that when each population reaches 300 words understood, the production ratio on the by-understood figure is 0.4 in both, with similar-looking spread — and asking whether that result can be trusted. It cannot be trusted _as the reader read it_, and the pages were inviting that reading against the code's own documentation.

## What the figure is

`plot_production_rate_by_understood` (`common_bivariate.py`) plots the population ratio $q$ against the population expected words understood, both read off the age curves at zero study and zero child effects. Its docstring, written for issue #233, is explicit: the x value at a point is the median child's comprehension _at some age_ and the y value is the median child's ratio _at that same age_, so the curve is a developmental-stage relationship and **not** $E[q \mid \text{understood} = U]$ for a child who understands $U$ words. Three templates captioned it as the conditional anyway — the shared body used by VG10, VG19, VG20 and VG22 ("children at this level of comprehension convert this fraction of it"), and VG21 and VG13 ("what share of a given comprehension vocabulary is typically spoken"). VG05, VG07 and VG16 already carried the correct reading; VG08 and VG09 said "population-level" and no more.

## What the children say

Rebuilding each fit's analysis frame and verifying it against the recorded hash, then taking every administration with a usable spoken count and an understood count within ±10% of a level:

| Words understood |            VG21 curve | TD children observed (median, IQR, n) |            VG22 curve | DS children observed (median, IQR, n) |
| ---------------: | --------------------: | ------------------------------------- | --------------------: | ------------------------------------- |
|              100 |                  0.09 | 0.09 (0.03–0.19), 527                 |                  0.05 | 0.06 (0.01–0.10), 52                  |
|              200 |                  0.22 | 0.16 (0.07–0.28), 591                 |                  0.12 | 0.10 (0.04–0.20), 73                  |
|              300 | **0.43** [0.40, 0.45] | **0.27** (0.11–0.47), 464             | **0.40** [0.36, 0.45] | **0.13** (0.05–0.32), 97              |
|              400 |                     — | —                                     |     0.66 [0.60, 0.72] | 0.38 (0.16–0.66), 78                  |

The curves agree at 300 words. The children do not: a typically developing child who understands 300 words typically speaks 27% of them, a child with Down syndrome 13%, and both are well below the 0.4 the curves show — though, as the next section sets out, the two groups of children at that level are not comparable and the Down syndrome figure is study-dependent. The distributions are not similar either; the Down syndrome one sits lower throughout its interquartile range. The gap between curve and children grows with the level and is widest in the Down syndrome pool.

## Why they diverge — corrected after a third check

The curve's value at 300 is the population ratio at the age when the reference child (zero study and child effects) reaches 300 words: 21 months for VG21, 47 months for VG20. Conditioning on 300 words instead selects every child who reached it at _any_ age, and the ratio rises with age, so the selected children are younger than that crossing age (median 17 and 38 months) and speak a smaller share. Decomposing the drop from curve to children through each population's own $q(a)$ at the children's median age, that age selection accounts for **0.23 in both populations** — not more in the Down syndrome pool, as the first version of this note claimed. The population ratio at those two ages is the same, 0.20, so the whole of the 0.14 gap between the two pools' children is residual: TD children 0.07 above their age's ratio, DS children 0.07 below.

The second version of this note read those residuals as a selection asymmetry — comprehension-advanced typically developing children converting more, typical children with Down syndrome converting like their peers. Splitting each pool's children at 300 words by study and adjusting each study for its own median age shows that reading was built on the pooled numbers and does not survive:

| Study         |   n | median age | observed q | population q(age) | residual |
| ------------- | --: | ---------: | ---------: | ----------------: | -------: |
| TD `Marchman` | 171 |         17 |       0.33 |              0.20 |    +0.13 |
| TD `Thal`     | 124 |         16 |       0.17 |              0.16 |    +0.02 |
| TD `Floccia`  | 119 |         19 |       0.33 |              0.30 |    +0.03 |
| TD `Caselli`  |  45 |         18 |       0.20 |              0.24 |    −0.04 |
| DS `it_01`    |  36 |         40 |       0.10 |              0.27 |    −0.17 |
| DS `uk_02`    |  21 |         38 |       0.34 |              0.21 |    +0.14 |
| DS `es_01`    |  19 |         34 |       0.15 |              0.13 |    +0.02 |
| DS `uk_07`    |  11 |         53 |       0.39 |              0.55 |    −0.16 |

Residuals scatter from −0.17 to +0.14 inside both pools, with no consistent sign in either. The pooled TD +0.07 is `Marchman`; the pooled DS −0.07 is `it_01` and `uk_07`. Age-adjusted, the four largest Down syndrome studies convert alike across all levels at 38 months (0.15–0.18), so the spread at 300 words is not a study-level conversion difference either; it is what 11–36 children at one level look like. At 200 words, where the children in both pools are typical of their age, no residual remains on either side and the children's gap (0.16 against 0.10) equals the population-stage gap. **No Down-syndrome-specific shortfall in conversion at 300 words is identifiable once age and study are in view, and no selection asymmetry needs to be invoked.**

One further thing the check turned up. "Typical of their age" was judged against the sample: the observed median comprehension near 38 months in the Down syndrome pool is 286 words, but VG20's population curve puts the reference child at 232 there — 54 words below the sample's median child, where VG21's reference child at 17 months (185) sits close to its sample median (174). The population curve is the zero-study-effect child, and study effects are centred over studies rather than over children, so a pool whose large studies sit above the average study will have a reference child below its own median child. It does not change the decomposition above, which uses the model's $q(a)$ consistently, but it means "median child" is not a safe label for the Down syndrome population curve; the pages now say "reference child".

## The reference child is not the pool's median child — and what it does to the milestones

The calibration check above generalised. Comparing each model's reference-child curve with the sample median administration at several ages:

|                                 |  TD 18 mo |      TD 21 mo |  DS 30 mo |      DS 38 mo |  DS 48 mo |
| ------------------------------- | --------: | ------------: | --------: | ------------: | --------: |
| Understood — reference / sample | 214 / 200 | **301 / 255** | 175 / 208 | **232 / 284** | 311 / 333 |
| Spoken — reference / sample     |   52 / 35 |  **129 / 77** |   14 / 16 |       48 / 39 | 142 / 129 |

The cause is the study effects and the pool's age coverage together. Study effects are centred over _studies_, unweighted; the reference child is the child in the average study. But studies are segregated by age: the three studies that sample children aged 30–40 months in the Down syndrome pool (`es_01` +0.45, `uk_02` +0.32, `it_01` +0.23 on the logit scale) all sit above the average study, and at 19–22 months the typically developing pool is one study (`Floccia`, −0.31) sitting below it. Where studies do not overlap in age the model has to split each age band's level between the trend and the studies present, and partial pooling does — so the reference child at 38 months is below every Down syndrome study sampled there and at 21 months above the only typically developing one. It is the trend-versus-study ridge the refit runbook already documents, showing up on the reporting side. The row-weighted mean study effect is near zero (+0.01 TD, +0.06 DS on comprehension), so the discrepancy is age-local, not a global offset. On production the study effects are far larger — `us_01` +1.10, `ie_02` −1.03, `uk_07` −0.70 — so the Down syndrome production ratio is extremely study-dependent.

**This reaches the milestone story.** The delay factors in the discussion that led here used reference-child crossings. On sample medians (rolling window):

| Milestone      | TD reference / sample | DS reference / sample | ratio, reference / sample |
| -------------- | --------------------: | --------------------: | ------------------------: |
| 100 understood |               14 / 14 |             23 / 21.5 |               1.64 / 1.54 |
| 200 understood |               18 / 18 |           34 / **29** |           1.89 / **1.61** |
| 300 understood |    21 / _not reached_ |         47 / **39.5** |                  2.24 / — |
| 50 spoken      |               18 / 19 |             39 / 39.5 |               2.17 / 2.08 |

The reading "the comprehension delay grows from 1.6× to 2.2× to meet a steady production delay" came from exactly the two places where the reference child and the sample diverge, and does not survive: on the sample the comprehension delay is about 1.6× throughout the measurable range and the production delay about 2.1× — a **constant differential delay**, which is a simpler statement and the one the data support. The "0.43 = 0.43" convergence of the production ratio at 300 words is likewise the reference child's, at the typically developing window's edge, where the sample's ratio at 21 months is 0.30 and the sample median never reaches 300 words in the window. Whether the comprehension delay grows with level cannot be settled in this pool, because the answer depends on which studies cover which ages.

## Decision (2026-09-02)

Agreed: **(a)** the reference child stays the estimand every population curve and milestone reports — it is what the model defines and it is comparable across models and populations — and is never again called the typical or median child; **(d)** every joint RE page carries a per-study fan (`study_fans.png`: one curve per study over its own ages, the reference child bold, the administration-weighted child dashed) and a calibration block (`render_reference_child_calibration`) setting the reference child beside the weighted child and the sample median at three ages; **(e)** milestones and DS/TD delays are reported under both the reference child and the administration-weighted child — the same fit re-weighted, with a Gaussian kernel in age, to the studies present at each age — and the gap is reported as the study-coverage sensitivity (`posterior_summary_monthly_weighted_{u,s}.csv` on the pages; `ds_td_*_attainment_delay_weighted.csv`, `ds_td_comprehension_q_at_U_weighted.csv` and `ds_td_comprehension_latency_weighted.csv` in the comparison). The limitation is recorded as a data-collection fact: trend and study offsets are separately identified only where studies overlap in age, and in this pool they mostly do not.

## What is and is not confident

As a **population-stage statement** the agreement is real and reasonably well supported: both curves are estimated with narrow intervals that overlap, and each sits within about 0.05–0.08 of the empirical median ratio of all children at the corresponding age (0.35 in both pools). The typically developing value is at the very end of its curve — VG21's window is capped at 22 months and the curve's x-axis stops at 328 words — so it is an edge estimate rather than a mid-range one. As a **statement about children at a milestone** it is not supported, and the pages should not let a reader make it.

## What changed

- The three captions now say what the axis is and that the figure is not the conditional share.
- A new shared block, `report_cells.render_conditional_production_check`, renders beneath the figure on all eight templates that carry it (VG10, VG19, VG20 and VG22 through the shared body). It states the reading, then sets the curve beside the observed children at each level the curve covers — count, median ratio, interquartile range, median age — from the hash-verified frame, and says that a comparison between populations at a comprehension milestone must be made in that column.

## Follow-through into the DS/TD comparison

Three surfaces carried the same curve into the cross-population contrast.

- **The comparison book's "Comprehension-matched production ratio"** (`compare_ds_td_re.py comprehension`) already stated the #233 reading, but tabulated N = 50/100/150 and concluded "converging by N ≈ 200" — and its TD comparator was VG13 (8–18 months), whose population median never reaches 250 words within support, so the TD curve column was blank above N = 200 and the contrast could say nothing about the 300-word milestone. Two changes. The TD joint comparator is now **VG21** (8–22 months), the registered form of the `window-22` extension adopted on 2026-08-21 that `MAX_MATCHED_U = 320` was already set for — in `compare_ds_td_re`, `compare_ds_td_expressive`, `compare_ds_td_trajectories` and `compare_models`' two by-age overlays; VG11 and VG12 remain the univariate comparators. And `run_comprehension_matched` now also writes `ds_td_comprehension_q_observed.csv` from the two hash-verified frames through the same `observed_production_ratio_at_levels` the model pages use, so the book's table runs to N = 300 with the observed children beside the curve in each population. On the book's own table the population curves now genuinely converge — Δq +0.10 at 200, +0.04 at 250, **0.00 [−0.07, +0.08] at 300 with P(Δq > 0) = 0.50**, both at about 0.43 — while the children who understood 300 words give 0.27 (TD, 464 children) and 0.13 (DS, 97 children). The prose says both, and that a statement about children at a milestone is made from the children columns.
- **`compare_models.ds_td_q_vs_understood`**, described in its docstring as a "headline matched-comprehension q overlay" (DS VG09 vs TD VG13, VG07 dashed), together with `ds_td_q_crossings.csv` — the words-understood at which q reaches 0.25/0.5/0.75/0.9 per population, the child-level reading bare — and the VG20 duplicate `ds_td_q_vs_understood_vg20` are retired. Neither carried a caveat and the first used a development model for the DS side. The book's figure is the one of record; `compare_ds_td.py`, the deprecated shim that still called the retired figure "canonical", now delegates to `compare_ds_td_re.run_comprehension_matched`. `ds_td_spoken_vs_understood_vg20`, the same contrast in words rather than a ratio, already carried the caveat and stays.
- **The report book** does not narrate the DS/TD production ratio at all, so nothing there needed changing.

## Open

- A model-derived $E[q \mid U]$ — integrating the joint child-effect posterior through the understood likelihood — is the quantity the reader wanted and does not exist as an output. The empirical column is the honest substitute until it does. #233 records the same gap.
- VG20's reference child sits 54 words below the Down syndrome pool's median child at 38 months. Whether the population curves on the Down syndrome pages should be study-weighted — the reference child in the average _administration_ rather than the average _study_ — is a reporting decision not yet made, and it bears on every "typical child" statement those pages carry.
- The Down syndrome column at 300 words draws a third of its administrations from one study (`it_01`, 36 of 112); a per-study breakdown would say whether the 0.13 is a pool property or a study property.
