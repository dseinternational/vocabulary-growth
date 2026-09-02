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

The curves agree at 300 words. The children do not: a typically developing child who understands 300 words typically speaks 27% of them, a child with Down syndrome 13% — half — and both are well below the 0.4 the curves show. The distributions are not similar either; the Down syndrome one sits lower throughout its interquartile range. The gap between curve and children grows with the level and is widest in the Down syndrome pool.

## Why they diverge

The curve's value at 300 is the population ratio at the age when the population _median_ reaches 300 words: 21 months for VG21, 47 months for VG22. Conditioning on 300 words instead selects every child who reached it at _any_ age, and the ratio rises with age, so the selected children are younger than that crossing age (median 17 and 38 months) and speak a smaller share. The docstring predicted the conditional would sit _above_ the curve because a positive $\rho_{uq}$ lifts the ratio of comprehension-advanced children; that effect exists, but the age selection dominates it, and by more where comprehension grows slowly over a long window — which is the Down syndrome pool, hence the larger gap there. At low levels (50–100 words) there is little age heterogeneity among the children who reach them and the two columns agree.

## What is and is not confident

As a **population-stage statement** the agreement is real and reasonably well supported: both curves are estimated with narrow intervals that overlap, and each sits within about 0.05–0.08 of the empirical median ratio of all children at the corresponding age (0.35 in both pools). The typically developing value is at the very end of its curve — VG21's window is capped at 22 months and the curve's x-axis stops at 328 words — so it is an edge estimate rather than a mid-range one. As a **statement about children at a milestone** it is not supported, and the pages should not let a reader make it.

## What changed

- The three captions now say what the axis is and that the figure is not the conditional share.
- A new shared block, `report_cells.render_conditional_production_check`, renders beneath the figure on all eight templates that carry it (VG10, VG19, VG20 and VG22 through the shared body). It states the reading, then sets the curve beside the observed children at each level the curve covers — count, median ratio, interquartile range, median age — from the hash-verified frame, and says that a comparison between populations at a comprehension milestone must be made in that column.

## Follow-through into the DS/TD comparison

Three surfaces carried the same curve into the cross-population contrast.

- **The comparison book's "Comprehension-matched production ratio"** (`compare_ds_td_re.py comprehension`, VG20 vs VG13) already stated the #233 reading, but tabulated N = 50/100/150 and concluded "converging by N ≈ 200" — true of the population curves and opposite to the children. `run_comprehension_matched` now also writes `ds_td_comprehension_q_observed.csv` from the two hash-verified frames through the same `observed_production_ratio_at_levels` the model pages use, and the book's table runs to N = 300 with the observed children beside the curve in each population: at 300 words, curves near 0.4 in both, children 0.23 (TD, VG13's 8–18-month frame, 381 children) and 0.13 (DS, 97 children). The prose now says the curves converge and the children do not, and that a statement about children at a milestone is made from the children columns.
- **`compare_models.ds_td_q_vs_understood`**, described in its docstring as a "headline matched-comprehension q overlay" (DS VG09 vs TD VG13, VG07 dashed), together with `ds_td_q_crossings.csv` — the words-understood at which q reaches 0.25/0.5/0.75/0.9 per population, the child-level reading bare — and the VG20 duplicate `ds_td_q_vs_understood_vg20` are retired. Neither carried a caveat and the first used a development model for the DS side. The book's figure is the one of record; `compare_ds_td.py`, the deprecated shim that still called the retired figure "canonical", now delegates to `compare_ds_td_re.run_comprehension_matched`. `ds_td_spoken_vs_understood_vg20`, the same contrast in words rather than a ratio, already carried the caveat and stays.
- **The report book** does not narrate the DS/TD production ratio at all, so nothing there needed changing.

## Open

- A model-derived $E[q \mid U]$ — integrating the joint child-effect posterior through the understood likelihood — is the quantity the reader wanted and does not exist as an output. The empirical column is the honest substitute until it does. #233 records the same gap.
- The Down syndrome column at 300 words draws a third of its administrations from one study (`it_01`, 36 of 112); a per-study breakdown would say whether the 0.13 is a pool property or a study property.
- The comparison book's children columns come from VG13's frame on the typically developing side (8–18 months, 381 children at 300 words), not VG21's (8–22 months, 464); the two agree to within 0.04 at 300 words, and the book names its source.
