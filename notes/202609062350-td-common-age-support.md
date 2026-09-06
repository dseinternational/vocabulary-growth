# What the 200-row study threshold costs, and how thin the age support really is

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-06. Answers the **common age support** clause of [#240](https://github.com/dseinternational/vocabulary-growth/issues/240)'s study-threshold item, and re-measures the threshold's cost on the current loader rules. Neither needs a fit; both were open only because nobody had counted. The threshold sensitivity and the random-slope sensitivity, the item's other two clauses, still need fits and are not answered here.

## The threshold's cost, on the current frames

`min_study_observations = 200` drops any contributing study with fewer than 200 admissible rows, before the model sees them. Measured on the loaders' current rules — so after the source-level deduplication of 2026-08-23 and the Romance-language extension:

| Model | Eligible           | Retained           | Dropped                                                                                     |
| ----- | ------------------ | ------------------ | ------------------------------------------------------------------------------------------- |
| VG11  | 15 studies, 18,815 | 10 studies, 18,500 | 5 studies, 315 rows (1.7%): Armon-Lotem 8, Byers 68, Frank 37, OToole 62, Poulin-Dubois 140 |
| VG12  | 9 studies, 7,185   | 6 studies, 7,049   | 3 studies, 136 rows (1.9%): Byers 41, Frank 37, Poulin-Dubois 58                            |
| VG13  | 9 studies, 6,492   | 6 studies, 6,356   | 3 studies, 136 rows (2.1%): Byers 41, Frank 37, Poulin-Dubois 58                            |

The asymmetry the review named is confirmed and is the thing to keep in view: **a third of the study units buys under 2% of the rows**. A hierarchical model retains small groups through partial pooling, so the threshold is not needed to make the fit work; what it changes is the population the study-effect scale describes and how much information there is about between-study variance. That remains an open design question — it is the item's second clause, and needs a threshold-free refit to answer.

## Common age support

For each retained frame, how many distinct studies contribute at each integer age:

| Model | Ages | Minimum studies at any age | Median | Where the minimum falls            |
| ----- | ---: | -------------------------: | -----: | ---------------------------------- |
| VG11  |   23 |                      **4** |      7 | 8-11 months                        |
| VG12  |   18 |                      **1** |      4 | 25 months (Floccia alone, 36 rows) |
| VG13  |   11 |                      **3** |      4 | 18 months                          |

**VG11 and VG13 have no thin-support region.** Every reported age in VG11 rests on at least four studies and the typical age on seven; every reported age in VG13 rests on at least three. The review's concern about uneven overlap was raised across all three models, and on these two it does not bite — the retained studies' age _spans_ do differ (VG13's ByersHeinlein and Karousou stop at 15 months where Floccia and Thal start at 12), but the spans overlap enough that no age is carried by a single laboratory.

**VG12 is the model where it does bite, and only at the top of its range.** Support is four to six studies from 8 to 18 months, falls to **two** (Caselli and Floccia) for every age from 19 to 24, and to **one** (Floccia) at 25:

```
age   8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
n     4  4  4  4  6  6  6  5  4  4  3  2  2  2  2  2  2  1
```

That matters because 19-25 months is where the model has both study intercepts and no study-specific age slope: above 18 months, the shape of the comprehension trajectory is estimated from two laboratories, and its last reported month from one. Language rides along with it — Caselli is Italian, Floccia English — so the 19-24 band is also where the "language cannot be separated from dataset" limit is at its sharpest, since the band is _two_ datasets rather than six.

This is not extrapolation: `report_max_age_understood = 25` already stops the report where the observations stop, and the 25-month row is a real observed month. It is a statement about how much independent evidence stands behind the top of the reported range, which is a different thing and had not been quantified. VG12's Limits section now says so.

## What this does and does not settle

- **Settled**: common-support reporting, for all three models. It is quantified above, and the two models where it is not a problem can now say so rather than inheriting a family-wide caveat.
- **Not settled**: whether the 200-row threshold should be there at all. Answering that needs a threshold-free refit of at least VG11 (which loses the most studies) and a comparison of the study-effect scale and the trajectory against the registered fit. Deferred to the VM refit window.
- **Not settled**: whether intercept-only study effects are adequate. VG12's 19-24 band is the strongest argument for a random-slope sensitivity, because that is exactly where a study-specific slope and the population shape would be least distinguishable — and it is also the region with fewest studies to estimate a slope distribution from, which cuts the other way. It needs the fit, not more counting.

## Reproducing this

```bash
uv run python scripts/experiments/td_common_age_support.py
```

Reads the prepared DuckDB only — no trace, no fit, nothing written. It calls `du.load_data` with each definition's own `population`, `td_languages`, `max_age_months` and ceiling-exclusion settings, applies `du.filter_studies_by_min_obs` at `definition.min_study_observations`, and counts distinct `study` values per rounded integer age. It **asserts** that the rebuilt frames hold 18,500 / 7,049 / 6,356 rows — the counts `tests/test_kappa_conditional_calibration.py::test_frames_use_the_registered_language_scope` pins — so a frame that is not the model's own fails the run rather than producing a plausible table.

Related: [202608231537](202608231537-vg11-vg12-vg13-statistical-review.md) §3.5, [202609062330](202609062330-vg11-vg13-calibration-regenerated.md).
