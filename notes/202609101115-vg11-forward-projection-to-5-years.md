# A forward projection of VG11's typically-developing spoken trajectory to 5 years

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-10. **Question:** the study owner asked for an approximate forward projection of the typically-developing words-spoken trajectory to about 5 years, as the companion to yesterday's Down syndrome projection to 11 years ([`notes/202609091900`](202609091900-vg20-forward-projection-to-11-years.md)). **Evidence:** the `rep` fit of VG11, the TD reference for the typically-developing spoken trajectory. **Record:** `scripts/experiments/vg11_forward_projection.py`, which writes the tables and figure under `<output-root>/experiments/vg11-forward-projection/`. This is an exploratory extrapolation, not a model output, and nothing in the report or the fit pipeline reads it.

**The short answer, and the caveat that comes with it.** The pooled projection puts the typically-developing median child at **552 words of 810 at 5 years, 89% [475, 684]**, and an individual child at **570, 89% [228, 760]**. But the projection's own internal disagreement is larger than those intervals suggest, and the model's own mean function says 810 — the whole checklist — by 4 years. The honest reading is that this extrapolation **brackets** the 5-year count between about 480 and the ceiling rather than locating it, and that the bracket is that wide because a typically-developing child of 5 has left an early-vocabulary checklist behind. §5 gives the reading that _is_ well supported: the age at which the typically-developing median child reaches a given count.

## 1. Why this is a weaker extrapolation than the Down syndrome one

The two projections use the same construction and are not equally trustworthy, for one structural reason: **where the data stop relative to the shape of the curve.**

VG20's window ends at 84 months with the median child at 0.55 of the reference form, gaining about 3 spoken words a month and slowing: the curve is well onto its plateau, so the fitted families are constrained on both sides of the shoulder and mostly argue about how flat that plateau is. VG11's frame ends at **30 months**, where the median child is at 0.53 and still gaining **24 words a month** — six months past its peak rate of 39 words a month at 24 months, so the deceleration has begun but has barely got anywhere. The asymptote is therefore set by how those last six months of curvature are read, and the 30-month rate is itself the least certain point on the rate curve (89% interval [16.5, 29.1] words a month, against [38.3, 40.9] at the peak). Nothing in the data locates where it levels off.

The frame itself is not thin at its top — 923 rows from 921 children and 7 studies at exactly 30 months, against VG20's 12 rows above 7 years — so this is not the Down syndrome problem of a trend running unchecked past a handful of observations. It is the opposite problem: dense data that stop early, on the part of the curve that says least about what comes next. VG11's stored grid stops at 30 months and there is no fitted number above it to quote or to argue with, which is why the model's own mean function is reported here as a reference line instead.

## 2. Method

Identical in construction to the Down syndrome harness, on one outcome.

1. Read the stored population curve on the 8–30 month plot grid (`p_plot`, at zero study and child effects, i.e. the median child), the between-child scale `tau_subject` and dispersion `kappa_plot`. 2,000 draws, every 18th of the 36,000.
2. For each draw, fit three saturating growth families to the model's own curve over 12–30 months: logistic, Gompertz and Hill (log-logistic), each with a free asymptote below the ceiling. Fit quality over the window is good — median RMSE 0.0009 (Hill), 0.0036 (logistic) and 0.0039 (Gompertz) on the proportion scale, 0.7 to 3.1 words — which is the point worth noticing: three families that describe the observed window to within a word or three disagree by 173 words at 5 years.
3. Above 30 months, continue each family from the model's own value there — a constant shift on the logit scale, so the projection is continuous and stays bounded. Families are pooled with equal weight.
4. Three levels are reported. **Population median child** carries posterior and extrapolation uncertainty about the curve. **Individual children** adds `tau_subject` (1.04 logits) on the logit scale, the persistent between-child differences. **One administration for one child** additionally draws the Beta-Binomial count, which is the estimand VG11's reported `Y_*` columns carry; above the window `kappa` is continued on the model's own exact functional form, `kappa_min + exp(a + b z)`, recovered per draw from the stored grid. It decays from 10.3 at 30 months toward `kappa_min` = 5.9, reaching 7.0 at 3 years and 5.9 at 5, so the continuation is bounded.

Inside the window the predictive level reproduces VG11's reported `Y_*` intervals to within a word or two — 231 [34, 574] at 24 months against the report's 232 [35, 572], 343 [63, 682] against 344 [64, 684], and 429 [91, 743] at 30 months against 430 [91, 744] — which is what checks the construction.

Intervals are equal-tailed. Everything is on the 810-word reference scale at zero study effect, the same footing as the report's own curve; see §4 on what that means here.

## 3. Results

### 3.1 Population median child

Posterior and extrapolation uncertainty about the curve itself. Words of 810; median with 50%, 75% and 89% intervals. The 30-month row is VG11's own posterior.

| Age   | Spoken | 50%     | 75%     | 89%     |
| ----- | ------ | ------- | ------- | ------- |
| 2.5 y | 430    | 426–434 | 423–437 | 421–440 |
| 3 y   | 512    | 484–547 | 474–561 | 468–570 |
| 3.5 y | 538    | 491–603 | 481–624 | 474–637 |
| 4 y   | 547    | 492–627 | 481–651 | 475–667 |
| 4.5 y | 550    | 492–636 | 481–662 | 475–679 |
| 5 y   | 552    | 492–640 | 481–667 | 475–684 |

### 3.2 Individual children

Adds the model's between-child spread: the range a child drawn from the population would fall in, not the uncertainty about the average.

| Age   | Spoken | 50%     | 75%     | 89%     |
| ----- | ------ | ------- | ------- | ------- |
| 2.5 y | 429    | 290–563 | 206–638 | 146–693 |
| 3 y   | 516    | 375–632 | 278–692 | 199–733 |
| 3.5 y | 550    | 409–659 | 305–712 | 218–749 |
| 4 y   | 563    | 421–671 | 314–723 | 225–756 |
| 4.5 y | 568    | 425–676 | 318–727 | 227–759 |
| 5 y   | 570    | 427–678 | 320–729 | 228–760 |

### 3.3 One administration for one child

Adds the occasion-level Beta-Binomial dispersion as well, which is the estimand the report's `Y_*` columns carry and the one to compare against an observed count.

| Age   | Spoken | 50%     | 75%     | 89%     |
| ----- | ------ | ------- | ------- | ------- |
| 2.5 y | 429    | 265–589 | 162–679 | 91–743  |
| 3 y   | 523    | 341–675 | 217–749 | 121–790 |
| 4 y   | 574    | 382–720 | 248–779 | 141–804 |
| 5 y   | 584    | 388–730 | 249–784 | 140–806 |

Child effects and dispersion are both constant-or-decaying on the logit scale while the median approaches the ceiling, so the upper tail compresses and the lower tail does not. At 5 years a quarter of children are still projected below about 430 words on this checklist — which is the projection's least believable feature and the one §4's last paragraph is about.

## 4. What the bands are made of

**Family — this is most of it.** At 5 years the three families put the median child at 485 (logistic), 552 (Hill) and 658 (Gompertz) words, from fitted asymptotes of 0.59, 0.68 and 0.82 of the ceiling. That 173-word spread is wider than the pooled 89% band (475–684) and is a statement about extrapolation form, not a measurement. Above about 3.5 years every family is flat: each has committed to its asymptote within a year of the last observation, and which asymptote is not something the data chose.

**Fit window — and this is worse than in the Down syndrome case.** Fitting to 27 months instead of 30 — discarding three months of a curve the data support densely — leaves the median almost unchanged (550 against 552 at 5 years) but moves the 89% upper limit from 684 to **802**, because Gompertz's fitted asymptote goes to the ceiling (1.000, against 0.820 on the full window). Three months of curve is the difference between "about two-thirds of the checklist" and "all of it". Read the 89% upper limits in §3 as arbitrary to within about that much; the medians are the more stable half of the interval.

**The model's own mean function — the upper bracket.** VG11's mean is a logit-linear age trend with an HSGP deviation, and both parts are recoverable exactly: the trend is `intercept + slope * z`. Continued from the model's own value at 30 months at its fitted logit-linear rate, it reaches 719 words at 3 years, 795 at 3.5 and the full 810 by 4.5. Taken as the mean function itself — where the curve goes if the HSGP deviation reverts to zero above the window, as it does past the data in the Down syndrome models — it gives 775 at 3 years and 810 by 4.5. Either reading exhausts the checklist inside the projected range, and both sit above every family. This is the same phenomenon VG20's unchecked climb above 8 years is, and here it is _not_ obviously the wrong answer.

**Dispersion.** Holding `kappa` at its 30-month value instead of continuing the model's own decay narrows the predictive 89% band at 5 years from [140, 806] to [180, 796]. It does not affect the other two levels at all.

**The instrument, and why the low tail is the projection's weakest part.** A typically-developing 5-year-old knows several thousand words and would be at or near the ceiling of any early-vocabulary checklist. The typically-developing pool's spoken counts come from 680-item Words & Sentences forms scored against the 810-word reference, so 680/810 = 0.84 is the largest proportion any observation in the frame can express — and the 89% upper limits above 3 years already sit at or above 680, i.e. outside what the instrument the data come from could record. Saturation is not yet a problem _inside_ the window (25 of 18,500 rows are at the 680 ceiling, 0.14%, and 3 of the 923 rows at 30 months), so it does not distort the fit; it distorts the extrapolation. The projection's claim that a quarter of children at 5 years are below 430 checklist words is therefore best read as an artefact of carrying a constant logit-scale between-child spread past the point where the measure discriminates, not as a statement about children.

**The study scale.** Like the report's own curve, everything here is at zero study effect. The pool's row-weighted mean study effect is +0.16 logits, which at 30 months is the difference between 430 words and 463 (the raw pool median at 30 months is 483). A projection for a study-average rather than a study-median child would sit roughly 30 words higher throughout.

## 5. The reading that is well supported: age at a given count

The count at a fixed age is badly determined above the window, but the **age at a fixed count** is not, for the same reason: the curve is flat there, so a horizontal reading is stable where a vertical one is not. Below is the age at which the population median child reaches a count, under the pooled families and under the model's own logit-linear continuation — the two ends of the bracket.

| Words | Pooled families | Logit-linear continuation |
| ----- | --------------- | ------------------------- |
| 300   | 25.7 mo         | (inside the window)       |
| 430   | 30.0 mo         | (inside the window)       |
| 525   | 38.1 mo         | 31.5 mo                   |
| 600   | beyond 5 y      | 32.9 mo                   |
| 680   | beyond 5 y      | 34.8 mo                   |

The 525-word row is the one worth quoting, because it is the Down syndrome projection's answer at 11 years: VG20 puts the median child with Down syndrome at 524 spoken words of 810 at 11 years (89% [445, 612]). The typically-developing median child reaches that count at **somewhere between about 2.6 and 3.2 years** — a gap of roughly eight years on this checklist, and one that both extrapolations agree on despite disagreeing about everything above it, because both are reading an age off a curve rather than a count. Note that this compares VG11 (typically developing, spoken, English and Romance pool) with VG20 (Down syndrome, spoken) on the shared 810-word scale, not through the matched-comprehension comparison `compare_ds_td_trajectories.py` runs, which uses VG21 and is confined to ages both models support.

## 6. Not offered: a typically-developing comprehension projection

There is no companion for words understood. The typically-developing pool has **no comprehension observations above 25 months** (#228), which is why VG04 and VG12 report comprehension only to 25 — so a comprehension projection to 5 years would start from a window ending 5 months earlier, on an even steeper part of the curve, with no shoulder and no ceiling reference. Everything §1 and §4 say about the spoken projection would hold with less to constrain it. It can be run on request; it should not be quoted.

## 7. Reproduce

```bash
uv run python scripts/experiments/vg11_forward_projection.py
uv run python scripts/experiments/vg11_forward_projection.py --fit-hi 27 --draws 1000 --tag _fit27 --no-figure
uv run python scripts/experiments/vg11_forward_projection.py --kappa-mode hold --draws 1000 --tag _kappahold --no-figure
```

The harness validates the VG11 fit through `vocab_growth.fit_consumers` before reading it (registered definition, raw-data fingerprint, exact prepared-frame hash), so a refit or a data-rule change stops it until `--allow-stale-fit` is passed. Outputs land under `<output-root>/experiments/vg11-forward-projection/`: `projection_intervals.csv` (monthly, all levels), `projection_family_medians.csv`, `projection_fit_diagnostics.csv` and `vg11_projection_to_5y.png`.
