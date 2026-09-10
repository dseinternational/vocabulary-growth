# A forward projection of VG20's understood and spoken trajectories to 11 years

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

**Date:** 2026-09-09. **Question:** the study owner asked for an approximate forward projection of the Down syndrome words-understood and words-spoken trajectories to about 11 years, with 50%, 75% and 89% credible intervals, from the reporting model. **Evidence:** the 2026-09-08 `rep` fit of VG20, the model of record for the Down syndrome joint understood + spoken estimands (0 divergences, maximum R-hat 1.003, minimum ESS 1,713). **Record:** `scripts/experiments/vg20_forward_projection.py`, which writes the tables and figure under `<output-root>/experiments/vg20-forward-projection/`. This is an exploratory extrapolation, not a model output, and nothing in the report or the fit pipeline reads it.

## 1. Why the fitted model cannot simply be read past 8 years

VG20's population curve is stored on a grid to 115 months, so a number exists at 9.6 years. It is not evidence. The understood mean is a logit-linear age trend anchored at 24 and 84 months, with an HSGP deviation that is anchored to zero at 54 months and reverts to the trend a few length-scales past the last observation; the production-ratio mean is levelled off above 84 months for exactly this reason (`notes/202608042030-q-mean-extrapolation.md`). Above 7 years the frame holds 12 comprehension and 49 spoken rows, above 8 years 4 and 12, and 57 of the 84 rows above 78 months are `uk_01`, whose study effects sit at −0.20 logits on understood and +0.52 on the production ratio. So above about 8 years the stored curve is the trend running with nothing to correct it, and it climbs accordingly: 575 [517, 626] understood words at 96 months, 670 [596, 729] at 108 and 733 [666, 775] at 115, against 469 at 72. The report already stops quoting comprehension at 72 months and spoken at 90 (`notes/202608221200-reporting-source-by-quantity.md`).

## 2. Method

The projection is built from the part of the curve the data support and extended by assumption, with the assumption's spread carried into the bands.

1. Read the stored population curves on the plot grid — `p_u_plot` (understood proportion of 810) and `q_plot` (production ratio), both at zero study and child effects, i.e. the median child — and the between-child scales `tau_subj_u`, `tau_subj_q` and their correlation `rho_uq`. 2,000 draws, every 18th of the 36,000.
2. For each draw, fit three saturating growth families to the model's own curve over 12–84 months (84 is the high slope anchor): logistic, Gompertz and Hill (log-logistic), each with a free asymptote below the ceiling. The target is a smooth curve, so each fit is the projection of that draw onto the family; the spread comes from the posterior and from the family, not from fit noise. Fit quality is adequate for the purpose: median RMSE over the window is 0.010–0.016 on the proportion scale (8–13 words) for understood and 0.012–0.015 for the ratio, the logistic fitting worst because the trajectory is not symmetric about its inflection.
3. Above 84 months, continue each family from the model's own value at 84 — a constant shift on the logit scale, so the projection is continuous there and stays bounded. Spoken is 810 × p_u × q, crossing every understood family with every ratio family (nine combinations). Families are pooled with equal weight.
4. Child-level bands add the model's own correlated child effects on the logit scale. Inside the window this reproduces the report's subject-marginal intervals exactly (understood at 72 months: median 470, 89% [219, 679]; spoken: 336, [78, 616]), which checks the construction.

Intervals are equal-tailed. Everything is on VG20's 810-word reference scale; "spoken" is VG20's outcome — the model does not carry signing, so this is not the produced (spoken ∪ signed) count VG15 reports.

## 3. Results

### 3.1 Population median child

Posterior and extrapolation uncertainty about the curve itself. Words of 810; median with 50%, 75% and 89% intervals.

| Age  | Understood | 50%     | 75%     | 89%     | Spoken | 50%     | 75%     | 89%     |
| ---- | ---------- | ------- | ------- | ------- | ------ | ------- | ------- | ------- |
| 6 y  | 469        | 455–483 | 446–492 | 437–501 | 360    | 346–374 | 336–384 | 327–392 |
| 7 y  | 525        | 509–544 | 498–555 | 486–567 | 442    | 423–458 | 410–469 | 398–480 |
| 8 y  | 558        | 536–581 | 521–596 | 506–608 | 478    | 456–500 | 439–515 | 425–529 |
| 9 y  | 579        | 550–610 | 531–629 | 515–642 | 501    | 473–529 | 454–548 | 437–565 |
| 10 y | 592        | 557–633 | 537–655 | 519–669 | 515    | 482–549 | 461–572 | 442–592 |
| 11 y | 601        | 561–650 | 539–677 | 520–691 | 524    | 488–565 | 465–591 | 445–612 |

The rows to 7 years are VG20's own posterior. The production ratio projects to about 0.87 by 9 years and holds there, so the comprehension–production gap of roughly 75 words persists in the projection rather than closing.

### 3.2 Individual children

Adds the model's between-child spread: the range a child drawn from the population would fall in, not the uncertainty about the average.

| Age  | Understood | 50%     | 75%     | 89%     | Spoken | 50%     | 75%     | 89%     |
| ---- | ---------- | ------- | ------- | ------- | ------ | ------- | ------- | ------- |
| 6 y  | 470        | 357–571 | 282–632 | 219–679 | 336    | 206–467 | 131–549 | 78–616  |
| 7 y  | 527        | 418–619 | 338–671 | 268–710 | 416    | 281–538 | 193–608 | 124–664 |
| 8 y  | 558        | 453–645 | 373–692 | 299–726 | 454    | 318–571 | 225–636 | 148–685 |
| 9 y  | 580        | 477–663 | 397–706 | 320–737 | 478    | 342–592 | 245–653 | 163–699 |
| 10 y | 596        | 494–676 | 413–717 | 334–746 | 494    | 357–606 | 258–665 | 173–708 |
| 11 y | 608        | 506–687 | 422–726 | 342–753 | 505    | 367–616 | 266–674 | 179–716 |

Child effects are constant on the logit scale, so as the median approaches the ceiling the upper tail compresses and the lower tail does not: at 11 years a quarter of children are still projected below about 500 understood and 370 spoken words on this checklist.

## 4. What the bands are made of

**Family.** At 11 years the three families put the median child at 558 (logistic), 606 (Gompertz) and 655 (Hill) understood words, and 473, 535 and 572 spoken. That 100-word spread is most of the pooled 89% band, so the band is a statement about extrapolation form, not a measurement. The fitted asymptotes say the same thing from the other side: 0.67, 0.76 and 0.96 of the ceiling for understood under the three families.

**Fit window.** Fitting to 72 months instead of 84 lowers the 11-year medians to 574 understood [481, 688] and 484 spoken [392, 601]; fitting to 96 raises them to 624 [545, 699] and 541 [466, 614] (1,000 draws each). Read the medians as carrying roughly ±40 words on top of the intervals in §3.

**The model's own climb.** VG20's stored curve reaches 733 understood and 618 spoken at 115 months, above the pooled 89% bands from about 9 years. The four comprehension observations above 8 years (758, 455, 555 and 676 words, all `uk_06`, whose understood effect is +0.45 logits) sit nearer the projection than the model's climb once the study effect is taken off, but four rows settle nothing. The figure draws the model's curve as a dashed reference.

**The instrument.** Growth is nearly flat after about 9 years in every family. Part of that is real deceleration and part is an 810-word early-vocabulary checklist saturating; it says nothing about vocabulary measured by any other instrument.

## 5. Reproduce

```bash
uv run python scripts/experiments/vg20_forward_projection.py
uv run python scripts/experiments/vg20_forward_projection.py --fit-hi 72 --draws 1000 --tag _fit72
uv run python scripts/experiments/vg20_forward_projection.py --fit-hi 96 --draws 1000 --tag _fit96
```

The harness validates the VG20 fit through `vocab_growth.fit_consumers` before reading it (registered definition, raw-data fingerprint, exact prepared-frame hash), so a refit or a data-rule change stops it until `--allow-stale-fit` is passed. Outputs land under `<output-root>/experiments/vg20-forward-projection/`: `projection_intervals.csv` (monthly, all three levels), `projection_family_medians.csv`, `projection_fit_diagnostics.csv` and `vg20_projection_to_11y.png`.
