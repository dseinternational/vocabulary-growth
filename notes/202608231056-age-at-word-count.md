# Ages by which a share of children reach a word count

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Computed from the VG20 and VG19 `rep` traces of the 2026-08-22 Down syndrome refit (VG19 at the high-tune sampling override, see §6). The VG20 table is a candidate reported quantity; the VG19 table is a **sensitivity comparison only** — VG19 is not the model of record and [202608220748](202608220748-vg19-individual-trajectories.md) §6.6 records its individual-level tables as internal.

## 1. The question, and the quantity it is not

"By what age would we expect 50% / 75% / 90% of children with Down syndrome to have learned N words?" This is a **between-child** quantity: the distribution, across children, of the age at which a child reaches N.

It is not what `scripts/time_to_milestone.py` reports, and the difference is not presentational. That script gives the age at which the **population trajectory** reaches N, with an interval expressing posterior uncertainty about that one age. Its own docstring is explicit that the interval "is not a spread across individual children (that would need new-child posterior-predictive draws)". Reading its HDI as a spread across children would understate the spread by roughly an order of magnitude — at 150 understood words the population milestone is 27.7 months with an 89% HDI of about ±1.2 months, while the 50th-to-90th percentile of _children_ spans 27.7 to 48.3 months.

Both numbers are legitimate; they answer different questions. The population milestone answers "when does the average child reach N"; this note answers "how long do we wait for most children".

## 2. How it is computed

A child's curve is the population curve displaced by that child's own random effect, so the estimand is a threshold crossing on a displaced curve, propagated over posterior draws.

**Understood, under VG20, is exact.** VG20 gives each child a constant offset `z_u` on the logit of understood, so the child reaches N words exactly where `f_u(a) = logit(N/810) - z_u`. Crossing age is monotone decreasing in `z_u`, so the child at the X-th percentile of crossing age is the child at the (1-X)-th percentile of `z_u`, and no simulation is needed.

**Everything else is simulated.** Spoken is `810 * expit(f_u + z_u) * expit(h + z_q)`, which depends on two effects at once and is not monotone in a single index; VG19's offsets vary with age (`b0 + b1 * D(a)`, `D(a) = (a - 36)/12`) and so are not a single index either. For each posterior draw a synthetic cohort of 2,000 children is drawn from the fitted covariance, the empirical percentile of crossing age is taken within that draw, and posterior uncertainty is the spread of that percentile across draws.

The simulator is checked against the closed form: run on VG20 understood it reproduces the exact answer to 0.1 months at every cell.

Two deliberate choices:

- **The estimand is the child's latent vocabulary, not a questionnaire score.** The Beta-Binomial observation noise (`kappa`) is measurement scatter around a child's own trajectory. Including it would answer "what would this child score on one administration", which is a different and less useful question.
- **Censoring is reported, never extrapolated.** Crossing ages beyond a quantity's reporting cap — 72 months for understood per `report_max_age_understood`, 90 for spoken under the per-quantity policy of [202608221200](202608221200-reporting-source-by-quantity.md) — are reported as outside the window. Draws that never reach the target are carried as `+inf` through the median rather than dropped; dropping them would pull each figure earlier, which is exactly the bias that makes a milestone look easier than the model says.

## 3. VG20 — the model of record

Median and 89% HDI, in months, on the 810-word reference scale.

**Words understood** (window ends at 72 months)

| words | 50% of children   | 75% of children   | 90% of children   |
| ----: | :---------------- | :---------------- | :---------------- |
|    50 | 18.1 [17.5, 18.7] | 21.4 [20.6, 22.2] | 25.2 [24.0, 26.4] |
|   150 | 27.7 [26.5, 28.9] | 37.5 [34.9, 40.4] | 48.3 [45.6, 51.1] |
|   300 | 47.1 [44.9, 49.4] | 61.0 [55.8, 66.6] | not by 72 mo      |
|   500 | not by 72 mo      | not by 72 mo      | not by 72 mo      |
|   750 | not by 72 mo      | not by 72 mo      | not by 72 mo      |

**Words spoken** (window ends at 90 months)

| words | 50% of children   | 75% of children   | 90% of children   |
| ----: | :---------------- | :---------------- | :---------------- |
|    50 | 38.7 [37.6, 39.8] | 46.0 [44.7, 47.7] | 54.8 [52.0, 57.2] |
|   150 | 50.2 [48.5, 51.9] | 62.8 [59.2, 66.3] | 78.3 [72.4, 83.8] |
|   300 | 68.6 [64.1, 72.7] | 87.2 [78.6, >90]  | not by 90 mo      |
|   500 | not by 90 mo      | not by 90 mo      | not by 90 mo      |
|   750 | not by 90 mo      | not by 90 mo      | not by 90 mo      |

The 50% understood column reproduces the population median-of-crossings exactly (18.1, 27.7, 47.1), as it must — a median child's offset is zero. The spoken 50% column does **not** match its population crossing exactly (38.7 against 38.3 at 50 words), and should not: spoken depends on two correlated offsets, so the median child is not the child at (0, 0).

**500 and 750 words are outside the reporting window at every percentile.** The population median reaches 500 understood at 78.8 months and 500 spoken at 98.8, both past their caps. 750 is 93% of the 810-item checklist, where the instruments no longer discriminate.

## 4. VG19 — where the two models disagree

Comprehension agrees; production does not.

**Understood** — VG19's tail is marginally _earlier_, not later (46.2 against 48.3 months at 150 words, 90th percentile). `rho_u` is negative (−0.219), so lower-starting children gain slightly faster and the spread compresses.

**Spoken** — the tails separate materially:

| words | percentile | VG20              | VG19                  |
| ----: | ---------- | :---------------- | :-------------------- |
|    50 | 90%        | 54.8 [52.2, 57.8] | **64.8 [55.3, 76.2]** |
|   150 | 75%        | 62.9 [58.9, 66.3] | **67.9 [62.3, 74.0]** |
|   150 | 90%        | 78.5 [73.0, 84.1] | **not by 90 mo**      |
|   300 | 75%        | 87.3 [78.7, >90]  | **not by 90 mo**      |

Every 50% figure agrees to within a month across both models and both outcomes.

The mechanism is `tau_q1 = 0.639` with `rho_q = +0.472`: under VG19 the production-ratio spread compounds with age, so children who start slower fall further behind and the late tail stretches. VG20 cannot express this — its band is the same width at 24 months and at 84. This is the dissociation [202608220748](202608220748-vg19-individual-trajectories.md) describes, expressed as ages rather than correlations.

**Consequence for reporting.** The 50% figures are robust to the choice between these two child-effect structures. The 75% and especially the 90% **spoken** figures are not, and VG20's are the optimistic ones. If these numbers are used to set expectations about how long spoken vocabulary takes for slower-developing children, that gap is the finding, not a technicality.

Neither model is nested in the other — VG20 correlates the outcomes but freezes the spread in age, VG19 grows the spread but forces the outcomes independent — so this cannot be settled by preferring the more general model. It is the same open question [202608220748](202608220748-vg19-individual-trajectories.md) §4 records.

## 5. Reproducing

```bash
uv run python scripts/experiments/vg20_age_at_word_count.py
uv run python scripts/experiments/vg20_vg19_age_at_word_count.py
```

Both read the fitted traces through `vocab_growth.comparison` and write to `<output-root>/comparisons/`. The first carries the exact closed form for VG20 understood and is the reference the second is checked against.

## 6. Provenance

The traces are from the 2026-08-22 Down syndrome refit at `--config rep`. VG19 missed the convergence gate at the tier default on one parameter (`tau_subj_u_1`, R-hat 1.01020 against 1.01, ESS 376 against 400) and was refitted at `tune=12000, draws=8000, target_accept=0.99`, where it cleared comfortably (R-hat 1.0063, ESS 1,672). Its recovered scales match [202608220748](202608220748-vg19-individual-trajectories.md) closely — `tau_u0` 0.750 against 0.751, `tau_u1` 0.179 against 0.176, `rho_u` −0.219 against −0.219, `tau_q1` 0.639 against 0.640 — which is independent evidence that the refit reproduced the earlier fit.

VG20 carries a soft-tier convergence caveat from this run (1 divergent transition in 36,000 draws), disclosed through `convergence_caveats.csv`. Everything above inherits it.
