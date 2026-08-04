# The mean extrapolates past the data: clamping above the high anchor

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Analysis and implementation note, 2026-08-04. **Implemented**: `trend_and_gp` gains a one-sided `clamp_above_hi` option, exposed as the model-definition field `clamp_mean_above_hi_anchor` and switched on for the eight Down syndrome joint models (VG05, VG07–VG10, VG14–VG16). It is a graph change, so every one of those models needs a refit; only VG10 has been refitted (§7).
>
> **This note corrects §3 and §5 of [202608041730](202608041730-ds-spoken-q-trajectory-prior.md).** That note concluded the residual the `q` mean cannot carry is an S-shape in the developmental curve, and that fixing it needed a mean form able to represent an S. Scored against the observed data rather than the fitted curve, that is wrong — see §2. The real defect is unbounded extrapolation above the high anchor, which is a different problem with a much smaller fix.

> [!IMPORTANT]
> **Validated (§7).** Two implementations were tried, a hard `min` and the soft form now in the tree. Both halve the GP's correction at the top of the domain and lower both GP amplitudes with their contraction rising, which confirms the mechanism; the soft form additionally keeps the fitted curves monotone, which the hard one did not. An apparent convergence cost — VG10 losing a clean gate — **is withdrawn**: six fits across three sampler seeds show the clean gate at seed 47 was luck (the configuration without it passes 1 of 3), the two arms' R-hat and ESS ranges overlap, and the clamped arm has zero divergences in all three fits against 0, 1 and 4 without. VG10 does not reliably meet the gate at `test` either way, which is a pre-existing property of the model.

## 1. Where the previous note went wrong

[202608041730](202608041730-ds-spoken-q-trajectory-prior.md) §3 scored candidate mean forms by the residual they leave against the **fitted** `logit(q)` curve, evaluated on the fourteen query ages with equal weight. Both choices were wrong. The fitted curve already contains the GP's correction to the mean, so scoring against it measures the consequence of the defect rather than the shape of the data; and equal weighting gives 12 months — where `q` is essentially unobserved — the same say as 36 months, where 134 administrations sit.

## 2. What the observed data actually say

Scored against the directly observed `spoken / understood` ratio, in three-month bands from 12 to 78 months with at least eight rows apiece, weighted by band size:

| mean form                          | weighted RMS | max abs residual |
| ---------------------------------- | -----------: | ---------------: |
| **linear in age (current)**        |    **0.214** |            0.555 |
| linear in log(age)                 |        0.285 |            0.481 |
| linear in sqrt(age)                |        0.222 |            0.485 |
| quadratic in age                   |        0.207 |            0.494 |
| best three-anchor (knots 12/48/84) |        0.193 |            0.386 |

A straight line on the logit scale is already an adequate description of `logit(q)` where there are data. Linear in `log(age)` is **worse**, not better. The best three-anchor piecewise-linear alternative — which is what `tent_and_gp` would give — improves the weighted RMS by about 10%, and adds a parameter and a prior to do it. That does not justify a graph change, and §3 of the previous note should not be relied on.

## 3. The actual defect

The Down syndrome GP domain runs to **115 months** while `slope_anchors` are 24 and 84. Above 84 months — a quarter of the domain, and 3.2% of spoken rows — the mean is extrapolation that no prior constrains. On the logit scale a line that climbs 4.6 logits between the anchors keeps climbing, and saturates:

| age (months) | `q` mean alone | `q` realised | GP pull (logit) | `u` mean alone | `u` realised | `u` GP pull |
| ------------ | -------------: | -----------: | --------------: | -------------: | -----------: | ----------: |
| 48           |          0.465 |        0.484 |       **+0.08** |          0.372 |        0.386 |       +0.06 |
| 72           |          0.843 |        0.723 |           −0.72 |          0.692 |        0.571 |       −0.53 |
| 84           |          0.930 |        0.684 |           −1.82 |          0.815 |        0.720 |       −0.54 |
| 115          |      **0.993** |        0.842 |       **−3.29** |          0.962 |        0.908 |       −0.93 |

Across the posterior, **P(`q` mean alone > 0.99) = 0.896 at 115 months** — the mean asserts, with 90% probability, that a child speaks more than 99% of the words they understand. Nothing in the data says so: only five rows in the whole frame carry both outcomes above 78 months.

Two things follow. First, the GP is **idle where the data are** (+0.08 logits at 48 months) and spends its amplitude almost entirely correcting the mean's asymptote where they are not. That is a better explanation of why `eta_q` was pinned at `HalfNormal(0.20)` than the S-shape story in the previous note's §5 — the amplitude was never needed for developmental curvature. Second, the understood trajectory has the same defect, about three times milder, so this is a property of the parameterisation rather than of `q`.

## 4. The change

`trend_and_gp` gains `clamp_above_hi`. With it on, the mean levels off above the high anchor:

```text
intercept + slope * z_eff,    z_eff = sb_z - softplus(beta * (sb_z - z)) / beta
```

and the GP's nuisance basis becomes `[1, z_eff]` — the same coordinate, because the GP must be orthogonalised against what the mean can actually express. Orthogonalising against `[1, z]` while the mean uses `z_eff` would leave the GP free to mimic a slope change above the high anchor, which is exactly the aliasing the anchoring exists to remove.

### Why a soft minimum rather than `min(z, sb_z)`

The first implementation used a hard `pt.minimum`. It is continuous but its derivative jumps at the anchor, and the fitted curve inherits an elbow: in the VG10 refit (§7) understood ran 591, 595, 595, 595, 596 words across 83.7–87 months before resuming, and **spoken went briefly non-monotone**, peaking at 428.6 words at 84.3 and dipping to 426.6 at 85.6. A vocabulary growth curve that dips is not defensible in a report figure.

`beta` is set from the anchor span, `_CLAMP_SOFTNESS / (sb_z - sa_z)` with `_CLAMP_SOFTNESS = 50`, so the rounding is scale-free across models. On the Down syndrome grid (`z = (age − 39.0) / 20.0`, anchor span 2.993 in `z`) that gives `beta` = 16.7 and confines the rounding to a narrow window:

| age (months)                |   24 |   48 |   72 |    80 |    **84** |    88 |   96 |  115 |
| --------------------------- | ---: | ---: | ---: | ----: | --------: | ----: | ---: | ---: |
| soft − hard `z_eff`, months | 0.00 | 0.00 | 0.00 | −0.04 | **−0.83** | −0.04 | 0.00 | 0.00 |

The two forms are indistinguishable outside roughly ±4 months of the anchor, and identical at 96 and 115 — so the extrapolation fix, which is the whole point, is fully retained.

Four properties are deliberate:

- **One-sided.** Below the low anchor the line still extrapolates. That extrapolation is accurate — it puts `q(12)` at 0.019 against a fitted 0.022 and an observed 0.026 at 18 months — whereas clamping there would pin young-age values at the 24-month level, an error far larger than the one being fixed. `tent_and_gp` clamps both ends, which is why it was not reused.
- **The low anchor keeps its meaning exactly**, and the high anchor keeps it approximately. This is the price of smoothness and it is worth stating plainly: `p_slope_hi` is no longer exactly the mean at 84 months but short of it by `slope * log(2) / beta`, which on the DS grid moves the implied `q` at the anchor from 0.9402 to 0.9363. Both anchor priors carry over unchanged. A test pins the offset to its formula so it cannot drift silently.
- **No new random variable.** The clamp is one `softplus`, so the free-RV stream and its order are identical and the RNG-reproducibility discipline in `_gp_from_mean` is preserved. A test asserts this.
- **Monotone through the anchor.** A test asserts it, and a companion control test asserts the same statistic fails on a hard clamp — so the test cannot silently stop testing anything.

Flat is not true either — the realised `q` keeps rising to 0.842 at 115 months. The argument is asymmetric cost: a GP can add a gentle rise onto a flat mean far more cheaply than it can subtract a large fall from a saturating one, and where there is no data a flat continuation is a more defensible default than an assertion of near-total production.

### Prior predictive, before and after

| age (months)             |    24 |    48 |    72 |    84 |    96 |   105 |       115 |
| ------------------------ | ----: | ----: | ----: | ----: | ----: | ----: | --------: |
| prior median `q`, before | 0.121 | 0.359 | 0.679 | 0.810 | 0.900 | 0.938 | **0.963** |
| prior median `q`, after  | 0.121 | 0.359 | 0.678 | 0.811 | 0.817 | 0.815 | **0.811** |
| P(`q` > 0.99), before    |     — |     — | 0.011 | 0.046 | 0.124 | 0.193 | **0.294** |
| P(`q` > 0.99), after     |     — |     — | 0.011 | 0.047 | 0.054 | 0.052 | **0.051** |

Identical at and below the high anchor, as intended. Prior median spoken at 115 months falls from 670 to 427 words out of 810.

## 5. Scope

Switched on for the eight Down syndrome joint models. VG13 keeps it off — its domain is 8–18 months against anchors at 10 and 16, so there are two months of extrapolation and nothing to fix. VG11 and VG12 are likewise unaffected (8–30 months, anchors 12 and 26).

**VG01 and VG02 have the same defect and are not changed here.** They are Down syndrome univariate models with the same `slope_anchors` (24, 84) and the same `gp_domain_months` (8, 115), so their means extrapolate identically; they are built from `UnivariateModelDefinition`, which does not carry the new field. This was a deliberate scoping decision, not an oversight — see §8.

## 6. Tests

Seven tests in [`tests/test_gp_utils.py`](../tests/test_gp_utils.py): that the mean is flat above the high anchor and equals `intercept + slope * sb_z` there; that it is unchanged at and below the anchor; that it still slopes below the low anchor; that without the clamp it keeps extrapolating (the behaviour being removed); that no free RV is added or reordered; and that the anchored GP is orthogonal to the **clamped** slope column and demonstrably not to the raw one, while still pinned to zero at the reference row.

The orthogonality assertions are on the _centred_ GP. `_orthogonalise_and_anchor` projects and then shifts the residual to zero at the anchor row, which re-introduces a constant, so orthogonality to the constant column does not survive by construction; the slope-column invariant does.

## 7. VG10 refits — the elbow is fixed, the convergence cost is not

VG10 refitted at `test` (4 chains x 2,000 draws, seed 47, no overrides) twice: once with the hard `min(z, sb_z)`, then again with the soft form now in the tree. 5m 56s and 5m 30s against 5m 42s without the clamp. The full test suite passes for both.

|                | no clamp | hard clamp | **soft clamp** |
| -------------- | -------: | ---------: | -------------: |
| gate `passed`  |     True |      False |      **False** |
| divergences    |        0 |          3 |          **0** |
| max R-hat      |   1.0084 |     1.0143 |     **1.0139** |
| min ESS        |      433 |        404 |        **343** |
| min BFMI       |    0.468 |      0.453 |      **0.424** |
| R-hat failures |     none |          2 |          **5** |

### What the soft form fixed

The elbow is gone. The fitted spoken and understood curves are **monotone over the entire plot grid**, and through the anchor the spoken curve now runs 413.2, 421.6, 427.0, 430.7, 434.7, 439.8, 445.9 words across 83.3-91.0 months — smallest step +0.48 words — against the hard clamp's dip from 428.6 at 84.3 to 426.6 at 85.6. Divergences also return to zero. The change at the 84-month query age drops from +33.7 words to +22.8, and every other queried age moves by 3 words or less.

### What it did not fix

**Neither variant recovers the clean convergence gate.** The soft form trades the hard form's 3 divergences for a lower minimum ESS (343 against 404) and _more_ R-hat failures — five against two, now including `ell_unit_u`, `ell_u` and `tau_q`, where the hard clamp's were confined to two understood GP coefficients.

That two independent implementations both regress from a clean pass weakens the "probably run-to-run variation" hedge offered after the first refit. It does not settle it: this is still one fit per variant, and one no-clamp fit to compare against, so a lucky baseline is not excluded. Item 6 is the experiment that would decide it, and it is cheap.

None of these are catastrophic values — max R-hat 1.0139 against a 1.01 threshold, min ESS 343 out of 8,000 draws — and VG10 carried a REVIEW verdict throughout its history until the `eta_q` widening earlier the same day. But the honest summary is that a better-specified prior has been bought at the cost of the gate, and that trade has not been ratified.

### The benefit is unchanged between the two forms

|                              |      no clamp |    hard clamp |        soft clamp |
| ---------------------------- | ------------: | ------------: | ----------------: |
| `q` mean alone at 115 mo     |         0.993 |         0.940 |             0.940 |
| GP pull at 115 mo (logit)    |         -3.29 |         -1.46 |         **-1.43** |
| `eta_q` median / contraction |  0.855 / 0.35 |  0.696 / 0.42 |  **0.685 / 0.42** |
| `eta_u` median / contraction |  0.916 / 0.29 |  0.833 / 0.33 |  **0.842 / 0.33** |
| `p_slope_hi_q` / prior CDF   | 0.931 / 0.819 | 0.940 / 0.845 | **0.940 / 0.846** |

Both GP amplitudes fall and their contraction rises, identically under either form. That is §3's mechanism confirmed, and it is the substantive result of this note: the amplitudes were being consumed correcting the mean's asymptote, and stopping that lets the data inform them.

### Seed repeats: the convergence cost is not real

The regression above rested on one fit per variant. Four further fits at `test` — sampler seeds 101 and 202, clamp on and off, into a scratch output root, with `definition.random_seed` untouched so every fit sees an identical analysis frame — settle it.

| clamp | seed | passed | divergences | max R-hat | min ESS | min BFMI | R-hat failures |
| ----- | ---: | ------ | ----------: | --------: | ------: | -------: | -------------: |
| off   |   47 | True   |           0 |    1.0084 |     433 |    0.468 |              0 |
| off   |  101 | False  |           1 |    1.0115 |     333 |    0.459 |              1 |
| off   |  202 | False  |           4 |    1.0132 |     437 |    0.465 |              2 |
| on    |   47 | False  |           0 |    1.0139 |     343 |    0.424 |              5 |
| on    |  101 | False  |           0 |    1.0095 |     364 |    0.462 |              0 |
| on    |  202 | False  |           0 |    1.0129 |     432 |    0.484 |              3 |

**The clean gate at seed 47 without the clamp was luck.** The same configuration fails at both other seeds. VG10 at `test` passes 1 of 3 without the clamp and 0 of 3 with it, and at n = 3 that difference carries no weight.

The two arms' ranges overlap almost exactly — max R-hat 1.0084-1.0132 against 1.0095-1.0139, min ESS 333-437 against 343-432 — so the clamp is not detectably worse on either. On divergences it is arguably **better**: zero in all three clamped fits against 0, 1 and 4 without. That is suggestive rather than established at three fits each, but it is the opposite sign to the concern.

Two conclusions, and one correction:

- **The convergence cost reported above is withdrawn.** It was seed noise read as a treatment effect, from a single fit either side.
- The R-hat failure counts are volatile in both arms (0, 1, 2 without; 5, 0, 3 with), so the "failures spread to `ell_unit_u`, `ell_u`, `tau_q`" observation in the previous subsection is also noise and should not be pursued. Open item 8 is withdrawn.
- **VG10 does not reliably meet the convergence gate at `test` either way.** Max R-hat sits at 1.008-1.014 against a 1.01 threshold in every fit. That is a property of the model and the sampling configuration, not of anything changed today, and it means single-fit gate outcomes on this model should not be quoted as evidence for or against a change.

### The gap at the anchor was never extrapolation

The GP pull **at** 84 months is essentially unchanged across all three fits: -1.82, -1.77, -1.74. The clamp only alters behaviour above the anchor, and at 84 the mean sits at 0.936 against a realised 0.721. So a substantial part of what §3 characterised as the GP correcting the mean is not extrapolation at all — it is the straight trend overshooting its own upper anchor because the realised curve is concave approaching it. §3's framing ("the GP is idle where the data are") holds at 48 months but overstates the case at 72-84.

## 8. Open

1. **Refit the family.** VG05, VG07, VG08, VG09, VG14, VG15 and VG16 all carry the graph change with stale fits.
2. **VG01 and VG02 — measured, and much less exposed than expected.** Same anchors and domain, but the saturation does not follow:

   | model    | mean alone at 115 mo | realised | GP pull | P(mean > 0.99) |
   | -------- | -------------------: | -------: | ------: | -------------: |
   | VG01     |            671 words |      488 |   −1.16 |          0.000 |
   | VG02     |            725 words |      694 |   −0.36 |          0.003 |
   | VG10 `q` |                0.993 |    0.842 |   −3.29 |      **0.896** |

   The reason is that `q` is a conditional **ratio** that genuinely approaches 1, so its logit saturates hard, whereas a vocabulary proportion out of 810 stays well short of the ceiling even at 115 months and its logit does not. The defect is therefore worst on ratio-valued trajectories, which is `q` — extending the clamp to `UnivariateModelDefinition` is low priority on this evidence.

   VG01 does have a separate and larger problem this measurement turned up: its mean sits at 47 words against a realised 185 at **48 months**, a GP pull of +1.57 inside the data-rich range. That is not extrapolation and is not addressed by anything here.

3. **Whether `eta_q` should now come back down.** It was widened to `HalfNormal(0.8)` earlier the same day, on evidence that §3 here reinterprets. Partly answered: with the clamp, `eta_q` falls from 0.855 to 0.685 at prior CDF 0.643 with contraction 0.42, so the wider prior is no longer binding and is doing no harm — but 0.685 is well above what `HalfNormal(0.4)` would comfortably allow, so reverting the widening would reintroduce a conflict. Leave it at 0.8; revisit only if the clamp is kept and the family refit shows it settling lower.
4. **The residual is still there.** Clamping does not make the mean match the data better between the anchors; §2 says a straight line is already adequate there, and the previous note's remaining prior-predictive gap at 54 months is not addressed by this change.
5. ~~**A smooth clamp would remove the elbow.**~~ Done and kept — it is the implementation in the tree. It removed the elbow and the dip and took divergences back to zero, but did **not** recover the convergence (§7): min ESS fell further, 404 to 343, and R-hat failures rose from two to five. The predicted cost also materialised as predicted — `p_slope_hi` is short of the anchor value by `slope * log(2) / beta`, moving the implied `q(84)` from 0.9402 to 0.9363.
6. ~~**Repeat the convergence comparison.**~~ Done (§7). The regression was seed noise; the clean gate without the clamp reproduces 1 time in 3. The clamp is not detectably worse on R-hat or ESS and has zero divergences across all three seeds against 0, 1 and 4 without.
7. **Decide whether to keep this.** With item 6 answered the objection is gone: the prior-predictive gain is real, the mechanism is confirmed, the curves stay monotone, and there is no measurable convergence cost. The remaining question is scope rather than merit — see item 9.
8. ~~**Why the R-hat failures moved.**~~ Withdrawn (§7). The failure counts swing 0, 1, 2 across seeds without the clamp and 5, 0, 3 with it, so which parameters appear is noise at this sampling configuration and there is nothing to explain.
9. **VG10 does not reliably meet the convergence gate at `test`.** Max R-hat sits at 1.008-1.014 against a 1.01 threshold across all six fits in §7, so pass or fail turns on the seed. This predates everything in this note. Two consequences: single-fit gate outcomes on this model must not be quoted as evidence for or against a change, and the reporting-quality configuration's chain count should be confirmed adequate before any of this is reported.
10. ~~**Whether the analysis should be capped at 84 months**~~ was raised separately and is _not_ the right fix for the defect in this note. Comprehension data effectively stop at about 72 months (9 rows in 72-84, 5 above 84), but spoken data above 84 are real and informative: within uk_01 alone the median runs 363 words at 72-84, 479 at 84-96 and 539 at 96-115, across 43 rows and 36 children, 22 of whom appear only above 84. A data cap would discard that to fix an unobserved-comprehension problem the likelihood already handles by outcome-wise missingness. The supportable version is to trim what is _reported_ for `u` and `q` to where their data end — **done, §9**.

## 9. Trimming what is reported for `u` and `q`

Item 10's supportable version, implemented. This is the reporting counterpart of the clamp: §4 fixes what the mean _does_ above the high anchor, this stops quoting an age that has no data behind it.

### The asymmetry

The query grid is shared by every outcome a model reports, but the Down syndrome outcomes are not observed over the same range. Measured on VG10's own analysis frame (1349 rows, 737 children):

| Outcome    | Rows |  95th pct |  Rows ≥ 72 mo | Rows ≥ 84 mo |
| ---------- | ---: | --------: | ------------: | -----------: |
| Understood |  905 | 64 months |  15 (15 kids) |   5 (5 kids) |
| Spoken     | 1346 | 78 months | 104 (80 kids) | 51 (44 kids) |
| Both (`q`) |  902 | 64 months |  15 (15 kids) |   5 (5 kids) |

`q` tracks understood almost exactly — 902 of the 905 understood rows also carry spoken — because it is a ratio _of_ comprehension and inherits the narrower range. Understood's 95th percentile is 64 months, so the grid's top three ages (78, 84, 90) rest on at most eight administrations, and two of them sit at or past the high anchor where the mean is now a levelled-off extrapolation rather than an estimate.

The counts differ trivially from item 10's because that used a strict `> 72`; one administration falls at exactly 72.0.

### The change

`report_max_age_understood = 72` on VG02, VG05, VG07-VG10 and VG14-VG16 — the nine models that report comprehension. It trims the understood and `q` summary tables and the production-ratio figure; spoken keeps the full grid.

Deliberately **report-time only**. The query grid, the model graph, the `query_id` dimension and the traces on disk are all untouched, so this needs no refit and cannot move a number that is still reported — verified by asserting the trimmed frame equals the full frame's surviving rows, for both the table and the figure's CSV companion.

### Scope, and what was left alone

- **VG01** is production-only and its data run to 115 months. Trimming it would discard exactly the evidence that argued against the 84-month cap in item 10. Validation now _rejects_ the field on a non-comprehension model rather than letting it be a silent no-op.
- **The whole-month companion tables** keep the full observed span. That is deliberate rather than an oversight: they carry an `n_obs` column that records the emptiness directly, which the curated 6-monthly table has no equivalent of. They are the exhaustive companion; the query table is the headline.
- **The typically-developing models** stop at 30 months, well inside their data.
- **The understood trajectory figures** (`posterior_predictive_median_trend_u`, `expected_learning_rate_u`, the joint u+s trajectory) still run the full plot grid. The production-ratio figure was trimmed because it sits directly beside the trimmed `q` table in the report chapter and the two disagreeing would be worse than either alone; the rest is a wider design question about whether a joint u+s figure should show its two curves over different spans — see open item 11.

### Tests

`tests/test_reported_age_range.py`, 30 tests. Beyond the helper's own behaviour, two failure modes specific to this design are covered directly: the value reaches the engines through the _configuration_ object rather than the definition, so a missing pass-through would raise only mid-fit — caught statically per engine, and mutation-checked by deleting one pass-through and confirming the test fails; and the cap is asserted to lie on the query grid and to actually remove ages, so it cannot silently become inert.

## 10. The family refit

All seven models carrying the §4 graph change with pre-clamp fits were refitted at `test` (4 chains × 2000 draws, seed 47 — the configuration their previous fits used), with `--render`, on a tree carrying both §4 and §9. Total wall time 44 minutes; every model exited 0.

Fitting with §9 included rather than on plain `main` is deliberate. `--render-only` renders Quarto against the _existing_ summary CSVs and does not re-run `posterior_summary`, so the tables can only be regenerated by refitting; fitting without the trim would have forced a second 44-minute round. §9 is report-time only and provably does not touch the trace, so the posterior is identical either way.

| model | divergences |           max R-hat |         min ESS |     gate |
| ----- | ----------: | ------------------: | --------------: | -------: |
| VG05  |  33 → **3** | 1.0076 → **1.0024** |  420 → **1830** |    False |
| VG07  |       4 → 4 | 1.0058 → **1.0031** | 1014 → **1499** |    False |
| VG08  |       1 → 2 |     1.0217 → 1.0398 |       208 → 135 |    False |
| VG09  |  23 → **5** |     1.0132 → 1.0136 |   200 → **219** |    False |
| VG14  |   7 → **2** | 1.0048 → **1.0041** | 1241 → **1997** |    False |
| VG15  |       2 → 3 | 1.0248 → **1.0074** |   379 → **479** |    False |
| VG16  |   1 → **0** | 1.0113 → **1.0099** |   402 → **529** | **True** |

R-hat improved in five of seven, min ESS in six of seven, divergences in four of seven.

> [!CAUTION]
> **None of this is evidence that the clamp improves convergence.** §7 established that at `test` these gate outcomes turn on the sampler seed: VG10's max R-hat spans 1.008–1.014 across six fits against a 1.01 threshold. These are single fits at one seed, so VG16's newly clean gate and VG08's regression are equally likely to be noise. Quoting either as a treatment effect would repeat exactly the error §7 withdrew. Before any of this is reported, the reporting-quality configuration must be confirmed adequate (open item 9).

Six of the seven now fail on divergences alone, having cleared R-hat, ESS and BFMI. The exceptions are **VG08** and **VG09**, whose R-hat and ESS failures cluster on the understood mean and the leading HSGP coefficients — `p_slope_low_u`, `slope_u`, `intercept_u`, `g_unit_u_hsgp_coeffs[0..2]`. That is the mean/GP aliasing signature, and it is structural to their place in the ladder rather than anything this note introduced: both carry subject random effects with `anchor_g_u_at_ref = False`, and the per-draw GP anchoring that breaks the degeneracy is precisely what VG10 and VG16 add.

The §9 trim landed on all seven — understood and `q` at 11 rows to 72 months, spoken at 14 rows to 90. VG14 and VG15 additionally confirm that the _signed_ ratio `r` correctly keeps the full grid: it is a ratio of comprehension, but signing data do not stop where comprehension does.

## 11. Open

11. **Whether the `u` trajectory figures should stop at 72 too.** The tables and the production-ratio figure now do. Leaving the trend and learning-rate figures on the full grid is defensible — they show the fitted curve, not a claim about evidence — but it is not obviously right, and the joint u+s figure would need a deliberate decision about showing two curves over different spans.
12. ~~**The seven stale DS joint models.**~~ Done — §10.
13. **VG10's tables are now the odd ones out.** It was refitted before §9 existed, so its understood and `q` tables still run to 90 while the other seven stop at 72. A six-minute refit fixes it, but it overwrites a current model of record and was outside the seven.
14. **The report figure cache is stale.** `docs/report/figures/` still holds the pre-refit plots and tables. `sync_report_figures.py` validates reporting quality, so at `test` it needs `--allow-provisional`.
