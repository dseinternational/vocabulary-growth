# Three prior–data conflicts: options, decisions and results

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Working record, 2026-08-06. Follows the R3 sweep in [202608051500](202608051500-report-critical-review.md) §4a, which flagged three distinct problems across the model family. This note records the options considered for each, the study owner's decisions, and what the tests actually showed. **Two of the three are resolved; the third is still running.** §5 and §5a each record a claim of mine that was withdrawn — the first under challenge, the second on finding a confound the project had already documented. Both were the same mistake: comparing a population-level model curve against a raw pooled empirical mean.

## 0. Summary

| Problem                                              | Decision                            | Status                       |
| ---------------------------------------------------- | ----------------------------------- | ---------------------------- |
| 1. `b_kappa_mag_s` pinned ~4σ beyond its prior       | Disclose for VG05/07/08; fix VG14   | **VG14 refitting**           |
| 2. `eta` presses its prior in four univariate models | Test the mean-clamp hypothesis      | **Partially confirmed** (§3) |
| 3. VG13's GP hyperparameters uninformed              | Test whether the window hides curve | **Running** (§5)             |

Two things found along the way were defects in the tooling rather than the models: the R3 sweep did not cover two model families (§2), and the trivariate engine could not express the fix Problem 1 needed (§4).

## 1. What the sweep found

Extended to all 15 registered models, 219 parameters, 15 flagged. Three distinct problems, not one:

- **`b_kappa_mag_s` at prior CDF 1.000 with negative contraction** in VG05, VG07, VG08 and VG14. The legacy dispersion form constrains `b_kappa_mag ≥ 0`, forcing dispersion to _fall_ with age, and `HalfNormal(0.3)` caps how fast. The data want ~1.22 — about four standard deviations out — and the posterior is _wider_ than the prior. A parameter pinned against a boundary is not an estimate.
- **`eta` pressing** in VG01 (0.960), VG03 (0.983), VG11 (0.958) and VG12. These are exactly the univariate models.
- **GP hyperparameters uninformed**: VG13's `eta_q`, `ell_unit_u`, `ell_unit_q` (contraction ≤ 0.03), and VG15's `ell_unit_sign` (0.033).

## 2. The sweep had missed two model families

VG14 and VG15 were excluded from `prior_vs_posterior.py` entirely, because their signed-ratio, `psi` and concentration priors were not reconstructed. The exclusion was **wholesale rather than partial**, so VG14's conflict went unseen on `b_kappa_mag_s` — a parameter in the _shared_ bivariate block the sweep could already build.

Fixed by `signing_priors` + a family-aware `model_priors` dispatch. Coverage went from 167 parameters across 13 models to 219 across 15, and immediately turned up **VG15's `ell_unit_sign` at contraction 0.033 — on a headline model**.

The lesson generalises: a diagnostic that skips a subject entirely reports nothing rather than reporting a gap, and silence reads like a pass.

## 3. Problem 2 — the clamp hypothesis, partially confirmed

The four models where `eta` presses are exactly the univariate ones, and `clamp_mean_above_hi_anchor` — the 2026-08-04 fix that demonstrably lowered GP amplitudes with their contraction rising — **does not exist on their definition class**. It was added to the joint models and never reached the univariate engines.

Hypothesis: `eta` presses because the parametric mean extrapolates past its high anchor and the GP spends amplitude correcting that instead of describing developmental curvature.

Both univariate engines now read the flag through `getattr`, and VG01 was fitted at `test` config with and without it. VG01 is the strongest test case — its extrapolation region is 84–115 months, ~29% of the GP domain, against ~18% for the typically-developing models.

| arm         |     `eta` | prior CDF | contraction | `ell_unit` | divergences | min BFMI |
| ----------- | --------: | --------: | ----------: | ---------: | ----------: | -------: |
| baseline    |     1.031 |     0.961 |       0.265 |      0.490 |           0 |    0.873 |
| **clamped** | **0.883** | **0.923** |       0.284 |      0.491 |           1 |    0.868 |

**The mechanism is real and reproduces the joint models' signature** — amplitude down 14%, contraction up, and VG01 drops below the 0.95 flag threshold — with no side effects: the length-scale is unmoved, BFMI unchanged, and 0 → 1 divergences is noise.

**But it reduces the pressure rather than removing it.** Prior CDF 0.923 is still the upper fifth of the prior. And VG01 is the best case; the typically-developing models have far smaller extrapolation regions, so expect less there.

**Position:** adopt the clamp for the univariate models on the _extrapolation_ argument, which is settled independently — an unbounded mean above the high anchor is a defect whatever `eta` does — and treat the amplitude improvement as a secondary benefit rather than a solution to Problem 2. `eta` remains an open problem, and the one obvious remedy is already known to fail: widening cost VG12 27 divergences without identifying the amplitude.

## 4. Problem 1 — and why VG14 had never been fixed

The study owner's position, 2026-08-06: **VG05, VG07 and VG08 supply no reported number**, being superseded later in the lineage. That makes their boundary-pinned parameter _disclosable rather than fixable_ — a development step can carry one provided the report says so. It also preserves the lineage: changing a prior partway through VG05 → VG07 → VG08 → VG09 → VG10 would confound the contrast the sequence exists to show.

**VG14's role is undecided**, so it was fixed for completeness pending that decision.

The migration then failed, and the failure was the interesting part. `common_trivariate` read `a_kappa_mu` and `b_kappa_mag_sigma` off the definition directly instead of dispatching on the parameterisation. **It accepted only the legacy form.** The bivariate and joint engines had both moved to `_configure_kappa_priors` / `build_kappa_for_config`; this one never did.

So VG14's legacy dispersion was never an oversight in its definition — the engine could not express anything else. The engine now mirrors the others, with per-outcome dispatch, which VG14 relies on: understood and spoken anchored, signed still legacy (its signed block sits comfortably inside its prior at CDF 0.25, contraction 0.49, and has no reason to move).

A stale justification was also corrected. The guard test excluded VG14 on the grounds that its "frame is the signing subset, not this one". **That is factually wrong**: VG10, VG14 and VG15 all fit the same 1,349-row frame, and VG15 already uses these exact prior objects.

## 5. Problem 3 — a claim withdrawn, and the test that replaces it

VG13's GP is unidentifiable **by construction**. `ell_months_range = (6, 18)` puts the median length-scale at 12 months over a **10-month** window (8–18), with 95% of prior mass on 8.3–15.7 months. A GP whose length-scale exceeds its domain is close to linear across it, and the per-draw anchor orthogonalises it against `[1, z]` — precisely what such a GP can produce. It can express almost nothing, and the amplitude of nothing is unidentifiable. The range was inherited unchanged from VG12 (22-month window) and VG10 (107-month window) without rescaling.

That much stands. What follows did not.

> [!CAUTION]
> **Withdrawn.** I argued from VG13's posterior predictive calibration — mean errors of 0.01 words on 147, PIT means at 0.497–0.504 — that "there is no curvature the GP is failing to capture", and recommended dropping the GP. **The evidence does not support the claim.** The calibration has only two 12-month bands, and a two-band mean error is structurally blind to within-band shape: a straight chord across a curve gives near-zero average error in each band while being wrong throughout it.
>
> A finer monthly check does not settle it either, in the opposite direction. Residuals against the population-level curve are positive at ten of eleven ages (mean +14.3, two sign runs over eleven), which looks systematic — but that is almost certainly random-effect marginalisation, not trend misfit: with subject effects of sd ≈ 1 on the logit scale the mean over children sits above the sigmoid at the population mean, and the gap shrinks as `p` rises, which is the pattern observed. The whole-frame calibration compares the _marginal_ predictive and matches to 0.007.
>
> So: a fine-grained comparison against the wrong quantity, and a right-quantity comparison at the wrong resolution. Neither answers the question.

The deeper error was treating an inert GP as evidence about the data. **A GP that cannot express curvature looks inert whether or not there is any to find**, so its inertness — and `eta_u` sitting mid-prior at CDF 0.582 — is uninformative about curvature. There is also substantive reason to expect some: 8–18 months is where typically developing vocabulary acceleration begins.

### The test

VG13 fitted at `test` config with `ell_months_range` at its current `(6, 18)` and at `(2, 8)`, short enough to express structure inside a 10-month window, compared by ELPD. If the flexible GP earns predictive improvement there is curve the current model is blind to; if not, there is not.

This inverts the recommendation back to **rescale** rather than **drop**, and for a better reason than the first time: not "the mean fits, so remove it", but "the current model cannot answer the question, and this makes it answerable".

It matters beyond VG13's own fit — VG13 supplies the typically-developing side of the matched-comprehension contrast in the only written results chapter. If it has been smoothing over an acceleration, that bears on the contrast.

### Result: the test does not work, and that is the finding

Both arms fitted at `test` config.

| arm             | `ell_unit_u` | implied `ell` | `eta_u` | contraction | divergences | min BFMI | max R-hat |
| --------------- | -----------: | ------------: | ------: | ----------: | ----------: | -------: | --------: |
| baseline (6–18) |        0.509 |       12.1 mo |   0.319 |       0.090 |           3 |    0.240 |    1.0105 |
| rescaled (2–8)  |        0.495 |        5.0 mo |   0.233 |       0.192 |     **140** |    0.258 |    1.0292 |

**Three things, none of which answers the original question.**

1. **Rescaling to (2, 8) is not a viable configuration.** 140 divergences against 3. Whatever the shorter length-scale buys, it wrecks the geometry, and a fit with 140 divergences cannot be trusted for parameter estimates either — which undercuts reading anything into its `eta_u`.
2. **`ell_unit` sits at 0.50 in _both_ arms** — dead centre of its Beta(3,3) prior. The length-scale is unidentified whichever range it is given. That is a stronger and broader statement than "the window is shorter than the length-scale": rescaling the range did not make it identifiable, so the diagnosis in this section is right about the conclusion and incomplete about the cause.
3. **PSIS-LOO cannot adjudicate.** On understood it reports 45% of Pareto k values above 0.7 with `p_loo` 3723 on 6358 observations; on spoken it fails outright ("All tail values are the same"). With per-child random effects, dropping one observation moves that child's effect substantially, which is exactly the regime where importance sampling breaks. The project already knows this — the 12 May review recorded PSIS-LOO instability and used K-fold leave-one-subject-out instead, and `scripts/kfold_loso.py` exists for it.

So the ELPD comparison proposed above **cannot be run with the tool proposed**. The question — is there curvature in 8–18 months that VG13 is blind to? — remains open, and answering it needs K-fold LOSO rather than PSIS-LOO, at materially greater cost.

What is now settled: **rescaling the length-scale is not the fix**, so of the four options in the original review, A is eliminated on evidence. Dropping the GP (B) or fixing its hyperparameters (C) remain, and both are consistent with a length-scale that is unidentified at any range.

## 5a. The signed ratio: why the GP will not find the hump

Investigating VG15's `ell_unit_sign` (contraction 0.033) led into the signed-ratio mean. Recorded here because the structure is easy to misread: VG14 and VG15 _do_ carry a Gaussian process on the signed ratio, and it nonetheless cannot supply or move the hump.

    logit r(a) = tent_mean(a)  +  eta_sign * g_hsgp(a)

**Four independent reasons the GP cannot find the peak.**

1. **A zero-mean GP has a flat prior median.** It can add smooth departures from a shape, not supply one. This is why the signed mean is parametric at all: #54 made it intercept-only and the level went prior-dominated; #154 replaced that with the three-anchor tent explicitly to obtain a hill-shaped prior median.
2. **In VG15 the GP is orthogonalised against the whole tent basis** — all three hats, not just `[1, z]`. `tent_and_gp`'s docstring is explicit that this is so it "cannot mimic a shift of any anchor". The GP is _designed_ to be unable to move the peak.
3. **Its length-scale is unidentified.** Contraction 0.033 in VG15 and 0.073 in VG14, which has no orthogonalisation at all — so this is not caused by (2). The cause is the data: 516 signed observations with **85% between 12 and 48 months**, 23 above 60, 7 above 72, none between 84 and 96, against a GP domain running to 115.
4. **Its amplitude is only partly identified** (contraction 0.29–0.35), so even the departures it can express are weakly determined.

The consequence for reporting is larger than the peak. VG15's signed trajectory is **parametric in practice**, so the credible band on `r(a)` is the uncertainty in three anchor heights propagated through a fixed shape — not uncertainty about the shape. It will read tighter than the evidence supports, most severely above 60 months where the shape is asserted and 23 observations remain.

### Withdrawn: that the fixed peak demonstrably biases the fitted curve

> [!CAUTION]
> I compared VG15's population-level `r(a)` against the observed pooled ratio by age band, found mean residuals of −0.006 below 36 months and **+0.059 above** — worst at 48–54 months, observed 0.365 against fitted 0.242 — and argued the fixed knot was biasing the curve. I checked the random-effect marginalisation confound (the residual sign flips at the knot, which a level effect cannot produce) and concluded the bias was real.
>
> **It is not established.** The comparison is confounded by _study composition_, which I did not check, and which [202606151700](202606151700-vg14-signed-ratio-shape-and-p-any-bias.md) had already documented. Per-study pooled ratios span **uk_01 0.014, es_01 0.149, ie_02 0.320, uk_02 0.511** — a 36-fold spread. uk_02 supplies **52% of the 48–54 month band** against roughly 20–25% of the bands below it, and the ratio collapses at 54–60 when es_01 takes over. The apparent late peak tracks uk_02's rising share, not age.
>
> That is the same error as §5's, in a new place: comparing a population-level curve (random effects at zero) against a raw pooled empirical mean, when the grouping the model explicitly accounts for is also changing with age. Checking one confound is not checking the confounds.

### What stands, and what the test now tests

Unaffected by the above, because none of it rests on that comparison:

- The peak sits at 36 months **by construction** — 77% of posterior draws peak within a month of the fixed anchor. Its height is estimated (contraction 0.76); its age is not estimated at all.
- Therefore "signing peaks around three years" is a statement about anchor placement and must not be reported as a finding.
- The GP cannot rescue this, for the four structural reasons above.
- Signed evidence effectively stops around 60 months while `r(a)` is reported to 115.

The June note concluded that "the peak age cannot be identified from these data with this model", on the composition argument now confirmed. **That conclusion was reached for VG14, which has no study random effects** — its age curve necessarily absorbs composition. VG15 _does_ carry study random intercepts, so between-study level is absorbed and the age curve has a chance to separate "this age signs more" from "uk_02 children sign more".

Whether it actually does is the open question, and it is exactly what the free-peak arm tests. If `peak_unit` is identified, VG15's study effects did the work VG14 could not. If it comes back at its prior, the June conclusion holds for VG15 too and the peak age is simply not recoverable from this data. Either outcome is worth having; neither is assumed.

### Result: the peak age is identifiable in VG15, and it is earlier than the anchor

| arm      | divergences | min BFMI | max R-hat | `peak_unit` contraction | peak age                      |
| -------- | ----------: | -------: | --------: | ----------------------: | ----------------------------- |
| fixed    |           2 |    0.510 |    1.0073 |                       — | 36.0 (asserted)               |
| **free** |       **0** |    0.504 |    1.0097 |               **0.481** | **29.4 mo, 89% [23.9, 46.2]** |

Three things, and none was guaranteed.

1. **A moving knot samples.** Zero divergences against the fixed arm's two, BFMI and R-hat unchanged. The draw-dependent nuisance basis and the free knot in a piecewise-linear mean — both plausible geometry hazards — cost nothing here.
2. **The peak age is identified.** Contraction 0.481, and the posterior interval [23.9, 46.2] against the prior's [21.5, 67.5]. Not sharp, but the data plainly inform it.
3. **The estimate is 29.4 months, about six months _earlier_ than the fixed anchor** — and the peak height is unmoved (0.319 → 0.314), so this is a shift in shape, not level.

That last point is worth dwelling on, because it is the reverse of §5a's withdrawn claim. The confounded analysis suggested the true peak was _later_ than 36; the properly adjusted answer is _earlier_. Study random effects absorb uk_02's high signing baseline, and once they do, the developmental peak moves earlier than the raw pooled ratio implies. A confounded comparison did not merely overstate a real effect — it pointed the wrong way.

**The June conclusion does not extend to VG15.** [202606151700](202606151700-vg14-signed-ratio-shape-and-p-any-bias.md) found the peak age unidentifiable, but for VG14, which has no study random effects and whose age curve must therefore absorb composition. VG15 has them, and they are what makes the peak recoverable. The earlier finding stands for the model it was made about.

**Not yet established**: that 29.4 months is the right answer rather than a better one. This is a single `test`-config arm against a prior deliberately centred elsewhere (Beta(2, 4), median 40 months), which is reassuring — the data pulled it down against the prior's pull — but it wants a `rep` fit and a prior-sensitivity check before it is reported. Enabling it is a graph change requiring a VG15 refit, and VG15 is a headline model.

## 5b. VG15's two flags: both resolve to disclose, not change

Settled 2026-08-06 after three `test`-config arms and direct measurement. Recorded because the conclusion reversed twice on the way, and the reversals are the useful part.

### `kappa_sign` — leave legacy

|                     |                                                          |
| ------------------- | -------------------------------------------------------- |
| `b_kappa_mag_sign`  | median 0.074, prior CDF 0.276, **contraction 0.429**     |
| constraint binding? | P(< 0.02) = 0.170 — approaches zero but does not pile up |

Well identified, comfortably inside its prior, and the legacy form's sign constraint (dispersion non-increasing with age) is not binding. This is **not** the defect VG05/VG07/VG08/VG14 carried, where the same parameter sat four standard deviations beyond its prior with the posterior wider than it. Migrating would be consistency churn on a headline model with a refit cost and no defect to fix.

The resulting asymmetry — VG14 migrated, VG15's sign block not — is therefore two separate correct calls rather than an inconsistency, and is recorded as such so nobody "tidies" it later.

### `ell_unit_sign` — unidentified, and inconsequential

Three arms, all at `test`:

| arm           | R-hat    | ESS      | div | min BFMI | max R-hat |
| ------------- | -------- | -------- | --: | -------: | --------: |
| baseline      | pass     | pass     |   2 |    0.510 |    1.0073 |
| **fixed-ell** | pass     | pass     |   2 |    0.508 |    1.0065 |
| no-gp         | **fail** | **fail** |   1 |    0.494 |    1.0143 |

**`fixed-ell` changes nothing measurable.** Holding the length-scale at its prior median (0.2644, ell = 9.17 months — almost exactly the sampled posterior mean of 0.2685) gives a maximum absolute median shift of **0.0023** on `r(a)` and a mean band-width change of **+0.1%**. Convergence is marginally better. It removes one sampled parameter and moves nothing else.

So the R3 flag is **real but benign**: `ell_unit_sign` genuinely is unidentified (contraction 0.033), and that has no consequence for anything VG15 reports. Fixing it would be tidiness bought with a headline refit.

**`no-gp` is worse, for two reasons.** It failed the hard convergence tier outright (R-hat and ESS), and — the more interesting reason — removing the GP narrowed the band by 63% at 96 months, an age with essentially no signed observations. That is not an improvement. The wide band there is the model's _only honest signal of ignorance_ above 60 months, and the GP is what produces it.

> [!CAUTION]
> **Two positions of mine were wrong on the way to this.** First, that the signed GP's band was "wider than the data warrant" — the width where data are absent is honest ignorance, not inflation. Second, that the GP should therefore be fixed or dropped — fixing does nothing and dropping is actively harmful. The measurements corrected both. The direct measurement that started this (median GP contribution 7% of the tent's range against a per-age sd six times larger) was accurate; the _inference_ from it to "the band is miscalibrated" was not.

### What this reframes

The real defect is not the GP. It is that **`r(a)` is reported to 115 months at all**, on 23 observations above 60 and none between 84 and 96. Neither arm addresses that; trimming does, and the recommendation now has a second independent argument behind it — the band above 60 months is doing the work of saying "we do not know", which is exactly the range that should not be reported.

### Disclosure obligations this creates

"Disclose, don't change" is only honest if the disclosure happens. For the signing chapter (currently a stub, so it can be written correctly rather than corrected):

1. `r(a)`'s **peak age is fixed at 36 months by construction**, not estimated — unless the free peak is adopted (§5's result: identifiable at 29.4 months).
2. The signed GP's **length-scale is not identified**; the trajectory is parametric in practice.
3. **ψ is estimated from uk_02 alone** (56 four-cell rows) and applied pool-wide.
4. Signed **evidence stops around 60 months** while the reported range runs to 115.

## 6. Open

1. Whether VG14 is partially informative or wholly replaced by VG15 — the decision its migration was done in anticipation of.
2. `eta` in VG01, VG03, VG11, VG12. The clamp helps but does not resolve it; widening is known to fail. Needs something that _identifies_ the amplitude.
3. VG15's `ell_unit_sign` at contraction 0.033, on a headline model. Not yet investigated; likely the same class of problem as VG13's, since the signed data are sparser than the spoken.
4. VG13's predictive intervals are much too wide — empirical coverage 0.99 at nominal 0.89. Separate from the GP question, and possibly the subject-marginal predictive being scored against children who are in the sample.
