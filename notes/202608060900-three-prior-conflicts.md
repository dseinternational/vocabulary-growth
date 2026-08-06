# Three prior–data conflicts: options, decisions and results

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Working record, 2026-08-06. Follows the R3 sweep in [202608051500](202608051500-report-critical-review.md) §4a, which flagged three distinct problems across the model family. This note records the options considered for each, the study owner's decisions, and what the tests actually showed. **Two of the three are resolved; the third is still running.** §5 records a claim of mine that was withdrawn under challenge.

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

## 6. Open

1. Whether VG14 is partially informative or wholly replaced by VG15 — the decision its migration was done in anticipation of.
2. `eta` in VG01, VG03, VG11, VG12. The clamp helps but does not resolve it; widening is known to fail. Needs something that _identifies_ the amplitude.
3. VG15's `ell_unit_sign` at contraction 0.033, on a headline model. Not yet investigated; likely the same class of problem as VG13's, since the signed data are sparser than the spoken.
4. VG13's predictive intervals are much too wide — empirical coverage 0.99 at nominal 0.89. Separate from the GP question, and possibly the subject-marginal predictive being scored against children who are in the sample.
