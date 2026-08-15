# Is the R-hat gate too stringent? Calibrating it against the family's own fits

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **This note refutes the hypothesis it was written to test.** It was prompted by VG11 failing the convergence gate on a single parameter, and by the reasonable suspicion that a fixed threshold applied to a maximum over thousands of parameters penalises large models. The measurement says otherwise, and §4 is the retraction. Reproduced by `scripts/experiments/rhat_gate_calibration.py`; the figures below are the 2026-08-15 scan.

## 1. The question

The convergence gate fails closed when **any** sampled parameter exceeds R-hat 1.01. On 2026-08-14 VG11 failed it on one parameter of 14,597 — `g_unit_hsgp_coeffs[4]` at 1.0125 — while every reported quantity converged with margin. That invited three worries, each of which would, if true, mean the rule was measuring something other than convergence:

1. **Multiplicity.** R-hat is estimated with Monte Carlo error, so a maximum over N parameters is an extreme order statistic whose distribution rises with N. A 14,597-parameter model would then be held to a stricter standard than a 30-parameter one for reasons having nothing to do with either.
2. **Inconsistency with the project's own ESS floor.** R-hat's sampling error shrinks with effective sample size, so a fixed R-hat threshold implies a minimum ESS. If that implied minimum is far above the explicit `ESS >= 400` gate, the two halves of the same gate disagree.
3. **Reparameterisation.** R-hat on internal parameters is not invariant — the same posterior written centred or non-centred has different parameters with different R-hat — whereas the reported estimands are invariant.

Worth stating at the outset: **the threshold itself is not unusual.** 1.01 is the Vehtari et al. (2021) recommendation, adopted across the field precisely because the older 1.1 convention was lax enough to pass visibly unconverged chains. Nothing here argues for a looser number. The question is only whether the *rule built on it* — a maximum over every sampled parameter — behaves.

## 2. Method

For every fitted trace in the output root, R-hat and ESS are computed elementwise for two sets:

- **Parameters** — every posterior variable that is not a function of a grid (observations, the concatenated predictor grid, the plot and query grids), minus any scaled random effect whose `_raw` counterpart is also stored, since those are two names for one degree of freedom.
- **Reported quantities** — the `*_plot` and `*_query` grids, which are what the report reads and what a reader sees.

The distinction is the point: the gate screens the first set, the study reports the second.

## 3. Result: the gate is well calibrated for this family

The scan covers **29 fits and 67,328 parameters** — every model of record, every registered sensitivity variant, the parameter-recovery replicates, and the retained VG11 fit that prompted the question.

**Three parameters in total exceed 1.01, across two fits.** Two of them are in the A1 recovery replicate `r02`, whose minimum ESS is **347** — below the project's own 400 floor, so it fails the ESS gate independently and is a genuinely under-sampled fit. The third is VG11's `g_unit_hsgp_coeffs[4]` at 1.0125. Of the fourteen models of record present in the scan, **none has a parameter over 1.01**; VG11 is the fifteenth and the sole model-of-record failure.

And the quantity the study actually reports never fails at all:

> **Across all 29 fits, not one reported grid point exceeds 1.01. The maximum anywhere is 1.0079.**

R-hat against ESS, pooled over every parameter and fit:

| ESS bin       |      n | median R-hat | 99th pct |    max | fraction > 1.01 |
| ------------- | -----: | -----------: | -------: | -----: | --------------: |
| ≤ 400         |      2 |       1.0051 |   1.0056 | 1.0056 |               0 |
| 400–800       |     33 |       1.0039 |   1.0106 | 1.0115 |            3.0% |
| 800–1,600     |    242 |       1.0030 |   1.0089 | 1.0133 |            0.8% |
| 1,600–3,200   |    710 |       1.0013 |   1.0047 | 1.0058 |           **0** |
| 3,200–6,400   |  1,949 |       1.0007 |   1.0025 | 1.0046 |           **0** |
| 6,400–12,800  |  3,985 |       1.0004 |   1.0019 | 1.0035 |           **0** |
| > 12,800      | 60,407 |       1.0002 |   1.0006 | 1.0013 |           **0** |

The relationship is orderly and exactly as theory predicts: R-hat's excess over 1 shrinks with effective sample size. **Above ESS 1,600 there is not a single exceedance in more than 67,000 parameters.**

## 4. The retraction: parameter count does not drive the maximum

The multiplicity hypothesis is wrong for this family, and two comparisons kill it outright:

| model    | parameters | min ESS | max R-hat (parameters) | passes |
| -------- | ---------: | ------: | ---------------------: | ------ |
| VG08     |        856 |     938 |                 1.0046 | ✅     |
| VG15     |      4,803 |   1,335 |                 1.0077 | ✅     |
| VG12     |      5,859 |   3,371 |                 1.0017 | ✅     |
| **VG13** | **11,068** |   1,247 |             **1.0054** | ✅     |
| **VG11** | **14,597** |   1,139 |             **1.0125** | ❌     |

**VG12 carries seven times VG08's parameters and is three times cleaner.** And the decisive one: **VG13 has nearly VG11's parameter count and nearly its minimum ESS, and passes comfortably.** Size does not explain VG11's failure, and neither does sampling volume. VG11 is a genuine outlier — the family's worst-mixing fit on its worst parameter.

Worry 2 also fails as stated. The implied ESS demand is not extreme: models pass routinely at minimum ESS between 900 and 1,700. It is only below roughly 800 that exceedances appear at all, and the explicit floor of 400 sits below that — the two halves of the gate are not in conflict, they are merely calibrated at different points on the same curve.

Worry 3 (reparameterisation) is not tested here and remains true in principle. It is simply not doing any work: if the parameter-space screen almost never fires, its non-invariance costs nothing in practice.

**One earlier claim of mine is withdrawn.** In the course of this investigation I described the A1 recovery replicate `r02` as showing "the same dissociation as VG11". It does not: its minimum ESS is 347, below the explicit floor, so it fails the ESS gate independently and is a genuinely under-sampled fit. It is not evidence of the rule misfiring, and the argument must rest only on fits that clear the ESS floor.

## 5. What does survive

One finding stands, and it is the one the VG11 exception rests on: **the parameter screen and the reported quantities can dissociate.**

| VG11 quantity                    |  max R-hat | min ESS | points above 1.01 |
| -------------------------------- | ---------: | ------: | ----------------- |
| `f_plot` / `p_plot` (trajectory) | **1.0032** |   3,407 | **0 of 500**      |
| `f_query` / `p_query`            | **1.0028** |   3,409 | **0 of 8**        |
| `kappa_plot` / `kappa_query`     | **1.0019** |   4,364 | **0 of 500**      |
| `g_unit_hsgp_coeffs[4]`          |     1.0125 |   1,139 | —                 |

Individual HSGP basis coefficients trade off against one another and are weakly identified; the function they sum to is not, and only the function is reported. So a fit can miss the gate on a coordinate nobody reads while every published number is converged. That is a real property of the rule — but §3 shows it is a **rare** property, not a systematic one, and it is the justification for a per-model exception rather than for changing the rule.

## 6. VG11 specifically: no pathology to find

The obvious follow-up is whether VG11's failure signals a misspecification that a better model would fix. It does not appear to.

**The HSGP basis is correctly sized.** Coefficients 9–15 sit at their prior (sd 0.90–1.00, |mean| ≤ 0.29), so the expansion has saturated and 16 functions is enough. Coefficients 0–1 are also prior-like, as expected: the GP is anchored and orthogonalised against the explicit logit-linear trend, so the lowest frequencies are already absorbed by it. The informed band is 2–8, and the failing coefficient 4 is simply the dominant one.

**The geometry is mild.** `corr(g[4], eta) = −0.286` and `corr(g[4], ell) = +0.131`, with essentially zero correlation against `tau_subject`, `v_total`, the subject variance share and the intercept. A modest amplitude-versus-coefficient trade-off is intrinsic to a Gaussian process; there is no funnel here and no obvious reparameterisation to reach for.

**No chain is stuck and there is no multimodality**: per-chain SDs are 0.76–0.84 and per-chain means span 0.19. The sampler is otherwise healthier than two models already published with caveats — 16 divergences in 48,000 draws (0.033%, spread 5/1/4/1/4/1) and BFMI 0.359–0.395 on every chain, against VG12's 0.215 and VG13's 0.242.

So VG11 is a correctly-specified model whose dominant GP direction mixes slowly. The only honest remedy is **more draws** — roughly double, to lift that coefficient's ESS from 1,139 past about 2,300. More *chains* would buy the same ESS at the same wall-clock, but twelve will not fit in 251 GB.

## 7. Consequences

**The gate should not be loosened.** It fires on one fit in this whole family, and that fit is the one thing in the family that genuinely mixes badly. That is a rule doing its job.

**The VG11 exception stands on its recorded justification, and only that.** [`202608142000`](202608142000-refit-run-record-and-disk-failure.md) §5a records the acceptance as resting on the reported quantities converging, the basis coefficients being nuisance directions, and the sampler being healthier than two caveated models. None of those claims depended on the hypothesis refuted here. What changes is emphasis: "provisional, pending a longer refit" is the operative phrase, because VG11 is an outlier rather than a representative case.

**Retiring VG11 would be the wrong response.** The project's retirement precedents are validity (VG06, whose comprehension field was production-only), supersession (VG14, which VG15 does everything of) and vacuity (the sensitivity variants that could no longer vary anything). VG11 meets none. Its estimates are sound, and nothing replaces it: it is the only typically-developing **spoken** model with child random effects, it supplies the TD arm of the DS-versus-TD between-child spread contrast in [`comparison.py`](../src/vocab_growth/comparison.py), and it anchors two Target-8 sensitivities still to run. Retiring a model for a 0.25% threshold excess, while its estimates are sound and irreplaceable, would set a poor precedent. Whether the TD spoken heterogeneity contrast earns its place in the findings is a separate scope question for [#190](https://github.com/dseinternational/vocabulary-growth/issues/190), to be decided on scientific grounds.

## 8. Caveats

- **One family, one sampler.** Every fit here is nutpie/HMC on the same likelihood family. Nothing generalises beyond that.
- **The "parameters" set is a reconstruction**, not the gate's own list: non-grid posterior variables minus scaled random effects with a stored `_raw`. It should match what the gate screens, but it is inferred from the trace rather than read from the model.
- **Exceedance rates in the low-ESS bins rest on few parameters** — 2 below ESS 400 and 29 between 400 and 800 — so those rows describe this family's fits, not the sampling distribution of R-hat.
- **Worry 3 is untested.** Demonstrating the reparameterisation point would need the same model fitted centred and non-centred, which nothing here does.
