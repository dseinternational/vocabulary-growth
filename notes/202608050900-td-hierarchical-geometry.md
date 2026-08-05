# What makes the TD hierarchical models sample badly

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Diagnostic note, 2026-08-05. Written during the reporting-config refit run, from the `rep` traces of VG10, VG12 and VG13 produced by that run. **Nothing here is implemented yet.** Every change proposed in §7 is a model-graph change that would invalidate the fits currently being published, so the four of them are staged for the next reporting iteration rather than applied now. §6 **retracts** a recommendation made earlier in the run.

## 1. The problem

VG12 (TD understood) and VG13 (TD understood + spoken, young window) clear the hard convergence tier — R-hat and effective sample size — but fail the soft tier on energy BFMI: 0.202 and 0.245 against a 0.3 threshold, with VG12 also recording 2 divergent transitions. This is not new to this run; it is the standing reason the typically-developing hierarchical models have repeatedly needed hightune retries, and it is why the `--allow-caveats` publication path was built during this run.

The question this note answers is *which parameters* cause it, measured rather than guessed.

## 2. The BFMI driver is the variance partition

BFMI is a property of the marginal energy distribution, so the parameters that matter are the ones whose draws track the energy. Correlating every scalar parameter in VG12's posterior against `sample_stats.energy`, pooled across the six chains:

| parameter            | corr with energy |
| -------------------- | ---------------: |
| `tau_subject`        |       **−0.812** |
| `kappa_young`        |       **−0.783** |
| `kappa_excess_young` |           −0.591 |
| `a_kappa`            |           −0.573 |
| `b_kappa`            |           +0.354 |
| `kappa_old`          |           −0.347 |
| `p_slope_low`        |           +0.339 |
| `slope`              |           −0.334 |
| `kappa_excess_old`   |           −0.289 |
| `intercept`          |           +0.265 |
| `p_slope_hi`         |           −0.198 |
| `kappa_min`          |           −0.033 |

The subject random-effect scale and the young-age dispersion dominate, and they are directly coupled to each other: `corr(tau_subject, kappa_young) = +0.755`. These two parameters are competing to explain the same variance. Both describe how much observations at a given age scatter around the population trajectory — `tau_subject` attributes that scatter to persistent between-child differences, `kappa` to within-child measurement noise — and the data cannot cleanly separate them.

`kappa_excess_young` and `a_kappa` appear high in the table but are not independent findings: `a_kappa` is a `pm.Deterministic` function of the anchored dispersion parameters, so `corr(kappa_excess_young, a_kappa) = 0.964` is structural, not pathological.

## 3. What is *not* the problem

**The study scale.** `tau` is VG12's *worst-mixing* parameter — ESS 1313, against 2279 for `kappa_young` and 2325 for `tau_subject` — which made it the obvious first suspect. It is not the energy driver: its correlation with energy is **−0.023**, ranking it 13th, below every parameter in the table above. Centring the study block will therefore fix an ESS problem and is worth doing for that reason, but should not be expected to move BFMI.

**The HSGP.** The lengthscale is comfortably interior to its bounds — `ell_unit` posterior mean 0.405, 89% equal-tailed interval [0.160, 0.676] — so there is no boundary pathology and no case for changing the basis size or the domain.

**Per-draw GP anchoring.** See §6; the TD models already have it.

## 4. Why: singletons, and a falsification test

The mechanism is the ratio of singly- to repeatedly-measured children. A child observed once contributes a single residual that the model can attribute *either* to that child's random intercept *or* to dispersion, with no information to choose. Only children measured more than once carry the within-child replication that identifies the split. The more singletons, the flatter the ridge.

If that is the cause, then a model with a *higher* repeat rate should show a *weaker* version of the same signature. VG10 (Down syndrome, understood + spoken, subject REs on both) is the natural test: different population, different outcome, different structure, fitted by the same engine.

|                            | repeat-measured subjects | `corr(tau_subj, kappa_young)` | `corr(tau_subj, energy)` | min BFMI |
| -------------------------- | -----------------------: | ----------------------------: | -----------------------: | -------: |
| **VG10** (DS, `_u` block)  |       308 / 737 = **42%** |                    **+0.273** |                   −0.444 |    0.465 |
| **VG12** (TD)              |     1000 / 5819 = **17%** |                    **+0.755** |                   −0.812 |    0.202 |

Two-and-a-half times the repeat rate, roughly a third of the coupling, half the energy correlation, and more than double the BFMI. The prediction holds on a model that shares none of VG12's population, outcome or structure, which is much stronger evidence than the within-VG12 correlation alone. The TD pool is large but shallow — 7,052 observations across 5,819 children, 1.21 observations per child — and it is the shallowness, not the size, that hurts.

This also explains why the problem is intrinsic rather than a tuning failure: no amount of extra tuning creates within-child replication that the data do not contain.

## 5. A second, separate finding: the TD `eta` priors were never recalibrated

The 2026-08-04 pass recorded in [202608041730](202608041730-ds-spoken-q-trajectory-prior.md) widened the GP amplitude priors for the Down syndrome joint models, on the finding that they sat at prior CDF 0.95–0.99 with contraction 0.03–0.16. That pass did not reach the TD models. Computing the same diagnostic (contraction = 1 − posterior sd / prior sd, prior `HalfNormal(sigma)`):

| model | parameter | sigma | posterior mean | posterior sd | prior CDF | contraction |
| ----- | --------- | ----: | -------------: | -----------: | --------: | ----------: |
| VG12  | `eta`     |  0.50 |          0.855 |        0.269 | **0.913** |   **0.106** |
| VG13  | `eta_u`   |  0.40 |          0.324 |        0.222 |     0.582 |       0.077 |
| VG13  | `eta_q`   |  0.20 |          0.159 |        0.121 |     0.573 |  **−0.003** |
| VG10  | `eta_u`   |  0.60 |          0.862 |        0.243 |     0.849 |       0.327 |
| VG10  | `eta_q`   |  0.80 |          0.737 |        0.280 |     0.643 |       0.419 |

VG12's `eta` shows the same signature that justified the DS widening: pressed into the prior's upper tail with the data barely narrowing it. VG13's two amplitudes are differently wrong — well-centred, but with contraction at or below zero, meaning the posterior is no narrower than the prior and the amplitudes are reported straight back from it. VG10, after the widening, is the contrast case: both amplitudes sit mid-prior with contraction three to four times higher.

**This is a calibration defect, not the geometry defect.** Neither `eta` nor `ell_unit` appears among VG12's energy correlates, so widening the amplitude priors should not be expected to improve BFMI. It is worth doing because the reported GP amplitudes are currently prior artefacts, which is a reporting problem in its own right.

## 6. Withdrawn: the GP-anchoring proposal, in both its forms

Earlier in this run I recommended "porting the per-draw GP anchoring to the TD trio" as a geometry fix. **That recommendation was wrong and is withdrawn entirely.** It failed twice, for two different reasons, and the second is the interesting one.

First, the TD trio already has it: VG11 and VG12 set `anchor_g_at_ref=True`, VG13 sets `anchor_g_u_at_ref` and `anchor_g_q_at_ref`. The models *without* anchoring are VG01–VG05, VG07, VG08, VG09 and VG14.

That suggested redirecting the proposal to VG08 and VG09 — the only two models in this run to fail the gate at plain `rep` and need hightune retries, and two of the three subject-RE models lacking the anchoring. **That redirection is also wrong.** VG05 → VG07 → VG08 → VG09 → VG10 is a deliberate one-change-at-a-time ladder, documented as such in the model inventory, and the per-draw GP anchor at 54 months **is the single change that distinguishes VG10 from VG09** — VG10's `config_name` is literally `…-uq-anchored` against VG09's `…-uq`. Enabling the anchor on VG09 would collapse VG09 into VG10 and destroy the very contrast that produced the evidence for anchoring in the first place. Enabling it on VG08 would break the ladder's one-change-per-rung structure.

The ladder has already answered the question the proposal was trying to ask. VG10 is the anchored model, it exists, it converges cleanly at `rep`, and it is the headline model of record for the Down syndrome joint family. VG08 and VG09 are diagnostic rungs whose job is to be worse; their needing a longer tuning budget is part of what the ladder demonstrates, not a defect in them to be repaired.

## 7. Proposed changes, ranked

1. **Reparameterise the variance partition.** Sample the total between-observation variance and the *share* attributable to persistent child differences, rather than `tau_subject` and `kappa` competing for the same variance. This turns the diagonal ridge in §2 into an axis, separating a strongly-identified quantity (total scatter) from a weakly-identified one (the split), and puts the prior on the split, which is the honest place for it. `tau_subject` and `kappa` remain recoverable as deterministics, so the DS/TD heterogeneity comparison survives unchanged. This is the only proposal that targets the measured driver.
2. **Centre the study random-effect block.** With thousands of observations per study the non-centred parameterisation is the wrong side of the funnel trade-off. Expect the `tau` ESS to improve; do not expect BFMI to move (§3).
3. **Widen the TD `eta` priors** (§5). Corrects reported GP amplitudes; no expected geometry effect.
4. ~~**Enable per-draw GP anchoring.**~~ Withdrawn — see §6.

All three surviving changes are graph changes requiring refits of the affected models. Items 1 and 2 are implemented behind definition fields defaulting to off, so that enabling them per model is a deliberate, reviewable act and no existing model's graph changes silently.

### Implementation status, 2026-08-05

Implemented on branch `geometry/variance-partition`, in an isolated worktree so the reporting run in flight is untouched:

| change              | state                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 1. Variance partition | `SubjectVariancePartitionParams` + `gp_utils.build_variance_partition`; 12 tests. Calibrated for VG12 as `_TD_UNDERSTOOD_VARIANCE_PARTITION` but **not attached** to any registered model. |
| 2. Centred study block | `UnivariateModelDefinition.centred_study_re`; 6 tests including prior equivalence at two study counts. Not enabled on any model. |
| 3. TD `eta` widening  | VG12 `eta_sigma` 0.5 → 1.0. **VG13 deliberately not changed** (§5).                                              |
| 4. GP anchoring       | Withdrawn (§6). No code change.                                                                                  |

Nothing is attached to a registered model, because none of it is validated yet — see §9 item 1, and the measurement below.

**Why none of this could land on `main` when it was written.** A fit is validated for render, sync and publication by comparing its manifest against the *current* registered definition, field for field (`fit_artifacts.validate_fit_output`). The definition is serialised with `dataclasses.asdict`, so **adding any field at all** — even one defaulting to off, even one no model sets — changes the serialised form and invalidates every existing fit of that definition class. VG12's stored definition matches the current one at exactly 30 keys; a 31st breaks it. The three surviving changes therefore have to wait for the TD models to be published, and cannot be staged incrementally on `main` in the meantime.

### Rejected

**Restricting subject REs to repeatedly-measured children.** This would identify the split cleanly but is not available: the DS/TD heterogeneity contrast is a reported estimand, and dropping singletons from one population and not the other would make the two sides incomparable. It is worse than it first appears — the singletons' variance would not disappear, it would migrate into `kappa`, which is *itself* a reported cross-population quantity.

**Making study effects fixed rather than random.** Removes `tau` entirely, but forfeits partial pooling across studies and the population-level trajectory's interpretation as a study-averaged curve. Since `tau` is not the energy driver (§3), this would pay a real interpretive cost for a benefit that the measurements do not support.

## 8. Consequence for the DS/TD heterogeneity contrast

Low BFMI means the sampler explored the energy distribution's tails poorly, and tail quantiles are what interval bounds are made of. VG12's and VG13's reported posterior *intervals* are therefore the least trustworthy part of those fits, while their point estimates are comparatively safe. `tau_subject`'s interval is strikingly tight (0.686 ± 0.015) for a parameter whose identification rests on 17% of children; that precision comes substantially from the Beta-Binomial functional form separating child variance from dispersion, not from replication in the data.

Because the DS side of the comparison has markedly better energy behaviour (§4), **the contrast is asymmetrically affected**: the TD interval is the less reliable of the two. Any reported DS/TD heterogeneity difference should carry that caveat in the text, not only in the convergence appendix.

## 9. Open items

1. Whether the §7.1 reparameterisation actually improves BFMI is untested. It should be measured at `test` config on VG12 under a separate `--output-dir`, against the current parameterisation, before any reporting-quality refit is committed to.
2. The residual noted in the DS/TD dispersion comparator fix still stands: VG10's `kappa_s` is the dispersion of the production ratio `q` conditional on understood, while VG11's is the dispersion of spoken counts marginally. These are not the same quantity, and the comparator now pairs them. This predates the fix and remains unaudited.
3. VG13's `eta_q` contraction of −0.003 may indicate the short young-age window simply cannot inform a GP amplitude, in which case widening the prior is the wrong response and fixing the amplitude would be better. Not investigated.
