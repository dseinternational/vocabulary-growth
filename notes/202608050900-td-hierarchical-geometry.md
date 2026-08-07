# What makes the TD hierarchical models sample badly

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Diagnostic and implementation note, 2026-08-05. Written during the reporting-config refit run, from the `rep` traces of VG10, VG12 and VG13 produced by that run, and revised the same day once the proposals were tried.
>
> **Two of the four proposals did not survive.** §6 withdraws the GP-anchoring proposal entirely — twice wrong, and the second reason is the interesting one. §9 records a `test`-config trial on VG12 which **falsifies §7.1's central claim**: the variance partition does not fix the energy BFMI, because the ridge rotates rather than dissolving. It is retained for a 4× reduction in divergences, not as a BFMI remedy. Centring the study block is a clear win (22× ESS on `tau`) and both are now enabled on VG11 and VG12.
>
> §10 records an OOM that killed a 7h20m VG11 fit, and why subsampling the typically-developing pool — the obvious memory lever — would make the geometry worse.

## 1. The problem

VG12 (TD understood) and VG13 (TD understood + spoken, young window) clear the hard convergence tier — R-hat and effective sample size — but fail the soft tier on energy BFMI: 0.202 and 0.245 against a 0.3 threshold, with VG12 also recording 2 divergent transitions. This is not new to this run; it is the standing reason the typically-developing hierarchical models have repeatedly needed hightune retries, and it is why the `--allow-caveats` publication path was built during this run.

The question this note answers is _which parameters_ cause it, measured rather than guessed.

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

## 3. What is _not_ the problem

**The study scale.** `tau` is VG12's _worst-mixing_ parameter — ESS 1313, against 2279 for `kappa_young` and 2325 for `tau_subject` — which made it the obvious first suspect. It is not the energy driver: its correlation with energy is **−0.023**, ranking it 13th, below every parameter in the table above. Centring the study block will therefore fix an ESS problem and is worth doing for that reason, but should not be expected to move BFMI.

**The HSGP.** The lengthscale is comfortably interior to its bounds — `ell_unit` posterior mean 0.405, 89% equal-tailed interval [0.160, 0.676] — so there is no boundary pathology and no case for changing the basis size or the domain.

**Per-draw GP anchoring.** See §6; the TD models already have it.

## 4. Why: singletons, and a falsification test

The mechanism is the ratio of singly- to repeatedly-measured children. A child observed once contributes a single residual that the model can attribute _either_ to that child's random intercept _or_ to dispersion, with no information to choose. Only children measured more than once carry the within-child replication that identifies the split. The more singletons, the flatter the ridge.

If that is the cause, then a model with a _higher_ repeat rate should show a _weaker_ version of the same signature. VG10 (Down syndrome, understood + spoken, subject REs on both) is the natural test: different population, different outcome, different structure, fitted by the same engine.

|                           | repeat-measured subjects | `corr(tau_subj, kappa_young)` | `corr(tau_subj, energy)` | min BFMI |
| ------------------------- | -----------------------: | ----------------------------: | -----------------------: | -------: |
| **VG10** (DS, `_u` block) |      308 / 737 = **42%** |                    **+0.273** |                   −0.444 |    0.465 |
| **VG12** (TD)             |    1000 / 5819 = **17%** |                    **+0.755** |                   −0.812 |    0.202 |

Two-and-a-half times the repeat rate, roughly a third of the coupling, half the energy correlation, and more than double the BFMI. The prediction holds on a model that shares none of VG12's population, outcome or structure, which is much stronger evidence than the within-VG12 correlation alone. The TD pool is large but shallow — 7,052 observations across 5,819 children, 1.21 observations per child — and it is the shallowness, not the size, that hurts.

This also explains why the problem is intrinsic rather than a tuning failure: no amount of extra tuning creates within-child replication that the data do not contain.

> [!IMPORTANT]
> **Read the rate claim with §9's qualification.** VG11 was refitted after this section was written and clears the BFMI threshold despite having the _lowest_ repeat rate of the three typically-developing models (13.4%). It has roughly twice VG12's within-child replication in absolute terms — 1,947 repeatedly-measured children to 1,000. The mechanism holds, but the operative quantity is the **absolute amount** of within-child replication, not the proportion of children who have any.

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

First, the TD trio already has it: VG11 and VG12 set `anchor_g_at_ref=True`, VG13 sets `anchor_g_u_at_ref` and `anchor_g_q_at_ref`. The models _without_ anchoring are VG01–VG05, VG07, VG08, VG09 and VG14.

That suggested redirecting the proposal to VG08 and VG09 — the only two models in this run to fail the gate at plain `rep` and need hightune retries, and two of the three subject-RE models lacking the anchoring. **That redirection is also wrong.** VG05 → VG07 → VG08 → VG09 → VG10 is a deliberate one-change-at-a-time ladder, documented as such in the model inventory, and the per-draw GP anchor at 54 months **is the single change that distinguishes VG10 from VG09** — VG10's `config_name` is literally `…-uq-anchored` against VG09's `…-uq`. Enabling the anchor on VG09 would collapse VG09 into VG10 and destroy the very contrast that produced the evidence for anchoring in the first place. Enabling it on VG08 would break the ladder's one-change-per-rung structure.

The ladder has already answered the question the proposal was trying to ask. VG10 is the anchored model, it exists, it converges cleanly at `rep`, and it is the headline model of record for the Down syndrome joint family. VG08 and VG09 are diagnostic rungs whose job is to be worse; their needing a longer tuning budget is part of what the ladder demonstrates, not a defect in them to be repaired.

## 7. Proposed changes, ranked

1. **Reparameterise the variance partition.** Sample the total between-observation variance and the _share_ attributable to persistent child differences, rather than `tau_subject` and `kappa` competing for the same variance. This turns the diagonal ridge in §2 into an axis, separating a strongly-identified quantity (total scatter) from a weakly-identified one (the split), and puts the prior on the split, which is the honest place for it. `tau_subject` and `kappa` remain recoverable as deterministics, so the DS/TD heterogeneity comparison survives unchanged. This is the only proposal that targets the measured driver.

   > [!CAUTION]
   > **The last sentence is wrong.** Measured on VG12 (§9), the partition does not improve the energy BFMI at all. The rotation works — the total samples cleanly — but the _share_ inherits the entire energy correlation. The ridge does not dissolve; it rotates. The change is retained for a 4× reduction in divergences, not as a BFMI remedy.

2. **Centre the study random-effect block.** With thousands of observations per study the non-centred parameterisation is the wrong side of the funnel trade-off. Expect the `tau` ESS to improve; do not expect BFMI to move (§3).
3. **Widen the TD `eta` priors** (§5). Corrects reported GP amplitudes; no expected geometry effect.
4. ~~**Enable per-draw GP anchoring.**~~ Withdrawn — see §6.

All three surviving changes are graph changes requiring refits of the affected models. Items 1 and 2 are implemented behind definition fields defaulting to off, so that enabling them per model is a deliberate, reviewable act and no existing model's graph changes silently.

### Implementation status, 2026-08-05

Implemented on branch `geometry/variance-partition`, trialled at `test` config (§9), then merged and **enabled on VG11 and VG12**:

| change                 | state                                                                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Variance partition  | `SubjectVariancePartitionParams` + `gp_utils.build_variance_partition`. **Enabled** on VG11 and VG12, each calibrated for its own outcome level. Kept for divergences, not BFMI — §9. |
| 2. Centred study block | `UnivariateREModelDefinition.centred_study_re`. **Enabled** on VG11 and VG12. Prior equivalence tested at two study counts.                                                           |
| 3. TD `eta` widening   | VG12 `eta_sigma` 0.5 → 1.0. **VG13 deliberately not changed** (§5).                                                                                                                   |
| 4. GP anchoring        | Withdrawn (§6). No code change.                                                                                                                                                       |

20 tests across the two mechanisms. VG11 and VG12 require refits; VG13 and the eleven other registered models are untouched.

**Why the fields sit on a subclass.** A fit is validated for render, sync and publication by comparing its manifest against the _current_ registered definition, field for field (`fit_artifacts.validate_fit_output`). The definition is serialised with `dataclasses.asdict`, so **adding any field at all** — even one defaulting to off, even one no model sets — changes the serialised form and invalidates every existing fit of that class. VG12's stored definition matched at exactly 30 keys; a 31st broke it, and it broke VG01–VG04 too, four published models with no random effects whatever, VG03 alone a 2h50m refit.

The two options therefore live on a new `UnivariateREModelDefinition` subclass carried only by VG11 and VG12. VG01–VG04 stay at 30 keys and stay valid — verified directly against their manifests. The engine reads both fields through `getattr`, so a plain definition still builds; VG17 derives from VG01 and never becomes a subclass instance.

### Considered and rejected: removing the subject random effects

Raised by the study owner on 2026-08-07, on reasonable grounds — 1.21 observations per child, a BFMI failure, poor parameter recovery, and a reparameterisation that did not help. If a parameter cannot be identified, why carry it?

**The premise does not hold on its own.** VG11 has essentially the same observations per child (1.27) and a _lower_ repeat fraction (13.4% against VG12's 17.2%), carries subject random effects on the same engine, and clears BFMI comfortably at 0.359–0.390. Replication per child is therefore not what decides whether subject effects are viable; the absolute count is — 1,947 repeat-measured children against 1,000.

**The decisive objection is where the variance would go.** `tau_subject` is not a reported estimand anywhere: it appears only in per-model diagnostic prose and internal scripts. The DS/TD heterogeneity contrast is carried by **`kappa`**, `sd_Y` and the overdispersion ratio. Removing subject effects does not delete between-child variance — it migrates it into `kappa`, which is exactly the quantity the contrast reports.

So dropping them from VG12 while VG10 keeps them would give:

- VG10's `kappa` = within-child noise
- VG12's `kappa` = within-child noise **plus** between-child variance

contrasted against one another. **That is the defect fixed on 2026-08-05, in mirror image.** `DISP_DS_KEY` pointed at VG07, which has no subject effects, against typically-developing models that have them, and the contrast was inverted as a result. Removing them from the typically-developing side would reintroduce the same asymmetry from the other direction, and this time it would look deliberate.

Removing them from _both_ populations avoids that, but forfeits partial pooling, understates uncertainty for the 17% of children with repeat administrations, and changes VG10 — which converges cleanly and recovers well (coverage 0.867). Changing a model that is not broken to fix one that is, is the wrong trade.

**The general point.** The BFMI failure, the poor recovery and the `tau_subject`/`kappa` ridge are three symptoms of one cause: missing within-child replication. Removing the parameter does not supply the information. It relocates the problem into a quantity that _is_ reported, and hides it.

Checked rather than argued: VG12's registered `single-admin` variant — one administration per child, which makes subject effects unidentifiable by construction — was run on 2026-08-07. It had never been run before; the Target 8 comparison reported it as "Variant fit not found (skip)".

### Rejected

**Restricting subject REs to repeatedly-measured children.** This would identify the split cleanly but is not available: the DS/TD heterogeneity contrast is a reported estimand, and dropping singletons from one population and not the other would make the two sides incomparable. It is worse than it first appears — the singletons' variance would not disappear, it would migrate into `kappa`, which is _itself_ a reported cross-population quantity.

**Making study effects fixed rather than random.** Removes `tau` entirely, but forfeits partial pooling across studies and the population-level trajectory's interpretation as a study-averaged curve. Since `tau` is not the energy driver (§3), this would pay a real interpretive cost for a benefit that the measurements do not support.

## 8. Consequence for the DS/TD heterogeneity contrast

Low BFMI means the sampler explored the energy distribution's tails poorly, and tail quantiles are what interval bounds are made of. VG12's and VG13's reported posterior _intervals_ are therefore the least trustworthy part of those fits, while their point estimates are comparatively safe. `tau_subject`'s interval is strikingly tight (0.686 ± 0.015) for a parameter whose identification rests on 17% of children; that precision comes substantially from the Beta-Binomial functional form separating child variance from dispersion, not from replication in the data.

Because the DS side of the comparison has markedly better energy behaviour (§4), **the contrast is asymmetrically affected**: the TD interval is the less reliable of the two. Any reported DS/TD heterogeneity difference should carry that caveat in the text, not only in the convergence appendix.

## 9. The trial, and what it falsified

Four VG12 arms at `test` config, in a throwaway output root, with the `eta` change held out of the geometry arms so the two are not confounded:

| arm           | min BFMI | divergences |  max R-hat | ESS `tau` | ridge |
| ------------- | -------: | ----------: | ---------: | --------: | ----: |
| baseline      |    0.203 |          59 |     1.0133 |       310 | 0.747 |
| eta           |    0.202 |          76 |     1.0131 |       394 | 0.760 |
| **centred**   |    0.194 |          31 | **1.0057** | **6,950** | 0.758 |
| **partition** |    0.192 |      **14** |     1.0132 |       396 | 0.755 |

**Energy BFMI does not move.** 0.192 to 0.203 across every arm, against a 0.3 threshold. The energy correlations say why:

| arm       | `tau_subject` | `kappa_young` |  `v_total` |    `share` |
| --------- | ------------: | ------------: | ---------: | ---------: |
| partition |        −0.806 |        −0.786 | **−0.025** | **−0.737** |

The reparameterisation did exactly what it was designed to do — `v_total`, the strongly-identified direction, is now clean. But `share` inherited the whole energy correlation. **§7.1's claim that this targets the measured driver is false.** No change of coordinates can help, because the problem is not bad coordinates: `share` is the weakly-identified quantity, and only within-child replication can identify it. That is a _confirmation_ of the §4 mechanism reached by refuting the remedy it suggested, which is worth more than the fix would have been.

Two things did work, and both are adopted for VG11 and VG12:

- **Centring**: 22× the ESS on `tau` (310 → 6,950), max R-hat 1.0133 → 1.0057, divergences halved. Exactly the ESS-not-BFMI prediction of §3.
- **The partition**: divergences 59 → 14. Kept for that alone.

`tau_subject` is identical across all four arms — 0.6862 to 0.6877, sd ~0.0146, near-identical 89% intervals. The reparameterisations are faithful, and §8's caveat stands: that interval's tightness is not a parameterisation artefact, and it is not evidence that the split is well identified.

### Consequence

VG12 and VG13 still fail the soft tier on BFMI and will still need the disclose-and-publish path. That is now a measured property of the data rather than an untried hypothesis: **the only remedy is more repeat measurement**, which no reparameterisation, prior, or tuning budget can substitute for.

### VG11's refit — and a qualification to §4

VG11 was refitted at plain `rep` carrying both changes, and the result does not fit the story above cleanly:

| check           | VG11                        |
| --------------- | --------------------------- |
| R-hat           | passes, max 1.0068          |
| ESS             | passes; `tau` = **27,017**  |
| **Energy BFMI** | **passes — 0.359 to 0.390** |
| Divergences     | fails — 22                  |

**VG11 clears the BFMI threshold that VG12 and VG13 fail.** §4 predicted the opposite: VG11 has the _lowest_ repeat rate of the three at 13.4%, against VG12's 17.2%.

The rate is the wrong measure. In absolute terms VG11 has **1,947 repeatedly-measured children to VG12's 1,000**, over 18,522 observations to 7,052 — roughly twice the within-child replication and 2.6× the data, despite the lower fraction. Its `subject_variance_share` is correspondingly well identified (posterior sd 0.0105). So §4's mechanism survives, but it should be read as _the absolute quantity of within-child replication_, not the proportion of children who have any.

That reading is not clean either: VG10 has only 308 repeat-measured children and the best BFMI of all (0.465). VG10 is a different engine on a different population, so the cross-engine comparison is weak — but it means the absolute-count reading is a better description than the rate, not a law.

**What cannot be claimed:** that centring and the partition fixed VG11's BFMI. VG11 has never been fitted without them, so there is no baseline. It may simply be an easier model. The `test`-config trial on VG12 (above) is the only controlled comparison available, and there the changes did not move BFMI at all.

VG11 still misses the soft tier on 22 divergent transitions, so it publishes through the caveated path — for divergences, not for energy.

## 10. Operational: VG11 is memory-bound, and subsampling is the wrong lever

VG11's first attempt at `rep`-hightune was killed by the OOM killer after 7h20m, at 247 GB anon-RSS on a 251 GB machine. Four concurrent `test`-config arms were running alongside it and brought the crash forward, but VG11 alone accounted for essentially the whole machine.

The scaling is arithmetic, not bad luck. Memory is dominated by observation-sized deterministics stored per draw, so it goes as `n_obs × draws`. VG11 has 18,522 rows against VG12's 7,052 and was running 48,000 draws against VG12's 36,000 — 3.5× the storage. VG12's successful `rep` fit implies ~65 GB; 3.5× that is ~227 GB, which matches what the kernel recorded.

Plain `rep` is the cheapest reporting-quality configuration available, because `_sampling_parameter_errors` treats chains, draws, tune and `target_accept` as **minimums** — dropping to four chains would fail the reporting-quality gate. 36,000 draws puts the estimate near 185 GB, which fits if nothing else runs.

**Subsampling the typically-developing pool is a real memory lever but the wrong one here.** `sample_fraction` already draws a fraction of _subjects_ and keeps all their administrations — `data_utils` records why, including that a 10% row-wise draw of VG11 produces exactly this family's pathology. Even done correctly, it removes repeat-measured children, and §10 has just established that within-child replication is the binding constraint on the geometry. It would buy memory by making the thing we are trying to fix worse. The order of resort is: plain `rep` alone; then stop storing the observation-sized deterministics (`f_obs`, `p_obs`, `kappa_obs`, `z_obs`), which are recomputable functions of the parameters and cost nothing statistically; and only then subsample.

## 11. Open items

1. ~~Whether the §7.1 reparameterisation actually improves BFMI is untested.~~ Answered in §9: it does not. What remains open is whether the BFMI can be improved **at all** without more repeat measurement. Nothing tried so far touches it, and §4's mechanism predicts that nothing will.
2. VG11's share prior is calibrated blind. VG12's share prior could be checked against a posterior (0.598, at prior CDF 0.368); VG11 has never completed a fit, so its share prior is weak by construction rather than by evidence. Revisit once the refit lands — if VG11's implied share sits in the prior's tail, recalibrate as VG12's was.
3. Whether the observation-sized deterministics (`f_obs`, `p_obs`, `kappa_obs`, `z_obs`) need storing at all. They dominate VG11's memory (§10) and are recomputable functions of the parameters. Dropping them costs nothing statistically and would remove the memory ceiling that currently constrains the TD models' sampling configuration.
4. The residual noted in the DS/TD dispersion comparator fix still stands: VG10's `kappa_s` is the dispersion of the production ratio `q` conditional on understood, while VG11's is the dispersion of spoken counts marginally. These are not the same quantity, and the comparator now pairs them. This predates the fix and remains unaudited.
5. VG13's `eta_q` contraction of −0.003 may indicate the short young-age window simply cannot inform a GP amplitude, in which case widening the prior is the wrong response and fixing the amplitude would be better. Not investigated.
