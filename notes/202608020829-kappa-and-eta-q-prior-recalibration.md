# Dispersion and `q`-GP prior recalibration: `b_kappa_mag`, `kappa_min`, `eta_q`

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Analysis and implementation note, 2026-08-02. Seven things are **implemented**: the production `kappa` priors on VG01, VG03 and VG11 (§6), the subject-wise `sample_fraction` fix (§13), the VG10 GP anchoring on VG16 (§16 — a structural change to a registered model), the two-anchor `kappa` reparameterisation on VG01, VG03 and VG11 (§18), which **supersedes §6 on those three models**, the conditional `kappa` calibration on VG11, VG12 and VG13 (§19), which **supersedes §18's prior values on VG11**, the marginal calibration of the two comprehension models VG02 and VG04 (§20), the Down syndrome joint dispersion calibration on VG09, VG10, VG15 and VG16 (§22), which **supersedes §19's decision to exclude them**, and the family-wide subject random-effect scale (§23). §§4–5 (`kappa_min` family-wide, `eta_q`) remain **proposals**. The supporting fits in §§2–8 are `dev` configuration (2 chains x 500 draws) on a 10% typically-developing subsample and are indicative only; §§10, 12, 13, 16, 18, 19, 22 and 23 supersede them with `test`-config fits.
>
> **Read §§18–23 before acting on any `kappa` recommendation in §§2–17.** Neither the parameterisation nor the calibration those sections describe is still in use on the migrated models. Every model that carries an empirical dispersion calibration is now on the two-anchor form; the four that are not (VG05, VG07, VG08, VG14) are named and reasoned in §22.
>
> **Every `kappa` figure before §19 is a _marginal_ estimate**, valid only for the models with no grouping structure — VG01 to VG04. A model with study and subject random intercepts needs the conditional estimate, which is three to ten times larger. Reading a number from §§1–18 across to VG11, VG12 or VG13 reintroduces exactly the error §19 exists to correct.
>
> **§21 is a caveat on how every `kappa` in this note should be read.** On the understood outcomes it is not a pure dispersion: counts collected on 396- and 418-item forms are scored out of 810, which compresses the modelled scale as children work up a form, and the models' age-constant subject scale cannot follow that — so `kappa(age)` absorbs it. The priors are unaffected (the calibration mirrors the model, as it must), but "typically-developing comprehension dispersion rises with age" describes the parameter, not children.

> [!IMPORTANT]
> Extended the same day with §§10–17. The recalibration works — VG03 on the corrected frame puts `b_kappa_mag` at 1.369 with contraction 0.82, against prior CDF 1.00 and contraction 0.18 before (§13) — but `b_kappa_mag` remains censored in VG01 (§10). The validation also turned up a bimodality in VG11, which a control fit cleared the recalibration of causing (§10). §11 identifies the cause: **`sample_fraction` drew rows rather than children, cutting replication from 1.32 to 1.04 administrations per child and leaving the subject random effect and the observation-level dispersion indistinguishable.** §12 confirms it — a subject-wise draw at the same volume gives max R-hat 1.010 against 1.718 — and **corrects §11's ranking of the two modes, which was wrong**: the comparator was a published US-English monolingual norm matching this pool's mean rather than its median. §13 records the fix, its tests, and the VG03 and VG04 refits. Results for VG11, VG12 and VG13 obtained under row-wise subsampling should be discarded, including the `dev` run in §2. §15 then diagnoses the two remaining open items — VG01's censored `b_kappa_mag` is a units problem, and the Down syndrome models' elevated R-hat is the known understood-trajectory ridge — and §16 implements the one action that came out of it, giving VG16 the VG10 anchoring. **§17 is a review pass over this note's own work** and corrects it in two places: the empirical `b_kappa_mag` in §2 was estimated with the wrong functional form and is biased low (correct values 1.71 and 2.17, not 0.77 and 1.38), which makes the implemented `HalfNormal(0.75)` still too tight and invalidates §13's explanation of the VG03 gap; and `_subsample_subjects` depended on input row order until sorted. It also confirms the `kappa` decline is not a CDI-form artefact.
>
> **§18 then implements the reparameterisation §17 called for, corrects §17 in turn, and turns up a limitation none of the earlier sections saw.** VG01 and VG03 come out clean and centred — every dispersion parameter between prior CDF 0.46 and 0.56, against 0.998 and 0.932 for `b_kappa_mag` before — but **VG11 puts `kappa` at ten times its marginal estimate**, because its subject random effects absorb the between-child spread the Beta-Binomial carries in VG01 and VG03. A dispersion prior calibrated on raw per-age counts does not transfer to a model that conditions on random effects; that is the error §7 identified for `kappa_s` and declined to make, and §6 made it for VG11. **VG01 and VG03 are the two models §18 validates.**
>
> **§19 then closes §18's top open item and finds it was wider than §18 thought.** A conditional (GLMM) estimator — saturated mean, study effects, subject effect integrated out by quadrature — puts `kappa` 3 to 10 times above its marginal counterpart on every random-effect pool measured. VG11's posterior turns out to have been right all along at 310 against a conditional estimate of 317, so re-centring removes a prior conflict without moving the answer. **The new finding is VG12 and VG13**, whose understood dispersion posteriors sat at ~16 against a data estimate of ~42 — prior-dominated rather than prior-conflicted, and so invisible to the check that caught VG11 — and whose dispersion _rises_ with age, which the legacy `b_kappa_mag >= 0` forbids outright. Both are migrated and recalibrated, and their refits move to 42.3 and 42.0 with `b_kappa` posteriors entirely in the region the old form excluded (`+0.211` and `+1.397`). Across all three models every dispersion parameter now sits between prior CDF 0.41 and 0.63. **The Down syndrome joint frame fails the recovery check** — a known `kappa(24) = 317` comes back between 260 and 842 — so VG09/VG10/VG16 are deliberately excluded; that is a limitation, not a clearance. (§22 revisits this and reaches a different verdict: the failure is a measurable one-directional bias, not scatter, so all four are now calibrated as lower bounds.) §19 also disproves §18's claim that `tau_subject` and `kappa` cannot be told apart on this design, and leaves the **subject random-effect scales** as the loudest remaining prior-data conflict (VG13's `tau_subj_q` at prior CDF 0.975).
>
> On the parameterisation itself: The floor-free form §17 simulated is rejected by the data (10 to 168 log-likelihood units across six pools), so `kappa_min + exp(a + bz)` is kept and only its _parameterisation_ changes: priors go on the age term at two reference ages in months, and `a_kappa` / `b_kappa` are derived. That also supplies the argument this note had been missing — the anchor values are stable across age-cell inclusion rules that swing `b_kappa_mag` by 44%, so the legacy form asks for a prior on the one quantity the data do not pin down. §18 also retires the `P(kappa > 200)` criterion that `_PRODUCTION_KAPPA` was tuned against: at `n = 810`, `kappa = 200` is 2.24x the binomial standard deviation, not "near-binomial", and the observed young-age cells really do sit there.

## Summary

A full `dev`-config run of all fifteen models (2026-08-02) surfaced three priors whose posteriors are prior-limited rather than data-limited — the posterior is no narrower than the prior and sits in its extreme upper or lower tail. Fitting a per-age Beta-Binomial (n = 810, matching the likelihood) to the analysis frame recovers the model's own dispersion parameters directly, and shows that two of the three are mis-scaled by roughly a factor of four, in a direction the empirical data identify unambiguously. The third, `eta_q`, is mis-scaled for a different reason: it was tightened for sampler convenience, and the tightening is now doing statistical work it was never justified to do.

The headline prior finding is that **the `b_kappa_mag` prior should not be shared between production and comprehension outcomes**. Production dispersion rises steeply with age; typically-developing comprehension dispersion is flat to slightly _rising_ in `kappa`, which the model's sign constraint cannot represent at all. The shared prior is badly wrong for the former and the wrong shape for the latter.

Validating that change then turned up something with wider consequences than the priors it set out to fix: **`sample_fraction` subsampled rows rather than children**, destroying the within-child replication that identifies a subject random effect and giving VG11 a bimodal posterior at R-hat 1.72. That is now fixed and tested (§13). Every typically-developing result in this note's own §2 was produced under the defective draw.

## 1. Evidence and its independence

The calibration below fits a two-parameter Beta-Binomial `(mu, kappa)` by maximum likelihood to the counts at each whole month of age, then regresses `log kappa` on standardised age `z` (weighted by cell size). Age cells with fewer than 25 administrations, or with a mean proportion outside `(0.01, 0.95)`, are dropped — `kappa` is not identified against the floor or the ceiling, and including those cells produced spurious estimates up to 600.

These are the rows the models train on. This is therefore **scale calibration, not an independent anchor**, and it is a weaker class of evidence than the Wordbank normative deciles used for the trajectory anchors in [`docs/models/PRIORS.md`](../docs/models/PRIORS.md). Two things nonetheless distinguish it from prior-data double-dipping of the kind the `q` anchors were corrected for. First, the target is the prior _scale_, chosen so the prior does not exclude behaviour the data plainly exhibit, and not the prior _location_ read off a posterior. Second, PRIORS.md already admits exactly this evidence class for `kappa` — its "Dispersion (`kappa`)" section derives the sign of the age trend from the same per-age fit. What follows extends that from the sign to the magnitude.

The estimates are marginal: they carry no GP mean, no study effects and no subject effects, all of which absorb spread in the fitted models. A marginal `kappa` is therefore a **lower bound** on the model's `kappa`, and the empirical `kappa_min` in particular should be read as a floor rather than a target.

## 2. What the data say

| Data          | usable age cells | empirical `b_kappa_mag` | fitted posterior (`dev`) | min `kappa` observed |
| ------------- | ---------------- | ----------------------- | ------------------------ | -------------------- |
| DS spoken     | 12 (18–60 mo)    | 1.38                    | VG01 **1.72**            | 4.2 (rho 0.19)       |
| TD spoken     | 20 (11–30 mo)    | 0.77                    | VG03 1.18, VG11 1.24     | 3.0 (rho 0.25)       |
| DS understood | 5 (12–37 mo)     | 0.76                    | VG02 0.73                | 6.4 (rho 0.13)       |
| TD understood | 18 (8–25 mo)     | **−0.04**               | VG04 0.07, VG12 0.05     | 7.5 (rho 0.12)       |

> [!CAUTION]
> The `empirical b_kappa_mag` column is estimated by regressing `log kappa` on `z`, which is **not** the model's parameter and is biased low — see §17. Corrected three-parameter values: DS spoken **2.17**, TD spoken **1.71**, TD understood **−0.89**. The direction and the outcome-specific split are unaffected; the magnitudes are not.

The empirical column uses the full pools throughout and stands. The fitted-posterior column does not: the typically-developing entries come from the `dev` run and were produced under the defective row-wise subsample (§§11–13), and VG11's in particular sits in a spurious mode. Superseded values, on the corrected frame at `test`: VG03 `b_kappa_mag` 1.369 (§13), VG04 0.031 (§13). The qualitative pattern is unchanged by the corrections.

The empirical estimates and the model posteriors agree closely across all four rows, including the two where both are near zero. That agreement is not independent corroboration — both use the same rows — but it does establish that the posteriors are tracking a stable feature of the data rather than sampler pathology, and it is what licenses treating the marginal fit as a guide to the prior scale.

Against the current shared prior `b_kappa_mag ~ HalfNormal(0.3)` — median 0.20, 95th percentile 0.59 — the two production rows sit at prior CDF 0.99–1.00, with contraction (1 − sd_post/sd_prior) of 0.07 for VG01 and 0.18 for VG03 and VG11. The posterior is not narrower than the prior; it is pressed against it.

## 3. The sign constraint, and why the prior must be outcome-specific

The dispersion model is

```text
kappa(z) = kappa_min + exp(a_kappa + b_kappa * z),   b_kappa = -b_kappa_mag <= 0
```

so `kappa` is constrained to be non-increasing in age and dispersion `rho = 1 / (kappa + 1)` non-decreasing. For production that is the right shape and the prior is merely far too tight. For typically-developing comprehension it is the wrong shape: the per-age series runs 11.1 at 8 mo, 11.3 at 12, 12.7 at 16, 11.2 at 22, 15.6 at 24 — flat, with a slight _rise_. VG04 and VG12 respond by piling up against the boundary at 0.05–0.07.

Widening a half-normal here would make matters slightly worse, not better, by moving prior mass into the one direction the data reject. Two coherent options:

- leave comprehension outcomes at `HalfNormal(0.3)`, accepting a boundary pile-up that is at least harmless; or
- make the coefficient signed — `b_kappa ~ Normal(0, 0.5)` — so rising dispersion is representable. This needs an engine change in [`build_utils.build_kappa_of_z`](../src/vocab_growth/models/build_utils.py) and is the better fix, but it is out of scope here.

The first option was taken. §13 confirms the diagnosis on better evidence: refitted at `test` on 2.5 times the data, VG04's `b_kappa_mag` moves _closer_ to the boundary, 0.073 to 0.031 with a 5th percentile of 0.002 — the tightening-onto-a-constraint signature.

Note that DS comprehension (VG02, empirical 0.76) does fall with age, unlike TD comprehension. The two populations are observed over different windows (12–37 vs 8–25 months), so this is not necessarily a population difference. The outcome-specific split proposed below is drawn on the production/comprehension line because that is where the _sign_ differs; whether DS and TD comprehension warrant different scales is a further question this note does not settle.

## 4. `kappa_min`

`kappa_min` is the asymptotic dispersion floor at old ages, so it sets maximum overdispersion: `rho_max = 1 / (kappa_min + 1)`. The current prior `LogNormal(log 5, 0.6)` has a 5th percentile of 1.86 and implies `rho_max` around 0.17 at its centre. The empirical minima are 3.0 (TD spoken, rho 0.25) and 4.2 (DS spoken, rho 0.19), and the fitted DS joint models put `kappa_min_u` at 1.28–1.34 without subject random effects and 2.4 with them — the latter group at prior CDF 0.11, the former at 0.01–0.02.

Proposed: **`LogNormal(log 3, 0.8)`** — median 3, 5–95% [0.80, 11.2].

| target                          | current prior CDF | proposed |
| ------------------------------- | ----------------- | -------- |
| VG05 / VG07 (no subject RE) 1.3 | 0.01              | 0.15     |
| VG08–VG10 (subject RE) 2.4      | 0.11              | 0.39     |
| TD spoken empirical 3.0         | 0.20              | 0.50     |
| VG04 6.4                        | 0.66              | 0.83     |
| VG12 8.4                        | 0.81              | 0.90     |

This covers both ends rather than only the top. It places 1.3% of prior mass below `kappa_min = 0.5` (`rho > 0.67`, developmentally implausible — near-total between-child bimodality), which is acceptable but worth watching in the prior-predictive audit.

## 5. `eta_q`

This is not a mis-scaling from evidence. PRIORS.md is explicit that `eta_q` was tightened from 0.4 to 0.20 to cure the weakly-identified `q` slope/intercept ridge in the subject-RE-on-`q` models (VG10 `test` min ESS 120 → 450, divergences 6 → 2), and describes the result as "a smoothness assumption on `q`, not a data-tuned value".

The `dev` fits say otherwise. Five DS models land at 0.487–0.513 with contraction between −0.03 and 0.11 — that is, no contraction at all — and prior CDF 0.98–0.99. Because the prior is truncating, **0.50 is a lower bound on what the data want, not an estimate of it.**

Proposed: revert to the family-standard **`HalfNormal(0.4)`**, the value every other GP amplitude in the family uses. It reintroduces no data-tuned quantity and puts 0.50 at prior CDF 0.79 — informative but not binding.

Two caveats belong with this proposal.

First, it will re-expose the ridge. But VG10 carries _two_ independent fixes for that ridge — the `eta_q` tightening and the per-draw GP anchoring — and only one of them distorts a prior. The test is whether the anchoring alone suffices: refit VG10 at `test` with `eta_q = 0.4` and watch min ESS and divergences. If it holds, the tightening is redundant.

Second, the conflict is confined to the models with both study and subject random effects (VG07–VG10, VG16). VG05, which has neither, sits at 0.352, and young-TD VG13 at 0.129 — neither presses the prior. In the same models `tau_subj_q` is running at 1.36 against a prior median of 0.34. That joint pattern is what aliasing between the `q` GP and the subject effects on `q` would look like. If `eta_q` inflates past roughly 0.8 under the wider prior while `tau_subj_q` stays high, the problem is structural identifiability and no prior choice will fix it — the tightening was papering over it.

## 6. The joint prior set, and why `a_kappa` moves too

Widening `b_kappa_mag` alone breaks the young-age end. The exponential is anchored at `z = 0` and extrapolated across roughly ±2 standard deviations of age, so the upper tails of `a_kappa` and `b_kappa_mag` compound. With `b ~ HalfNormal(1.0)` and `a_kappa ~ Normal(log 8, 1.0)`, 15.1% of prior mass puts `kappa > 200` at `z = −2` — effectively binomial, and empirically false, since the observed young-age `kappa` is 22–36.

Tightening `a_kappa`'s scale from 1.0 to 0.75 compensates, and costs nothing: every fitted `a_kappa` posterior in the family (1.41–2.06) sits comfortably inside `Normal(log 8, 0.75)`, whose 5–95% range is [0.85, 3.31]. This is a compensating change made for prior-predictive reasons, not because the data asked for it, and it should be labelled as such.

Prior-predictive `kappa(z)`, 300,000 draws:

| prior set                                                          | z = −2              | z = 0              | z = +2            | P(kappa > 200 \| z = −2) |
| ------------------------------------------------------------------ | ------------------- | ------------------ | ----------------- | ------------------------ |
| current: `kmin LN(log5, .6)`, `a N(log8, 1.0)`, `b HN(0.30)`       | 19.3 [6.5, 82]      | 14.4 [5.4, 48]     | 11.3 [4.4, 35]    | 0.6%                     |
| `b HN(1.00)`, `a N(log8, 1.0)`                                     | 39.0 [6.9, 638]     | 12.4 [3.9, 46]     | 6.0 [1.7, 23]     | 15.1%                    |
| `b HN(0.75)`, `a N(log8, 1.0)`                                     | 29.1 [6.3, 279]     | 12.4 [3.9, 46]     | 6.8 [2.0, 26]     | 7.7%                     |
| **proposed: `kmin LN(log3, .8)`, `a N(log8, 0.75)`, `b HN(0.75)`** | **28.5 [7.6, 218]** | **12.2 [4.7, 33]** | **6.7 [2.1, 21]** | **5.6%**                 |
| `b HN(1.00)`, `a N(log8, 0.75)`                                    | 38.0 [8.3, 529]     | 12.2 [4.7, 33]     | 6.0 [1.7, 19]     | 13.4%                    |
| empirical (TD spoken)                                              | 22–36               | ~12                | 3–4               | —                        |

The proposed set tracks the empirical series at all three points while the current set is flat by comparison — it cannot fall below `kappa` 11 at `z = +2` where the data show 3–4.

The residual weakness is the upper tail at `z = −2`: 5.6% of prior mass still implies near-binomial dispersion at the youngest ages. `HalfNormal(0.75)` also leaves VG01's `dev` posterior of 1.72 at prior CDF 0.98, so that model may still press the prior after refitting. Both are consequences of the parameterisation rather than of the values chosen — see §8.

That prediction was borne out, and understated: refitted at `test`, VG01 moved to 2.337 and prior CDF 1.00 (§10). VG03 and VG04 are comfortable at 0.93 and 0.08; VG01 alone is not.

### Implemented

> [!CAUTION]
> **Superseded by §18.** These three models no longer carry a `KappaPriorParams` block at all: the same curve is now parameterised by two age anchors, and `a_kappa` / `b_kappa_mag` have no priors of their own. The `kappa_min` values below survive unchanged; the other two do not. The block is kept here because §§10 and 13 report fits made under it.

For the three univariate spoken models — VG01, VG03, VG11 — in [`definitions.py`](../src/vocab_growth/models/definitions.py):

```python
kappa=KappaPriorParams(
    kappa_min_mu=math.log(3.0), kappa_min_sigma=0.8,
    a_kappa_mu=math.log(8.0),   a_kappa_sigma=0.75,
    b_kappa_mag_sigma=0.75,
)
```

## 7. Why the production prior stops at the univariate spoken models

The obvious extension is `kappa_s` in the joint models. It does not follow, because `kappa_s` does not govern the same quantity. In the univariate spoken models the likelihood is `BetaBinomial(n = 810, p = p_spoken, kappa)` — exactly what §2 calibrates. In the joint engine the nested spoken likelihood is `BetaBinomial(n = observed understood count, p = q, kappa_s)` ([`common_bivariate.py:688`](../src/vocab_growth/models/common_bivariate.py:688)): the denominator is per-child and varies, and the mean is the production _ratio_, not the spoken proportion. Dispersion on that scale is a different quantity from dispersion of spoken-out-of-810, and the marginal per-age fit does not transfer to it.

Extending to `kappa_s` therefore needs its own calibration, on the conditional scale: per-age Beta-Binomial fits of spoken counts against observed understood counts, restricted to the nested rows. That is tractable with the existing data and is the natural next step, but it is not done here and the joint models are left unchanged.

## 8. The deeper issue: `a_kappa` and `b_kappa_mag` are not separately interpretable

Both residual weaknesses in §6 have the same origin. `a_kappa` and `b_kappa_mag` parameterise an intercept and a slope, and it is their _joint_ tails that determine dispersion at the ends of the age range — the only place where dispersion can actually be checked against data. Neither prior is separately interpretable on the observable `rho` scale, so calibrating them requires the kind of simulation table above rather than direct reasoning.

This is precisely the problem the project already solved for the mean trajectory by parameterising it through the expected proportion at two reference ages. The same move is available here: put priors on `kappa` (or `rho`) at a young and an old reference age, joined on the log scale. Both priors would then be directly checkable against the per-age table in §2, the compensating `a_kappa` change would be unnecessary, and — because the two anchors are free — the sign constraint of §3 would disappear, making TD comprehension representable without a separate signed-coefficient change.

That is a larger change, touching [`build_utils.build_kappa_of_z`](../src/vocab_growth/models/build_utils.py) and every model's `KappaPriorParams`. It is recorded here as the direction of travel, not as a proposal for this cycle.

> [!NOTE]
> Implemented in §18, for VG01, VG03 and VG11. Two details of the sketch above did not survive contact with the data: the floor stays (dropping it costs 10 to 168 log-likelihood units), so the anchors are on the age term _above_ the floor rather than on total `kappa`; and the helper is in `gp_utils`, not `build_utils`, since it emits PyMC ops.

## 9. Consequences and next steps

> [!NOTE]
> This list was written before §§10–13. Items 1 and 6 are superseded; the rest stand. A consolidated status is in §14.

1. ~~**Validate the implemented change.** Refit VG01, VG03, VG11 at `test`.~~ Done — see §10, and §13 for the reruns on the corrected frame.
2. **Rerun the prior-predictive audit.** [`scripts/prior_predictive_audit.py`](../scripts/prior_predictive_audit.py) must be rerun for the affected models before any reporting fit; PRIORS.md's audit table and its `kappa` row are stale for VG01, VG03 and VG11.
3. **Calibrate `kappa_s` on the conditional scale** (§7) before extending the production prior to the joint models.
4. **Test `eta_q = 0.4` on VG10 at `test`** (§5), watching min ESS, divergences and `tau_subj_q` together.
5. **Update PRIORS.md.** The "Dispersion (`kappa`)" section's conclusion — that the prior is "slightly too tight at the high-dispersion end" — understates the finding: the age-_slope_ prior is out by roughly a factor of four for production, and is the wrong shape for TD comprehension. The `eta_q` review note, which describes the tightening as carrying no data-informed content, needs the §5 correction.
6. ~~**These are `dev` fits.**~~ Superseded: VG01, VG03 and VG04 now have `test`-config fits (§§10, 13). Nothing here is reporting-quality, and no fit in this note has passed a convergence gate.

## 10. Validation: `test`-config refits, 2026-08-02

VG01, VG03 and VG11 refitted at `test` (4 chains x 2,000 draws) with the §6 priors. Typically-developing pools remain at 10%, so VG03 and VG11 differ from the `dev` fits in both prior and sampling configuration; VG01 is on the full Down syndrome pool.

### `kappa_min` — resolved

|      | `dev` (old prior) | prior CDF | `test` (new prior) | prior CDF | contraction |
| ---- | ----------------- | --------- | ------------------ | --------- | ----------- |
| VG01 | 2.414             | 0.11      | 2.775 [2.38, 3.19] | 0.46      | 0.94        |
| VG03 | 2.626             | 0.14      | 3.075 [2.46, 3.60] | 0.51      | 0.91        |

Centred in the prior with contraction above 0.9 in both models. This part of §4 can be regarded as settled for the univariate spoken models.

### `b_kappa_mag` — resolved for VG03, not for VG01

|      | `dev` (old prior) | prior CDF | `test` (new prior) | prior CDF | contraction  |
| ---- | ----------------- | --------- | ------------------ | --------- | ------------ |
| VG01 | 1.719             | 1.00      | 2.337 [1.99, 2.70] | 1.00      | 0.07 -> 0.52 |
| VG03 | 1.175             | 1.00      | 1.425 [1.15, 1.69] | 0.94      | 0.18 -> 0.64 |

VG03 is now properly informed. VG01 is not: loosening the prior moved the posterior _further out_, 1.72 -> 2.34, and at 3.1 prior standard deviations it remains censored, so 2.34 is still a lower bound.

Widening again would be the wrong response. At VG01's fitted values `kappa(z) = 2.775 + exp(1.111 - 2.337 z)` gives kappa about 34 at z = −1, 5.8 at z = 0, 3.1 at z = +1 and 2.80 at z = +2 — the exponential term is extinguished by z = +1 and `kappa_min` carries everything beyond it. `b_kappa_mag` is therefore measuring how fast the exponential is switched off so the floor can take over, not a dispersion gradient. This is §8's non-identifiability observed directly, and it is an argument for the two-anchor reparameterisation rather than for a third widening.

Note also that `a_kappa` drifted down to prior CDF 0.10 (VG01) and 0.16 (VG03). Not a conflict, but the compensating tightening to sigma = 0.75 may now be pinching from the other side, and should be revisited if `a_kappa` is seen to press further.

### Sampling

VG01 and VG03: max R-hat 1.003, min ESS 1,141 and 2,139, **zero divergences**, min BFMI 0.82 and 0.94. The steeper `kappa` gradient costs nothing geometrically. The sampling configuration changed alongside the prior, so this is not a clean attribution, but the absolute values are good.

### VG11 is bimodal, and the prior change is not the cause

VG11 came back at max R-hat 1.737, min ESS 6, min BFMI 0.18, with the four chains splitting into two stable pairs. A control fit — VG11 at `test`, everything identical except `kappa` reverted to the shared default, written to a scratch output root — reproduces the failure almost exactly:

|                       | max R-hat | min ESS | div | min BFMI | `tau_subject` chain medians | `a_kappa` chain medians |
| --------------------- | --------- | ------- | --- | -------- | --------------------------- | ----------------------- |
| treatment (new prior) | 1.737     | 6       | 1   | 0.18     | 0.07, 1.08, 0.07, 1.08      | 1.25, 4.15, 1.23, 4.10  |
| control (old prior)   | 1.740     | 6       | 3   | 0.20     | 0.07, 0.08, 1.08, 1.08      | 1.46, 1.46, 4.28, 4.27  |

The two modes are the same under both priors: between-child overdispersion is attributed either to the Beta-Binomial (`tau_subject` about 0.07, `a_kappa` about 1.3) or to the subject random effects (`tau_subject` about 1.08, `a_kappa` about 4.2), with nothing in the likelihood choosing between them. **This is aliasing between `kappa` and the subject random effects, revealed by running four chains rather than two.** The recalibration neither caused it nor made it materially worse. §11 identifies the actual cause, which is neither the prior nor the model specification.

It follows that VG11's `dev` diagnostics — max R-hat 1.025, min ESS 99 — were falsely reassuring: both chains happened to land in the same mode. Two chains cannot detect this failure mode, and the `dev` configuration uses two.

### The wider implication

Sorting the `dev` run's max R-hat by whether a model carries subject random effects:

- **with** subject REs: VG08 1.075, VG09 1.280, VG10 1.083, VG11 1.025, VG12 1.075, VG13 1.038, VG15 1.137, VG16 1.127
- **without**: VG01 1.009, VG02 1.028, VG03 1.044, VG04 1.011, VG05 1.058, VG07 1.022, VG14 1.034

Every model above 1.07 carries subject random effects, and VG09's `dev` signature (R-hat 1.280, min ESS 6) is the same shape as VG11's confirmed bimodality. That is suggestive, not established — these are two-chain fits, and R-hat is unreliable at that width.

**§11 substantially qualifies this.** VG11, VG12 and VG13 were all fitted on a 10% subsample, which is now identified as the cause of VG11's bimodality; their diagnostics say nothing about the models as specified. VG08, VG09, VG10, VG15 and VG16 are Down syndrome models fitted on the full pool, so their elevated R-hat needs a different explanation and remains open.

It also bears on §5. The `eta_q` ridge was diagnosed as a `q` slope/intercept problem and treated by tightening a prior. If the underlying issue is instead that `kappa`, the `q` GP and the subject effects on `q` are three mechanisms competing to explain the same overdispersion, then the tightening suppressed a symptom of a structural problem in one of the three.

### Follow-up

- ~~Refit VG11 at `test` with four or more chains~~ Done — see §11.
- Treat the reporting-quality configuration's six chains as the minimum for any model with subject random effects; two-chain `dev` diagnostics cannot be trusted for them.
- VG01's residual censoring is a further argument for the §8 reparameterisation.

## 11. The VG11 bimodality is an artefact of row-wise subsampling

VG11 refitted at `test` with six chains. nutpie does not accept per-chain `initvals` — PyMC 6.2's `_sample_external_nuts` warns and drops a sequence, honouring only a single shared `initial_points` dict — so chain dispersion comes from nutpie's own per-chain jitter, which demonstrably reaches both basins.

Result: max R-hat 1.718, min ESS 9, 5 divergences, and a **clean 3/3 split with no within-chain migration whatever** — each chain's median `tau_subject` is constant to three decimals across its 2,000 draws. The modes are entirely separated.

|        | chains  | `tau_subject` | `a_kappa` | `kappa_min` | `b_kappa_mag` | BFMI      |
| ------ | ------- | ------------- | --------- | ----------- | ------------- | --------- |
| mode A | 2, 4, 5 | 0.070–0.077   | 1.23–1.25 | 3.60–3.63   | 1.51–1.53     | 0.81–0.85 |
| mode B | 0, 1, 3 | 1.076–1.078   | 4.04–4.07 | 9.68–10.6   | 1.18–1.22     | 0.18–0.21 |

### The cause

The 10% subsample draws **rows, not children**, so it destroys the within-subject replication that identifies a subject random effect:

|               | rows   | subjects | with >1 observation | obs/subject |
| ------------- | ------ | -------- | ------------------- | ----------- |
| full pool     | 16,235 | 12,266   | 1,947 (15.9%)       | 1.32        |
| 10% subsample | 1,626  | 1,565    | 60 (3.8%)           | 1.04        |

At 1.04 observations per child a subject random effect and the observation-level Beta-Binomial dispersion are **the same thing** — two per-observation noise terms with nothing to separate them. Mode A puts that noise in `kappa`; mode B puts it in `tau_subject`. That is the whole bimodality.

LOO confirms the mechanism and must not be read as favouring mode B:

|        | elpd_loo        | p_loo       | Pareto k > 0.7    |
| ------ | --------------- | ----------- | ----------------- |
| mode A | −8,622.8 ± 70.0 | 36.3        | 0 / 1,626         |
| mode B | −8,020.2 ± 62.1 | **1,113.1** | **1,097 / 1,626** |

Mode B's apparently better elpd is an artefact: 1,113 effective parameters for 1,626 observations, and PSIS has failed on two thirds of the points. That is the signature of a model absorbing the data into per-observation parameters, not of a better fit.

### Which mode is right — see §12, this section's first answer was wrong

The original comparison here set the two modes against the published Wordbank normative median (11 spoken words at 12 months) and concluded that mode A was correct, because mode A gives 10.6 words at 12 months and mode B gives 4.1. §12 shows that comparison used the wrong comparator and reached the wrong answer: **VG11's own pool has an empirical median of 4.0 spoken words at 12 months**, and it is mode B that reproduces it. The 11-word figure is the pool's _mean_ (10.2), not its median.

What survives from this section is the diagnosis of the _cause_ — row-wise subsampling — not the ranking of the modes.

### Consequences

1. **This says nothing against VG11 as specified.** The full pool has 15.9% of children with repeat administrations; the artefact is created by the subsample, not by the model.
2. **Row-wise subsampling is unsafe for any model with subject random effects** — VG11, VG12, VG13 here, and any future use of `sample_fraction`. Subsampling should draw **subjects and keep all their observations**, which `load_data`'s `sample_fraction` does not currently do ([`data_utils.py:1027`](../src/vocab_growth/data_utils.py:1027)). That is a latent trap in the definitions: VG03 and VG04 ship with `sample_fraction=0.25` and are safe only because they have no subject effects.
3. **The §10 speculation about a family-wide problem is narrowed.** VG11, VG12 and VG13 were all subsampled, so their diagnostics are uninformative about the models. VG08, VG09, VG10, VG15 and VG16 are Down syndrome models on the full pool, where the artefact cannot apply — their elevated R-hat is a separate, still-open question.
4. **Decisive confirmation** would be VG11 at `test`, six chains, on the full typically-developing pool. The 10% six-chain fit took 4m 03s, so the full pool is roughly a 40-minute run. A cheaper equivalent is to subsample 10% of _subjects_ rather than rows, preserving replication at the same data volume.
5. The promoted VG11 `test` fit (R-hat 1.718) is diagnostic evidence, not a usable fit, and its recorded sampling parameters carry 6 chains rather than the registered `test` 4, so it will not revalidate under `--render-only`.

## 12. Subject-wise subsampling resolves it — and reverses §11's mode ranking

VG11 refitted at `test`, six chains, drawing **10% of subjects and keeping all their observations**. Replication is restored at the same data volume: 1,662 rows, 1,249 subjects, 16.7% with more than one observation, 1.33 observations per subject — against the full pool's 15.9% and 1.32, and the row-wise draw's 3.8% and 1.04.

|                      | max R-hat | min ESS | div | BFMI              | `tau_subject` chain medians        |
| -------------------- | --------- | ------- | --- | ----------------- | ---------------------------------- |
| row-wise 10%         | 1.718     | 9       | 5   | 0.18–0.85 (split) | 0.07, 0.07, 0.08, 1.08, 1.08, 1.08 |
| **subject-wise 10%** | **1.010** | **406** | 28  | 0.39–0.44         | 1.07 x 6                           |

**The bimodality is gone.** All six chains agree to two decimals on every parameter. §11's diagnosis is confirmed: row-wise subsampling was the cause, and it is a defect in how `sample_fraction` draws, not in VG11.

### The correction

The converged solution is `tau_subject` = 1.07, `a_kappa` = 4.31 — that is, **mode B**, the one §11 called spurious. §11's ranking was wrong, and the error was in the comparator, not the arithmetic.

VG11's pool is seven Wordbank datasets including three bilingualism labs (ByersHeinlein, Floccia, Kalashnikova). Its spoken-count distribution at 12 months is strongly right-skewed — 25th percentile 1 word, median 4, 75th percentile 9, mean 10.2. The published Wordbank norm of 11 words is a US-English **monolingual** figure, and it happens to sit almost exactly on this pool's _mean_. `p_query` is the median-child trajectory (subject effects are centred Normal on the logit scale and the logistic is monotone), so it must be compared with the pool's median of 4.0, not with 11. Against the right comparator the ranking inverts:

| age   | empirical median (own pool) | subject-wise fit | row-wise mode A |
| ----- | --------------------------- | ---------------- | --------------- |
| 9 mo  | 0.0                         | 1.5              | 3.5             |
| 12 mo | **4.0**                     | **4.1**          | 10.6            |
| 15 mo | 15.0                        | 14.5             | 30.7            |
| 18 mo | 50.0                        | 50.5             | 76.9            |
| 24 mo | 264.0                       | 249.6            | 253.2           |
| 30 mo | 487.0                       | 422.0            | 430.3           |

Integrating over the subject-effect distribution recovers the mean as well — model population mean 7.0 at 12 months and 73.9 at 18 against empirical 10.2 and 79.2 — so the fit is reproducing both moments of a skewed distribution, which is what a large `tau_subject` is _for_.

Two earlier claims in this conversation's record therefore need withdrawing. The first review's validation that "the 10% subsample holds up" rested on the `dev` VG11 fit giving 11.4 words at 12 months against the 11-word norm; that fit was in the artefact mode, and the agreement was with the wrong statistic. §11's "three independent lines agree that mode A is correct" was wrong on all three: the norm comparison used the pool's mean as if it were a median, and mode B's poor BFMI (0.18–0.21) and inflated `p_loo` were symptoms of the _unidentified_ subject effects under row-wise draw, not properties of the high-heterogeneity solution — with replication restored the same solution gives BFMI 0.39–0.44.

### What is still unsatisfactory

- 28 divergences and BFMI around 0.40. Converged, but the geometry is not comfortable.
- `p_loo` is 914.6 on 1,619 observations with 796 Pareto k above 0.7. Better than the row-wise 1,113 / 1,097, but still the signature of a model with close to one random effect per observation. At 1.33 observations per child, subject effects remain weakly identified even under subject-wise draw — this is a property of the data, not of the subsample, and it applies to the full pool too (1.32 obs/subject).
- The fit undershoots the empirical median at 27–30 months (350 against 378, 422 against 487).

### Actions

1. ~~**Fix `sample_fraction` to draw subjects, not rows.**~~ Done — see §13.
2. **Any VG11/VG12/VG13 result obtained under row-wise subsampling should be discarded**, including the `dev` run of 2026-08-02 that opens this note. The Down syndrome models are unaffected — they use the full pool.
3. Re-examine whether subject random effects are worth their cost at 1.32 observations per child, given `p_loo` near the observation count in both draws.
4. The `kappa` recalibration is untouched by all this: VG01 and VG03 have no subject effects, and their §10 results stand.

## 13. Implemented: `sample_fraction` now draws subjects

`load_data` subsamples whole children and keeps all their administrations, via a new `_subsample_subjects` helper ([`data_utils.py`](../src/vocab_growth/data_utils.py)). Subjects are keyed by `study`/`subject_id` together, matching the `subject_key` convention in the random-effect engines, so an identifier repeated across datasets is not merged. The behaviour is unconditional rather than opt-in: row-wise draw is never the right choice for this pool, so leaving it reachable would only preserve the trap.

Measured on the real pool, spoken outcome: 1.325 administrations per child at full size, **1.331** under the new 10% draw, 1.039 under the old one.

Four tests in [`tests/test_data_utils.py`](../tests/test_data_utils.py) cover it — that every retained child keeps all its rows and the draw is over children rather than rows, that it is study-scoped, that it is reproducible under a fixed seed, and an integration test against the real database asserting that a 10% draw stays within 10% of the full pool's observations-per-child. The last of these fails on the old behaviour (1.039 against 1.325), which is the regression that was missing.

### Consequence for VG03 and VG04

Both ship at `sample_fraction=0.25`, so their analysis frames change: VG03 goes from 4,138 to 4,152 rows and VG04 from 1,533 to 1,550. Neither carries subject random effects, so nothing about their specification is affected and no estimate should move materially — but the frames differ, so their `analysis_frame_hash` changes and their existing fits were stale. Both have now been refitted (below), and in both cases the estimates are indeed unmoved, which is the expected result and a check on the change.

### VG03 refitted on the corrected frame

VG03 refitted at `test` (4 chains x 2,000 draws) at its own `sample_fraction=0.25` — 4,152 rows, no overrides of any kind, so unlike the VG11 diagnostics this fit revalidates normally. 22m 40s. (§17's ordering fix later moved this frame to 4,075 rows; the fit below predates it.)

Max R-hat 1.003, min ESS 2,233, **zero divergences**, min BFMI 0.86.

|               | §10 fit (10% override, old draw) | this fit (0.25, subject-wise) | prior CDF | contraction |
| ------------- | -------------------------------- | ----------------------------- | --------- | ----------- |
| `b_kappa_mag` | 1.425                            | 1.369 [1.24, 1.50]            | 0.93      | 0.82        |
| `kappa_min`   | 3.075                            | 2.540 [2.18, 2.86]            | 0.42      | 0.95        |
| `a_kappa`     | 1.324                            | 1.537 [1.40, 1.67]            | 0.23      | 0.89        |

At 2.5 times the data the estimates agree with the §10 fit — the `b_kappa_mag` intervals overlap heavily — and contraction rises from 0.64 to 0.82. **This is the clean confirmation the recalibration needed**: `b_kappa_mag` is now genuinely data-informed and sits inside the prior at CDF 0.93, against 1.00 with contraction 0.18 before the change. `a_kappa` has also come back up off the low tail (CDF 0.23 against 0.16 at 10%), so the compensating tightening to sigma = 0.75 is not pinching after all.

~~The fitted 1.37 sits above the marginal per-age estimate of 0.77 from §2, as expected: the model's GP mean and study effects absorb mean-trend structure that the marginal fit leaves in the residual.~~ **This explanation is wrong — see §17.** The 0.77 came from a log-linear estimator that is biased low; the correct three-parameter value is 1.71, so the fitted 1.369 sits _below_ the empirical estimate, not above, and the gap it explained does not exist. The likelier reading is that `HalfNormal(0.75)` is still pulling the estimate down.

VG01 remains the outstanding case — still censored at prior CDF 1.00 (§10) — which continues to point at the §8 reparameterisation rather than a further widening.

### VG04 refitted, and the sign constraint confirmed

VG04 refitted at `test` on the corrected frame, 1,550 rows, 6m 07s. (§17's ordering fix later moved this frame to 1,555 rows; the fit below predates it.) It is a **comprehension** model, so it keeps the shared default `kappa` prior — the production recalibration deliberately excluded it — and the fit therefore tests the frame change, not the prior change.

Max R-hat 1.002, min ESS 1,992, zero divergences, min BFMI 0.93.

|               | `dev` (10%, row-wise) | this fit (0.25, subject-wise) | prior CDF | contraction |
| ------------- | --------------------- | ----------------------------- | --------- | ----------- |
| `b_kappa_mag` | 0.073                 | **0.031 [0.002, 0.178]**      | 0.08      | 0.64        |
| `kappa_min`   | 6.420                 | 6.412 [2.41, 9.79]            | 0.66      | 0.41        |
| `a_kappa`     | 1.512                 | 1.456 [−0.04, 2.12]           | 0.27      | 0.32        |
| `p_slope_low` | 0.081                 | 0.087 [0.038, 0.176]          | 0.43      | 0.58        |
| `p_slope_hi`  | 0.413                 | 0.416 [0.201, 0.652]          | 0.40      | 0.48        |
| `eta`         | 0.687                 | 0.661 [0.335, 1.152]          | 0.81      | 0.18        |

Every trajectory parameter is unmoved, as it should be for a model with no subject random effects — the frame correction changes which children are drawn, not what the pool looks like.

The result of interest is `b_kappa_mag`. With 2.5 times the data and four chains it has moved **closer** to the boundary, 0.073 to 0.031, with a 5th percentile of 0.002. That is what a posterior does when the true value lies at or beyond a constraint: it collapses onto it and tightens. §3's reading is therefore confirmed on better evidence — typically-developing comprehension dispersion does not fall with age, the model can only represent falling dispersion, and VG04 is answering "as close to flat as you will let me". Leaving the comprehension prior at `HalfNormal(0.3)` is the right call in the sense that widening it would be actively wrong, but the boundary pile-up is a specification limit, not a resolved question: the fix is a signed `b_kappa ~ Normal(0, 0.5)`, or the §8 reparameterisation, which removes the constraint as a side effect.

One refinement to §4. VG04 puts `kappa_min` at 6.41, comfortably inside the _current_ prior (CDF 0.66) — so the recentring proposed in §4 is not needed for typically-developing comprehension, and would move this model to CDF 0.83. It is still needed for Down syndrome comprehension, where VG02, VG05 and VG07 sit at 1.28–1.43 against CDF 0.01–0.02. A single family-wide `kappa_min` prior therefore has to span roughly 1.3 to 8.4, which `LogNormal(log 3, 0.8)` does (CDF 0.15 to 0.90). §4's proposal stands; what VG04 adds is that its _upper_ end is load-bearing, so the recentring must not be pushed lower than log 3.

## 14. Status

Everything done and found on 2026-08-02, in one place.

> [!CAUTION]
> Written before §18. The first row of "Changed in the repository" and the VG01, VG03 and VG11 rows of "Fits" are superseded: those three models no longer use the `_PRODUCTION_KAPPA` block and have been refitted under the two-anchor form. Open items 1 and 6 of the list at the end are also resolved by §18. Everything else stands.

### Changed in the repository

| Change                                                                                                                       | Where                                 | Status                            |
| ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------------- |
| Production `kappa` priors (`kappa_min` LN(log 3, 0.8), `a_kappa` N(log 8, 0.75), `b_kappa_mag` HN(0.75)) on VG01, VG03, VG11 | `definitions.py`, `_PRODUCTION_KAPPA` | Implemented, validated (§§10, 13) |
| `sample_fraction` draws subjects, not rows                                                                                   | `data_utils._subsample_subjects`      | Implemented, 4 tests (§13)        |
| VG16 GP anchoring at 54 mo (`anchor_g_u_at_ref`, `anchor_g_q_at_ref`)                                                        | `definitions.py`, VG16                | Implemented, validated (§16)      |
| `bimodality`, `parameterising`, `recentring`, `unrepresentable`                                                              | `config/spellcheck/allow-en.txt`      | Added                             |

### Fits

| Model  | Config           | Frame                  | max R-hat | min ESS | div | Verdict                                              |
| ------ | ---------------- | ---------------------- | --------- | ------- | --- | ---------------------------------------------------- |
| VG01   | `test`, 4 chains | full DS pool           | 1.003     | 1,141   | 0   | Clean, but `b_kappa_mag` still censored at CDF 1.00  |
| VG03   | `test`, 4 chains | 0.25 subject-wise      | 1.003     | 2,233   | 0   | Clean; the recalibration's confirmation              |
| VG04   | `test`, 4 chains | 0.25 subject-wise      | 1.002     | 1,992   | 0   | Clean; confirms the sign-constraint limit            |
| VG11   | `test`, 6 chains | 10% row-wise           | 1.718     | 9       | 5   | Bimodal — diagnostic evidence only, not a usable fit |
| VG11   | `test`, 6 chains | 10% subject-wise       | 1.010     | 406     | 28  | Unimodal; scratch output, not promoted               |
| VG16   | `test`, 4 chains | full DS pool, anchored | 1.013     | 331     | 1   | Clean; `beta_lag` 0.308 [0.182, 0.438]               |
| all 15 | `dev`, 2 chains  | 10% row-wise (TD)      | —         | —       | —   | Superseded; TD entries discard                       |

The promoted VG11 output is the row-wise bimodal fit and should not be quoted. Its sampling parameters record 6 chains against the registered `test` 4, so it will not revalidate under `--render-only` either.

### Findings, and how firm each is

- **`b_kappa_mag` was mis-scaled by roughly 4x for production outcomes.** Firm — empirical calibration and four model fits agree, and the corrected prior moves VG03 from CDF 1.00 / contraction 0.18 to CDF 0.93 / contraction 0.82.
- **The sign constraint makes typically-developing comprehension unrepresentable.** Firm — VG04 tightens _onto_ the boundary as data increase.
- **`sample_fraction` destroyed within-child replication.** Firm, fixed, and covered by a regression test that fails on the old behaviour.
- **`kappa_min` needs recentring, and its upper end is load-bearing.** Firm for the univariate spoken models where it is implemented; the family-wide extension (§4) is still a proposal.
- **`eta_q`'s tightening is doing unjustified statistical work.** Firm as a description of the posteriors (five models, no contraction); the proposed revert to `HalfNormal(0.4)` is untested.
- **VG01's `b_kappa_mag` is censored even at HN(0.75).** Firm. §15 attributed it to units; §17 shows the prior is too tight in absolute terms for VG03 too, and that the parameterisation blocks the fix. Partial explanation (§15): `b_kappa_mag` is a per-standardised-age slope, so a shared prior is about 3.5x tighter on VG01 than on VG03 in per-month terms, purely because the Down syndrome pool's age sd is 20.8 months against 5.9.
- **Elevated R-hat across the subject-RE models.** Largely dissolved (§15). It is the understood-trajectory GP/linear ridge, which VG10's anchoring already fixes; VG08 and VG09 are documented lineage steps. The one real finding was **VG16**, the only model with subject effects on both `u` and `q` and no anchoring. Now anchored (§16): 47 divergences to 0 at `dev`, ridge correlations removed, `beta_lag` unchanged at 0.308. Note §16 also corrects §15 on where the benefit shows — at `test` the unanchored model is already fine, so more tuning masks the geometry rather than the geometry being harmless.

### Open

1. **Restate `b_kappa_mag`'s prior in per-month units, or reparameterise** (§15). Do not widen it again.
2. ~~**Give VG16 the VG10 stabilisation** and refit.~~ Done — §16. Kept: the ridge is removed and the `dev` divergences clear, at the cost of roughly halving `beta_lag`'s ESS at `test`.
3. Items 2–5 of §9: prior-predictive rerun, `kappa_s` conditional calibration, the `eta_q` test on VG10, and the PRIORS.md update.
4. VG12 and VG13 have no valid fit — their `dev` runs used the row-wise draw.
5. Whether subject random effects earn their place at 1.32 administrations per child in the typically-developing pool. Even under the corrected draw, VG11's `p_loo` is 914.6 on 1,619 observations with 796 Pareto k above 0.7. That is a property of the data, not the subsample, and it applies to the full pool. It does **not** apply to the Down syndrome models, which have 1.87 administrations per child (§15).
6. Re-read §15's R-hat figures at `test` or better; they come from 2-chain `dev` fits.

## 15. The two open items diagnosed

### VG01's censored `b_kappa_mag`: the prior is stated in the wrong units

`b_kappa_mag` is a slope per _standardised_ age unit, and `z = (age - mean) / sd` is standardised within each model's own pool. A shared prior therefore encodes a different developmental claim in every model, scaled by that pool's age spread:

| model | pool          | age sd (months) | `b_kappa_mag` posterior | per month | prior median, per month |
| ----- | ------------- | --------------- | ----------------------- | --------- | ----------------------- |
| VG01  | DS spoken     | 20.8            | 2.337                   | 0.112     | 0.0243                  |
| VG03  | TD spoken     | 5.9             | 1.369                   | 0.232     | 0.0858                  |
| VG04  | TD understood | 3.5             | 0.031                   | 0.009     | 0.0575                  |

The last column is each model's _own_ prior median converted to a per-month rate — `HalfNormal(0.75)` for the two production models, `HalfNormal(0.30)` for comprehension VG04, which the recalibration deliberately left alone (§3). The like-for-like comparison is VG01 against VG03.

Read in per-month terms the models are not even ordered the way the raw parameter suggests — typically-developing dispersion changes about twice as fast per month as Down syndrome dispersion, yet VG01's `b_kappa_mag` is the larger number. The whole difference is the standardisation. `HalfNormal(0.75)` allows VG01 a median rate of 0.024 per month against VG03's 0.086: **the same prior is about 3.5 times tighter on VG01 in the units that carry developmental meaning.** That, not anything about Down syndrome dispersion, is why VG01 is the censored one.

The mechanism is exact. VG01's dispersion transition runs from about 12 to about 48 months — 1.74 standard deviations of its own age distribution — across a 57-fold drop in the exponential term. That requires `b = ln(57) / 1.74 = 2.33`, which is the posterior median to two decimal places. The wide Down syndrome age range (8–115 months, sd 20.8) compresses a transition that occupies only the first third of the range into a short interval of `z`, and a short interval demands a steep slope.

Two further observations support treating this as a parameterisation problem rather than a data one:

- `corr(a_kappa, b_kappa_mag) = −0.789` in the VG01 posterior — the intercept/slope ridge of §8, measured.
- The exponential term is extinct over most of the range it is defined on: 87% of `kappa` at 24 months, 32% at 48, 11% at 60, 0.8% at 84, 0.0% at 115. For the upper half of the Down syndrome age range the model's dispersion is simply the constant `kappa_min`.

The realised `kappa(age)` curve is nonetheless reasonable where the data can check it — model 7.7 against empirical 6.5 at 36 months, 5.3 against 5.9 at 42, 4.1 against 4.7 at 48, 3.1 against 4.2 at 60 — so the fit is not distorted. It is the _parameter_ that is extreme, not the _function_.

> [!CAUTION]
> §17 revises this section. The claim that `b_kappa_mag` measures "how fast the floor takes over" rather than a dispersion gradient does not survive: a three-parameter fit in the model's own functional form independently gives 2.17 for Down syndrome spoken, against the fitted 2.337. VG01 is recovering the data. The units argument below stands, but explains only why the prior is _relatively_ tighter on VG01 — §17 shows it is too tight in absolute terms for VG03 as well.

**Recommendation: do not widen `b_kappa_mag` again.** Either state the prior per month and multiply by the pool's age sd when building the model, or adopt the §8 two-anchor reparameterisation, which is scale-free by construction because its anchors sit at fixed ages. As an illustration of the first option, taking VG03's `HalfNormal(0.75)` as the intended rate gives VG01 `sigma = 0.0857 x 20.8 / 0.6745 = 2.64`, under which its posterior sits at prior CDF **0.62** instead of 1.00 — with no change to the data or the fit, only to the units the prior is stated in.

### The Down syndrome models' elevated R-hat: mostly expected, one real finding

Three candidate explanations, tested against the `dev` traces.

**Not the VG11 aliasing.** The Down syndrome pool has strong replication — 1.87 administrations per child, 53.2% of children with more than one, maximum 8 — against 1.32 and 15.9% for typically-developing. Subject random effects are well identified here, and no model in this group is subsampled.

**Not the subject random effects.** They dominate the _count_ of parameters failing the 1.01 threshold (VG09 90 of 135, VG16 217 of 273) simply because there are several hundred of them, but their worst R-hat is low:

| model | subject REs | n params | max R-hat (all) | max R-hat (scalars) | max R-hat (subject REs) |
| ----- | ----------- | -------- | --------------- | ------------------- | ----------------------- |
| VG05  | no          | 22,864   | 1.058           | 1.042               | —                       |
| VG07  | no          | 22,914   | 1.022           | 1.022               | —                       |
| VG14  | no          | 26,666   | 1.034           | 1.034               | —                       |
| VG08  | yes         | 24,107   | 1.097           | 1.065               | 1.034                   |
| VG09  | yes         | 25,300   | 1.280           | **1.252**           | 1.043                   |
| VG10  | yes         | 25,252   | 1.095           | **1.017**           | 1.037                   |
| VG15  | yes         | 14,363   | 1.164           | 1.071               | 1.021                   |
| VG16  | yes         | 25,301   | 1.127           | **1.126**           | 1.057                   |

**It is the understood-trajectory ridge, and VG10 already fixes it.** The worst scalars are `intercept_u`, `p_slope_low_u`, `p_slope_hi_u`, `eta_u` and `ell_unit_u` — the linear trend competing with the GP. Posterior correlations, VG09 against VG10:

| pair                        | VG09 (no anchor) | VG10 (GP anchored at 54 mo) |
| --------------------------- | ---------------- | --------------------------- |
| `intercept_u`, `slope_u`    | −0.54            | −0.27                       |
| `intercept_u`, `eta_u`      | −0.37            | +0.03                       |
| `intercept_u`, `ell_unit_u` | −0.47            | −0.09                       |
| `eta_u`, `ell_unit_u`       | +0.43            | +0.46                       |

The per-draw GP anchoring removes the ridge; the residual `eta_u`/`ell_unit_u` correlation is the intrinsic amplitude/length-scale relationship and is untouched, as expected. VG10's max scalar R-hat is 1.017 against VG09's 1.252.

So VG08 and VG09 are lineage steps whose ridge is documented — it is the stated reason VG10 exists — and VG10 and VG15 carry the fix. Nothing new there.

**The finding is VG16.** It is the only model with subject random effects on both `u` and `q` that does **not** carry the GP anchoring:

|      | `anchor_g_u_at_ref` | `anchor_g_q_at_ref` | subject RE on u / q |
| ---- | ------------------- | ------------------- | ------------------- |
| VG09 | False               | False               | yes / yes           |
| VG10 | **True** (54 mo)    | **True** (54 mo)    | yes / yes           |
| VG15 | **True** (54 mo)    | **True** (54 mo)    | yes / yes           |
| VG16 | **False**           | **False**           | yes / yes           |

VG16 was built as "VG09 plus a cross-lag", so it inherited VG09's unanchored geometry rather than VG10's stabilised one. It is also the worst-behaved model in the family by some margin: **47 divergences (4.7% of draws)** against 0–0.4% everywhere else, and max scalar R-hat 1.126 on `eta_u`.

**Recommendation:** give VG16 the VG10 stabilisation — `anchor_g_u_at_ref=True`, `anchor_g_q_at_ref=True`, `gp_anchor_age_months=54.0` — and refit at `test`, watching the divergence count and `eta_u`. This is a one-line change to a registered model's structure, so it needs its own note entry and a before/after comparison of `beta_lag`, which is the estimand VG16 exists to report (currently 0.308, 89% ETI [0.170, 0.452]).

### Caveat

Every diagnostic in this section comes from 2-chain `dev` fits, where R-hat is unreliable and the maximum over 20,000-plus parameters is noisy. The parameter-family breakdown and the posterior correlations are robust to that — they describe posterior geometry, not convergence — but the R-hat figures themselves should be re-read at `test` or better before any of this is treated as settled.

## 16. Implemented: VG16 gets the VG10 GP anchoring

`anchor_g_u_at_ref=True`, `anchor_g_q_at_ref=True`, `gp_anchor_age_months=54.0` added to VG16 in [`definitions.py`](../src/vocab_growth/models/definitions.py), bringing it into line with VG10 and VG15. This is a **structural change to a registered model**, not a prior adjustment.

Three fits: the `dev` baseline, an unanchored control at `test`, and the anchored fit at `test`. The control exists so the effect can be attributed to the anchoring rather than to the configuration change.

| run                           | divergences | max R-hat (all) | max R-hat (scalars) | min ESS | min BFMI |
| ----------------------------- | ----------- | --------------- | ------------------- | ------- | -------- |
| `dev`, unanchored (2 chains)  | **47**      | 1.127           | 1.126               | 12      | 0.48     |
| `test`, unanchored (4 chains) | **0**       | 1.011           | 1.011               | 345     | 0.51     |
| `test`, anchored (4 chains)   | 1           | 1.013           | 1.013               | 331     | 0.49     |
| `dev`, anchored (2 chains)    | **0**       | 1.043           | —                   | 59      | 0.41     |

### A correction to §15's reasoning

§15 attributed VG16's 47 divergences to the missing anchoring and proposed a `test` refit as the test of that. **The `test` refit does not test it.** The unanchored control at `test` already has zero divergences: 2,000 tuning steps at `target_accept` 0.9 overcome the bad geometry on their own, so at that budget the two specifications are indistinguishable — 0 against 1 divergence, R-hat 1.011 against 1.013, min ESS 345 against 331.

The valid comparison is at the configuration where the problem appeared. At `dev`, anchoring takes VG16 from **47 divergences to 0**, max R-hat 1.127 to 1.043, and min ESS 12 to 59. So the diagnosis was right in substance and wrong about where it would show: the unanchored geometry is genuinely bad, and more tuning masks it rather than fixing it.

### The geometry does change, as designed

Posterior correlations at `test`, unanchored control against anchored:

| pair                        | unanchored | anchored |
| --------------------------- | ---------- | -------- |
| `intercept_u`, `slope_u`    | −0.51      | −0.19    |
| `intercept_u`, `eta_u`      | −0.31      | +0.04    |
| `intercept_u`, `ell_unit_u` | −0.50      | −0.04    |
| `eta_u`, `ell_unit_u`       | +0.45      | +0.51    |

The same signature as VG09 to VG10: the intercept/GP redundancy goes, the intrinsic amplitude/length-scale correlation stays. The ridge is real even where the sampler copes with it.

### `beta_lag` is unchanged

VG16 exists to report the cross-lag, so this is the number that had to survive the change:

| run                | `beta_lag` median | 89% ETI        | P(>0) | ESS   | R-hat |
| ------------------ | ----------------- | -------------- | ----- | ----- | ----- |
| `test`, unanchored | 0.310             | [0.178, 0.442] | 1.00  | 5,045 | 1.000 |
| `test`, anchored   | 0.308             | [0.182, 0.438] | 1.00  | 2,677 | 1.002 |

Identical to three decimal places, with near-identical intervals. The structural change does not move the estimand.

### Costs, and the recommendation

Two costs. `beta_lag`'s ESS roughly halves at `test` (5,045 to 2,677) and min ESS is marginally lower, so the anchoring is not free at high tuning budgets. And anchoring carries the interpretability cost documented for VG10: once the GP is anchored at 54 months, the anchor parameters stop matching the realised trajectory at the anchor ages (VG10's `p_slope_hi_q` is 0.935 against a realised `q(84)` of 0.734), so VG16's `p_slope_*` values must not be read as expected proportions.

**Keep the change.** It removes a real ridge, eliminates the divergences at exploratory budgets where most iteration happens, leaves the headline estimand untouched, and makes VG16 consistent with the two sibling models that already carry the stabilisation. The ESS cost is worth the geometry.

`beta_lag` = 0.308, 89% ETI [0.182, 0.438] is now the `test`-config value on the anchored model. It remains a `test` fit, not reporting quality, and the "population-relative headline is approximately null" framing in the model inventory refers to a different baseline variant — that discrepancy is not addressed here and should be checked before either figure is quoted.

## 17. Review of this note's own work, 2026-08-02

A deliberate review pass over the methodology and the code. It found one material statistical error, one code defect, and one robustness confirmation. The corrections are recorded here rather than silently edited into the earlier sections; those sections carry pointers.

### Statistical error: the empirical `b_kappa_mag` was estimated with the wrong functional form

§2 estimates the empirical slope by regressing `log kappa_a` on `z`. The model is `kappa(z) = kappa_min + exp(a_kappa - b_kappa_mag * z)`, so total `kappa` **flattens onto the floor** at older ages and its log-slope is shallower than `b_kappa_mag`. The log-linear regression therefore estimates something that is not the model's parameter, and estimates it low.

Refitting the actual three-parameter form to the same per-age cells:

| data          | usable cells | §2 log-linear `b` | correct 3-parameter `b` | 3-parameter `kappa_min` | model posterior |
| ------------- | ------------ | ----------------- | ----------------------- | ----------------------- | --------------- |
| DS spoken     | 12           | 1.38              | **2.17**                | 3.01                    | 2.337 (VG01)    |
| TD spoken     | 20           | 0.77              | **1.71**                | 3.21                    | 1.369 (VG03)    |
| TD understood | 18           | −0.04             | **−0.89**               | 10.64                   | 0.031 (VG04)    |
| DS understood | 5            | 0.76              | 5.08                    | 8.72                    | 0.725 (VG02)    |

The Down syndrome comprehension row rests on five age cells and its three-parameter fit is not stable; treat it as uninformative rather than as an estimate.

Three consequences.

**§13's explanation of the VG03 gap is wrong.** It reads the fitted 1.369 as sitting _above_ a marginal estimate of 0.77 and attributes the difference to the GP mean and study effects absorbing trend structure. With the correct estimator the comparison reverses — empirical 1.71 against fitted 1.369 — so the model sits _below_ the data and the absorption story explains a gap that does not exist. It was an artefact of the estimator, not a property of the fit.

**§15's account of VG01 is half right.** The claim that `b_kappa_mag` is "measuring how fast the exponential is switched off rather than a dispersion gradient" is undercut: the three-parameter empirical fit uses the model's own functional form and independently lands at 2.17 against the fitted 2.337. VG01 is recovering the data, not producing a parameterisation artefact. What survives from §15 is the **units** argument — `b_kappa_mag` is a per-standardised-age slope, so a shared prior is about 3.5 times tighter on VG01 than on VG03 in per-month terms — but that now explains only the _relative_ tightness.

**The implemented prior is still too tight, in absolute terms, for both models.** `HalfNormal(0.75)` has a 95th percentile of 1.47 against corrected targets of 1.71 and 2.17:

| prior                            | median | 95th | CDF at TD 1.71 | CDF at DS 2.17 |
| -------------------------------- | ------ | ---- | -------------- | -------------- |
| `HalfNormal(0.75)` (implemented) | 0.51   | 1.47 | 0.98           | 1.00           |
| `HalfNormal(1.00)`               | 0.67   | 1.96 | 0.91           | 0.97           |
| `HalfNormal(1.50)`               | 1.01   | 2.94 | 0.75           | 0.85           |

The recalibration was a large improvement — VG03 went from prior CDF 1.00 and contraction 0.18 to 0.93 and 0.82 — but it did not go far enough, and VG03's fitted 1.369 against an empirical 1.71 is consistent with the prior still pulling the estimate down.

### And the parameterisation, not the prior value, is now the binding constraint

Widening `b_kappa_mag` to the value the data want breaks the young-age prior predictive, and recentring `a_kappa` does not rescue it — the tails compound as `exp(2b)` at `z = −2`:

| prior set                                    | `kappa` at z = −2, median [5–95%] | P(`kappa` > 200 \| z = −2) |
| -------------------------------------------- | --------------------------------- | -------------------------- |
| implemented: `a N(log 8, .75)`, `b HN(0.75)` | 28.4 [7.6, 219]                   | 0.056                      |
| `a N(log 8, .75)`, `b HN(1.50)`              | 68.8 [9.6, 3396]                  | 0.302                      |
| `a N(log 4, .75)`, `b HN(1.50)`              | 36.7 [6.1, 1740]                  | 0.209                      |
| `a N(log 4, .60)`, `b HN(1.50)`              | 36.0 [6.5, 1617]                  | 0.204                      |

There is no setting of the current three parameters that both admits `b = 2.2` and keeps young-age dispersion plausible. The two-anchor form of §8 does, because it constrains the endpoints directly instead of an intercept and a slope. Priors on `kappa` at `z = ±1.5`, centred on the empirical young and old values and deliberately broad — `kappa_young ~ LogNormal(log 25, 0.6)`, `kappa_old ~ LogNormal(log 4, 0.6)` — give:

|        | `kappa` median | 5–95%         | P(`kappa` > 200) |
| ------ | -------------- | ------------- | ---------------- |
| z = −2 | 33.8           | [10.6, 108.4] | 0.006            |
| z = 0  | 10.0           | [5.0, 20.1]   | 0.000            |
| z = +2 | 2.9            | [0.9, 9.4]    | 0.000            |

against an empirical reference of 22–36 young, about 12 mid, 3–4 old. Both priors are directly checkable against the §2 table, and the explosive tail is gone.

Note that the slope implied by that anchor pair is about 0.61 — **not** comparable with `b_kappa_mag` of 1.7–2.2. The two-anchor slope describes total `kappa`; `b_kappa_mag` describes the exponential term sitting above a floor of about 3. The two parameterisations describe the same curve with different parameters, and the numbers must not be read across.

**§8 is therefore promoted from "direction of travel" to the recommended next change.** Widening `b_kappa_mag` a third time is not viable.

### Robustness check that passed

The typically-developing `kappa` decline could have been an artefact of the WG-to-WS form switch: WG dominates below 16 months and WS above, the forms have different item ceilings (396 and 680 on the common 810 scale), and mixing forms within an age cell would inflate apparent dispersion. It is not. Within each form the decline is as steep as in the pooled series:

| age | WG `kappa` | WS `kappa` | pooled |
| --- | ---------- | ---------- | ------ |
| 12  | 30.8       | —          | 34.3   |
| 14  | 21.6       | 14.7       | 20.4   |
| 16  | 16.0       | 14.6       | 15.2   |
| 18  | 10.2       | 9.0        | 9.4    |
| 24  | —          | 3.9        | 4.1    |
| 30  | —          | 3.0        | 3.0    |

The central empirical claim of this note survives the check.

### Code defect: the subject draw depended on row order

`_subsample_subjects` (§13) took `subject_key.unique()` and sampled by position. `Series.unique` preserves order of first appearance and the loader's DuckDB query carries no `ORDER BY`, so the same seed could select a **different set of children** if row order changed. Verified on a synthetic frame: shuffling the input and re-drawing at seed 47 shared only 10 of 30 subjects.

Fixed by sorting the key list before sampling, matching what the random-effect engines already do when assigning study and subject codes (`unique_studies = sorted(...)`, `unique_subjects = sorted(...)`). A fifth test covers it, asserting the draw is invariant to input row order.

The old row-wise code had the same latent defect, so this is not a regression introduced by §13 — but it was introduced into new code and should not have been.

**Consequence:** the analysis frames move again. VG03 goes 4,152 to 4,075 rows and VG04 1,550 to 1,555, so the §13 `test` fits are attached to the pre-sort frame. The shift is under 2% and no conclusion in this note depends on it, but both models need a refit before their numbers are quoted.

### Checked and correct

- `_PRODUCTION_KAPPA` is applied to VG01, VG03 and VG11 only; VG02, VG04 and VG12 retain the shared default, as §3 intends.
- VG16's anchoring matches VG10 and VG15, and correctly omits `anchor_g_sign_at_ref` (VG16 is bivariate).
- The mode comparison in §12 compares `p_query` — the median child, since the random effects are centred Normal on the logit scale and the logistic is monotone — with the pool's empirical **median**. That is the right pairing, and it is the pairing whose absence caused §11's error.
- The LOO reading in §12 is correct: mode B's better elpd is invalidated by `p_loo` of 1,113 on 1,626 observations with 1,097 Pareto k above 0.7, not accepted as evidence.
- Contraction is reported throughout as `1 - sd_post / sd_prior`, with prior sd taken analytically or from 20,000 prior draws.

### Revised open list

1. **Reparameterise `kappa` on two age anchors** (§8, §17). This now blocks a correct `b_kappa_mag` prior rather than merely tidying one.
2. **Refit VG03 and VG04** on the post-sort frame.
3. VG16, VG11, VG12, VG13 all need fits on current code; only VG01 and VG16 currently have fits matching the code as it stands.
4. Items 2–5 of §9 are unchanged: prior-predictive rerun, `kappa_s` conditional calibration, the `eta_q` test on VG10, and the PRIORS.md update — which should now also carry §17's estimator correction, since PRIORS.md's own `kappa` section uses the same per-age method.

## 18. Implemented: `kappa` reparameterised on two age anchors

§17 closed by promoting §8's two-anchor form from "direction of travel" to the blocking next change. This section implements it for the three univariate spoken models, and corrects §17's own proposal along the way: the form §17 simulated — a pure log-linear `kappa` with no floor — is not supported by the data.

The implemented curve is unchanged from the one every model already uses:

```text
kappa(z) = kappa_min + exp(a_kappa + b_kappa * z)
```

What changes is that `a_kappa` and `b_kappa` are no longer given priors. Priors go on the _age term_ `exp(a_kappa + b_kappa * z)` at two reference ages in months, and the intercept and slope are solved for so the curve passes through both. This is the move the mean trajectory already makes through `slope_anchors`.

### §17's proposed form was wrong: the floor is real

§17's simulation table dropped `kappa_min` and interpolated total `kappa` log-linearly between two anchors. Fitting the candidate forms to a _saturated_ mean — a free proportion per whole-month age cell, so the dispersion estimate carries no assumption about the mean model — rejects that:

| pool                   |      n | cells | no-floor log-linear | best clamped |
| ---------------------- | -----: | ----: | ------------------: | -----------: |
| DS spoken (VG01)       |  1,114 |    25 |              +10.50 |        +1.49 |
| TD spoken (VG03 frame) |  4,075 |    23 |              +60.14 |       +10.55 |
| TD spoken (VG11 frame) | 16,235 |    23 |             +167.78 |       −23.95 |
| TD understood (VG04)   |  1,555 |    16 |               +0.28 |        −0.54 |
| TD understood (VG12)   |  5,997 |    18 |               +1.84 |        −0.27 |
| DS understood (VG02)   |    671 |    15 |               +0.98 |        +0.49 |

Negative log-likelihood relative to the three-parameter floored form; positive is worse. The no-floor form costs 10 to 168 units on the three spoken pools, because a single log-linear slope cannot be steep through the second year and flat afterwards at the same time: on the VG11 frame it predicts `kappa` 8.9 at 19 months against an observed 6.1, and 2.4 at 30 months against 3.0. Total `kappa` genuinely flattens onto a plateau near 3 rather than continuing to decay. The 168 units come from the dense 15–22 month band — 40% of that frame's 16,235 observations — not from the sparse extremes. It costs nothing on the comprehension pools only because their `kappa` is flat, so a floor has nothing to do.

The "clamped" column is the other candidate: log-linear between the anchors, held flat outside them, which is the idiom `gp_utils.tent_and_gp` already uses for the signed ratio's mean. Its apparent advantage on VG11 is not real. The column reports the best over a grid of anchor pairs — fifteen for the spoken pools — and on VG11 the spread across that grid runs from −24 to **+1176**. A form whose fit swings by 1,200 log-likelihood units on a choice made by profiling has no defensible fixed setting. It also introduces gradient discontinuities in the dispersion of the likelihood itself, which is a worse place for them than in a mean a GP can smooth over. Rejected.

So the change is a reparameterisation and nothing more: the likelihood family is identical, and a fit under the new priors is directly comparable with one under the old.

### Where the anchors go

Place them where the age term is roughly an order of magnitude above the floor, and where it has fallen back to it. Between those two ages the exponential carries the curve and outside them the floor does, so both priors sit where the data can identify them. Applied to the three-parameter fits above:

| pool                   | `kappa_min` |  `b` | anchors | total `kappa` at the anchors |
| ---------------------- | ----------: | ---: | ------- | ---------------------------- |
| DS spoken (VG01)       |        3.54 | 2.78 | 18, 36  | 49.1, 7.66                   |
| TD spoken (VG03 frame) |        3.08 | 1.78 | 12, 20  | 37.1, 6.15                   |
| TD spoken (VG11 frame) |        3.08 | 1.50 | 12, 20  | 29.9, 6.55                   |

The implemented excess medians are those totals minus the floor, rounded: 45 and 4.0 for Down syndrome, 30 and 3.0 for typically-developing (splitting the two frames, which share their trajectory priors and now share their dispersion prior too).

### The anchors are stable where the slope is not

This is the argument for the reparameterisation that the earlier sections missed, and it is the strongest one. §17 reported an empirical `b_kappa_mag` of 2.17 for Down syndrome spoken; §18's fit gives 2.78. The difference is entirely the age-cell inclusion rule — §1 drops cells with fewer than 25 administrations or a mean proportion outside `(0.01, 0.95)`, this section keeps every cell with at least 15.

Neither rule is obviously right, and the proportion filter is doing something arbitrary at exactly the wrong place. It is meant to exclude cells where `kappa` is unidentified against the floor, and on the 12-month Down syndrome cell it is correct to: 27 administrations, 81% of them zero, `kappa` estimated at 602 on a profile interval of [57, 9201]. But it also drops the 17-month cell, which is well identified (`kappa` 69.4 on 53 administrations, [36.1, 119.5]) and adjacent to the young anchor, purely because its mean of 5.6 words is below `0.01 * 810`. The 18-month cell survives on 8.15 words against a threshold of 8.1. A rule that separates two neighbouring, comparably-identified cells by a fifth of a word is not measuring identifiability.

Refitting under four inclusion rules:

| pool | `b_kappa_mag` range | young-anchor excess | old-anchor excess   |
| ---- | ------------------- | ------------------- | ------------------- |
| VG01 | 1.93 – 2.78 (1.44x) | 29.6 – 45.5 (1.54x) | 4.11 – 5.58 (1.36x) |
| VG03 | 1.67 – 1.78 (1.07x) | 30.3 – 34.0 (1.12x) | 3.06 – 3.19 (1.04x) |
| VG11 | 1.50 – 1.78 (1.19x) | 26.8 – 33.0 (1.23x) | 2.93 – 3.48 (1.19x) |

The legacy form asks for a prior on the one quantity that no cell-selection rule pins down. The anchored form asks for priors on quantities the data determine better, and makes the slope a derived consequence. That is a property of the parameterisation, not of this particular calibration.

### Calibrating the spread, and a correction about `kappa > 200`

`sigma = 0.7` on each anchor — a 5–95% range of about ±3.2x — chosen so each anchor's range covers the spread of defensible estimates for it, including the per-age cells on either side of the anchor age, which scatter more than any smooth fit does. At the typically-developing young anchor the 11-, 12- and 13-month cells give total `kappa` of 20.3, 89.4 and 37.0 on profile intervals that do not overlap ([12.2, 31.2] on 86 administrations, [65.6, 119.1] on 162, [29.4, 45.7] on 271). The scatter is real between-study composition, not noise, and no smooth curve passes through all three. Against an excess prior centred at 30, `sigma = 0.6` would have put the high cell at the 96th percentile; 0.7 brings it to the 93rd. Erring wide is deliberate — the failure being repaired is a prior too tight to let the data speak.

`kappa_min` is carried over unchanged at `LogNormal(log 3, 0.8)`, so this is a single-factor change against the §10 and §13 fits.

The resulting prior, simulated on each model's own age grid (median, 5–95%):

| model      | 8 mo           | 12 mo         | 18 mo        | 20 mo         | 36 mo       | oldest        |
| ---------- | -------------- | ------------- | ------------ | ------------- | ----------- | ------------- |
| VG01       | 177 [30, 1150] | 105 [24, 491] | 49 [18, 147] | —             | 7.9 [3, 20] | 3.1 [0.8, 12] |
| VG03, VG11 | 99 [19, 586]   | 34 [13, 99]   | 9.2 [4, 20]  | 6.8 [2.8, 17] | —           | 3.6 [1.0, 13] |

> [!CAUTION]
> Those young-age upper tails would fail the criterion `_PRODUCTION_KAPPA` was tuned against — "P(`kappa` > 200), near-binomial, empirically false". **That criterion was wrong, and it is worth retiring explicitly.** At `n = 810` the Beta-Binomial variance inflation over binomial is `(n + kappa) / (1 + kappa)`, so `kappa = 200` still gives 2.24x the binomial standard deviation. Near-binomial at this `n` would need `kappa` in the tens of thousands. And the observed dispersion at those ages really is in that range: the Down syndrome 18-month cell estimates 64.5 and the typically-developing 12-month cell 89.4, both on intervals that exclude anything close to binomial. The old comment's "observed young-age `kappa` is 22–36" came from a per-age table whose youngest cells had been filtered out.

The one genuine caution is at the _other_ end. Beyond the old anchor the floor alone sets the level, where in the legacy form the exponential term propped it up, so `kappa_min`'s ~8% of prior mass below 1 now shows at old ages (P(`kappa` < 1) reaches 0.08 for VG01 at 115 months). Tightening `kappa_min_sigma` is a candidate follow-up, deliberately not folded into this change.

### A property worth naming

The prior on `kappa` at a given age is **exactly invariant** to the pool's age standardisation. The interpolation weight is `(age − young) / (old − young)`, and since `z` is affine in age the standardisation cancels identically. Resampling, a study filter, or a change to `sample_fraction` cannot move what the prior says — which is precisely the failure mode §11 and §13 spent this note chasing in the data pipeline, and which `a_kappa` (defined at the pool mean age) is subject to by construction. There is a test asserting it across two deliberately different standardisations.

### Changed in the repository

- `models/definitions.py` — new `KappaAnchorPriorParams`; `_PRODUCTION_KAPPA` replaced by `_DS_SPOKEN_KAPPA` and `_TD_SPOKEN_KAPPA`, one per population rather than one shared block (the units problem of §15: a single `b_kappa_mag` prior is ~3.5x tighter on the DS pool than the TD pool in per-month terms; anchors in months are immune). Anchor ages are validated as ordered and inside the GP domain, like every other reference age.
- `models/gp_utils.py` — `build_kappa_of_z_anchored` alongside the unchanged `build_kappa_of_z`. It stores `a_kappa` and `b_kappa` as deterministics under the legacy names, so the dispersion posterior stays comparable across the two forms, plus `kappa_young` / `kappa_old` carrying total `kappa` at the anchors.
- `models/common.py` — `AnchoredKappaPriors`; `ModelConfiguration` now rejects a mixed or partial dispersion specification in `__post_init__`; `build_kappa_for_config` is the single dispatch point, used by both univariate engines and by VG17.
- `models/build_utils.py` — `standardize_anchor_ages`, with `slope_anchor_logit_coeffs` delegating to it, so every anchored parameterisation reaches the graph through one conversion.
- `models/model_vg17.py` — follows whichever form VG01 carries, so "reuses VG01's priors" stays true.
- `scripts/prior_vs_posterior.py` — plots only the _free_ parameters of whichever form a model uses, and raises rather than silently mis-plotting if a bivariate model migrates.
- `docs/models/vg01|vg03|vg11/index.qmd`, `docs/models/PRIORS.md` — the dispersion prior figures and the prior rationale.
- Tests: 6 for the builder's algebra, 15 for the configuration contract and dispatch, 4 for the definitions. The full suite is 391 tests, all passing.

**Not migrated:** VG02, VG04, VG12 (comprehension) and `kappa_s` in the joint models. Comprehension would benefit most from the freed sign — its dispersion is flat to slightly rising, which `b_kappa <= 0` cannot represent at all — but its anchors need their own calibration (the TD understood floor is about 10.6, not 3), and `kappa_s` governs dispersion of the production ratio on a per-child denominator, to which none of this marginal calibration transfers. A test asserts the split rather than leaving it to review.

### Results

`test`-config refits, 4 chains x 2,000 draws, `target_accept` 0.9. Prior CDF is evaluated at the posterior mean; for the derived `b_kappa_mag` the implied prior is exactly Normal, being a difference of two Normal log-anchors over a constant.

| fit           | parameter               | post mean | prior CDF | contraction |    ESS |
| ------------- | ----------------------- | --------: | --------: | ----------: | -----: |
| VG01 legacy   | `kappa_min`             |     2.779 |     0.462 |       0.937 |  6,825 |
|               | `a_kappa`               |     1.102 | **0.096** |       0.783 |  4,957 |
|               | `b_kappa_mag`           |     2.340 | **0.998** |       0.520 |  4,881 |
| VG01 anchored | `kappa_min`             |     2.975 |     0.496 |       0.934 |  8,398 |
|               | `kappa_excess_young`    |    42.828 |     0.472 |       0.865 | 11,268 |
|               | `kappa_excess_old`      |     4.220 |     0.530 |       0.827 |  8,319 |
|               | `b_kappa_mag` (derived) |     2.685 | **0.460** |       0.763 |  8,523 |
| VG03 legacy   | `kappa_min`             |     2.532 |     0.416 |       0.948 | 10,199 |
|               | `a_kappa`               |     1.537 |     0.235 |       0.893 | 10,030 |
|               | `b_kappa_mag`           |     1.371 | **0.932** |       0.823 | 10,093 |
| VG03 anchored | `kappa_min`             |     2.938 |     0.490 |       0.958 | 10,866 |
|               | `kappa_excess_young`    |    33.293 |     0.559 |       0.927 |  9,320 |
|               | `kappa_excess_old`      |     3.164 |     0.530 |       0.900 | 10,456 |
|               | `b_kappa_mag` (derived) |     1.745 | **0.521** |       0.871 | 10,201 |

**The prior-data conflict is gone.** Under the legacy form VG01's slope sat at prior CDF 0.998 _and_ its intercept at 0.096 — the compensating pair §6 predicted and §8 named. Every anchored parameter, in both models, now sits between 0.46 and 0.56. Contraction is unchanged or better on every comparable parameter despite much wider priors, and the dispersion block's effective sample size roughly doubled on VG01.

**And the fits recover the empirical curve.** Reading each posterior back onto total `kappa` at the two anchor ages:

| model | source                 | `kappa_min` | young anchor | old anchor | `b_kappa_mag` |
| ----- | ---------------------- | ----------: | -----------: | ---------: | ------------: |
| VG01  | empirical, 3-parameter |        3.54 |        49.06 |       7.66 |         2.781 |
|       | legacy posterior       |        2.78 |        40.27 |       7.74 |         2.340 |
|       | anchored posterior     |        2.98 |        45.80 |       7.20 |         2.685 |
| VG03  | empirical, 3-parameter |        3.08 |        37.12 |       6.15 |         1.783 |
|       | legacy posterior       |        2.53 |        29.42 |       6.76 |         1.371 |
|       | anchored posterior     |        2.94 |        36.23 |       6.10 |         1.745 |

VG03's anchored posterior matches the marginal empirical fit to within a few percent on all four quantities, having previously understated 12-month dispersion concentration by 26%. This confirms §17's prediction that "VG03's fitted 1.369 against an empirical 1.71 is consistent with the prior still pulling the estimate down" — it was, and releasing the parameterisation releases the estimate.

VG03 also **passes the convergence gate outright**: 0 divergences, max R-hat 1.0018, min ESS 2,569. That is the first clean gate pass anywhere in this note.

> [!NOTE]
> Only VG01 is a clean single-factor comparison — it does not subsample, so the prior is the only thing that changed. VG03's legacy fit was made on the pre-sort frame (4,152 rows against 4,075), so its comparison crosses the §17 row-order fix as well; the shift is under 2% and the direction of every change is large against it, but it is not a controlled contrast. VG11 has no usable "before" at all: its promoted fit is the row-wise bimodal one at R-hat 1.72.

### Sampling cost

VG01's sampling took 1m 08s; VG03's took 20m 15s. The gap is not the reparameterisation acting on the dispersion block, whose posterior geometry is benign in both — the two anchors correlate at −0.08 (VG01) and −0.20 (VG03), and the expected `kappa_min` / old-anchor trade-off at −0.59 and −0.75. It is step size and trajectory length: VG01 runs at step size 0.17 and a mean of 50 leapfrog steps, VG03 at 0.08 and 230. With VG03 also carrying 3.7x the observations per gradient, 3.7 x 4.6 = 17x accounts for the 17.9x observed almost exactly.

Per-draw statistical efficiency is unchanged by the reparameterisation: VG03's dispersion ESS is 9,320–10,866 anchored against 10,030–10,199 legacy, from the same 8,000 draws. What the two fits cost _per draw_ is not established, because the legacy VG03 trace was overwritten and only its summaries were kept. A paired `dev`-config control on the same post-sort frame is the outstanding check.

### Open list after §18

Resolved by this section: §17's items 1 (the reparameterisation), the VG03 half of item 2, the PRIORS.md half of item 4, and §14's open items 1 (restate `b_kappa_mag` in interpretable units) and 6.

> [!CAUTION]
> **Items 1, 2 and 8 below are superseded by §19.** Item 1 is done — the estimator exists and VG11, VG12 and VG13 are recalibrated — and it turned up two things this list did not anticipate: VG12 and VG13 were mis-centred as badly as VG11 but silently, and the Down syndrome joint frame cannot support the calibration at all, so VG09/VG10/VG16 are deliberately left out. Item 2's premise is wrong in its specifics: the typically-developing understood dispersion is not "nearly flat" on the conditional scale, it rises. Item 8's supporting argument — that `tau_subject` and `kappa` are indistinguishable on this design — is disproved in §19.

Still open, in rough priority order:

1. **Calibrate `kappa` conditionally for the random-effect models.** Newly the top item: VG11's fit (below) shows a marginally-calibrated dispersion prior is out by a factor of ten once subject effects absorb the between-child spread. Needs per-age Beta-Binomial fits to residuals after study and subject effects, not to raw counts. This is the same piece of work §7 scheduled for `kappa_s`; do it once and apply it to VG11, VG12, VG13 and the joint models together. **Until then `_TD_SPOKEN_KAPPA` on VG11 is known to be mis-centred** — it is left in place only because reverting to `_PRODUCTION_KAPPA` would substitute one mis-calibrated prior for another, and the anchored form at least makes the miss legible.
2. **Extend the two-anchor form to comprehension** — VG02, VG04, VG12. They gain the most, because the freed sign is what their data need, but each needs its own anchor calibration: the typically-developing understood floor is about 10.6 against 3 for spoken, and the curve is nearly flat, so both anchors sit close together and the implied slope prior will be near-symmetric about zero. Do not reuse either block from §18. VG12 also needs item 1 first.
3. **Refit VG04** on the post-sort frame (§17's item 2, still outstanding), and **VG12, VG13, VG16** on current code.
4. **Rerun `scripts/prior_predictive_audit.py`** for VG01, VG03 and VG11. Its `kappa` rows are stale twice over now.
5. **Test `eta_q = 0.4` on VG10** (§5). Untouched by any of this.
6. **Consider tightening `kappa_min_sigma`.** Newly consequential: with the age term interpolated between anchors, the floor alone sets the level beyond the old anchor, so its 8% of prior mass below `kappa = 1` is now visible at old ages. Held back deliberately so §18 stays a single-factor change.
7. **Pair a `dev`-config control against VG03's anchored fit** to settle whether the parameterisation costs anything per draw. Statistical efficiency is already known to be unchanged; wall-clock is not.
8. **Whether VG11 should carry subject random effects at all** at 1.32 administrations per child (§14's item 5). VG11's `tau_subject` posterior of 1.059 with a standard deviation of 0.009, alongside `kappa` at ten times its marginal estimate, is that question presenting itself rather than a separate finding.

### VG11 rejects the shared typically-developing anchors, for a reason that is not the parameterisation

> [!NOTE]
> **Diagnosis confirmed by §19, conclusion partly overturned.** The mechanism described below is right, and §19 measures it: VG11's conditional `kappa` at 12 months is 317.5 against the 30 this section's prior encoded. But the closing paragraph's two claims do not survive. `tau_subject` and `kappa` _are_ separable on this design (§19 recovers each from simulation), and VG11's dispersion posterior was accurate rather than untrustworthy — 310 against a conditional estimate of 317.5. The prior was wrong; the fit was not.

VG11's `test` fit (16,235 observations, 12,266 subjects, 39m 55s) is unimodal — max R-hat 1.017 against the 1.718 of the row-wise fit it replaces — but it does **not** pass the convergence gate (13 divergences, min ESS 203, R-hat and ESS failures on `ell_unit`, `eta`, `tau`, `ell`, three HSGP coefficients and `delta_raw[0]`). And its dispersion posterior is nowhere near its prior:

| parameter            | post mean | prior median | prior CDF | contraction |
| -------------------- | --------: | -----------: | --------: | ----------: |
| `kappa_min`          |      5.78 |         3.00 |     0.794 |       0.748 |
| `kappa_excess_young` |    305.80 |        30.00 | **1.000** |       0.405 |
| `kappa_excess_old`   |     44.16 |         3.00 | **1.000** |       0.188 |
| `b_kappa_mag`        |     1.422 |         1.69 |     0.355 |       0.913 |
| `tau_subject`        |     1.059 |         0.34 |     0.966 |       0.970 |
| `tau` (study)        |     0.413 |         0.34 |     0.591 |       0.589 |

Total `kappa` at 12 months is 312, against 36 for VG03 on the same outcome and 30 from the marginal empirical fit to VG11's own frame. A factor of ten.

**This is not a failure of the reparameterisation, and it would have happened to any prior calibrated the same way.** VG11 carries subject random intercepts and `tau_subject` lands at 1.06 on the logit scale — pinned, at prior CDF 0.966 and contraction 0.970. Those effects absorb the between-child spread that, in VG03, the Beta-Binomial has to carry itself, so what is left at the observation level is nearly binomial and `kappa` rises accordingly. §1 anticipated the direction — "a marginal `kappa` is a lower bound on the model's `kappa`" — but not the magnitude, and §6 applied the calibration to VG11 anyway.

It is the same error §7 identified for `kappa_s` and declined to make: **a dispersion prior calibrated marginally does not transfer to a model that conditions on random effects.** VG01 and VG03 have neither subject nor study effects, which is exactly why their fits are clean and centred.

There is a second reading, and the two are not separable on this evidence. `tau_subject` at 1.06 with a posterior standard deviation of 0.009 on a pool with 1.32 administrations per child — only 1,947 of 12,266 subjects have a repeat — is what §14's open item 5 predicted: the subject random effect and the observation-level dispersion are competing for the same variance, and the data cannot arbitrate. A prior at CDF 1.000 on one side and 0.966 on the other is what that looks like from the outside. Whether VG11 needs a conditional dispersion calibration, or should not carry subject effects at this replication, is now the question — and it is a question about VG11's structure, not about `kappa`'s parameterisation.

**Action:** VG11 should not be quoted from this fit either. It needs `kappa` anchors calibrated _conditionally_ — per-age Beta-Binomial fits to residuals after subject and study effects, not to raw counts — which is the same piece of work §7 scheduled for `kappa_s`, and should be done once for both. Until then VG01 and VG03 are the two models this section validates.

## 19. Implemented: `kappa` calibrated conditionally for the random-effect models

§18's open item 1, and the resolution of the VG11 failure it ended on. §7 first identified this error class and declined to make it; §6 made it for VG11; §18 measured it at a factor of ten and could not fix it. This section builds the estimator that can, validates it, and applies it to the three typically-developing random-effect models. It also finds that VG12 and VG13 were affected as badly as VG11 and in a way that had gone unremarked, and that the Down syndrome joint frame cannot support the calibration at all.

### What the marginal estimate is an estimate of

The per-age fits in §§1–2 and §18 answer one question: how much do counts vary at a given age? For VG01 and VG03 that is the right question, because nothing in those models removes any of that variation before the likelihood sees it — the Beta-Binomial carries all of it.

A model with study and subject random intercepts has already removed most of it. Its `kappa` answers a different question: how much variation is left once this child's own level is known? That residual is necessarily smaller, so `kappa` is necessarily larger, and the marginal number is a lower bound rather than an approximation. §1 said as much — "a marginal `kappa` is a lower bound on the model's `kappa`" — and then §6 used it anyway.

The size of the gap is the point. A lower bound off by 20% is a usable prior centre. Off by a factor of ten it is a prior for a different quantity.

### The estimator

`scripts/kappa_conditional_calibration.py` fits the same saturated mean the marginal calibration uses, with the random effects present:

```text
logit p_ij = m_c(ij) + s_k(i) + b_i,     b_i ~ N(0, tau^2)
y_ij       ~ BetaBinomial(N_ij, p_ij, kappa(a_ij))
```

`m_c` is free per integer-age cell, so dispersion is still estimated given whatever the mean does. Study effects are fixed and sum-to-zero, matching the engines' `ZeroSumNormal` intercepts (with a handful of large studies the shrinkage is negligible, so fixed and random coincide). The subject effect is integrated out by Gauss-Hermite quadrature; the log-likelihood is differentiated with JAX and maximised with L-BFGS. `kappa` is parameterised directly as the floor plus the age term at each of two anchor ages, so the Hessian returns standard errors on the quantities the prior is actually stated on.

For the nested spoken outcome this is the conditional scale §7 asked for: successes are the spoken count, the denominator is that child's own observed understood count, and the mean is the production ratio `q`.

### Three things had to hold before reading a prior off it

**The design must be able to tell `tau` from `kappa`.** For a child measured once both add variance to the same single number, and 84% of VG11's children are measured once. They are not, however, strictly confounded: a logit-normal random effect and a Beta-Binomial leave differently _shaped_ count distributions, and on a synthetic 900-singleton design a `tau` of 1.0 still recovers to within 9%. What the repeats buy is resolution at the small end — strip them out and a `tau` of 0.3 comes back anywhere in 0.001–0.48, which is precisely the regime that decides whether a model needs a conditional prior at all. §18 concluded from that "the data cannot arbitrate". **That was too pessimistic**, and this is the first correction this section makes to §18. Simulating from two opposite truths on VG11's real subject/study/age structure and refitting recovers each. (This check is what excludes the Down syndrome pool below, so it is not a formality that every pool passes.)

| truth            | `tau` | `kappa(12)` | `kappa(20)` | recovered `tau` | recovered `kappa(12)` | recovered `kappa(20)` |
| ---------------- | ----: | ----------: | ----------: | --------------: | --------------------: | --------------------: |
| subject-heavy    |  1.06 |       317.5 |        50.5 |   1.059 / 1.050 |         337.7 / 297.1 |           50.4 / 51.0 |
| dispersion-heavy |  0.15 |        30.0 |         6.6 |   0.137 / 0.146 |           30.4 / 30.3 |             6.6 / 6.7 |

Two seeds each. The 1,947 children with a repeat carry the separating information: on this design the posterior correlation between `log tau` and the anchors stays below 0.33 in every recovery run, and the two regimes come back an order of magnitude apart rather than collapsing to a common answer. What §18 saw — a prior at CDF 1.000 on one side and 0.966 on the other — is not two parameters trading off; it is two priors that were both wrong.

The separation is not equally comfortable everywhere. On the real fits the same correlation is 0.27 for VG11 and 0.39 for VG13's `q`, but **0.71 for VG12** — its `tau` and young anchor are genuinely competing, which is a further reason its prior is left wide. A test asserts the negative control too: with repeats stripped from the design, the estimator does _not_ recover the truth, so the checks above are testing something.

**The quadrature has to converge.** It does not at the node count a first pass would choose. At `tau` near 1 the subject distribution is wide, and under-integrating it understates the spread the random effect accounts for, so the dispersion absorbs the difference and `kappa` comes out low. On VG11: 292.6 at 24 nodes, 313.3 at 48, 317.6 at 96, 318.5 at 240. All four pools are converged by 160, which is the default. The recovery table above is at 160; at 24 the same simulation returned 265 against a truth of 303.

**The answer must not depend on the mean model.** The saturated mean fits every age cell exactly where the models use a smooth HSGP, so a gap between this estimate and a posterior could be an artefact of the estimator tracking the age curve more closely. Sweeping from a 6-parameter spline to saturated, and separately across polynomial means:

| pool            | linear mean | quadratic | spline[8] | saturated |
| --------------- | ----------: | --------: | --------: | --------: |
| VG11 spoken     |       166.2 |     318.3 |     316.3 |     317.5 |
| VG12 understood |        33.4 |      44.2 |      42.4 |      43.0 |
| VG13 understood |        41.6 |      41.7 |      41.7 |      42.2 |
| VG13 `q`        |        35.3 |      35.2 |      35.6 |      36.0 |

`kappa` at the young anchor. From a quadratic upward nothing moves by more than 3%, so the gaps reported below are properties of the data and not of this estimator.

### Results

Conditional fits at 160 nodes on a saturated mean, against the marginal estimate on the same rows and against each model's posterior. **The posterior column is the fit that preceded this section** — the pre-change priors — since that is what makes the comparison informative; the refits are further below and overwrite those figures in `output/`. Only VG11's was `test` config; VG12's, VG13's and the Down syndrome joint models' were `dev` (2 chains x 500 draws) and are indicative, so the gaps below are read as order-of-magnitude evidence and not as precise discrepancies.

| pool                     |      n | obs/child | `tau` | conditional `kappa` at anchors | marginal | posterior [94% CI]            |
| ------------------------ | -----: | --------: | ----: | ------------------------------ | -------- | ----------------------------- |
| VG11 spoken (12, 20)     | 16,235 |      1.32 |  1.06 | **317.5**, **50.5**            | 30, 6.7  | 310 [281, 339], 50.0 [47, 53] |
| VG12 understood (12, 20) |  5,997 |      1.26 |  0.74 | **43.0**, **66.4**             | 11, 13   | 16.5 [13, 21], 15.6 [12, 19]  |
| VG13 understood (12, 17) |  5,406 |      1.19 |  0.77 | **42.2**, **124.1**            | 10.8, 13 | 15.9 [12, 21], 14.4 [11, 19]  |
| VG13 `q` (12, 17)        |  5,320 |      1.19 |  1.12 | **36.0**, **29.7**             | 5.8, 4.1 | 40.4 [20, 64], 29.7 [17, 50]  |
| VG09/10/16 U (24, 48)    |    671 |      1.73 |  0.85 | 81.6, 20.3 _(unstable)_        | 12, 5.1  | 66.3 [55, 79], 19.5 [16, 23]  |
| VG09/10/16 `q` (24, 48)  |    645 |      1.74 |  1.15 | 13.8, 7.6 _(unstable)_         | 4.5, 2.0 | 20.5 [17, 24], 16.5 [15, 19]  |

Every conditional estimate is 3–10x its marginal counterpart, and the likelihood-ratio statistic against `tau = 0` is between 117 and 4,353 on 1 degree of freedom. The random-effect absorption is not a subtle correction to any of these pools.

Three separate patterns sit in that table.

**VG11 and VG13's `q` were right all along.** Their posteriors match the conditional estimate closely — 310 against 317.5, 50.0 against 50.5; 40.4 against 36.0, 29.7 against 29.7 — which means the likelihood was overwhelming a bad prior rather than being distorted by it. Re-centring removes a prior-data conflict without moving the answer. It also retrospectively vindicates VG11's `test` fit, which §18 said should not be quoted: the dispersion figure it reported was correct, and the reason to distrust it was the prior it was fighting, not the number it reached.

**VG12 and VG13's understood outcome were not.** Both posteriors sit near 16 where the data say 42, with the 94% credible interval nowhere near it. These models were on the legacy form, whose prior puts `kappa` at roughly 13 at the pool mean age — and the posteriors had barely moved off it, on 5,997 and 5,406 observations.

Their `b_kappa_mag` posteriors show why, and are the most direct evidence in this section that the constraint rather than the data was in charge:

| model          | `b_kappa_mag` mean | 89% ETI        | prior             |
| -------------- | -----------------: | -------------- | ----------------- |
| VG12           |              0.075 | [0.004, 0.224] | `HalfNormal(0.3)` |
| VG13 `kappa_u` |              0.131 | [0.013, 0.352] | `HalfNormal(0.3)` |

Both are pressed against the zero boundary with the interval's lower limit at the constraint. `b_kappa_mag >= 0` forces dispersion to fall with age; these data want it to rise, so the posterior does the only thing it can and piles up at "as close to flat as permitted". No setting of the three legacy parameters fixes that — it is the parameterisation, not the numbers.

This is the finding that was not visible before: §18 flagged VG11 because its prior conflict was loud, and these two were quietly prior-dominated instead — a posterior sitting _on_ its prior median attracts no attention, which is exactly the failure mode a prior-CDF check is meant to catch and a glance at the trajectory plots is not.

One caveat on the strength of this comparison. Neither `dev` fit passes the convergence gate, and VG12's `a_kappa` in particular has a bulk ESS of 23 and R-hat 1.075, so its posterior mean is not precisely located. That is why the argument above rests on the boundary behaviour of `b_kappa_mag` (ESS 283 and 372, R-hat 1.017 and 1.003, both well mixed) rather than on the exact value of `kappa`. The refits below are the proper test.

**The Down syndrome joint frame cannot be calibrated from.** Its replication is the best of any pool — 1.73 administrations per child, 208 of 387 children with a repeat — but 671 rows spread over 12–46 months is not enough. The recovery check settles it directly: on this design, at the same spline basis and node count the estimate would use, a known `kappa(24) = 317` comes back as 465 and 842 for understood and 260 and 517 for `q`, with one of the four fits failing to converge. Even in the easier low-dispersion regime, where `kappa(24) = 30` recovers to within 11%, `kappa(48) = 6.6` comes back at 8.3–12.9. The estimator is not wrong here so much as unconstrained: it cannot recover a truth on this design, so it cannot report one.

The mean sweep says the same thing more mildly. Varying only the spline knot count, on identical rows, moves `kappa` at the young `q` anchor from 13.0 to 23.6, a factor of 1.8; the typically-developing pools move by 3% under the same sweep. (Its saturated fit moves further still, to 30.9, but that is not a like-for-like comparison: the 15-observation cell rule drops 325 of the 671 rows, so the mean model and the row set change together — which is why the saturated mean is unusable here and the spline basis was added in the first place.)

A second limitation compounds it. 469 of 1,114 spoken rows fall back to the engine's marginal out-of-810 likelihood because the understood count is missing or violated, so `kappa_s` there governs two different scales and this calibration covers only the 58% on the nested one. On VG13 the same figure is 2%, which is why it does not arise there.

**VG09, VG10 and VG16 therefore keep their existing priors, and that is a finding rather than an omission.** It should not be read as evidence they are correctly calibrated — the presumption from every pool that _could_ be measured is that a marginally-calibrated prior on a random-effect model is several times too low, and nothing here rules that out for them. It means only that this estimator cannot say by how much. VG15 is out of scope for a different reason: its cross-tabulated four-cell frame is not one this estimator reproduces.

> [!NOTE]
> **Superseded by §22.** The recovery check above varied `tau` and `kappa` together and read the two-sided spread. Holding `tau` at its fitted value and varying only `kappa` shows the error is one-directional and monotone in the level, so the estimates are lower bounds rather than noise — usable, with the bias divided back out and a wide `sigma`. All four models are now migrated. What stands from this paragraph is the caution: the frame is the thinnest in the family and the resulting prior is correspondingly the weakest.

### The typically-developing understood rise is real but not smooth

The two-anchor fit summarises VG13's understood dispersion as rising from 42 at 12 months to 124 at 17. Fitting a free `kappa` per age cell, with the random effects still present, shows what that is smoothing:

| age (months) |   8 |   9 |  10 |  11 |  12 |   13 |   14 |   15 |    16 |   17 |    18 |
| ------------ | --: | --: | --: | --: | --: | ---: | ---: | ---: | ----: | ---: | ----: |
| VG13 `kappa` |  23 |  32 |  29 |  28 |  45 |   51 | 21.6 | 23.4 | 183.2 | 80.3 | 127.9 |
| n            | 252 | 289 | 261 | 321 | 586 | 1036 |  504 |  322 |   917 |  482 |   436 |

A factor of eight between adjacent cells at 15 and 16 months, on 322 and 917 observations — too large and on too much data to be sampling noise, and not a monotone trend either. VG12 shows the same shape on the same rows (19.6, 21.0, 110.7 at 14, 15, 16). **Why the 16–18 month cells sit so far above their neighbours is not established here**, and the obvious candidates — a CDI form boundary, a single dataset dominating the 16-month cell — are not tested. The prior response is to centre on the fitted rise but widen both understood anchors to `sigma = 0.9` (5–95% about ±4.4x) rather than the 0.7 the spoken and ratio anchors use, so the prior admits a flat or falling curve too. Treating the log-linear fit as if it described the profile would overstate what is known.

### Changed in the repository

- **`scripts/kappa_conditional_calibration.py`** (new) — the estimator, the six pool definitions, and `--recover` / `--mean-sweep`, which are the checks above and are meant to be run before adding a pool.
- **`tests/test_kappa_conditional_calibration.py`** (new) — recovery in both regimes on a small synthetic design, the negative control (with repeats removed, the truth is _not_ recovered), the node-count bias, and the anchor algebra.
- **The two-anchor form now works per outcome in the joint engines.** `build_kappa_for_config`, `kappa_prior_rows`, `kappa_anchor_derived_rows` and `_configure_kappa_priors` all take a suffix that selects both the configuration field and the variable names, and a new shared `validate_kappa_fields` rejects a half-specified or doubly-specified outcome. `BivariateModelConfiguration` gained `kappa_anchored_u` / `kappa_anchored_s`; both bivariate build paths route through the shared builder. Outcomes are independent — VG13 anchors both, the Down syndrome joint models anchor neither — and a test covers the mixed case, which no registered model currently uses.
- **Four new conditional prior blocks** in `definitions.py`, applied to VG11 (`kappa`), VG12 (`kappa`), and VG13 (`kappa_u`, `kappa_s`). VG12 and VG13 also migrate from the legacy form in the process, which resolves §18's open item 2 for VG12 and was not previously scheduled for VG13.
- **`scripts/prior_vs_posterior.py`** now handles anchored joint models; it previously raised `NotImplementedError` as a deliberate tripwire for exactly this case, and VG13 is the model that would have hit it.
- **The migration guard in `tests/test_model_definitions.py`** now records which outcomes are anchored and why, so the DS joint models' exclusion is asserted rather than inferred from absence.

### `kappa_min` stops meaning "floor"

Worth recording because it is easy to misread in the definitions. With `b_kappa > 0` the exponential term vanishes at _young_ ages rather than old ones, so `kappa_min` becomes the young-age asymptote. VG13's is 30, not 3, and that is not a discrepancy: a third of its 8–18 month frame sits below the 12-month anchor, where the rising exponential contributes almost nothing, and the 8–11 month cells estimate 23–32. VG12's conditional fit puts no mass on a floor at all — it goes to zero with an unbounded standard error, since a rising curve never reaches one inside the frame — so it keeps the weak default and lets the anchors carry the level.

### Corrections to §18

1. **"The data cannot arbitrate" between `tau_subject` and `kappa` (§18) is wrong.** They are separable on VG11's design; the recovery check above demonstrates it in both directions. §18's open item 8 — whether VG11 should carry subject random effects at 1.32 administrations per child — loses its main supporting argument, though the question of whether the effects _earn their place_ is untouched.
2. **"VG11 should not be quoted from this fit" (§18) was right for the wrong reason.** Its dispersion posterior was accurate; the prior conflict was the defect. The convergence failures §18 also recorded remain a reason not to quote it.

The sensitivity registry needed updating with them, and restating the variants exposed what two of them had actually been asking. `kappa-flat` set `a_kappa_mu` from `log 8` to 0 — an eight-fold cut in the _level_ of the age term, not a flattening of it, despite the name — and `kappa-const` pinned `b_kappa_mag` near zero, which is the genuinely constant-in-age one. Both are now stated at the anchors: `kappa-flat` divides both anchor medians by eight, and `kappa-const` sets them equal, since under the anchored form the slope is derived and cannot be set directly. `replace_kappa` now validates against whichever form the block uses and names it in the error, so the next migration breaks a stale variant loudly instead of silently.

### Refits

All three at `test` (4 chains x 2,000 draws), on the frames and code described above.

**Every dispersion parameter is now centred.** Prior CDF at the posterior mean, against 1.000 for VG11's two anchors under §18's prior:

| model | parameter              | post mean | prior median | prior CDF | contraction |
| ----- | ---------------------- | --------: | -----------: | --------: | ----------: |
| VG11  | `kappa_min`            |      5.83 |          6.0 |     0.485 |       0.876 |
| VG11  | `kappa_excess_young`   |    311.35 |        311.0 |     0.501 |       0.943 |
| VG11  | `kappa_excess_old`     |     44.39 |         44.0 |     0.505 |       0.945 |
| VG12  | `kappa_min`            |      3.59 |          3.0 |     0.589 |       0.261 |
| VG12  | `kappa_excess_young`   |     38.87 |         40.0 |     0.487 |       0.942 |
| VG12  | `kappa_excess_old`     |     62.19 |         63.0 |     0.494 |       0.953 |
| VG13  | `kappa_min_u`          |     33.88 |         30.0 |     0.580 |       0.777 |
| VG13  | `kappa_excess_young_u` |      8.13 |         10.0 |     0.409 |       0.695 |
| VG13  | `kappa_excess_old_u`   |     84.63 |         90.0 |     0.473 |       0.894 |
| VG13  | `kappa_min_s`          |      3.94 |          3.0 |     0.633 |       0.258 |
| VG13  | `kappa_excess_young_s` |     31.34 |         33.0 |     0.471 |       0.867 |
| VG13  | `kappa_excess_old_s`   |     24.98 |         27.0 |     0.456 |       0.864 |

Nothing outside 0.41-0.63. The two `kappa_min` rows with low contraction (0.26) are the two outcomes whose conditional fit put no mass on a floor: the data have little to say about a parameter that never binds inside the frame, and the prior is correspondingly left to carry it.

**Posterior `kappa` at the anchor ages matches the independent conditional estimate:**

| outcome         | posterior at anchors [94% CI]          | conditional estimate | before         |
| --------------- | -------------------------------------- | -------------------- | -------------- |
| VG11 spoken     | 315.8 [288, 347], 50.2 [47.6, 53.1]    | 317.5, 50.5          | 310, 50.0      |
| VG12 understood | 42.3 [38.3, 47.1], 65.7 [59.6, 71.8]   | 43.0, 66.4           | **16.5, 15.6** |
| VG13 understood | 42.0 [37.8, 46.4], 116.6 [91.8, 145.6] | 42.2, 124.1          | **15.9, 14.4** |
| VG13 `q`        | 35.0 [29.8, 40.3], 28.8 [24.7, 33.2]   | 36.0, 29.7           | 40.4, 29.7     |

VG12 and VG13's understood dispersion moves by a factor of 2.6. VG11's and VG13's `q` do not move, as predicted.

**The freed sign is what did it, and the posteriors prove the constraint was binding.** `b_kappa` is a derived quantity under the anchored form, so it can be read directly against the region the legacy form allowed:

| outcome          | `b_kappa` mean | 89% ETI          | legacy form admitted |
| ---------------- | -------------: | ---------------- | -------------------- |
| VG11             |         -1.432 | [-1.531, -1.333] | yes                  |
| VG12             |     **+0.211** | [+0.143, +0.279] | **no**               |
| VG13 `b_kappa_u` |     **+1.397** | [+0.782, +2.083] | **no**               |
| VG13 `b_kappa_s` |         -0.126 | [-0.277, +0.019] | yes                  |

Both comprehension outcomes put their entire 89% interval in `b_kappa > 0`, which `b_kappa = -b_kappa_mag <= 0` made unreachable at any prior setting. That is the cleanest statement of what was wrong: not a mis-tuned number but a parameterisation that excluded the answer.

**A caution about what this does and does not confirm.** The priors are centred on the conditional estimate from the same rows the models fit, so posterior agreeing with prior agreeing with estimate is not independent corroboration of the value — it is confirmation that the prior no longer fights the data and that the model can now reach what the data say. The evidence that the data rather than the prior are placing these posteriors is the contraction (0.94-0.95 on the anchors, so the posterior is far narrower than a prior spanning roughly [10, 190] at 5-95%) together with the `b_kappa` sign, which no prior here forces.

**None of the three passes the convergence gate**, and that is unchanged by this work rather than caused by it:

| model | divergences | max R-hat | min ESS | failing parameters                  | sampling |
| ----- | ----------: | --------: | ------: | ----------------------------------- | -------- |
| VG11  |          30 |     1.035 |     174 | `tau`, `ell_unit`, `ell`, `eta`     | 30m 38s  |
| VG12  |          11 |     1.017 |     359 | `tau`                               | 9m 22s   |
| VG13  |          80 |     1.012 |     266 | `tau_u`, `kappa_old_u`, `b_kappa_u` | 15m 45s  |

The failures are concentrated in the study random-effect scale and the GP hyperparameters, not the dispersion block: every `kappa` parameter has bulk ESS between 398 and 1,357 with R-hat at most 1.009. VG11's profile is the same one §18 recorded before this change. **These fits still should not be quoted for reporting**, but the dispersion estimates in them are well mixed and are what the tables above rest on.

One thing worth flagging that this section does not fix: VG11's `tau_subject` is 1.060 with a posterior standard deviation of 0.009, and VG13's `tau_subj_q` is 1.117 at prior CDF 0.975 and contraction 0.932. The subject random-effect scales remain pinned against a `HalfNormal(0.5)` prior in both, which is a separate mis-calibration from the one this section corrects and is now the loudest remaining prior-data conflict in these models.

### Open list after §19

Resolved by this section: §18's items 1 (the conditional calibration), 2 for VG12 (VG02 and VG04 remain, and both still need their own calibration — neither carries random effects, so the _marginal_ estimator is the right one for them), the VG12 and VG13 parts of item 3, and item 8's supporting argument.

Still open, in rough priority order:

1. **Why typically-developing understood `kappa` jumps at 16–18 months.** The per-age profile goes 21.6, 23.4, 183.2 at 14, 15, 16 months on 504, 322 and 917 observations, and the 16-month cell is the largest in the frame. Candidates not yet tested: a CDI form boundary (Words & Gestures runs to 18 months), one dataset dominating that cell, or a genuine developmental feature. Until it is understood, both understood anchors are deliberately wide and VG12's and VG13's dispersion should be read as a level rather than a trend.
2. ~~**Extend the two-anchor form to VG02 and VG04**~~ — done in §20.
3. **Dispersion for VG09, VG10, VG16 and VG15 remains uncalibrated and probably too low.** Every pool that _could_ be measured came out 3–10x its marginal counterpart, and these four carry marginally-derived defaults. The frame cannot support the estimator as written; what might work is pooling the Down syndrome joint rows with VG01's and VG02's single-outcome rows to buy replication, or a hierarchical fit that borrows the dispersion curve's shape across populations. Neither is attempted here.
4. **Refit VG04** on the post-sort frame (§17's item 2, still outstanding), and **VG16** on current code.
5. **Rerun `scripts/prior_predictive_audit.py`** for VG01, VG03, VG11, VG12 and VG13. Its `kappa` rows are now stale three times over.
6. **Test `eta_q = 0.4` on VG10** (§5). Untouched by any of this.
7. **Consider tightening `kappa_min_sigma`** (§18's item 6). Now more delicate than it looked: on a rising curve `kappa_min` is the young-age asymptote and carries real weight, so a single family-wide setting is no longer obviously right.
8. **Pair a `dev`-config control against VG03's anchored fit** (§18's item 7) to settle the per-draw cost of the parameterisation.
9. **Whether VG11 should carry subject random effects at all.** §19 removes the argument that `tau_subject` and `kappa` are inseparable, so this reverts to the ordinary question of whether the effect earns its place at 1.32 administrations per child — worth answering by model comparison rather than by inspection.

## 20. Implemented: the comprehension models VG02 and VG04

§19's open item 2, and the smallest of these sections: VG02 and VG04 carry no random effects, so the estimator needs neither a study nor a subject term and there is nothing to separate. What made the Down syndrome joint frame unusable in §19 was the difficulty of telling `tau` from `kappa` on few rows; with `tau` absent by construction that difficulty does not arise, and VG02's thin frame is workable where the joint one was not.

`scripts/kappa_conditional_calibration.py` gained a `Pool.study_effects` / `Pool.subject_effects` declaration for this. The estimator now mirrors whatever the model carries rather than always fitting the conditional form — which is the whole lesson of §19 stated as code, and it fixed a real defect in the process: `--mean-sweep` had been fitting a conditional specification regardless of the pool, so a no-random-effects model was being tested for the stability of a fit it never uses.

### Results

Marginal fits, saturated mean, no grouping — the specification these two models actually have:

| pool                 |     n | cells | anchors | `kappa` at anchors | per-age cell range |
| -------------------- | ----: | ----: | ------- | ------------------ | ------------------ |
| VG02 understood (DS) |   346 |    15 | 18, 36  | **15.4**, **7.1**  | 3.6-35.9           |
| VG04 understood (TD) | 1,538 |    16 | 12, 18  | **11.8**, **11.3** | 5.8-15.6           |

Both are stable across every mean model tried — VG02 gives 14.8-15.4 and 7.1-7.2, VG04 11.6-11.8 and 11.1-11.4, from a 6-parameter spline through to saturated. The Down syndrome frame's thinness shows up instead in `kappa_min`, which ranges over 0.76-6.01 depending on the mean model while the anchor totals move by under 4%: the floor and the excess trade off along a ridge and only their sum at the anchors is identified. That is precisely the ridge the two-anchor form exists to sidestep, and it is why both blocks keep the shared weak `LogNormal(log 3, 0.8)` floor and let the anchors carry the level.

### Two things worth recording

**Typically-developing comprehension dispersion is flat, and the legacy form could not say so.** 11.8 at 12 months against 11.3 at 18, with no trend across the whole 8-24 month range. The two anchors are consequently near-equal, they sit only six months apart — there is no decay to span, so they go where the data are densest (n = 115 and 128) — and the implied slope prior comes out near-symmetric about zero, P(`kappa` rising) 0.476 against 0.007 for Down syndrome spoken. §18's open item 2 predicted exactly this shape. Under `b_kappa_mag >= 0` a flat curve was reachable only by pushing the slope to the boundary, which is what VG12's and VG13's posteriors were seen doing in §19.

**VG04 against VG12 is the cleanest available demonstration that the marginal/conditional distinction is real.** They are the same outcome and the same population, differing in whether the model carries random effects:

| frame          | fitted marginally | fitted conditionally | the model's own specification |
| -------------- | ----------------: | -------------------: | ----------------------------- |
| VG04 (n=1,538) |          **11.8** |                 42.8 | marginal -> prior at 11.8     |
| VG12 (n=5,997) |              11.0 |             **43.0** | conditional -> prior at 43.0  |

Two different frames and two different estimators agree to within 7% on both readings. The factor of nearly four between the two priors is not a disagreement about the data; it is the difference between two quantities, and which one a model needs is settled by its structure rather than by anything empirical. §19 argued this from VG11's failure; here it falls out of a controlled comparison.

### Anchors and spread

VG02 takes `sigma = 0.8` rather than the spoken blocks' 0.7, because its cells hold 15-35 observations each and the per-cell estimates scatter 3.6-16.3 around the anchors. VG04 keeps 0.7. Neither floor is identified, so both keep the shared default. Prior medians are tuned so the _simulated median total_ at each anchor matches the fitted value rather than summing the component medians, which overshoots by about 10%.

### Refits

Both at `test` (4 chains x 2,000 draws), and both are far better behaved than the random-effect models of §19 — 3 divergences each, max R-hat 1.004 and 1.003, min ESS 1,189 and 1,626. Neither passes the convergence gate, which requires no divergences, but nothing here is near a diagnostic failure.

| model | parameter            | post mean | prior median | prior CDF | contraction |
| ----- | -------------------- | --------: | -----------: | --------: | ----------: |
| VG02  | `kappa_min`          |      1.34 |          3.0 |     0.157 |       0.870 |
| VG02  | `kappa_excess_young` |     13.16 |         11.0 |     0.589 |       0.907 |
| VG02  | `kappa_excess_old`   |      5.55 |          3.2 |     0.754 |       0.820 |
| VG04  | `kappa_min`          |      4.12 |          3.0 |     0.654 |       0.463 |
| VG04  | `kappa_excess_young` |      7.30 |          7.6 |     0.477 |       0.721 |
| VG04  | `kappa_excess_old`   |      7.07 |          7.2 |     0.490 |       0.705 |

Posterior `kappa` at the anchors: VG02 14.4 [12.3, 16.4] and 6.9 [6.2, 7.6] against the calibration's 15.4 and 7.1; VG04 11.4 [10.5, 12.3] and 11.2 [10.3, 12.1] against 11.8 and 11.3.

**VG02 is a live demonstration of the ridge.** Its posterior pulls the floor down to 1.34 (prior CDF 0.157, the least-centred parameter in either fit) and pushes both excesses up — 13.2 and 5.6 against priors at 11.0 and 3.2 — while the _totals_ at the anchors stay within 7% of the calibration. The decomposition moves; the identified quantity does not. Under the legacy form this would have shown up as a prior-data conflict in `a_kappa` with no way to tell it apart from a genuine disagreement about the level.

**VG04's `b_kappa` comes back at −0.022, 89% interval [−0.143, +0.086].** It straddles zero: the data say flat and the posterior is now able to say so. Under `b_kappa = -b_kappa_mag <= 0` that interval is unrepresentable, and the posterior would have been pressed against the boundary exactly as VG12's and VG13's were in §19. `kappa_min`'s contraction of 0.46 is the expected consequence of a flat curve — with no decay, the floor and the level are only weakly distinguishable — and is why it keeps a deliberately weak prior.

### Changed in the repository

- Two new blocks in `definitions.py`, `_DS_UNDERSTOOD_KAPPA` and `_TD_UNDERSTOOD_KAPPA`, applied to VG02 and VG04. Every univariate model is now on the two-anchor form.
- `Pool` in the calibration script declares its model's grouping; `run_pool` and `run_mean_sweep` both honour it, and a pool with no subject effect reports the marginal fit as the answer rather than as a contrast.
- The migration guard in `tests/test_model_definitions.py` records the two additions. Two tests in `tests/test_kappa_parameterisation.py` had used VG02 as their legacy-form exemplar and now build a synthetic definition instead — with no univariate model left on the legacy form, pinning them to the registry made them break on exactly the change they were meant to allow.

### Open list after §20

> [!NOTE]
> Items 1, 2, 3 and 9 below are resolved by §§21–23. The current list is **Open list after §23**, at the end of this note.

Resolved by this section: §19's item 2. Every univariate model (VG01–VG04, VG11, VG12) now carries an empirically calibrated two-anchor dispersion prior, each matched to whether its own model has grouping structure.

Still open, in rough priority order — unchanged from §19 except for the numbering:

1. **Why typically-developing understood `kappa` jumps at 16–18 months** (§19). Now slightly sharper: VG04's marginal per-cell profile over the same ages is flat at 5.8–15.6 with no such spike, so whatever produces it appears only once subject effects are in the model. That points at the composition of the repeat-measured children rather than at the age cells themselves, and is worth a look before either understood prior is narrowed.
2. **Dispersion for VG09, VG10, VG16 and VG15 remains uncalibrated and probably too low** (§19). Unchanged, and now the only outstanding dispersion work: the DS joint frame cannot support the conditional estimator, and pooling its rows with VG01's and VG02's single-outcome rows to buy replication is the obvious thing to try next.
3. **Refit VG04** on the post-sort frame — now done as part of §20, so this drops to **refit VG16** on current code.
4. **Rerun `scripts/prior_predictive_audit.py`** for every migrated model. Its `kappa` rows are stale for all six.
5. **Test `eta_q = 0.4` on VG10** (§5). Untouched by any of this.
6. **Consider tightening `kappa_min_sigma`** (§18's item 6). §20 argues against a single family-wide setting more strongly: VG04's floor has contraction 0.46 and VG02's posterior pulls it to 1.34, so on the comprehension models it is doing real work that a tighter prior would suppress.
7. **Pair a `dev`-config control against VG03's anchored fit** (§18's item 7) to settle the per-draw cost of the parameterisation.
8. **Whether VG11 should carry subject random effects at all** (§19).
9. **The subject random-effect scales.** VG13's `tau_subj_q` sits at prior CDF 0.975 and VG11's `tau_subject` is pinned at 1.060 ± 0.009 against a `HalfNormal(0.5)`. With dispersion now calibrated everywhere it can be, this is the largest remaining prior-data conflict in the family and deserves the same treatment `kappa` has just had.

## 21. Diagnosed: the 16–18 month understood spike is a form-scale artefact

§20's open item 1. The answer is not in the age cells, and not in the children: **counts collected on a 396-item form are scored out of 810, so the modelled proportion compresses as children work up the form, the apparent between-child spread falls with age, and a constant subject scale cannot follow it. `kappa(age)` is the only age-varying spread parameter left, so it absorbs the difference.** The 16-month cell is where the observed spread first drops below the fitted `tau`, and `kappa` there has nothing left to explain.

### Reproducing it

A free `kappa` per age cell on VG13's understood frame, random effects present, reproduces §19's profile:

| age (months) |    8 |    9 |   10 |   11 |   12 |   13 |   14 |   15 |    16 |   17 |    18 |
| ------------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ----: | ---: | ----: |
| `kappa`      | 23.5 | 32.2 | 29.0 | 27.8 | 44.9 | 50.6 | 21.8 | 23.2 | 186.8 | 75.0 | 141.1 |
| n            |  252 |  289 |  261 |  321 |  586 | 1036 |  504 |  322 |   917 |  482 |   436 |

### Four candidates, all eliminated

**One study dominating the 16-month cell.** True but not sufficient. Thal contributes 634 of the 917 rows (69%), and it is a two-wave longitudinal study: the _same_ 634 children at 13 and at 16 months, nothing in between. Its within-cell spread is the lowest in the frame — logit sd 0.617 against Marchman's 0.831 and Floccia's 1.017 — so it does pull the cell down. But Marchman's own spread collapses over the same boundary, 1.433 at 15 months to 0.831 at 16, on rows Thal has no part in.

**Hard censoring at the form ceiling.** Ruled out on magnitude. Only 0.43% of the 16-month cell sits at the WG ceiling of 396 and 2.0% within 5% of it. Whatever compresses the distribution is not clipping it.

**Selection into the WG form.** Ruled out directly. From 16 months the WS form becomes available and only WG and Oxford CDI rows carry comprehension, so a study that streamed its abler children onto WS would leave a truncated WG sample behind. Thal did not stream: it gave _both_ forms to the same 644 children at 16 months, so its 634 WG rows are the whole cohort. Marchman does split children between forms (165 WG and 183 WS at 16 months, disjoint), but the WS children are barely ahead — median production 28 against 21 — and Marchman is 18% of the cell.

**Between-study heterogeneity in the subject scale.** Real, and much too small. Fitting one `tau` per study costs three extra parameters and buys 34.8 log-likelihood units; the age term below costs one and buys 115.2.

### What it is

Let the subject effect keep one scalar per child but give it an age-varying loading, `logit p_ij = m_c + s_k + lambda(a_ij) b_i` with `b_i ~ N(0, 1)` and `lambda` on the same two-anchor form as `kappa`. The quadrature is unchanged; the model gains one parameter.

| model on VG13 understood      | parameters | nll       | vs constant `tau` |
| ----------------------------- | ---------: | --------- | ----------------: |
| no subject effect             |         25 | 30,188.50 |            −405.3 |
| constant `tau` (§19's form)   |         26 | 29,783.17 |                 — |
| **age-varying `lambda`**      |     **27** | 29,668.01 |        **+115.2** |
| one `tau` per study           |         29 | 29,748.40 |             +34.8 |
| per-study `tau` + age loading |         30 | 29,648.32 |            +134.9 |

One parameter buys 115 units. Adding per-study scales on top of it buys a further 19.7 for three more, and the age slope is unchanged when they are present, so the two are not competing for the same variance. Under the loading model the fitted `lambda` falls from 0.906 at 12 months to 0.576 at 17, and the `kappa` profile changes character completely — 177.6 at 8 months down to 44.9 at 18, with the 16-month cell at 99.1 rather than 186.8. **It falls with age instead of rising.**

The same contrast on the two-anchor `kappa` form the models actually parameterise, which is what `--loading` reports:

| pool                     | constant `tau` (as modelled)      | with an age-varying loading                         |       gap |
| ------------------------ | --------------------------------- | --------------------------------------------------- | --------: |
| VG11 spoken (12, 20)     | `tau` 1.056; `kappa` 317.5 → 50.5 | `lambda` 1.304 → 1.028 (−21%); `kappa` 516.9 → 48.0 | **237.1** |
| VG12 understood (12, 20) | `tau` 0.736; `kappa` 43.0 → 66.4  | `lambda` 0.933 → 0.525 (−44%); `kappa` 84.6 → 53.0  | **162.3** |
| VG13 understood (12, 17) | `tau` 0.770; `kappa` 42.2 → 124.1 | `lambda` 0.941 → 0.622 (−34%); `kappa` 79.0 → 63.5  | **111.3** |
| VG13 `q` (12, 17)        | `tau` 1.119; `kappa` 36.0 → 29.7  | `lambda` 0.981 → 1.216 (+24%); `kappa` 22.7 → 53.4  |       0.8 |

Both understood outcomes reverse: 43.0 → 66.4 becomes 84.6 → 53.0, and 42.2 → 124.1 becomes 79.0 → 63.5. `q` alone shows no gap, and it is the outcome whose denominator cancels the form.

### The estimator can tell the two apart

Simulating on VG13's real subject, study and age structure, two seeds each:

| truth                                       | fitted `lambda`               | fitted `kappa`(16) | loading buys |
| ------------------------------------------- | ----------------------------- | -----------------: | -----------: |
| constant `tau` = 0.721, `kappa`(16) = 186.8 | 0.740 → 0.688 / 0.743 → 0.711 |      135.0 / 170.7 |    1.0 / 0.6 |
| `lambda` 0.906 → 0.526, `kappa`(16) = 99.1  | 0.890 → 0.549 / 0.904 → 0.549 |       131.9 / 95.6 |  86.3 / 80.9 |

When the truth is a constant scale the loading model correctly returns a flat one and buys nothing — a 1-degree-of-freedom null behaving like one. When the truth is a falling loading, the constant-`tau` model invents a 16-month spike of **484.0 and 273.5** against a truth of 99.1, and the loading model recovers both the scale and the profile. The real data behave like the second row.

### Why the scale falls: it is the denominator

The obvious explanation — the logit link exaggerating spread at small `p` — is wrong, and VG13's own `q` outcome is what refutes it. `q` is measured on the same children, in the same design, with almost the same mean profile as understood (0.115 → 0.263 against 0.104 → 0.232 across 12–17 months), and it shows **no loading effect at all**: with a free `kappa` per cell, `lambda` 1.084 → 1.103 for 0.03 log-likelihood units, and on the two-anchor form 0.8 units. Both are a 1-degree-of-freedom extension paying for nothing, on the same rows where understood pays 111. Nor is it the young low-`p` cells: restricted to 12 months and up, VG13 still gives `lambda` 0.960 → 0.560 for 71.1 units and VG12 0.905 → 0.509 for 89.7.

What distinguishes `q` is its denominator. `q` is spoken out of that child's _own observed understood count_, so both sides come from the same form and the form's extent cancels. `understood` is scored out of 810 while the instrument holds 396 items (WG) or 418 (Oxford CDI) — and they are the _easiest_ 396. Two children who differ by hundreds of words on a full inventory can differ by a handful on a form they have nearly exhausted, and by 16–18 months the mean row sits at 0.48–0.53 of its form with 46–55% of children past halfway. The measure compresses, progressively, with age.

Rescoring the identical rows out of each row's own form instead of 810 is the test, and it is decisive:

| pool            | out of 810                 | out of the row's own form | removed |
| --------------- | -------------------------- | ------------------------- | ------: |
| VG12 understood | 141.0 units, `lambda` −44% | 6.1 units, `lambda` −11%  | **96%** |
| VG13 understood | 115.2 units, `lambda` −36% | 15.8 units, `lambda` −15% | **86%** |
| VG11 spoken     | 199.9 units, `lambda` −22% | 32.1 units, `lambda` −11% | **84%** |
| VG13 `q`        | 0.0 units, `lambda` +2%    | — (shares its form)       |       — |

The ordering is the one the mechanism predicts. Comprehension is worst affected because WG's 396 comprehension items are the tightest constraint in the export; spoken production is milder because WS carries 680 and the mean spoken row is only 0.26 of its form; and the ratio outcome, which cancels the form, shows nothing.

### What this does and does not change

> [!IMPORTANT]
> **No prior changes.** §19's rule is that the calibration must mirror the model's own structure, and the registered models carry a constant `tau_subject`. Their `kappa` therefore has to absorb this, and a prior calibrated under the constant-`tau` estimator is the right prior for them. The refits in §19 and §20 stand.

What changes is the **interpretation**, and the docs state it too strongly in three places. "Typically-developing comprehension dispersion rises with age" is true of the model's `kappa` parameter and false of typically-developing comprehension: on the instrument's own scale the dispersion falls, like every other outcome. `kappa` on the understood outcomes is a compound of observation-level dispersion and a subject scale the model cannot vary, and it should not be read as a statement about children. Corrected in [`docs/models/PRIORS.md`](../docs/models/PRIORS.md) and the VG12 and VG13 model pages.

The 810-item reference scale itself is not in question. It is a deliberate harmonisation choice, documented in PRIORS.md's "Instrument scale" section, endorsed by Laudańska et al. (2026), and load-bearing for every cross-population comparison the project makes. What is new here is a consequence of it that had not been traced: it makes the between-child scale age-dependent on the modelled scale, and a constant-`tau` model routes that into its dispersion.

### Changed in the repository

- **`scripts/kappa_conditional_calibration.py`** gained `--loading`, which fits the age-varying loading alongside the constant-`tau` form and reports the likelihood-ratio gap. It is a diagnostic, not a calibration path: a large gap means the pool's `kappa` is carrying subject-scale drift and should not be read as dispersion.
- **`tests/test_kappa_conditional_calibration.py`** covers both directions of the control above — a constant-`tau` truth must return a flat loading, and a falling-loading truth must not be recoverable by the constant-`tau` form.

## 22. Implemented: the Down syndrome joint frame, calibrated as a lower bound

§20's open item 2, and a reversal of §19's verdict on it. The suggestion §19 left — pool the joint rows with VG01's and VG02's single-outcome rows to buy replication — **cannot be done**, and finding out why is what forced the rest of this section.

### There are no rows to pool

Every Down syndrome model calls `load_data` with no `sample_fraction`, no `min_study_observations`, no age bound and no exclusions. They all get the same 1,218 rows, of which 671 carry comprehension, 1,114 production and 670 both. **VG02's understood frame and VG09's are not two frames; they are the same 671 rows** — VG02's appears smaller in §20's table only because the saturated mean's 15-observation cell rule drops 325 of them.

So "pool VG01 and VG02 in" resolves to "relax filters that are already absent". The joint frame is not a subset of a larger Down syndrome pool; it is the pool. Whatever is done here has to be done on 671 rows.

### No configuration recovers `kappa`, and 36 per outcome were tried

The remaining lever is to spend fewer parameters. Sweeping spline flexibility (4, 5, 6, 8 knots) × age window (8–115, 10–60, 12–48 months) × anchor pair ((24, 48), (18, 36), (20, 40)), scoring each by how well two known truths come back on two seeds — 36 configurations per outcome, 288 fits in all: **none recovers everything to within 30%.** The best understood configuration is off by 39% at the young anchor and the best ratio configuration by 61%.

`tau` is the exception, and a large one: it recovers to within 6% on the understood frame at the configuration the estimate uses, and under 14% at every setting tried. That result is what §23 rests on.

### But the failure is bias, not scatter — which makes it usable

The grid varied `tau` along with `kappa`, conflating two questions. Holding `tau` at its fitted value and varying only the truth, five seeds each at the configuration the estimate uses:

| pool              | truth `kappa`(24) | recovered (median) | bias | seed range |
| ----------------- | ----------------: | -----------------: | ---: | ---------- |
| VG09/10/15/16 U   |              12.0 |               11.7 |  −2% | 10.4–13.3  |
| VG09/10/15/16 U   |              40.8 |               39.0 |  −4% | 34.9–42.5  |
| VG09/10/15/16 U   |              81.6 |               60.2 | −26% | 54.6–68.9  |
| VG09/10/15/16 U   |             163.2 |              104.1 | −36% | 94.4–108.6 |
| VG09/10/15/16 `q` |               6.9 |                5.5 | −20% | 4.7–8.4    |
| VG09/10/15/16 `q` |              13.8 |               11.5 | −17% | 8.4–11.9   |
| VG09/10/15/16 `q` |              27.6 |               15.9 | −42% | 13.3–20.5  |
| VG09/10/15/16 `q` |             100.0 |               33.4 | −67% | 26.1–42.9  |

The error is **one-directional and monotone in the level**, and the seed ranges are tight around each median. That is not an estimator returning noise; it is one walking down a flat ridge. A large `kappa` is near-binomial, the data stop distinguishing bigger from biggest, and the optimum slides toward where the likelihood still has curvature. §19 read the two-sided spread of a `tau`-and-`kappa` grid and concluded "no number read off it would mean anything". The correct reading is narrower and more useful: **the estimates are lower bounds.**

The understood pool is also more stable than §19's single mean sweep suggested. Across spline knot counts 6, 8, 10 and 14 it gives `kappa`(24) of 81.3, 81.6, 81.9 and 81.5 — a spread of 0.7%. It is the ratio outcome that moves, 13.0 to 23.6 across the same sweep. Stability and recoverability are different properties and this frame has one without the other.

### The prior was wrong regardless, and demonstrably so

None of the above is needed to establish that the legacy block does not fit these models. Prior CDF at the posterior mean, `dev` fits, across every Down syndrome joint model:

| model | `b_kappa_mag_u` | prior CDF | contraction | ESS   |     | `b_kappa_mag_s` | prior CDF | contraction |
| ----- | --------------: | --------: | ----------: | ----- | --- | --------------: | --------: | ----------: |
| VG05  |           0.825 |     0.994 |        0.38 | 999   |     |           0.855 |     0.996 |       −0.22 |
| VG07  |           0.813 |     0.993 |        0.40 | 1,031 |     |           0.906 |     0.998 |       −0.18 |
| VG08  |           1.100 |    0.9998 |        0.24 | 318   |     |           0.746 |     0.987 |       −0.31 |
| VG09  |           1.141 |    0.9999 |        0.27 | 313   |     |           0.309 |     0.697 |        0.07 |
| VG10  |           1.157 |    0.9999 |        0.29 | 381   |     |           0.304 |     0.689 |        0.09 |
| VG14  |           0.807 |     0.993 |        0.41 | 951   |     |           0.856 |     0.996 |       −0.26 |
| VG15  |           1.154 |    0.9999 |        0.28 | 496   |     |           0.361 |     0.771 |       −0.07 |
| VG16  |           1.144 |    0.9999 |        0.29 | 3,028 |     |           0.311 |     0.700 |        0.11 |

Against `HalfNormal(0.3)`. Every understood slope is in the extreme upper tail, all eight well mixed, and five of the eight spoken slopes have **negative contraction** — the posterior wider than the prior, the likelihood pushing outward against a wall. This is §18's pathology, worse than VG01's 0.998 was, and it needs no view on the level to diagnose: the two-anchor form removes it by having no slope prior at all.

### What was changed

Two new blocks, `_DS_JOINT_UNDERSTOOD_KAPPA_RE` and `_DS_JOINT_Q_KAPPA_RE`, on **VG09, VG10, VG15 and VG16** — the four models carrying subject intercepts on _both_ outcomes, which is what makes them share one calibration target. Medians are each estimate divided by the bias measured at it: understood 81.6/0.74 = 110 at 24 months and 20.3/0.62 = 33 at 48; ratio 13.8/0.83 = 17 and 7.6/0.70 = 11. `sigma` is **1.0** on all four anchors, wider than anywhere else in the family, because the correction is itself uncertain and the ratio's mean sensitivity is a factor of 1.8.

**VG15 needed the joint-modality engine extending first.** §19 gave the two bivariate engines a per-outcome suffix; `common_joint_modality.py` still called `build_kappa_of_z` directly with a hard-coded triple. It now routes through the same shared helpers with suffixes `_u`, `_s` and `_sign`, and `JointModelConfiguration` gained the three `kappa_anchored_*` fields and the shared validator. VG15 is consequently **the only registered model mixing the two forms** — anchored on understood and spoken, legacy on the signed ratio, which has no calibration — and a test now pins that.

**VG05, VG07, VG08 and VG14 are deliberately not migrated**, and for a reason the §19 rule supplies: the calibration must match the specification, and theirs differ. VG05 carries no random effects, VG07 only study ones, and VG08 a subject effect on understood but not on `q` — so VG08 would need one calibration per outcome. All three are steps in the VG05 → VG07 → VG08 → VG09 → VG10 lineage, whose whole purpose is to isolate what adding each random effect does; changing a prior partway along would confound exactly that contrast. VG14's frame is the signing subset.

### Refits

All four at `test` (4 chains x 2,000 draws). Only VG16's is a like-for-like prior comparison — VG09, VG10 and VG15 had only ever been fitted at `dev`, so their improvement confounds the prior change with the extra draws — but VG16's is clean, and it improves on every count:

| model | divergences | max R-hat | min ESS |     | before (VG16 only) |
| ----- | ----------: | --------: | ------: | --- | ------------------ |
| VG09  |           0 |     1.011 |     313 |     | —                  |
| VG10  |           1 |     1.007 |     527 |     | —                  |
| VG15  |           0 |     1.023 |     442 |     | —                  |
| VG16  |           0 |     1.007 |     478 |     | 1, 1.013, 331      |

None passes the convergence gate, which requires zero divergences _and_ R-hat below 1.01, but VG09, VG15 and VG16 now have no divergences at all and VG15 — previously the worst-behaved model in the family at max R-hat 1.137 and min ESS 15 — comes back at 1.023 and 442 with its headline association `psi` at 1.92 [1.37, 2.59] on 3,635 effective samples.

**The four agree with each other, as four models sharing a frame should:**

| model | `kappa`(24) U | `kappa`(48) U | `kappa`(24) `q` | `kappa`(48) `q` |
| ----- | ------------: | ------------: | --------------: | --------------: |
| VG09  |          77.9 |          17.3 |            23.1 |            16.1 |
| VG10  |          78.2 |          17.3 |            23.0 |            16.0 |
| VG15  |          78.1 |          17.5 |            21.2 |            12.1 |
| VG16  |          78.3 |          17.4 |            23.7 |            16.7 |

**And they land where the lower-bound reading predicted, in both directions.** The understood posteriors sit at 78 against a prior median of 110 and the uncorrected estimate of 81.6 — so the bias correction was not needed there, and at `sigma = 1.0` it cost nothing (prior CDF 0.37 and 0.26). The ratio posteriors sit at 23 and 16 against a prior median of 17 and 11 and an estimate of 13.8 and 7.6 — _above_ both, which is the direction a lower bound is supposed to be wrong in. Recording this plainly because it is the useful part: the correction was applied a priori from the recovery simulation, it over-corrected one outcome and under-corrected the other, and a prior wide enough to be honest about the frame absorbed both.

Prior CDF at the posterior mean, VG09 (the others are within 0.01):

| parameter              | post mean | prior median | prior CDF | contraction |
| ---------------------- | --------: | -----------: | --------: | ----------: |
| `kappa_min_u`          |      2.03 |          3.0 |     0.312 |       0.753 |
| `kappa_excess_young_u` |     75.88 |        106.0 |     0.369 |       0.958 |
| `kappa_excess_old_u`   |     15.26 |         28.7 |     0.264 |       0.962 |
| `kappa_min_s`          |      9.23 |          3.0 | **0.920** |  **−0.052** |
| `kappa_excess_young_s` |     13.87 |         12.6 |     0.538 |       0.840 |
| `kappa_excess_old_s`   |      6.85 |          6.7 |     0.509 |       0.680 |

`kappa_min_s` is the one parameter not centred, and it is §20's ridge again rather than a new problem: the floor and the excess trade off, only their sum at the anchors is identified, and that sum is tight (23.1, 89% interval [18.3, 28.9], 2,964 effective samples). VG02 did the same thing in the opposite direction. The negative contraction says the same — a posterior no narrower than its prior on a parameter the data cannot separately see.

**`b_kappa` is freed and the constraint is shown to have been binding.** Under the legacy form these models put `b_kappa_mag_u` at 1.14–1.16 against a `HalfNormal(0.3)`. Derived from the anchors instead:

| outcome          | `b_kappa` mean | 89% ETI        |
| ---------------- | -------------: | -------------- |
| VG09 `b_kappa_u` |         −1.395 | [−1.64, −1.15] |
| VG09 `b_kappa_s` |         −0.804 | [−1.86, −0.13] |
| VG15 `b_kappa_u` |         −1.385 | [−1.63, −1.15] |
| VG15 `b_kappa_s` |         −1.289 | [−2.28, −0.30] |

Both signs are negative, so unlike the comprehension models of §19 the legacy form's _direction_ was right here — it was the magnitude the prior would not allow. VG15's signed ratio, still on the legacy form, sits at `b_kappa_mag_sign` 0.153 [0.01, 0.41] and is not fighting anything, which is consistent with leaving it there.

## 23. Implemented: the subject random-effect scales

§20's open item 9, the last of the three and the simplest, because the measurement had already been made and nobody had read it. The conditional estimator reports `tau` alongside `kappa` for every pool — it has to, since separating them is the whole point of it — so the subject scales have had a calibration sitting in §19's output since §19 was written.

### Fourteen parameters, one prior, none of them centred

Every subject random-effect scale in the registry was `HalfNormal(0.5)`, median 0.34:

| model | parameter       | posterior | prior CDF | estimate |
| ----- | --------------- | --------: | --------: | -------: |
| VG08  | `tau_subj_u`    |     0.824 |     0.901 |    0.847 |
| VG09  | `tau_subj_u`    |     0.831 |     0.904 |    0.847 |
| VG09  | `tau_subj_q`    |     1.360 |     0.994 |    1.147 |
| VG10  | `tau_subj_u`    |     0.827 |     0.902 |    0.847 |
| VG10  | `tau_subj_q`    |     1.363 |     0.994 |    1.147 |
| VG11  | `tau_subject`   |     1.060 |     0.966 |    1.056 |
| VG12  | `tau_subject`   |     0.735 |     0.858 |    0.736 |
| VG13  | `tau_subj_u`    |     0.768 |     0.876 |    0.770 |
| VG13  | `tau_subj_q`    |     1.117 |     0.975 |    1.119 |
| VG15  | `tau_subj_u`    |     0.826 |     0.901 |    0.847 |
| VG15  | `tau_subj_q`    |     1.253 |     0.988 |    1.147 |
| VG15  | `tau_subj_sign` |     1.082 |     0.970 |        — |
| VG16  | `tau_subj_u`    |     0.831 |     0.904 |    0.847 |
| VG16  | `tau_subj_q`    |     1.362 |     0.994 |    1.147 |

Not one below 0.858. A prior that every parameter in the family sits in the upper tail of is not weakly informative; it is wrong in a consistent direction.

**The agreement in the last two columns is worth pausing on.** On all four typically-developing parameters the independent estimator and the model posterior agree to three significant figures — 1.056 against 1.060, 0.736 against 0.735, 0.770 against 0.768, 1.119 against 1.117 — and on the five Down syndrome understood ones to within 3%. Two different estimation machines, one a quadrature-integrated maximum-likelihood GLMM with a saturated mean and the other a Hamiltonian sampler with an HSGP, landing on the same number. That is the strongest validation of the conditional estimator anywhere in this note, and it arrives as a by-product.

The four that differ are all the Down syndrome ratio, estimate 1.147 against posteriors of 1.25 to 1.38, and §22 measured a bias on exactly that pool: a `tau` of 1.15 recovers there as 1.06, about 8% low, so 1.147 implies a truth near 1.24. That accounts for VG15's 1.253 exactly and about half the gap to the 1.36 the other three reach. The residual is the estimator's known weakness on its weakest pool rather than a conflict, and it is in the direction that makes the prior below _more_ comfortable, not less.

### One number covers the family

The estimates span 0.74 to 1.15, a factor of 1.6 — narrow enough that one prior serves. `HalfNormal(1.5)` has median 1.01 and 5–95% of 0.09 to 2.94:

|                                   | at the estimate | at the posterior |
| --------------------------------- | --------------- | ---------------- |
| prior CDF under `HalfNormal(0.5)` | 0.86–0.994      | 0.86–0.994       |
| prior CDF under `HalfNormal(1.5)` | **0.376–0.556** | **0.376–0.636**  |

Every parameter lands in the central half. The family stays HalfNormal rather than moving to the LogNormal the `kappa` anchors use, deliberately: a scale prior with mass at zero lets a subject effect the data do not support shrink away, and that property is worth keeping even on frames where the effect is overwhelming (§19's likelihood ratios against `tau = 0` run from 117 to 4,353 on one degree of freedom). Widening the scale fixes the conflict without giving it up.

`tau_subj_sign` has no calibration — nothing estimates a signing subject scale — so it inherits the family setting rather than being fitted to one. Its posterior at 1.082 was in the same tail as the rest, and definitions.py already recorded the conflict and declined to act on it ("kept at 0.5, porting VG10"); it is now resolved by the same change, at prior CDF 0.53.

**The study scales stay at `HalfNormal(0.5)`.** Their posteriors sit at prior CDF 0.43 to 0.82 across every model that has them, so there is nothing to fix, and the estimator treats study effects as fixed and therefore has no opinion to offer. That the subject scales and the study scales shared one default was the accident; only one of them was mis-set.
