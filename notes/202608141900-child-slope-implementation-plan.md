# Implementation plan: a child-level random slope (the fix Proposal A1 measures the need for)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> A plan, not a change. Nothing here is implemented, and **none of it should start while [#215](https://github.com/dseinternational/vocabulary-growth/issues/215) is running** — it adds a definition class and a model, and the run needs a stable tree. Gate 1 has already been executed and is recorded in [202608141600](202608141600-rank-stability-tracking.md) §10; it changed the design before any engine work began, which is the point of putting it first.

## 1. What is being fixed, in one paragraph

VG08–VG10 give each child a **constant** offset from the population trajectory. Three distinct quantities have to live in two parameters as a result: persistent between-child differences (`tau_subj`), occasion-to-occasion movement, and drift — a child systematically changing standing over months. The last two have nowhere of their own and land on `kappa`, which is why `kappa`'s fitted decline cannot be read developmentally. The fix is a child-level effect that varies with age.

## 2. Gate 1 — which structure, decided before writing any of it

[202608141600](202608141600-rank-stability-tracking.md) §10.3 fits three candidate structures to the adjusted residuals by maximum likelihood, against a known per-observation binomial sampling variance. Result, as `2 x delta logL` over the constant-intercept baseline:

| structure                              | spoken, all 767 | spoken, repeats 334 | understood, all 610 | understood, repeats 253 |
| -------------------------------------- | --------------: | ------------------: | ------------------: | ----------------------: |
| **+ random slope** (2 df)              |       **36.05** |           **20.81** |           **27.09** |                    0.82 |
| **cost of imposing `rho01 = 1`** (1 df) |          −1.49 |           **−6.28** |               −2.56 |                   −0.32 |
| **+ AR(1) transient**                  | `ell -> 0`, no gain |                 — | `ell -> 0`, no gain |                       — |

Three conclusions, and they set the whole design:

1. **A random slope, not an AR(1).** The mean-reverting alternative collapses to zero persistence on both outcomes — the within-child deviation has no memory beyond the occasion — so an autocorrelated child process is not what is missing.
2. **The production slope is genuine within-child drift**, not cross-sectional widening: it survives restriction to the 334 children with repeated spoken measures, where `tau1` *rises* to 0.0243 logit/month.
3. **Comprehension shows the opposite pattern**: worth 27.09 across all 610 children and **0.82** across the 253 with repeats. Its widening is cross-sectional. The model must be able to represent both cases and let the outcome decide, which the random slope does — `tau1 = 0` is the null and is inside it.

This is also the first direct test of Proposal A1's structure: A1 is the random slope with `rho01` pinned to 1 (one deviate scaled by an age function is a rank-one covariance). Freeing it costs 6.28 on 1 df on the repeats-only production fit (p ≈ 0.012). **A1's no-crossing assumption is rejected where there is power to test it.**

## 3. Model identity and where the fields live

**A new registered model, `VG19`.** The family's convention is that a structural refinement of a model of record gets a number — VG09 is VG08 plus subject intercepts, VG10 is VG09 plus the anchored GP — and this is the same kind of step. It leaves VG10 standing as the model of record while VG19 is developed and compared, and its role assignment goes to [#190](https://github.com/dseinternational/vocabulary-growth/issues/190) with the others. (VG17 and VG18 are taken by the exploratory sign-group modules.)

**The definition fields go on a subclass, not on `BivariateModelDefinition`.** A fit is validated by comparing `dataclasses.asdict` field for field, so **adding a field to a definition class invalidates every existing fit of that class** — here VG05, VG07–VG10 and VG16, six models, mid-refit. The precedent is exact and already documented in the codebase: `UnivariateREModelDefinition` exists for this reason, so that VG01–VG04 were not made stale by fields only VG11/VG12 use.

```python
@dataclass
class BivariateChildSlopeModelDefinition(BivariateModelDefinition):
    subject_slope_u: SubjectSlopePriorParams | None = None
    subject_slope_q: SubjectSlopePriorParams | None = None
    subject_slope_ref_age_months: float = 36.0
```

`None` on both means "behave exactly as VG10", so the subclass is safe to instantiate before the engine supports it. The engine reads them through `getattr(definition, ..., None)`, as it already does for the variance partition.

`subject_slope_ref_age_months` is the age at which `tau0` is the between-child spread. **36 months, the pool's median age** — centring at the design's centre is what keeps the intercept and slope from trading off in the sampler, and it makes `tau0` a spread where the data are dense rather than an extrapolated one.

## 4. The graph

```python
tau0     ~ HalfNormal(tau0_sigma)          # spread at the reference age
tau1     ~ HalfNormal(tau1_sigma)          # spread of rates, PER YEAR
rho_raw  ~ Beta(2, 2);  rho01 = 2*rho_raw - 1
z        ~ Normal(0, 1), dims (subject_id, 2)

L    = [[tau0, 0], [rho01*tau1, tau1*sqrt(1 - rho01**2)]]
b    = z @ L.T                              # (subject_id, 2), non-centred
shift(obs) = b[subject_obs, 0] + b[subject_obs, 1] * (age_obs - ref) / 12
```

Four notes on choices that are not arbitrary.

**The Cholesky is written out rather than drawn from `LKJCholeskyCov`.** For `n = 2`, LKJ(η) on the correlation is exactly `(rho + 1)/2 ~ Beta(η, η)`, so `Beta(2, 2)` *is* LKJ(2) here. Writing it explicitly keeps `tau0`, `tau1` and `rho01` as named free variables that the posterior summaries, the prior-vs-posterior check and the recovery scorer can each read directly, instead of a packed triangular vector they would all have to unpack.

**The slope is per year.** `tau1` in logit/month is 0.02-ish and a prior on that scale is unreadable; per year it is 0.12–0.29, and `HalfNormal(0.5)` has median 0.34, covers both fitted values comfortably, and keeps mass near zero so a slope the data do not support shrinks away.

**Both subject effects carry a slope.** The tracking analysis measures drift on the *spoken proportion*, and VG10 builds spoken as `p_u * q` — so the evidence cannot say whether the drift belongs to `f_u` or to `h`. Expect the two `tau1`s to be individually poorly identified and correlated, and report their joint implication for spoken rather than each alone.

**The seam already exists.** Proposal A1 built the observation-level, age-dependent subject shift in `common_bivariate_re` (commit `06d2502`); a random slope is a different age function through the same seam. `common_univariate_re` and `common_joint_modality` are untouched by this plan.

### Names to preserve

Downstream code reads scalar `tau_subj_u` / `tau_subj_q` — the posterior summaries, `comparison.py`'s heterogeneity contrast, `prior_vs_posterior.py`, the recovery scorer. Emit `tau_subj_u` as a `Deterministic` equal to `tau0` (the spread at the reference age), exactly as A1 does. Emit `b0_subj_u` and `b1_subj_u` as **separate one-dimensional** deterministics over `subject_id`: `recovery/compare.py` routes by dimension and only handles one-dimensional random-effect variables, so a `(subject_id, 2)` array would be silently skipped.

### What genuinely changes downstream

- **`comparison.py`'s child-spread figures read a scalar.** Under a slope the between-child spread is age-varying — `sqrt(tau0² + 2 rho01 tau0 tau1 D + tau1² D²)` — so `child_spread_product` and its callers need a grid rather than a number. This is the one non-trivial downstream change in the plan; everything else is name preservation.
- **The posterior-predictive unseen child gets better, not just different.** One `(b0, b1)` pair per posterior draw gives a coherent trajectory that fans with age, which is what a "what to expect for a child" figure should show. A1 generalised `sample_posterior_predictive` for the age-varying case already; this extends the same branch.

## 5. Gates

Each is a stop, not a checkpoint.

**G1 — structure selection. PASSED** (§2). Reproduced by `report_child_structure` in `scripts/experiments/rank_stability.py`.

**G2 — prior predictive.** A linear-in-age slope on the logit scale over an 8–115 month domain is a strong extrapolation, and this is where it would show. At `+1 SD` (0.34/year) a child at 115 months sits about 2.2 logits above the population trajectory before `tau0` is counted; against a population `p_u` near 0.96 there that saturates rather than diverging, so the prior is expected to pass — but "expected to" is why it is a gate. If it fails, the remedies in order are a tighter `tau1_sigma`, a slope in log-age, or clamping the slope's age argument the way the mean is clamped above its high anchor.

**G3 — parameter recovery under the new structure**, at `test`, sited at young ages on production for the reason [202608141600](202608141600-rank-stability-tracking.md) §5 gives: below 30 months on production the binomial measurement bound exceeds the entire within-child variance, and that is precisely where a slope has to be told apart from noise. Add `vg19` to `_TARGETS` in `recovery/spec.py` as `_BIVARIATE_RE` — the engine is unchanged, so no new recovery specification is needed. **A recovery failure here is an informative result, not a blocked one.**

**G4 — it must beat VG10, and say what it did to `kappa`.** LOO against VG10, and the headline diagnostic: does `kappa_u`'s 24→48 month decline shrink once the child slope can absorb drift? That number is the whole reason for the exercise. Also check `rho01` against the ML value of +0.43 and the implied spread-by-age against [202608141600](202608141600-rank-stability-tracking.md) §9's table.

**G5 — convergence at `rep`,** on the family's usual tier. The risk is a funnel on `tau1`: 433 of 767 children on production contribute one observation and inform the marginal spread but not the drift. Non-centring and the prior's mass near zero are the mitigations; `rep-hightune` is the fallback, as for the TD hierarchical models.

## 6. Sequence and cost

| step | work                                                                     | cost                   |
| ---- | ------------------------------------------------------------------------ | ---------------------- |
| 1    | `SubjectSlopePriorParams`, the definition subclass, VG19's registration   | half a day             |
| 2    | `build_child_slope` in `gp_utils.py`; engine branch in `common_bivariate_re` | half a day          |
| 3    | Name preservation, predictive branch, `comparison.py` age-varying spread  | a day                  |
| 4    | G2 prior predictive; `dev` fit; tests                                     | half a day + minutes   |
| 5    | G3 recovery, 3 replicates at `test`                                       | a few hours of sampling |
| 6    | G5 fit at `rep`; G4 comparison; report page                               | ~2h sampling + a day   |

Sampling cost: the random-effect dimension doubles (832 children x 2), so budget **1.5–2x VG10's 60 minutes** at `rep`, and memory in the same band as the rest of the DS joint family — this is not a VG11/VG13-class job.

## 7. What happens to Proposal A1

A1 becomes the constrained special case rather than a separate line of work. `rho01 = 1` is A1; `tau1 = 0` is the model of record. Fitting VG19 with `rho01` free tests both in one model, and the registered `vg10 / a1-tau-age-varying` variant stays where it is — as the diagnostic of how much of `kappa`'s decline is at stake, run before any of this was built.

## 8. What this plan does not cover

- **The typically-developing models.** TD spoken has 2,017 children with repeats but spans only 8–30 months; whether a slope is identified there is a separate Gate 1, on TD data, and it has not been run.
- **VG15 and the signing outcomes.** `common_joint_modality` would follow only if VG19 earns it.
- **The `rho01`-versus-`tau1` interpretation for policy.** A positive `rho01` means children ahead pull further ahead. That is a finding with practical weight, and how it should be reported — and hedged, given it is estimated from 334 children over a median 12-month span — is a question for the report, not for the engine.
