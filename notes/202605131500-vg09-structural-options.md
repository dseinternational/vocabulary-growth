# VG09: Structural options for q-trajectory diagnostics

Date: 2026-05-13

> **Note:** This document was generated with assistance from an AI model (Claude, Anthropic) and should be independently verified.

## Context

See `notes/202605131400-vg09-sampler-diagnostics.md` for the diagnostic investigation that motivates this note. In short: VG09 produces small but persistent `r_hat ≈ 1.01–1.02` and `ess_tail` shortfalls on a clustered set of global hyperparameters of the production-ratio (`q`) trajectory. Tighter sampling does not help. The cause is structural: VG09 has more ways to encode the same global level/slope on `q` than the data can disambiguate.

## The redundancy

The `q` trajectory in VG09 is built as four additive components on the logit scale (`common_bivariate_re.py`):

```text
h(a, s, i) = intercept_q + slope_q · a_z   ← linear mean trend
           + eta_q · g_unit_q(a)            ← HSGP correction
           + delta_q,study[s]               ← study random intercept
           + delta_q,subj[i]                ← subject random intercept   (new in VG09)
```

`intercept_q` and `slope_q` are deterministic from two anchor-Beta probabilities at ages 24 and 84 months (`p_slope_low_q`, `p_slope_hi_q`). Three of the four components carry a global level: the linear trend, the GP, and each RE family. The data identify the _sum_; the decomposition is only weakly constrained by the priors.

This is why the flagged parameters cluster on exactly the components that overlap (anchor probabilities, `eta_q`, and `intercept_u` / `p_slope_hi_u` on the analogous u-side). Tighter `target_accept` cannot help because the geometry is a ridge, not a step-size problem.

## Options

> **Correction (2026-07-12, Claude Code/Opus 4.8, re [#155](https://github.com/dseinternational/vocabulary-growth/pull/155)):** the specific values proposed below were superseded. The VG07-posterior-derived Option A anchors (`Beta(3, 22)` / `Beta(20, 4)`) were adopted in VG10 but later judged prior-data double-dipping and **broadened** across the DS-joint family to the weakly-informative, non-double-dipping `Beta(2, 12)` (low) / `Beta(3, 2)` (high). The Option B `eta_q` tightening was adopted at `HalfNormal(0.20)` (not the `HalfNormal(0.15)` floated below) to tame the `q`-GP ridge that broadening the anchors surfaced. Option D (the per-draw GP anchor) remains VG10's structural fix. The options below are preserved as the original proposal record; the current priors live in [`docs/models/PRIORS.md`](../docs/models/PRIORS.md) and `src/vocab_growth/models/definitions.py`.

### Option A — Tighten the anchor-Beta priors (smallest change)

Current priors are essentially diffuse:

| Anchor                     | Current prior | Empirical posterior (VG07 rep) | Proposed prior                   |
| -------------------------- | ------------- | ------------------------------ | -------------------------------- |
| `p_slope_low_q` (at 24 mo) | Beta(1, 1.5)  | ~0.10–0.15                     | Beta(3, 22) — mean 0.12, sd 0.06 |
| `p_slope_hi_q` (at 84 mo)  | Beta(2, 1.2)  | ~0.85–0.95                     | Beta(20, 4) — mean 0.83, sd 0.08 |

**Pros:** one-line change in `definitions.py`; weakly informative; defensible because VG07 has no subject REs on q so its posterior is not subject to the redundancy being addressed.
**Cons:** narrows the ridge but does not remove it. Reviewers may question the use of a previous model's posterior to set the prior.

### Option B — Tighten `eta_q` (constrain GP magnitude)

`eta_q ~ HalfNormal(0.4)` permits the GP to wander ±~0.4 logit units. Drop to `HalfNormal(0.15)`. Forces the linear trend to do the work and reserves the GP for genuine non-linearity.

**Pros:** directly addresses GP ↔ mean-trend competition; trivial change.
**Cons:** the GP is also the model's safety net when the linear trend is wrong; over-tightening biases mid-age `q` estimates. Best paired with A so the anchors are credible enough to carry the trajectory.

### Option C — Tighten subject-RE shrinkage on q

`tau_subj_q ~ HalfNormal(0.5)` is the SD of subject-level shifts in logit-q space. Tighten to `HalfNormal(0.25)`.

**Pros:** removes much of the third redundant dimension; subject REs on q are VG09-specific, so this targets exactly what changed.
**Cons:** subject REs were added because individual variation in q is real — over-shrinking defeats the purpose. Inspect the VG09 posterior of `tau_subj_q` first: if it sits well below 0.5, tightening is free; if near 0.5, you are pushing against signal.

### Option D — Anchor `g_q` to zero at a reference age (reparameterise)

Replace `g_q = eta_q · g_unit_q` with `g_q = eta_q · (g_unit_q − g_unit_q(a_ref))`, where `a_ref` is a reference age (e.g. 48 months — the midpoint of the DS query range). This forces every posterior draw of the GP to pass through zero at `a_ref`, so the linear trend uniquely defines the level there and the GP can only describe deviations.

> **Note on the relationship to the current HSGP.** The existing GPs (univariate models in `common.py`, bivariate models in `common_bivariate.py`) use PyMC's `HSGP`, which is a _zero-mean Gaussian process in prior expectation_ — i.e. `E[g(a)] = 0` averaged over draws. That is **not** the same as `g(a_ref) = 0` for each draw. For any single posterior draw, the GP function carries an arbitrary constant component which trades off freely with `intercept_q`. The zero-mean prior centres the trade-off; it does not remove the ridge. Option D would be a stronger, per-draw constraint that no other GP in the codebase currently has.

**Pros:** the cleanest structural fix for the GP ↔ intercept redundancy. Doesn't require strong priors. Standard trick in additive models. Would also tighten the analogous understood-side issue if applied symmetrically to `g_u`.
**Cons:** modest change in `common_bivariate_re.py` (and parallel changes in `common_bivariate.py` if applied to VG05–VG08). `g_q` becomes a deviation field, not the GP itself — slight reinterpretation when reporting. Does not address the subject-RE ↔ intercept redundancy.

### Option E — Drop the linear mean trend; pure GP with a single anchor

Replace `mean_trend_q = intercept_q + slope_q · a_z` with a single anchor probability (e.g. at the midpoint age) and let the GP carry the whole shape.

**Pros:** structurally eliminates slope ↔ intercept ↔ GP collinearity. One global level, one nonparametric shape.
**Cons:** large change. Loses the interpretable "slope between A and B" parameter the report leans on. GP needs a wider `eta_q` prior; more sensitive to GP basis tuning. Propagates to other bivariate models if you want consistency.

### Option F — Reparameterise anchors as midpoint + difference

Model `mu_q = (logit p_low + logit p_hi)/2` (overall level) and `dlt_q = logit p_hi − logit p_low` (slope on logit scale), each with its own prior. These are typically much less correlated in the posterior than the raw anchors.

**Pros:** orthogonalises the most r-hat-marginal pair; mostly a coordinate change rather than a model change.
**Cons:** does not touch the GP or RE redundancy. May improve only `slope_q` / `intercept_q` r-hat.

## Recommendation

If the goal is to clear diagnostic warnings without substantively changing the model's output, **A + D** is the smallest credible fix:

- **A** narrows the anchor priors to where VG07 (no subject REs) already located them.
- **D** removes the structural GP ↔ intercept overlap with a one-line constraint per GP that does not change the model's interpretation, only its parameterisation.

A more conservative variant is **C + D**: tighten the new VG09-specific subject-RE shrinkage and add the GP anchor constraint, with no prior changes on the trajectory itself.

Option E is a larger commitment than the diagnostic issue warrants on its own, and should be considered only as part of a broader move toward GP-only mean functions across the model family.

## Suggested first experiment

Implement **A + D** as a VG09 variant (e.g. `VG09b`) without touching VG09. Fit at `rep` config. Compare:

1. `r_hat` and `ess_tail` for the previously flagged parameters.
2. `q_query` and the joint trajectory medians/HDI bands against the canonical VG09 run — these should be substantively unchanged if the diagnosis is correct.
3. The posterior of `tau_subj_q`; if it is similar to VG09's, the subject RE on q is robust to the parameterisation.

If A + D resolves the diagnostics without moving the derived quantities, promote VG09b to VG09 and apply the GP anchor constraint symmetrically to `g_u` and (for consistency) to VG05–VG08.
