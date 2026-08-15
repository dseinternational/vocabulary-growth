# Should the cross-lag enter the other models of record? No — and the fitted evidence says why

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Decision analysis on the fitted `rep` traces of VG10 and VG16. Every figure is reproduced by `scripts/experiments/vg16_crosslag_quantification.py` (§5 of its output is the comparison table below). Companion note: [202608151120](202608151120-vg16-cross-lag-quantified.md) quantifies what the cross-lag measures. The constructive follow-up is the correlated subject random-effects proposal (VG20).

## 1. The question

VG16's cross-lag is reliably positive (`beta_lag` = +0.203, 89% ETI [0.093, 0.316]) and interpretable — about a quarter of the spoken-vocabulary gap between children who differ in comprehension is carried by better conversion. Does a real, positive, interpretable term warrant inclusion in the models of record — VG10 in particular, and the joint family generally?

**No.** The comparison below isolates the term's effect on everything the models of record report, and it is nil; the mechanism analysis says the term is also the wrong parameterisation for what it detects. Keeping VG16 as the single-purpose model that reports the coefficient, and pursuing the association it detects with a correlated random-effect block instead, dominates adoption on every criterion.

## 2. The isolating comparison

The two `rep` fits differ in exactly one live field. Diffing the recorded `model.definition` blocks of the two manifests yields five differences: `model_id`, `config_name`, `banner` (naming), `use_cross_lag` (False vs True), and `lag_baseline` — which is inert when `use_cross_lag=False`, since nothing else reads it. So VG16 minus VG10 *is* the cross-lag's effect on the reported model, with data, priors, anchoring, sampler and sampling budget all held fixed.

That effect, at the population level across the reporting ages (12–84 months):

| age (mo) | Δ understood (words) | Δ `q` (pp) | Δ spoken (words) |
| -------: | -------------------: | ---------: | ---------------: |
| 12 | −0.20 | +0.10 | +0.01 |
| 18 | −0.19 | +0.14 | +0.06 |
| 24 | −1.14 | +0.13 | +0.08 |
| 30 | −2.09 | +0.02 | −0.15 |
| 36 | −2.13 | −0.19 | −0.77 |
| 42 | −1.91 | −0.37 | −1.56 |
| 48 | −2.23 | −0.33 | −2.00 |
| 54 | −2.55 | −0.08 | −1.94 |
| 60 | −2.45 | +0.25 | −0.66 |
| 66 | −1.80 | +0.30 | +0.21 |
| 72 | −0.60 | +0.17 | +0.32 |
| 78 | −0.29 | +0.12 | +0.86 |
| 84 | −0.23 | +0.35 | +2.03 |

The largest movements anywhere: **2.6 words understood (≤ 1.2%), 0.4 percentage points of `q`, 2.0 words spoken (≤ 2.0%, peaking in percentage terms only where the counts are fractions of a word)**. The random-effect scales are equally unmoved: `tau_subj_u` 0.7970 → 0.7966, `tau_subj_q` 1.2855 → 1.2826. Adding the cross-lag to VG10 would reproduce every figure, table and downstream comparison to within rounding. There is nothing to adopt *for*.

## 3. Why that is structural, not an accident of this fit

**It touches 29% of the data.** 412 of 1,431 observations have a prior-wave comprehension source, from 250 of 767 children; for the rest the term is identically zero. Acceptable in a model built to estimate one coefficient; poor structure for a model whose job is a population trajectory.

**It absorbs no misattributed variance.** The usual reason to adopt a term into a model of record is that its absence forces structure into the wrong parameter — the argument that registered A1 as a sensitivity ([202608141600](202608141600-rank-stability-tracking.md) §9). The cross-lag carries ~2.9% [0.6, 6.9] of the variance in the `q` logit and moves `tau_subj_q` by 0.2%. Nothing in VG10 is currently wearing this term's name.

**It is the wrong parameterisation for what it detects.** The tracking analysis rejected the latent AR(1) on both outcomes — the within-child deviation has no memory beyond the occasion — so the occasion and sampling components of `x_lag` carry no signal and `beta_lag` measures a **between-child** association (persistent receptive standing ↔ persistent conversion efficiency) through a proxy with reliability 0.53. The direct evidence is in VG16's own posterior: with the cross-lag fitted, the realised subject intercepts still correlate at **+0.135** [0.087, 0.180] across children, an association the independent-prior structure has nowhere to put. Baking a mis-parameterised term into the models of record is worse than leaving it in one diagnostic model whose limits are documented.

**It would make the headline model's structure depend on study design.** Which children have a prior-wave comprehension measure is an artefact of which of the fourteen studies followed children longitudinally, and at what intervals. A population trajectory meant to generalise should not carry a term defined by that. There is also a quiet interpretive cost: with the term present, the reported population curve becomes "a child whose previous comprehension was exactly at expectation" rather than simply the population level.

**It is not portable anyway.** The cross-lag is implemented only in `common_bivariate_re`. VG14 (`common_trivariate`) and VG15 (`common_joint_modality`) would each need it built from scratch; VG05 (`common_bivariate`) has no subject random effects to define either baseline against, which `_validate_cross_lag` rejects by construction. The realistic candidate set is VG09/VG10 alone — so "more generally" reduces to "VG10", and §2 shows what it buys there.

## 4. What to do instead

The association the cross-lag detects is real and worth estimating properly. The right instrument is a **correlated subject random-effect block**: replace the independent priors on `(delta_subj_u, delta_subj_q)` with a joint prior through an `LKJCholeskyCov`, and read the correlation `rho_uq` directly. Against the cross-lag, it:

- applies to **all 767 children**, not the 250 with longitudinal repeats, and does not depend on study design;
- estimates the between-child association on the correct footing, rather than through a noisy lagged proxy — no attenuation to argue about;
- absorbs the +0.135 the current structure cannot express;
- **nests the model of record at `rho_uq = 0`**, so it is a testable one-parameter extension, exactly the criterion the child-slope plan ([202608141900](202608141900-child-slope-implementation-plan.md)) sets for structural candidates — and a strictly smaller change than the random slopes planned there (one correlation parameter against a second per-child coordinate);
- uses the seam Proposal A1 already opened in the bivariate engine's subject-RE block.

This is proposed as **VG20**, fitted alongside VG10 for comparison — VG17/VG18 are the exploratory sign-group modules and VG19 is reserved by the child-slope plan. The two extensions are orthogonal (`rho_uq` correlates the two intercepts across outcomes; the slope plan's `rho01` correlates intercept and slope within one), so VG20 can be built and judged first, and its covariance folded into VG19's if both earn their place. Proposed in [#224](https://github.com/dseinternational/vocabulary-growth/issues/224).

## 5. Decision

1. **VG16 stays as it is**: a single-purpose registered model reporting `beta_lag`, with its between-child reading and attenuation documented ([202608151120](202608151120-vg16-cross-lag-quantified.md) §5–6).
2. **The cross-lag is not propagated** to VG10 or any other model of record.
3. **The correlated-RE model (VG20) is the proposed follow-up** for the association the cross-lag detects.

The stale "≈ null" description of VG16's headline in `docs/models/README.md` and the `dev`-tier figure in `docs/models/vg16/index.qmd` remain to be corrected separately ([202608151120](202608151120-vg16-cross-lag-quantified.md) §7).
