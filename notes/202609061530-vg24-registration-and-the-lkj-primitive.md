# VG24 registered, and why it is not built on `pm.LKJCorr`

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-06. Registers **VG24** ([#296](https://github.com/dseinternational/vocabulary-growth/issues/296)) — VG15 with its three child random intercepts drawn from one joint Normal with a free correlation — ahead of the `us_03` refit, so it fits in the same window as VG15 rather than needing a second one. Nothing is fitted here; the numbers below are smoke checks from a short `dev`-tier run, not results.

## The finding worth carrying forward: `pm.LKJCorr` is not usable here

The obvious primitive for a 3x3 correlation is `pm.LKJCorr`, and its own docstring shows exactly the construction this model needs — draw the correlation, Cholesky it, transform an uncorrelated Normal. **On the locked PyMC 6.3.1 that construction is wrong in two independent ways**, and the first is silent.

**It returns the Cholesky factor, not the correlation matrix.** A draw is lower-triangular with rows of unit norm; `L @ L.T` is the correlation matrix with the unit diagonal. So `corr[0, 1]` — the natural way to read a correlation off it, and what the docstring's example implies — reads a **structural zero**. VG24's first build did precisely this, compiled, sampled, and reported `rho_uq`, `rho_u_sign` and `rho_sign_q` as `+0.000` in every draw of every chain. A model that has silently fixed its headline parameter at zero looks exactly like a model that fitted and found nothing, which is the failure mode worth naming: the check that caught it was reading the draws, not reading the code.

**Its marginals are not exchangeable, and its sampler and its density disagree.** Under LKJ every off-diagonal has the same marginal. Measured on 200,000 forward draws, and again by sampling the density with NUTS:

| Route                                 | SD of $\rho_{01}$ | SD of $\rho_{02}$ | SD of $\rho_{12}$ |
| ------------------------------------- | ----------------- | ----------------- | ----------------- |
| `LKJCorr`, forward draws              | 0.408             | 0.378             | 0.378             |
| `LKJCorr`, NUTS on the density        | 0.450             | 0.409             | 0.407             |
| `LKJCholeskyCov`, NUTS on the density | 0.405             | 0.410             | 0.409             |
| LKJ(2), $n = 3$ theory                | 0.408             | 0.408             | 0.408             |

The forward figures are stable across three seeds at 200,000 draws, so this is not noise. Two consequences for a model like VG24. The density is what NUTS samples, so the forward/density disagreement means prior predictive draws would not describe the prior the fit uses. And because the marginal depends on **position in the matrix**, `rho_sign_q` — at (1, 2), the quantity VG24 exists to estimate — would carry a different prior from `rho_uq` at (0, 1), the quantity it is compared against. That is not a rounding difference; it is a different amount of regularisation on the two ends of a comparison.

`pm.LKJCholeskyCov` is exchangeable, matches theory, and its `sd_dist` reproduces VG15's three independent `HalfNormal(1.5)` scales exactly (marginal mean 1.18–1.26, SD 0.90, against the theoretical 1.197 and 0.904). VG24 uses it. `tests/test_joint_correlated_subject_re.py` pins the exchangeability and the scale marginals from the density, and separately pins the `LKJCorr` shape defect as the recorded reason for the choice — if a later PyMC fixes it, that test fails and the decision gets re-read rather than silently inherited.

**Not audited here:** whether VG20's, VG22's and VG23's `rho_uq` are affected. They are not built on `LKJCorr` — the bivariate engine writes the correlation as an explicit `Beta` on $(\rho+1)/2$, which is exactly LKJ at $n = 2$ and was chosen so the correlation stays a named variable — so the defect does not reach them by construction. Worth stating because "an LKJ bug" sounds like it should.

## What VG24 is

VG15 with `subject_re_correlation_eta = 2.0` on a `JointCorrelatedSubjectREModelDefinition` subclass, derived from VG15 through `_as_definition_subclass` so VG15's serialised definition — and every VG15 fit on disk, sensitivity arms included — is untouched. The engine seam is the three gated `subject_shift` calls in `common_joint_modality`.

- **Only the correlation is added.** `tau_subj_u`, `tau_subj_q` and `tau_subj_sign` keep their names, their `HalfNormal(1.5)` priors and their per-child meaning, so VG15's and VG24's child scales are directly comparable. They move from free variables to Deterministics on `sd_dist`'s output; `report_cells._DERIVED_NAMES` already records that these three are sampled in some models and derived in others, so nothing downstream needs a special case.
- **VG15 is nested exactly at the identity**, not approximately: `LKJCholeskyCov` returns the Cholesky factor of the _covariance_, which at the identity correlation is `diag(tau)`, so `z @ chol.T` is `z * tau` — the expression VG15's independent branch emits, op for op. Asserted numerically rather than by inspection.
- **`eta = 2` matches VG20 and VG23** so the three models' `rho_uq` are estimated under the same concentration. The induced marginal is not identical: at $n = 3$ the per-correlation prior SD is 0.41 against VG20's 0.45 at $n = 2$. Close enough that the comparison is fair, different enough that the model page says so; the priors table now computes its quoted bounds from the block's actual size, which also makes it right for VG20 and VG23 rather than assuming $n = 2$.

## Build check on the real frame

Graph diff against VG15, both prepared from the current (post-`us_03`) loader:

- Free variables **42 → 40**: three `tau_subj_*` scalars replaced by one packed `subject_re`.
- Added deterministics: `rho_uq`, `rho_u_sign`, `rho_sign_q`, plus the three `tau_subj_*` under their existing names.
- Observed nodes identical; `delta_subj_*` keep their `subject_id` dims.

**The free-variable count is worth recording against [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.1.** That task is open because `vg15 fallback-dispersion`, at 44 free variables, failed numba's register allocator in `np_concatenate` on the linux-aarch64 refit VM where the 42-variable model of record compiled. VG24 sits at **40**, below the model of record, so it is not expected to hit that ceiling — and it compiled and drew under numba on win-amd64 here (577 s to compile). That is an expectation, not a result: the failing platform is the VM, and `--nutpie-backend jax` remains the escape hatch if it appears.

A 2 x (150 + 150) smoke run on the real frame gave `rho_uq` +0.32, `rho_u_sign` +0.26, `rho_sign_q` +0.39, with the three scales at 0.83 / 1.27 / 1.18 against VG15's fitted 0.79 / 1.27 / 1.16. **These are smoke numbers from an unconverged run and are not evidence of anything.** They are recorded only because they show the block is live and the scales did not move when the correlation was added, which is what a working nested parameterisation should do.

## Registration surface

Beyond the definition and the engine: catalogue entry on the `joint` engine (role left `UNCLASSIFIED`, which fails closed to full publication strictness — classifying it is a study-owner decision), `model_vg24.py`, `docs/models/vg24/index.qmd` written from scratch rather than derived from VG15's, `recovery/spec.py` on `JOINT_SPEC`, and the priors-table rows for the two new correlations. Six pinned test sets needed updating, each of which exists to make a silent registration visible, and each of which caught this one: the clamp scope, the shared DS-joint prior blocks, the kappa-form map, the subject-effect plan, the engine dispatch in `test_trend_gp_consolidation.py`, and the model count in the three agent-instruction files.

## Open, deliberately

- **Which children inform `rho_sign_q`.** The child shifts enter the marginal likelihoods only — not the four-cell or produced-cell Dirichlet-Multinomials, so the child block cannot pull `psi`. So `rho_sign_q` and `rho_u_sign` are identified by the children contributing both marginals, not by the whole frame. The model page computes and states the support from the fit's own stored data. Extending the shifts into the cell likelihoods would bring the `es_01`, `uk_07` and `nz_01` children in but changes what `psi` means; that is a separate proposal.
- **The reporter confound.** All three counts come from one questionnaire completed by one parent, and the graph has no informant term, so shared reporting tendency loads onto every correlation and biases all three upward in magnitude. Same caveat VG20 and VG23 carry, with more force here because signing and speech are reported item by item on the same form.
- **The cross-lag.** Still [#297](https://github.com/dseinternational/vocabulary-growth/issues/297), still deferred on [#242](https://github.com/dseinternational/vocabulary-growth/issues/242). VG24 is the between-child estimate and carries no temporal direction.
