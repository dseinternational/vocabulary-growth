# Marginalising the singleton child effects: what it took, and what it costs

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Implementation record, 2026-08-23. Lever №3 of [202608231410](202608231410-td-geometry-remaining-levers.md) §3, which the study owner adopted second (§6 there). The lever is implemented and tested for the univariate random-effect engine — VG11 and VG12 — and **no model of record adopts it**: a guard test enforces that, because adopting it is a graph change that invalidates the adopting model's fit. Whether it is worth those refits is the VG12 bench's question (§8), and the cost measured in §5 makes that question sharper than the proposing note expected.

## 1. What was built

`src/vocab_growth/models/subject_marginal.py` carries the machinery: the quadrature rule, the row partition, the zero-padded child shift, and the observed-variable factory. `SingletonMarginalisationParams(n_nodes=...)` selects it, and lives on a new subclass, `UnivariateMarginalisedREModelDefinition`, for the reason `UnivariateREModelDefinition`'s docstring already gives — a fit is validated by comparing `dataclasses.asdict` field for field, so a definition class that gains a field invalidates every fit of that class, including models that never set it. VG11 and VG12 stay instances of the parent until the bench says otherwise.

In the engine (`common_univariate_re.py`) three things change in the model graph when the flag is set, and nothing changes when it is not:

- `delta_subject` is sampled over a new `repeat_subject_id` coordinate — one effect per repeat-measured child, labelled by the child code it keeps — instead of over every child;
- the per-row child shift becomes a gather from that vector extended by one trailing zero, with marginalised rows indexed at the sentinel, so `f_obs` on those rows is the population-and-study prediction and the zero is structural rather than estimated; and
- the likelihood becomes `subject_marginal_betabinomial` in place of `pm.BetaBinomial`.

That likelihood is **one observed variable over every row**, not two. A repeat-measured row takes the unchanged conditional Beta-Binomial density on the `mu` that carries its child's explicit effect; a marginalised row takes the quadrature integral of the same density over that child's prior. The two blocks are computed separately and scattered back into row order — running the quadrature everywhere with a zero spread on the repeat rows would have been simpler and cost about 40% more likelihood evaluations for an identical answer. Keeping one variable is what lets the pointwise `log_likelihood`, the posterior predictive, the calibration table and every LOO path keep the name, shape and row order they have always had.

The two blocks are **contiguous slices**, which is why the data preparation reorders the rows so that every marginalised row comes before every conditional one — stably, so within each block the order is the one the merge produced. That is not tidiness; §3 records what the indexing alternatives did. One consequence to know: an explicit and a marginalised fit of the same data have different `obs_id` orders, so anything joining two fits row by row has to join on the analysis frame rather than on position.

## 2. The quadrature was the hard part, and the proposing note under-specified it

[202608231410](202608231410-td-geometry-remaining-levers.md) §3 described the integral as "a one-dimensional integral … evaluated by Gauss–Hermite quadrature (~20 nodes)". Twenty nodes on the `Normal(0, 1)` prior is **not** accurate enough for these models, and the measurement that shows it is the first real result of this work.

The reason is that one administration of an 810-word inventory is informative about that child's own level. The Beta-Binomial's information about a row's logit scales with `kappa`, so the integrand is a narrow spike inside a unit-width prior: at VG11's fitted `kappa` the spike's standard deviation is about 0.2 in prior units, while twenty prior-centred nodes sit roughly 0.6 apart near the centre. The rule then integrates a function it never resolves.

Measured against a mode-centred fine-grid integral, on the real rows and real posterior draws of the two models of record (VG12 from its stored trace; VG11's trace is compacted, so its row predictor was rebuilt from the reported median trajectory and dispersion curve):

| rule             | VG12 worst row | VG11 worst row |
| ---------------- | -------------: | -------------: |
| prior nodes, 20  |        6.6e-02 |        1.5e+00 |
| prior nodes, 40  |        2.1e-03 |        3.9e-01 |
| prior nodes, 80  |        6.7e-06 |        6.1e-02 |
| **adaptive, 20** |    **3.4e-06** |    **4.0e-05** |
| adaptive, 30     |              — |        4.5e-06 |

VG11 is the harder model: its dispersion reaches `kappa` 714 at the youngest ages against VG12's 98, and a larger `kappa` is a sharper spike. Note what the table says about the naive fix — eighty prior nodes, four times the cost, still leave VG11 wrong by 6e-02 on a row.

Re-measured against the shipped implementation rather than a prototype of it, on 800 of VG12's singleton rows at four posterior draws: twenty nodes give a worst row error of 3.4e-06 and 1.2e-05 summed over those rows; forty give 5.8e-10 and 2.8e-09. The rule converges steeply once the nodes are in the right place, which is what makes the node-doubling check in §8 a real check rather than a formality — and what says twelve nodes (1.1e-04 worst) is the wrong economy.

**The adopted rule places the nodes at each row's own integrand mode and scales them by its curvature** — adaptive Gauss–Hermite, which `lme4` and `glmmTMB` use for exactly this reason. Three details are load-bearing:

- **The mode search runs on finite differences, not on analytic derivatives.** Analytic first and second derivatives of the Beta-Binomial log density in the child effect need digamma and trigamma, and differentiating the resulting expression with respect to the model's parameters would then need the third derivative of `gammaln`, which PyTensor does not implement. Central differences keep the whole expression a composition of ordinary Beta-Binomial evaluations, so autodiff returns the exact gradient of the value the sampler is given. Three damped Newton steps are used; measured on VG11's rows the search converges in two, so the third is headroom rather than tuning.
- **The rule degrades to an underestimate, never an overestimate.** The mode is confined to ±8 prior standard deviations and the node scale is capped at the prior's, which the exact conditional posterior of a log-concave likelihood cannot exceed. Rows whose child effect would have to sit far outside that range have log-densities of several hundred negative nats — numerically absent from the sum — and a rule that inflated them would be an attractor for a warmup chain rather than a rounding error. A test pins that the marginal never exceeds a log-density of zero, which no probability may.
- **Everything is forced to double precision.** PyTensor types a bare Python scalar constant as the narrowest dtype that holds it, so `pt.gammaln(811)` and `pt.gammaln(5.0)` both return `float32`, and single-precision `gammaln` of an argument in the hundreds is wrong in the fourth decimal. That is a hundred times larger than the quadrature error the adaptive placement buys, and it is invisible: the graph runs, the fit converges, and the density is quietly wrong. The library casts defensively and a test compares its density against `pm.logp(pm.BetaBinomial.dist(...))` to 1e-9.

The Beta-Binomial density is written out rather than called generically, split into the terms that move with the node and the terms that do not. The binomial coefficient and the `kappa` normaliser are the same at every node of a row, so they are added once after the node sum and cancel outright in the mode search's differences. `alpha + beta` is `kappa` exactly by construction, so the normaliser needs no addition of its own.

## 3. Two defects that only an end-to-end fit could show

Both were invisible to tests that evaluate the density, because in both the **value** is right and the **gradient** is not — and a sampler dies on a gradient.

**PyMC's own Beta-Binomial log density returns a non-finite gradient inside a `CustomDist`.** Calling `pm.logp(pm.BetaBinomial.dist(...), value)` for the conditional block gave a non-finite gradient at 132 of 150 jittered points, against 0 of 150 for the same density written out in `gammaln` terms. NUTS cannot start from a point whose gradient is not finite, so nutpie rejected every initial point it tried, reporting only an opaque `Logp function returned error: ErrorCode(3)`. Both blocks now use this module's own density, which a test pins against PyMC's to 1e-9.

**Advanced indexing to put the two blocks back in row order was worse than useless.** First as a scatter into a zero vector: PyTensor rewrites `set_subtensor` into an in-place write, and nutpie evaluates the density from several threads at once, so it raced — correct at one chain, a Windows access violation at two. Then as a gather of a concatenation, which sampled at one chain but still returned non-finite gradients on the assembled term. Ordering the rows so that the two blocks are slices removes the whole class: the likelihood now contains no advanced indexing at all. After both fixes the model-level check is **0 of 400** jittered points with a non-finite gradient, against 251 of 400 before, and nutpie samples at one chain, at two chains on two cores, and at two chains on one.

A third change came out of the same hunt and is kept on its own merits: the node placement is held out of the gradient with `disconnected_grad`. The mode search divides second differences by the square of a 1e-3 step, so differentiating through it is numerically violent, and pruning that backward pass is most of what the gradient used to cost. It is legitimate because the rule's value is, to quadrature accuracy, invariant to where its nodes sit: what the sampler gets is the exact gradient of the same rule with the nodes held fixed, which is that same quadrature applied to the derivative of the integrand.

## 4. What it removes

At full VG12 size the sampled space goes from **5,849 dimensions to 1,030** — 4,819 singleton children leave, 1,000 repeat-measured children keep an explicit effect, and every other free parameter is untouched. That is the funnel mass [202608050900](202608050900-td-hierarchical-geometry.md) §4 measured as the energy-BFMI driver, removed by algebra rather than by changing the model: the marginal likelihood still contains `tau_subject`, so `kappa` keeps its meaning as within-child dispersion and the Down-syndrome-versus-typically-developing heterogeneity contrast is untouched.

It also takes a large bite out of what a fit stores, which has been a binding constraint on this hardware since [202608050900](202608050900-td-hierarchical-geometry.md) §10. At VG12's `rep` shape — six chains of eight thousand draws — the two child-effect arrays are about 4.5 GB of the stored trace and marginalisation takes them to 0.8 GB; under `compact`, which already drops the scaled copy, 2.2 GB becomes 0.4 GB. That is on top of what [202608231530](202608231530-observation-deterministics-not-sampled.md) removed, and it is the one benefit that does not wait on the bench.

## 5. What it costs, which is the part the proposing note did not anticipate

| VG12, full size     | explicit | marginalised |
| ------------------- | -------: | -----------: |
| sampled dimensions  |    5,849 |        1,030 |
| gradient evaluation |  1.62 ms |      34.7 ms |
| graph compilation   |     ~1 s |        ~36 s |

**A gradient costs 21 times more.** The arithmetic is not subtle: a marginalised row evaluates the Beta-Binomial at twenty nodes plus twelve mode-search points, against one evaluation before, and the gradient of `gammaln` is a digamma of the same cost. Fewer nodes buy back only a little — the mode search is a fixed twelve — and the accuracy table above says twelve nodes is where the error starts to matter.

So the lever's arbiter cannot be energy BFMI alone. Removing 4,819 dimensions has to buy back a factor of 21 in the sampler's own work before it is free, and only what it buys beyond that is a gain. In tree-depth terms the fit needs about a twenty-fold cut in gradient evaluations per iteration, or the same number of them with far higher effective sample size, to break even. That is not obviously out of reach for a geometry that currently fails BFMI — the whole point of a funnel is that the sampler takes many small steps through it — but it is now a measured question rather than an assumption, and **the bench must report effective samples per second beside the diagnostics** (§8).

Two costs that are not in the table but are real: compile time is a fixed per-fit cost that grows with the node count, and PyTensor caches between runs so the figures above are the warm ones -- a cold `nutpie` compilation for a full `rep` fit will be longer; and the posterior-predictive path runs the custom draw function in Numba object mode, which is slow but happens once per fit rather than per gradient.

## 6. Semantics that change, and must be said wherever the fit is reported

- **The pointwise `log_likelihood` of a marginalised row is the marginal predictive density, not the conditional one.** That is the right quantity for a leave-one-subject-out reading of a singleton row — the conditional predictive overfits its own single observation, which is where PSIS-LOO's Pareto-k pathology on these models sits — but it means `elpd` values do not compare across the change, and `loso_compare` must not mix a marginalised fit with an explicit one.
- **Posterior predictive draws for a marginalised row draw a fresh child effect** rather than reusing the fitted one, which widens those rows' predictive intervals to their honest marginal width. The predictive calibration table therefore measures something different, and better, on those rows.
- **`f_obs` and `p_obs` on marginalised rows carry no child effect.** They are the population-and-study prediction for the row, which is what they mean once the child effect is integrated out. Nothing in the univariate reporting path reads them per row, but anything that starts to must know this.

## 7. What is not done

- **The bivariate random-effect engine** (VG13, VG21) is untouched. A child there carries a two-dimensional effect across the understood and spoken outcomes, so a singleton child needs a two-dimensional integral — a product of one-dimensional rules only if the two effects are independent, and VG13's are. That is the natural second stage, and it should wait for the univariate bench: if the cost in §5 does not pay there, it will not pay on a model with twice the effects and a nested likelihood.
- **No model of record adopts the flag**, and `test_the_models_of_record_do_not_carry_the_flag` fails loudly if one does, with the reminder that its published fit is then stale.
- **Parameter recovery has not been run** on a marginalised model. It is an obligation before adoption, not before the bench.

## 8. The bench

```
scripts/experiments/marginal_arm.py explicit --output-dir <throwaway>
scripts/experiments/marginal_arm.py marginal --output-dir <throwaway>
scripts/experiments/marginal_arm.py marginal --nodes 40 --output-dir <throwaway>
```

`compare_marginal_arms.py --output-dir <throwaway>` then scores them. The first two arms are the fifth arm of [202608050900](202608050900-td-hierarchical-geometry.md) §9's table, at `test` config in a throwaway output root; the third is the node-doubling sensitivity check the proposing note requires, and it must move nothing. What to read off them:

1. **Equivalence** — `tau_subject`, `kappa` and the reported trajectory must agree between `explicit` and `marginal` within Monte Carlo error. The marginalisation is exact, so this test has teeth: a disagreement is a bug, not a trade-off. It cannot be a bit-for-bit comparison, because the sampled spaces have different dimensions.
2. **Geometry** — energy BFMI, divergences, R-hat, and the `tau_subject`/`kappa` ridge correlation, against the four arms already in that table.
3. **Cost** — effective samples per **gradient evaluation**, which is the comparison that survives running the arms on different machines: the sampler's own leapfrog count is in `sample_stats.n_steps`, and §5's factor of 21 converts it to work. This is the number that decides adoption; wall-clock is printed beside it but covers the whole pipeline rather than sampling alone.

**Expectations, set before the first arm runs**, so the bench can falsify them:

- **Equivalence holds.** A disagreement beyond Monte Carlo error is a bug in the marginalisation, not a property of it.
- **Divergences fall**, from the 14 the partition arm reported. The funnel that produces them is what leaves the sampled space.
- **Energy BFMI rises materially**, from the 0.19–0.20 every arm of [202608050900](202608050900-td-hierarchical-geometry.md) §9 reported towards the 0.981 its single-administration arm reached by _deleting_ the child effects. Marginalisation removes the same dimensions without changing the model, so if BFMI stays near 0.2 the mechanism that note attributes the failure to is wrong — which would be worth more than the lever.
- **Effective samples per gradient is genuinely open.** The gradient is 21 times dearer and the tree should be shallower; nothing measured so far says by how much.

### A desk-scale rehearsal

Both arms were run at 5% of the children and `dev` sampling (two chains, 500 draws) to exercise the harness end to end. This is **not** the bench and cannot stand in for it, but it is the first side-by-side evidence:

| arm            | sampled dimensions | min BFMI | divergences | max R-hat | min ESS | ridge |
| -------------- | -----------------: | -------: | ----------: | --------: | ------: | ----: |
| explicit       |                325 |    0.223 |          15 |    1.1172 |      16 | 0.759 |
| marginal, K=20 |                 81 |    0.632 |           6 |    1.0128 |      51 | 0.725 |

Equivalence held on every shared parameter — `tau_subject`, `kappa_young`, `kappa_old`, `v_total`, `subject_variance_share`, `tau`, `eta`, `ell` — with |z| at most 1.01, which is what the exactness claim predicts and what a botched marginalisation would have failed.

**Two cautions.** The explicit arm is not converged at this budget (max R-hat 1.117, min ESS 16), so its effective sample size — and therefore the work comparison, which came out at 617 against 119 effective samples per million gradient evaluations in the explicit arm's favour — is not a number to quote. And a twentieth of the children is a different funnel from the whole pool: 294 children against 5,819, of whom 50 rather than 1,000 are repeat-measured. What the rehearsal does say is that the plumbing works end to end, that the direction of the geometry change is the predicted one, and that the equivalence check has been exercised rather than merely specified.

## 9. What the tests pin

`tests/test_subject_marginal.py` (29 tests, 83 s) covers: the quadrature weights as a normalised expectation; the written-out density against PyMC's to 1e-9; the marginal against a fine-grid integral at both models' fitted regimes; that doubling the nodes moves nothing; that a vanishing child scale reproduces the conditional density; that the marginal is never a probability above one, including on rows a fit reaches only in early warmup; the row partition, the sentinel gather, the singleton-first order and the likelihood's refusal to build without it; the definition guards; and, on a real VG12 subsample, that data preparation applies that order while a flag-off build keeps the rows as they came, that only repeat-measured children keep an effect, that marginalised rows' `f_obs` does not move when the effects move, that repeat rows keep the conditional density to 1e-9, that marginalised rows match an independent numerical integration, and that the whole path — NUTS, `compute_log_likelihood`, posterior predictive — runs and returns the shapes every consumer expects.

Beyond the suite, a 5% VG12 subsample was fitted end to end at `dev` — prior predictive, sampling, diagnostics, posterior predictive through the custom draw function, calibration, summary, plots, report — which is what §3's two defects were found by and what the suite alone could not have caught.
