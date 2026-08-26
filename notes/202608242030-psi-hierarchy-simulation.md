# VG15's four-group `psi` hierarchy: the prior is not the cause

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Measurement record, 2026-08-24, answering items 2 and 3 of [#226](https://github.com/dseinternational/vocabulary-growth/issues/226). **`tau_psi ~ HalfNormal(1.0)` should stay.** At the precision the four cross-tabs actually carry it is well calibrated — 89% coverage 0.885 against nominal 0.89 for the per-study associations, and per-study tracking at or above what a model that _knew_ `tau_psi` could achieve. The registered `tau-psi-narrow` variant, by contrast, fails recovery outright. The one bias the hierarchy does reproduce is on the population `psi`, from its own `Normal(0.3, 0.5)` prior, and it is about **−3%**, not the −25% three replicates suggested. Reproduced by `scripts/experiments/psi_hierarchy_simulation.py --replicates 2000`.

## 1. Why not just refit VG15

[#226](https://github.com/dseinternational/vocabulary-growth/issues/226) reports `psi` and `psi_study` recovering biased low through an underestimated `tau_psi`, and proposes the standard few-groups pathology: with four informing sources the between-study spread is barely identified, and a prior median of 0.67 against truths of 0.9–1.3 pulls it down, over-shrinking the group estimates. Its evidence is three `test`-tier VG15 replicates, one of which cleared the convergence gate.

More VG15 replicates would cost days and would still confound the hierarchy with everything else in a joint three-outcome model. This isolates the hierarchy instead, by replacing each study's cross-tab likelihood with its Laplace approximation and leaving every other line of the association block exactly as VG15 writes it:

```
log_psi   ~ Normal(0.3, 0.5)
tau_psi   ~ HalfNormal(tau_psi_sigma)
z_psi     ~ ZeroSumNormal(sigma = sqrt(J/(J-1)), shape = J)
log_psi_j  = log_psi + tau_psi * z_psi[j]
y_j       ~ Normal(log_psi_j, s_j^2)      <- the only substitution
```

Everything then has a closed form. No MCMC, no convergence gate, no sampling noise, and 2,000 replicates per cell in place of three. `--verify` fits one replicate with PyMC, using PyMC's own `ZeroSumNormal`, and the two agree on every quantity to within MCMC error (`|z| <= 1.4`).

## 2. What each source actually carries

The `s_j` are not invented. Each is the observed information of the model's own Dirichlet-Multinomial likelihood on that study's real cells, at each row's own observed margins, with `conc` at VG15's prior median of 20.1. So the _relative_ precision of the four sources is the data's.

| study   | table     | administrations | children | information | `s_j` |
| ------- | --------- | --------------: | -------: | ----------: | ----: |
| `es_01` | four-cell |             185 |      185 |       83.89 | 0.109 |
| `nz_01` | produced  |             111 |   **33** |       20.43 | 0.406 |
| `uk_02` | four-cell |              56 |       28 |       38.63 | 0.228 |
| `uk_07` | four-cell |              82 |       30 |       67.41 | 0.201 |

The table reproduces #226's structural observation exactly: `nz_01` has the second-largest number of administrations and the fewest children, and it ends up with much the largest `s_j`. That is because `s_j` is discounted for repeated administrations of one child — `--design-rho 1`, the default, makes a study's information track children rather than administrations, which is the conservative choice and the one #226 argues for. At `--design-rho 0` `nz_01`'s `s_j` would be 0.221 rather than 0.406, and every shrinkage result below would be _weaker_.

**A fidelity check.** The reduced model's own posterior SD for `nz_01` is 0.408 on the log scale. VG15 reports `nz_01` at 3.29 with an 89% interval of [1.67, 5.33], which implies a log-scale SD of about 0.363. The reduction is 12% wider than the model it stands in for — close enough that its answers are about VG15 and not about a caricature of it.

## 3. `tau_psi` recovery, by prior and by truth

2,000 replicates per cell. Truths span VG15's own three recovery truths (0.446, 0.923, 1.279).

| `tau_psi_sigma`  | true `tau` | median `tau` |   bias | `tau` coverage | group slope | oracle slope | group coverage |
| ---------------- | ---------: | -----------: | -----: | -------------: | ----------: | -----------: | -------------: |
| **0.3** (narrow) |       0.45 |        0.317 | −0.133 |          0.803 |       0.747 |        0.631 |          0.853 |
| **0.3**          |       0.70 |        0.441 | −0.259 |          0.544 |       0.838 |        0.796 |          0.852 |
| **0.3**          |       0.95 |        0.549 | −0.401 |          0.234 |       0.889 |        0.875 |          0.856 |
| **0.3**          |       1.30 |        0.678 | −0.622 |      **0.022** |       0.919 |        0.928 |          0.854 |
| **1.0** (record) |       0.45 |        0.491 | +0.041 |          0.940 |       0.858 |        0.631 |      **0.884** |
| **1.0**          |       0.70 |        0.698 | −0.002 |          0.916 |       0.913 |        0.796 |      **0.886** |
| **1.0**          |       0.95 |        0.878 | −0.072 |          0.924 |       0.943 |        0.875 |      **0.885** |
| **1.0**          |       1.30 |        1.102 | −0.198 |          0.918 |       0.962 |        0.928 |      **0.883** |
| **2.0** (wide)   |       0.45 |        0.540 | +0.090 |          0.924 |       0.859 |        0.631 |          0.878 |
| **2.0**          |       0.70 |        0.808 | +0.108 |          0.912 |       0.937 |        0.796 |          0.885 |
| **2.0**          |       0.95 |        1.041 | +0.091 |          0.908 |       0.957 |        0.875 |          0.882 |
| **2.0**          |       1.30 |        1.342 | +0.042 |          0.926 |       0.971 |        0.928 |          0.893 |

**Group slope** is the regression of the recovered group deviation on the true one, pooled: 1.0 means the estimates track their truths, 0 means they are pulled entirely to the centre. It is the quantity #226 reads off its own table by eye ("posterior means cluster at 1.9–2.4 largely independently of the truth").

**Oracle slope** is the same quantity for a model that _knew_ `tau_psi` and the centre, `(tau^2/(tau^2+s^2))^2`. It is below 1 even for a correctly specified model, because two attenuations compose: the posterior shrinks the observed deviation, and the observed deviation is itself the true one plus noise. Without that benchmark every row above would look like evidence of over-shrinkage; with it, the reading reverses.

### 3.1 The registered prior is well calibrated

At `HalfNormal(1.0)` the per-study coverage is 0.883–0.886 against a nominal 0.89 — as close as 2,000 replicates can resolve. `tau_psi` coverage is 0.92, mildly conservative. And the group slope **exceeds the oracle at every truth**, by as much as 0.23 at the smallest: estimating `tau_psi` rather than knowing it makes the group estimates track their truths _better_, because `tau_hat` moves up in exactly the replicates where the realised spread is large, so the model shrinks less where there is more to preserve.

`tau_psi` is underestimated, and in #226's direction — −0.198 at a truth of 1.30, about −15%. But that underestimation does not propagate into over-shrunk group estimates. Item 3 of #226 asked whether `HalfNormal(1.0)` should be reconsidered for four groups. On this evidence: **no**.

### 3.2 `tau-psi-narrow` fails recovery

`HalfNormal(0.3)` returns 0.678 for a truth of 1.30, and its 89% interval contains the truth in **2.2% of replicates**. It is not a plausible alternative prior; it is a miscalibration.

That matters for how its existing robustness verdict is read. The reported sensitivity run found `tau-psi-narrow` **robust** — the _reported quantities_ did not move outside VG15's own intervals. That is a real finding and it is not contradicted here, but it is a much weaker claim than it looks: a prior can leave the headline numbers alone while destroying the calibration of the parameter it acts on. Item 2 of #226 asked for these variants to be scored against recovery rather than against reported-quantity robustness. Scored that way, the narrow variant fails and the wide one passes.

## 4. What the hierarchy does _not_ explain

#226's sharpest number is `nz_01`, whose truth of 6.26 came back as 1.97. Read as deviations from the population value in each replicate, two of its three replicates show essentially **zero** tracking (slopes of about −0.03), and the third 0.69.

At the sources' own precision this simulation gives `nz_01` a slope of **0.881** — the worst of the four, in the right order, but nowhere near zero. So how much less informative would the sources have to be for the hierarchy alone to produce the reported collapse? `--se-scale` multiplies every `s_j` by a common factor, holding the data's relative precisions fixed:

| `s` scale | `nz_01` `s_j` | `nz_01` slope | oracle | pooled slope | group coverage |
| --------- | ------------: | ------------: | -----: | -----------: | -------------: |
| 1         |         0.406 |     **0.881** |  0.715 |        0.943 |          0.885 |
| 2         |         0.812 |         0.699 |  0.334 |        0.820 |          0.864 |
| 4         |         1.623 |         0.408 |  0.065 |        0.577 |          0.849 |
| 8         |         3.246 |         0.210 |  0.006 |        0.317 |          0.848 |

(at `HalfNormal(1.0)`, true `tau` 0.95)

Even at **eight times** the SEs the cross-tabs imply, `nz_01`'s slope is 0.21 rather than ~0. Two readings are available and the evidence does not separate them:

- The three-replicate estimate is noisy. Two replicates of one study is very little to conclude "essentially zero tracking" from, and only one of the three cleared the convergence gate.
- VG15's per-study likelihood really is far less informative than its cross-tab alone, because `r`, `q` and `conc` are estimated jointly with `psi` rather than held fixed as they are here. That is a statement about joint estimation, not about having four groups.

Either way, **the number of groups is not the mechanism**, and no change to `tau_psi`'s prior addresses it. The first thing to do about it is the cheap one: more replicates at `rep`, which #226 already lists as item 1.

## 5. The bias that does reproduce

`log_psi` comes back biased by **−0.033** in every one of the twelve cells, at every prior and every truth, with coverage 0.84–0.87 against nominal 0.89. On the reported scale that is `psi` low by about 3.3%: a reported 2.34 against a truth nearer 2.42.

This one is not the hierarchy's; it is `log_psi ~ Normal(0.3, 0.5)` pulling toward a prior median of 1.35. It is also exactly predictable in closed form. Under the zero-sum constraint the group deviations cancel in the mean, so the centre is informed with variance `sum(s_j^2)/J^2 = 0.0168`; combining that with the prior at the truth `log(2.34) = 0.850` gives a posterior mean of 0.816, a bias of **−0.034** — the simulation's −0.033 to within Monte Carlo error.

So the direction #226 reports for the population association is real and reproducible. Its size, from the hierarchy and the population prior alone, is about a seventh of what three replicates suggested (−0.033 against roughly −0.24 on the log scale).

## 6. What this changes

- **Keep `tau_psi ~ HalfNormal(1.0)`.** #226 item 3 is answered and closed. Its prior median of 0.67 sitting below the plausible truths is real but does not produce the reported harm.
- **`tau-psi-narrow`'s robust verdict needs a caveat wherever it is cited.** It is robust in reported quantities and badly miscalibrated in the parameter it varies. #226 item 2 is answered.
- **`psi` is biased low by about 3%** by its own population prior, reproducibly, and the report can now say so with a number rather than a direction. That is far smaller than the shrinkage caveat already in `_caveats-signing.qmd`, and it sits alongside — not in place of — the Type-M warning that selecting on `psi > 1` inflates the estimate.
- **The per-study over-shrinkage remains unexplained**, and this rules out the explanation #226 proposed for it. The next step is item 1 (more `rep` replicates), not a prior change.

## 7. What this cannot say

The reduction holds `r`, `q` and `conc` fixed and replaces each study's Dirichlet-Multinomial by a Gaussian, so it drops every correlation between `psi` and the rest of VG15, and every way each study's own likelihood departs from a normal one. A bias found here is therefore a **lower bound** on VG15's. A bias _not_ found here — which is what §3 reports — locates the problem outside the hierarchy but does not say where outside it lies.

The `s_j` also depend on `conc`, fixed at VG15's prior median. A smaller fitted `conc` would raise every `s_j` together, which is what §4's scale sweep spans.
