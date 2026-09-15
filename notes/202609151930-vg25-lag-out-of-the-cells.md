# VG25's first `rep` fit was bimodal, and the lag leaves the cross-tab cells

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-15. **Decision, by the study owner, the same day:** fix VG25 within the current refit cycle, by keeping the within-child baseline and confining the lag to the spoken marginal (`sign_lag_in_cells=False`). **What this note is:** the failed fit, the probes that located the cause, and what changed. **Nothing here is a fit of record.** The probe numbers come from short many-chain runs and show which designs have one mode, not what the coefficient is.

## 1. The fit

VG25 was fitted at `rep` for the first time on 2026-09-15, from `main` at `bd34ef5`, in pass 1 of the nine-model refit alongside VG15, VG20 and VG24. Those three passed the gate first time. VG25 did not:

|                  |                                                                                 |
| ---------------- | ------------------------------------------------------------------------------- |
| convergence gate | **REVIEW**: max R-hat **1.614**, min ESS **10**, 252 R-hat and 229 ESS failures |
| divergences      | 0 in every chain                                                                |
| energy BFMI      | 0.58–0.63 in every chain                                                        |
| wall time        | 63 min                                                                          |
| retained at      | `failed/VG25-age-joint-signspeech-ds-corr-signlag-20260915T170113Z/`            |

Zero divergences and a healthy BFMI with an R-hat of 1.6 is the signature of chains that each sampled well and disagreed. Per-chain means, from the retained trace:

|                 | chains 0, 2, 4, 5 | chains 1, 3 | VG24, refitted the same day |
| --------------- | ----------------: | ----------: | --------------------------: |
| `beta_sign_lag` |        **+0.685** |  **−0.501** |                           — |
| `tau_subj_sign` |             1.573 |       1.872 |                       1.186 |
| `rho_u_sign`    |            −0.024 |      +0.409 |                       0.264 |
| `rho_sign_q`    |             0.261 |       0.375 |                       0.388 |
| `psi`           |             2.921 |       2.984 |                       2.519 |

Within each group the chains agree to the second decimal, with within-chain SDs the same size in both groups. **The coefficient changes sign between the modes, and neither mode leaves VG24's child signing block where VG24 has it.** Heavier tuning would not help: each chain was mixing well inside its own mode.

## 2. The mechanism

VG25 was registered with the lag in the cross-tab compositions as well as the spoken marginal. [202609111704](202609111704-vg25-registration-and-the-boundary-share.md) §3 argued for that: VG15 keeps its child shifts out of the four-cell and produced-cell Dirichlet-Multinomials because a free per-child offset is co-identified with `psi` there, but `beta_sign_lag` is one scalar multiplying a covariate fixed by the data, with no per-child freedom to chase a composition with. The same section recorded the caveat that under the within-child baseline an estimated per-child quantity does reach the composition, through one coefficient, on lagged rows.

The caveat turned out to be the whole problem. The within-child predictor is

```
x = has_lag * (logit(signed / understood) at the prior wave  -  g at the prior wave)
```

where `g` includes the child's estimated signing intercept. So `beta_sign_lag * x` puts `-beta_sign_lag * (child signing intercept)` into the production-ratio logit of every lagged row. In the spoken marginal, the child's own `q` intercept, correlated with the signing intercept through `rho_sign_q`, sits beside it. In the compositions it does not, because the compositions are built on population-and-study marginals by design. There the lag is the only route by which a child's signing standing reaches `q`, and the child block and the coefficient can trade against each other two ways. The two modes are those two trades.

## 3. The probes

To separate the baseline from the scope, `scripts/experiments/vg25_sign_lag_modes.py` builds the joint engine's real graph on the real frame for each of the four combinations and samples it with nutpie: **12 chains**, 1,500 tuning and 500 draws, `target_accept` 0.95, seed 20260915. Sampling took 400–470 s per design, two designs at a time. Chains are grouped by the sign of their mean `beta_sign_lag`:

| design                            | chains per mode | `beta_sign_lag` | `tau_subj_sign` |    `rho_u_sign` |  `rho_sign_q` |       `psi` |
| --------------------------------- | --------------- | --------------: | --------------: | --------------: | ------------: | ----------: |
| within, in the cells (registered) | **8 + 4**       | +0.687 / −0.502 |   1.570 / 1.878 | −0.029 / +0.414 | 0.265 / 0.368 | 2.92 / 2.99 |
| population, in the cells          | 12              |          +0.144 |           1.187 |          +0.264 |         0.319 |        2.48 |
| **within, spoken marginal only**  | 12              |          +0.440 |           1.119 |          +0.263 |         0.336 |        2.52 |
| population, spoken marginal only  | 12              |          +0.368 |           1.196 |          +0.258 |         0.219 |        2.52 |
| _VG24, `rep` fit of record_       | —               |               — |         _1.186_ |        _+0.264_ |       _0.388_ |      _2.52_ |

All 48 chains have zero divergences. Across chains within one mode, `beta_sign_lag` spans no more than 0.02.

Three things follow.

- **The probes reproduce the fit.** The registered design splits again, a third of the chains in the negative mode, at the same values to two decimals. The two modes' mean log densities are −20,649.6 and −20,654.8, so this is a genuinely bimodal posterior and not a minor side mode that sampling happened to find.
- **Either change alone removes the second mode.** Moving to the population baseline makes the predictor a fixed function of the data and population parameters. Leaving the lag out of the cells removes the only route by which the child's signing intercept reached `q` unaccompanied by the child's own `q` intercept.
- **The within-child baseline on the spoken marginal is the design that leaves VG24's model alone.** `rho_u_sign` and `psi` sit at VG24's values; `tau_subj_sign` is 6% lower and `rho_sign_q` 13% lower. The population baseline in the cells leaves the scale and `psi` alone but takes `rho_sign_q` to 0.32. On the marginal it takes `rho_sign_q` to 0.22, which is the "second, noisier reading of `rho_sign_q`" the registration predicted for that baseline.

The `beta_sign_lag` values are **not results**. They come from 500 draws per chain, and the coefficient moves by a factor of three across designs that differ in scope alone (+0.14 against +0.37 under the population baseline). They are recorded to show that each design has one mode and where it sits, not what VG25 estimates.

## 4. What changed

The study owner chose the within-child baseline on the spoken marginal. The alternative was the population baseline in the cells, which keeps `uk_07`'s evidence but changes the estimand the model was registered for.

- **Definition.** `VG25.sign_lag_in_cells` is `False`, and the field's class default is `False`, with the measurements on its docstring. The baseline stays `within`, so the banner and the three-quantities framing on the model page are unchanged.
- **Support.** On the 2026-09-15 frame the lag rests on **110 supporting observations from 79 children** in four studies (ie_02 42, uk_05 30, uk_04 25, uk_02 13), against 190 from 128 with the compositions. `uk_07` contributes nothing, because its rows carry no spoken marginal. The boundary share is larger on the smaller support: 27 of 110 rows (24.5%) sit at a signed share of exactly 0 or 1, and carry 50.6% of the source logit's sum of squares under the continuity correction, against 45.8% in the cells scope.
- **Sensitivity arms.** There are still ten.
  - `sign-lag-marginal-only` would now be the headline under another name. It is replaced by **`sign-lag-in-cells`**, the population baseline in the compositions, the one in-cells combination with one mode.
  - **`no-uk07` is replaced by `no-uk05`**, because `uk_07` no longer supports the lag. The leave-one-study-out pair is `no-ie02` (support 68 from 37 children) and `no-uk05` (80 from 64).
  - `sign-lag-uk07-marginal` now brings `uk_07`'s children into the support (162 from 106) rather than moving them between branches.
  - `sign-lag-population` now sits on the marginal scope.
  - None of the arms has been probed except the two designs in §3.
- **Graph tests.** The graph-property tests now pin the lag out of `cells_obs` on the headline and in it on the in-cells variants. The recovery spec's consumer list for VG25 is `("y_s_obs",)`, which it derives from the flag. The wave-sequential simulation loop is still selected, because the source is drawn in the same stage as the consumer.
- **Leave-one-out suppression.** It is unchanged, as its own comment already said: the source wave's counts reach later rows through the predictor whichever likelihoods it enters.

## 5. Consequences for the refit

The change is in `src/vocab_growth/`, so it moves the executable-code signature. The fits pass 1 produced from `bd34ef5` (VG15, VG20 and VG24) are therefore superseded for publication and have to be refitted. Pass 2 was stopped during VG12's sampling, before it replaced anything. The refit restarts once this change is on `main`, with the same nine models.

## 6. Not established, and a correction

- **The probes are not `rep` fits.** Twelve chains are twice what `rep` runs, so a second mode with a basin the size of this one would have been hard to miss, but a small basin could still escape. The restarted refit's VG25 fit, at six chains and 12,000 iterations each, is the check.
- **A correction to the registration record.** [202609111704](202609111704-vg25-registration-and-the-boundary-share.md) §3 took the cells decision on the premise that the predictor is a fixed covariate. That premise holds for the population baseline only, and §3 recorded the within-child exception as a caveat rather than as a reason. A flagged block at the head of that note points here. The note is otherwise unchanged.
- **VG16 is not implicated by this.** VG16's own within-child arm gave `beta` −0.60 at `dev` and +0.10 at `test`, which [202608151500](202608151500-within-child-crosslag-feasibility.md) §7.1 attributes to `dev`-tier non-convergence. The sign flip is suggestive, but VG16 has no cross-tab compositions. The mechanism in §2 needs a likelihood in which the child's signing intercept reaches `q` without the child's own `q` intercept, and VG16's bivariate engine has none. Nothing here tests that attribution either way.
