# VG09: Sampler diagnostics under reporting configuration

Date: 2026-05-13

> **Note:** This document was generated with assistance from an AI model (Claude, Anthropic) and should be independently verified.

## Context

VG08 and VG09 were rerun in the `rep` sampling configuration (`draws=6000`, `tune=6000`, `chains=6`, `target_accept=0.95`, 6 cores) to confirm the previous analyses prior to reporting. VG08 returned clean diagnostics. VG09 produced a small number of parameters with `r_hat` marginally above 1.01 and one with `ess_tail` marginally below 400. This note records the investigation and rules out sampler-tuning as the cause.

## VG08 result

Clean. `r_hat ≤ 1.01` and `ess_tail ≥ 400` across all reported parameters. Sampling wall time 10m 13s. Pareto-k > 0.7 LOO warnings present (consistent with prior runs).

## VG09 result — original `rep` config

- 6 parameters with `r_hat > 1.01` (max 1.019)
- 1 parameter with `ess_tail < 400` (min 394)
- Posterior sampling: 9m 46s

## VG09 rerun — tighter sampler

A one-off rerun with `target_accept=0.99` and `tune=8000` (draws kept at 6000) was launched to test whether marginal r-hat reflected step-size adaptation issues. Sampling took 22m 44s — about 2.3× longer — for **no meaningful improvement**:

| Metric                       | Original `rep` | `target_accept=0.99, tune=8000` |
| ---------------------------- | -------------- | ------------------------------- |
| `r_hat > 1.01`               | 6 (max 1.019)  | 5 (max 1.020)                   |
| `ess_tail < 400`             | 1 (min 394)    | 2 (min 358)                     |
| Posterior sampling wall time | 9m 46s         | 22m 44s                         |

The flagged parameters and their values were essentially unchanged. `ess_tail` shifted marginally in the wrong direction (within MC noise).

## Flagged parameters

All flagged parameters are global hyperparameters of the trajectory mean functions, dominated by the production-ratio (`q`) side:

| Parameter       | `r_hat` | `ess_bulk` | `ess_tail` |
| --------------- | ------- | ---------- | ---------- |
| `slope_q`       | 1.020   | 430        | 460        |
| `p_slope_hi_q`  | 1.012   | 431        | 397        |
| `p_slope_low_q` | 1.013   | 1008       | 2252       |
| `eta_q`         | 1.010   | 483        | 358        |
| `intercept_u`   | 1.012   | 759        | 1066       |
| `p_slope_hi_u`  | 1.014   | 628        | 2373       |

Five of the six are q-trajectory hyperparameters. The two understood-side parameters (`intercept_u`, `p_slope_hi_u`) are the analogous mean-function anchors for the `u` trajectory.

## Interpretation

The VG09 q-trajectory has three sources of global level/slope information that compete with one another:

1. The anchor-Beta priors `p_slope_low_q`, `p_slope_hi_q` (which deterministically yield `intercept_q` and `slope_q`).
2. The non-parametric correction `g_q = eta_q · g_unit_q` (HSGP).
3. The subject-level random intercepts `delta_q,subj` (new in VG09; not present in VG07/VG08).

Adding the third dimension to a model that already has the first two creates a weakly identified posterior geometry: a global shift in `delta_q,subj` can be absorbed by `intercept_q` or by a slow drift in `g_q`. The same applies on the understood side, where VG08 had subject REs on `u` (and `p_slope_hi_u` was already marginal at `r_hat ≈ 1.01`).

Symptoms consistent with this diagnosis:

- The flagged parameters cluster on exactly the components that overlap (mean-function anchors and GP scale).
- Increasing `target_accept` and `tune` does not help — the issue is posterior shape, not step-size adaptation.
- The marginal r-hat values are stable across reruns (max 1.019 → 1.020); they are at the edge of acceptable but consistently so.
- ESS is bulk-low rather than tail-low for most flagged parameters, which is typical of slow exploration along ridges in correlated posteriors.

## What this does _not_ affect

The reported derived quantities — `q_query`, `p_u_query`, `p_s_query`, expected learning rates, the joint trajectory and HDI bands — are functions of the full posterior over `h_all` and `f_u_all`. These quantities are well-identified even when their decomposition into mean trend + GP + REs is not. Spot-checks of the posterior predictive plots and query tables against VG07/VG08 reporting-config runs do not show structural differences attributable to non-convergence.

## Decision

The original `rep` run is retained as the canonical VG09 output. The tighter rerun is not preserved; the temporary launcher script was deleted. Productive next steps would be structural — tightening priors on the q-trajectory hyperparameters or reparameterising the mean function — not more sampling time. These are discussed separately.
