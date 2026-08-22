# Gate 1 for the 4x4: the cross-outcome coupling is level-to-rate, not rate-to-rate

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Structure selection for the successor to VG19 and VG20, run 2026-08-22 before any model is written, by the method [202608141600](202608141600-rank-stability-tracking.md) §10.3 used for VG19's own Gate 1. **The headline is a negative that changes the design.** The element that motivated the 4x4 — `corr(b1u, b1q)`, do children who gain comprehension faster also convert faster — is **not supported**: 2Δ logL = 2.47 on 1 df across all children and **0.57** on the repeated-measures children. What the data does support is an asymmetric coupling nobody proposed: a child's comprehension **level** strongly predicts their production-ratio **rate**. Reproduced by `scripts/experiments/four_by_four_gate1.py`.

## 1. Why this had to be gated before it was built

[202608211500](202608211500-vg19-registration.md) §2 recorded that VG19 and VG20 are not composable and that their union is a 4x4 over `(b0u, b1u, b0q, b1q)` with six correlations, and §5 of [202608212000](202608212000-vg19-gates-g2-g4-g5.md) called that 4x4 the successor. Both described it as one object with six free elements. It is cheap to ask which of those six the data actually carries, and doing so before writing a definition class is the same discipline that produced VG19: Gate 1 chose the random slope over an AR(1) on the residuals, and the fitted model then agreed.

Method: age- and study-adjusted logit residuals, maximum likelihood against a **known** per-observation binomial sampling variance, one multivariate normal per child, no PyMC and no registered-model fit. The residual scale mirrors the models' own nested likelihood — `u` is scored against the administration's form ceiling and `q` against that same administration's understood count — so the two sampling variances are the ones the Beta-Binomial layer would use. 602 children and 970 paired administrations; 246 children and 614 administrations with repeats.

The two columns answer different questions and both are reported throughout, because a term that appears only in the all-children column is cross-sectional spread rather than within-child drift.

## 2. The fitted 4x4

|               | all children | repeats only |
| ------------- | -----------: | -----------: |
| SD `b0u`      |        0.982 |        0.903 |
| SD `b1u`      |        0.237 |    **0.079** |
| SD `b0q`      |        1.164 |        1.218 |
| SD `b1q`      |        0.424 |        0.564 |
| `b0u`,`b1u`   |       +0.555 |       +0.820 |
| `b0u`,`b0q`   |       +0.272 |       +0.297 |
| `b0u`,`b1q`   |   **+0.754** |   **+0.648** |
| `b1u`,`b0q`   |       +0.455 |       +0.710 |
| `b1u`,`b1q`   |       +0.370 |       +0.648 |
| `b0q`,`b1q`   |       +0.766 |       +0.724 |
| occasion SD u |        0.591 |        0.595 |
| occasion SD q |        0.704 |        0.704 |

`b1u` falls from 0.237 to **0.079** when the singletons are dropped, which independently replicates the original Gate 1's second conclusion: **comprehension's widening is cross-sectional, not drift.** `b1q` moves the other way (0.424 to 0.564), replicating the first: the production slope is genuine within-child drift. Two Gate 1 runs, two years of implementation apart in the plan's terms and on a differently constructed residual, agree on the asymmetry.

## 3. Which elements the data carries

`2 * delta logL` of the full 4x4 over each nested restriction. Positive favours the 4x4; `p` is the chi-square tail.

| restriction removed from the full model     | all children | p         | repeats only | p         |
| ------------------------------------------- | -----------: | --------- | -----------: | --------- |
| block-diagonal (**= VG19**, 4 df)           |        60.22 | 2.6e-12   |        26.79 | 2.2e-05   |
| intercepts only (**= VG20**, 5 df)          |       122.71 | —         |        54.16 | —         |
| `b0u`,`b1q` — u **level** -> q **rate**     |    **29.74** | 4.9e-08   |    **22.47** | 2.1e-06   |
| `b0u`,`b0q` — the `rho_uq` VG20 estimates   |        17.48 | 2.9e-05   |        11.69 | 6.3e-04   |
| `b1u`,`b0q` — u **rate** -> q **level**     |         4.76 | 0.029     |         1.15 | 0.284     |
| `b1u`,`b1q` — **rate -> rate**              |     **2.47** | **0.116** |     **0.57** | **0.450** |
| VG19 + `rho_uq` only, one cross term (3 df) |        59.64 | 7.0e-13   |        26.28 | 8.3e-06   |

The VG20 row is on a variance boundary and its `p` is not reported for that reason; it is listed to show the magnitude.

Four things follow.

1. **The rate-to-rate correlation is not there.** This is the element that made the 4x4 interesting, and it is the weakest of the six on both columns — weakest of all on repeats only, where a within-child rate coupling would have to show if it existed. The apparent +0.370 is what a correlation matrix produces when the other four cross terms are doing the work.
2. **The strongest cross-outcome term is `b0u` -> `b1q`**, and it is stronger than `rho_uq` by a factor of nearly two on both columns and survives restriction to repeats. **A child's comprehension standing predicts how fast they convert comprehension into speech.** That is a more useful clinical statement than the one the 4x4 was proposed to test, and it is not estimated by VG19, VG20 or VG16.
3. **`rho_uq` is confirmed independently.** VG20's parameter is supported at p = 2.9e-05 on a method that shares nothing with the fitted model but the data. That is worth having on the record given that §4c of [202608212000](202608212000-vg19-gates-g2-g4-g5.md) found it buys almost nothing in held-out prediction.
4. **VG19 + `rho_uq` is not enough.** Adding only the intercept-intercept term to VG19's block structure leaves 59.64 on 3 df on the table, almost all of it the `b0u` -> `b1q` coupling.

## 4. What to build

Not the symmetric 4x4. The data supports **four SDs and three correlations** — the two within-outcome terms VG19 already has, plus `b0u`,`b0q` and `b0u`,`b1q` — with `b1u`,`b1q` and `b1u`,`b0q` fixed at zero. Seven covariance parameters rather than ten, and the two dropped are the two the repeats column cannot see.

That structure is not a Cholesky of a free correlation matrix, so it needs its own parameterisation with a positive-definiteness constraint, which is a real implementation cost and should be weighed against simply fitting the full 4x4 and reporting two of its six correlations as null. Given that `b1u` itself is largely cross-sectional (§2), a third option is worth pricing: drop the comprehension rate entirely and fit a 3x3 over `(b0u, b0q, b1q)`, which is exactly the reduced structure above with `b1u` removed and needs no constrained parameterisation at all.

## 5. Caveats

1. **Residual ML is not the model.** The residuals are adjusted by a cubic in age plus study fixed effects, not by the fitted HSGP, and the child structure is Gaussian on the logit rather than Beta-Binomial. Gate 1 is a structure-selection instrument, and the standard it has to meet is the one it met for VG19: the fitted model agreed with it.
2. **The correlation matrix is highly interdependent.** `b0u`,`b1q` = +0.754 sits alongside `b0q`,`b1q` = +0.766, so the level-to-rate term and the within-outcome term are partly competing to explain the same covariance. The likelihood-ratio tests are the right reading of the table and the individual correlations are not.
3. **`q` is undefined where a child understands nothing**, so 602 of the pool's 767 children enter, and the youngest are the ones most likely to be excluded.
4. **No multiplicity adjustment.** Six one-df tests are reported; the two that matter are at 1e-8 and 3e-5 and the two nulls are at 0.12 and 0.45, so nothing here is near a threshold where that would change the reading.
