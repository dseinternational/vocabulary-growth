# Gate 1 for the 4x4: the coupling is level-to-rate, and the covariance is not four-dimensional

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Structure selection for the successor to VG19 and VG20, run 2026-08-22 before any model is written, by the method [202608141600](202608141600-rank-stability-tracking.md) §10.3 used for VG19's own Gate 1. **Two negatives, and the second is the bigger one.** The element that motivated the 4x4 — `corr(b1u, b1q)`, do children who gain comprehension faster also convert faster — is **not supported** (§3). And the 4x4 itself is **not identified**: its maximum-likelihood correlation matrix is singular, a rank-3 fit reaches the identical likelihood, and rank 2 costs only 2.60 on 2 df, so the successor should be a low-rank factor form rather than a correlation matrix of any shape (§4-5). What the data does support among the cross terms is an asymmetric coupling nobody proposed: a child's comprehension **level** strongly predicts their production-ratio **rate**. Reproduced by `scripts/experiments/four_by_four_gate1.py`.

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

## 4. The 4x4 is not identified by these data

Everything in §3 was computed at a maximum-likelihood optimum whose **correlation matrix is singular**. The smallest eigenvalue is 2.1e-08 across all children and 8.5e-07 on the repeats — numerically zero, on both subsets. The MLE sits exactly on the positive-definiteness boundary, and one linear combination of the four child effects has no variance at all.

This is not an artefact of the tanh-per-entry parameterisation walking into its own rejection region. Refitting `Sigma = L L'` with `L` of shape `(4, rank)` — unconstrained, always positive semi-definite, sharing no coordinates with the other search — reaches the identical optimum:

| rank | free covariance params | negll, all children | negll, repeats only |
| ---: | ---------------------: | ------------------: | ------------------: |
|    1 |                      4 |           1437.3791 |            820.3019 |
|    2 |                      7 |           1326.7271 |            715.3349 |
|    3 |                      9 |       **1325.4277** |        **714.6210** |
|    4 |                     10 |       **1325.4277** |        **714.6210** |

**Rank 3 and rank 4 are the same fit to four decimal places.** The tenth covariance parameter buys nothing. Rank 1 is decisively rejected (2Δ logL = 221 on 3 df), but **rank 2 costs only 2.60 on 2 df across all children and 1.43 on the repeats** — so the data cannot distinguish a two-dimensional child covariance from a four-dimensional one.

At rank 2, all children, the standardised loadings are:

| factor | `b0u` | `b1u` | `b0q` | `b1q` | reading                               |
| ------ | ----: | ----: | ----: | ----: | ------------------------------------- |
| 1      | +0.83 | +0.97 | +0.74 | +0.99 | a general child language factor       |
| 2      | −0.56 | −0.26 | +0.67 | +0.10 | comprehension level against `q` level |

Factor rotations are not unique and the second factor's reading should not be leaned on; the **dimension count** is what is invariant and what matters here.

### What this does to §3

The likelihood comparisons in §3 stand as likelihood comparisons — they survived a two-way warm start, in which projecting the full solution down into each restricted space and each restricted solution up into the full space improved the full model by 0.00 in every case, so neither side is under-optimised. **The `p` values do not stand.** A chi-square reference requires the MLE to be interior, and it is not. Read the §3 column as an ordering — `b0u`->`b1q` (29.74) far ahead of `rho_uq` (17.48), `b1u`->`b0q` (4.76) weak and gone on the repeats, rate-to-rate (2.47, 0.57) smallest of the six and smallest of all where drift would have to show — and not as four calibrated tests. The two conclusions that matter are robust to any reference distribution: 29.74 is large under all of them and 0.57 is small under all of them.

## 5. What to build

**Not a 4x4 with a free correlation matrix, and not the constrained correlation matrix an earlier draft of this note recommended.** Both fit ten or seven covariance parameters to data that maximise their likelihood at nine and are indifferent between nine and seven. A constrained correlation matrix also needs a positive-definiteness constraint hand-written into the graph, which is exactly the awkward part.

The structure the evidence points to is a **low-rank factor form**: `b = L z`, with `z ~ Normal(0, 1)` of dimension 1 or 2 and `L` a free `(4, k)` loading matrix. That is unconstrained, positive semi-definite by construction, needs no boundary handling, and at `k = 2` costs seven covariance parameters against the free 4x4's ten while losing 2.60 in log-likelihood. It also expresses the substantive claim directly — that a child's four effects are driven by one or two underlying dimensions rather than by four separately-varying quantities — and a general-factor-plus-contrast reading is a more natural object for the report than six pairwise correlations.

Two caveats before this is treated as a specification.

**Residual ML sits on the boundary; a Bayesian fit will not.** An LKJ or a factor prior regularises away from singularity, so a fitted 4x4 in PyMC would return a proper posterior rather than a rank-3 point. The finding is therefore not "the fourth dimension is exactly zero" but "**the data carry almost no information about it, so the prior would supply nearly all of it**" — which is the more useful warning, and one that a posterior alone would not make obvious.

**`k` is a choice this analysis cannot make.** Rank 2 and rank 3 are 2.60 apart on 2 df and rank 3 and rank 4 are identical, so 2 and 3 are both defensible and 4 is not. Registering the factor model with `k` as a definition field, and fitting `k = 1, 2, 3` as a registered sensitivity family, is the honest way to settle it — and it is cheap, because the three differ by one column of `L`.

## 6. Caveats

1. **Residual ML is not the model.** The residuals are adjusted by a cubic in age plus study fixed effects, not by the fitted HSGP, and the child structure is Gaussian on the logit rather than Beta-Binomial. Gate 1 is a structure-selection instrument, and the standard it has to meet is the one it met for VG19: the fitted model agreed with it.
2. **The `p` values in §3 are not calibrated**, because the MLE is on the positive-definiteness boundary (§4). They are retained in the table because the ordering is the point and the two conclusions that matter are far from any threshold in either direction, but they should not be quoted as tests. No multiplicity adjustment is applied either, and for the same reason it would not change the reading.
3. **The individual correlations are not interpretable.** `b0u`,`b1q` = +0.754 sits alongside `b0q`,`b1q` = +0.766 on a matrix that is singular, so the level-to-rate term and the within-outcome term are partly competing to explain the same covariance and the entries are not separately estimable. The likelihood-ratio ordering and the rank analysis are the readings this fit supports.
4. **`q` is undefined where a child understands nothing**, so 602 of the pool's 767 children enter, on 970 paired administrations. The 14 rows dropped for `understood = 0` and the excluded children skew young, which is also where `q` is least informative for a second reason: **15.2% of the retained rows have `q` exactly 0** and sit against the half-item clip, where a Gaussian-on-logit approximation is at its weakest. The corresponding ceiling problem is much smaller — 1.2% at `q` exactly 1.
5. **Nine rows have `spoken > understood`** and are clipped to `q = 1` by this harness. That is 0.9% of its paired set, too few to move anything here. It is **not** an unhandled defect in the pipeline, and this note should not be read as reporting one: `nested_outcome_spec` classifies such rows per observation, retains them through the **marginal** spoken likelihood rather than the conditional one, and counts them as `n_parent_violations`, which every joint model prints in its own report. There are 11 in the full frame of 1,428 spoken rows. Seven are `ie_01`, and those seven are the records whose previous `GREATEST(says_total, understands_total)` repair was deliberately removed — see the `ie_01` block in `data_utils.py`, which explains that repairing a count from the outcome being modelled is selection on the outcome. The concentration is real (`ie_01` 15.2% of its paired rows against 0.43% across every other study) and is a reason to treat that study's comprehension column with care, but the handling is documented policy rather than an oversight. The only thing this harness adds is that it **clips** rather than routing, which is the right simplification for a structure-selection instrument and the wrong one for a likelihood.
