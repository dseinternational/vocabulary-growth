# Is VG22 the better description? Yes of the structure, no of the magnitude

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-09. **Answers:** the question [`202609091200-vg20-vg22-gate-resolved.md`](202609091200-vg20-vg22-gate-resolved.md) left explicitly open — _"Whether VG22 is the better description of these children is not answered here and is not what the gate asked."_ **Supersedes on the refitted pool:** [`202609031415-vg20-vs-vg22.md`](202609031415-vg20-vs-vg22.md), whose numbers all predate the `us_03` ingestion and which flagged that the comparison "could move materially". **Fits read:** the `rep` refits of VG20 and VG22 from the refit that completed on 2026-09-08 (created 2026-09-07T23:58Z and 2026-09-08T00:30Z), the `rank-2` sensitivity arm, the criterion-1 recovery replicates, and the two `kfold_loso.py` runs of 2026-09-09.

## Headline

**VG22 is the better description of the phenomenon and cannot estimate it.** Every structural claim the 2026-09-03 comparison made survives the ingestion of 104 repeat-visit children and gets slightly stronger. And the one parameter that carries those claims — `tau_subj_q_1`, the between-child spread of the production-ratio rate — is not identified at this data size: three recovery replicates return it 24–34% too small, each time with an 89% interval that excludes the value it was simulated from.

Those two findings are not in tension, and separating them is the point of this note. **A model can be right about what varies and wrong about how much**, and VG22 is. The gate asked which model should carry the reported numbers and answered VG20. This asks which model is true of the children, and the answer is that VG22's _qualitative_ structure is supported while its _magnitudes_ are not reportable from this pool.

## What reproduced on the refitted pool

The child-level comparison, refreshed on the refitted fits. Both models are `rep`, on the byte-identical frame `sha256:b27f64ea0640…`, 1,708 rows and 943 children.

| quantity                         |         VG20 (refit) |         VG22 (refit) | pre-refit 20/22 |
| -------------------------------- | -------------------: | -------------------: | --------------- |
| level–level correlation `rho_uq` | 0.433 [0.355, 0.507] | 0.340 [0.243, 0.434] | 0.390 / 0.311   |
| comprehension **level** spread   | 0.822 [0.776, 0.870] | 0.808 [0.761, 0.857] | 0.781 / 0.757   |
| ratio **level** spread           | 1.361 [1.275, 1.451] | 1.259 [1.173, 1.348] | 1.348 / 1.251   |
| comprehension **rate** spread    |     not in the model | 0.068 [0.015, 0.121] | — / 0.094       |
| ratio **rate** spread            |     not in the model | 0.730 [0.632, 0.830] | — / 0.718       |
| between-**study** comprehension  | 0.394 [0.267, 0.559] | 0.406 [0.276, 0.575] | 0.360 / 0.380   |
| between-**study** ratio          | 0.648 [0.453, 0.895] | 0.637 [0.442, 0.883] | 0.638 / 0.635   |

Three things carry over unchanged. **VG20's correlation is inflated by variance it cannot express** — `rho_uq` falls from 0.433 to 0.340 when the rates are given somewhere to live, a 21% reduction against 20% before the refit. **The level spreads shrink the same way**, most visibly the ratio level, 1.361 to 1.259. And **the ratio rate spread excludes zero decisively**, 0.730 [0.632, 0.830]: children with Down syndrome do differ in how fast their production share moves, and VG20 has no parameter for it.

The implied 4×4 correlation at the 36-month reference age (`subject_factor_corr.csv`) reproduces the headline result:

| pair                                     | refitted                 | pre-refit              |
| ---------------------------------------- | ------------------------ | ---------------------- |
| **comprehension level ↔ ratio rate**     | **0.402 [0.266, 0.531]** | 0.393 [0.265, 0.518]   |
| ratio level ↔ ratio rate                 | 0.389 [0.264, 0.507]     | 0.385 [0.260, 0.503]   |
| comprehension level ↔ ratio level        | 0.340 [0.243, 0.434]     | 0.311 [0.218, 0.400]   |
| comprehension rate ↔ ratio rate          | 0.536 [-0.124, 0.913]    | 0.625 [0.209, 0.894]   |
| comprehension rate ↔ ratio level         | 0.416 [-0.246, 0.884]    | 0.448 [-0.021, 0.842]  |
| comprehension level ↔ comprehension rate | -0.228 [-0.683, 0.267]   | -0.249 [-0.560, 0.094] |

**A child's comprehension standing predicts their conversion _rate_ (0.402) at least as strongly as their conversion _level_ (0.340)**, on the same fit, under the same reference age, with both intervals excluding zero — and VG20 estimates only the second. Two qualifications the 2026-09-03 note did not make. The ordering is a point-estimate one: the two intervals overlap heavily, and the gap between them narrowed from 0.082 to 0.062 on the refit, so "more strongly" is not a distinguishable claim. And the magnitude is rank-dependent in a way `rho_uq` is not. Read off the rank-2 arm's trace, the same correlation is **0.860 [0.732, 0.956]** against rank 3's 0.402, while `rho_uq` is 0.349 against 0.340. At rank 2 the rate direction is forced into the span of the two level directions, so its correlations with them are a constraint of the parameterisation rather than a measurement — [`202608231420`](202608231420-vg22-factor-anchor-bimodality.md) §5's reading of why the rank-2 scale is the compromised one. What survives both ranks is the sign: the coupling is positive, and VG20 has no parameter for it.

One thing did move, and against VG22. **The comprehension rate is fading**: its spread fell from 0.094 [0.042, 0.147] to 0.068 [0.015, 0.121] on 104 more children, and all three of its correlations widened, the one that previously excluded zero (comprehension rate ↔ ratio rate, 0.625 [0.209, 0.894]) now spanning it. More data made that effect smaller and less identified, which is the signature of an effect that is not there. VG22's four-effect structure is behaving like a three-effect one.

## The predictive evidence, with its caveat

Holding out whole children and refitting per fold — the comparison PSIS-LOO could not give, because it degenerated across this whole family — favours VG22:

| comparison                          | VG22 − VG20 | paired SE |  SEs |
| ----------------------------------- | ----------: | --------: | ---: |
| whole-child holdout, 943 children   |      +65.56 |     19.78 | 3.31 |
| later-visit holdout, 425 children   |      +33.61 |      9.85 | 3.41 |
| later-visit **comprehension alone** |       +0.63 |      2.05 | 0.31 |

**This must not be read as a model comparison**, for the reason [`202609091200`](202609091200-vg20-vg22-gate-resolved.md) records: 9 of 10 and 8 of 10 fold fits miss R-hat 1.01 or ESS 400 at `test` tier, and `kfold_loso.py` flags both runs `all_folds_converged=False`. Taken as a direction rather than a magnitude, it says VG22 predicts a held-out child better, and that the one dimension where VG20 previously led — later comprehension — is now a tie.

So VG22 predicts these children better while being unable to estimate the parameter it predicts them better _with_. That is coherent: some rate flexibility helps prediction even when its scale is mis-estimated. It does not convert into a reportable number.

## Where it fails: criterion 1

Parameter recovery at `rep`, three replicates, truth drawn from the model of record's own posterior, simulated data of exactly the real size and design (1,708 rows, 943 children). All three replicates are scored by the harness, all three cleared its convergence assessment, and pooled coverage is **0.834** against a nominal 0.89. `tau_subj_q_1` is in the `not recovered` list of all three.

| quantity                      | rep |  truth | recovered |     sd |         z |       bias | covered? |
| ----------------------------- | --- | -----: | --------: | -----: | --------: | ---------: | -------- |
| `tau_subj_q_0` ratio level    | r01 | 1.2811 |    1.2921 | 0.0544 |     +0.20 |      +0.9% | yes      |
|                               | r02 | 1.2657 |    1.2453 | 0.0529 |     −0.38 |      −1.6% | yes      |
|                               | r03 | 1.2324 |    1.2485 | 0.0521 |     +0.31 |      +1.3% | yes      |
| `tau_subj_u_0` compr. level   | r01 | 0.7727 |    0.7478 | 0.0293 |     −0.85 |      −3.2% | yes      |
|                               | r02 | 0.8785 |    0.8053 | 0.0272 |     −2.69 |      −8.3% | **no**   |
|                               | r03 | 0.8439 |    0.7900 | 0.0287 |     −1.88 |      −6.4% | **no**   |
| `tau_subj_u_1` compr. rate    | r01 | 0.0962 |    0.1283 | 0.0293 |     +1.10 |     +33.4% | yes      |
|                               | r02 | 0.0606 |    0.0643 | 0.0303 |     +0.12 |      +6.2% | yes      |
|                               | r03 | 0.0936 |    0.0776 | 0.0412 |     −0.39 |     −17.1% | yes      |
| **`tau_subj_q_1` ratio rate** | r01 | 0.7634 |    0.5807 | 0.0544 | **−3.36** | **−23.9%** | **no**   |
|                               | r02 | 0.7216 |    0.4748 | 0.0571 | **−4.33** | **−34.2%** | **no**   |
|                               | r03 | 0.6785 |    0.5182 | 0.0537 | **−2.98** | **−23.6%** | **no**   |
| `rho_uq`                      | r01 | 0.3649 |    0.2516 | 0.0578 |     −1.96 |     −31.1% | **no**   |
|                               | r02 | 0.3378 |    0.3373 | 0.0533 |     −0.01 |      −0.1% | yes      |
|                               | r03 | 0.2958 |    0.3067 | 0.0578 |     +0.19 |      +3.7% | yes      |

**The pattern is exact: VG22 recovers what it shares with VG20 and fails on what distinguishes it.** The ratio level scale comes back within 1.6% every time. The ratio _rate_ scale — the parameter the whole low-rank factor exists to deliver — misses all three times, always downward, by an average of 27%, with a posterior standard deviation of about 0.055 that puts the truth three to four standard deviations away. This is not a wide posterior admitting ignorance; **it is a narrow posterior in the wrong place.**

The per-child effects say the same thing from the other side. Recovered per-child production _rates_ have **41–44% of the spread** of the ones they were simulated from (0.326 against 0.750, 0.295 against 0.714, 0.305 against 0.687) and their interval coverage falls to 0.81, 0.67 and 0.77 against a nominal 0.89 — while the comprehension and ratio _levels_ hold coverage at 0.83–0.90 throughout. The rates are being shrunk towards zero child by child, which is the same finding as the scale coming back too small.

`tau_subj_u_1` recovers, but it is the wrong kind of success: the truths are 0.06–0.10 against a posterior sd of 0.03–0.04, so the interval covers by being wide relative to a near-zero quantity. It is consistent with that effect not being there.

Where the missing rate variance goes is an interpretation, but the recovery output points one way consistently. In all three replicates the spoken dispersion is recovered _lower_ than it was simulated — more overdispersion — somewhere on the age range: `kappa_young_s` 121 → 65 (z −4.8) and `kappa_excess_young_s` 89 → 36 (z −4.2) in r01, `kappa_young_s` 45 → 31 (z −3.3) in r02, `kappa_min_s` and `kappa_old_s` at z −1.7 and −2.0 in r03. Between-child variation in the production rate and extra Beta-Binomial spread in production counts are the two places the model can put the same variance, and for a child seen once they are not separable. That is consistent with the design reading below; it is not a demonstration of it.

### The correlations recover; the scale does not

The harness scores no element of `subject_factor_corr`, so the implied correlations were read from the recovery traces directly, with each replicate's truth rebuilt from its sampled loadings by the model's own construction (checked against the record trace's stored deterministic: maximum discrepancy 4e-16, and it reproduces the harness's `rho_uq` rows exactly).

| pair                             | rep |  truth | recovered |    sd |     z | covered? |
| -------------------------------- | --- | -----: | --------: | ----: | ----: | -------- |
| comprehension level ↔ ratio rate | r01 | +0.528 |    +0.448 | 0.102 | −0.79 | yes      |
|                                  | r02 | +0.434 |    +0.552 | 0.112 | +1.06 | yes      |
|                                  | r03 | +0.425 |    +0.453 | 0.113 | +0.25 | yes      |
| ratio level ↔ ratio rate         | r01 | +0.376 |    +0.342 | 0.085 | −0.40 | yes      |
|                                  | r02 | +0.464 |    +0.495 | 0.099 | +0.31 | yes      |
|                                  | r03 | +0.506 |    +0.504 | 0.088 | −0.03 | yes      |

Every one covered, with posteriors honestly wide (sd about 0.1) where `tau_subj_q_1`'s was narrow and wrong. **The factor's shape is identified and its size along the rate dimension is not.** That is a sharper statement than "the rate block fails": the data can say which way the rate effect points relative to the levels, and cannot say how large it is.

### It is not the prior

`tau_subj_q_1 ~ HalfNormal(0.5)`, so a truth of 0.76 sits 1.5 prior scales out and prior shrinkage is the obvious innocent explanation. It does not account for the displacement. Comparing log-densities at the truth against at the posterior mean, the prior prefers the posterior mean by 0.49 (r01) and 0.59 (r02) log-density units; the posterior prefers it by 5.64 and 9.36. **The prior accounts for 9% and 6% of the gap.** The rest is the likelihood.

### It is not a rotation artefact

The rotational freedom of a factor model — `L` and `LQ` give the same covariance for any orthogonal `Q` — is pinned in this model by the triangular anchor constraint, so it is not in play. What is unidentified is narrower: each normalise-a-Normal row of the loading matrix carries one inert radial magnitude, which `build_child_factor`'s own comment names as the price of avoiding a wrapped coordinate, and `subject_factor_w_32` — the diagonal entry of the production-rate row — is such an entry. Its appearing outside its interval in r02 is expected and says nothing. `tau_subj_q_1` is a different kind of quantity: the factor is built so that `Sigma_ii = tau[i]**2` exactly, which makes each `tau` the marginal between-child standard deviation of its effect however the loadings are parameterised. The comparison is like for like.

## The two failures are the same failure

Criterion 2 and criterion 1 are independent tests that land on the same parameter and point the same way.

|                                     | `tau_subj_q_1`   |
| ----------------------------------- | ---------------- |
| fit of record, rank 3               | 0.730 (sd 0.062) |
| the same data at rank 2             | 0.407 (sd 0.050) |
| recovery: simulated 0.76 → returned | 0.581            |
| recovery: simulated 0.72 → returned | 0.475            |
| recovery: simulated 0.68 → returned | 0.518            |

The rank comparison says two defensible parameterisations of the same data disagree about this number by 4.06 combined standard errors. Recovery says that even when the number is _known_, the fit does not return it. And the recovered values, 0.47 to 0.58, sit between the rank-2 and rank-3 answers.

**Read together they say the rank disagreement is not two candidate answers one of which is right — it is a ridge the data cannot resolve.** Choosing rank 3 and reporting 0.730 would state a number that this model returns as 0.52 when 0.68 is true.

## Why: it is a design limit, not a modelling fault

A per-year rate on a child's production ratio is identified, in practice, by children seen at two separated ages with production measurable at both. A single-visit child contributes only through the way between-child spread changes with age — a signal the model's age-varying dispersion competes for directly. The pool does not have many of the children who count.

| the pool                                       | children | share of pool |
| ---------------------------------------------- | -------: | ------------: |
| total                                          |      943 |               |
| seen once — next to nothing for a rate         |      518 |         54.9% |
| repeat-measured                                |      425 |         45.1% |
| … with ≥ 6 months between first and last visit |      352 |         37.3% |
| … producing at two or more separate visits     |      272 |         28.8% |
| **… both: carrying real rate information**     |  **232** |     **24.6%** |

1,708 rows across 1,610 visits, median span between first and last visit 12 months. So `tau_subj_q_1` is being estimated from about **232 informative children**, most contributing a single rate contrast over a one-year window — and the recovery replicates show that is not enough. (The 425 matches `kfold_loso.py`'s `n_subjects` for the later-visit holdout exactly, which cross-checks the wave definition.)

**This is what would change the answer, and it is not a modelling change.** More children seen once will not help, and the last ingestion is the measurement of that: `us_03` added 104 children with two visits, and `tau_subj_q_1` moved from 0.718 on the fit of record immediately before it to 0.730 after. (The 0.576 quoted in the gate notes is from the 2026-08-23 fits, before the #281 refit; most of the movement between 0.576 and 0.730 predates the ingestion and is not the pool's doing.) What identifies a rate is longer follow-up on children already in the pool.

## What this means

**Reportable from VG22, as direction and sign:** that children differ in how fast their production share moves at all (`tau_subj_q_1` excludes zero at both ranks, 0.407 [0.328, 0.487] at rank 2); that the coupling between comprehension standing and conversion _rate_ is positive (positive at both ranks, and recovered within about one standard error in all three replicates); and that about a fifth of VG20's `rho_uq` is coupling VG22 attributes to the rates rather than the levels — a caveat on how a published number should be read rather than a new number, since VG20's correlation is of time-constant offsets and VG22's is of levels at 36 months, and the two coincide only if the rates are zero.

**Not reportable:** any magnitude from VG22's rate block — `tau_subj_q_1` itself, the per-child rates, the size of the level-to-rate correlation (0.40 at rank 3, 0.86 at rank 2), and anything involving the comprehension rate.

**Unchanged:** VG20 stays the model of record, decided on criterion 2 and not revisited here. Nothing in this note bears on a published population trajectory; the two models remain interchangeable at every reported age.

The honest summary for the report is that the richer structure is real and the study cannot yet measure it. That is a finding about the evidence base, not a defeat, and it is more useful to a reader than either "VG22 fits better" or "VG22 was rejected".

## What would reopen it

In order of what each decides.

1. **Longer follow-up**, not more children. A second wave on the 518 single-visit children, or a third on the 425, is the only thing that moves the identification. This belongs in the study's data-collection planning, not its modelling queue.
2. **A simulation study over follow-up designs** — recovery at `--truth prior` with synthetic visit structures — would say how many repeat visits, at what spacing, `tau_subj_q_1` needs. That is cheap relative to collecting them and would make the ask concrete.
3. **A rank the evidence can choose.** [`202609091200`](202609091200-vg20-vg22-gate-resolved.md) already says this; recovery now explains _why_ it cannot, which is that the quantity the ranks disagree about is the one the data do not identify.
4. **Converged fold fits** at `rep-lite` (~3 h) would let the k-fold advantage be read as a magnitude. It cannot change any conclusion here, and it is the least valuable of the four.

## Caveats

- Recovery coverage over three replicates is **indicative, not simulation-based calibration** — the harness says so in its own verdict column. Three replicates cannot estimate a coverage rate. What they can do, and did, is show the same parameter missing in the same direction three times, at |z| of 3.0 to 4.3, which is not a coverage argument.
- All three replicates cleared R-hat and ESS and failed the four-check gate on **divergences alone** — 6, 1 and 56 — which the harness counts as `converged: yes` with caveats, so criterion 1's "all three assessed" condition is met and the failure is in the recovery rather than in the assessment. r03's 56 is high enough to be worth naming; its R-hat is 1.006 and its minimum ESS 917, and its `tau_subj_q_1` miss is the mildest of the three, so the divergences are not what produced the result.
- The k-fold numbers are from fold fits that did not converge and are quoted as direction only.
- The correlation-recovery table is a direct reading of the traces, not a harness output. The reconstruction of each truth from its sampled loadings was checked against the record trace's stored `subject_factor_corr` on 2,000 draws and reproduces the harness's own `rho_uq` rows; the rank-2 correlations are read from that arm's trace the same way, because the sensitivity arm writes no `subject_factor_corr.csv`.
- VG20's own recovery evidence, cited in [`202609031415`](202609031415-vg20-vs-vg22.md) as the decisive asymmetry, was run **before** the `us_03` ingestion and its outputs are not on the current output root. The comparison "VG20 recovers and VG22 does not" is therefore not like-for-like on data version. It is not needed: VG22's failure is established against its own generating truth on its own current design, without reference to VG20.
- The comprehension-rate effect fading with more data is one refit's movement, not a trend. It is recorded because it points the same way as its correlations widening, not because either is decisive alone.
