> [!NOTE]
> Drafted or revised with assistance from Claude Code/Opus 5.5.

# A harness for the Gompertz mean-function comparison

**Date:** 2026-10-01. **Question:** issue [#330](https://github.com/dseinternational/vocabulary-growth/issues/330) asks whether a Gompertz mean function predicts our data as well as the logit-linear trend plus HSGP mean, with everything else held fixed. This note records the harness built to answer it, the priors and their prior-predictive evidence, the VG20 spoken decision, `dev`-tier smoke runs and the commands for the `test`-tier runs. **It does not answer the question.** The smoke runs exercise the code; their numbers are not evidence. **Code:** `scripts/experiments/gompertz_mean_arm.py` and `tests/test_gompertz_mean_arm.py`, on branch `feat/gompertz-mean-harness` from `main` at `8b5e402`. **Data:** the prepared database of that checkout. **Output:** `<output-root>/experiments/gompertz-mean/`.

[The shape comparison](202609101230-gompertz-comparison.md) found that a Gompertz expresses typically-developing production almost exactly, typically-developing comprehension less well, and the Down syndrome curves only with a free asymptote. It could not say how a Gompertz _mean_ would predict, and its §6 set out the arm that would. This is that arm.

## 1. What the harness does

The Gompertz arm replaces `expit(trend + GP)` with

$$p_{ij} = \operatorname{expit}\left(\operatorname{logit}\left(A \exp\left(-\exp\left(-k_g (t - T_i)\right)\right)\right) + \text{study}_j + \text{child}_i\right)$$

with the asymptote share $A$ free. Only the population latent changes. Study and child effects, the registered sex covariate, the Beta-Binomial likelihood with its two-anchor age-varying dispersion, the random-effect scales and priors, the analysis frame, the sampling configuration and the seed are those of the registered graph.

**How.** Each engine builds its mean through a module-level call to `gp_utils.trend_and_gp`. During the build, the harness replaces that name in the engine module (`common_univariate_re` for VG11 and VG12, `common_bivariate_re` for VG20) with a function that builds the Gompertz. It restores the original afterwards. Every other line of the registered build runs unchanged, and `src/vocab_growth/` is not modified, so no fit of record is restaled. The engine passes only standardised ages. The harness recovers the standardisation exactly from the standardised slope anchors and their values in months. The logit is computed as $\log p - \log(1 - p)$ with $\log p = \log A - e^{-k_g(t - T_i)}$ and a stable $\log(1 - e^{x})$. A share that underflows in probability therefore remains a finite logit for the likelihood's clip.

**What the Gompertz arm does not use.** The registered trend anchors (`p_slope_low`, `p_slope_hi`) have no counterpart. The mean clamp stops a linear trend extrapolating, and the GP anchor fixes the GP's level. Neither has anything to act on in a curve bounded by $A$, so both are ignored. The registered definition still carries these fields, so the priors stage still draws their prior plots into the arm's directory. They are not in the graph.

**What the tests pin.** `tests/test_gompertz_mean_arm.py` checks the mean function against known values: $A/e$ at $T_i$, $A e^{-e^{-1}}$ one growth time later, Day et al.'s 250 words at $T_i$, monotonicity, the asymptote and finite logits in the tails. It checks that the PyTensor form matches NumPy and that the age standardisation is recovered exactly. On the synthetic frame from `tests/support`, it builds each registered graph and its Gompertz arm for VG12, VG11 and both VG20 options. It then checks that the free random variables differ only by the mean's own parameters and that all others keep their order and dims. Observed variables must be identical. Only the mean's deterministics (`slope`, `intercept`, `ell`, `g_unit`, `g`) may disappear. It also evaluates the built latent at a fixed point against the NumPy Gompertz in months. It checks that the patch is removed after the build and that the output-root refusal works.

**Output root.** Everything goes under `<output-root>/experiments/gompertz-mean/`. `<output-root>` defaults to the project's resolved root, `D:\output\vocabulary-growth` on the reporting workstation. The harness refuses any destination inside a canonical `models/` directory, either `<checkout>/output/models` or `$DSE_VOCAB_GROWTH_OUTPUT_DIR/models`. It checks again after setting the override. Arm fits land in `experiments/gompertz-mean/models/<id>-gompertz-mean-<arm>/`.

**Provenance caveat.** The pipeline writes an ordinary `fit_manifest.json`. Its definition is the registered one under the config name `gompertz-mean-<arm>`, and its code signature is the package's. **Neither describes the patch.** Each arm directory therefore also carries the exploratory marker and `gompertz_mean_arm.json`. The latter records the curve, arm, ratio option, sampling tier and priors. `compare` refuses an arm whose recorded priors, tier or ratio option differ from those requested. Directory names do not include the tier, so a `test` fit replaces the `dev` smoke fit of the same arm.

## 2. VG20's spoken mean: the Gompertz replaces both means

VG20 models spoken as $p_u \times q$. `--ratio-mean` decides what happens to $q$.

- **`gompertz` (the default):** a second free-asymptote Gompertz replaces the production-ratio mean as well.
- **`flexible`:** the Gompertz replaces the comprehension mean only, and $q$ keeps its trend plus HSGP.

The default is `gompertz` for three reasons. First, the question is whether a _parametric_ mean predicts as well. With the HSGP left on $q$, the spoken arm would still carry a 16-coefficient HSGP. It would then test how the comprehension mean propagates into speech, not whether a Gompertz describes speech. Second, $q$ is a saturating curve the family can express. In the VG20 fit of record it rises from 0.017 at 12 months to 0.30 at 42 and 0.76 at 72. Third, there is a precedent: the [forward projection](202609091900-vg20-forward-projection-to-11-years.md) fitted a Gompertz to the ratio. The spoken mean is then a product of two Gompertz curves. That product is itself a Gompertz when the two rates are equal, and close to one otherwise.

`flexible` was cheap to add and is kept as a decomposition arm. If the default loses on speech, `flexible` shows whether the comprehension mean or the ratio mean is responsible. Run it alongside the default at `test` (§5).

## 3. Priors, set by prior-predictive check

$k_g \sim \text{LogNormal}(\log m, s)$ per month, $T_i \sim \text{Normal}$ in months and $A \sim \text{Beta}$, per curve. The centres follow the issue. Typically-developing production uses Day et al. (2025). Typically-developing comprehension, which the paper does not model, uses our own pooled least-squares and free-asymptote fits ([shape comparison](202609101230-gompertz-comparison.md) §§4–5). So do the Down syndrome curves, as the issue suggests. The prior-predictive check set the spreads and moved two centres.

| Slot               | $k_g$                     | $T_i$ (months)  | $A$         | Centre from                            |
| ------------------ | ------------------------- | --------------- | ----------- | -------------------------------------- |
| VG12 TD understood | LogNormal(log 0.14, 0.4)  | Normal(16, 3)   | Beta(6, 4)  | Our fits: 0.131–0.177, 15.0–17.3       |
| VG11 TD spoken     | LogNormal(log 0.15, 0.35) | Normal(23.3, 3) | Beta(16, 4) | Day et al. 0.161–0.17, 23.3; ours 0.14 |
| VG20 DS understood | LogNormal(log 0.04, 0.5)  | Normal(38, 10)  | Beta(7, 3)  | Our fits: 0.031–0.043, 37.1–46.6       |
| VG20 DS ratio $q$  | LogNormal(log 0.07, 0.5)  | Normal(42, 10)  | Beta(9, 2)  | VG20 fit-of-record $q$ curve           |

`prior` builds each arm's graph through the engine's own preparation, prior and build stages, draws 500 prior-predictive samples (seed 20261001) and compares them with the observed rows. It reports two tables. The first gives 5/50/95% of the population (median-child) curve at the query ages. The second gives, by age band, the observed median against 5/50/95% of the prior-predictive band median, with the share of rows inside their own 90% prior-predictive interval.

**Observed band median against the prior-predictive band median, 5/50/95%.** Gompertz arm with the final priors. The registered arm's prior, also checked, is shown for reference.

| Curve              | Age band | Observed | Gompertz prior  | Registered prior |
| ------------------ | -------- | -------: | --------------- | ---------------- |
| VG12 TD understood | 8–12     |       32 | 0 / 33 / 172    | 0 / 55 / 318     |
| VG12 TD understood | 12–16    |       94 | 10 / 107 / 275  | 6 / 102 / 311    |
| VG12 TD understood | 16–20    |      197 | 57 / 191 / 376  | 32 / 162 / 396   |
| VG12 TD understood | 20–25    |      294 | 119 / 284 / 532 | 53 / 264 / 624   |
| VG11 TD spoken     | 12–16    |        8 | 0 / 7 / 109     | 1 / 23 / 95      |
| VG11 TD spoken     | 16–20    |       41 | 0 / 53 / 215    | 9 / 63 / 214     |
| VG11 TD spoken     | 20–25    |      170 | 36 / 188 / 397  | 34 / 185 / 553   |
| VG11 TD spoken     | 25–31    |      420 | 174 / 358 / 554 | 51 / 474 / 779   |
| VG20 DS understood | 24–36    |      180 | 15 / 141 / 303  | 12 / 128 / 401   |
| VG20 DS understood | 36–48    |    316.5 | 80 / 228 / 426  | 33 / 198 / 468   |
| VG20 DS understood | 48–60    |      360 | 150 / 312 / 537 | 102 / 304 / 552  |
| VG20 DS understood | 72–116   |    537.5 | 212 / 455 / 682 | 152 / 567 / 801  |
| VG20 DS spoken     | 24–36    |       14 | 0 / 9 / 57      | 1 / 21 / 73      |
| VG20 DS spoken     | 36–48    |       70 | 1 / 63 / 172    | 10 / 62 / 168    |
| VG20 DS spoken     | 48–60    |      195 | 35 / 130 / 263  | 36 / 110 / 233   |
| VG20 DS spoken     | 72–116   |      394 | 136 / 306 / 467 | 57 / 343 / 623   |

Every observed band median lies inside the central 90% of the Gompertz prior's band median. The share of rows inside their own 90% prior-predictive interval is 0.94–0.99 for every band and curve. The Gompertz priors are narrower than the registered ones at the top of each range. This is expected: three parameters cannot reach the registered prior's extreme late curves, under which 11–16% of draws exceed 95% of the inventory at 30 months (TD), as do 11% of comprehension draws at 84 months (DS). No Gompertz prior draw did so at those ages.

**What the check changed.** The first VG20 priors were centred at $k_g$ 0.035 and $T_i$ 42 for comprehension, and 45 for the ratio with $A \sim$ Beta(8, 2). They sat low: the observed understood median of 316.5 words at 36–48 months was at the top of a prior band of 57 / 188 / 364. Spoken medians of 195 and 394 were above the prior's medians of 104 and 273. The final priors move both curves earlier and the asymptotes up. The first VG11 prior was $k_g$ 0.165 with $A \sim$ Beta(12, 4). It put the 25–31 band at 156 / 340 / 550 against an observed 420. It also put 12–16 months at 3 words against 8. The final prior uses $k_g$ 0.15, between our pooled 0.141 and the paper's 0.161–0.17, and $A \sim$ Beta(16, 4), mean 0.8 against Words and Sentences' 680/810 = 0.84.

**Two structural properties, not prior choices.** First, the double exponential falls to zero faster in the early tail than the registered logit-linear mean. At 12 months, 81% of the VG20 Gompertz prior's draws put the spoken population curve below one word, against 25% for the registered prior. The 8–24 month band's observed median of 2 words is still inside 0 / 0 / 7. Second, the VG12 asymptote is not observed. Comprehension forms stop at 309–418 items and the data at 25 months, so the posterior for $A$ there will be prior-sensitive.

## 4. Smoke runs at `dev`: the code paths work, and the numbers mean nothing

`dev` is 2 chains × 500 draws after 500 tuning steps, seed 47. Run on 2026-10-01 on the reporting workstation, with several fits running concurrently.

| Fit                        | Wall time | Divergences | Max R-hat | Min bulk ESS |
| -------------------------- | --------: | ----------: | --------: | -----------: |
| VG20 baseline              |  3 m 11 s |          14 |     1.048 |           32 |
| VG20 Gompertz (ratio too)  |  3 m 08 s |           0 |     1.111 |           27 |
| VG20 Gompertz (`flexible`) |  4 m 12 s |           1 |     1.070 |           47 |
| VG12 baseline              |  4 m 44 s |          16 |     1.066 |           34 |
| VG12 Gompertz              |  4 m 15 s |           0 |     1.233 |            7 |

No fit converged, as expected at `dev`. In every Gompertz arm the worst-mixing parameters are the Gompertz's own: VG12's $A$, $T_i$ and $k_g$ have bulk ESS 7–8, and VG20's comprehension trio 31–37. This is the ridge among the three when the asymptote is weakly observed. It is the main risk for the `test` runs (§5).

**Paired PSIS-LOO, Gompertz minus baseline, `dev`.** Same rows in both arms. A spoken row with zero observed comprehension is dropped from both.

| Curve              | Gompertz arm        | Rows | ELPD diff (SE) | Pareto k above threshold, baseline / Gompertz |
| ------------------ | ------------------- | ---: | -------------- | --------------------------------------------- |
| VG12 TD understood | Gompertz            | 7049 | +6.2 (17.7)    | 37% / 37%                                     |
| VG20 DS understood | ratio also Gompertz | 1300 | −27.1 (7.2)    | 28% / 26%                                     |
| VG20 DS spoken     | ratio also Gompertz | 1408 | −60.5 (16.4)   | 26% / 22%                                     |
| VG20 DS understood | `flexible`          | 1300 | −18.7 (6.5)    | 28% / 26%                                     |
| VG20 DS spoken     | `flexible`          | 1408 | +5.8 (6.6)     | 26% / 26%                                     |

The threshold is the sample-size-dependent good-k, 0.667 at these draw counts. **PSIS-LOO is unusable on every one of these comparisons**, as it is on the fits of record: 22–37% of rows exceed the threshold. The typically-developing models are the extreme case, because most children have one administration ([held-out validation note](202609131214-held-out-validation-for-the-td-models.md) §1). At `rep` the TD models of record have 36% (VG12) and 47% (VG11) of rows unusable, and VG20 has 23–25% per outcome. No better-tuned fit will change that, so the TD PSIS comparison will be reported with its Pareto diagnostics and not interpreted as a predictive score.

**Leave-one-subject-out, `dev`, 2 folds (VG20 frame: 1,707 rows, 943 children).** The fold mechanics of `scripts/kfold_loso.py` are reused: stratified subject folds, fold frames, holdout fits through `vocab_growth.fold_fits` and per-subject scoring. They are reused, not copied, by loading the script and patching only its output directories. The frame is VG20's own prepared frame rather than `kfold_loso.py`'s loader, so the analysis frame is held fixed. Gompertz minus baseline: both outcomes −15.9 (paired SE 16.1), understood −1.3 (5.6), spoken −17.7 (13.5). All four fold fits failed the convergence gate at `dev` (max R-hat 1.03–1.15), and two folds hold out half the children. These numbers show that the path runs, not what the result is. Fold fits took 43–181 s.

## 5. The `test`-tier runs, queued for a compute window

Not run here. Each `fit` runs the full engine pipeline at `test` (4 chains × 2000 draws after 2000 tuning steps). Each arm's directory is under the experiment root, so nothing else is touched. Run the commands sequentially, or at most two at once. These are full-trace fits, and concurrent memory pressure has killed rep fits on this workstation before.

```bash
# Prior-predictive record (minutes; already run for this note)
uv run python scripts/experiments/gompertz_mean_arm.py prior vg12
uv run python scripts/experiments/gompertz_mean_arm.py prior vg11
uv run python scripts/experiments/gompertz_mean_arm.py prior vg20

# Down syndrome (cheapest first)
uv run python scripts/experiments/gompertz_mean_arm.py fit vg20 baseline --config test
uv run python scripts/experiments/gompertz_mean_arm.py fit vg20 gompertz --config test
uv run python scripts/experiments/gompertz_mean_arm.py --ratio-mean flexible fit vg20 gompertz --config test

# Typically developing
uv run python scripts/experiments/gompertz_mean_arm.py fit vg12 baseline --config test
uv run python scripts/experiments/gompertz_mean_arm.py fit vg12 gompertz --config test
uv run python scripts/experiments/gompertz_mean_arm.py fit vg11 baseline --config test
uv run python scripts/experiments/gompertz_mean_arm.py fit vg11 gompertz --config test

# Scores
uv run python scripts/experiments/gompertz_mean_arm.py compare --config test
uv run python scripts/experiments/gompertz_mean_arm.py --ratio-mean flexible compare --curves vg20 --config test
uv run python scripts/experiments/gompertz_mean_arm.py loso --config test --folds 5
```

**Estimated cost.** The figures are scaled from the `rep` fits of record (6 chains × 12,000 iterations: VG20 53 min, VG12 1.5 h, VG11 3.7 h) to `test`'s 4,000 iterations per chain. They agree with the `dev` timings above. VG20 is about 15–20 minutes per arm (three arms). VG12 is about 30 minutes per arm and VG11 about 75 minutes per arm. Together that is about 4.5 hours of fits. `loso` makes 10 VG20 fold fits, without predictive or plotting stages, at about 10–15 minutes each: about 2 hours. A `--ratio-mean flexible loso` would add another 2 hours, and is worth running only if the default arm loses on speech. The total is about 6.5 hours sequentially. Trace disk at `test` should be a few GB per typically-developing arm, an estimate scaled from the `rep` traces.

**Read before interpreting.** Check each arm's gate in `gompertz_mean_test_convergence.csv`. A Gompertz arm that fails on its own $k_g$, $T_i$ or $A$ has not answered the question. The remedy is a reparameterisation, not more draws: for example, sample the curve's values at two ages, as the registered trend's anchors do, and derive $k_g$ and $T_i$ from them. For the TD arms, the paired PSIS-LOO is the only score this harness provides, and §4 shows it is unusable. A typically-developing answer needs a refit-per-fold path for the univariate engine, which `kfold_loso.py` does not support ([held-out validation note](202609131214-held-out-validation-for-the-td-models.md)). Building one is an open item, not part of this harness.

## 6. The prediction to test, recorded in advance in #330

> On the shape evidence: the typically-developing **production** arms should come out indistinguishable by LOO, and the Down syndrome arms should favour the flexible mean, because a free-asymptote Gompertz still cannot bend where the asymptote itself has to be inferred from data that stop at 7.5 years. Typically-developing comprehension should favour the flexible mean too, on the 0.49-versus-0.368 evidence.

The `dev` numbers in §4 are not a test of this and must not be read as one.

## 7. Done when

Done-when for #330 is unchanged. It needs converged `test` fits of both arms of all four curves, with ELPD differences and standard errors. The VG20 LOSO run is also required. A follow-up note must record the result against §6 and say whether any mean function should change. If the answer is yes for a reported model, a separate issue should propose the definition change, because it would restale every fit of that class.
