# Do children keep their relative vocabulary standing? Tracking in the repeated measures

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Exploratory analysis of the assembled data. It fits no model and changes no model of record. Reproduced by `scripts/experiments/rank_stability.py`; the output quoted here is `--boot 300` on the 2026-08-14 data tree.

## 1. The question, and why no fitted model answers it

If a child is ahead of others their age at one assessment, are they still ahead later? That is developmental **tracking**, and it matters for interpretation: it decides whether an early assessment is informative about later standing or close to noise.

VG08–VG10 cannot answer it. They give each child a **constant** random intercept, so a child's offset from the population trajectory is fixed by construction. That assumes perfect tracking; it does not test it. The answer has to come from the repeated measures directly, which is what this note does.

## 2. Method

Raw counts are not comparable across ages or instruments, so each administration is scored as a residual:

1. the count as a proportion of that administration's own form ceiling, clipped away from 0 and 1 by half an item;
2. its logit — the scale the models work on, and the one that linearises the trajectory's middle;
3. residualised on a cubic in standardised age **plus study fixed effects**.

The score is therefore "how this child compares with others of the same age, measured by the same study". The study term is not optional: instrument, country and recruitment differ enough between sources that an unadjusted comparison largely recovers the study rather than the child.

Three summaries, all with **cluster bootstrap** intervals over children, since children — not administrations — are the independent unit:

- **ICC**, the between-child share of residual variance among children with repeated measures. 1.0 means standing never changes; 0.0 means each assessment is independent of the last.
- **Lag-binned Spearman correlation** over all within-child pairs. A stable trait gives a flat curve; a drifting one decays.
- **Quartile transitions**, the same thing in a readable form.

`rho/rel` disattenuates for measurement error using a **binomial lower bound** on the error variance, computed on the **repeated-measures children only** — the rows the pairs come from. (An earlier version used the full set including singletons, giving 0.865 rather than 0.853 on DS spoken; small, but only one of the two makes the variance decomposition in §5 coherent.) Because the bound ignores the extra-binomial dispersion the models fit (`kappa`), it understates error and so under-corrects: treat it as a floor on true stability, not an estimate.

## 3. Result: tracking is substantial across the full Down syndrome age range

| pool              | ICC (89% CI)             | 1–6 mo | 6–12 mo | 12–24 mo |               24–60 mo |
| ----------------- | ------------------------ | -----: | ------: | -------: | ---------------------: |
| **DS spoken**     | **0.806** [0.780, 0.832] |  0.704 |   0.656 |    0.643 | **0.238** [0.04, 0.44] |
| **DS understood** | **0.786** [0.718, 0.847] |  0.701 |   0.780 |    0.721 |                      — |

334 children contribute repeated spoken measures and 253 repeated comprehension measures.

In readable terms, comparing each child's **first and last** observation — a median span of 12 months on spoken and 11 on comprehension — **half stay in the same quartile, 86–87% stay within one quartile**, and movement from bottom to top is rare: 5 of 334 on spoken, 4 of 253 on comprehension. (The lag-binned correlations below are over _all_ within-child pairs, whose median gap is 7 months.)

Correlations hold at roughly 0.65–0.72 out to two years and then **fall sharply beyond two years** (0.238 on spoken, n = 59 children). Whether that is genuine divergence or accumulated noise cannot be separated here.

## 4. The comparison that matters, and the trap in it

The typically-developing pool has 2,017 children with repeated spoken measures, which invites a direct contrast. Taken naively it looks striking:

| pool          | 1–6 mo | 6–12 mo | 12–24 mo |
| ------------- | -----: | ------: | -------: |
| DS understood |  0.701 |   0.780 |    0.721 |
| **TD spoken** |  0.800 |   0.660 |    0.461 |

DS comprehension looks _flat_ where TD production _decays_ — an appealing story about steadier development.

**It does not survive age matching.** The TD pool spans 8–30 months only, and that window is the vocabulary explosion, when rank order is least stable. The DS pool spans 8–115. Restricting DS to the same ages reverses the picture:

| pool (8–30 months)    | ICC (89% CI)         | 1–6 mo | 6–12 mo |                 12–24 mo |
| --------------------- | -------------------- | -----: | ------: | -----------------------: |
| **DS spoken ≤30**     | 0.710 [0.658, 0.754] |  0.690 |   0.380 | **−0.156** [−0.48, 0.21] |
| **DS understood ≤30** | 0.858 [0.800, 0.904] |  0.794 |   0.770 |       0.453 [0.13, 0.69] |
| TD spoken             | 0.748 [0.730, 0.762] |  0.800 |   0.660 |                    0.461 |

Age-matched, **DS spoken tracking decays faster than TD spoken**, and by 12–24 months is indistinguishable from zero. The apparent DS advantage in the first table was an artefact of comparing a wide age range against a narrow, fast-changing one.

## 5. Why young DS spoken tracking collapses: it is a floor effect

The reliability bound explains it. For DS spoken at ≤30 months it is **0.571**, against 0.85–0.98 everywhere else — by far the lowest in the analysis. At those ages DS spoken counts are tiny — the observed median at 22–26 months is **9 words, IQR 4–21** — so a difference of two or three words moves a child across quartiles. (An observed figure is quoted rather than VG10's fitted mean because VG10 was being refit as this was written.) The disattenuated 1–6 month correlation hits the 1.0 cap, which is the signature of noise rather than instability.

So the finding is **not** that young children with Down syndrome have unstable expressive ability. It is that **spoken-word counts are too small at those ages to measure standing reliably**. Comprehension in the same window tracks well (ICC 0.858, correlations 0.79/0.77), because the counts are large enough to carry information.

Splitting the adjusted variance three ways makes it explicit. "Occasion" is the remainder after between-child variation and the binomial measurement bound, i.e. genuine occasion-to-occasion movement:

| pool              | between-child (`tau_subj`) | within-child | of which measurement |   occasion |
| ----------------- | -------------------------: | -----------: | -------------------: | ---------: |
| DS spoken         |                      80.6% |        19.4% |                14.7% |      +4.7% |
| DS understood     |                      78.6% |        21.4% |                 3.4% |     +18.0% |
| **DS spoken ≤30** |                      71.0% |        29.0% |            **42.9%** | **−13.9%** |
| DS understood ≤30 |                      85.8% |        14.2% |                 5.3% |      +8.9% |
| TD spoken         |                      74.8% |        25.2% |                 5.0% |     +20.2% |
| TD understood     |                      87.1% |        12.9% |                 2.0% |     +11.0% |

The `DS spoken ≤30` row is **incoherent, and that is the result**: the measurement bound (42.9%) exceeds the entire within-child variance (29.0%), so the occasion share goes negative. A negative share is impossible, so at those ages either the ICC or the error bound is overstated — most likely both, since with a median of 9 spoken words the logit approximation underlying each is straining. The honest conclusion is that the between-child signal and the sampling noise **cannot be separated at all** in that regime.

That has a direct practical reading: **before about 30 months, an expressive-vocabulary count is a poor basis for judging a child's relative position; a comprehension measure is far better.**

## 6. Corrections to earlier statements

Two figures quoted from an ad-hoc version of this analysis earlier on 2026-08-14 were wrong and are superseded here.

**"88% of adjusted variance is between children" was inflated.** It included children with a single observation, which contribute no within-child variance and so cannot inform the ratio. Restricted to children with repeated measures, the ICC is **0.806** (DS spoken) and **0.786** (DS understood).

**The first cluster bootstrap for the ICC was broken.** It resampled children with replacement and then subset the frame with `isin(unique())`, which silently dropped the duplicates — sampling _without_ replacement over a shrinking subset, and biasing the estimate down. It was caught only because the intervals did not contain their own point estimates (TD spoken: 0.916 with CI [0.735, 0.761]). A milder bias would have looked entirely plausible. Duplicated clusters now receive fresh keys; the failure is recorded in `icc_ci`'s docstring.

## 7. Caveats

- **Cross-sectional adjustment within study.** Children measured by only one study contribute nothing to the between-study comparison, and the study fixed effect absorbs any real between-source difference in child ability.
- **The DS↔TD contrast is approximate.** DS rows carry a recorded `survey_vocab_max`; TD rows do not, so the TD ceiling is taken as the observed maximum per form. The two are normalised on comparable but not identical footings.
- **Regression to the mean** inflates apparent quartile persistence at the extremes.
- **ρ ≈ 0.65 is a population tendency, not a prediction for an individual.** Roughly half of children change quartile over a year.
- The 24–60 month DS bin rests on 59 children, and the ≤30-month 12–24 bin on 23. Both are thin.

## 8. What this says about Proposal A1

[202607261540](202607261540-item-difficulty-and-the-aggregate-likelihood.md) §9 proposes **A1**: make `tau_subj_*` age-varying and hold `kappa` constant, so "does the spread between children widen with age?" is asked of the parameter that can answer it. Its stated structural caveat is that scaling a single per-child offset by `tau(age)` "imposes perfect rank correlation of children across age — children never cross".

**That caveat was a logical property; this note makes it a measured quantity.** Perfect rank correlation is exactly what the lag-binned Spearman estimates:

| interval | observed ρ | disattenuated | A1 assumes |
| -------- | ---------: | ------------: | ---------: |
| 1–6 mo   |      0.704 |         0.825 |       1.00 |
| 6–12 mo  |      0.656 |         0.770 |       1.00 |
| 12–24 mo |      0.643 |         0.753 |       1.00 |
| 24–60 mo |  **0.238** |     **0.279** |       1.00 |

Three consequences.

**The assumption is workable to about two years and fails beyond.** Roughly 0.75–0.83 disattenuated against an assumed 1.0 up to 24 months; 0.28 beyond. Since the median within-child gap is 7 months, A1's fan shape is a fair approximation over the span children are actually observed, and a poor one over the 8–115 month range the models report on.

**It applies to the models of record too, not only to A1.** A constant `tau_subj` is A1 with `tau(age)` flat, so it imposes the _same_ perfect rank correlation. What changes is that the assumption now has a number against it.

**A1's identification warning can be located.** The note warns that `tau_subj` and `kappa` are "two names for the same deviation" for a singleton child. §5's decomposition shows where that becomes fatal rather than merely awkward: on DS spoken at ≤30 months the measurement bound exceeds the whole within-child variance. So A1's required parameter-recovery run should be designed at **young ages on production specifically** — a recovery whose truth sits where counts are large will succeed and prove nothing — and comprehension and production should be scored separately, since their measurement shares differ by an order of magnitude (3.4% against 14.7% overall; 5.3% against 42.9% at ≤30 months).

Finally, the relaxations the note names as future work — random slopes, or a child-level longitudinal function — are what the 24–60 month decay calls for. A1 cannot represent crossing at all, so if the widening it finds is real, some of it may be drift the model has nowhere else to put.

## 9. Does the spread between children widen with age? The direct measurement

A1 exists to estimate one quantity: whether the between-child spread grows. The same adjusted scores answer it descriptively, without fitting anything. Each age band's residual SD is decomposed by subtracting the mean binomial sampling variance, leaving between-child **plus** occasion variation — not between-child alone, because one cross-sectional band cannot separate the two. `spread_by_age_band` in the harness reproduces it.

| band (mo) |         n | DS spoken | DS understood |
| --------- | --------: | --------: | ------------: |
| 8–18      | 149 / 128 |  **0.00** |          1.22 |
| 18–30     | 364 / 289 |      1.18 |          1.21 |
| 30–42     | 307 / 261 |      1.64 |          1.21 |
| 42–60     | 338 / 205 |      1.69 |          1.40 |
| 60–84     |  211 / 91 |      1.71 |      **2.06** |
| 84–120    |    59 / — |      1.46 |             — |

_Non-measurement SD on the logit scale. Row counts are spoken / understood._

**The widening is real, and the two modalities do it at different times.** Production spreads out early — 1.18 at 18–30 months to 1.64 by 30–42 — and then **plateaus** at about 1.70 for the next forty months. Comprehension is flat at 1.21 to 42 months and then **rises late**, to 2.06 at 60–84.

Two artefacts to read past, both visible in the measurement column rather than argued around. The 8–18 month spoken zero is the floor effect of §5 in its starkest form: the measurement SD (1.108) exceeds the observed one (1.056), so the subtraction hits its bound. It is not an absence of variation between children; it is an inability to see any. And the 84–120 month dip rests on 59 observations from 50 children, at ages where the logit scale compresses against the 810-word ceiling — treat it as "no evidence of further widening", not as narrowing.

**The consequence for the models of record.** VG08–VG10 hold `tau_subj_*` constant, so nothing in them can represent any of this; the only parameter free to vary with age is `kappa`. Its fitted decline — `kappa_u` from roughly 110 at 24 months to 33 at 48 in the calibration the DS joint family carries — is therefore doing double duty, and some unknown share of it is between-child widening wearing the wrong parameter's name. **That is exactly A1's charge, and the table supports it.**

**It does not follow that A1 should be adopted**, and this note's own measurements are why.

- **The fan shape is wrong in both directions.** A1 scales one deviate by `tau(age)`; between two anchors that is monotone. Production rises then plateaus, comprehension is flat then rises. A two-anchor `tau` misfits each, and misfits them oppositely.
- **"Hold `kappa` constant" is right for one outcome and wrong for the other.** §5 puts genuine occasion variation at 4.7% of adjusted variance on spoken and **18.0%** on understood. Pinning `kappa` is defensible for production; on comprehension it would push real occasion-to-occasion movement into `tau_subj(age)` and inflate the widening being measured.
- **Below 30 months on production the two are not separable at all** (§5), which is where the production widening starts.

**Decision, 2026-08-14: A1 is registered as a sensitivity on VG10 and is not a candidate model of record.** The variant is `vg10 / a1-tau-age-varying`, and it is parameterised so the model of record is nested at zero — `tau_subj_*_young` keeps the record's `HalfNormal(1.5)` exactly, `log_tau_subj_*_ratio ~ Normal(0, 0.5)` is the single added parameter, and both `kappa` blocks are held flat over the same 24/48-month anchors. Its purpose is the diagnostic quantity and nothing more: **how much of `kappa`'s fitted decline is misattributed between-child widening.** The relaxations in §10 are the fix; A1 is the measurement that says how much fixing is needed.

> [!NOTE]
> **Frames do not match, deliberately.** The item-difficulty note's fitted frame has 832 children, 460 singletons and 372 with repeats. This note's tracking sets are per outcome and differently masked: 767 children with 334 repeats on spoken, 610 with 253 on comprehension. Do not equate 372 with 334 or 253.

## 10. The real fix: what a within-child age-varying random effect would buy

A1 is a measurement, not a repair. The repair is a model in which the child's departure from the population trajectory is itself a function of age. Two structures are usually proposed for that, and the repeated measures say which one these data want.

### 10.1 What the fix has to separate

Three things are currently entangled in two parameters. **Persistent** between-child differences sit in `tau_subj`, which is constant. **Occasion-to-occasion** movement has nowhere of its own and is absorbed by `kappa` alongside sampling noise. **Drift** — a child systematically changing standing over months — is representable by nothing at all, so it too lands on `kappa`. §5's decomposition is the same three quantities measured from outside the model; a structure that names all three is what would let the models estimate them.

### 10.2 Random slopes

Give each child a rate as well as a level, correlated:

```
f_i(a) = mu(a) + b0_i + b1_i * (a - a_ref)
(b0_i, b1_i) ~ MVN(0, Sigma)      Sigma = diag(tau0, tau1) Omega diag(tau0, tau1)
```

`tau0` is the between-child spread at the reference age; `tau1` is the spread of child-specific **rates**; the off-diagonal `rho01` says whether children who start ahead also gain faster. Three consequences matter here. Children **cross**, so the lag-correlation decay of §3 is representable rather than assumed away. The spread is quadratic in age — `tau0² + 2 rho01 tau0 tau1 D + tau1² D²` — so widening is _implied_ by the parameters rather than imposed as a shape, and it can widen, narrow, or do neither. And `rho01` is a reportable scientific quantity in its own right: fan-out versus catch-up, stated as a number.

**The data support a slope on production and cannot yet see one on comprehension.** Fitting each child's own slope through their residuals, and subtracting the sampling variance the within-child noise implies:

| outcome           | children (≥3 obs) | SD(child slope), logit/month | drift at 1 SD over 24 months |
| ----------------- | ----------------: | ---------------------------: | ---------------------------: |
| **DS spoken**     |               182 |     **0.051** [0.023, 0.070] | **1.22 logits** [0.55, 1.67] |
| **DS understood** |               100 |         0.000 [0.000, 0.024] |        0.00 logits [0, 0.58] |

Against a between-child SD of about 1.7 logits, two years of drift at one standard deviation moves a child roughly **70% of the whole between-child spread** on production. That is not a refinement; it is a first-order feature the models cannot represent. On comprehension the estimate hits its floor — the observed slope spread is _below_ what noise alone predicts — so the honest reading is an upper bound, and the windows are short (span median 11 months, p90 15) so it is a weak one.

`corr(child level, child slope)` is **+0.18** [0.05, 0.31] on production at a mean window age of 41 months: mild fan-out.

### 10.3 Latent AR(1), and why it is the better fit here

Replace the constant offset with a child-level process over age:

```
f_i(a) = mu(a) + u_i(a),    cov(u_i(a), u_i(a')) = tau_subj² exp(-|a - a'| / ell_child)
```

Two parameters where there was one. `tau_subj` is still the between-child amplitude; **`ell_child` is the persistence, in months, and it is the tracking statistic this note measures reported as a model parameter.** The current models are nested at `ell_child -> infinity`, and pure occasion noise is `ell_child -> 0`, so a posterior for `ell_child` is a direct answer to "how long does standing last" rather than a comparison between models.

**The measurement that chooses between §10.2 and §10.3.** A permanent random slope _accumulates_: a slope SD of 0.051 logit/month sustained across 18→84 months would add `(0.051 × 66)² = 11.3` to the variance, taking the spread far past the 1.18→1.71 actually observed in §9. The drift is real over one- to two-year windows and **does not accumulate over five years**. Drift that appears locally and cancels globally is **mean-reverting**, which is what an AR(1)/Ornstein–Uhlenbeck component is and what a permanent random slope is not.

The natural form is therefore both parts: a permanent intercept plus a transient mean-reverting component, whose three parameters map one-to-one onto §5's three-way decomposition. Fitting `rho(lag) = w + (1 - w) exp(-lag / ell)` to the disattenuated correlations gives `w = 0.77`, `ell ≈ 0` on comprehension — a permanent intercept plus occasion noise, i.e. close to what the models already assume — and on production a single exponential with `ell ≈ 35` months that fits poorly, because the observed decay is flat to 18 months and then falls off a cliff in a bin holding 59 children. **The production long-lag behaviour is the least well determined quantity in this note and the one the fix most depends on.** That is an argument for fitting the structure, not for asserting its parameters from these summaries.

### 10.4 Where A1 lands once the fix exists

A1's age-varying `tau` is not wrong; it is stacked on the wrong base. Placed on the _amplitude_ of a mean-reverting child process, `tau(age)` estimates whether the between-child spread widens **while children are still permitted to cross** — which is A1's question with its one measured-false assumption removed. That ordering is the recommendation: the relaxation first, A1's `tau(age)` on top of it second, and the registered A1 variant meanwhile as the diagnostic of how much of `kappa`'s decline is at stake.

### 10.5 What it would cost, and what has to be checked

The subject random effects have one construction site per engine (`common_univariate_re`, `common_bivariate_re`, `common_joint_modality`), and the shift they produce is already applied at observation level, so the seam is narrow. Random slopes are the cheaper change: a second coordinate per child, non-centred through an `LKJCholeskyCov`, doubling the random-effect dimension (832 children × 2). The AR(1) form needs the child's latent values at their own observation ages, which is block-diagonal with blocks of at most 8 — cheap in principle, but it must be written as a padded rectangular recursion rather than a scan to sample at a tolerable speed.

Two checks gate either of them, and neither is optional. **Parameter recovery under the new structure**, sited at young ages on production for the reason §5 gives. And **the singleton share**: 433 of 767 children on production and 357 of 610 on comprehension contribute one observation, so they inform the population trajectory and nothing about persistence. 182 children carry three or more production observations, 100 carry three or more comprehension observations, and those are the rows that would identify a persistence parameter. It is enough to try. It is not enough to assume it will succeed.
