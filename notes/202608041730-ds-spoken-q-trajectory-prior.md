# The Down syndrome spoken prior predictive: the `q` trend line and `eta_q`

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Analysis and implementation note, 2026-08-04. Two changes are **implemented** across the Down syndrome joint family (VG05, VG07–VG10, VG14–VG16): the high-age `q` anchor moves from `Beta(3, 2)` to `Beta(4, 1.2)` (§4), and the `q`-GP amplitude `eta_q` moves from `HalfNormal(0.20)` to `HalfNormal(0.8)` (§5). VG13 keeps `HalfNormal(0.20)` and its own `q` anchors, for the reason in §5. Every model in the family now has a stale fit; only VG10 has been refitted (§7).
>
> This note supersedes §5 of [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) on `eta_q`, which reached the same recommendation from a **wrong diagnosis** — see §5 here. It is a companion to [202608041216](202608041216-ds-understood-trajectory-prior.md), which fixed the understood anchors the same day; §3 explains why the two problems have different shapes and different fixes.

> [!IMPORTANT]
> Three things a reader should take away. **The `eta_q` change does not fix the prior predictive** — widening a zero-mean GP widens its band without moving its median, measured in §5. Only the anchor change moves the curve. **The two changes address different problems** that this investigation happened to surface together. And **§6's prediction failed**: `p_slope_hi_q` did not relax when `eta_q` was loosened, which is why the anchor had to be recalibrated directly rather than left to follow.

## 1. The observation

The VG10 prior sample trajectories for words spoken sit visibly below the data cloud between roughly 30 and 70 months. This is the same class of observation as the understood one recorded in [202608041216](202608041216-ds-understood-trajectory-prior.md), and it was checked the same way: against the **fitted population curve**, not against the visual centroid of the scatter. The scatter is not the right comparator — `p_s_plot` carries no study or subject random effects, so it is a median-child trajectory, and the spoken scatter contains a dense band of exact zeros that the eye discounts.

## 2. What the prior actually does

Prior median population spoken curve against the fitted one, words out of 810:

| age (months)  |    12 |   18 |   24 |   30 |    36 |    48 |    54 |    60 |    72 |    84 |
| ------------- | ----: | ---: | ---: | ---: | ----: | ----: | ----: | ----: | ----: | ----: |
| prior median  |   4.4 |  7.3 | 11.7 | 18.9 |  29.6 |  67.6 |  95.8 | 131.6 | 223.7 | 326.9 |
| fitted median |   0.4 |  1.7 |  6.0 | 16.6 |  41.0 | 147.8 | 209.3 | 257.7 | 329.7 | 406.8 |
| ratio         | 12.0× | 4.4× | 2.0× | 1.1× | 0.72× | 0.46× | 0.46× | 0.51× | 0.68× | 0.80× |

The prior is not displaced in one direction. It is **too flat**: 12× too high at 12 months, 2.2× too low at 48–54, converging again by 84. The undershoot that prompted the investigation is real, and the overshoot at the young end is larger in ratio — invisible on the plot because it is four words against half a word.

## 3. Why this is a different problem from the understood one

> [!CAUTION]
> **The S-shape conclusion below is wrong — corrected by [202608042030](202608042030-q-mean-extrapolation.md) §2, later the same day.** The residuals quoted here are scored against the _fitted_ `q` curve on the fourteen query ages with equal weight. The fitted curve already contains the GP's correction to the mean, so it measures the consequence of the defect rather than the shape of the data, and equal weighting gives 12 months — where `q` is essentially unobserved — the same say as 36 months, where 134 administrations sit. Scored against the **observed** ratio in size-weighted bands, a straight line on the logit scale is already adequate (weighted RMS 0.214); linear in `log(age)` is _worse_ (0.285); and the best three-anchor alternative improves it by about 10% (0.193). `logit(q)` is not meaningfully S-shaped where there are data. The real defect is unbounded extrapolation _above the high anchor_, which is now fixed by clamping the mean flat there — a much smaller change than the mean form this section calls for.

The understood prior was displaced in _level_, and the fix was to move both anchors up. Here the level is roughly right on average and the _slope_ is wrong, so the fix is a rotation, not a translation.

The two also differ in what the mean form can do about it. For understood, `logit(p_u)` is concave in age and a mean linear in `log(age)` cuts the residual the GP must carry from RMS 0.438 to 0.139. For `q` that reparameterisation barely helps — RMS 0.527 to 0.335, maximum residual 0.831 to 0.580 — because `logit(q)` is not concave but **S-shaped**. The fitted local slope runs +0.071 logit/month at 12–24, peaks at +0.118 at 24–36, then decays to +0.013 by 72–90. No monotone transformation of the age axis produces an S. Only the GP can.

That fact is the hinge of the whole note: it means the `q` trajectory _requires_ GP curvature, which makes the amplitude prior on that GP load-bearing in a way it is not for understood.

## 4. The `q` trend line, and the evidence for moving it

Decomposing the logit gap between prior and fitted (`p_s = p_u × q`, and both factors are small enough at the young end for the logits to add), `q` accounts for essentially all of the displacement from 36 months on: −1.03 of the −0.95 total at 54 months, with the understood side slightly offsetting.

Within `q`, it is the high anchor. The low anchor is already well-centred — fitted 0.117 against `Beta(2, 12)`'s median of 0.126, prior CDF 0.464, contraction 0.81. The high anchor is not: fitted 0.929 against `Beta(3, 2)`'s median of 0.614, at prior CDF 0.972 — **above the prior's own 95th percentile of 0.902**. The stated intent of `Beta(3, 2)`, recorded in PRIORS.md, was that a broad prior would let the DS dataset the 84-month level. The data went past its upper tail; the prior was constraining, not permitting.

The consequence for the prior median curve is direct. The prior median trend runs `logit(q)` −1.96 at 24 months to +0.44 at 84, a slope of 0.040/month. The fitted trend runs −2.02 to +2.57, or 0.0765/month — **1.9× steeper**. A steeper line pivoting about a fixed low anchor sits lower at 12 months and higher above 40, which is exactly the pattern in §2.

The evidence for the new value is deliberately **not** the posterior. It is the directly observed per-child production ratio, `spoken / understood`, on the 902 rows of the frame that carry both outcomes:

| age (months) |    18 |    24 |    30 |    36 |    42 |    48 |    54 |    60 |    66 |    72 |
| ------------ | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| observed `q` | 0.026 | 0.061 | 0.089 | 0.128 | 0.241 | 0.475 | 0.502 | 0.610 | 0.706 | 0.733 |
| rows         |   135 |   158 |   135 |   134 |    83 |    77 |    27 |    31 |    24 |    11 |

A size-weighted least-squares line through `logit(q)` over these bands implies a trend `q(24)` of 0.053 and `q(84)` of **0.946**; unweighted, 0.057 and 0.924; restricted to 36 months and up, 0.054 and 0.943. Three estimators agree, and they land on the fitted 0.929 without using it — independent corroboration that the posterior was tracking the data rather than a sampler artefact.

**Implemented: `Beta(4, 1.2)`** — median 0.805, 5–95% 0.438–0.978, finite density at 1. The median stops deliberately short of the 0.92–0.95 the data line implies, because 84 months is past the evidence: the last band carrying both outcomes is 72 months at 11 rows, and exactly one row in the pool has both outcomes above 78 months. The anchor age itself is extrapolated, so the lower tail stays wide. The fitted 0.929 sits at prior CDF 0.817 under the new prior, against 0.972 under the old.

This is scale calibration on the project's own frame — the same evidence class as the dispersion priors and the understood anchors, and explicitly not an independent norm. There is no normative DS production ratio.

The low anchor is left alone. The data line implies 0.053 there against the prior's 0.126, which is a real discrepancy, but the fitted trend value is 0.117 and sits comfortably inside the prior with contraction 0.81 — the posterior is not asking for the move, and making it would be changing a parameter the likelihood is already informing well. It is recorded in §8 as an open question rather than acted on.

## 5. `eta_q`, and the correction to the earlier diagnosis

> [!CAUTION]
> **The _explanation_ below is superseded by [202608042030](202608042030-q-mean-extrapolation.md) §3; the widening itself stands, pending recheck.** This section attributes the pinned `eta_q` to `logit(q)` being S-shaped across the DS age range, so that only the GP could carry the curvature. The measurement that replaces it is more specific: the GP is _idle_ where the data are (+0.08 logits at 48 months) and spends its amplitude almost entirely hauling the mean back from its asymptote above 84 months, where the extrapolated line reaches `q` = 0.993 at 115. The age-span pattern in the table below is real and still discriminates the models correctly — a longer domain means more extrapolation past the high anchor — but the mechanism is extrapolation, not developmental curvature. Whether `eta_q` still needs `HalfNormal(0.8)` once the mean is clamped is open; see that note's §8.

§5 of [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) recommended reverting `eta_q` from `HalfNormal(0.20)` to `HalfNormal(0.4)`, on the argument that the tightening was a sampler patch for a `q` slope/intercept ridge, that VG10's Option D anchoring already removes that ridge structurally, and that the conflict was "confined to the models with both study and subject random effects (VG07–VG10, VG16)".

**The recommendation was right and the reasoning was wrong.** Fitted `eta_q` across the family, against `HalfNormal(0.20)`:

| model                      | `eta_q` median | prior CDF | contraction | subject RE on `q` | Option D |
| -------------------------- | -------------: | --------: | ----------: | ----------------- | -------- |
| VG05                       |          0.390 |     0.952 |        0.07 | no                | no       |
| VG07                       |          0.413 |     0.963 |        0.03 | no                | no       |
| VG08                       |          0.455 |     0.979 |        0.09 | no                | no       |
| VG09                       |          0.486 |     0.986 |        0.15 | yes               | no       |
| VG10                       |          0.477 |     0.984 |        0.15 | yes               | yes      |
| VG14                       |          0.387 |     0.951 |        0.09 | —                 | —        |
| VG15                       |          0.452 |     0.978 |        0.16 | yes               | yes      |
| VG16                       |          0.482 |     0.985 |        0.14 | yes               | yes      |
| **VG13** (TD, 8–18 months) |      **0.135** | **0.572** |        0.01 | yes               | yes      |

It is not a subject-random-effect phenomenon: VG05, VG07 and VG08 carry no subject RE on `q` and press just as hard. It is not an Option D phenomenon: the anchored models sit alongside the unanchored ones. §5's characterisation of VG05 as "not pressing the prior" at 0.352 came from a two-chain `dev` fit; on the current trace it is 0.390 at prior CDF 0.952.

What separates the models is **age span**, which is §3's finding arriving from the other direction. Every model traversing the DS 8–115 month range has to render the whole `q` sigmoid and needs GP curvature to do it. VG13 is the only model that does not press the prior, and its window is 8–18 months — the bottom limb of the S, where a straight line on the logit scale is adequate. Contraction of 0.03–0.16 everywhere else says no model's data are informing this parameter at all; they are all reporting the prior.

### The control fit

VG10 at `test`, `eta_q_sigma = 0.4`, via `dataclasses.replace` into a scratch output root — the test §5 pre-registered. 5m 28s.

|                | model of record (0.20)       | control (0.4) |
| -------------- | ---------------------------- | ------------- |
| gate `passed`  | False                        | **True**      |
| divergences    | 2                            | **0**         |
| max R-hat      | 1.0110                       | 1.0093        |
| min ESS        | 283                          | **483**       |
| min BFMI       | 0.453                        | 0.471         |
| R-hat failures | `g_unit_u_hsgp_coeffs[2]`    | **none**      |
| ESS failures   | 4 understood GP coefficients | **none**      |

The ridge does not return, and `tau_subj_q` is unmoved (1.255 → 1.252, contraction 0.94 both). That last point matters: §5 warned that if `eta_q` inflated while `tau_subj_q` stayed high, the `q` GP and the subject effects would be aliased and no prior choice would help. Giving the GP room did not pull `tau_subj_q` down at all, which is evidence _against_ that aliasing rather than for it.

The failures that cleared were on the **understood** GP, while the change was to the **q** GP. The plausible mechanism is the `p_s = p_u × q` coupling — if `q` cannot bend, the understood GP is recruited to supply spoken curvature through `p_u` and fights the understood likelihood. That is a conjecture; nothing here tests it, and it should not be repeated as established.

`eta_q` at 0.4 is still pressing: 0.685, prior CDF 0.922, contraction 0.22. Better than 0.984/0.15, still a lower bound. **Implemented at `HalfNormal(0.8)`** rather than 0.4 on that basis.

### Widening `eta_q` does not fix the prior predictive

Prior median population spoken curve, same seed, three amplitudes:

| age (months) | fitted | `eta_q` = 0.2 |   0.4 |   0.8 |
| ------------ | -----: | ------------: | ----: | ----: |
| 12           |    0.4 |           4.4 |   4.4 |   4.3 |
| 54           |  209.3 |          95.8 |  95.9 |  95.8 |
| 84           |  406.8 |         326.9 | 321.7 | 314.8 |

Unmoved, because the GP is zero-mean and widening its amplitude widens the band symmetrically on the logit scale. The fraction of prior mass below the fitted value at 54 months is 91.6% under all three settings. `eta_q` governs whether the model _can_ represent the shape, not where the prior is centred; only the anchor change in §4 moves the curve.

## 6. A prediction that failed

Before the control fit, this investigation predicted that `p_slope_hi_q`'s conflict was an artefact of the squeezed GP — that the anchor was being levered out to buy slope the GP could not supply, and would relax once `eta_q` was loosened. It did not: 0.929 → 0.929, prior CDF 0.972 → 0.971, contraction 0.94 in both fits. The anchor is measuring something the GP was never going to absorb, which is why §4 recalibrates it directly.

## 7. Consequences

The combined effect on the prior predictive, `Beta(4, 1.2)` and `eta_q = 0.8` together, as fraction of prior mass below the fitted population value:

| age (months) |    12 |    24 |    36 |    48 |    54 |    60 |    72 |    84 |
| ------------ | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| before       |  8.7% | 29.7% | 63.7% | 88.9% | 91.6% | 88.7% | 76.2% | 65.6% |
| after        | 11.8% | 32.1% | 57.0% | 77.6% | 79.4% | 73.0% | 56.5% | 46.2% |

Improved at every age and worsened at none — unlike the understood recalibration, where lifting the line made the 12–18 month end worse. The pivot here is about the unchanged low anchor, so steepening the line lowers the young extrapolation and raises the middle at the same time.

It is a reduction, not a cure. At 54 months 79.4% of prior mass still sits below the fitted value, and the prior median spoken at 12 months is 3.5 words against a fitted 0.4. That residual is the S-shape a straight logit-linear mean cannot produce, and it is the same structural limitation §5 of [202608041216](202608041216-ds-understood-trajectory-prior.md) records for the understood trajectory. The GP now has the amplitude to carry it in the posterior; the prior mean still does not have the form to centre it.

## 8. Open

1. **Refit the family.** VG05, VG07, VG08, VG09, VG14, VG15 and VG16 all carry both new priors with stale fits. Only VG10 is refitted (§9).
2. **The `q` low anchor.** The observed-ratio line implies a trend `q(24)` of 0.053 against the prior's median 0.126 and a fitted 0.117. Left unchanged because the posterior is not asking for it, but the disagreement between the data line and the fitted trend at the low end is not explained. Two candidate reasons, neither checked: the model's linear component is fitted over the whole 8–115 month grid while the observed line covers 18–72, and the median of a per-child ratio is not the ratio for the median child.
3. ~~**`eta_q` may still be pressing at 0.8.**~~ Answered for VG10 in §9 — 0.855 at prior CDF 0.745, contraction 0.15 → 0.35. Still open for the other seven models, and contraction of 0.35 means it is not yet strongly data-determined.
4. ~~**`p_slope_hi_q` under the new prior.**~~ Answered for VG10 in §9 — unmoved at 0.931, prior CDF 0.819. Still to confirm on the rest of the family.
5. **The `q` overshoot at 36–54 months** (§9): the fitted `q` runs above the observed band medians there (0.190 against 0.128 at 36 months, 0.344 against 0.241 at 42), in both the old and new fits. Pre-existing and unexplained. The likeliest candidates are that the model's `q` is a median-child parameter while the observed figure is a median of per-child ratios, and that the bands are thinning fast in that range — neither checked.
6. **The log-age mean variant** (open item 2 of [202608041216](202608041216-ds-understood-trajectory-prior.md)) would help understood but not `q`, per §3. A mean form that can represent an S — a two-parameter logistic in age, or a third `q` anchor — is the corresponding proposal for the spoken side, and is a graph change.
7. ~~**VG13 is untouched but its `eta_q` default changed underneath it.**~~ Checked and closed. VG13 now sets `eta_q_sigma=0.20` explicitly rather than inheriting the family default. Its serialised definition was compared field by field against the one recorded in its own `fit_manifest.json`: identical throughout, so its existing fit remains valid and no refit is required.

## 9. VG10 refit

VG10 refitted at `test` (4 chains × 2,000 draws, seed 47, no overrides) with both changes in place. 5m 42s, rendered. The full test suite passes.

### VG10 passes its convergence gate

|                | before                       | after    |
| -------------- | ---------------------------- | -------- |
| gate `passed`  | False                        | **True** |
| divergences    | 2                            | **0**    |
| max R-hat      | 1.0110                       | 1.0084   |
| min ESS        | 283                          | **433**  |
| min BFMI       | 0.453                        | 0.468    |
| R-hat failures | `g_unit_u_hsgp_coeffs[2]`    | **none** |
| ESS failures   | 4 understood GP coefficients | **none** |

This reproduces the §5 control fit at the higher amplitude. VG10 has carried a REVIEW verdict throughout the 2026-08-03 and 2026-08-04 work; this is the first clean pass.

> [!CAUTION]
> **The clean pass is seed-specific and should not be read as a property of this change** — corrected by [202608042030](202608042030-q-mean-extrapolation.md) §7, later the same day. Refitting this exact configuration at two further sampler seeds gives max R-hat 1.0115 with 1 divergence and 1.0132 with 4, both failing the gate: it passes 1 time in 3. Max R-hat sits at 1.008-1.014 against a 1.01 threshold across every VG10 `test` fit measured, so pass or fail turns on the seed. The parameter results in this section are unaffected — the anchor and `eta_q` movements reproduce across seeds — but the diagnostics table above records one draw from a noisy distribution, not an improvement delivered by the recalibration.

### The priors now contain their posteriors

| parameter       | posterior before | posterior after | prior CDF before | prior CDF after | contraction before | contraction after |
| --------------- | ---------------- | --------------- | ---------------- | --------------- | ------------------ | ----------------- |
| `p_slope_hi_q`  | 0.929            | 0.931           | 0.972            | **0.819**       | 0.94               | 0.93              |
| `eta_q`         | 0.477            | **0.855**       | 0.984            | **0.745**       | 0.15               | **0.35**          |
| `p_slope_low_q` | 0.117            | 0.122           | 0.464            | 0.487           | 0.81               | 0.80              |
| `tau_subj_q`    | 1.255            | 1.253           | 0.598            | 0.597           | 0.94               | 0.94              |
| `eta_u`         | 0.900            | 0.916           | 0.880            | 0.886           | 0.28               | 0.29              |

`p_slope_hi_q` does not move (0.929 → 0.931) while its prior CDF falls from 0.972 to 0.819. That is the signature §4 predicted: the data were already setting this parameter, and the old prior was fighting them from outside its own 95th percentile. `eta_q` rises from 0.477 to 0.855 with contraction more than doubling, 0.15 → 0.35 — it was censored at 0.20, and the data are now informing it rather than merely reporting the prior. Open item 3 is answered for VG10: 0.8 is sufficient, though contraction of 0.35 means it is still only partly data-determined.

> [!CAUTION]
> **§5 of [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) set a threshold that this fit crosses.** It warned that if `eta_q` inflated past roughly 0.8 under a wider prior while `tau_subj_q` stayed high, the `q` GP and the subject effects on `q` would be structurally aliased and no prior choice would help. `eta_q` is 0.855 and `tau_subj_q` is unchanged at 1.253, so the stated condition is met. The accompanying signature is absent, however, and pointedly so: aliasing would show as contraction staying near zero and poor geometry, and instead contraction rose from 0.15 to 0.35 while divergences went to zero and every R-hat and ESS failure cleared. The threshold is recorded as crossed rather than quietly passed over, but on this evidence it is not diagnostic of what §5 feared.

### It did not move the answer

| age (months)           |   12 |   24 |   36 |   48 |   54 |   60 |   66 |   72 |    84 |   90 |
| ---------------------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ----: | ---: |
| spoken, change (words) | −0.1 | −0.1 | +0.4 | +2.3 | +1.5 | +3.9 | +5.6 | +4.3 | −11.1 | −8.7 |

At most 5.6 words below 78 months and 11.1 at the top, against a curve running from 0.3 to 438 words; `q` moves by at most 0.035 and understood by at most 14 words. As with the understood recalibration, correcting an off-centre prior changes the prior predictive and leaves the reported answer where it was. Nothing previously reported from VG10 is revised.

The refitted `q` also tracks the directly observed ratio closely at the ages where the evidence is thickest — 0.483 against 0.475 at 48 months, 0.715 against 0.706 at 66, 0.723 against 0.733 at 72 — while running above it at 36–54 (0.190 against 0.128; 0.344 against 0.241). That overshoot is unchanged from the previous fit and so is not caused by this recalibration; it is not explained here.
