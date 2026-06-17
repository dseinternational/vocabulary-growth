# VG15: joint sign/speech model (issue #49, Option 3)

Status: 2026-06-15, build + first fit. Author: Ethan (with Claude Code). Numbers
from `output/models/VG15-age-joint-signspeech-ds/` and the source data in
`data/`.

## 1. What VG15 is (and why)

VG14 modelled signing and speaking as two production ratios hanging off the
understood trajectory, but assumed they were **conditionally independent given
age**, so its total expressive vocabulary `p_any` was only an independence
**upper bound**, and its signed peak age was **study-confounded** (no study
random effects). VG15 fixes both:

1. A scalar **Plackett association** `psi` linking sign and speech _within_
   understood words, identified from uk_02's four-cell cross-tab. This yields a
   **data-identified** `p_any = p_U·(r + q − pi_both)` instead of an upper bound,
   plus the four-cell composition trajectory (sign-only → both → speak-only).
2. **Study and subject random intercepts** on `f_U`, `g` (sign), `h` (speak) —
   the VG07-VG10 pattern, with VG10's GP anchor for stability — so the reported
   (population) age curves are separated from study composition and within-child
   clustering. (Subject REs + anchor added in issue #59; see §7. The §5/§6
   results below are the earlier study-RE-only fit.)

Built in a new isolated engine `common_joint_modality.py` (does not import or
modify the bivariate / trivariate engines; VG01-VG14 untouched). Same
`*_all`-as-plain-tensors memory discipline as VG14. r/q/p_U prior specs are
seeded from the (uk_06-included) VG14 fit.

## 2. Data

DS only. Non-uk_02 studies contribute marginal understood/spoken/signed (from the
merged view). uk_02 is taken from its raw CSV and split:

- **Four-cell rows** (margins reconcile: `signed = signed_only + signed_spoken`,
  `spoken = spoken_only + signed_spoken`, positive cell-sum): ~62 rows, 19–56 mo.
  These feed the **Dirichlet-Multinomial** and identify `psi`. (The issue quoted
  ~56; the margin-reconciling set used here is a few rows larger — it does not
  change the science. Cell means as fractions of understood: neither ~0.43,
  sign-only ~0.31, speak-only ~0.12, both ~0.18.)
- **Marginal-only rows** (no cross-tab): ordinary marginal signed/spoken/understood.

uk_06 signed counts are **included** (inherits the VG14 decision); they contribute
marginal signed only (no cross-tab).

## 3. Model + priors

- `p_U`, `r = P(sign|understood)`, `q = P(speak|understood)`: HSGP on the logit
  scale. `p_U`/`q` have a linear trend; the signed mean `r` is **intercept-only**
  (no age slope, matching VG14 — see §6), with a loose+short GP carrying the hump.
  Priors seeded from VG14.
- `psi`: `log psi ~ Normal(0.3, 0.5)` — weakly positive (uk_02 shows both 0.18 >
  r·q 0.14) but spanning independence `psi = 1`.
- Dirichlet-Multinomial concentration: `log conc ~ Normal(3.0, 1.0)`.
- Study random intercepts: `tau_{u,q,sign} ~ HalfNormal(0.5)`, non-centred.
- Likelihoods: understood / spoken / signed Beta-Binomial marginals +
  uk_02 four-cell Dirichlet-Multinomial.

## 4. Caveats (state up front)

- `psi` is a **single scalar** resting almost entirely on ~62 uk_02 rows
  (19–56 mo) — descriptive/exploratory; no age-varying association at this n.
- Composition and `p_any` outside ~19–56 mo lean on the marginal trajectories +
  scalar-`psi` extrapolation.
- uk_06 contributes marginals only.

## 5. Fitted result (rep)

Fitted at `--config rep` (6 chains × 6000 draws). **Clean: 0 divergences /
36 000, max r̂ = 1.001, min ESS ≈ 5.3k** (dev had 2/1000 divergences; they clear
at `target_accept = 0.95`). _(This section is the flat-anchor signed-mean fit; the
current model is the intercept-only refit in §6 — ψ, the four-cell composition and
`p_any` are substantively the same; only the signed-mean parameterisation
differs.)_

**Association.** `psi` median **1.78**, 90% HDI **[1.21, 2.44]**, **P(psi > 1) =
0.997** — a clear positive within-understood sign-speech association. Words a
child understands are signed-and-spoken together more often than independence
predicts, consistent with signing bridging into speech word-by-word.

**Data-identified `p_any` vs the VG14 independence bound** (expected words):

| Age (mo) | identified p_any | independence bound |
| -------: | ---------------: | -----------------: |
|       24 |               42 |                 44 |
|       36 |              107 |                112 |
|       48 |              191 |                199 |
|       60 |              259 |                265 |

The identified total sits a few words **below** the independence bound at every
age — the positive association means the union is smaller than independence
implies, so VG14's `p_any` was indeed a (mild) over-estimate. The gap is modest
here because the model's in-window `r` is also lower than VG14's (see below).

**Four-cell composition** (fractions of understood, study REs marginalised) shows
the migration directly: sign-only 0.25 → 0.19 → 0.10 and speak-only 0.09 → 0.17 →
0.35 across 24 → 36 → 48 mo, with sign+speech peaking mid (0.07 → 0.12 → 0.18).

**Does adding study REs sharpen the signed peak?** No — it **flattens** it. The
population signed ratio `r(a)` peaks at ~0.36 around 30 mo (vs VG14's ~0.46), is
broad and flat across 18–60 mo, and crosses `q(a)` at ~37 mo. Once study
composition is absorbed by the random intercepts, the apparent age-peak in
signing is _lower and less pronounced_ — confirming the VG14 finding that the
signed peak was partly study-driven, not a sharp developmental feature. The
robust story is unchanged: signing rises from near-zero in infancy to a
substantial fraction of understood words by age 2–3 and hands off to speech in
the preschool years.

Key figures: `psi_posterior.svg`, `four_cell_composition.svg`,
`p_any_identified_vs_bound.svg`, `signed_vs_spoken_rate.svg`, `uk02_cell_ppc.svg`.

## 6. Signed marginal mean was prior-dominated → intercept-only refit (current)

VG15 inherited VG14's tight `Beta(15,90)` signed anchors, and — exactly as in
VG14 — they were **prior-dominated**: posterior sd ≈ prior sd (`p_slope_low_sign`
0.169 ± 0.036, `p_slope_hi_sign` 0.129 ± 0.030, both ≈ the 0.034 prior sd), so
the signed grand-mean **level** was set by the prior, not the data, while the
spoken anchors were data-informed. Applied the same fix as VG14 (PR #52
`da5750a`): the signed marginal mean is now **intercept-only** (drop the age
slope — the actual extrapolation failure mode) with one weakly-informative
intercept `intercept_sign ~ Normal(logit(0.15), 0.75)`. The **study random
intercepts are untouched** (`tau_sign` is the largest RE in the model, ~0.95);
with the REs handling between-study level, freeing the grand-mean intercept is
even more clearly correct.

**Result (rep, intercept-only):** clean — **2 / 36 000 divergences** (0.006%, a
documented near-zero rate), max r̂ = 1.001, min ESS ≈ 6.1k.

| quantity                   | flat-anchor                  | intercept-only                 |
| -------------------------- | ---------------------------- | ------------------------------ |
| `intercept_sign` posterior | (pinned) 0.17 / 0.13 anchors | mean −1.59 ≈ 0.17, **sd 0.48** |
| `eta_sign`                 | ~1.0                         | 0.81                           |
| `tau_sign` (study RE)      | ~0.95                        | 0.95 (unchanged)               |
| ψ median [90% HDI]         | 1.82 [1.26, 2.51]            | 1.78 [1.19, 2.42]              |
| P(ψ > 1)                   | 0.997                        | 0.997                          |

The intercept posterior sd (0.48) is now meaningfully below its prior sd (0.75) —
the data is speaking. `eta_sign` dropped (the free intercept carries level the GP
was lifting) and is not pinned. **ψ, the four-cell composition and the
data-identified `p_any` (still below the independence bound) are substantively
unchanged** — this is a level/uncertainty correction, not a re-estimation of the
association. The peak does **not** relocate; as throughout, study REs (not prior
tuning) are the lever for the signed level/peak. The signed specs of VG14
(trivariate) and VG15 (joint) are now consistently intercept-only.

## 7. Subject random intercepts + VG10 stabilisation (issue #59, 2026-06-17)

Study REs alone left a within-child pseudoreplication artefact: the understood
population median showed a spurious ~42–60 mo plateau (VG14 worst; VG07 — the
study-RE-only proxy — a residual shoulder; VG10, with subject REs, smooth). Added
**subject random intercepts** on `f_U`, `q` and `r` (non-centred, observation
level) plus VG10's stabilisation — GP anchor on `f_U`/`q` at 54 mo and tighter
q-anchor priors (`p_slope_low_q ~ Beta(3,22)`, `p_slope_hi_q ~ Beta(20,4)`).

**Design point (keeps ψ identified):** subject REs are applied to the **marginal**
Beta-Binomial likelihoods only. The uk_02 four-cell rows are aggregate cross-tabs
with no child id, so the four-cell Dirichlet-Multinomial keeps the study-level
rates `q_obs`/`r_obs`. Rows without a real subject id (uk_02 aggregate rows) get
singleton subject codes (prior-drawn, no pooling, harmless).

**Subject coverage:** 581 subjects, 335 single-observation, **246 with repeated
observations** — enough repeated structure for the subject REs to bite.

**Dev fit (2 chains × 500):** clean run, ψ ≈ 1.77 (unchanged), four-cell machinery
intact. **Headline outcome achieved — the understood-median plateau is gone:**
growth stays **+6.4 to +9.3 words/mo across 42–60 mo** (smooth, near-monotone;
VG14 went flat/negative there). Convergence at dev is borderline (max r̂ ≈ 1.06,
min ESS ≈ 53, concentrated in the understood linear-trend anchors `slope_u` /
`p_slope_hi_u` — the usual GP/trend/RE ridge under short tuning).

**Test fit (4 chains × 2000 tune, ~4 min, 0 divergences):** convergence tightens
to **max r̂ 1.017** — only the three understood linear-trend anchors (`slope_u` /
`intercept_u` / `p_slope_low_u`) sit marginally above 1.01 (≤ 1.017, min ESS ≈
467), while ψ (r̂ 1.000, ESS 18.6k) and all `tau_subj_*` (r̂ ≤ 1.003, ESS ≈ 2k)
are clean. Plateau removal holds (+6.9–7.8 words/mo across 42–60 mo). Those trend
anchors are the slowest-mixing globals and should clear ≤ 1.01 at rep.

**Rep fit (6 chains × 6000 tune/draw, `target_accept = 0.95`, ~14 min, 4.0 GB
trace):** fully clean — **max r̂ 1.0047 (0/40 over 1.01), 0 / 36 000 divergences,
min ESS ≈ 1.3k**; ψ 1.76 (r̂ 1.000, ESS 63k), `tau_subj_*` r̂ ≤ 1.001 (ESS ≈
6k). The understood linear-trend anchors that were marginal at test cleared
(`slope_u` r̂ 1.005). Plateau removal holds (+6.9–7.7 words/mo, 42–60 mo). Report
renders. **All #59 acceptance criteria met.**
