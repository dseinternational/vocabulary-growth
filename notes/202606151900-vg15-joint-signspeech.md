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
2. **Study random intercepts** on `f_U`, `g` (sign), `h` (speak) — the VG07-VG10
   pattern — so the reported (population) age curves are separated from study
   composition.

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

- `p_U`, `r = P(sign|understood)`, `q = P(speak|understood)`: linear trend + HSGP
  on logit scale; signed ratio reuses VG14's hump-capable spec (flat-pinned mean,
  loose+short GP). Priors seeded from VG14.
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
at `target_accept = 0.95`).

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
