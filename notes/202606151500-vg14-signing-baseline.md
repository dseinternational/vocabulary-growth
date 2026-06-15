# VG14 signing baseline: trivariate understood + spoken + signed (DS)

Status: 2026-06-15, post-build. First fit of the issue #49 **Option 1** trivariate model.
Author: Ethan (with Claude Code). All numbers below are taken from the fitted output in
`output/models/VG14-age-understood-spoken-signed-ds/` (`--config rep`), verified against the
`diagnostics.csv`, `posterior_summary_*.csv` and `trace.nc` files — not from prior prose.

> **Partly superseded — see `202606151700-vg14-signed-ratio-shape-and-p-any-bias.md`.**
> A data check found the baseline signed ratio was mis-shaped and `p_any` biased. The
> signed-ratio spec was re-fitted (flat-pinned mean + looser, shorter GP), so **§3
> (signed priors) and the §5 crossover _shape_ are superseded**: `r(a)` is now a
> **hump** (near zero in infancy, rising to ~0.4–0.5 of understood words by age 2–3),
> not the monotone decline reported below; `r(12)` is ~0.20, not 0.58. The fitted peak
> age (~24–30 mo) and the recede are **study-confounded and not identifiable** here (the
> 48–60 mo window is single-study uk_02; no study REs) — only the rise and the ~39-mo
> crossover are robust; see the review note. Also: **uk_06 signed is now excluded** by
> default (414 → 403 signed obs; different construct), and **`p_any` is an
> independence-based upper bound** (~3.7 pp high vs the uk_02 observed union). The §2
> data, §4 diagnostics shape, and §7–§8 are unchanged.

> **Numbering note (read first).** Three earlier notes (`202606121200-kappa-variance-vg14-handoff.md`,
> `202606121500-vg14-scoping-note.md`, `202606121700-vg14-assessment-vg15-proposal.md`) use "VG14"
> for a _different_ idea — the DS-vs-TD dispersion contrast `Δκ(age) = κ_DS(age) − κ_TD(age)` — and
> "VG15" for the uk_02 sign-composition model. **Issue #49 is authoritative and supersedes that
> numbering: VG14 is the trivariate signing model described here, and VG15 is the uk_02 four-cell
> multinomial.** The dispersion-contrast model now needs a fresh id (VG16+); it is not built.

## 1. What VG14 is

A DS-only trivariate extension of the bivariate (understood + spoken) engine, adding signing as a
third production modality via a second production-ratio curve:

```
p_U(a)    = sigmoid(f_U(a))                        # proportion of checklist understood
q(a)      = sigmoid(h(a))                          # fraction of understood words SPOKEN
r(a)      = sigmoid(g_sign(a))                     # fraction of understood words SIGNED
p_S(a)    = p_U(a) * q(a)                           # p_S    <= p_U by construction
p_Sign(a) = p_U(a) * r(a)                           # p_Sign <= p_U by construction
p_any(a)  = p_U(a) * (1 - (1 - r(a)) * (1 - q(a)))  # total expressive (sign ⟂ speech | age)
```

Three Beta-Binomial likelihoods (understood, spoken, signed), each with its own age-varying
dispersion `kappa(z) = kappa_min + exp(a_kappa + b_kappa·z)`, `b_kappa = -b_kappa_mag`. Built in a
new isolated module `src/vocab_growth/models/common_trivariate.py` — the bivariate engine
(`common_bivariate.py`, VG05–VG13) is untouched.

**Caveat (the Option 1 limitation):** signing and speaking are assumed _conditionally independent
given age_. The two ratios share `p_U(a)` but are otherwise modelled independently, and `p_any`
is computed under that independence. In reality a word is often signed before it is spoken, so
the within-understood sign–speech association is real and unmodelled here. VG15 (uk_02 four-cell
multinomial) is the planned relaxation; VG14's posteriors are intended to seed it.

## 2. Data

DS only (signing has no TD counterpart). Loaded via
`load_data(Population.DOWN_SYNDROME, columns=["age","understood","spoken","signed"])`; rows kept
where at least one of understood/spoken/signed is observed.

| Quantity            |   n |
| ------------------- | --: |
| Total observations  | 950 |
| Understood observed | 704 |
| Spoken observed     | 949 |
| Signed observed     | 414 |

Signed counts are sparse and small (observed range 0–641, mean ≈ 65, ages 15–115 months; signing
concentrated in the early years). All `signed ≤ understood` in the data, consistent with the
`p_Sign ≤ p_U` structural constraint. `build_model` validates `0 ≤ signed ≤ n_trials`.

## 3. Priors chosen

- **Understood (U):** reused from VG05 — `p_slope_low_u` Beta(1, 10), `p_slope_hi_u` Beta(1.1, 1.1),
  `ell_unit_u` Beta(3, 3), `eta_u` HalfNormal(0.4).
- **Spoken ratio q:** bivariate defaults — `p_slope_low_q` Beta(1, 1.5), `p_slope_hi_q` Beta(2, 1.2),
  `eta_q` HalfNormal(0.4).
- **Signed ratio r (new, weakly-informative; signing is sparse):** `p_slope_low_sign` Beta(1, 4)
  (≈0.2 at the 24-month anchor), `p_slope_hi_sign` Beta(1, 3) (≈0.25 at the 84-month anchor),
  `ell_unit_sign` Beta(3, 3), and a deliberately **tight** GP amplitude `eta_sign` HalfNormal(0.2)
  (vs 0.4 for q/U). `kappa_sign` uses the shared `KappaPriorParams` defaults.

The data move the signed ratio strongly: posterior `p_slope_low_sign` mean 0.44 (24 mo) and
`p_slope_hi_sign` mean 0.057 (84 mo) — i.e. a strong _decline_ in the signed fraction with age,
the opposite direction to the (rising) spoken ratio. `eta_sign` posterior mean 0.19 (HDI
[0.02, 0.44]) — the tight prior was not fighting the data. Signed dispersion: `kappa_min_sign`
1.07, `a_kappa_sign` 0.67, `b_kappa_mag_sign` 0.79.

## 4. Diagnostics — dev vs rep

| Config | draws × chains | max r̂ | r̂ > 1.01 | min ESS bulk | min ESS tail | divergences |   wall |
| ------ | -------------- | ----: | -------: | -----------: | -----------: | ----------: | -----: |
| dev    | 500 × 2        |  1.03 |        5 |         ~118 |         ~140 |           0 |  2m33s |
| rep    | 6000 × 6       | 1.001 |        0 |         6746 |         8555 |   0 / 36000 | 25m20s |

The dev fit is for eyeballing only; the rep fit is clean — `r̂ ≤ 1.001` on every parameter,
healthy ESS (min bulk ≈ 6.7k), and **zero divergences** at `target_accept = 0.95`.

## 5. The r(a) / q(a) crossover (headline)

The fraction of understood words that are _signed_, `r(a)`, starts high and falls monotonically;
the fraction _spoken_, `q(a)`, starts low and rises. They cross at **~38–40 months** — the
sign→speech hand-off. (Medians, from `posterior_summary_q.csv` / `posterior_summary_r.csv`.)

| Age (mo) | q(a) spoken | r(a) signed |
| -------: | ----------: | ----------: |
|       12 |       0.100 |       0.579 |
|       24 |       0.196 |       0.472 |
|       36 |       0.240 |       0.349 |
|       42 |       0.372 |       0.285 |
|       60 |       0.704 |       0.139 |
|       90 |       0.872 |       0.039 |

`q` overtakes `r` between 36 and 42 months (at 36 mo signed 0.349 > spoken 0.240; at 42 mo spoken
0.372 > signed 0.285).

Read: before ~3 years a _larger share_ of a DS child's understood words are signed than spoken;
by ~3.5 years speech overtakes signing, and signing recedes (but does not vanish) thereafter. This
is exactly the clinical picture signing-as-a-bridge is meant to describe — now a posterior quantity
with uncertainty. See `sign_speech_crossover.svg`.

## 6. Total expressive vocabulary p_any (query ages)

`p_any(a) = p_U·(1 − (1 − r)(1 − q))` — probability a word is produced in _any_ modality, under the
conditional-independence caveat (§1). Expected counts out of 800 (medians, `posterior_summary_p_any.csv`):

| Age (mo) | p_any median | E[words] |
| -------: | -----------: | -------: |
|       12 |        0.019 |       15 |
|       24 |        0.068 |       54 |
|       36 |        0.169 |      135 |
|       48 |        0.261 |      209 |
|       60 |        0.302 |      242 |
|       72 |        0.393 |      314 |
|       84 |        0.468 |      374 |
|       90 |        0.516 |      412 |

`p_any` exceeds spoken-alone at every age (most in the early years), quantifying the Mason-Apps
point that spoken scores understate DS communicative vocabulary — strongest where signing is most
active. Treat the early-age values as a lower-uncertainty _floor_: under conditional independence
`p_any` is the largest it can be for given marginals, so the real correlated total is somewhat
lower; VG15 will pin this down.

## 7. Implementation notes

- **Isolation:** all signed logic lives in `common_trivariate.py` + `TrivariateModelDefinition` in
  `definitions.py` + the `model_vg14.py` wrapper. No edits to `common_bivariate.py` /
  `common_bivariate_re.py` / VG01–VG13.
- **Memory:** the full-grid (`n_all`) intermediates (`f_u_all`, `p_u_all`, `g_*`, `q_all`, `r_all`,
  `p_any_all`, …) are kept as **plain tensors, not stored `pm.Deterministic` nodes** — only the
  obs/plot/query slices are recorded and extracted. This differs from the bivariate engine (which
  stores the `*_all` arrays) and was necessary: the first rep attempt (6×6000 with the full arrays
  stored) was OOM-killed at 16 GB. With the slices-only trace the rep fit completes in ~25 min and
  ~clean memory. Worth porting back to the bivariate engine if those models grow.
- The rep config (6 chains × 6000 draws, `target_accept` 0.95) is memory-heavy; budget ≥16 GB.

## 8. Next (out of scope here)

VG15 / Option 3: the uk_02 four-cell (sign-only / sign+speech / speech-only / neither)
Dirichlet-Multinomial, relaxing the conditional-independence assumption and estimating the
within-understood sign–speech association directly. VG14's `r(a)`, `q(a)` and `p_any` posteriors
seed it.
