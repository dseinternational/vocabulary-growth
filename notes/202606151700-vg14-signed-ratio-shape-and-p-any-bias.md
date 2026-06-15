# VG14 review: signed-ratio shape, p_any bias, and uk_06

Status: 2026-06-15, post-baseline review. Checks the first VG14 fit
(`202606151500-vg14-signing-baseline.md`) against the raw source data. Author:
Ethan (with Claude Code). All numbers are computed from `data/` (the merged
`vocab_combined` view and `data/vocab_data_uk_02.csv`) and the fitted output in
`output/models/VG14-age-understood-spoken-signed-ds/`.

Two problems with the baseline fit, and what was done about them.

## 1. r(a) was mis-shaped (monotone decline, should be a hump)

The baseline fit gave a signed ratio `r(a)` that **declined monotonically** from
0.58 at 12 months. The data says otherwise: the signed fraction of understood
words is a **hump** — near zero in infancy, rising through toddlerhood to a peak
in the preschool years, then receding as words move into speech.

Empirical signed fraction of understood words, by age bin (DS, **uk_06
excluded**, rows with both understood and signed observed):

| Age bin (mo) |   n | median per-child ratio | ratio of means (signed/understood) |
| ------------ | --: | ---------------------: | ---------------------------------: |
| < 18         |   3 |                  0.000 |                              0.000 |
| 18–24        |  38 |                  0.019 |                              0.192 |
| 24–30        |  30 |                  0.212 |                              0.312 |
| 30–36        |  60 |                  0.351 |                              0.409 |
| 36–42        |  30 |                  0.415 |                              0.427 |
| 42–48        |  19 |                  0.343 |                              0.360 |
| 48–54        |  21 |                  0.493 |                              0.523 |
| 54–60        |   5 |                  0.456 |                              0.412 |

Near zero below ~24 months, a broad plateau ~0.40–0.52 over 30–54 months, and a
decline after. **There is no signing data below 18 months** (3 obs at 15–17 mo,
all zero), so the baseline `r(12)=0.58` was extrapolation below the data floor.

**Cause.** Two structural defects in the signed block:

- The logit-linear mean trend is **monotone** between its anchors. The cheap way
  to fit the dominant feature (signing fades from ~50 mo to near zero by 115 mo,
  where 117 obs sit) is a declining mean — which then extrapolates _upward_ below
  24 months.
- The signed GP amplitude `eta_sign` was set **tight** (≈0.19), so the GP could
  not bend the ratio into a hump against that declining mean.

**Fix (applied).** In the signed-ratio block:

- Pin the mean trend **flat**: tight, equal low-fraction anchor priors
  (`Beta(15, 90)` at both 24 and 84 mo) so the linear trend cannot become a
  monotone decline.
- **Loosen** the GP amplitude (`eta_sign` HalfNormal scale 0.2 → 1.0) so the GP
  has the amplitude to carry the rise-then-fall on the logit scale.
- **Shorten** the signed GP lengthscale (`ell_unit_sign` → ~9 mo, vs ~12 mo for
  U/q) so the late signing can stand apart from the post-60-month collapse rather
  than being smoothed into a monotone decline.

### Remaining limitation — peak age is study-confounded, not identifiable here

After the re-spec the hump peaks at ~24–30 mo, not the ~48–60 mo the empirical
median bins suggested. This is not a tuning failure; the peak age cannot be
identified from these data with this model. Composition by window (population
ratio Σ signed / Σ understood):

| window   |   n | sources                      | pop. ratio |
| -------- | --: | ---------------------------- | ---------: |
| 24–36 mo |  90 | uk_04/uk_05/uk_02            |       0.38 |
| 36–48 mo |  49 | uk_02 (32), uk_01 (13)       |       0.40 |
| 48–60 mo |  26 | **uk_02 only** (all signers) |       0.50 |
| 60+ mo   |   1 | uk_01                        |       0.02 |

(Composition shown with uk_06 excluded, to isolate the 24–60 mo peak window;
uk_06 adds 11 heavy older-age signers at 60+ mo — see §3. The peak-window
argument below is uk_06-independent.)

The population ratio is still rising at 48–60 mo and only collapses when the
sample switches from uk_02 (ends ~56 mo, heavy signers) to uk_01/uk_06 — and once
uk_06's older signers are included, even the post-60 mo recede weakens (§3). The apparent "peak then recede" is largely a between-study boundary,
not a within-child developmental recede. VG14 has no study random effects (it
mirrors VG05), so the age curve absorbs this composition and cannot separate
"this age signs more" from "uk_02 children sign more."

- Do not force a later peak by excluding >60 mo data — that rides uk_02's
  range-end and manufactures a peak from one study's window.
- Robust claims (survive the confound): signing rises from near-zero in infancy
  to a substantial fraction (~0.4–0.5) of understood words by age 2–3, and
  crosses the rising spoken curve at ~39 mo. The peak age and the recede are
  uncertain and study-confounded, not a precise developmental month.
- Proper fix if peak timing matters: study random intercepts on the signed block
  (VG07–VG10 pattern), not more tuning or data exclusion. Candidate follow-up: a
  VG14-with-REs variant, or fold into VG15.

## 2. p_any over-stated the total (independence bias)

`p_any = p_U · (1 − (1 − r)(1 − q))` assumes signing and speaking are
conditionally independent given age. uk_02 (the only four-cell source) shows a
**positive** association, so independence over-states the union. Over 20–56
months (130 uk_02 rows, comprehension > 0):

| quantity                          | value |
| --------------------------------- | ----: |
| observed both modalities          | 0.177 |
| independence $r \cdot q$          | 0.140 |
| observed union                    | 0.571 |
| independence union $1-(1-r)(1-q)$ | 0.608 |

Observed both (0.177) > $r \cdot q$ (0.140) ⇒ positive association; observed union
(0.571) < independence union (0.608) ⇒ independence over-estimates by ~3.7 pp.

**Fix (Task C).** `p_any` is now labelled in the report as an **independence-based
upper bound**, and a validation plot (`p_any_validation.svg`) overlays the model's
`p_any / p_U` on the uk_02 observed union over 20–56 months, with the gap recorded
in `p_any_validation_gap.csv`. VG15 (the uk_02 multinomial) is what properly
identifies the association.

## 3. uk_06 signed is now INCLUDED (revised decision)

`uk_06`'s `signed` is a **real signing-production count** (11 obs, 60–115 mo,
often comparable to or exceeding spoken — e.g. ~387 signs at 60 mo). Signing
implies understanding, so "understands-and-signs" is itself a sign, not a
comprehension-adjacent construct. It is therefore **included in the signed
likelihood by default** (`TrivariateModelDefinition.include_uk06=True`, 414 signed
obs; the flag is kept for reversibility). The remaining open question for
Frank/Sue is **coding comparability** — are uk_06's signed counts coded the same
way as uk_02/04/05? uk_06 has no field dictionary, and its `imitated` column
sometimes exceeds `understood`, which warrants a check — but this is a
data-quality question, not a construct mismatch.

### Sensitivity: uk_06 in vs out

uk_06 are heavy older-age signers, so including them raises the old-age signed
estimate and weakens the post-peak recede (rep medians):

| Age (mo) | r(a) uk_06 out | r(a) uk_06 in | signed count out → in |
| -------: | -------------: | ------------: | --------------------: |
|       60 |          0.103 |         0.142 |               33 → 46 |
|       72 |          0.043 |         0.065 |               19 → 26 |
|       90 |          0.030 |         0.043 |               14 → 21 |

The young/mid shape is essentially unchanged — **peak ~0.46 at ~30 mo, crossover
at ~39 mo, rise to ~0.45–0.5 by age 2–3 all hold** — so including uk_06 does not
move the headline; it only lifts the old-age tail (substantively interesting and
consistent with DSE's signing emphasis: some children keep signing into school
age). Both fits are clean (0 divergences / 36 000, r̂ ≤ 1.001). The §1
study-confound on the exact peak age is unaffected (uk_06 sits at 60+ mo, past
the 48–60 mo window that drives the peak question).

## 4. Reporting (Task B)

The signed-rate and crossover plots now **shade ages outside ~18–54 months** as
extrapolation (no signing data < 18 mo; data thin after 54 mo), and the report
prose describes the result as a hump (signing rises then yields to speech), not
"signing dominant at 12 months."

## 5. Fitted result (rep)

Re-fitted at `--config rep` (6 chains × 6000 draws) after the changes above.
Diagnostics are clean: **0 divergences / 36 000**, max r̂ = 1.001, min ESS ≈ 6.5k.
Signed-block posteriors: `eta_sign` 1.15 (the loosened GP is used), `ell_sign`
≈ 10 mo, anchors pinned flat at 0.16 / 0.12 (the tight prior held).

`r(a)` is now a **hump**, not a monotone decline:

| Age (mo) | r(a) signed | q(a) spoken |
| -------: | ----------: | ----------: |
|       12 |       0.205 |       0.098 |
|       18 |       0.332 |       0.164 |
|       24 |       0.478 |       0.195 |
|       36 |       0.341 |       0.243 |
|       48 |       0.242 |       0.526 |
|       60 |       0.103 |       0.705 |
|       90 |       0.030 |       0.875 |

Near zero in infancy, rising to a substantial fraction (~0.48) by age 2–3, then
declining. The fitted peak is ~24–30 mo, but **the peak age and the decline are
not identifiable from these data** — they are confounded with study composition
(see the limitation above: the 48–60 mo window is single-study uk_02, the
population ratio is still rising there, and the model has no study random
effects). What is robust is the rise to ~0.4–0.5 by age 2–3 and the **crossover
with `q(a)` at ~39 months** — the sign→speech hand-off (down from the baseline
fit's spurious 12-month-dominant signing).

Total expressive `p_any` (independence upper bound) expected word counts: 55 (24
mo), 133 (36), 210 (48), 239 (60). The p_any validation against `uk_02`:
independence over-states the union by **3.7 pp** within `uk_02` (0.608 vs observed
0.571 over 20–56 mo); VG14's own `p_any / p_U` lands ~1.2 pp above observed
(`p_any_validation_gap.csv`).

Key figures: `signed_rate.svg` (hump, extrapolation shaded < 18 / > 54 mo),
`sign_speech_crossover.svg`, `p_any_validation.svg`.
