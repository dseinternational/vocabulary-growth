# Vocabulary growth in children with Down syndrome — plain-language findings

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> A general-reader summary of the main results from the 2026-07-12 reporting run. The
> headline models (VG10, VG12, VG15, VG16) are reporting-quality; the sign-group
> contrasts (VG17/VG18) are **exploratory** (`test` tier). Full detail is in the
> consolidated report (`docs/report`). Numbers are best estimates with credible
> intervals; see caveats at the end.

## How to read the numbers

- We use **Bayesian** models: each quantity comes with a full probability distribution.
  We quote a **best estimate** (the median) plus a **credible interval** — the range the
  true value lies in with a stated probability. Our house convention is a **50%** (inner)
  and an **89%** (outer) interval; 89% is a deliberately non-special width, not a pass/fail
  threshold. If an interval includes the "no-difference" value, we cannot claim a
  difference. (The tables below are from the 2026-07-12 run and quote the **90%** intervals
  in force then; the next reporting run will report the 50% and 89% intervals.)
- Vocabulary is counted out of a common **810-word** reference list, so every study and
  both groups sit on the same scale.
- "Understood" = words the child comprehends; "spoken" = words they say; "total
  expressive" = words they can say **or** sign (each counted once).

## 1. How vocabulary grows, and the comprehension–production gap (VG10, DS)

Children with Down syndrome understand far more than they say, especially early — and
the gap narrows with age.

| Age   | Understood (words of 810) | Spoken (words of 810) |
| ----- | ------------------------- | --------------------- |
| 24 mo | **101** (90% 82–121)      | **5** (90% 3–7)       |
| 48 mo | **281** (90% 240–323)     | **147** (90% 115–179) |
| 72 mo | **444** (90% 377–513)     | **371** (90% 318–424) |

**Plain reading:** at 2 years a typical child with Down syndrome understands ~100 words
but says only a handful; by 6 years they understand ~440 and say ~370. Comprehension
leads production throughout, but production catches up.

## 2. Production "catches up": the say-given-understand ratio (VG10)

The production ratio **q = P(says a word | understands it)** rises steeply with age:

| Age   | q (%)               |
| ----- | ------------------- |
| 24 mo | **5%** (90% 3–7)    |
| 48 mo | **53%** (90% 43–62) |
| 72 mo | **84%** (90% 73–95) |

**Plain reading:** at 2 years a child says about 1 in 20 of the words they understand;
by 6 years, more than 4 in 5. The early bottleneck is expressive, not receptive.

## 3. Compared with typically developing (TD) children (VG10 vs VG12)

At 24 months, on the same 810-word scale:

- **TD understood: ~357 words** (90% 303–411)
- **DS understood: ~101 words** (90% 82–121)

**Plain reading:** at 2 years, children with Down syndrome understand roughly a **third**
of what typically developing children do — a substantial receptive delay. (TD data here
cover only the young age range, so this contrast is anchored at 24 months.)

## 4. Signing — signs help the child who uses them, but signers are not ahead overall

Two questions, two answers, and they fit together:

**(a) For a child who signs, do signs add expressive words? Yes (VG15).** Total
expressive vocabulary sits clearly above spoken-only:

| Age   | Spoken | Total expressive (say-or-sign) | Signs add           |
| ----- | ------ | ------------------------------ | ------------------- |
| 36 mo | 41     | **107** (90% 78–135)           | **+65** (90% 41–89) |
| 48 mo | 143    | **188** (90% 151–223)          | **+44** (90% 24–63) |

The estimated sign–speech association **psi = 1.8** (90% 1.2–2.3) — a
positive link between signing and saying a given word. **Plain reading:** signs give a
signing child access to ~40–65 extra expressive words in the pre-school years.

**(b) Do signing children end up with MORE vocabulary than non-signers? No clear
difference (VG17/VG18, exploratory).** Comparing groups at the same age (odds ratio; 1 =
no difference):

| Comparison (clean, de-duplicated) | spoken (VG17)           | total expressive (VG18) |
| --------------------------------- | ----------------------- | ----------------------- |
| signer vs non-signer              | OR 1.11 (90% 0.89–1.37) | OR 0.93 (90% 0.74–1.17) |

Both include 1. (A first pass on all studies suggested a big total-expressive advantage,
OR 1.65, but that was a **measurement artefact** — "total expressive" was defined
inconsistently across studies; the clean analysis removes it.)

**Putting (a) and (b) together:** signs let signing children express words they cannot
yet say (compensation), but they do not end up ahead of non-signing children, who reach
similar totals through speech. Signing supports the children who rely on it rather than
conferring a surplus.

## 5. Does earlier understanding predict later talking? (VG16 — interpret cautiously)

A within-child cross-lag from earlier words understood to the later production ratio was
**small and positive**: `beta_lag` = **+0.20** on the logit scale (90% 0.06–0.33). **Plain reading:** understanding earlier is
weakly associated with talking relatively more later. **Big caveat:** with only two
measurement waves the design cannot cleanly separate a genuine within-child effect from
stable between-child differences, so treat this as suggestive, not causal (see the
report).

## Caveats (please read)

- Findings 1–3 and 4(a), 5 are from reporting-quality DS/TD models; **4(b) (VG17/VG18)
  is exploratory** (`test` tier).
- The sign-group comparisons in 4(b) are partly confounded with which studies collected
  sign data (country/instrument); the "sign-unknown" group's expressive total is
  under-measured.
- One study (`uk_01`) records signed-only words; handled for totals but relevant to
  cross-study "amount of signing" comparisons (a follow-up).
- All counts use the common **810-word** scale, separately **validated** as an
  appropriate denominator (report Methods).
- Convergence: 14 of 15 models pass all diagnostic gates; VG11 has a single divergence
  (otherwise excellent — R-hat 1.0004, ESS ~9,850), reviewed and **accepted** as the
  model of record.

## Bottom line

Children with Down syndrome understand much more than they say, with a large early
comprehension–production gap that narrows as the say-given-understand ratio climbs from
~5% at age 2 to ~84% at age 6; they show a substantial receptive delay versus typically
developing peers (~⅓ of the words understood at age 2). Manual signs give signing
children extra expressive words in the pre-school years, but signing children are not
ahead of non-signers overall — signs compensate rather than confer a surplus.

_Sources: VG10, VG12, VG15, VG16 (reporting-quality); VG17/VG18 (exploratory,
`model_vg17.py`/`model_vg18.py`). Intervals in these tables are the 2026-07-12 run's 90%
highest-density (HDI) intervals; the current reporting convention is a 50% and an 89%
equal-tailed interval (HDI reserved for skewed estimands) — see
`docs/report/methods-workflow.qmd`. Methods + convergence: `202607121753-...md`._
