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

`rho/rel` disattenuates for measurement error using a **binomial lower bound** on the error variance. Because that bound ignores the extra-binomial dispersion the models fit (`kappa`), it understates error and so under-corrects: treat it as a floor on true stability, not an estimate.

## 3. Result: tracking is substantial across the full Down syndrome age range

| pool              | ICC (89% CI)             | 1–6 mo | 6–12 mo | 12–24 mo |               24–60 mo |
| ----------------- | ------------------------ | -----: | ------: | -------: | ---------------------: |
| **DS spoken**     | **0.806** [0.780, 0.832] |  0.704 |   0.656 |    0.643 | **0.238** [0.04, 0.44] |
| **DS understood** | **0.786** [0.718, 0.847] |  0.701 |   0.780 |    0.721 |                      — |

334 children contribute repeated spoken measures and 253 repeated comprehension measures.

In readable terms, over a median gap of ~12 months: **half stay in the same quartile, 86–87% stay within one quartile**, and movement from bottom to top is rare — 5 of 334 on spoken, 4 of 253 on comprehension.

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

The reliability bound explains it. For DS spoken at ≤30 months it is **0.588**, against 0.87–0.97 everywhere else — by far the lowest in the analysis. At those ages DS spoken counts are tiny (VG10 puts the expected count at 24 months at about 6 words), so a difference of two or three words moves a child several quartiles. The disattenuated 1–6 month correlation hits the 1.0 cap, which is the signature of noise rather than instability.

So the finding is **not** that young children with Down syndrome have unstable expressive ability. It is that **spoken-word counts are too small at those ages to measure standing reliably**. Comprehension in the same window tracks well (ICC 0.858, correlations 0.79/0.77), because the counts are large enough to carry information.

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

## 8. What would strengthen it

A within-child model with an age-varying random effect — a random slope, or a latent AR(1) — would estimate drift directly rather than inferring it from binned correlations, and would handle the unbalanced designs properly. Note that this analysis also bears on **A1** in [202607261540](202607261540-item-difficulty-and-the-aggregate-likelihood.md): the constant `tau_subj` the models assume is a good approximation out to about two years and demonstrably poor for young DS spoken counts, where the reliability floor dominates.
