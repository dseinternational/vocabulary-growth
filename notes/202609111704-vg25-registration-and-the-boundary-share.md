# VG25 registered: the sign → speech cross-lag, and the boundary share that changed a default

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Registration record for [#297](https://github.com/dseinternational/vocabulary-growth/issues/297). No model has been fitted. Every count below is measured on the 2026-09-11 frame (1,708 rows, 943 children) by building VG25's real graph; the reproduction script is inline in §3. No fit of record was touched.

## 1. What was registered

**VG25 = VG24 + one coefficient.** A child's signed share of comprehension at their previous administration wave, relative to their own persistent signing standing, shifts the logit of their current production ratio `q` through `beta_sign_lag ~ Normal(0, 0.5)`. VG24 is nested exactly at zero, checked on the observed log density rather than by inspection (`tests/test_joint_sign_cross_lag.py`).

It extends **VG24 and not VG15**, which is the whole interpretive argument rather than a hierarchy detail. `rho_sign_q` is the persistent between-child sign–speech association; with it in the model the lag has that part taken away and is left measuring the prospective, occasion-level quantity it is named for. Without it, [202608151140](202608151140-cross-lag-not-for-models-of-record.md) §3's finding applies — within-child deviations in this pool have no memory beyond the occasion, so a lag mostly absorbs persistent covariance through a noisy proxy.

The deferral this closes rested on one item, the wave-sequential simulator, and [202609111500](202609111500-vg16-recovery-and-the-blocker-that-was-not-there.md) built it. VG25 is the model it was built for: its lag reads `signed`, which the joint engine draws in the **same stage** as the `spoken` it shifts, so a single pass would draw each consumer against a source still holding its real study value. Nothing declares that. `single_pass_is_sound` derives it from the stage order and selects the loop — measured end to end on 2026-09-11 at about 70 s at `dev`, seven waves, sixteen coherence checks passed.

## 2. The five design decisions

Four were named on the issue. The fifth was forced by measurement and is the only place VG25 departs from VG16 on something other than an argument.

|                                  | Registered                                         | Why |
| -------------------------------- | -------------------------------------------------- | --- |
| Which likelihoods the lag enters | spoken marginal **and** the cross-tab compositions | §3  |
| Baseline                         | `within` (VG16 registers `population`)             | §4  |
| `nz_01` denominator              | no lag source                                      | §5  |
| Gap                              | unrestricted, with a registered 12-month arm       | §6  |
| Boundary shares                  | **continuity**, not VG16's clip                    | §7  |

Seven sensitivity arms are registered with the model, in the same change as the fields they need — a field with no variant using it is dead weight in a fingerprint. What each does to the coefficient's support, measured:

| arm                         | supporting observations | children | what moves                                                           |
| --------------------------- | ----------------------: | -------: | -------------------------------------------------------------------- |
| _(registered)_              |                     191 |      129 | —                                                                    |
| `sign-lag-clip`             |                     191 |      129 | the predictor's **values**, not its rows (§7)                        |
| `sign-lag-marginal-only`    |                     111 |       80 | drops `uk_07` entirely and 28 of `uk_02`'s 41                        |
| `sign-lag-population`       |                     191 |      129 | the **baseline** subtracted, not the rows                            |
| `sign-lag-uk07-marginal`    |                     191 |      129 | `uk_07`'s 52 rows move from the `cells` branch to the `marginal` one |
| `sign-lag-gap-12`           |                     168 |      112 | drops the 23 lags reaching back more than a year                     |
| `beta-sign-tight` / `-wide` |                     191 |      129 | the prior scale only                                                 |

The four that leave the support at 191 are not inert — they change the predictor, the baseline, the likelihood carrying it, or the prior. Stated because a reader checking an arm against its row count would otherwise conclude three of them did nothing.

## 3. Which likelihoods the lag enters, and what it costs either way

VG15 keeps its **child shifts** out of the four-cell and produced-cell Dirichlet-Multinomials because a free per-child offset is co-identified with `psi` on those thin rows, and letting it in moved `psi` from 1.78 to about 2.8. The obvious move is to treat the lag the same way. It was not, and the reason is that `beta_sign_lag` is **one scalar multiplying a covariate fixed by the data** — one added dimension, not one per child, with no per-child freedom to chase a composition.

Measured on the current frame, building the real graph:

| design                              | supporting observations | children | gap median (IQR) | by study                                         |
| ----------------------------------- | ----------------------: | -------: | ---------------- | ------------------------------------------------ |
| marginal **and** cells (registered) |                     191 |      129 | 6 (4–11)         | uk_07 52, ie_02 43, uk_02 41, uk_05 30, uk_04 25 |
| marginal only                       |                     111 |       80 | 6 (3–7)          | ie_02 43, uk_05 30, uk_04 25, uk_02 13           |

The registered arm splits 111 rows onto the conditional spoken branch and 80 into the four-cell composition. `uk_07` contributes **nothing at all** under the narrow arm, because its rows carry no spoken marginal.

The issue's own table gave 201/136 and 121/87 for these two designs; the frame has moved since (`us_03` ingestion), and the ordering and the argument are unchanged.

Two things the issue's table implied that turned out not to hold on this frame:

- **`es_01` contributes 185 rows carrying a signed share and not one lag**, because no `es_01` child has two waves. It is a large source of the predictor's _input_ and none of its evidence.
- **`nz_01` contributes neither**, whatever is decided about its denominator — see §5.

**The caveat, stated because it is not nothing.** Under the `within` baseline the predictor itself contains `subject_shift_sign` at the prior wave, so an estimated per-child quantity does reach the composition — through one scalar coefficient, on lagged rows only, rather than as a free offset per row. `sign-lag-population` is the arm in which no estimated per-child quantity reaches the cells at all, because that baseline subtracts the shift back out. Read beside `sign-lag-marginal-only` it separates "the lag moved `psi`" from "a child effect reached `psi` through the lag".

Reproduces the two rows above (run 2026-09-11):

```python
import numpy as np

from vocab_growth.models.common_joint_modality import build_joint_analysis_frame
from vocab_growth.models.cross_lag import prev_wave_sign_share_lag, sign_share_counts
from vocab_growth.models.definitions import VG25

frame, _ = build_joint_analysis_frame(VG25)
signed, understood = sign_share_counts(frame)
subject = frame["subject_code"].to_numpy(int)
age = frame["age"].to_numpy(float)

marginal = frame["spoken"].notna().to_numpy()
cells = frame["signed_spoken"].notna().to_numpy()

for label, consumer in (("marginal + cells", marginal | cells), ("marginal only", marginal)):
    prev, has_lag, _ = prev_wave_sign_share_lag(
        subject, age, signed, understood,
        zero_handling=VG25.sign_lag_zero_handling,
    )
    rows = (has_lag > 0) & consumer
    gaps = age[rows] - age[prev[rows]]
    print(
        f"{label:18s} obs={rows.sum():4d} children={len(np.unique(subject[rows])):4d} "
        f"gap median={np.median(gaps):.0f} "
        f"IQR=({np.percentile(gaps, 25):.0f}-{np.percentile(gaps, 75):.0f})"
    )
```

## 4. The baseline, and why it differs from VG16's

`within` subtracts the child's own signed-ratio intercept, so a child is compared with their own persistent signing standing. VG16 registers `population`.

The two models differ because one of them has a correlation and the other does not. VG16 carries no `rho_uq`, so its population baseline still had a between-child association to measure. VG25 inherits `rho_sign_q`, so the population baseline would be a second, noisier reading of a quantity already in the model. Registered as `sign-lag-population`, where it does double duty as the check in §3.

## 5. `nz_01` supplies no source, and it is not free

Its cross-tab partitions **produced** words, so the only share it measures is the signed share of production — a different variable, not a differently-denominated version of this one. Pooling the two under one coefficient would make `beta_sign_lag` mean two things at once.

**It costs real support, and the issue's "+28 children" line reproduces almost exactly.** 28 of `nz_01`'s 33 children have more than one wave, and admitting a produced denominator takes the coefficient from 191 supporting observations over 129 children to **269 over 157** — 78 added observations from 28 children, at a median gap of 6 months, and no existing row's source moved. (The issue projected 279 over 164 on the older frame.)

So this is not the free decision an earlier draft of this note claimed. It is declined on the argument alone, which is the only way the rule survives the next `nz_01` follow-up: a coefficient whose meaning depends on how much data a denominator happens to bring is not one anyone can interpret. What would be defensible is a **second** coefficient on the production-denominated share, which is a different model and not this one.

## 6. The gap

191 supporting rows reach back 2 to 15 months, median 6 (IQR 4–11). One coefficient is fitted across all of them. `sign-lag-gap-12` is registered with the model rather than after a reviewer asks, which is what [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) asked for. Dropping a lag does not drop the row.

## 7. The boundary share: the decision measurement forced

**This is the finding of the registration**, and it is the one thing here that was not anticipated on the issue.

The predictor is `logit(signed / understood)`. A signed share of exactly 0 is common in a way an understood count of exactly 0 is not:

|                     | VG16's count lag | VG25's ratio lag |
| ------------------- | ---------------: | ---------------: |
| supporting rows     |              477 |              191 |
| at a logit boundary |         7 (1.5%) |   **28 (14.7%)** |

26 rows have a source wave where the child signed none of what they understood; 2 have one where they signed all of it. The clip puts all 26 at `logit(1e-4) = -9.21` **whatever the wave measured**, so a child who understood 2 words and signed none enters identically to one who understood 406 and signed none. The source denominators of those 26 rows run 2, 7, 7, 13, 15, … 129, 406 — two orders of magnitude, one predictor value.

Measured on the **source signed-share logit** — the predictor's observed input, before the model's latent baseline is subtracted, which is the part measurable without a fit:

| treatment  |   SD | range          | boundary rows' share of its sum of squares |
| ---------- | ---: | -------------- | -----------------------------------------: |
| clip       | 3.34 | [−9.21, +9.21] |                                  **76.1%** |
| continuity | 1.79 | [−6.70, +4.89] |                                      45.8% |

Under the clip, 14.7% of the rows carry three quarters of the variation the coefficient is estimated from — `beta_sign_lag` would be fixed mostly by a floor constant. The continuity correction `(k + 0.5) / (n + 1)` derives each boundary row from its own wave's denominator — −1.61 for the 2-word wave, −6.70 for the 406-word one — which is the ordering the data actually support: a larger denominator with no signs is stronger evidence of not signing.

**And it is not an artefact of the raw scale.** The fitted predictor subtracts a latent baseline, so the table above could in principle overstate the boundary's reach. Residualising the source logit on the source wave's age and study — a stand-in for that baseline, and the closest check available before a fit — leaves the same picture: SD 2.77 against 1.56, boundary leverage **66.1% against 42.3%**.

So **VG25 registers continuity and VG16 keeps the clip**. The _field_ default stays `LAG_ZERO_CLIP`, so the two lags are configured the same way and the off-state remains the historical treatment; what changed is what VG25 registers. `sign-lag-clip` measures the difference on a fit rather than on this arithmetic.

This is worth generalising: the two lags are not the same defect surface because one reads a count against a fixed 810-item inventory and the other reads a ratio of two counts from one administration. The ratio's boundaries are reachable at both ends and are reached often; the count's lower boundary is reached rarely and its upper one never. A treatment chosen for one is not automatically right for the other, which is exactly the assumption a port makes by default.

## 8. What does not carry over from VG16

**The same-form restriction, and it is a property of the predictor rather than of the pool.** `lag_same_form_only` exists for the count lag because `understood / 810` is deflated by a shorter source form and a study intercept cannot absorb a within-study form transition — 131 of VG16's 473 supporting rows (28%) cross a ceiling. The ratio lag divides one count by another **scored on the same form in the same administration**, so the truncation is very largely common to numerator and denominator and cancels. VG25 therefore carries no same-form field at all, rather than carrying one that would always be inert.

**The source-selection rule does carry over, restated.** VG16 takes the largest understood count at a source wave, on the ground that it is the least-truncated measurement available. For a ratio the same rule is "the largest comprehension denominator", which is what the shared walk applies. On the current frame no source wave offers more than one signed-share measurement, so the rule is registered ahead of need — as VG16's was.

## 9. What is still open

1. **No fit.** VG25 is `UNCLASSIFIED` in the catalogue, which fails closed: full publication strictness, and a place in the default refit scope (now 8 of 22). Whether it becomes a model of record, a development step, or neither is a study-owner decision taken with [#190](https://github.com/dseinternational/vocabulary-growth/issues/190)'s other scope questions (#297 check 7).
2. **Gate 4's three recovery cells.** VG25 is a registered recovery target and the wave loop is selected automatically, so the `(beta ≠ 0, rho ≠ 0)` cell is runnable today from the prior. The other two need a way to **set** a parameter in the truth draw rather than take what the posterior or prior offers, which the harness does not have. Not large; not built.
3. **No leave-one-study-out arm** (#297 check 5). `exclude_studies` is a `BivariateModelDefinition` field; adding it to `JointModelDefinition` would restale every VG15 and VG24 fit, which is a change to make deliberately with the refit it costs. `sign-lag-uk07-marginal` is the nearest available arm and answers a different question — it keeps `uk_07`'s children and moves their rows from the cross-tab branch to the marginal one.
4. **The LOO leakage** [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) raised against VG16 applies here unchanged: leaving an observation out does not remove it from the later rows whose predictor it feeds. `scripts/wave_forward_score.py` is the answer and has not been run for this model.
5. **Expect a small coefficient with an interval near zero.** The descriptive note's 89% interval only just cleared zero at n = 147; VG16's own recovery returned `beta_lag` 29% low on its one assessable replicate at `test`; and this lag's support is 191 rows against VG16's 473. Two of three VG16 replicates missed the R-hat gate at `test`, so gate 4 needs a higher tier here too.

## 10. Three things review found, and what measurement said about each

Automated review of [#339](https://github.com/dseinternational/vocabulary-growth/pull/339) raised nine points. Six were stale counts and headings. Three were substantive, and all three were real:

**The source-selection rule was not order-independent, though its docstring said it was.** The wave walk ranked candidate source measurements on the comprehension denominator alone and took `np.argmax`, which resolves a tie by whichever row the frame happens to list first. That is sound only while the ranked quantity is the _only_ thing read off the selected row — true of VG16's count lag as originally written, and false of this one, which reads the signed numerator as well. Demonstrated rather than argued: on a wave with two forms tied at 100 understood and disagreeing at 20 and 60 signed, merely swapping the two rows moved the predictor from −1.386 to +0.406. The same hole had been open in VG16 since [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) added `same_form_only`, which reads the source's _form ceiling_ off the chosen row: there the swap flipped whether the row had a lag at all. The rule is now stated as _every quantity read off the selected row is a selection key_, and both lags pass their keys. Nothing on the current frame exercises it — no wave that serves as a source offers a choice, for either lag — and every array is bit-identical across VG16's registered form, its same-form and gap arms, and all seven VG25 arms. Ten waves do carry two usable comprehension measurements, so one further administration for any of those children would have made this live.

**The report's support tables ignored `sign_lag_in_cells`.** They filtered the audit artefact on "has a likelihood branch", but a cross-tab row has a branch whether or not the lag reaches it. Rendering a `sign-lag-marginal-only` fit would therefore have reported the registered model's 191 observations over 129 children in place of its own **111 over 80** — and every gap, boundary, study and branch figure derived from that set with it. The template now reads the field from the fit's own manifest, as it already did for the boundary treatment. The caption had said the compositions count "only under `sign_lag_in_cells`" while the code counted them regardless, which is the tell.

**The pair plot could not show the ridge the report sends the reader to inspect.** This is [#233](https://github.com/dseinternational/vocabulary-growth/issues/233) again, on the other engine. ArviZ caps the grid at `floor(sqrt(max_subplots))` — six variables — and the joint engine led with `psi` and `conc` and then took build order, which puts `beta_sign_lag` eighteenth: the plot would have shown the two headline associations and four understood-GP hyperparameters, while @sec-diagnostics tells the reader to read the coefficient against the signing child block. Ordering, not filtering, so nothing is hidden. Only the lag's own block is prepended: routing the joint engine through the bivariate `pair_plot_priority` would also reorder VG14's, VG15's and VG24's plots, and a reporting change for three fitted models is not something to make inside a registration.

## 11. A note on scope of change

Registering VG25 moves the **executable-code signature**, as any change inside `src/vocab_growth/` does, so every existing fit needs the refit that [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) already schedules before it can be published or resumed. It does **not** move any other model's serialised definition or prepared-frame hash: the seven new fields live on `JointCrossLagModelDefinition`, a subclass no other model instantiates, and `tests/support/graph_baseline.json` gained 315 lines and changed none — which is the evidence that the registration was graph-inert for the other twenty-one.
