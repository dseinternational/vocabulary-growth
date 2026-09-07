# The `us_03` refit halved the young-age concentration, and widened the interval the project publishes

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-06. Measured on the refitted VG10, VG16 and VG20 against the pre-ingestion figure cache, while VG19, VG22 and VG24 were still sampling. Companion to [`202609062030-us03-refit-curve-shift.md`](202609062030-us03-refit-curve-shift.md), which covers the mean trajectory; this note is about everything else the refit moved, and the practical consequence is larger than the curve shift.

## The concentration at young ages roughly halved

`kappa` is the Beta concentration — the engine builds `alpha = p·kappa`, `beta = (1−p)·kappa` — so a **lower** `kappa` is **more** over-dispersion.

| model | `kappa_young_u` published → refitted | shift in published sd |       `kappa_old_u` |
| ----- | -----------------------------------: | --------------------: | ------------------: |
| VG10  |                      77.8 → **41.5** |             **−3.62** | 22.5 → 24.1 (+0.37) |
| VG16  |                      78.2 → **42.0** |             **−3.60** | 22.5 → 23.9 (+0.33) |
| VG20  |                      71.4 → **38.3** |             **−3.50** | 24.4 → 25.9 (+0.34) |

The old-age concentration barely moved. This is a change at the young end only, and it is by far the largest parameter movement in the refit — three and a half standard deviations, against about one and a half for the child scale and well under one for everything on the production side.

The legacy models show the same thing through their own parameterisation: VG08's `b_kappa_mag_u` fell 0.485 → 0.202, its 89% interval from [0.284, 0.706] to [0.031, 0.435].

## What that does to the published interval

The 89% predictive interval for **a new child's** comprehension, VG20, on the 810-item reference inventory:

| age | published |  refitted |      width |
| --- | --------: | --------: | ---------: |
| 18  |  [6, 166] |  [3, 192] | **+18.1%** |
| 24  | [23, 309] | [16, 321] |      +6.6% |
| 30  | [45, 415] | [32, 418] |      +4.3% |
| 36  | [61, 476] | [48, 482] |      +4.6% |
| 48  | [99, 581] | [86, 587] |      +3.9% |

Two things to read off it. The interval **widened at every age and most at the youngest**, and its **lower bound fell at every age** — 6 → 3, 23 → 16, 45 → 32, 61 → 48, 99 → 86. The model now admits materially lower comprehension counts than it did.

This matters more than the curve shift for what the project is for. These are the numbers that inform expectations; a family or practitioner reading the 18-month row now sees a wider band with a lower floor, and that change is driven by a cohort of 186 children whose data sit exactly there.

## The rest of the picture

| quantity                                 | direction            | size                                                                  |
| ---------------------------------------- | -------------------- | --------------------------------------------------------------------- |
| `tau_subj_u` (between-child, understood) | up                   | +1.4 to +1.6 sd in VG08, VG10, VG16, VG20 (VG20 0.781 → 0.822, +5.2%) |
| `intercept_u`, `slope_u`                 | down                 | −2.4 to −2.5 sd and −1.6 sd — the curve flattening                    |
| `rho_uq` (VG20's headline)               | up, but within noise | 0.390 → 0.433, +0.93 sd                                               |
| `tau_q`, `tau_subj_q`, `kappa_old_u`     | unchanged            | all under +0.4 sd                                                     |

**VG20's headline result survives.** The understood/production-ratio correlation moved less than one standard deviation, so the finding the model exists to report is not a casualty of the ingestion.

**The production side is untouched throughout**, which is the same fact the curve note records from the other direction: `us_03` supplies no production data at all, so `q` and everything scaled to it stay where they were.

## One thing to be careful about

`tau_subj_u` rising ~5% is on the Down syndrome side of the between-child contrast with the typically developing models — and [#225](https://github.com/dseinternational/vocabulary-growth/issues/225) finds the _typically developing_ child scale recovers **low**. Those push the same way: the reported DS-versus-TD difference in between-child spread is now wider through a real data change on one side and a suspected bias on the other. That is a reason to state the contrast with both caveats attached, not a reason to treat it as strengthened. The TD side cannot be resolved until #289 task 4.6 and the TD refit.

## Scope

Three anchored models, all agreeing closely. VG19, VG22 and VG24 were still sampling and should be added; VG05, VG07 and VG08 carry the legacy `kappa` form and are development steps, so their numbers are recorded above for corroboration rather than as reportable quantities — see [#236](https://github.com/dseinternational/vocabulary-growth/issues/236) item 11 on why their concentration trends must not be quoted at all.
