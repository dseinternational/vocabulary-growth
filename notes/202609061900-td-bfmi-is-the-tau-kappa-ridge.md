# The typically developing models' low BFMI is the child effect, and the partition is not what causes it

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **Revised the same day, after the Down syndrome refit supplied a control.** The first version of this note called the pathology "the child-scale / dispersion ridge" throughout. The Down syndrome family shows that is two claims, only one of which generalises: adding a child effect costs energy exploration in **both** populations, but the τ–κ _correlation_ is specific to the typically developing models. See [The Down syndrome family is the control](#the-down-syndrome-family-is-the-control) — it is the section that makes the argument, and it was not available when the rest was written.

Date: 2026-09-06. Measured on the existing `rep` fits of VG11, VG12, VG13, VG21 and VG23 — read-only, nothing refitted. Scopes [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.6, and bears on [#225](https://github.com/dseinternational/vocabulary-growth/issues/225) and [#229](https://github.com/dseinternational/vocabulary-growth/issues/229) because it is the same parameter in all three.

## What was measured

`#289` task 4.6 records four typically developing fits below the 0.3 energy-BFMI threshold and asks for the responsible blocks to be reparameterised. It does not say which blocks. The energy is stored in every trace, so the question is answerable from what is already on disk: take each fit's marginal energy and correlate it with every scalar parameter. In a hierarchical model the parameter that tracks the energy is the one the sampler is struggling to move through.

The answer is the same in all five, and it is not subtle. In VG12 the top three are `tau_subject` (−0.815), `kappa_young` (−0.789) and `subject_variance_share` (−0.718); in VG13, `tau_subj_u` (−0.569) ahead of the whole `kappa` block. **The child-effect scale and the Beta-Binomial concentration are what the energy is made of.**

Then the ridge itself, directly:

| model | partition? | min BFMI | corr(τ, κ_young) | corr(τ, energy) |   τ CV | free params | energy SD ÷ √(d/2) |
| ----- | ---------- | -------: | ---------------: | --------------: | -----: | ----------: | -----------------: |
| VG12  | yes        |    0.208 |       **+0.757** |          −0.815 | 0.0216 |       5,850 |               3.08 |
| VG23  | no         |    0.243 |           +0.566 |          −0.575 | 0.0183 |      11,055 |               2.84 |
| VG13  | no         |    0.248 |           +0.576 |          −0.569 | 0.0181 |      11,054 |               2.81 |
| VG21  | no         |    0.263 |           +0.692 |          −0.593 | 0.0192 |      11,476 |               2.68 |
| VG11  | yes        |    0.363 |       **+0.290** |          −0.631 | 0.0086 |      14,588 |               2.32 |

The last column is the marginal energy SD against the ≈√(d/2) a well-behaved posterior of that dimension would give (54.1 for VG12, 74.3 for VG13, 85.4 for VG11). Every fit is 2.3–3.1× it, so the energy distribution is heavy-tailed — which is what a BFMI below 0.3 is reporting. It orders _exactly_ with BFMI, but that is not independent corroboration: both are statistics of the same energy series, so the column restates the failure rather than explaining it. The explanatory column is `corr(τ, κ_young)`.

## What follows, and what does not

**The partition is not the cause.** VG13, VG21 and VG23 carry **no** `subject_variance_partition` — they have two free scales — and all three sit below the threshold with a τ–κ correlation of 0.57–0.69. Meanwhile VG11 _has_ the partition, has the weakest ridge (+0.29) and is the only one of the five that clears. This is the direct measurement of the claim [#229](https://github.com/dseinternational/vocabulary-growth/issues/229) already argues on structural grounds — _"the partition is not what creates the ridge — VG13 has the ridge with two free scales"_ — and it settles it.

**Do not over-read the ordering.** Across five fits the rank correlation between the ridge and the BFMI is about +0.6, not 1: VG21 has the second-strongest ridge and the best BFMI of the four failures, and the middle three (0.243–0.263) are too close to separate. The two extremes are clean and the structural conclusion does not depend on the ordering, but "the stronger the ridge the worse the BFMI" is a tendency here, not a law, and n = 5.

**Tightness is not the problem.** VG11's `tau_subject` has by far the smallest coefficient of variation (0.0086 against 0.018–0.022) and the best BFMI. So the "strikingly tight for a parameter identified by 17% of children" observation in `202608050900-td-hierarchical-geometry.md` §8 describes something real but is not itself the pathology.

## The Down syndrome family is the control

The refit running on 2026-09-06 supplied something the typically developing models cannot: a lineage in which the child effect is added **and nothing else is**, on a pool with real within-child replication. VG08 is VG07 plus a constant understood child effect.

| model                         | child effect |  min BFMI | corr(τ_subj_u, κ_young_u) | corr(τ_subj_u, energy) |
| ----------------------------- | ------------ | --------: | ------------------------: | ---------------------: |
| VG07 (study RE only)          | no           | **0.784** |                         — |                      — |
| VG14 (no child effect)        | no           | **0.816** |                         — |                      — |
| VG08 (+ constant child on U)  | yes          | **0.421** |                **−0.001** |                 −0.632 |
| VG10 (+ child on q, anchored) | yes          |     0.470 |                    +0.271 |                 −0.469 |
| VG16 (VG10 + cross-lag)       | yes          |     0.486 |                    +0.278 |                 −0.465 |

Two things follow, and they pull apart the single claim the first version of this note made.

**The energy cost of a child effect is general.** BFMI roughly halves at exactly the step that introduces one — 0.78 to 0.42, with nothing else changing — and the child scale is the strongest energy correlate in every model that has one, Down syndrome included (−0.63 in VG08). That is not a typically developing quirk.

**The τ–κ ridge is not general.** In VG08 the correlation between the child scale and the concentration is **−0.001**: no trade-off at all, while the typically developing models sit at +0.57 to +0.76. So "child scale versus dispersion" describes the TD pathology specifically, not the cost of having a child effect.

The obvious candidate for the difference is replication, and it is large. Measured on each model's own prepared frame:

| frame          |  rows | children | mean obs/child | children with ≥2 visits |
| -------------- | ----: | -------: | -------------: | ----------------------: |
| VG08/VG10 (DS) | 1,708 |      943 |           1.81 |               **46.6%** |
| VG12 (TD)      | 7,049 |    5,819 |           1.21 |                   17.2% |
| VG13 (TD)      | 6,356 |    5,496 |           1.16 |                   15.1% |

Nearly half of Down syndrome children are seen more than once, against a sixth of typically developing ones. Where the data can separate between-child from within-child variance directly, the child scale does not have to be prised apart from the dispersion by the shape of the likelihood — and the correlation that indicates it is being so prised apart is absent. That is [#229](https://github.com/dseinternational/vocabulary-growth/issues/229)'s thesis with a control group.

(The Down syndrome figures are lower than #229's table, which gives 1.95 and 50.7%. That table predates `us_03`, which adds 183 children who are mostly seen once.)

**What this does not license.** Two populations, one contrast, and the populations differ in far more than replication — different instruments, ages, pool sizes and study composition. Replication is the candidate explanation, not the established one; the test that would settle it is thinning the Down syndrome pool to TD-like replication and asking whether the ridge appears.

## Why this ties three issues together

The parameter with the strongest energy correlation is `tau_subject`, and that is exactly the parameter [#225](https://github.com/dseinternational/vocabulary-growth/issues/225) finds biased in recovery. So the recovery bias, the low BFMI and #229's design question are **one mechanism seen three ways**: a child effect and an observation-level dispersion competing for the same variance on the same probability scale, identified by functional form rather than by replication. The Down syndrome control above is what makes that last clause more than a phrase — where replication is available, the competition does not show up as a correlation.

That has a sequencing consequence which is the practical point of this note. #225's evidence is interval coverage, measured on VG12 recovery fits whose own BFMI caveat reads _"the interval bounds are less reliable than the point estimates"_. An interval-coverage failure cannot be diagnosed with an instrument that reports unreliable intervals. **4.6 comes first**, and it is unblocked: all seven typically developing fits were verified on 2026-09-06 as untouched by the `us_03` ingestion — every one reproduces its recorded prepared-frame hash exactly — so none of this waits on the Down syndrome refit.

## What it does not say

It does not say what the fix is. It localises the problem to the τ–κ block and rules out one candidate cause; choosing between the structural options — which is [#229](https://github.com/dseinternational/vocabulary-growth/issues/229)'s job — still needs the design work recorded there. It is also worth noting what a _reparameterisation_ implies for scheduling: it changes the graph, so it invalidates every existing typically developing fit. Those fits are already unpublishable for an unrelated reason (none records an executable-code signature, which arrived on 2026-09-05), so the right order is to fix 4.6 and then refit the typically developing set once, rather than refitting it now and again afterwards.
