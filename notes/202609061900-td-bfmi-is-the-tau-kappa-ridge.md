# The typically developing models' low BFMI is the child-scale / dispersion ridge, and the partition is not what causes it

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

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

## Why this ties three issues together

The parameter with the strongest energy correlation is `tau_subject`, and that is exactly the parameter [#225](https://github.com/dseinternational/vocabulary-growth/issues/225) finds biased in recovery. So the recovery bias, the low BFMI and #229's design question are **one mechanism seen three ways**: a child effect and an observation-level dispersion competing for the same variance on the same probability scale, identified by functional form rather than by replication.

That has a sequencing consequence which is the practical point of this note. #225's evidence is interval coverage, measured on VG12 recovery fits whose own BFMI caveat reads _"the interval bounds are less reliable than the point estimates"_. An interval-coverage failure cannot be diagnosed with an instrument that reports unreliable intervals. **4.6 comes first**, and it is unblocked: all seven typically developing fits were verified on 2026-09-06 as untouched by the `us_03` ingestion — every one reproduces its recorded prepared-frame hash exactly — so none of this waits on the Down syndrome refit.

## What it does not say

It does not say what the fix is. It localises the problem to the τ–κ block and rules out one candidate cause; choosing between the structural options — which is [#229](https://github.com/dseinternational/vocabulary-growth/issues/229)'s job — still needs the design work recorded there. It is also worth noting what a _reparameterisation_ implies for scheduling: it changes the graph, so it invalidates every existing typically developing fit. Those fits are already unpublishable for an unrelated reason (none records an executable-code signature, which arrived on 2026-09-05), so the right order is to fix 4.6 and then refit the typically developing set once, rather than refitting it now and again afterwards.
