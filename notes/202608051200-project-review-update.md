# Project review update: what has changed since 12 May 2026

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Written 2026-08-05, mid-way through a reporting-quality refit of every model. Numbers describing model output are from that run and one model (VG11) was still fitting as this was written. Successor to [202605120945](202605120945-meeting-project-review.md), which should be read as superseded on the headline-model question — see §3.

## The short version

Since the 12 May review there have been **213 commits**. Almost none of them added new statistical ideas. Nearly all of them did one of four things: **got more data in**, **checked that the data were what we thought they were**, **checked that the models were saying what we thought they were saying**, or **made the whole thing reproducible by someone else**.

That is the right shape of work for a study that intends to publish numbers other people will act on, but it means progress does not look like new findings. The headline developmental story has not changed much since May. What has changed is how much of it we can defend.

**The single biggest remaining obstacle to publication is not statistical — it is that the report itself is largely unwritten.** The modelling is close to finished. The prose is not.

## 1. What the project is, briefly

We are trying to describe how children with Down syndrome learn words — how many words they understand, how many they say, how many they sign, and how those relate to each other as children get older. We pool data from several studies in different countries, and we compare against typically developing children as a reference.

We use Bayesian models, which means the output is not a single number but a range of plausible values with probabilities attached. That matters here because the practical question — "what should I expect for this child?" — is a question about a range, not a point.

## 2. What changed

### 2.1 More data, and then hard scrutiny of it

Three new datasets were brought in: **Ireland (ie_02)**, **New Zealand (nz_01)** and **Spain (es_01)**. The Down syndrome pool now draws on **13 studies**; the model of record uses **1,349 observations from 737 children**.

More consequential than the additions was what auditing them turned up. The US dataset (`us_01`) had to be **rebuilt from the original item-level files** because the public download page silently truncates every administration to the age window its questionnaire was designed for — that alone had cut 345 administrations to 194, and it could not distinguish four all-blank forms from genuine zeros.

Several other classes of bad record were found, documented and excluded, each with a switch to put them back for sensitivity checks: partial administrations, duplicated outcome columns, implausible production patterns, counts above what a form can physically score, and children recorded only at their questionnaire's ceiling. One anomalous child was removed entirely.

The typically developing comparison pool was widened from English-only to **English, Italian and Spanish**, because the Down syndrome side was already a quarter non-English and the comparison was not like-for-like.

**Why this matters:** almost every one of these fixes changed the numbers. A study that had not looked would have published the unaudited ones.

### 2.2 The model family was completed, then independently reviewed

Six models were added — the typically developing hierarchical models, a signing model, a joint sign-and-speech model, and a model that asks whether understanding earlier predicts speaking later *within the same child*. There are now **15 models**, and a canonical inventory that has to be updated whenever one changes.

They were then **independently statistically reviewed**, and the review's recommendations implemented. Duplicated model-building code across engines was consolidated, which sounds like housekeeping but matters: the same bug had been sitting in several copies.

### 2.3 Priors were put on an evidence footing

A "prior" is what the model assumes before seeing the data. Get it wrong and the model reports your assumption back to you.

A systematic audit found several places where this was happening. Priors for the youngest ages were recalibrated across the family. A case of **double-dipping** — using the data to set a prior and then fitting to the same data — was found and removed. The dispersion parameters were re-expressed in terms of two real ages, so that the assumptions could be checked against what the data actually show at those ages, instead of at a mathematical convenience point that moves whenever the sample changes.

Most recently the trajectory curves were found to **extrapolate implausibly past the last data** — asserting with 90% probability that a child speaks more than 99% of the words they understand at an age where only five children contribute both measurements. The mean now levels off where the evidence stops.

### 2.4 Making the computation trustworthy

These models are fitted by a randomised algorithm that has to be checked for whether it explored the space properly. A **two-tier gate** now enforces this: the serious failures stop a fit outright and cannot be overridden; the softer warnings are recorded and must travel with the results.

Two genuine geometry problems were diagnosed and fixed, and a **parameter-recovery harness** was built — the model generates fake data from known answers, then has to recover them. It is the most direct check available that the machinery works.

### 2.5 Reproducibility

The environment was migrated to current versions and pinned so the exact stack can be reconstructed. Every fit now records what code, what data and what settings produced it, and cannot be published if any of them has drifted. Output is promoted atomically, so an interrupted run cannot leave a half-finished model looking complete.

There is a convention requiring AI-assisted content to be labelled as such, and **512 automated tests**.

### 2.6 The report was started

A full report book now exists in draft with methods chapters, a descriptive data chapter, and a comparison report. Credible intervals were standardised. Comprehension results are now trimmed to 72 months, where the comprehension evidence actually stops.

## 3. Things we got wrong, and corrected

This section matters more than §2 for judging how much to trust the rest.

- **The 12 May recommendation is superseded.** That review recommended VG09 as the headline Down syndrome model. The current model of record is **VG10**, which adds a stabilisation that VG09 lacks.
- **An inverted headline claim** was found and corrected — a reported direction of effect was backwards.
- **A dispersion parameter was being read as between-child variability.** It is not; it is driven by the overall level. Three places in the report and the docstring that licensed them were corrected today. No corrected estimate of that quantity is currently available.
- **The Down-syndrome-versus-typical dispersion comparison was pointed at the wrong model** — one without the per-child effects its counterpart had, which inverted the contrast. Fixed today.
- **The report claimed a disclosure mechanism it did not have.** The appendix that was supposed to record convergence caveats was an empty stub with a broken cross-reference. It has been built.
- **A recommendation made yesterday was withdrawn today** after being checked against the model inventory, and a second was falsified by direct experiment (§4).

## 4. Where we are now

Thirteen of fifteen models are fitted at reporting quality and valid. One (VG11) is finishing as this is written; one (VG12) needs a refit to pick up a sampling improvement.

Two of the typically developing models carry a **soft convergence caveat** that we now understand and cannot remove. The cause was diagnosed this week: the models cannot cleanly separate "children genuinely differ from each other" from "measurements are noisy", because most typically developing children in the pool were measured only **once** — 1.21 observations per child. A test confirmed the mechanism by comparing against a Down syndrome model with 2.4× the repeat rate and much better behaviour.

We tried to fix it by re-expressing the two quantities as a total and a split. **It did not work**, and the reason is informative: the problem rotated rather than disappearing. This is what happens when the obstacle is *missing information* rather than an awkward parameterisation. **No amount of computation will fix it; only measuring more children more than once would.**

That result is worth having. It converts a nagging technical warning into a stated limitation with a known cause.

## 5. What remains before we could publish

**Modelling — days, mostly mechanical**

1. Finish the current refit (VG11, then VG12).
2. Regenerate the cross-model comparisons and the report figures.
3. Re-verify every number quoted in the report against the new fits. Several are currently pinned to superseded ones.
4. Re-run the parameter-recovery checks against the new fits.
5. Complete one outstanding prior-sensitivity check.

**Writing — the real bottleneck, weeks**

6. The following chapters are **stubs or TODO placeholders**: the summary, the introduction, the discussion, the signing and total-expressive results chapter, the shared caveats, the reference tables, and the acknowledgements. The methods chapters are in reasonable shape; the interpretation is not written.
7. The model specifications appendix is still outstanding.
8. A plain-language summary was promised in the report's own introduction and does not exist.

**Decisions that are not ours to make**

9. Confirm VG10 as the headline Down syndrome model for publication.
10. Decide how the convergence caveats are presented — they are disclosed correctly now, but how prominently they appear in a public summary is an editorial judgement.
11. Decide what happens to the superseded July publication still live in public storage.

**Before external release**

12. An independent statistical read of the final numbers, not just the code.
13. A decision on whether the sample's representativeness — these are not a random sample of children with Down syndrome — is stated prominently enough for the intended audience.

## 6. Honest risks

- **The interpretation is the unwritten part, and it is the part that will be quoted.** Everything above is machinery. The claims a reader takes away have not yet been drafted, and drafting them will surface disagreements that the modelling has so far deferred.
- **We have corrected several substantive errors in the last two weeks alone** (§3). The rate is not obviously falling. It would be unwise to treat the current numbers as final before the verification pass in §5.3.
- **The typically developing comparison is the weaker half of every contrast we report.** Its intervals are less trustworthy than the Down syndrome side's, for a reason we can now state precisely. Any comparison of variability between the two groups should carry that caveat in the text, not only in an appendix.
