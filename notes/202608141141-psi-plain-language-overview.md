# Signing and speech: what the sign–speech association tells us, in plain language

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> A general-reader account of the sign–speech association $\psi$ and what the four studies that record signing word-by-word show, written for readers who do not work with the statistics. The technical investigation is [202608121030](202608121030-psi-heterogeneity-and-age-invariance.md); every figure here comes from it or from [`scripts/psi_heterogeneity_audit.py`](../scripts/psi_heterogeneity_audit.py). Descriptive counts are current as of 2026-08-14. The **fitted** $\psi$ quoted below predates the change that gives each study its own value, so it is a pooled figure that the next reporting refit will replace — see [Cautions](#cautions-for-interpreting-the-numbers).

## The question

Every word a child understands falls into one of four boxes: they sign it, they say it, they do both, or they do neither. Our models already estimate two summary figures — what share of the words a child understands they sign, and what share they say. Those two totals do not tell us how the words overlap.

Picture a child who understands 100 words, signs 40 of them and says 40 of them. That is consistent with two very different children. One signs and says the _same_ 40 words. The other signs 40 words and says a _different_ 40 — 80 words expressed in total, twice as many. Nothing in the two totals distinguishes them.

$\psi$ is the single number that settles which picture is closer to the truth. It is set up so that **1 means no relationship**: knowing that a child signs a particular word would tell you nothing about whether they also say it. Above 1 means signed and spoken words tend to be the same words. Below 1 means children sign the words they cannot say.

## Which studies can answer this, and which cannot

Eight of our studies record signing. Only four record it word by word in a way that reveals the overlap: **uk_02** and **uk_07** in the UK, **nz_01** in New Zealand, and **es_01** in Spain, contributing 434 observations between them. The other four give totals only — how many words signed, how many spoken — which cannot separate the two children in the example above.

The association is therefore measured in half the signing studies and applied to all of them.

## What the four studies show

The most direct way to put it: _of the words a child signs, what share do they also say?_

| study               | share of signed words also spoken |
| ------------------- | --------------------------------- |
| uk_07 (UK)          | 72%                               |
| uk_02 (UK)          | 50%                               |
| nz_01 (New Zealand) | 45%                               |
| es_01 (Spain)       | 31%                               |

Another angle on the same data — the share of children in each study who show the _reverse_ pattern, signing mainly words they cannot say:

| study | children showing the reverse pattern |
| ----- | ------------------------------------ |
| uk_02 | 4%                                   |
| nz_01 | 4%                                   |
| uk_07 | 11%                                  |
| es_01 | **45%**                              |

In the three English-language studies, signing and speech land largely on the same words and only a small minority of children run the other way. In the Spanish study it is close to a coin flip.

Stated as $\psi$ itself the gap looks enormous — roughly 0.9 in Spain against 6 to 15 elsewhere, an order of magnitude. That framing is partly an artefact of how the statistic is built. On the simpler "share also spoken" measure the four studies run from 31% to 72%: a real difference, but a gradient rather than two separate worlds. The order-of-magnitude phrasing should not be over-read.

## What has been ruled out

**It is not age.** The obvious explanation is that uk_07's children are older — median 60 months against 32 and 38 elsewhere — and that the association simply grows with age. This was tested and rejected. Once you know which study a child is in, their age tells you nothing further about the association; and in the age band where three studies overlap, es_01 and uk_02 children matched at exactly the same age still differ six-fold.

**It is not uk_07's intervention.** uk_07 is a randomised trial, so the natural suspicion is that its programme created the pattern. Its _control_ group in fact shows the stronger association, and the difference was present before the intervention began.

**It is not the instrument or the language.** This is the cleanest evidence available. The Spanish study includes a comparison group of typically developing children, matched on developmental level and assessed on exactly the same form. Those children show a positive association where the children with Down syndrome show none. Same country, same questionnaire, same question — so what differs is something about the children or their circumstances, not about how the question was asked.

## What has not been explained

We do not know why the studies differ. Signing instruction is the obvious candidate: where children are taught to sign and say words together, both channels would naturally land on the same words. But **no study in our data records whether its children were taught to sign**. That explanation is a reasonable guess about practice rather than something measured, and the one study with experimental variation in signing instruction points the other way. It belongs in any write-up as an open question, not as a finding.

## Cautions for interpreting the numbers

**One number covers all ages.** $\psi$ carries no age term, deliberately, because the data cannot support one. A reported value therefore applies equally to an 18-month-old and a 7-year-old. That is a limit on what can be estimated, not evidence that the relationship is stable across development.

**One number covers all children in a study.** $\psi$ is a study-level average, and individual children vary a great deal — in Spain 45% ran counter to their own study's average. It should never be read as a description of a particular child.

**The headline moves when the pool changes.** Because the studies genuinely disagree, a single pooled value is a compromise that shifts with which studies are included: it moved from 1.80 to 2.49 when uk_07 alone was added. The per-study values are the meaningful read, and a pooled figure quoted without the spread misleads.

**It cannot separate "modality" from "easy words".** Some words are simply easier to express in any form — common, concrete, well practised. That alone would make signed and spoken words overlap, with no relationship between the two channels as such. Our data are word counts, which carry no information about _which_ words, so the two explanations cannot be told apart.

**It affects the total expressive vocabulary figure.** Where signing and speech land on the same words, a child's combined vocabulary is _smaller_ than adding the two channels and assuming no overlap would suggest. The earlier signing model (VG14) made that no-overlap assumption and so reports an upper bound. The difference is modest — at most around 9 words on the 810-word scale — but the direction is worth knowing.

**None of this is about cause.** Nothing here says that signing helps or hinders speech. It describes which words appear in which channel, at one point in time, as reported by a parent.

**The current published figures are out of date.** The fitted output predates the change giving each study its own value, so the per-study numbers that ought to be the primary read do not yet exist in the model output. Signing evidence also thins at older ages, so figures beyond about six years rest on very few observations. $\psi$ is flagged as descriptive and exploratory in the VG15 model report and should stay that way.
