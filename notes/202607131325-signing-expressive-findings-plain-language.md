# Signing and expressive vocabulary in Down syndrome — plain-language findings

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> These are **exploratory** results from two side-models (VG17, VG18) fitted at the
> `test` sampling tier, not confirmatory conclusions. Read the caveats at the end. The
> full study results live in the consolidated report (`docs/report`).

This note summarises, for a general science reader, what we found when we asked a
simple question: **do children with Down syndrome who use manual signs differ, in how
much they can express, from children who do not (or whose sign use we don't know)?**

## How to read the numbers

- We fit **Bayesian** models, which return a whole **probability distribution** for
  each quantity rather than a single number. From that we quote:
  - a **best estimate** (the posterior mean), and
  - a **credible interval** — the range the true value lies in with a stated
    probability. A **90% credible interval** has a 90% chance of containing the true
    value; a **95%** interval is wider (more cautious). If an interval **includes the
    "no-difference" value, we cannot claim a difference.**
- We express group differences as an **odds ratio (OR)**. OR = 1 means **no
  difference**; OR > 1 means the first group expresses _more_ of the vocabulary,
  OR < 1 _less_. (Formally it is the ratio of odds on the model's logit scale.)
- Groups, by whether the study recorded signing: **signer** (recorded signing at least
  one word), **non-signer** (assessed for signing, signed none), **sign-unknown**
  (signing was never assessed).

## Background in one paragraph

Children with Down syndrome typically understand many more words than they can say, and
many use manual signs as an early bridge to communication. A natural question is whether
signing children end up expressing _more_ vocabulary overall (signs **adding** to
speech) or _the same_ (signs **substituting** for speech). We tested this two ways:
**spoken words only** (model VG17) and **total expressive vocabulary** — any word the
child can _say or sign_, each word counted once (model VG18). Both adjust for
between-study differences using study-level random effects.

## Finding 1 — Spoken vocabulary does not differ by sign group (VG17)

Comparing spoken vocabulary at the same age, no group stands out:

| Comparison             | odds ratio | 90% credible interval | 95% credible interval |
| ---------------------- | ---------- | --------------------- | --------------------- |
| signer vs non-signer   | 1.11       | 0.89 – 1.37           | 0.86 – 1.44           |
| signer vs sign-unknown | 0.97       | 0.76 – 1.23           | 0.73 – 1.29           |
| non-signer vs unknown  | 0.87       | 0.64 – 1.18           | 0.60 – 1.25           |

Every interval comfortably includes 1. **Plain reading:** children who sign speak about
as much as those who don't — signing is not associated with less (or more) _spoken_
vocabulary.

## Finding 2 — Total expressive vocabulary: the honest answer is "no clear difference" (VG18)

Here a measurement subtlety mattered, and it flips the headline.

**First pass (all studies):** signers looked well ahead of non-signers on total
expressive vocabulary — signer vs non-signer **OR 1.65** (90% CI 1.34–2.03; 95% CI
1.28–2.12; the interval sits entirely above 1). Taken at face value this says "signs add
a lot."

**But** the "total expressive" measure was **not built the same way in every study**:
some studies counted words expressed by _speech or sign_ (a true combined total), while
others recorded only spoken words. Mixing those definitions inflates the apparent signer
advantage.

**Clean analysis (only the studies with a consistent, de-duplicated speak-or-sign
total: uk_01, uk_02, nz_01):**

| Comparison             | odds ratio | 90% credible interval | 95% credible interval |
| ---------------------- | ---------- | --------------------- | --------------------- |
| signer vs non-signer   | 0.93       | 0.74 – 1.17           | 0.71 – 1.22           |
| signer vs sign-unknown | 0.83       | 0.66 – 1.05           | 0.63 – 1.11           |
| non-signer vs unknown  | 0.89       | 0.64 – 1.21           | 0.60 – 1.31           |

Now the signer-vs-non-signer interval **includes 1** (0.74–1.17). **Plain reading:** once
total expressive vocabulary is measured consistently and cohort is accounted for, signers
and non-signers reach **comparable** overall expressive vocabulary — the earlier "signs
add a lot" result was largely a **measurement artefact**, not a real effect.

## What this means

- On this (exploratory) evidence, using signs is associated with **neither markedly more
  nor markedly less** expressive vocabulary at a given age: signs appear to travel _with_
  a child's expressive development rather than clearly boosting or replacing speech.
- The spoken-only and the clean total-expressive analyses **agree** (both null), which is
  reassuring.

## Caveats (please read)

- **Exploratory, `test`-tier** models — not the study's confirmatory models.
- **Groups differ by study/country/instrument**, not just by signing. The "signer vs
  non-signer" comparison is the cleanest because it varies _within_ the signing studies;
  the "vs sign-unknown" comparisons are confounded with which studies collected sign data.
- The **sign-unknown** group's expressive total is under-measured (signing was never
  assessed for them), so comparisons involving it are not like-for-like.
- One study (`uk_01`) records **signed-only** words rather than all signed words; this is
  correctly handled for the _total_ measure but means cross-study "amount of signing"
  comparisons need care (a separate follow-up).
- All counts are on the common **810-word reference scale**, which we separately
  **validated** as an appropriate denominator (see the report's Methods).

## Bottom line

**Children with Down syndrome who use signs express about as much vocabulary, spoken or
total, as comparable children who do not — with no clear signing advantage or deficit
once measurement and cohort are handled consistently.** This is a tentative,
exploratory finding that would need confirmatory (reporting-tier) modelling and, ideally,
data where every child's signing was assessed the same way.

_Sources: models VG17 (`model_vg17.py`) and VG18 (`model_vg18.py`); intervals are
highest-density (HDI) credible intervals from the fitted posteriors. See the dated run
record `202607121753-...md` for methods and convergence._
