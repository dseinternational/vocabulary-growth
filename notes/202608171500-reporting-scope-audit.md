# Scope-restriction audit: does every model stop where its own evidence stops?

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-08-17. **Issue:** #228. **Trigger:** a question about the typically-developing curve in the Down-syndrome-versus-typically-developing comparison plot — why it ends at about 220 understood words when the Down syndrome curve runs to 528.

The answer was VG13's 18-month age cap rather than any shortage of data, and the cap's stated justification turned out not to hold. That prompted an audit of every scope restriction in the model family against the evidence each **outcome** actually has, which found the same mistake in two more places.

## The shared defect

> A restriction justified by a property of the **pool** is only sound if that property holds for every **outcome** the pool carries.

Each finding below is an instance. None of them is a coding error; each is a rule whose stated reason was true of the data as a whole, or true when it was written, and not true of the thing it was applied to.

## Finding 1 — VG13's 18-month cap rests on two reasons, and neither survives

`VG13.max_age_months = 18`, commented: _"Restrict to 8–18 months where WG/Oxford CDI data are dense and the WS bias (production proxy comprehension) is avoided entirely."_

**The Words & Sentences half is vacuous.** `load_data` sets `td_forms = list(WORDBANK_BIVARIATE_FORMS)` whenever `understood` is requested, and only extends it with WS for a spoken-only model. WS is therefore never loaded for a comprehension model at any age. The form filter avoids the production-proxy bias unconditionally; the age cap contributes nothing to it. Verified directly: loading VG13's pool at a 25-month cap returns forms `['Oxford CDI', 'WG']` and zero null comprehension values.

**The other half was true in July and was retired in August.** `notes/202607121200-statistical-model-review.md` gives the real reason: _"above that only Oxford CDI provides bivariate rows (single study → unreliable)"_. Correct at the time — the pool was English-only. The Romance extension of 2026-08-03 (`notes/202608031500-td-romance-extension.md`) admitted Italian Words & Gestures, which Wordbank registers from **7 to 24 months**. Caselli has sat above 18 months ever since. Nobody revisited the cap; the romance note itself records "Age ranges are unchanged (VG11 8–30, VG12 8–25, VG13 8–18) — the widening adds density, not range", which is a description of what happened rather than a decision that it should.

**What the cap discards**, measured on the loader's own code path in VG13's language scope, after `min_study_observations = 200`:

| Age | Administrations | Studies | Mean understood | Max |
| --- | --------------- | ------- | --------------- | --- |
| 19  | 128             | 2       | 227             | 414 |
| 20  | 163             | 2       | 246             | 414 |
| 21  | 87              | 2       | 271             | 406 |
| 22  | 50              | 2       | 254             | 416 |
| 23  | 129             | 2       | 299             | 418 |
| 24  | 101             | 2       | 315             | 417 |
| 25  | 36              | 1       | 307             | 417 |

**694 administrations from 323 children**, all on forms the pipeline already treats as genuinely bivariate: Floccia (Oxford CDI, 591 rows, 19–25 months) and Caselli (Italian Words & Gestures, 103 rows, 19–24 months). Nothing at all is observed above 25.

**The sharpest way to put it: raising VG13's cap to 25 gives it VG12's pool exactly.** Both come to 7,052 rows over the same six studies — same rows, because the two models differ in nothing else that touches admission. VG12 already fits and reports every one of these observations. VG13 is the only model that drops them, and VG13 is the one carrying the Down-syndrome-versus-typically-developing comparison. Raising the cap adds no study and removes none; it takes the pool from 6,358 rows and 5,496 children to 7,052 and 5,819.

### Why the cap is still defensible, and what the extension costs

Two things argue for keeping it, and both are quantitative rather than decisive.

**Coverage thins sharply.** 499 administrations at 18 months against 36–163 a month above it, and 25 months is Floccia alone. Floccia's share of the pool goes from 9.7% to 17.2%, which moves the study random-effect balance.

**The Oxford CDI's 418-item ceiling binds unevenly.** Share of administrations above 90% of their own form's observed ceiling: 1.3–4.9% at 8–18 months, 7–8% at 19–22, and **20.2%, 27.7%, 36.1%** at 23, 24 and 25. Ceiling compression biases the comprehension trend down where it bites, and it bites in exactly the top three months.

**The extension is a respecification, not a `max_age_months` override.** Over 8–18 months the production ratio `q` runs from a median of 0.04 to 0.22 — the bottom limb of its S, which is what justifies `eta_q_sigma = 0.20` and a logit-linear trend between anchors at 10 and 16 months. Over 8–25 it runs to 0.83. Extending the window alone would extrapolate that trend nine months past its high anchor: on understood it reaches p = 0.85 (687 words) at 25 months against an observed median of 0.42 (340 words). So the high anchor, the GP domain, the GP anchor, the query grid and `eta_q` all have to move with the window.

### What was done

Two registered variants rather than a change to the model of record, so nothing published moves and the question gets a fit instead of an argument:

|                    | `window-25`                | `window-22`                | VG13                     |
| ------------------ | -------------------------- | -------------------------- | ------------------------ |
| `max_age_months`   | 25                         | 22                         | 18                       |
| rows / children    | 7,052 / 5,819              | 6,786 / 5,707              | 6,358 / 5,496            |
| `slope_anchors`    | (10, 24)                   | (10, 21)                   | (10, 16)                 |
| `p_slope_hi_u`     | Beta(2, 2.8), median 0.404 | Beta(2, 3.2), median 0.369 | Beta(2, 6), median 0.229 |
| `p_slope_hi_q`     | Beta(2, 1.3), median 0.630 | Beta(2, 2.6), median 0.425 | Beta(2, 7), median 0.201 |
| `gp_domain_months` | (8, 25)                    | (8, 22)                    | (8, 18)                  |
| `eta_q_sigma`      | 0.5                        | 0.5                        | 0.20                     |
| kappa anchor ages  | (12, 20)                   | (12, 20)                   | (12, 17)                 |

The high-anchor Betas are recentred just below the in-sample median at their new anchor age (24 mo: understood 0.415, `q` 0.675; 21 mo: 0.359 and 0.417), per the house convention that an anchor is recentred toward the empirical level and not tightened onto it. **They come from in-sample statistics rather than published norms, because no CDI comprehension norm exists above 18 months** — the same gap that makes VG12's 26-month anchor a named sensitivity target. That is weaker footing than the current window's priors have, and it is a limitation of the variants rather than a claim about them. The kappa anchor ages are VG12's, since VG12 is calibrated on exactly this comprehension pool over exactly this window; the magnitudes are inherited rather than recalibrated, which is a second named limitation.

Two windows rather than one because of the ceiling: `window-22` is the ceiling-safe half, and the difference between the pair measures the exposure. `window-25` is the one to fit first, because it is the one that answers the question. How far it actually extends the comparison is for the fit to say, but the in-sample medians bracket the expectation: 340 understood words at 25 months and 265 at 22, against the 220.9 at which VG13's `production_rate_by_understood` grid currently stops. VG10's runs to 527.7, so even the larger of the two leaves the top of the Down syndrome range without a matched comparator.

Both build real PyMC graphs with the same 24 free random variables as VG13 and the same six studies, so the window is the only factor. Verified by test rather than asserted: the earlier `single-admin` breakage showed that a variant can have a valid definition and an invalid graph, and `build_variant` alone does not catch it.

## Finding 2 — VG04 and VG12 publish comprehension where nothing was observed

The same mistake, in the module written to prevent it.

`reporting_ages.py` is the project's reporting-age policy, and its rule is deliberately per **quantity** — "a single figure can carry two outcomes with different support". Its closing paragraph then exempted an entire population on a pool-level premise:

> "Typically-developing models are unaffected either way: `TD_POOL_AGE_MONTHS` is `(8, 30)` and their GP domains stop at 30 or 18, so no cap here can bind."

`TD_POOL_AGE_MONTHS` is honest for **spoken**, which keeps WS and does reach 30. It is five months too generous for **understood**, which rides only on the bivariate forms and stops at 25.

**The consequence reached published output.** VG04 and VG12 both carry `ages_query` to 30. Their `posterior_summary.csv` — and the report table rendered from it — therefore quote comprehension at 27 and 30 months. VG12's published figures: 368 words \[89% 338–397] at 27 months and 380 \[304–457] at 30, on **zero** observations. Both models' _plots_ already stop at 25, because they are drawn over the observed support rather than the query grid, so each model's own report was internally inconsistent.

VG04's definition even recorded the fact and drew the wrong conclusion from it: _"Comprehension observations end at 25 months, but the declared reporting range reaches 30. This preserves the existing 8-30 month HSGP domain."_ Keeping the HSGP domain at 30 and reporting to 30 are two decisions, and `report_max_age_understood` is precisely the field that separates them — it is what lets the Down syndrome models run a domain to 115, a grid to 90 and a comprehension cap at 84.

**Fixed** by setting `report_max_age_understood = 25` on VG04 and VG12, and by replacing the exemption paragraph with the reasoning above. Reporting only: it cannot move either posterior. It does mark both fits stale by design — the field is fingerprinted for exactly this reason — so **VG04 and VG12 need refits before the next publish**. VG04 is cheap; VG12 is a high-tune fit.

A consequence worth noting separately: VG12's 26-month high slope anchor now sits one month past its last comprehension observation. That is a different matter from reporting, it is already registered as the `hi-anchor-broad` sensitivity, and it is not changed here.

## Finding 3 — the report's "TD models span 8–30 months" claim

`docs/comparison/index.qmd` states "**The TD models only span ~8–30 months**" and "the window is capped at 30 months by TD support". True of spoken, wrong of understood by five months. Not corrected in this pass, because the sentences are load-bearing in a chapter that will be re-rendered after the VG04/VG12 refits; flagged here so the correction happens with them.

## Cleared

Checked against the data and left alone.

**Every model's reported age grid against its own outcome's support.** A programmatic sweep over the whole registry — for each model and each outcome it carries, the top reported age (after any declared cap) against the last age at which that outcome is observed — returns exactly two over-reporters, VG04 and VG12, both on understood, both at 27 and 30. Every other model and outcome stops at or below its evidence.

**The Down syndrome reporting caps (84 understood, 84 signed, 90 spoken).** Re-derived against the pool four days ago in `df91f80` and `4ff48e5` and still correct: understood has 36 rows above 72 months and 12 above 84; spoken has 119 and 50; signed 107 and 47. The comprehension cap is also the high trend anchor, above which the mean is levelled off rather than fitted.

**`TD_POOL_AGE_MONTHS`'s upper bound of 30.** Its own docstring concedes the bound was inherited — "already implicit, the loader defaulted to 30" — which is the profile of a rule nobody justified. Measured anyway: extending to 36 would admit 150 further spoken rows of 18,987 (0.8%) from two datasets, and would require widening `_TD_GP_DOMAIN_MONTHS`, which VG03 and VG04 share. Same trade, and the same answer, as the five 7-month Italian rows that gave the pool its lower bound. Left as it is, with the measurement recorded in `reporting_ages.py` so the next person does not have to redo it.

**VG03 and VG04 staying English-only.** Justified structurally rather than empirically — they carry no random effects, so between-language spread would be absorbed by the Beta-Binomial dispersion and misreported as child-level dispersion. That reason does not decay.

**`min_study_observations = 200`.** Drops three studies and 136 of 6,494 rows on VG13, and the same three across the widened window. No study crosses the threshold when the window changes.

**`TD_POOL_EXCLUDED_DATASETS`, `SIGNED_ONLY_STUDIES`, `INCOMPLETE_ADMINISTRATION_CEILINGS`.** Each names a specific documented defect with a reinstatement path. `UNCERTAIN_SIGN_STUDIES` is the counter-example that shows the process works: it held `uk_06` until the source confirmed the construct, and was then emptied rather than left to ossify.

## Follow-ups

1. Fit `vg13 window-25`, then `window-22`, and compare against VG13 — the deliverables are how far the matched-comprehension comparison extends and whether the two windows disagree in the way the ceiling analysis predicts.
2. Refit VG04 and VG12 before the next publish (stale by the cap change).
3. Correct the "TD models span 8–30 months" sentences in the comparison chapter alongside those refits.
4. If `window-25` converges and the comparison is worth publishing, promote it from a variant to a registered model rather than leaving a headline comparison resting on a sensitivity entry.
