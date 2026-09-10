# Our fitted trajectories against Gompertz growth curves

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-10. **Question:** the study owner asked whether our typically-developing and Down syndrome understood/spoken fits can be evaluated against Gompertz curve fits, which Day, Borovsky, Thal & Elison (2025) propose as the functional form for CDI inventory totals. **Evidence:** the `rep` fits of VG12, VG11, VG20, VG02 and VG01, and the paper's published parameters (retrieved from PubMed: PMID 40521961, [doi:10.1111/desc.70036](https://doi.org/10.1111/desc.70036)). **Record:** `scripts/experiments/gompertz_comparison.py`, writing under `<output-root>/experiments/gompertz-comparison/`. Descriptive; nothing in the report or the fit pipeline reads it.

**Four findings.** (1) Our **typically-developing production** curve is Gompertz to within 3 words over its whole range, and its age of maximum growth and peak rate reproduce the paper's published values on a partly different pool — an independent corroboration of both. Its growth-rate parameter sits 12–17% below theirs, for the measurement reason §5 gives. (2) Our **typically-developing comprehension** curve is not: its maximum growth falls at about half its reachable ceiling, where a Gompertz requires 1/e. (3) For **Down syndrome**, the paper's two-parameter form — asymptote fixed at the form's item count — is not merely a worse fit but the wrong model: freeing the asymptote halves to thirds the error and puts it at 0.66–0.72 of the checklist. The fixed-asymptote version has to place maximum growth at 47–66 months and reaches 0.9 of the checklist at 10–11 years, which contradicts our own forward projection. (4) In the paper's own currency the Down syndrome delay is large: age at maximum growth 41.5 months against 15.0 for comprehension and 64.4 against 23.3 for production, with growth rates 4–5 times lower. Their clinical comparison group was 3.7 months late.

What this does **not** establish is whether a Gompertz mean would predict our data as well as our Gaussian-process mean does. §6 says why, and what would settle it.

## 1. What the paper proposes

According to PubMed, Day et al. (2025), _Developmental Science_ 28(4):e70036 ([doi:10.1111/desc.70036](https://doi.org/10.1111/desc.70036)), make two contributions. Part 1 is a procedure for projecting Words & Gestures scores collected above the instrument's normed age range onto Words & Sentences scores (§7 below). Part 2, the one asked about here, fits **Gompertz growth curves** to CDI production totals:

$$W(t) = A \exp\left(-\exp\left(-k_g (t - T_i)\right)\right)$$

with the asymptote $A$ **fixed** at the form's item count (they use max + 1, so 681) and the lower one effectively at zero, leaving two free parameters: $T_i$, the age of maximum growth, and $k_g$, a dimensionless growth rate. The maximum rate in words per month is $k_U = k_g A/e$, reached exactly when the curve is at $A/e$. Fitted by nonlinear least squares, per individual (requiring at least four timepoints, one of them past the inflection) or per group.

**The form above is reconstructed, not copied.** The PMC rendering of their equation loses its exponents, so what is fitted here is the two-parameter member of the Gompertz family that satisfies both properties they state and use. Three arithmetic checks confirm the reconstruction reproduces their published numbers: with $k_g = 0.161$ and $A = 680$ it gives a maximum rate of 40.3 words a month, which is their headline rate; it puts 250 words — the threshold whose age they report as 23.3 months — at exactly $T_i$, because $680/e = 250.2$; and refitting it to a noiseless curve generated from those parameters recovers them to six decimal places.

Their published values, for comparison below. Pooled fits to the Wordbank pool give $k_g = 0.161$ in Part 1 and 0.17 in Part 2 — read our number against that band, not against one end of it — with $T_i = 23.3$ months, and 0.15 and 0.17 for their two other cohorts. Individual-level group means give $k_U$ between 34.3 and 39.3 words per month and $T_i$ between 22.1 months (children later screened as having no language disorder) and 25.0 (children who received a speech, language, learning or reading diagnosis at 4–7 years), with their other cohort's mean at 24.3. Their headline is that maximum growth is about **40 words a month at around 24 months**, much faster than the 10–17 words a month the older "vocabulary spurt" literature reports, and that growth rates are unimodally distributed — no distinct classes of "spurters" and "non-spurters".

Justifying the fixed asymptote, they argue it need not vary because every child starts at zero and "will eventually reach 680 words in typical development". That premise is the hinge of finding (3).

## 2. What can be tested without a refit

Three legs, none of which needs any model refitted.

**The form's parameter-free implications.** A Gompertz reaches its maximum growth at exactly $A/e$ = 36.79% of its asymptote, whatever $k_g$ and $T_i$ are (a logistic would say 50%). So "where, as a fraction of the ceiling, does maximum growth happen?" tests the shape of a fitted curve against the form without fitting the form at all. Our models already report the age of maximum growth as an estimand (`expected_learning_rate_peak.csv`); this adds the fraction of the ceiling there.

**The form fitted to our own posterior curves.** Per posterior draw, the two-parameter version (asymptote fixed) and a three-parameter version (asymptote free) fitted to the stored population curve over the model's own reporting age range, giving $k_g$, $T_i$ and the RMSE in words. This measures how much of our fitted shape the form can express.

**The paper's method on our rows.** Their nonlinear least squares fit, applied to the observed administrations in each analysis frame, with a child-clustered bootstrap for intervals. This is what makes our numbers comparable with their published ones, and it is a property of the pool rather than of the model — VG01 and VG02 consume exactly the rows VG20 does, so it is fitted once per population and outcome.

**One measurement caveat runs through all of it.** $k_g$ and the 1/e test are both defined relative to an assumed asymptote, and our pools have no single one: every count is scored against 810 whatever form produced it, and each frame is a mixture of instruments with different item counts (`data_utils.WORDBANK_FORM_ITEMS` is the authority). The typically-developing **spoken** frame is roughly three-fifths Words & Sentences — the paper's instrument, 680 English items — with Words & Gestures production and the 416-item Oxford CDI making up the rest. The typically-developing **comprehension** frame carries no Words & Sentences rows at all, because Wordbank's Words & Sentences comprehension column repeats production (the defect that retired VG06, `notes/202605151630-vg06-ws-comprehension-issue.md`); it is Words & Gestures and the Oxford CDI, whose comprehension lists run 309 (Spanish European), 396 (English American), 408 (Italian) and 416–418 (Oxford) items. The Down syndrome frames are on native 810-item forms.

Both denominators are therefore reported: the modelled 810, and the largest count in the frame as an empirical proxy for the reachable ceiling. Note which way each choice cuts. For typically-developing comprehension the proxy, 418, is the **most generous** available — it is the largest form in the pool, and the modal form has 396 — so it flatters the Gompertz, and §3's conclusion survives it. For typically-developing production the proxy is exactly the paper's 680, which is right for the three-fifths of rows on that form and too high for the rest, so our $k_g$ is biased slightly low against theirs. $T_i$ is scale-free and needs none of this care, which is why it carries the strongest comparison in §5.

## 3. Where maximum growth happens: the parameter-free test

Population median child, over each model's own reporting age range. A Gompertz requires 0.368 of its asymptote; a logistic 0.500.

| Curve              | Peak age (89%)    | Rate at peak | Of 810 | Of frame max | Implied asymptote |
| ------------------ | ----------------- | -----------: | -----: | -----------: | ----------------: |
| VG12 TD understood | 17.7 [17.3, 18.2] |  29 words/mo |  0.254 |  0.492 (418) |               559 |
| VG11 TD spoken     | 24.0 [23.7, 24.5] |  40 words/mo |  0.287 |  0.342 (680) |               633 |
| VG20 DS understood | 38.1 [22.8, 64.0] |  11 words/mo |  0.288 |  0.300 (776) |               634 |
| VG20 DS spoken     | 52.0 [46.4, 73.9] |  12 words/mo |  0.228 |  0.244 (759) |               503 |
| VG02 DS understood | 38.7 [24.1, 43.7] |  12 words/mo |  0.337 |  0.351 (776) |               741 |
| VG01 DS spoken     | 45.7 [41.5, 89.2] |  12 words/mo |  0.188 |  0.200 (759) |               413 |

"Implied asymptote" inverts the test: the asymptote a Gompertz would need for its 1/e point to land where our curve's maximum growth actually falls. Read that way the test is not a verdict but a measurement of the parameter the paper fixes — and for the two headline models it agrees with the free-asymptote fit of §4 independently (VG11: 633 against 665; VG20 spoken: 503 against 532).

**VG11 TD spoken is the case that matters most**, because it is the same instrument and population as the paper. Its maximum growth is at 0.342 of the Words & Sentences ceiling against the required 0.368 — a 7% discrepancy in the implied asymptote, on a curve fitted with no Gompertz assumption anywhere. Its peak rate of 39.6 words a month at 24.0 months [23.7, 24.5] is the paper's headline claim almost exactly.

**VG12 TD understood is the clear departure.** At 0.492 of its reachable ceiling it is nearly at the logistic's symmetric half-way point. That conclusion does not turn on the denominator: getting the peak to 1/e would need a comprehension form of 559 items, and the largest in the pool has 418 (the modal one 396). The paper does not claim comprehension — Words & Sentences asks only about production, on the grounds that parents are unreliable reporters of comprehension past 16 months — so this is our extension of their form, not a failure of it. Note also that VG12's window stops at 25 months and its asymptote is entirely unobserved.

Two peak estimates are too uncertain to carry weight: VG20 DS understood, 89% [22.8, 64.0], and VG01 DS spoken, whose 89% upper limit is the top of its trimmed reporting range (90 months) with 5% of draws peaking there.

## 4. The Gompertz fitted to our own curves

RMSE in words against our own population curve, over the model's reporting range. `A` fixed at the frame's observed maximum is the closest analogue of the paper's choice; the free-`A` variant adds one parameter.

| Curve              | Variant     |          $k_g$ (89%) | $T_i$ | Asymptote |         RMSE (89%) | 0.9 A at |
| ------------------ | ----------- | -------------------: | ----: | --------: | -----------------: | -------: |
| VG12 TD understood | fixed A=418 | 0.177 [0.168, 0.185] |  15.4 |       418 |    9.8 [7.9, 11.8] |  28.1 mo |
| VG12 TD understood | free A      | 0.131 [0.122, 0.140] |  17.3 |       529 |     5.3 [3.7, 7.4] |  34.4 mo |
| **VG11 TD spoken** | fixed A=680 | 0.150 [0.147, 0.153] |  24.5 |       680 | **3.1 [2.3, 4.3]** |  39.5 mo |
| VG11 TD spoken     | free A      | 0.153 [0.146, 0.162] |  24.3 |       665 |     3.0 [2.3, 4.0] |  39.0 mo |
| VG20 DS understood | fixed A=776 | 0.031 [0.028, 0.033] |  46.6 |       776 |   13.8 [9.5, 20.0] | 120.1 mo |
| VG20 DS understood | free A      | 0.043 [0.035, 0.051] |  37.1 |       580 |    8.4 [5.0, 12.3] |  89.8 mo |
| VG20 DS spoken     | fixed A=759 | 0.034 [0.031, 0.038] |  64.1 |       759 |   15.7 [9.8, 24.1] | 130.0 mo |
| VG20 DS spoken     | free A      | 0.054 [0.044, 0.068] |  54.0 |       532 |    5.6 [2.4, 11.0] |  95.7 mo |
| VG02 DS understood | fixed A=776 | 0.030 [0.028, 0.033] |  43.0 |       776 |  27.2 [19.2, 35.3] | 117.5 mo |
| VG02 DS understood | free A      | 0.061 [0.051, 0.071] |  29.7 |       492 |    9.1 [3.9, 16.7] |  66.7 mo |
| VG01 DS spoken     | fixed A=759 | 0.025 [0.023, 0.027] |  67.6 |       759 |  24.2 [18.1, 33.0] | 159.0 mo |
| VG01 DS spoken     | free A      | 0.055 [0.042, 0.073] |  46.2 |       410 |   10.9 [5.4, 18.0] |  87.1 mo |

**Typically-developing production: the form is enough.** 3.1 words of error over 8–30 months on a curve that runs from 1 to 430 words, and freeing the asymptote buys 0.1 of a word — the two-parameter Gompertz has essentially captured our whole fitted trajectory. Whatever the Gaussian process is doing in VG11's mean, it is not shape a Gompertz cannot express. The free asymptote lands at 665 words, which independently reproduces the 664 the forward projection's own Gompertz arm fitted over 12–30 months ([`202609101115`](202609101115-vg11-forward-projection-to-5-years.md) §4, 0.82 of 810) — two harnesses, one number.

**Down syndrome: the fixed asymptote is the problem, not the family.** Freeing it cuts the error from 13.8 to 8.4 words (understood) and 15.7 to 5.6 (spoken), and lands it at 0.72 and 0.66 of the checklist. The direction is the one the fixed version cannot represent: our Down syndrome curves are heading for a ceiling well below the form's, and forcing them at 810 makes the form defer its inflection to 47–66 months and its 0.9 point to 10–11 years. That last number is checkable against our own work, and it fails: the forward projection recorded in [`202609091900`](202609091900-vg20-forward-projection-to-11-years.md) puts the median child with Down syndrome at 524 spoken words at 11 years, not the 0.9 × 759 = 683 the fixed-asymptote Gompertz implies. The free-asymptote version agrees with it: the 532-word asymptote it fits to the spoken curve here is within three words of the 535 the projection's Gompertz arm gives at 11 years. That is a cross-check rather than an identity — the projection reaches its number by crossing a Gompertz on comprehension with one on the production ratio, not by fitting one to spoken directly — but the two routes landing three words apart is the reassurance it looks like.

**The single-level baselines fit the form worst** (27.2 and 24.2 words fixed; 9.1 and 10.9 free), which is a point about our models rather than the form: VG01 and VG02 carry no random effects to absorb between-study variation, and their population curves are correspondingly more contorted.

## 5. The paper's method on our rows, against their published values

Nonlinear least squares on the observed administrations, asymptote at the frame's observed maximum, 200 child-clustered bootstrap resamples.

| Pool          |   Rows |   A |         $k_g$ (boot) |         $T_i$ (boot) | $k_U$ words/mo |
| ------------- | -----: | --: | -------------------: | -------------------: | -------------: |
| TD understood |  7,049 | 418 | 0.137 [0.132, 0.142] | 15.02 [14.92, 15.12] |           21.0 |
| TD spoken     | 18,500 | 680 | 0.141 [0.139, 0.144] | 23.26 [23.16, 23.34] |           35.4 |
| DS understood |  1,225 | 776 | 0.033 [0.031, 0.036] | 41.54 [40.40, 42.89] |            9.5 |
| DS spoken     |  1,394 | 759 | 0.029 [0.027, 0.031] | 64.41 [62.28, 66.69] |            8.0 |
| _Day et al._  |      — | 680 |         _0.161–0.17_ |               _23.3_ |        _34–39_ |

**The typically-developing production row is the comparison the paper licenses, and it agrees.** $T_i$ 23.26 months against their 23.3, and $k_U$ 35.4 words a month inside their 34.3–39.3 individual-level range. $k_g$ is 12–17% below their 0.161–0.17, in the direction the measurement caveat in §2 predicts: their fit is Words & Sentences throughout, ours mixes three form sizes on one 810-item denominator, so rows from the smaller forms enter deflated and flatten the fitted rate. Our pool is also scoped to English **and** Italian and Spanish (European), applies its own admission and measurement rules, and mixes studies whose intercepts span 1.2 logits. Two pools, two methods, two research groups, the same age of maximum growth to within a tenth of a month.

**The Down syndrome rows put the delay in their units.** Maximum growth 26 months later for comprehension (41.5 against 15.0) and 41 months later for production (64.4 against 23.3), with growth rates a quarter to a fifth of the typically-developing ones and maximum rates of 8–10 words a month against 21–35. For scale, the largest contrast the paper reports — children who went on to receive a speech, language, learning or reading diagnosis, against those screened clear — is 3.7 months of $T_i$ and 34.3 against 39.3 words a month.

## 6. What this does not settle, and the arm that would

Everything above is **shape**. None of it says a Gompertz mean would _predict_ our data as well as our mean function does, and three things stop it being read that way.

- The RMSEs in §4 are against our own fitted curve, not against data. They measure how much of our shape the form can express, which is not the same as how well either describes a child.
- Against the observed rows the two mean functions are within a few words of each other — TD spoken, Gompertz 120.9 words against our curve's 124.8; DS understood, 112.5 against 115.2 (VG20) and 110.8 (VG02) — but those numbers are dominated by between-child scatter that neither mean function models, and the comparison is biased against ours, because our curve is the **median** child at zero study and child effects while the least-squares fit targets the row mean (the pools' row-weighted mean study effect is positive: +0.16 logits in the typically-developing spoken pool). They establish only that nothing dramatic separates the two at the population-mean level.
- A Gompertz mean inside our likelihood would be a different model in more ways than its shape: two parameters against a logit-linear trend plus a 16-coefficient HSGP, and no natural place for the trend anchors the priors are calibrated on.

**The arm that would settle it** is a mean-function swap with everything else held fixed: replace `p(age) = expit(trend + GP)` with `p(age) = exp(-exp(-k_g(age - T_i)))` (asymptote free, so the Down syndrome case is representable), keep the Beta-Binomial likelihood with age-varying dispersion, keep the study and child random effects, fit both arms at the same sampling configuration, and compare by LOO on the same rows. That is four models × two arms. It must be built as an experiments-local graph, not a definition field, because a new mean-function branch in `src/vocab_growth/` restales every existing fit's executable-code signature — which is exactly what `scripts/experiments/` exists for. At `test` configuration the Down syndrome arms are minutes each and the typically-developing hierarchical arms are the expensive half (VG11 and VG12 carry 14,553 and 5,819 child effects); budget a few hours for the set, and note that a `test`-tier comparison of two mean functions is a legitimate answer to "which mean function", not to "what is the number".

Worth predicting before running it, so the result is not read after the fact: on §4's evidence the typically-developing production arms should come out indistinguishable by LOO, and the Down syndrome arms should favour the flexible mean, because a free-asymptote Gompertz still cannot bend where the free asymptote itself has to be inferred from data that stop at 7.5 years.

## 7. Two side notes worth recording

**Their Part 1 speaks to our above-window admissions.** Day et al. validate using Words & Gestures **above** its normed age range for typically-developing children, and publish (with an R package) a conversion of those scores to Words & Sentences ones. It is validated between 18 and 27 months, with a conservative recommendation to stop at 22–24 months or 250 words. We admit above-window administrations for the Down syndrome pool on the developmental argument that an early-vocabulary form given to an older child is appropriate, and `us_01`'s only comprehension observations between 19 and 27 months are of exactly that kind. Their result is independent support for the direction of that decision, with one limit they state plainly: their conversion is built on typically-developing Wordbank data and they decline to validate it for children with intellectual and developmental disabilities. So it corroborates admitting such rows; it does not license converting ours.

**Their reasoning about the ceiling is ours.** They are explicit that the apparent slowing of growth near the form's ceiling is not a real fall in the rate of word learning but the checklist running out of items to record it with — each newly learned word is progressively less likely to be one of the 680. That is the same argument [`202609101115`](202609101115-vg11-forward-projection-to-5-years.md) §4 makes for why a projection past the observed range describes the checklist rather than the child, arrived at independently.

## 8. Reproduce

```bash
uv run python scripts/experiments/gompertz_comparison.py
```

Each fit is validated through `vocab_growth.fit_consumers` before it is read (registered definition, raw-data fingerprint, exact prepared-frame hash), so a refit or a data-rule change stops the harness until `--allow-stale-fit` is passed. Outputs land under `<output-root>/experiments/gompertz-comparison/`: `shape_diagnostics.csv`, `curve_fits.csv`, `data_fits.csv` and `gompertz_comparison.png`.
