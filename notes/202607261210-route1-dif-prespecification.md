# Route 1 differential item functioning: pre-specification

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Pre-specification, 2026-07-26, committed **before** any item-level data enters the repository — the git history is the timestamp. It binds the Route 1 analysis described in [`202607261048-incorporating-item-difficulty.md`](202607261048-incorporating-item-difficulty.md) §6 once execution begins; until then two decisions marked **[owner decision]** remain open and must be settled by the study owner, not the analyst. Deviations after execution begins are permitted but must be logged in this note's §10 with their reason, and the primary estimand, threshold and decision rule may not be changed after any Down syndrome item-level response has been seen.

## 1. Purpose and scope

This note pre-specifies the first empirical test of the working scientific assumption that **children with Down syndrome learn words in roughly the same order as typically-developing children, but later**. It covers Route 1 only: the Wordbank Down syndrome subset (the Edgin dataset, which is this repository's `us_01`) against typically-developing children on the _same_ CDI forms, which requires no item crosswalk. Route 2 (the DSE-inventory calibration and its crosswalk) will be pre-specified separately after the item-data ingest audit, and nothing here licenses any change to a model of record — under the revised plan on the PR #183 thread, all model changes (Proposals A/B/C) are downstream registered sensitivities gated on this analysis and on observed-scale predictive validation.

Pre-specifying before acquisition matters here for a specific reason: the analyst has already seen the _aggregate_ Down syndrome data extensively, and the companion notes make directional predictions. Fixing the estimand, threshold and decision rule now is what keeps the answer from being fitted to the expectation.

## 2. The claim under test, stated correctly

The scientific hypothesis decomposes onto two measurement parameters, and — per the correction adopted on the PR thread — cross-sectional item data can test only the measurement claim, not the developmental mechanism:

- **"Later"** is an ability offset: the age shift `Delta(age)` such that `mu_TD(age - Delta) = mu_DS(age)` on the latent scale. This is expected, is not in dispute, and is estimated as a nuisance-with-interest, not tested.
- **"Same order"** is measurement invariance of item difficulty: the population-specific perturbations `delta_j` being small relative to the shared difficulty spread. This is the claim under test.

What a conclusive answer licenses: order-equivalence supports sharing calibrated difficulties across populations and gives the matched-comprehension comparisons a tested foundation at these ages on this instrument; material DIF means matched-count comparisons conflate propensity with composition in a way no reweighting fixes, and cross-population reporting moves to the latent scale under an explicit linking convention. Either outcome is informative; the design is not rigged for one of them.

Two scope limits stated up front. First, this tests a shared item _hierarchy_ plus an age-associated offset — it says nothing about within-child acquisition order or learning speed, which need longitudinal item-level data. Second, invariance is tested over the region the Down syndrome data inform: ages 11–30 months, floor-heavy, so hard items will carry wide posteriors and the verdict is scoped to the item range the coverage report (§8) shows is actually informed.

## 3. Data

**Down syndrome.** Wordbank, `dataset_name = "Edgin"`, `language = "English (American)"`, `health_conditions` matching Down syndrome: 196 administrations from 119 children — WG 87 administrations / 53 children (ages 11–18 months; comprehension and production), WS 109 / 66 (ages 17–30; production only). 75 children have two administrations and one has three, so a child-level effect is mandatory, and many children link the two forms within-child.

**Six WG administrations are excluded before fitting**, under the duplicated-outcome rule established in [`202607261245-edgin-duplicated-outcome-records.md`](202607261245-edgin-duplicated-outcome-records.md) and implemented in `data_utils.mask_duplicated_outcome_administrations`: production within 1–9% of comprehension at 11–18 months, which is a data-preparation artefact rather than exceptional ability on three independent lines of evidence. This exclusion is **not** outcome-selection of the kind §5 forbids — it is a pre-specified validity criterion fixed before any item-level response is seen, its false-positive rate is measured (0.69% in a 2,480-administration typically-developing reference), and §8.7 both re-runs the analysis with the records reinstated and verifies the mechanism directly once item-level vectors exist. The Route 1 sample is therefore **81 WG administrations from 50 children** for comprehension (three children — 81114, 81122, 81132 — lose their only WG record and leave the comprehension analysis entirely) and **190 administrations from 116 children** overall for production. Beyond this rule, no administration may be excluded on the basis of its responses.

**Typically developing.** Wordbank, `language = "English (American)"`, `form` in {WG, WS}, `typically_developing = true`, `health_conditions` null: 4,924 WG administrations (4,230 children, ages 8–18) and 8,859 WS (6,826 children, 16–30). A stratified random subsample of **3,000 administrations** (stratified by form and age in months, proportional allocation, `random_seed = 47`) is taken for computational tractability; the subsampling script and the selected administration identifiers are committed with the ingest.

**Acquisition and provenance.** The item-level ("instrument data") export for both forms is pulled from Wordbank with: download date, source URL, query filters, row counts and file checksums recorded in a manifest committed alongside the data, per the provenance requirement adopted on the PR thread. The pull must include item metadata (`item_definition`, lexical category) — the class analysis depends on it. Storage follows the repository's current direction (Parquet registered with the DuckDB build rather than CSV) — an ingest decision, not an analysis one. The known descriptive facts above come from the by-child export already in the repository; if the item-level pull's administration counts disagree with them, the discrepancy is resolved and documented before any model is fitted.

**Item identity across forms.** Production items shared between WG and WS (expected roughly 380–400; the exact linking set is reported by the ingest audit) are matched on `item_definition` and treated as the same item with one difficulty. The primary analysis assumes no form effect for a shared item; a form-offset sensitivity is pre-specified in §8.

## 4. Model

One-parameter logistic (Rasch) item response model, binary response of administration `t` of child `i` to item `j`:

```text
P(y_itj = 1) = logistic( theta_it - d_j - delta_j * 1[i in DS] )

theta_it = mu_g(i)(age_it) + u_i + e_it        person side: population trend, child effect, occasion effect
d_j      ~ Normal(0, sigma_d),  sum_j d_j = 0   shared difficulties (location fixed: the Rasch alias)
delta_j  ~ Normal(0, sigma_DIF), sum_j delta_j = 0   DIF (zero-sum: the group-shift alias)
```

- `mu_g(age)` is population-specific (linear plus quadratic in standardised age; a spline alternative is a §8 sensitivity). Person-side misspecification inflates the person variances, not `delta` — the person side is genuinely nuisance for the invariance estimand.
- Both identification constraints are required, per the review: `sum(delta_j) = 0` alone leaves the Rasch location alias between `theta` and `d`, so `sum(d_j) = 0` is imposed as well. Implementation follows the codebase's existing `ZeroSumNormal` idiom with the `sqrt(J/(J-1))` marginal-variance rescale.
- Comprehension and production are **separate models** (separate `theta`, `d`, `delta`, and hyperparameters). No cross-outcome structure on the first pass.
- Discrimination is fixed at 1: this tests **uniform** DIF only. Order-failure through unequal discriminations (a 2PL) is an acknowledged blind spot, handled as exploratory (§6).
- Priors: `sigma_DIF ~ HalfNormal(0.5)` primary, with `HalfNormal(0.25)` and `HalfNormal(1.0)` as pre-specified sensitivity brackets; remaining hyperpriors are fixed in the analysis script, which is committed and frozen **before** the Down syndrome item file is opened, and a prior-predictive check is run and reviewed per the project workflow before sampling. The committed script, not this prose, is the authority on every hyperparameter not named here.
- Estimation: PyMC with nutpie, the project's standard. The project's hard convergence gate applies (R-hat ≤ 1.01, bulk and tail ESS ≥ 400 on every sampled parameter); a fit failing it is refitted or reported as failed, never summarised.

## 5. Primary estimand and threshold

**Primary estimand.** `r = sigma_DIF / sigma_d` for **production** (both forms, all 196 Down syndrome administrations), where both scales are the fitted hierarchical standard deviations. `r` is an SD ratio; its variance share is `r^2 / (1 + r^2)` — the two must not be conflated (adopted from the review).

`r` translates into ordering language in closed form under the normal working model, which is what makes a threshold on it meaningful. For a random item pair, the probability that the Down syndrome difficulty ordering reverses the shared ordering is `arctan(r) / pi`, and the correlation between the two difficulty vectors is `1 / sqrt(1 + r^2)`:

| `r`  | pairwise reversal probability | difficulty-vector correlation |
| ---- | ----------------------------- | ----------------------------- |
| 0.15 | 4.7%                          | 0.989                         |
| 0.30 | 9.3%                          | 0.958                         |
| 0.60 | 17.2%                         | 0.857                         |
| 1.00 | 25.0%                         | 0.707                         |

(25% is the reversal rate when population-specific variation equals the shared spread; 50% would be unrelated orderings.) Alongside the closed form, the **empirical posterior reversal rate** — computed per posterior draw from the realised `d_j` and `delta_j`, overall and as a function of the pair's difficulty gap — is reported as the model-free companion, since near-tied items reverse trivially and the gap-resolved curve is the honest picture. Rank-reversal probability is the estimand closest to the scientific phrase "same order" (adopted from the review); `r` is primary because it is the parameter the model estimates directly and the closed form ties them together.

**Practical-equivalence threshold [owner decision].** Provisionally `r* = 0.30` — under 10% of random pairs reversed, difficulty vectors correlated at 0.96. Justification: (i) it is comfortably above the pseudo-DIF noise floor the negative control (§8) will measure, or the design is reworked; (ii) at the CDI's difficulty spread it permits item-level perturbations of a few tenths of a logit, about the size of routine instrument variation the project already tolerates when pooling forms; (iii) it is far below `r = 1`, where ordering is substantially population-specific. **This threshold is the scientific judgement in the analysis and must be confirmed or amended by the study owner before execution; after execution it is frozen.** The benchmark comparison in §8 (TD form-DIF) will contextualise it but cannot move it.

**Decision rule.** Using the project's interval convention (89%, ETI) on the posterior for `r`:

- **order-equivalent** if the 89% interval lies entirely below `r*` _and_ the negative control passed;
- **material DIF** if the 89% interval lies entirely above `r*`;
- **inconclusive** otherwise — reported as such, with the interval, and _not_ resolved by post-hoc threshold movement.

**Consequences, fixed in advance.** Order-equivalent: shared difficulties are licensed for these ages and this instrument; Proposals B/C may proceed to sensitivity fitting with a shared calibration; the matched-comprehension contrast gains a tested (age-scoped) foundation. Material DIF: cross-population contrasts move to the latent scale; the pre-specified linking convention is the all-item zero-sum reported alongside an anchor-purified variant (lowest-|`delta`| quartile as anchors, one refit — exploratory grade); the class profile of `delta` becomes a primary descriptive result. Inconclusive: Route 2 (more children, wider ages, the DSE inventory) is the remedy, not more analysis of these data.

## 6. Secondary and exploratory quantities

**Secondary, with pre-registered directions.**

- Comprehension `r` (WG only, 87 administrations / 53 children). Same pipeline, same threshold. Directional prediction, from the companion investigation's finding that the production gradient sits in `P(say | imitate)` while `P(imitate | understand)` is flat: **DIF, if material anywhere, is larger for production than comprehension.**
- Class-level DIF: mean `delta` by lexical class (sound effects/routines; nouns; predicates; function words — the standard CDI groupings, from the item metadata). Directional prediction: production DIF, if present, concentrates in phonologically demanding classes rather than function-versus-content per se. Class contrasts are estimated from the fitted `delta_j` with their posterior uncertainty, not by refitting per class.
- The ability offset `Delta(age)`, reported as a curve with its interval — the "later" parameter, descriptive.

**Exploratory, labelled as such wherever reported:** per-item `delta_j` (shrinkage-dominated at these n); a 2PL refit (non-uniform DIF); word-length or phoneme-count as a difficulty moderator if the metadata supports it.

## 7. Power and honesty about this sample

The Down syndrome responses are heavily degenerate, and the design analysis must use the real layout rather than idealised n. Known from the by-child export: WG production has median 2 words with 33 of 87 administrations at zero; WS production has median 11 with p90 at the 680-item form ceiling (the known Edgin ceiling block); WG comprehension is the least degenerate outcome (median 38, only 4 of 87 at zero). An all-zero or all-ceiling administration informs ability bounds but contributes nothing to ordering, so the effective sample for DIF is the informative middle, which is thinner than 196 administrations suggests — and concentrated on the easy half of the inventory.

Consequently the **design analysis is a gate, not a formality** (§8): if it shows the production analysis has a low probability of a conclusive verdict at plausible `r`, the pre-specified response is to report Route 1 as a feasibility study with the interval it achieved — not to quietly re-run variants until something concludes. The same applies to comprehension, which may turn out to be the better-powered outcome despite fewer administrations; the design analysis will say so _before_ any Down syndrome item response is seen, and the primary-outcome designation in §5 **[owner decision]** may be revised on the basis of the design analysis alone, never on the basis of the real data.

## 8. Controls, design analysis, and sensitivities

1. **Negative control (noise floor).** Split the TD subsample into random halves (seed 47), run the identical pipeline with the half indicator in place of the population indicator. The resulting pseudo-`r` is the measurement floor; the primary threshold must sit clearly above it, and the equivalence verdict requires this control to have passed.
2. **Simulation-based design analysis, before real data.** Simulate the full pipeline at the actual layout — real administration counts, ages, forms, child structure, and response-margin degeneracy — under true `r` in {0, 0.1, 0.2, 0.3, 0.5}, with enough replicates per point to estimate the decision-rule's operating characteristics (probabilities of each verdict at each true `r`). This is the recovery-and-boundary diagnostic the review required before treating `sigma_DIF` as identified.
3. **Prior sensitivity.** The `sigma_DIF` prior brackets in §4, reported alongside the primary.
4. **Ceiling sensitivity.** Refit with the Edgin WS 680 block _reinstated_ (`include_implausible_production=True`), not excluded. The direction reversed after the audit: that block is excluded by default now, and the retired `us01-ceiling-excluded` variant is no longer a precedent to mirror — it could only have excluded records already gone. Since the source author's original files are no longer available, no exclusion here can be confirmed at source, which makes reporting `r` both ways obligatory rather than optional. The aggregate models take the same direction through the registered `us01-implausible-reinstated` variants, so Route 1 and the models of record answer the same question the same way.
5. **Form-DIF benchmark.** Within TD only, estimate WG-versus-WS "DIF" for shared production items — the same word asked on two forms. This is instrument variation the project already tolerates by pooling forms, and gives the threshold an empirical comparator: Down syndrome `delta` of similar magnitude is no worse than routine cross-form variation.
6. **Person-trend sensitivity.** Replace the quadratic `mu_g` with a spline; `r` should be insensitive if the person side is truly nuisance.
7. **Duplicated-outcome sensitivity, and a direct test of the mechanism.** Refit with the **eight** excluded WG administrations reinstated (`include_duplicated_outcomes=True` — the count rose from six when the ratio threshold was set from the measured gap), and report `r` both ways: if the verdict turns on eight records, that is a result about fragility and must be stated as one. Separately, the item-level pull may afford a decisive check the aggregate data cannot give — a duplicated column appears as two _identical_ response vectors, and a form written to its maximum as every item marked known. This check does **not** depend on the source team, whose original files are no longer available: `us_01` is a subset of the public Wordbank export, so the relevant item-level responses, if published at all, come from Wordbank. Whether an item-level export exists for the Edgin dataset is unknown. Report, for each of the eight, whether the comprehension and production vectors are identical, near-identical, or unrelated — or record "no item-level export available for this dataset", which is a permitted and reportable outcome, not a failure to be worked around. A refutation reinstates them and is logged in §10.
8. **Retained-but-unusual sensitivity.** Refit excluding the two 18-month administrations with high infant comprehension and a normal production gap (comprehension 213 and 217). These are retained in the primary analysis on the study owner's judgement that they are clinically unusual but not invalid; this sensitivity shows whether the comprehension verdict depends on them.

## 9. Known limitations, conceded in advance

- **Age and range.** Ages 11–30 months, floor-heavy: the verdict covers the easier region of the inventory at young ages. It does not test ordering at the ages where the DSE data live (to 86 months) — that is Route 2's job.
- **Response process.** Parent report. A uniform reporting-style difference between populations is absorbed by the ability offset and does not contaminate `r`; an item-_specific_ reporting difference (say, over-reporting comprehension for gesture-associated words in Down syndrome) is formally indistinguishable from DIF. A material-DIF verdict is therefore "measurement non-invariance", which includes reporting effects, not necessarily "different acquisition order".
- **Uniform DIF only** under the 1PL; discrimination differences are exploratory.
- **One instrument, one language, one contributing DS dataset.** Generalisation beyond the American-English CDI and the Edgin cohort is Route 2's question.
- **The repeated administrations are close in age** (WG then WS for many children), so the child effect is identified mainly from that pairing; `u_i` and `e_it` may be weakly separated, which harms nothing primary but will show in their posteriors.

## 10. Execution order and deviations log

1. Owner settles the two **[owner decision]** items: the threshold `r*`, and primary-outcome designation (production, as drafted, or revised per the design analysis).
2. Ingest: pull, manifest, audit, linking-set report, coverage report. No model fitting.
3. Freeze the analysis script; prior-predictive check; run the design analysis (§8.2) and the negative control on TD data only. Any narrowing of scope is decided here.
4. Fit the primary and secondary models on the real data; apply the decision rule; report all pre-specified quantities regardless of outcome.
5. Log any deviation below, dated, with its reason.

_Deviations: none — not yet executed._
