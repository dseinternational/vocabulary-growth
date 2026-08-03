# Widening the typically-developing pool to Italian and Spanish (European)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Date: 2026-08-03.

## Summary

The Down syndrome pool is a quarter non-English — `es_01` (Spanish, 186 children) and `it_01` (Italian, 54 children) are 25.5% of its observations and 29.5% of its children — while the typically-developing reference it is compared against was drawn from English alone. Admitting Italian and Spanish (European) to the hierarchical typically-developing models (VG11, VG12, VG13) removes that asymmetry.

**The gain is defensibility, not power.** The typically-developing trajectory is already precisely estimated from 4,899–12,488 children; the Down syndrome-versus-typically-developing contrast is limited by the 764 Down syndrome children, and adding reference children barely moves it. What the widening buys is an answer to a question that currently has none: _is the reported gap an artefact of comparing Spanish and Italian Down syndrome children against an English-only reference?_

## Which languages, and why not more

Wordbank has five European Romance languages. Two are admitted.

| Language                        | Form items (WG / WS) | Norming sample | Comprehension validity | Admitted     |
| ------------------------------- | -------------------- | -------------- | ---------------------- | ------------ |
| Italian                         | 408 / 670            | yes            | 0.000                  | **yes**      |
| Spanish (European)              | 309 / 594            | yes            | 0.000                  | **yes**      |
| Catalan                         | 423 / 678            | no             | 0.000                  | no           |
| Portuguese (European)           | 317 / 639            | no             | 0.000                  | no           |
| French (French)                 | **713** / 690        | no             | **0.209**              | no           |
| _English (American), reference_ | 396 / 680            | partly         | 0.001                  | (already in) |
| _English (British) Oxford CDI_  | 418                  | no             | 0.005                  | (already in) |

Item counts were counted from each instrument's definition file in `langcog/wordbank` (`type == "word"` rows); the method reproduces English (American) 396 and 680 exactly. Comprehension validity is the fraction of rows with `comprehension` exactly equal to `production`, restricted to rows with comprehension ≥ 20 and excluding all-zero rows, so it is not a low-count coincidence — this is the VG06 proxy-defect signature.

**Italian** is the only pairing that is same-language _and_ same-instrument on both sides: `it_01`'s two form ceilings, 408 and 670, are exactly Wordbank's Italian Words & Gestures and Words & Sentences item counts. Its comprehension reaches 24 months, and its Words & Gestures contributor is Caselli — the same group behind the Italian Down syndrome CDI literature.

**Spanish (European)** matches `es_01`'s language but not its instrument: `es_01` uses the 651-item CDI-Down, where Wordbank's Spanish forms are 309 and 594. Worth recording separately: `es_01` also carries the study's own **186 mental-age and sex matched typically-developing children on the same CDI-Down**, fully paired via `pair_id`, with understood, spoken _and_ gestured counts. For a matched Spanish analysis those remain the better comparison — and they are the only typically-developing gesture data anywhere in the project, where the signing models have no typically-developing counterpart at all. They are still excluded from `vocab_combined` and used nowhere.

**French (French) is excluded on measurement grounds.** Its Words & Gestures form carries 713 word items where every other Words & Gestures adaptation in Wordbank has 309–457, so it is a Words & Sentences-sized inventory administered at 8–16 months; and 20.9% of its substantial-count rows record comprehension exactly equal to production. Either alone would disqualify it.

**Catalan and Portuguese (European)** are clean but are neither norming samples nor matched to any Down syndrome study, and Portuguese would have contributed 45% of the added observations on its own — the largest single block of the widened pool coming from the language with the weakest claim to be in it.

## Two measurement checks

### Ceiling exposure is no worse than the existing pool

No row in any candidate form exceeds its own item count. Fraction of rows within 90% of their form's ceiling (typically-developing, age ≤ 30):

| Form                         | Comprehension | Production |
| ---------------------------- | ------------- | ---------- |
| English (British) Oxford CDI | **8.0%**      | 2.6%       |
| English (American) WS        | 6.5%          | 6.5%       |
| Italian WS                   | 5.0%          | 5.0%       |
| Spanish (European) WS        | 4.9%          | 4.9%       |
| English (American) WG        | 3.1%          | 0.2%       |
| Spanish (European) WG        | 2.9%          | 0.0%       |
| Italian WG                   | 1.5%          | 0.8%       |

The admitted Words & Gestures forms are _less_ ceiling-exposed than English Words & Gestures, because the shorter Spanish form is administered over a narrower, younger window (8–15).

### The fixed 810-item denominator survives

Every model scores counts against `n_trials = 810` with no per-row denominator, so the scheme rests on shorter checklists being _nested_ — omitting rare, late-acquired words — rather than proportionally sampled. If they were proportionally sampled, a child on the 309-item Spanish form would produce a raw count ~22% below an equivalent child on the 396-item English form.

Tested across the 8–15 month window these forms share, comparing how well language medians align on raw counts against percent-of-own-form:

| Age      | CV, raw counts | CV, percent of own form | Tighter |
| -------- | -------------- | ----------------------- | ------- |
| 8        | 0.251          | 0.242                   | pct     |
| 9        | 0.249          | 0.272                   | raw     |
| 10       | 0.194          | 0.303                   | raw     |
| 11       | 0.267          | 0.284                   | raw     |
| 12       | 0.194          | 0.248                   | raw     |
| 13       | 0.163          | 0.188                   | raw     |
| 14       | 0.187          | 0.220                   | raw     |
| 15       | 0.146          | 0.194                   | raw     |
| **mean** | **0.206**      | **0.244**               | **raw** |

Raw counts align better, at 7 of 8 ages. And there is no monotone relation between item count and raw count — Spanish (309 items) produces the _highest_ medians and Portuguese (317) the lowest, with Catalan (423) mid-pack — so instrument length is not the dominant driver. This is the first test of the 810-denominator assumption _across_ languages rather than within English, and it passes. Note that `it_01`'s Italian Down syndrome counts were already pooled onto that scale, so the widening applies an existing assumption symmetrically rather than making a new one.

Caveat: medians on modest per-age samples, cross-sectional, confounded by dataset composition and recruitment. The margin (0.206 against 0.244) is not large.

## What it costs

**Between-language heterogeneity of roughly ±20% at matched age.** At 15 months the candidate-language medians run 108 (Portuguese) to 159 (Spanish), a 1.47× range. That variation has to go somewhere: into a study random intercept, or into the dispersion.

**Language is very nearly collinear with dataset.** Each admitted language contributes one dataset per form — Italian WG = Caselli, Italian WS = CLEX, Spanish = Karousou — so a language effect cannot be separated from a sample effect and will be estimated as between-_dataset_ heterogeneity. Report it as such. One trap for later: the `CLEX` label spans several languages in Wordbank (Croatian, Danish, Russian, Swedish, Turkish as well as Italian). Only its Italian rows enter the pool today, so the study label is unambiguous — admitting a further CLEX language would silently pool two languages under one study intercept.

**VG03 and VG04 stay English-only.** They carry no random effects, so the between-language spread would be absorbed by the Beta-Binomial dispersion and reported as child-level dispersion. This does mean the VG03→VG11 and VG04→VG12 steps now differ in language scope as well as hierarchy; the language scope is on the definition (`td_languages`) so the difference is explicit rather than implicit.

**Multilingual comprehension coverage fades above 18 months.** Spanish (European) Words & Gestures is registered 8–15, Italian Words & Gestures 7–24. VG12 reports understood across 9–30 months, so its upper half remains as English-dominated as before. The widening is best matched to VG13 (8–18), which is the joint model carrying the Down syndrome-versus-typically-developing comparison.

**The headline reference changes meaning.** It becomes a four-language European average rather than an English one. Better for generalisability, worse for a UK-family-facing number, and it needs saying wherever the reference is quoted.

## The pool now has an explicit lower age bound

Italian Words & Gestures is registered from **7** months, where every English CDI form
in Wordbank starts at 8. Five Italian administrations at 7 months therefore fell below
the floor of the typically-developing GP domain (`_TD_GP_DOMAIN_MONTHS = (8, 30)`), and
`build_utils` refused to build — correctly.

Widening the GP domain would have been the wrong fix: the constant is shared with
VG03/VG04, so it would have made those models stale for the sake of five observations
at the least informative end of the range. Instead the pool's age window is now stated
explicitly as `TD_POOL_AGE_MONTHS = (8, 30)` and applied in the loader. The upper bound
was already there implicitly (the loader defaulted to 30); the lower bound had been
implicit in the English forms' 8-month floor, and widening the language scope is what
made it matter. `max_age_months` still overrides the upper bound per model (VG13 uses
18); there is deliberately no per-model lower override, since a model wanting one would
be asking to sit outside its own GP domain.

## Resulting pools

Loader output, default flags, after each model's `min_study_observations`, English-only
against widened on the same code path:

| Model                | English only | Widened             | Languages    | Datasets  |
| -------------------- | ------------ | ------------------- | ------------ | --------- |
| VG11 (spoken)        | 16,235       | **18,522** (+2,287) | 3 → 5        | 7 → 10    |
| VG12 (understood)    | 5,997        | **7,052** (+1,055)  | 2 → 4        | 4 → 6     |
| VG13 (joint, ≤18 mo) | 5,406        | **6,358** (+952)    | 2 → 4        | 4 → 6     |
| VG03, VG04           | unchanged    | unchanged           | English only | unchanged |

Age ranges are unchanged (VG11 8–30, VG12 8–25, VG13 8–18) — the widening adds density,
not range. Romance share of the widened pools: about 12% of VG11, 15% of VG12 and 15% of
VG13 by observation, against 25.5% of the Down syndrome pool. Closer to symmetric than
0%, and deliberately short of overshooting.

`td_languages` is part of the model graph, so **VG11, VG12 and VG13 need reporting-quality
refits.** `TD_POOL_AGE_MONTHS` changes no English-only pool, so VG03/VG04 are unaffected —
verified by running both scopes through the loader on the same code path.

## What was not done

- Catalan and Portuguese (European) are not admitted; the criteria are on `ROMANCE_LANGUAGES` if that is revisited.
- No language-level random effect was added. With one dataset per language it would not be identified separately from the study intercept.
- The cross-language comparison the widening enables — Italian typically-developing against English typically-developing on the common scale, which is the actual validation of the pooling `it_01` already relies on — is not yet run. It needs the refits first.
- `es_01`'s 186 matched typically-developing children remain unused.
- No lower age bound was added per model; `TD_POOL_AGE_MONTHS` applies pool-wide.
