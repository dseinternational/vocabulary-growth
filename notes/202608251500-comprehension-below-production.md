# Comprehension below production: a sixth defect class, and the denominator that nearly got it wrong

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Study owner's ruling, 2026-08-25. Closes the pipeline half of [#190](https://github.com/dseinternational/vocabulary-growth/issues/190) item C's `uk_01` question and generalises it: an inclusive comprehension field cannot be exceeded by production, so administrations where it is are masked rather than retained and flagged.

## The rule

`understood < produced` masks `understood`, on ten administrations across three studies:

| study   | rows | worst case                        |
| ------- | ---: | --------------------------------- |
| `ie_01` |    7 | 13 understood against 366 spoken  |
| `uk_01` |    2 | 142 understood against 164 spoken |
| `it_01` |    1 | 58 understood against 373 spoken  |

Two of the `ie_01` rows record **0** understood against 83 spoken. The comprehension observation count for the Down syndrome pool goes from 987 to 977.

Implemented as a masked defect class in `data_utils.load_combined_data` with `include_comprehension_below_production=True` to reinstate — the same shape as the other five, so it is visible to the sensitivity machinery rather than disappearing into the prepared data.

## Three choices inside the rule, each of which could have gone wrong

**The denominator is `produced`, not `spoken + signed`.** This is the one that matters and it was nearly missed. `produced` is a _union_ — distinct words the child produces — while the two modality columns overlap wherever a child both says and signs the same word. In the signing studies the gap is large: `uk_07` has `produced < spoken + signed` on 77 of 82 rows, `nz_01` on 101 of 111. Reconstructing production as the sum flags **87** administrations instead of 10, almost all of them bimodal children penalised for being counted twice. A first pass at this rule did exactly that. The function now requires a `produced` column and raises without one rather than substituting the sum.

**Only the comprehension count is masked.** The production figure is corroborated by two columns that agree, and in both diagnosed studies the fault is localised to comprehension: `uk_01`'s `understood` appears to _exclude_ words the child also produces, which is why `spoken / understood` reaches 1.95 there (#190 item C), and `ie_01`'s seven rows sit in the wave whose Checklist 1 comprehension is already documented as unreliable — pooled comprehension _falls_ between waves while the mean understood total rises. Masking the row wholesale would discard production counts that are not in question.

**Equality is kept.** `understood == produced` is a child who produces everything they understand, which is legitimate. Forty-five administrations meet it, of which 18 are `0 == 0` and most of the rest sit at exactly 396 — the Words & Gestures ceiling, where both counts are censored rather than equal. Excluding them would have removed genuine young-age observations from the band where the Down syndrome anchors are calibrated, and would have taken the ceiling question out of the rule that already governs it.

## Relationship to the `uk_07` withheld row

`UK07_WITHHELD_ADMINISTRATIONS` already handled one row with this signature — 191 understood against 489 produced — by dropping it at CSV load, so it never reaches `vocab_combined` and this rule never sees it. The two mechanisms stay separate deliberately: the ten here are a stable property of closed sources, while the `uk_07` row is an open question with a reachable source team. That docstring used to draw its contrast against `ie_01`'s seven "retained-and-flagged" records; it now records that those are masked.

## What this makes stale, and what will not tell you

**Nothing is marked stale automatically, and that is the trap.** The rule lives in the loader, not in the data: `data/*.csv` and the DuckDB build are untouched, so `source_data_hash` does not move. `validate_fit_output` compares only that hash — `rows`, `source_row_counts` and `observed_outcome_counts` are recorded in the manifest but never checked, and the code block requires only that the fit came from a clean tree, not a current commit. So every affected fit will keep passing `check_fit.py --purpose publish` while having been fitted on a frame with ten more comprehension observations than the current one.

The refits therefore have to be forced by hand. Affected: every Down syndrome model that observes comprehension — **VG02**, VG05, VG07-VG10, VG14, VG15, VG16, VG19, VG20, VG22. VG01 observes only production and is unaffected; the typically-developing models (VG03, VG04, VG11, VG12, VG13, VG21) draw on the Wordbank pool and never see these rows.

Of those, **VG02 is the one that matters operationally**: it is the only affected model `check_fit.py` currently reports as valid, so it is the one that would be silently missed. Every other affected model is already in the refit set for other reasons.

Closing the gap properly — validating the recorded frame counts against a freshly prepared frame, so a preparation or loader change invalidates fits the way a definition change does — was considered and deferred by the study owner for this run.
