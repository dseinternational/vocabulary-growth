# Vocabulary data - United States 01 (Edgin)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Down syndrome and comparison-group CDI administrations from the Edgin cohort (University of Arizona), derived from the item-level contributor files committed to the public [`langcog/wordbank`](https://github.com/langcog/wordbank) repository — the same files that populate the Wordbank database.

Regenerate with:

```bash
python scripts/build_us01_source.py --verify
```

`data/vocab_data_us_01_manifest.json` records the source URLs, retrieval date, SHA-256 of each downloaded file, per-form row counts and the excluded empty administrations.

## Why this is not read from the Wordbank by-child export

`data/wordbank_administration_data.csv` is still the source for the typically-developing reference pool, but it cannot serve `us_01`, for two reasons.

**It is age-truncated.** Wordbank's By-Child Summary Data page calls `wordbankr::get_administration_data()` without `filter_age = FALSE`, and that default filters every administration to its instrument's registered age window — English (American) Words & Gestures 8–18 months, Words & Sentences 16–30. The filter runs *before* the page's own age slider is constructed, so no setting in the web UI can recover the rest. For the Down syndrome subset that reduced 345 administrations to 194. The out-of-window administrations are in the Wordbank database (`instruments/import_dataset.py` applies no age filter); they are simply unreachable through the download page.

**It cannot identify empty administrations.** Four source rows — three Words & Gestures, one Words & Sentences — have *every* word item blank. Wordbank scores them as zero. Two are Down syndrome rows inside the age window, and at 12 months the export holds two `(0, 0)` rows of which only one is the empty form, so the export cannot separate a form nobody filled in from a child who understands no words. They are excluded when this CSV is built and listed in the manifest.

A third consequence of moving to the item-level files, not a reason for it but the change with the largest effect on estimates: **Wordbank issues a separate `child_id` per form.** In the export the 119 apparent Down syndrome children were 53 Words & Gestures records plus 66 Words & Sentences records with *no child linked across the two forms*. They are 71 children, 46 of whom took both forms. Keying on the study's own subject identifier links them, which matters for every model carrying child random effects and for VG16's within-child cross-lag, and which makes one further cross-form longitudinal-collapse defect visible (see `CEILING_ONLY_CHILD_STUDIES` in `src/vocab_growth/data_utils.py`).

## Groups

The cohort has four developmental-status groups, from the source `DevStatus` (Words & Gestures) / `DevelopmentalDiagnosis` (Words & Sentences) column. All are carried here; `vocab_combined` selects `down_syndrome`.

| `dev_status`      | Source code | Rows |
| ----------------- | ----------- | ---- |
| `down_syndrome`   | 1           | 345  |
| `comparison`      | 0           | 321  |
| `pre_term`        | 2           | 42   |
| `autism_spectrum` | 3           | 23   |
| `unrecorded`      | (blank)     | 7    |

The `comparison` group needs care when reading anything derived from the export. Its source code is `0`, which the instrument's `values.csv` maps to an **empty** condition name, and Wordbank's importer links that empty-named condition anyway (`""` is not `None`). Those children therefore appear in the export as `typically_developing = false` with a blank `health_conditions` label — which looks like Down syndrome children whose condition was not recorded. They are a separate group: no child carries both codes. They are also excluded from the typically-developing reference pool by `TD_POOL_EXCLUDED_DATASETS`, so at present they are used nowhere. Their profile (mean 138 words understood at 14.7 months against the Down syndrome group's 69) is consistent with a typically-developing comparison group, but no codebook states this, so it remains an inference.

## Fields

<!-- spellchecker: disable -->

- `subject_id` — the study's own subject identifier (`SubID` / `ID`), shared across the two forms
- `form` — `WG` (Words & Gestures) or `WS` (Words & Sentences)
- `age` — age in months at administration (source `CDIAge`)
- `sex` — `M`, `F`, or blank
- `dev_status` — see Groups above
- `comprehension` — words understood. On WG, items coded `1` (understands) or `2` (understands and says). On WS there is no comprehension section; Wordbank reports comprehension equal to production by data convention and this file reproduces that, so the downstream `WORDBANK_BIVARIATE_FORMS` guard stays the single place the proxy is discarded
- `production` — words said. WG items coded `2`; WS items coded `1`
- `survey_vocab_max` — word-item count of the form (WG 396, WS 680)
- `in_norming_window` — whether `age` falls inside the form's registered Wordbank age window (WG 8–18, WS 16–30). Retained for provenance and for the descriptive tables; the admissibility rules derive what they need from `age` and `survey_vocab_max`, so nothing in the pipeline reads this column

<!-- spellchecker: enable -->

## Administrations outside the form's age window

The source carries every administration Wordbank holds, including the 149 outside their form's registered age window. They are **not** treated as one block, because they are not one thing.

**Above the window: admitted.** 50 Words & Gestures administrations at 19–27 months with ordinary counts (median 110 understood, 10 spoken), from 47 children all of whom are already in the pool — so they are repeat visits carrying within-child information. For a Down syndrome cohort an early-vocabulary form given to a chronologically older child is developmentally appropriate rather than an error, and the age window governs whether Wordbank's *percentile norms* apply, which this project does not use: every model scores raw counts against the 810-item reference with a per-form ceiling guard.

These rows matter more than their number suggests. `us_01` contributes 58 administrations between 19 and 27 months and every one is Words & Sentences, whose comprehension is a production proxy discarded by `WORDBANK_BIVARIATE_FORMS` — so without them the study contributes **no comprehension observations at all** in that band. Excluding them would also have been the more biased choice: a child still on Words & Gestures at 25 months is plausibly lower-ability than one who had moved to Words & Sentences.

**Below the floor: dropped** (`exclude_below_form_floor`, reinstate with `include_below_form_floor=True`). 16 administrations at 5–7 months. Three report 236, 364 and 368 words *spoken*, which no 6-month-old in any population produces, and two of the same children show comprehension collapsing from 247–371 words at 6 months to 5–19 by 11–12. The block is unreliable — most likely mis-keyed ages — and the rest of it is near-zero counts carrying almost no information.

**Ceiling-saturated: dropped as whole children** (`exclude_ceiling_only_children`, reinstate with `include_ceiling_only_children=True`). 64 children and 98 administrations whose *every* record sits at the form ceiling: 23 Words & Gestures at 39–173 months all at exactly 396 spoken, 62 Words & Sentences at 31–88 months with 61 at exactly 680, and 13 Words & Sentences at 24–30 months whose counts every other rule already masked.

The criterion is provenance, not age or count, and that is deliberate. A near-ceiling count is a defect signature only in infancy; at older ages it is ordinary, and removing the near-ceiling rule's `age <= 30` bound would mask 19 apparently legitimate records across six other studies (uk_01 at 115 months with 658 of 690 words, ie_01 at 69 with 741 of 810, es_01 at 54 with 637 of 651). What separates the batch is that the affected children have **no non-ceiling record of their own** — a fact about how the data were prepared, not about the values.

The rule is not free: 23 of the removed administrations carry a live comprehension value, all at exactly 396 between 39 and 173 months. A child recorded as understanding every word *and* saying every word at 173 months is the artefact, so they go with the rest of their record.

## Verification

`scripts/build_us01_source.py --verify` re-derives the in-window Down syndrome administrations and checks them against the by-child export as a multiset of `(age, comprehension, production)`. Words & Sentences matches exactly (109 = 109); Words & Gestures differs by precisely the two empty administrations, `(12, 0, 0)` and `(17, 0, 0)`. The reconstruction is trustworthy for the out-of-window rows because it is exact on the in-window ones.

## References

<!-- spellchecker: disable -->

1. Edgin, J. O., Tooley, U., Demara, B., Nyhuis, C., Anand, P., & Spanò, G. (2015). Sleep disturbance and expressive language development in preschool-age children with Down syndrome. _Child Development_, 86(6), 1984–1998. https://doi.org/10.1111/cdev.12443
2. Frank, M. C., Braginsky, M., Yurovsky, D., & Marchman, V. A. (2016). Wordbank: An open repository for developmental vocabulary data. _Journal of Child Language_. https://dx.doi.org/10.1017/S0305000916000209

<!-- spellchecker: enable -->

## License

Wordbank and its datasets are licensed under the [Creative Commons Attribution 4.0 International License, CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([see Wordbank FAQ](https://wordbank.stanford.edu/faq/)).
