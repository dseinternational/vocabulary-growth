# Vocabulary data - Ireland 2

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

This Irish source records repeated assessments on DSE Checklists 1 and 2. The pooled view retains rows recorded as English-speaking and supplies understood, spoken and signed counts. It sets `produced` equal to spoken because the modality overlap is not supplied. The source ceiling is 476 achievable items; the model reference remains 810.

## Columns

- `sex` — **1 = male, 2 = female**, the pipeline's canonical coding. This source carries 1/2 with no value label saying which is which, so it was held outside the standard as `sex_source_code` until the contributor confirmed the mapping on 2026-09-04; the values never changed, only the column name. `vocab_growth.data_utils` decodes it to `M`/`F` in the `vocab_combined` view. Sex is recorded for every row and is consistent within each child.
- `survey_vocab_max` — **476**, the DSE Checklists 1 + 2 ceiling (127 + 349 achievable words). Checklist 3 was not administered. Added upstream on 2026-09-15 so the ceiling travels with the data, and read by the `vocab_combined` view from the same day, which until then assigned `ie_02` the 810 of the full three-checklist instrument.

## A short form, not a partial administration

The study owner decided on 2026-09-15 to keep `ie_02`'s counts in the pool on the 810-item reference scale, as the nested Oxford CDI and MB-CDI forms are, rather than to mask them as a partial administration, as `ie_01`'s Checklists 1 + 2 baseline is (`data_utils.INCOMPLETE_ADMINISTRATION_CEILINGS`). The evidence the decision rested on, from `ie_01`'s follow-up wave, the only wave in the pool with all three checklists recorded: Checklist 3 adds little below about 300 words on Checklists 1 + 2, but a median of 100 words at 300–400 and 233 at 400–476, where it is 22–35% of the full count. 17.5% of `ie_02`'s comprehension counts are at or above 300, and its spoken counts are almost all small. The rule is `data_utils.DSE_SHORT_FORM_CEILINGS`.

Three consequences follow.

- **One more administration leaves the pool.** Three totals exceed 476 (`understood` 477, from a Checklist 2 count of 350 against 349 achievable words): both rows of the already-withheld `ID_79C464EF367C4D5B`, and `ID_FCFE8CE511D0687B` at t1, which the form-ceiling guard now drops. Every Down syndrome prepared frame goes from 1,708 rows to 1,707.
- **`ie_02` is no longer DSE-native.** `dse-native-only` keeps `ie_01`'s 810 wave, `uk_02`'s DSE form and `uk_06`: 153 fitted rows from 116 children, where it kept 264 from 181. On VG15 that leaves 50 signed observations, from `uk_02` and `uk_06`.
- **The understatement has a registered check.** `ie02-comprehension-masked` on VG10, VG15 and VG20 masks `ie_02`'s 110 comprehension counts and keeps its spoken and signed counts, through the `mask_dse_short_form_comprehension` definition field.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.