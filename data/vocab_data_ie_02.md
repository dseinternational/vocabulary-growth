# Vocabulary data - Ireland 2

[Description to follow]

## Columns

- `sex` — **1 = male, 2 = female**, the pipeline's canonical coding. This source carries 1/2 with no value label saying which is which, so it was held outside the standard as `sex_source_code` until the contributor confirmed the mapping on 2026-09-04; the values never changed, only the column name. `vocab_growth.data_utils` decodes it to `M`/`F` in the `vocab_combined` view. Sex is recorded for every row and is consistent within each child.
- `survey_vocab_max` — **476**, the DSE Checklists 1 + 2 ceiling (127 + 349 achievable words). Checklist 3 was not administered. Added upstream on 2026-09-15 so the ceiling travels with the data. **The `vocab_combined` view does not read it yet**: it still assigns `ie_02` the 810 of the full three-checklist instrument, so adding the column changed no prepared frame. Whether `ie_02` should instead be treated as a partial administration, as `ie_01`'s Checklists 1 + 2 baseline is (`data_utils.INCOMPLETE_ADMINISTRATION_CEILINGS`), is an open decision. Three totals exceed 476 (`understood` 477, from a Checklist 2 count of 350 against 349 achievable words): both rows of the already-withheld `ID_79C464EF367C4D5B` and `ID_FCFE8CE511D0687B` at t1.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.