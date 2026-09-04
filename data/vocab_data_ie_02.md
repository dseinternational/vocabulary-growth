# Vocabulary data - Ireland 2

[Description to follow]

## Columns

- `sex` — **1 = male, 2 = female**, the pipeline's canonical coding. This source carries 1/2 with no value label saying which is which, so it was held outside the standard as `sex_source_code` until the contributor confirmed the mapping on 2026-09-04; the values never changed, only the column name. `vocab_growth.data_utils` decodes it to `M`/`F` in the `vocab_combined` view. Sex is recorded for every row and is consistent within each child.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.