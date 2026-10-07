# Data

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6.

Source notes describe the original checklists, data preparation and known limits. Run `uv run python scripts/prepare_data.py` to rebuild the merged CSV and DuckDB database. Model-specific exclusions are applied by `vocab_growth.data_utils`; a source row need not enter every model.

- [`ie_01`: Ireland](vocab_data_ie_01.md)
- [`ie_02`: Ireland 02](vocab_data_ie_02.md)
- [`it_01`: Italy](vocab_data_it_01.md)
- [`nz_01`: New Zealand](vocab_data_nz_01.md)
- [`es_01`: Spain](vocab_data_es_01.md)
- [Wordbank](wordbank_administration_data.md)
- [`uk_01`: UK 01](vocab_data_uk_01.md)
- [`uk_02`: UK 02](vocab_data_uk_02.md)
- [`uk_03`: UK 03](vocab_data_uk_03.md)
- [`uk_04`: UK 04](vocab_data_uk_04.md)
- [`uk_05`: UK 05](vocab_data_uk_05.md)
- [`uk_06`: UK 06](vocab_data_uk_06.md)
- [`uk_07`: UK 07](vocab_data_uk_07.md)
- [`us_01`: US 01](vocab_data_us_01.md)
- [`us_02`: US 02](vocab_data_us_02.md)
- [`us_03`: US 03](vocab_data_us_03.md)

## Measures and model restrictions

The models score raw counts against a common 810-item reference. This does not make the original lists identical, and preparation retains each form's own ceiling. A count near a short form's ceiling can omit words outside that form.

Speech and signing can overlap. Use the recorded union or a valid cross-tabulation for distinct expressive vocabulary, rather than adding their marginal totals. Some sources record sign-only words; others record total signs or symbolic gestures. Their source notes explain which field can enter a signing model.

Read the [model inventory](../docs/models/README.md) for age, language and outcome restrictions. Read the relevant constant's docstring in [data_utils.py](../src/vocab_growth/data_utils.py) before changing an exclusion.

## Age

The harmonised `age` field records complete months. Where a source supplies a fractional age, preparation rounds it down (DSE decision, 2026-09-15; `dsegroup/research-data-analysis` [#23](https://github.com/dsegroup/research-data-analysis/pull/23)). Each is derived from the finest age its source supplies: months-and-days for `es_01`, `uk_03` and `uk_04`, a day count for `uk_05`, fractional months for `us_03`, and `uk_07`'s chronological ages, which are already complete months. Whole-month ages from `ie_01`, `ie_02`, `it_01`, `nz_01`, `uk_01`, `uk_02`, `uk_06` and `us_02` are used as supplied. We cannot recover whether their providers rounded or floored them. `us_01` (built here by `scripts/build_us01_source.py`) and the Wordbank export record integer months.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). See [LICENSE](LICENSE) for details.