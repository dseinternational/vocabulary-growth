# Data

- [Ireland](vocab_data_ie_01.md)
- [Ireland 02](vocab_data_ie_02.md)
- [Italy](vocab_data_it_01.md)
- [New Zealand](vocab_data_nz_01.md)
- [Spain](vocab_data_es_01.md)
- [Wordbank](wordbank_administration_data.md)
- [UK 01](vocab_data_uk_01.md)
- [UK 02](vocab_data_uk_02.md)
- [UK 03](vocab_data_uk_03.md)
- [UK 04](vocab_data_uk_04.md)
- [UK 05](vocab_data_uk_05.md)
- [UK 06](vocab_data_uk_06.md)
- [UK 07](vocab_data_uk_07.md)
- [US 01](vocab_data_us_01.md)
- [US 02](vocab_data_us_02.md)

## Age

Every source's `age` is **complete months of age**: a fractional age is floored, never rounded, so a child is counted a month older only on reaching that month (DSE decision, 2026-09-15; `dsegroup/research-data-analysis` [#23](https://github.com/dsegroup/research-data-analysis/pull/23)). Each is derived from the finest age its source supplies: months-and-days for `es_01`, `uk_03` and `uk_04`, a day count for `uk_05`, fractional months for `us_03`, and `uk_07`'s chronological ages, which are already complete months. The sources that record whole months only — `ie_01`, `ie_02`, `it_01`, `nz_01`, `uk_01`, `uk_02`, `uk_06` and `us_02` — are taken as given, because whether their providers rounded or floored cannot be recovered. `us_01` (built here by `scripts/build_us01_source.py`) and the Wordbank export record integer months.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.