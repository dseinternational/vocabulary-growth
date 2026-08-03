# Wordbank By-Child Summary Data (All Languages)

Downloaded from the [Wordbank website](https://wordbank.stanford.edu/data/?name=admin_data) on 15 June 2026.

Language = All, Form = All, Health Conditions = All, Language Status = All.

This file is the source for the **typically-developing reference pool only**. The Down syndrome subset (`us_01`, Edgin) was previously derived from it and is now built from the item-level contributor files instead — see [US 01](vocab_data_us_01.md) for why, and `tests/test_prepare_data_sources.py` for the guard that keeps `us_01` on exactly one source.

## Age truncation

This export is **not** the whole database. Wordbank's By-Child Summary Data page calls `wordbankr::get_administration_data()` without `filter_age = FALSE`, which filters every administration to its instrument's registered age window before the page's age slider is applied. Administrations outside a form's window are in the database but unreachable through the download page; retrieving them needs `wordbankr` against the Wordbank MySQL connection.

For the typically-developing pool this truncation is appropriate — the forms are used within the range they were normed for — so the export is kept as-is. It mattered for the Down syndrome subset, where it cut 345 administrations to 194.

## Language scope

Queries restrict to `ENGLISH_LANGUAGES` (`English (American)`, `English (Australian)`, `English (British)`, `English (Irish)`) by default. The hierarchical typically-developing models (VG11, VG12, VG13) use `ENGLISH_AND_ROMANCE_LANGUAGES`, which adds `Italian` and `Spanish (European)` so the reference pool spans several languages on both sides of the Down-syndrome-versus-typically-developing comparison. VG03/VG04 stay English-only. Both constants are defined in `src/vocab_growth/models/definitions.py`, with the admission criteria and measurement checks on `ROMANCE_LANGUAGES`.

## Health conditions

Only the `Edgin` dataset populates `health_conditions`, and it is the only dataset with any `typically_developing = false` rows. Checked 2026-08-03 against the contributor listing (42 languages, 128 dataset entries) and all 147 `raw_data/*/*_fields.csv` field-mapping files in `langcog/wordbank`: **no non-English dataset codes a health condition at all**, and there is no Down syndrome data in Wordbank in any language other than English (American). Note what that does and does not establish — a contributor who supplied no condition column would leave a clinical cohort indistinguishable from a typical one, so the negative result rests on the contributor listing, not on the condition field being empty.

`typically_developing` is not a database field. It is computed per child in the Shiny app as `is_null(health_conditions)`, so `false` requires a linked condition row — which is why Edgin's comparison group, whose condition code maps to an empty name, arrives flagged `false` with a blank label. See [US 01](vocab_data_us_01.md).

## Fields

<!-- spellchecker: disable -->

- downloaded
- language
- form
- dataset_name
- child_id
- age
- comprehension
- production
- is_norming
- birth_order
- caregiver_education
- ethnicity
- race
- sex
- birth_weight
- born_early_or_late
- gestational_age
- zygosity
- language_exposures
- health_conditions
- monolingual
- typically_developing

<!-- spellchecker: enable -->

## License

Wordbank[[1](#ref-frank2016)] and its datasets are licensed under the [Creative Commons Attribution 4.0 International License, CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([see Wordbank FAQ](https://wordbank.stanford.edu/faq/)).

## References

<!-- spellchecker: disable -->

1. <a id="ref-frank2016"></a>Frank, M. C., Braginsky, M., Yurovsky, D., & Marchman, V. A. (2016). Wordbank: An open repository for developmental vocabulary data. *Journal of Child Language*. https://dx.doi.org/10.1017/S0305000916000209