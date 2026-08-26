# Published Down syndrome CDI study samples (Laudańska et al. 2026, Table S2)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

`laudanska_2026_cdi_ds_table_s2.csv` is **literature reference data**, not study data: it plays no part in the data pipeline (`scripts/prepare_data.py` maps its inputs explicitly and does not read this directory). It records the published Down syndrome CDI literature's sample sizes and scores, extracted verbatim from a systematic review's supplementary material, for use in report and paper prose about the scale of existing studies.

## Source

Laudańska Z, van der Venne P, Preis H, Sachse S, Schaaf CP, Borjon JI, D'Souza H, Holzinger D, Haman E, Mani N, Poustka L, Zhang D, Marschik PB (2026). Communicative Development Inventories (CDIs) in etiologically diverse developmental conditions: A systematic review. _Research in Developmental Disabilities_, 170, 105256. <https://doi.org/10.1016/j.ridd.2026.105256>

Open access under **CC BY 4.0**, which permits redistribution of this extract with attribution. The CSV is Supplementary Table S2 ("Summary of publications using the CDI (expressive vocabulary) on study samples with Down syndrome"), one of four condition tables in the supplementary file `1-s2.0-S089142222600051X-mmc1.docx` (S1 autism, S3 Williams syndrome, S4 other conditions).

## Extraction and verification

Extracted 2026-08-26 directly from the supplementary `.docx` XML (`word/document.xml`, the table following the "Supplementary Table 2" caption), not transcribed from a rendering. Verified three ways: structurally (the only merged cells sit in the two banner header rows; all 24 data rows have exactly 13 unmerged cells, and the table contains no break/tab/hyphen elements a text extraction could drop); against pandoc's independent docx parser (27 rows, zero mismatched cells); and visually against the Word-rendered pages, every data row checked cell by cell.

## Structure and reading notes

The CSV is verbatim, including the source's three header rows (condition banner; section banner Paper/Participants/Scores/CDI; column names). Read with, e.g., `pd.read_csv(..., skiprows=2)`. Conventions preserved from the source:

- **Blank `Authors`/`Year` cells continue the study above** — a later timepoint or second measure on the same children (Deckers 2016 × 4, Zampini 2009 × 2, Zampini 2015 × 2), or a genuinely distinct subgroup (Bird 2005: monolingual n=14 and bilingual n=8; Dulin 2023: n=13 and n=10).
- `n.r.` = not reported; a **blank** cell is blank in the source (e.g. Bello 2014's percentile, both Zampini 2009 SDs, Foster-Cohen 2023's population).
- Four age-range cells use **European decimal commas** verbatim (e.g. `26,4-72,19`).

## Derived sample-size summary

Computed 2026-08-26 from the CSV. A _distinct participant group_ is a unique (population, sample size) pair within a study, which collapses repeated measurements of the same children and keeps genuine subgroups; per-study children sum a study's distinct groups.

| Level                                    |   k | Mean n | Median n | Range | Total |
| ---------------------------------------- | --: | -----: | -------: | ----- | ----: |
| Per row (verbatim; repeats re-measured children) |  24 |   22.2 |     23.5 | 6-40  |   534 |
| Per distinct participant group           |  19 |   22.2 |       22 | 6-40  |   421 |
| Per study (papers)                       |  17 |   24.8 |       25 | 6-40  |   421 |

No study exceeds 40 children. Two caveats: Dulin (2023)'s two subgroups (13, 10) may overlap — if the 10 are a subset of the 13, the per-study mean falls to 24.2 and the median and range are unchanged; and Foster-Cohen 2022 and 2023 (both n=35, mean ages within a month of each other) are almost certainly one cohort published twice, so unique children across the whole table are somewhat fewer than 421. These figures summarise the review's table as published; fidelity to the seventeen underlying papers is the review authors'.
