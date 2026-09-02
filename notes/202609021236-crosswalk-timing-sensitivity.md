# The DSE/Oxford crosswalk and the timing of the two forms

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

**2026-09-02.** The report's case for scoring every form against the fixed 810-item reference rests partly on the uk_02 dual-form crosswalk ([202607121200](202607121200-statistical-model-review.md) §3(A); `methods-models.qmd`, "Forms align more closely on raw counts than on proportions"). The study owner asked whether that comparison is valid if the DSE checklists were completed at a different time from the Oxford CDIs. This note records what the raw files say about when each form was completed, how far apart the two forms are recorded, what the crosswalk model of record already does about that, and a sensitivity analysis under alternative treatments of the gap, now reproducible through `scripts/crosswalk_dse_oxford.py --variant`. The short answer: the concern describes the data correctly — for 24 of the 34 children the two forms are recorded at different ages — and the conclusion survives it.

## What the raw files say about timing

- `data/vocab_data_uk_02.csv` is a compilation. Its `study` column carries `edg2008`, `edg2009` and `wates2010`; only the `wates2010` rows belong to the 2010 project, and all 40 Oxford CDI rows are `wates2010`. The earlier DSE rows on some of the same children come from the two EDG sub-studies.
- The Oxford CDI was the study's first-assessment instrument. The item-level workbook in `dsegroup/research-data-analysis` (`projects/vocabulary/original-data/CDIwates2010.xlsx`) records a test date for each of 41 children, all between 25 January and 26 May 2010 (6 in January, 21 in February, 13 in March, 1 in May). The SPSS file beside it (`Wates spss spreadsheet.sav`) carries those ages as `Age_in_months` with three vocabulary columns — understood only, understood and signed, understood and said — which are exactly the Oxford rows' `understood_only`, `signed_only` and `spoken`. Its second-assessment age (`Age_in_months_2`) is about 16 months later, so the DSE rows are not the second assessment.
- The DSE checklist completion dates are not recorded in either repository. The combined CSV (`edg-wates-combined.csv`) arrives with the ages already assigned, and the sibling repository's preparation script only anonymises it; nothing documents where the DSE rows' ages came from. Its Oxford ages also differ slightly from the workbook's own (they match 37 of the 40 rows as a multiset, and the workbook carries one child more than the CSV), so the ages that reached the pipeline were computed somewhere upstream of both files.

## How far apart the two forms are recorded

Pairing each child's Oxford administration with their nearest-in-age DSE row (`pair_forms` in the script):

| Gap, DSE age minus Oxford age (months) | Children |
| -------------------------------------- | -------- |
| −2                                     | 1        |
| 0                                      | 10       |
| +1                                     | 5        |
| +2                                     | 7        |
| +3                                     | 10       |
| +4                                     | 1        |

So 10 pairs are recorded at the same age (which still means only within a month of each other) and 24 are not, the DSE checklist being the later of the two in 23 of those. The raw count ratio tracks the gap, as it would if the children grew between the two forms: the median raw DSE/Oxford ratio for words understood is 1.14 for the 10 concurrent pairs, 1.14 for gaps of 1–2 months and 1.41 for gaps of 3–4 (Spearman rank correlation with the gap 0.33 over 34 pairs). A raw ratio pooled across all 34 would therefore overstate the crosswalk.

## What the model of record already does

The review's crosswalk was built for this. Each row enters at its own recorded age, a shared per-child random intercept links a child's rows, and a population age trend (`b1`, 0.86 logit per 12 months for words understood) absorbs the growth over the gap, so the form offset `delta` is estimated at matched age rather than as a raw ratio. That handles the gap correctly if the recorded ages are the ages at which each form was completed. It does not cover the case where the DSE age is later than the checklist's true completion — for instance if the DSE checklist was handed out at the Oxford test and dated when returned — and it cannot, because the completion dates are unknown. Hence the sensitivity.

## Sensitivity: `--variant`

Four treatments of the gap, all run through `scripts/crosswalk_dse_oxford.py --variant all` at the script's defaults (4 chains, 2,000 tune, 2,000 draws, seed 20260712; every fit converged, max R-hat ≤ 1.005):

- `base` — every dual-form child, every row at its recorded age; the analysis of record.
- `concurrent` — only the 10 children whose Oxford age equals one of their DSE ages, keeping all their rows (22 rows).
- `strict` — only those 10 children's concurrent DSE/Oxford row pairs (20 rows).
- `realigned` — every child, but the DSE row nearest each Oxford administration is given the Oxford age. This is the bound in which the DSE checklist reflects the child at the Oxford test date despite its later recorded age.

R = DSE count / Oxford count at the population level, median and 90% equal-tailed interval.

Words understood:

| Variant    | Children | R at 25 mo        | 31 mo             | 37 mo             | 43 mo             | 49 mo             |
| ---------- | -------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| base       | 34       | 0.85 [0.72, 0.99] | 0.96 [0.83, 1.11] | 1.10 [0.98, 1.24] | 1.25 [1.13, 1.38] | 1.40 [1.27, 1.53] |
| concurrent | 10       | 0.95 [0.64, 1.31] | 1.03 [0.74, 1.37] | 1.12 [0.85, 1.44] | 1.22 [0.95, 1.52] | 1.33 [1.03, 1.61] |
| strict     | 10       | 0.87 [0.56, 1.27] | 0.96 [0.67, 1.33] | 1.07 [0.79, 1.40] | 1.19 [0.93, 1.49] | 1.32 [1.02, 1.61] |
| realigned  | 34       | 0.91 [0.77, 1.06] | 1.03 [0.89, 1.17] | 1.17 [1.03, 1.31] | 1.32 [1.18, 1.45] | 1.47 [1.32, 1.59] |

Words spoken:

| Variant    | Children | R at 25 mo        | 31 mo             | 37 mo             | 43 mo             | 49 mo             |
| ---------- | -------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| base       | 34       | 0.86 [0.69, 1.08] | 0.89 [0.72, 1.11] | 0.94 [0.77, 1.16] | 1.03 [0.86, 1.25] | 1.17 [0.98, 1.38] |
| concurrent | 10       | 0.99 [0.65, 1.59] | 1.03 [0.69, 1.61] | 1.08 [0.73, 1.64] | 1.15 [0.79, 1.67] | 1.24 [0.87, 1.71] |
| strict     | 10       | 0.98 [0.64, 1.59] | 1.01 [0.67, 1.61] | 1.05 [0.71, 1.63] | 1.12 [0.78, 1.66] | 1.21 [0.84, 1.70] |
| realigned  | 34       | 0.98 [0.79, 1.24] | 1.01 [0.82, 1.26] | 1.07 [0.88, 1.32] | 1.16 [0.97, 1.40] | 1.31 [1.10, 1.52] |

Reading it:

- For words understood no variant moves the median ratio by more than 0.10 at any reported age (the largest shifts are +0.10 at 25 months under `concurrent` and −0.09 at 49 months under `strict`), every variant sits near 1 at the young end and rises with age as the shorter form saturates, and every variant excludes the length ratio 1.95 with probability 1.00. The report's sentence stands under any of these readings of the timing.
- `realigned` shifts R up by about 0.06 at every age, as it must: attributing the DSE count to a younger age makes the DSE form look more generous relative to Oxford. That is the direction and the size of the bias if every DSE checklist in fact reflected the Oxford test date, and it is smaller than the interval width.
- The concurrent subset is older (median Oxford age 43 months against 37 for all 34), so its young-end values are extrapolations, which is why its intervals widen there. In that subset the age-varying offset `delta1` for words understood excludes zero (+0.68 [+0.06, +1.27]) where it does not in the full set (+0.24 [−0.12, +0.59]); with 10 children that is weakly identified and is not read further here.
- Spoken is noisier throughout, as the review noted, and moves more (by up to 0.13), but its pattern is the same: R ≈ 1 at the centre and 1.95 excluded (probability ≥ 0.99 in every variant).

## What the timing does not explain

Eight of the 34 pairs record fewer words understood on the DSE checklist than on the Oxford CDI, which exact nesting of the inventories forbids, and three of those eight are concurrent pairs. That is administration noise of the kind the review already flagged — separate form-filling, recall, fatigue on the longer form — not a timing artefact, and no treatment of the gap removes it.

## Actions

- `scripts/crosswalk_dse_oxford.py` now takes `--variant {base,concurrent,strict,realigned,all}` (default `base`, so the analysis of record is unchanged), prints the gap distribution on every run, and reports the fitted age slope alongside the offset. `pair_forms` and `apply_variant` are the reusable pieces.
- A clause for `methods-models.qmd` stating that the forms were mostly not concurrent and that the estimate is insensitive to it has been drafted and is with the study owner for review; the chapter is unchanged by this note.
- Open: the DSE checklist completion dates. If they survive in DSE's own checklist records they would replace the `realigned` bound with each child's actual gap. Also open, and unchanged by this note: the provenance of the ages in `edg-wates-combined.csv`, which no document in either repository records.

## Addendum, same day: the paragraph's second check, tabulated

The same report paragraph's second reassurance — that typically-developing counts on forms of different length align more closely on raw counts than on proportions of each form's maximum — carried a `[TODO: table?]`. The claim came from [202608031500](202608031500-td-romance-extension.md), which ran the check on the _candidate_ Romance set, with Catalan and Portuguese, and found raw counts tighter at 7 of 8 ages. Rerun on the pool as admitted (English plus Italian and Spanish, the hierarchical models' scope) over the 8–15 month window the comprehension forms share, the result is cleaner: raw counts are tighter at all 8 ages (mean coefficient of variation of the form medians 0.191 on counts against 0.263 on proportions), the 309-item Spanish form records the highest median at three ages and never the lowest, and from 12 months the form recording the lowest counts is the 416-item Oxford CDI. The candidate-set rerun reproduces the August note's 7 of 8, with 8 months the exception (means 0.219 against 0.259, against the note's 0.206 and 0.244 on the August loader). The report's "at every age from 8 to 15 months" is therefore right for the pool it describes.

Median words understood, as a count and as a percentage of the form's own item count, with the coefficient of variation of the four (three, below 12 months) form medians on each scale:

| Age | English WG (396) | Oxford CDI (416) | Italian WG (408) | Spanish WG (309) | CV, counts | CV, proportions |
| --- | ---------------- | ---------------- | ---------------- | ---------------- | ---------- | --------------- |
| 8   | 16 (4.0%)        | —                | 11.5 (2.8%)      | 16.5 (5.3%)      | 0.153      | 0.253           |
| 9   | 26 (6.6%)        | —                | 29.5 (7.2%)      | 27 (8.7%)        | 0.054      | 0.121           |
| 10  | 34 (8.6%)        | —                | 22 (5.4%)        | 38 (12.3%)       | 0.217      | 0.322           |
| 11  | 57 (14.4%)       | —                | 74.5 (18.3%)     | 62.5 (20.2%)     | 0.113      | 0.137           |
| 12  | 62 (15.7%)       | 45 (10.8%)       | 94 (23.0%)       | 83.5 (27.0%)     | 0.267      | 0.329           |
| 13  | 96 (24.2%)       | 37 (8.9%)        | 107.5 (26.3%)    | 94.5 (30.6%)     | 0.328      | 0.364           |
| 14  | 99 (25.0%)       | 85.5 (20.6%)     | 145.5 (35.7%)    | 132 (42.7%)      | 0.210      | 0.281           |
| 15  | 136 (34.3%)      | 91.5 (22.0%)     | 142 (34.8%)      | 159 (51.5%)      | 0.189      | 0.294           |

Per-age administrations: English WG 245–1,037; Italian 34–69; Spanish 36–65; Oxford 7 (13 months) to 88. The caveats are the August note's: medians on modest per-age samples for the non-English forms, cross-sectional, and confounded with dataset composition — the Oxford CDI's low medians are one dataset (Floccia), which is why the comparison is a check on the scoring convention rather than a measurement of the forms.

The table is now a descriptive artefact rather than a note-only calculation: `td_form_alignment_table` and `form_alignment_spread` in `src/vocab_growth/descriptive.py`, written by `scripts/generate_descriptive_report.py` as `td_form_alignment.csv` and `td_form_alignment_spread.csv` into the report's figure cache alongside the summary tables, with the form item counts in `WORDBANK_FORM_ITEMS` (`data_utils.py`; the Oxford CDI at the project's 416 rather than the definition file's 418), and unit-tested in `tests/test_td_form_alignment.py`.

**Display layer, later the same day.** The study owner placed the cell in `methods-models.qmd` and moved its display code into `src/vocab_growth/report_tables.py`. Tables are emitted as Markdown from `output: asis` cells rather than returned as DataFrames, for three reasons found while doing it: a returned DataFrame renders the pandas index as a column of row numbers; the size can then be set per format (a styled div for HTML, a `\footnotesize` group for PDF); and the label travels in the caption. Two width regimes were needed, both checked by rendering to PDF at the report's A4 geometry and font: the alignment table (seven columns) sits at natural width with `tbl-colwidths="false"`, while the two twelve-column study-summary tables in `methods-data.qmd` overflowed the margin at natural width and wrapped the `mean (SD)` values under pandoc's proportional split, so they use `print_wide_table` — `\scriptsize`, half the cell padding, and widths proportioned to each column's longest value (`SUMMARY_TABLE_COLWIDTHS`), under which every value and dataset name keeps to one line and only the headers wrap. Their formatter (`summary_table`) moved from an inline chapter cell into the module. `tests/test_report_tables.py` pins the emitted structure, both builders' formatting and the widths; the rewired chapter renders cleanly to HTML.
