# Vocabulary data - UK (1)

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

This data was collected during the 1990s through to 2000.

## Fields

<!-- spellchecker: disable -->

- subject_id
- survey
- age
- sex
- noun1c
- noun1v
- noun1s
- noun2c
- noun2v
- noun2s
- noun3c
- noun3v
- noun3s
- noun4c
- noun4v
- noun4s
- noun5c
- noun5v
- noun5s
- noun6c
- noun6v
- noun6s
- noun7c
- noun7v
- noun7s
- noun8c
- noun8v
- noun8s
- noun9c
- noun9v
- noun9s
- noun10c
- noun10v
- noun10s
- tnoun10
- noun11v
- noun11s
- tnoun11
- noun12c
- noun12v
- noun12s
- tnoun12
- noun13c
- noun13v
- noun13s
- tnoun13
- verb14c
- verbs14v
- verbs14s
- tverb14
- adjec15c
- adject15
- adjec15s
- tadjec15
- noun16c
- noun16v
- noun16s
- tnoun16
- pron17c
- pron17v
- pron17s
- tpron17
- quest18c
- quest18v
- quest18s
- tquest18
- prep19c
- prep19v
- prep19s
- tprep19
- quant20c
- quant20v
- quant20s
- tquant20
- verb21v
- verb21s
- tverb21
- conn22v
- conn22s
- tconn22
- inounv
- inouns
- tinoun
- iverbv
- iverbs
- tiverb
- onoun
- overb
- complex
- sent
- survey_vocab_max
- spoken
- signed
- understood_only
- understood
- understood_imputed
- produced

<!-- spellchecker: enable -->

## Measurement and column semantics

> [!NOTE]
> This section was drafted by an LLM-based AI tool (Claude Code/Opus 4.8), from a
> review of the uk_01 study write-up and the data (2026-07-13). The WS checklist total
> was corrected to 680 on 2026-08-31 (Claude Code/Fable 5) after verification against
> the original study report and the source data — see "WS checklist total" below.

uk_01 is a Down syndrome study (~218 children, ages ~1–9 y, Sarah Duffen Centre,
Portsmouth) using the MacArthur-Bates CDI — Words & Gestures (396-word checklist) and
Words & Sentences (680 words, including the 396); the two were combined for the vocabulary
analysis.

### WS checklist total: 680 (verified 2026-08-31)

The `survey_vocab_max` recorded for WS rows was 690 for most of this dataset's history, and briefly 689; neither survives verification. The original project report (Sarah Duffen Centre, August 2000, `project_2.doc` in the DSE research archive) describes administering and scoring the standard MacArthur CDI per the Fenson et al. (1993) manual and comparing against the Fenson normative sample, with no mention of any restructured UK form; its instrument paragraph states "a checklist of 689 words" — the source of the 689 that stood here until 2026-08-31 — but that same paragraph also miscounts the WS categories (19 + 2 where the actual form has 22), and no published CDI version has 689 items. The published CDI:WS vocabulary checklist is 680 words across 22 semantic categories. The source data agrees: over the 154 WS administrations, the maximum count attained equals the standard American category size exactly in every category where the ceiling is reached (five children at exactly 103 action words, 14 at exactly 7 question words, 43 at exactly 12 sound effects, …), the remaining categories stay below their sizes, and the handful of single-row overshoots are isolated entry errors — two of them sit in rows that are corrupt on other fields as well (e.g. pronouns recorded as 55 of 25). The recorded 690 had no traceable source at all. No observed count exceeds 680 (WS maximum spoken/produced is 669), so the correction drops or masks nothing; it changes the recorded form ceiling only.

The per-item columns carry a `c` / `v` / `s` suffix per semantic category:
**c = understood only** (understands, but neither says nor signs), **v = vocalised** (understands and says), **s = signed**. The three are **alternative responses for each word**, not independent ticks, so a category's words understood are `c + v + s`. Signing
was recorded as a **per-word add-on question** ("indicate if the child _signs_ the
word"), and — per the write-up — was added to **only some** questionnaires.

Summary columns (verified against the category counts):

| Column            | Meaning in uk_01                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------- |
| `understood_only` | words understood but neither said nor signed = sum of the 19 `c` categories              |
| `understood`      | **total words understood** = `understood_only + produced` on Words and Gestures rows; empty on Words and Sentences rows |
| `spoken`          | **vocalised** words (words the child says) = sum of the `v` categories                   |
| `signed`          | **signed-only** words (signed but not vocalised) — see note below                        |
| `produced`        | **total expressive union** = `spoken + signed` (each word once)                          |

### `understood` counts every word understood, including those said or signed (corrected 2026-09-14)

> [!NOTE]
> This section was drafted by an LLM-based AI tool (Claude Code/Opus 5), from the source file and the original study report ([#320](https://github.com/dseinternational/vocabulary-growth/issues/320)).

Until 2026-09-14 `understood` was the sum of the `c` columns alone — words understood but **not** produced — so `spoken / understood` could exceed 1 (it did for 2 of the 29 rows then carrying a value, to 1.95), and those two were masked by `mask_comprehension_below_production`. The exclusive reading rests on three checks, detailed in `prepare/uk_01_edg.md` in `research-data-analysis`, where the CSV is built and where the correction was made:

- **The form.** The original project report describes the Words and Gestures form as asking whether the child "understands" each word or "understands and says" it.
- **The source's own totals.** `Combined groups.sav` holds both `WORDSUND` ("Words understood") and `UNDERST` ("Total words understood"), and `UNDERST == WORDSUND + WORDS` (total words produced) on all 224 rows.
- **The category ceilings.** On the 70 Words and Gestures rows `c + v + s` reaches the published category size exactly in 16 of 19 categories, while no row in any category has `c` at the category size alongside a single word said or signed.

The same correction retired an upstream rule that set a category's `c` to missing wherever it was zero while `v` or `s` was not — under this coding, simply a child who produces every word they understand in that category. It had emptied the comprehension of **41 of the 70** Words and Gestures rows, so uk_01 now contributes 69 comprehension counts to the pool (one row belongs to a withheld subject, below) where it contributed 27.

Comprehension was recorded on the **Words and Gestures form only**: every `c` is zero on all 149 Words and Sentences rows, so `understood` is empty there and `survey_vocab_max` for every comprehension count is 396.

### `produced` is a de-duplicated union — NOT a double-count

The study reports total production as **"vocalised and signed-only words"** (Table 9,
"Total Population (Vocalised)" vs **"Signers (Vocalised + Signed)"**) — i.e. spoken words
plus words signed-but-not-spoken, each word counted once. In the data `produced` (a
source column) equals `spoken + signed` for every row, which matches that union **iff
`signed` is the signed-only count** — so uk_01's `signed` is read as signed-only. A word
both said and signed is counted once (in `spoken`), so `produced` does **not**
double-count. Caveat: word-level say/sign overlap is not in this aggregated file (only
category counts), so the de-duplication is taken from the study's definition, not
re-derived here.

### `signed` is defined differently here than in uk_02 / nz_01

uk_01's `signed` = **signed-only** (excludes words also spoken). In the harmonised `uk_02` and `nz_01` data,
`signed` = **total signed**, including words also spoken (verified: uk_02
`signed == signed_only + signed_spoken`; nz_01 `signed == signs-only + both`). This is
immaterial for `produced` (all three yield the correct union) but **does** bias the
signing models VG14/VG15, whose signed ratio `r(a) = P(sign | understood)` treats
`signed` as total sign use — uk_01's `r` is understated relative to uk_02/nz_01.
The primary signing analyses mask uk_01's signed-only counts through `SIGNED_ONLY_STUDIES`. Reinstating them as total signed counts would require the original word-level overlap data. See
`notes/202607121753-reporting-config-fit-run-and-findings.md`.

## Withheld subjects (probable homonym fusion)

> [!NOTE]
> This section was drafted by an LLM-based AI tool (Claude Code/Fable 5), from the 2026-08-31 subject-id audit.

uk_01 has no per-child identifier in its source: the child's name is the longitudinal linker, so two different children sharing a name are silently fused under one `subject_id` (the homonym caveat documented in `research-data-analysis`'s `prepare/uk_01_edg.md`). One id shows the fused pattern in the committed data: `ID_E33ADE657109EBB8` (F) interleaves a signer who barely speaks (66 mo: spoken 8 / signed 225; 78 mo: 27 / 126) with a speaker who never signs (76 mo: 451 / 0; 88 mo: 483 / 0) — read as one child, a 424-word production collapse followed by a 456-word surge. The rows remain in this CSV; `scripts/prepare_data.py` drops them at load (see `UK01_WITHHELD_SUBJECTS` in `src/vocab_growth/data_utils.py`), pending adjudication against the original study records. See `notes/202608311600-uk01-homonym-fusion.md`.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.