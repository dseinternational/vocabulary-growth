# Vocabulary data - UK (1)

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
- understood
- understood_imputed
- produced

<!-- spellchecker: enable -->

## Measurement and column semantics

> [!NOTE]
> This section was drafted by an LLM-based AI tool (Claude Code/Opus 4.8), from a
> review of the uk_01 study write-up and the data (2026-07-13). WS word count corrected
> 689 → 690 on 2026-08-31 (Claude Code/Fable 5): 689 had no traceable source, while 690
> is the ceiling recorded throughout the pipeline (`survey_vocab_max`; hard-coded in
> `research-data-analysis`'s `prepare/uk_01_edg.py` and stated in its `uk_01_edg.md`).

uk_01 is a Down syndrome study (~218 children, ages ~1–9 y, Sarah Duffen Centre,
Portsmouth) using the MacArthur-Bates CDI — Words & Gestures (396-word checklist) and
Words & Sentences (690 words per the recorded `survey_vocab_max`, including the 396);
the two were combined for the vocabulary analysis.

⚠️ The WS ceiling of 690 has not been verified against the original UK-adaptation form,
and the source data hints it may be an overstatement: in 18 of 22 WS categories the
largest per-category count attained by two or more children equals the standard American
680-item WS category size exactly, three categories stay below it, and only question
words repeatedly exceeds it (9 vs the American 7) — pointing to a form total near
680–682 rather than 690. Since `survey_vocab_max` is the Beta-Binomial denominator for
WS rows, the recorded 690 should be confirmed against the original instrument if it can
be consulted; changing it would invalidate fitted output for every model consuming uk_01
WS observations.

The per-item columns carry a `c` / `v` / `s` suffix per semantic category:
**c = comprehension** (understands), **v = vocalised** (says), **s = signed**. Signing
was recorded as a **per-word add-on question** ("indicate if the child _signs_ the
word"), and — per the write-up — was added to **only some** questionnaires.

Summary columns (verified against the category counts):

| Column       | Meaning in uk_01                                                        |
| ------------ | ----------------------------------------------------------------------- |
| `understood` | words understood (comprehension)                                        |
| `spoken`     | **vocalised** words (words the child says) = sum of the `v` categories  |
| `signed`     | **signed-only** words (signed but not vocalised) — see note below       |
| `produced`   | **total expressive union** = `spoken + signed` (each word once)         |

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

### ⚠️ `signed` is defined differently here than in uk_02 / nz_01

uk_01's `signed` = **signed-only** (excludes words also spoken). In `uk_02` and `nz_01`,
`signed` = **total signed**, including words also spoken (verified: uk_02
`signed == signed_only + signed_spoken`; nz_01 `signed == signs-only + both`). This is
immaterial for `produced` (all three yield the correct union) but **does** bias the
signing models VG14/VG15, whose signed ratio `r(a) = P(sign | understood)` treats
`signed` as total sign use — uk_01's `r` is understated relative to uk_02/nz_01.
Harmonising `signed` across studies (or deriving uk_01's total-signed from the original
word-level forms) is needed before cross-study signed-ratio comparisons. See
`notes/202607121753-reporting-config-fit-run-and-findings.md`.

## Withheld subjects (probable homonym fusion)

> [!NOTE]
> This section was drafted by an LLM-based AI tool (Claude Code/Fable 5), from the 2026-08-31 subject-id audit.

uk_01 has no per-child identifier in its source: the child's name is the longitudinal linker, so two different children sharing a name are silently fused under one `subject_id` (the homonym caveat documented in `research-data-analysis`'s `prepare/uk_01_edg.md`). One id shows the fused pattern in the committed data: `ID_E33ADE657109EBB8` (F) interleaves a signer who barely speaks (66 mo: spoken 8 / signed 225; 78 mo: 27 / 126) with a speaker who never signs (76 mo: 451 / 0; 88 mo: 483 / 0) — read as one child, a 424-word production collapse followed by a 456-word surge. The rows remain in this CSV; `scripts/prepare_data.py` drops them at load (see `UK01_WITHHELD_SUBJECTS` in `src/vocab_growth/data_utils.py`), pending adjudication against the original study records. See `notes/202608311600-uk01-homonym-fusion.md`.

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.