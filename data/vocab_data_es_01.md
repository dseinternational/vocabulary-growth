# Vocabulary data - Spain

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5), from the source document and the data (2026-08-03).

A Spanish sample of **186 children with Down syndrome** and **186 typically developing children**, matched pair-wise on mental age and sex, provided directly by the author (Miguel Galeote). Cross-sectional: one observation per child, 372 rows in total.

The sample matches that reported in Galeote, M., Sebastián, E., Checa, E., Rey, R., & Soto, P. (2011). The development of vocabulary in Spanish children with Down syndrome: comprehension, production, and gestures. _Journal of Intellectual & Developmental Disability_, 36(3), 184–196. <https://doi.org/10.3109/13668250.2011.599317> — "186 children with DS and 186 children with TD, with a mental age (MA) of 8–29 months and matched on gender and MA".

This is the only source in this repository that carries a **typically developing comparison group**. Its non-vocal expressive modality is a **symbolic gesture** lexicon rather than a formal sign lexicon; it is read as this repository's `signed` construct (see [Measurement and column semantics](#measurement-and-column-semantics)).

## Fields

<!-- spellchecker: disable -->

- subject_id
- pair_id
- group
- sex
- age
- age_days
- mental_age
- mental_age_level
- understood
- spoken
- gestured
- spoken_or_gestured

<!-- spellchecker: enable -->

## Instrument

The **CDI-Down** — the Spanish adaptation of the MacArthur-Bates Communicative Development Inventories for children with Down syndrome (Galeote, M., Checa, E., Sánchez-Palacios, C., Sebastián, E., & Soto, P. (2016). _American Journal of Speech-Language Pathology_, 25(3), 371–380. <https://doi.org/10.1044/2015_AJSLP-15-0007>). Its vocabulary checklist is **651 words organised into 21 categories**; the instrument also has actions-and-gestures and sentences-and-grammar sections, which were not supplied here.

For each word the parent reports whether the child understands it, says it, and/or expresses it by gesture. The gesture column is not a tally of generic communicative gestures: the adaptation "added a third column to assess the comprehension and production of gestures representing specific lexical items" (Galeote et al., 2011) — that is, **symbolic or referential gestures**, each tied to one of the 651 words. The study's stated objective was to analyse vocabulary size "both in the oral and gestural (symbolic gestures) modalities".

## Measurement and column semantics

| Column               | Meaning in es_01                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `subject_id`         | Anonymised identifier, prefix `ID_`                                                             |
| `pair_id`            | Matched-pair key, 1–186; one DS and one TD child share each value                                |
| `group`              | `DS` (Down syndrome) or `TD` (typically developing)                                             |
| `sex`                | `M` or `F`                                                                                      |
| `age`                | Chronological age in months, rounded (11–71 DS, 6–33 TD)                                        |
| `age_days`           | Chronological age in days, on the source's 30-day month                                         |
| `mental_age`         | Brunet-Lézine Psychomotor Development Scale-Revised total developmental age, months (8.00–29.40) |
| `mental_age_level`   | CDI-Down age band used for matching, 1–7 (1 = 8–10 months … 7 = 26–28 months)                    |
| `understood`         | Words understood                                                                                |
| `spoken`             | Words said                                                                                      |
| `gestured`           | Words expressed by symbolic gesture — a **total**, including words also spoken                   |
| `spoken_or_gestured` | Words said **or** expressed by symbolic gesture — a union, each word counted once                |

The number of words expressed in _both_ modalities is `spoken + gestured − spoken_or_gestured`. Among the Down syndrome children it is 0 for 52 and up to 212; for 6 of them every gestured word is also spoken. All 186 carry a non-zero gestural total.

### The matched-pair design

`pair_id` links a child with Down syndrome to their typically developing match. Across the 186 pairs the mental age level is identical for all 186, the Brunet-Lézine total developmental age is within 0.3 months for all 186 (identical for 87), and sex is identical for 185 — pair 186 is a DS boy matched to a TD girl, retained as supplied.

### How this source enters the analysis pool

The `vocab_combined` view admits the **Down syndrome children only** (`group = 'DS'`, 186 rows). The typically developing children are a Spanish-normed comparison sample on a different instrument from the Wordbank forms the TD reference pool draws on, so pooling them would put a second instrument into the pool the Down syndrome exclusions are benchmarked against. They remain in the `vocab_es_01` table for a matched-pair analysis.

`gestured` becomes the view's **`signed`** column and `spoken_or_gestured` becomes **`produced`**. The symbolic gestures are a non-vocal expressive lexicon scored per word, which is the construct the signing models (VG14, VG15, VG17) estimate, and — like `uk_02` and `nz_01`, and unlike `uk_01` (see `SIGNED_ONLY_STUDIES`) — the count is a **total**, covering words gestured whether or not they are also spoken. It therefore needs no item-level re-derivation to be comparable. This makes es_01 the second-largest signing source in the pool at 185 usable signed observations, and its `produced` is a de-duplicated union recorded by the source itself rather than reconstructed (see the `model_vg18` docstring for how `produced` varies across sources). `produced` exceeds `spoken` by a mean of 28 words.

Two caveats sit behind that mapping. Symbolic gestures are not a formal sign language, so es_01's signing construct is a near neighbour of the taught-sign lexicons in `uk_01`, `uk_02` and `nz_01` rather than the identical thing; and the view masks the `signed` value of any row whose gestural total exceeds its own union, which is impossible (one row, see Known issues).

es_01 also supports a **four-cell within-understood cross-tab** — neither, spoken-only, gestured-only, both — of the kind `common_joint_modality` builds for `uk_02`, since `both = spoken + gestured − spoken_or_gestured` and `neither = understood − spoken_or_gestured` are both derivable. Wiring that would let es_01 inform the sign–speech overlap `psi` directly instead of only through its marginals. It is not wired here: it is a model change rather than a data change, and the one inconsistent row yields a negative cell.

## Known issues

- **Comprehension ceiling.** Two children with Down syndrome (`pair_id` 9 and 17, aged 60 and 59 months) understand all 651 words. These are legitimate ceiling observations rather than errors, but they are censored: the true receptive vocabulary of both is _at least_ 651 words. Nothing exceeds the ceiling on any of the four counts, so the form-ceiling guard drops no row.
- **Age scale.** The source gives age both as months-and-days and as a day total, related by exactly 30 days per month, so one is a rendering of the other rather than an independent measurement. `age` and `age_days` follow the document's months-and-days column, which is plausible on every row; the day total contains one impossible value. If the day total is instead the true count, the ages here run about 1.4% high (≈0.5 months at the DS mean of 34). Taking the document at its word puts the TD group's mean developmental quotient at 102.5 against 104.0 for the alternative reading — mildly in its favour, but not decisive. Eight rows disagree between the two columns; five shift the rounded age by 1–3 months.
- **One impossible gestural total.** One DS child (`pair_id` 148) records 1 word spoken and 15 gestured but a union of 11 — a union cannot be smaller than either part. Which of the three numbers is wrong cannot be determined, so all three are retained verbatim in this file, and the `vocab_combined` view masks that row's `signed` value instead: a gestural total above its own union is not a usable total-sign count. The row's `understood`, `spoken` and `produced` are unaffected and it is not dropped. It is the only row of 186 affected, and the only one whose within-produced cross-tab would have a negative cell.

Full preparation detail, including the anonymisation scheme and the source transcription, is documented alongside the preparation script in the `research-data-analysis` repository (`projects/vocabulary/prepare/es_01_galeote.md`).

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
