# Vocabulary data - Spain

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5), from the source document and the data (2026-08-03).

A Spanish sample of **186 children with Down syndrome** and **186 typically developing children**, matched pair-wise on mental age and sex, provided directly by the author (Miguel Galeote). Cross-sectional: one observation per child, 372 rows in total.

The sample matches that reported in Galeote, M., Sebastián, E., Checa, E., Rey, R., & Soto, P. (2011). The development of vocabulary in Spanish children with Down syndrome: comprehension, production, and gestures. _Journal of Intellectual & Developmental Disability_, 36(3), 184–196. <https://doi.org/10.3109/13668250.2011.599317> — "186 children with DS and 186 children with TD, with a mental age (MA) of 8–29 months and matched on gender and MA".

This is the only source in this repository that carries a **typically developing comparison group**. Its non-vocal expressive modality is described by the source as *gestural*, but it is a **lexical** one — gestures representing specific lexical items, each tied to one of the 651 checklist words and scored per word, which is the same coding `uk_02`, `uk_07` and `nz_01` apply to signs. It is read as this repository's `signed` construct (see [Measurement and column semantics](#measurement-and-column-semantics)); the terminology differs, the measurement does not.

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

### The source's own column labels

The four counts are labelled in the original Galeote table as:

| original label                    | column here          |
| --------------------------------- | -------------------- |
| **TOTAL COMPREHENSIÓN**           | `understood`         |
| **TOTAL PRODUCTION**              | `spoken`             |
| **TOTAL GESTURES**                | `gestured`           |
| **WORD PRODUCED + GESTURES ONLY** | `spoken_or_gestured` |

These labels settle the two questions that matter, and are recorded here so the convention is readable from the source rather than re-derived.

First, **`gestured` is a total**, counting words gestured whether or not they are also spoken. The label says so, and the fourth column's construction confirms it: the qualifier "ONLY" is there because the union adds just the gestures *not* already counted in production — which is only necessary if the third column is not gesture-only. Galeote et al. (2011) describe the fourth column as "total lexical production combining the two modalities (oral + gestural production)".

Second, **`spoken_or_gestured` is a union, not an intersection.** "WORD PRODUCED + GESTURES ONLY" could be misread as "words produced by word *and* gesture only". The data rules that out: the column exceeds `min(spoken, gestured)` on all 186 rows, and is at least `spoken` on 186 of 186.

The reading is also forced arithmetically, independently of the labels. If `spoken` and `gestured` were disjoint cells — the `nz_01`/`uk_07` convention — then `spoken_or_gestured = spoken + gestured` would hold identically on every row. It holds on only 52 of 186 (and 56 of 186 TD rows); the other 134 have a union strictly smaller than the sum, by a median of 7 words and up to 212. Those 52 are not a rival explanation: 17 have zero spoken words, so their overlap is arithmetically forced to zero, and the median spoken vocabulary among them is 2 words against 58 among the rest — the overlap appears exactly where a child says enough words for some to also be gestured.

The number of words expressed in _both_ modalities is therefore `spoken + gestured − spoken_or_gestured`. Among the Down syndrome children it is 0 for 52 and up to 212; for 6 of them every gestured word is also spoken. All 186 carry a non-zero gestural total.

> [!NOTE]
> Contrast `uk_07` and `nz_01`, where `spoken` and `signed` **are** disjoint cells and the additive identity does hold exactly (83 of 83 rows for uk_07). The same arithmetic test separates the two conventions, so it is worth running on any new source rather than reading the column names.

### The matched-pair design

`pair_id` links a child with Down syndrome to their typically developing match. Across the 186 pairs the mental age level is identical for all 186, the Brunet-Lézine total developmental age is within 0.3 months for all 186 (identical for 87), and sex is identical for 185 — pair 186 is a DS boy matched to a TD girl, retained as supplied.

### How this source enters the analysis pool

The `vocab_combined` view admits the **Down syndrome children only** (`group = 'DS'`, 186 rows). The typically developing children are a Spanish-normed comparison sample on a different instrument from the Wordbank forms the TD reference pool draws on, so pooling them would put a second instrument into the pool the Down syndrome exclusions are benchmarked against. They remain in the `vocab_es_01` table for a matched-pair analysis.

`gestured` becomes the view's **`signed`** column and `spoken_or_gestured` becomes **`produced`**. The symbolic gestures are a non-vocal expressive lexicon scored per word, which is the construct the signing models (VG14, VG15, VG17) estimate, and — like `uk_02` and `nz_01`, and unlike `uk_01` (see `SIGNED_ONLY_STUDIES`) — the count is a **total**, covering words gestured whether or not they are also spoken. It therefore needs no item-level re-derivation to be comparable. This makes es_01 the second-largest signing source in the pool at 185 usable signed observations, and its `produced` is a de-duplicated union recorded by the source itself rather than reconstructed (see the `model_vg18` docstring for how `produced` varies across sources). `produced` exceeds `spoken` by a mean of 28 words.

One caveat sits behind that mapping: the view masks the `signed` value of any row whose gestural total exceeds its own union, which is impossible (one row, see Known issues).

### The four-cell cross-tab, and the heterogeneity it exposed

es_01 supports a **four-cell within-understood cross-tab** — neither, spoken-only, gestured-only, both — of the kind `common_joint_modality` builds for `uk_02` and `uk_07`:

```
understood_only = understood        − spoken_or_gestured
spoken_only     = spoken_or_gestured − gestured
gestured_only   = spoken_or_gestured − spoken
both            = spoken + gestured  − spoken_or_gestured
```

These sum to `understood` identically, and **185 of 186 rows yield a valid partition** at 11–71 months. The exception is the known defective row (`pair_id` 148, see Known issues), whose `spoken_only` is −4; it routes to the marginal set, keeping its comprehension and spoken counts while its gestural total stays masked.

The loader (`common_joint_modality._load_es01_four_cell`) is implemented and tested, and **`include_es01_cells` defaults to `True`** since 2026-08-12. It was `False` for the nine days before that, and the reason was never the construct — it was that the sources already informing $\psi$ disagree about it substantially, and $\psi$ had nowhere to put that.

| source  | rows | MH odds ratio | reference set     | per-child OR < 1 | non-vocal words also spoken |
| ------- | ---- | ------------- | ----------------- | ---------------- | --------------------------- |
| `uk_02` | 56   | 6.09          | within understood | 4%               | 50.4%                       |
| `uk_07` | 82   | 13.90         | within understood | 11%              | 72.2%                       |
| `nz_01` | 111  | 14.63         | all 675 items     | 4%               | 44.8%                       |
| `es_01` | 185  | **0.90**      | within understood | **45%**          | 30.8%                       |

Two caveats. Mantel-Haenszel is a crude descriptive statistic on the observed cells, not $\psi$ itself, which is population-conditioned against the fitted $r$ and $q$. And `nz_01` has no comprehension total, so its "neither" cell spans all unproduced items rather than understood-but-unproduced, which inflates its odds ratio — the same data for `uk_07` reads 13.90 within understood and 40.72 over all 674 items. Magnitudes compare only within a reference set; the per-child sign and the share-also-spoken column need no "neither" cell and compare throughout.

What survives every control is that es_01 sits at independence while the three sign sources are positive. By age band it runs 0.30–1.12 against 4.4–41.6 for `uk_02` and 4.4–18.1 for `uk_07`, with no overlap in any band; matched on expressive vocabulary (30–300 words) it is 1.05 against 4.80 and 9.68. On the conditioning-free share-also-spoken measure it is the low end of a continuous gradient rather than categorically apart. Either way the spread is large, plausibly reflecting whether signing was taught alongside speech — both UK sources come from contexts where it is, and `uk_07` is an intervention trial — though four studies cannot test that.

That heterogeneity was disqualifying only because **$\psi$ was the only latent in VG15 with no study-level term.** `delta_u`, `delta_q` and `delta_sign` are all study random intercepts; `log_psi` was a bare global scalar. A pooled $\psi$ was therefore a precision-weighted average over whichever sources happened to be in the pool — which is why it moved from 1.80 to 2.49 when `uk_07` arrived, and why adding es_01's 185 rows would have dragged the headline toward independence as an artefact of composition rather than a finding.

$\psi$ now carries `delta_psi`, a zero-sum study random intercept over the $\psi$-informed studies, with `tau_psi` quantifying the spread. Each source keeps its own association and the reported population value is a shrunk centre, so these cells add evidence instead of moving the headline by composition — which is why the flag defaults `True`. Setting it `False` isolates es_01's contribution. The age question that the table below might suggest was tested and rejected: see [202608121030](../notes/202608121030-psi-heterogeneity-and-age-invariance.md).

## Known issues

- **Comprehension ceiling.** Two children with Down syndrome (`pair_id` 9 and 17, aged 60 and 59 months) understand all 651 words. These are legitimate ceiling observations rather than errors, but they are censored: the true receptive vocabulary of both is _at least_ 651 words. Nothing exceeds the ceiling on any of the four counts, so the form-ceiling guard drops no row.
- **Age scale.** The source gives age both as months-and-days and as a day total, related by exactly 30 days per month, so one is a rendering of the other rather than an independent measurement. `age` and `age_days` follow the document's months-and-days column, which is plausible on every row; the day total contains one impossible value. If the day total is instead the true count, the ages here run about 1.4% high (≈0.5 months at the DS mean of 34). Taking the document at its word puts the TD group's mean developmental quotient at 102.5 against 104.0 for the alternative reading — mildly in its favour, but not decisive. Eight rows disagree between the two columns; five shift the rounded age by 1–3 months.
- **One impossible gestural total.** One DS child (`pair_id` 148) records 1 word spoken and 15 gestured but a union of 11 — a union cannot be smaller than either part. Which of the three numbers is wrong cannot be determined, so all three are retained verbatim in this file, and the `vocab_combined` view masks that row's `signed` value instead: a gestural total above its own union is not a usable total-sign count. The row's `understood`, `spoken` and `produced` are unaffected and it is not dropped. It is the only row of 186 affected, and the only one whose within-produced cross-tab would have a negative cell.

Full preparation detail, including the anonymisation scheme and the source transcription, is documented alongside the preparation script in the `research-data-analysis` repository (`projects/vocabulary/prepare/es_01_galeote.md`).

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
