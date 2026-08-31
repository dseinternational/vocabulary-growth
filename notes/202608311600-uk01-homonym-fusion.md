# uk_01 homonym fusion: two children under one subject id

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

**2026-08-31.** An exploratory look at individual repeated-measures trajectories against the pooled median raised two questions: why some within-child segments are vertical, and why some decline. Chasing the declines led to the subject-id scheme, and to one uk_01 id that almost certainly fuses two different children. This note records the audit, the evidence, and what was done.

## Context: what the spaghetti plots showed

Plotting every Down syndrome child with two or more observations (lines joining visits, pooled 6-month-bin median and mean overlaid) showed three kinds of surprising segment. All were classified by joining consecutive same-child observations and asking whether the pair crosses forms (`survey_vocab_max` differs):

- **Vertical segments** are all same-age dual-form administrations — uk_02 children given the Oxford CDI (416) and DSE checklist (810) at one visit, us_01 children given Words & Gestures and Words & Sentences at one age, and uk_01 WG+WS pairs. There are no same-age pairs on a single form anywhere in the pool.
- **Declines onto a shorter form** are largely mechanical: of the five understood declines landing on a shorter form, four end above 80% of the destination form's ceiling (one exactly at the Oxford 416 ceiling), and even sub-ceiling a shorter form has fewer items to endorse.
- **Within-form declines** are mostly small (median drops of 2–32 words by study) and read as parent-report noise — except the two largest, both in uk_01, which turned out not to be declines at all.

## The finding

uk_01 is the one source whose subject identifier derives from the child's **name** alone (`research-data-analysis`, `prepare/uk_01_edg.py`): the raw SPSS file has no per-child identifier, so the name is the longitudinal linker, and the preparation docs carry a homonym caveat — two different children sharing a name fuse silently into one `subject_id`. `assert_injective` cannot catch this, because identical name → identical id is the intended longitudinal behaviour.

A merge-signature scan over the whole DS pool (spoken count dropping ≥ 100 words then recovering to at least the pre-drop maximum) returned exactly one child, and it is a uk_01 id:

| id                  | sex | age (mo) | spoken | signed | form |
| ------------------- | --- | -------- | ------ | ------ | ---- |
| ID_E33ADE657109EBB8 | F   | 66       | 8      | 225    | 396  |
| ID_E33ADE657109EBB8 | F   | 76       | 451    | 0      | 690  |
| ID_E33ADE657109EBB8 | F   | 78       | 27     | 126    | 690  |
| ID_E33ADE657109EBB8 | F   | 88       | 483    | 0      | 690  |

The four rows split perfectly into two interleaved, internally consistent modality profiles: a signer who barely speaks (66, 78) and a speaker who never signs (76, 88). Read as one child this is a 424-word production collapse in two months followed by a 456-word surge — and it is precisely the "uk_01 record at 76 months" that the longitudinal-collapse rule's docstring (`COLLAPSE_FACTOR` in `data_utils.py`) deliberately left for separate investigation, its age scope being justified developmentally. Read as two children, both trajectories are ordinary.

Verified at source (printing no names): the four rows sit under **one exact canonicalised name** in `Combined groups.sav`, and the file's non-item columns are survey, name, age, age group and sex only — no date of birth, record number or site, so no mechanical disambiguation is possible. The 24 source name-groups with ≥ 3 rows map one-to-one onto the 24 multi-row prepared ids, so the pipeline preserves the source grouping faithfully; the defect is the source's lack of a per-child key, not the hashing.

A follow-up name inspection (2026-08-31, same day) sharpened this. The four rows carry one **byte-identical** raw string — two tokens (given name and surname), title case, no case or whitespace variants — and the HMAC of the canonicalised name reproduces `ID_E33ADE657109EBB8` exactly, so the linkage is verified cryptographically, not just by age signature. No other row in the file shares even the first name, so there is no spelling variant that should have been linked instead. The file keys on full names throughout (222 of 224 rows are two-token), which makes a true homonym — two children in a ~130-child cohort sharing given name _and_ surname — less likely than a data-entry mix-up in which one child's name was written on another child's form. Either way the aggregate file cannot resolve it. The same inspection found no duplicate rows: no two rows are identical, none share (name, survey, age), and the only identical item-response vectors under different names are the five all-blank Words & Sentences forms the preparation already drops as structurally incomplete.

## What was done

`ID_E33ADE657109EBB8` is withheld whole — all four rows — at CSV load in `scripts/prepare_data.py`, via `UK01_WITHHELD_SUBJECTS` / `drop_uk01_withheld_subjects` in `data_utils.py`, the same mechanism as `UK07_WITHHELD_ADMINISTRATIONS`. Whole-subject rather than per-row, because which rows belong to which child is exactly what the aggregate data cannot establish, and assigning rows to children by their outcome profile would be selection on the outcome. Reinstatement is removing the id from the tuple and re-running `scripts/prepare_data.py`. In the default pool the cost is four spoken observations: uk_01's `signed` is masked by default (`SIGNED_ONLY_STUDIES`) and `understood` is missing on all four rows.

Deliberately **not** withheld: `ID_CEBD1F6C4348C78C` (M, eight rows, 35–80 months), the only other uk_01 id pairing a substantial-signer row (35 mo: 11–20 spoken / 163–183 signed, same-day WG+WS) with non-signing speaker rows (240 spoken / 0 signed by 44 mo, then a non-monotone series to 80 mo including a −106 drop). Two same-named boys would explain it, but so would a genuine sign-to-speech transition — which is what the signing models estimate — so it stays in as a sensitivity target. If the study records can be consulted, both ids should be adjudicated together.

## The opposite defect: spelling variants splitting one child (same-day follow-up)

A review of the exported names found the mirror-image failure: spelling variants of one child's name that hash to two subject ids, splitting the child and hiding repeat measures. A systematic scan (full-name edit distance ≤ 2, same-given-name/surname variants, diminutive prefixes, token containment) over the 224-row source produced six candidate pairs. Five were merged; one was deliberately kept separate:

- **Merged**: three one-letter typos and one diminutive-versus-full given name, each pairing a WG administration with a later WS administration of the same sex and developmentally coherent counts (WG 23 mo → WS 42 mo, WG 35 mo → WS 55 mo, WG 69 mo → WS 84 mo, plus a variant whose only row is an all-blank form already dropped); and one name with the middle names written out, whose row has no recorded age and so never reaches the models.
- **Kept separate**: a distance-1 pair whose recorded sexes differ (`ID_C8BB632B295E2AE2` M / `ID_731F1BADD111A15B` F) — the pattern of two genuinely different children with near-identical names.

The mechanism lives in `research-data-analysis`: `original-data/name-corrections.csv` maps each canonicalised variant to its corrected spelling (evidence in the `note` column), and `prepare/uk_01_edg.py` applies the mapping before hashing, failing the build on a stale or chained entry. For the typo pairs the direction of correction is arbitrary — either spelling may be the true one — and only the linkage matters; flip an entry if the real spelling is known. Regenerating remapped 4 rows and changed nothing else (219 rows, every non-id column identical); distinct ids in the CSV fell from 134 to 130, and the modelled uk_01 pool from 132 children to 129, three of them now cross-form repeat-measures children rather than six singletons.

## Consequences and follow-ups

- The prepared frame changes for every model consuming uk_01 spoken observations, so `data.analysis_frame_hash` no longer matches existing fits of those models — they are stale by design and need refitting before the next report sync.
- The subject-id audit otherwise came back clean: no truncation collisions, subject ids disjoint across studies in `vocab_combined`, no within-id sex conflicts anywhere, and the us_01 DuckDB `hash()` step collision-free (243 raw ids → 243 hashed).
- Two observations worth separate consideration: (1) the two us_01 same-day WG/WS pairs at 23 months disagreeing wildly (WG 50 vs WS 406; WG 11 vs WS 385) — the WS side includes the record the implausible-production docstring retains as uncontradicted, but a same-day WG count of 50 arguably contradicts it; (2) us_01's re-hash in `vocab_combined_view_sql` uses DuckDB's internal `hash()`, which is unkeyed and not guaranteed stable across DuckDB versions — a version bump could silently re-id every us_01 child (frame-hash validation would catch the staleness; moving the step onto the shared HMAC scheme would remove the dependency).
