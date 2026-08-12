# Vocabulary data - UK (7)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5), from the deposit documentation, the upstream preparation record and the data (2026-08-11).

**PACT-DS** — the feasibility randomised controlled trial "Feasibility of a Parent-Delivered Early Language Intervention for Children with Down Syndrome" (Burgoyne, Baxter, Hartwell, Pagnamenta & Stojanovik; UKRI Economic and Social Research Council grant ES/V016946/1, Universities of Reading and Manchester). Received as a UK Data Service archive deposit.

**30 children with Down syndrome**, randomised to **intervention** (n = 15) or **waiting control** (n = 15), each assessed at three points: T1 baseline, T2 immediately after the 30-week intervention, and T3 four months later. Three children were lost to the project after T1 and one further child did not return the T2 parent survey, giving **30 + 26 + 27 = 83** assessment points at **34–95 months**, of which **82 are analysed** (one is withheld — see Known issues). This is the pool's longest-running longitudinal Down syndrome source at school-entry ages, and — with signing recorded at 81 of its 83 points, from all 30 children — its largest signing source.

## Fields

<!-- spellchecker: disable -->

- subject_id
- group
- sex
- timepoint
- age
- understood
- spoken
- signed
- spoken_signed
- produced
- survey_vocab_max

<!-- spellchecker: enable -->

## Instrument

The **Reading CDI** — a **674-item** adaptation of the MacArthur-Bates Communicative Development Inventories used at the University of Reading, derived from the British-English Oxford CDI of Hamilton, A., Plunkett, K., & Schafer, G. (2000). Infant vocabulary development assessed with a British Communicative Development Inventory. _Journal of Child Language_, 27(3), 689–705. <https://doi.org/10.1017/S0305000900004414>. Administered online to parents at each of the three assessment points.

Each of the 674 items carries a comprehension tick and a **three-way expressive coding**: says only (no sign), signs only (no says), or signs and says (both checked). The three expressive cells are mutually exclusive and sum to the deposit's total expressive score.

The deposit also holds direct child assessments (Mullen, EOWPVT, Action Picture Test, YARC, and a bespoke 12-item expressive/receptive test) and parent measures (AC-QoL, HADS, BRIEF-P). Only the CDI, age, sex and trial arm are carried here.

## Measurement and column semantics

| Column             | Meaning in uk_07                                                              |
| ------------------ | ----------------------------------------------------------------------------- |
| `subject_id`       | Anonymised identifier, prefix `ID_`; repeats across a child's three time points |
| `group`            | `intervention` or `control` — the trial arm                                   |
| `sex`              | `M` or `F`                                                                     |
| `timepoint`        | `t1` (baseline) / `t2` (post-test) / `t3` (follow-up)                          |
| `age`              | Chronological age in months at that assessment point (34–95)                   |
| `understood`       | Items understood                                                               |
| `spoken`           | Items produced by **word only** — says checked, sign not                       |
| `signed`           | Items produced by **sign only** — sign checked, says not                       |
| `spoken_signed`    | Items produced by **both** word and sign                                       |
| `produced`         | Items produced in any modality — the union, `spoken + signed + spoken_signed`  |
| `survey_vocab_max` | 674, constant — the form's item count                                          |

### ⚠️ `spoken` and `signed` are modality-exclusive cells, not totals

This follows the **`nz_01`** (Foster-Cohen) convention, **not** the one in `uk_01`, `ie_02`, `uk_04` and `uk_05`, where those columns are already any-modality totals. Reading uk_07's cells as totals would understate both marginals by the say-and-sign overlap, which is substantial here (up to 331 items on one row).

The `vocab_combined` view therefore re-derives them exactly as it does for `nz_01`:

- `spoken` (any modality) = `spoken + spoken_signed`
- `signed` (any modality) = `signed + spoken_signed`
- `produced` = the source column, already a de-duplicated union

The re-derived `signed` is consequently a **total** sign count, comparable with `uk_02`, `nz_01` and `es_01` without item-level re-derivation. uk_07 is **not** a `SIGNED_ONLY_STUDIES` case (as `uk_01` is), and its construct is documented rather than unverified (as `uk_06`'s is), so its signing values are not masked in the primary signing analyses.

### The four-cell cross-tab that identifies $\psi$

Unlike `nz_01`, uk_07 records comprehension at every retained point. It is therefore the **second source after `uk_02`** supplying the four-cell **within-understood** cross-tab that identifies the sign–speech association $\psi$ in VG15 — neither, spoken-only, signed-only, both — since `understood_only = understood − produced` and the other three cells are recorded directly. All 82 retained rows yield a non-negative `understood_only`, and the four cells sum to the recorded comprehension total exactly.

VG15 consumes those cells (`common_joint_modality._load_uk07_four_cell`), which takes the rows identifying $\psi$ from **56 to 138** and their age span from 20–56 months to 20–95, with the two sources overlapping between 34 and 56. On the four-cell rows uk_07's marginal `spoken` and `signed` are suppressed, because the composition term already carries them; every other model reads uk_07's marginals from `vocab_combined` as usual.

The `include_uk07_cells` definition flag (default `True`) turns this off. Unlike `include_nz01_cells`, turning it off does **not** drop the study — uk_07's marginals stand on their own — so the flag isolates uk_07's pull on the association alone.

Two caveats bear on that contribution. uk_07 is a randomised trial sample, so the intervention arm's growth across the three assessment points is partly programme-driven; and the association is still modelled as a single age-invariant scalar, so uk_07's older window and uk_02's younger one are pooled rather than compared.

### Both trial arms are pooled

The models describe vocabulary against age rather than treatment effect, and the arm is a property of the child rather than of the measurement, so `vocab_combined` admits both. `group` stays in the `vocab_uk_07` table for a stratified analysis. The caveat is real and recorded below: the intervention arm's T1–T3 growth is partly programme-driven.

## Known issues

- **One administration withheld: production above comprehension.** One child at 58 months records 191 understood against 489 produced. Its comprehension is reported as falling across the study (349 → 291 → 191) while production rises (185 → 263 → 489), and it is the only row in the source where `produced > understood` — most likely a parent-report artefact (only the expressive columns ticked at the later visit) rather than a real loss of comprehension, but that is a hypothesis about how the form was completed, not something the counts can settle. It is **withheld pending clarification with the source team**, so 82 of the 83 assessment points are analysed. This is stricter than the treatment of `ie_01`'s seven comparable records, which are retained and counted as source-data violations; the difference is that those are a settled property of a closed source, while this one has a reachable source team and an open question. It is also the one row whose `understood_only` cross-tab cell would be negative. The row is dropped at CSV load, so it is absent from the DuckDB table, `vocab_combined`, `vocab_data_merged.csv` and VG15's cross-tab path alike. Reinstating it means removing its entry from `data_utils.UK07_WITHHELD_ADMINISTRATIONS` and re-running `scripts/prepare_data.py`. The child's other two assessments are unaffected and remain in the pool.
- **Small, age-shifted, and a trial sample.** 30 children at 34–95 months — older than most sources in the pool — and the intervention arm's growth between T1 and T3 is partly attributable to the programme. Neither is a defect, but both bear on how a study-level effect for uk_07 should be read.
- **Longitudinal.** `subject_id` repeats across a child's time points; a row is identified by `(subject_id, timepoint)`. The repeated-measures models cluster on `subject_id` within study.
- **No ceiling exposure.** Nothing in the source reaches 674 on any of the four counts. Five of the 83 rows sit within 90% of the ceiling on comprehension, and one on production.
- **`444` is both a missing code and an attainable score** in the deposit. The upstream preparation resolves every case in this deposit by treating `444` as missing only when it marks all six CDI columns of a time point, but any future `444` should be checked against source records.

Full preparation detail — the anonymisation scheme, the sentinel missing codes (`9000` lost to project, `999` non-responsive, `888` refusal, `444` survey not returned), and the arithmetic identities asserted against the deposit's own totals — is documented alongside the preparation script in the `research-data-analysis` repository (`projects/vocabulary/prepare/uk_07_pactds.md`).

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) — see `LICENSE` for details.
