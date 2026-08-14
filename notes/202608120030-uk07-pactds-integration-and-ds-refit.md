# uk_07 (PACT-DS) integration and the Down syndrome refit, 11–12 August 2026

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Run record and findings. The fits described here were produced from a **dirty checkout** and are therefore analytically valid but **not publishable** — see §7. Every difference reported is attributable to uk_07 alone: the baseline is the 2026-08-06 published set, whose `source_data_hash` matches this data tree with `vocab_data_uk_07.csv` removed, byte for byte. **These are not the models of record.** uk_06's signing was reinstated the day after these fits ran, which moves §5 in particular — see §5a for the numbers to quote.

## 1. What was added

`uk_07` — the **PACT-DS** feasibility randomised controlled trial (Burgoyne, Baxter, Hartwell, Pagnamenta & Stojanovik; ESRC ES/V016946/1), received as a UK Data Service deposit. 30 children with Down syndrome, randomised to intervention or waiting control, each assessed at three points on a 674-item "Reading CDI" (the University of Reading adaptation, which adds a per-item sign coding). **82 analysed administrations at 34–95 months.** Source detail is in [`data/vocab_data_uk_07.md`](../data/vocab_data_uk_07.md); the preparation record lives in the `research-data-analysis` repository.

Two properties make this source unlike any other in the pool.

**It is the pool's only substantial source of older-age signing.** Signed observations above 60 months go from 46 to 87; above 72 months, from **0 to 23**. The band 72–96 months previously contained no signed observation at all.

**It carries a within-understood four-cell cross-tab.** Its expressive coding is modality-exclusive — says-only, signs-only, both — and it records comprehension, so the fourth cell follows as `understood − produced`. It is the second source after `uk_02` able to identify the sign–speech association `psi`, taking the rows that identify it from 56 to 138 and their span from 20–56 months to 20–95.

The expressive columns follow the `nz_01` convention (exclusive cells), **not** the `uk_01`/`ie_02`/`uk_04`/`uk_05` one (any-modality totals). Read as totals they would understate both marginals by the say-and-sign overlap, which reaches 331 items on one row. The `vocab_combined` view re-derives `spoken = a + c` and `signed = b + c`; the derived `signed` is a genuine total, so uk_07 needs no signing mask.

## 2. One administration withheld

One row (58 months) records 191 words understood against 489 produced — the only row in the source where production exceeds comprehension, at the end of a reported comprehension decline (349 → 291 → 191) while production rises. The likeliest reading is a parent-report artefact, but that is a hypothesis about how the form was completed, not something the counts can settle.

It is **withheld pending clarification with the source team** (`data_utils.UK07_WITHHELD_ADMINISTRATIONS`), dropped at CSV load so it is absent from the DuckDB table, the view, the merged CSV and VG15's cross-tab path alike. This is stricter than `ie_01`'s seven comparable records, which are retained and counted as source-data violations; the difference is that those are a settled property of a closed source while this one has a reachable source team. It is also the one row whose `understood_only` cross-tab cell would be negative.

The fit logs confirm the treatment worked: the DS "child > understood" violation count stayed at **11 / 0**, exactly as before.

## 3. The refit

All ten DS models refit at `rep` on 2026-08-11, **2h 40m** total on a 48 GB / 12-performance-core Mac. Every model passes the convergence gate with **0 divergences**; R-hat and minimum ESS are unchanged or slightly better throughout.

|        | rows  | understood | spoken | signed (VG14 scope) | four-cell rows |
| ------ | ----- | ---------- | ------ | ------------------- | -------------- |
| before | 1,349 | 905        | 1,346  | 593                 | 56             |
| after  | 1,431 | 987        | 1,428  | 675                 | 138            |

Two models needed the documented hightune escalation, and the ladder **flipped** relative to the August run: VG09 failed at plain `rep` (3 R-hats to 1.016) and passed at hightune, as before; VG08 **passed** at plain `rep` but **failed** the same hightune settings that had rescued it in August (3 R-hats to 1.013). Both fits converged on their own terms — VG08 sits near the gate boundary either way — but it means the VG08 comparison below mixes a data change with a sampling change unless the stronger hightune retry lands.

## 4. Finding: the sign–speech association is substantially stronger

|              | before         | after          |
| ------------ | -------------- | -------------- |
| `psi` median | 1.797          | **2.495**      |
| 90% CI       | [1.250, 2.388] | [1.946, 3.084] |
| P(`psi` > 1) | 0.998          | 1.000          |

The interval no longer approaches independence. On the log scale it narrowed by about 29%, which is what 2.5× the identifying data should buy. Because a stronger association means more overlap between the sign and speech lexicons, VG15's total expressive `p_any` falls 3–5% at 36–60 months — moving further from VG14's independence bound, which is the point of VG15 over VG14.

`psi` remains a single age-invariant scalar, so uk_07's older window and uk_02's younger one are pooled rather than compared. Whether the association varies with age is now a question the data could begin to answer; it could not before.

## 5. Finding: the signing decline after the peak is much shallower

This is the largest movement anywhere in the refit, and it is where uk_07 supplies evidence the pool did not have.

| VG14                  | before | after     |
| --------------------- | ------ | --------- |
| `r(48)`               | 0.318  | 0.344     |
| `r(60)`               | 0.205  | 0.232     |
| `r(72)`               | 0.156  | 0.198     |
| `r(90)`               | 0.098  | **0.182** |
| words signed at 90 mo | 57.5   | **100.2** |

Note [202606151700](202606151700-vg14-signed-ratio-shape-and-p-any-bias.md) identified the apparent peak-then-recede as largely a between-study boundary and could not settle it. It now has data behind it rather than the tent prior's extrapolation, and the answer is that the recede is real but far weaker than the published fits show.

**This was invisible in the reports, and the cap has moved.** `report_max_age_signed = 60` trimmed every signed table and figure at 60 months, and the docstring justifying that cap had gone stale: it read "23 above 60, 7 above 72 and none between 84 and 96", so "above about 60 months `r(a)` is the tent's extrapolation, not an estimate". It is now 87 above 60 and 23 above 72, from 18 additional children.

On that evidence the study owner **raised the cap to 84**. The 72–84 band now carries 15 observations from real children rather than nothing, so reporting it is no longer extrapolation. 84–96 stays out deliberately: 8 observations from a single source is evidence but not enough to publish a curve on, and 84 is also the Down syndrome models' high trend anchor, above which the mean is clamped rather than fitted. The cap is part of the recorded definition, so the change is folded into the clean-tree refit rather than needing a third pass.

### 5a. Superseded by the clean-tree refit — read these numbers instead

The table above is a record of the 11 August fit, in which uk_07 was the only change. It is **not** the model of record. Two things happened after it, both of which move this finding, so the "after" column should not be quoted forward.

**uk_06's signing was reinstated** ([`de0c18b`](https://github.com/dseinternational/vocabulary-growth/commit/de0c18b), 12 August, closing #211) once the study owner confirmed its construct is the standard DSE checklist total rather than uk_01's sign-only field. That adds 11 signed observations, **all of them at 60–115 months** — precisely the band this finding is about. Signed observations go 675 → 686, and the above-60 count goes from the 87 quoted above to 98. So the effect measured here is uk_07's alone only up to 12 August; from then on it is uk_07 **and** uk_06, and the two cannot be separated in the models of record.

**The signed reporting cap became its own field** ([`4ff48e5`](https://github.com/dseinternational/vocabulary-growth/commit/4ff48e5)). Until then `report_max_age_signed` existed only on `JointModelDefinition`, so VG14's sign-derived figures and its `r(a)` table were trimmed by `report_max_age_understood` — which meant raising the _comprehension_ cap from 72 to 84 moved VG14's _signed_ output as an unintended side effect, and the `r(a)` table ran to 90 untrimmed. `r(90)` in the table above is therefore beyond what VG14 now reports at all, and should not be quoted.

Current models of record (VG14 at `rep`, clean tree at `4ff48e5`, uk_07 + uk_06), median with 89% CI, to the 84-month cap:

| VG14                  | 2026-08-06 fit | model of record      |
| --------------------- | -------------- | -------------------- |
| `r(48)`               | 0.318          | 0.340 [0.311, 0.378] |
| `r(60)`               | 0.205          | 0.249 [0.213, 0.281] |
| `r(72)`               | 0.156          | 0.211 [0.172, 0.249] |
| `r(84)`               | —              | 0.198 [0.152, 0.261] |
| words signed at 84 mo | —              | 105.2 [79.2, 141.4]  |

Only the 48- and 60-month rows of the left column were ever _published_: `report_max_age_signed` was 60 then, so 72 and beyond were read off the trace, not the report. They are shown for continuity with the table above, which used the same source.

The finding itself stands and is slightly strengthened. The recede after the peak is real but far weaker than the published fits show, and on the current evidence `r` is close to flat from 72 to 84 (0.211 → 0.198, intervals overlapping heavily) rather than continuing to fall.

## 6. Finding: older-age comprehension dispersion falls sharply

| VG10 `kappa_u` | before            | after             |       |
| -------------- | ----------------- | ----------------- | ----- |
| 24 mo          | 66.6 [55.8, 77.7] | 65.3 [55.0, 76.2] | −2%   |
| 48 mo          | 18.2 [14.5, 21.6] | 24.6 [20.3, 29.2] | +36%  |
| 72 mo          | 6.2 [4.2, 8.2]    | 11.2 [7.9, 14.7]  | +82%  |
| 90 mo          | 3.5 [2.0, 5.1]    | 7.2 [4.3, 10.5]   | +108% |

The intervals barely overlap at 72 and 90 months, so this is not sampling noise. The implied variance inflation factor falls from 114 to 67 at 72 months and from 181 to 99 at 90. Production dispersion `kappa_s` is essentially untouched (+0% to +6% across the range).

The reading is that the pool's older-age comprehension counts previously came from few and heterogeneous sources, and 82 internally consistent observations from 30 children across 34–95 months remove much of the overdispersion the model needed to accommodate that tail. This bears directly on the report's `kappa` section and on the DS/TD dispersion comparator, which was repointed to VG10 in the August run and will now report materially different older-age numbers.

## 7. Why these fits cannot be published

`check_fit.py all --config rep --purpose publish` rejects every one of them: **"The fit was produced from a dirty or unverifiable checkout."** The wiring was uncommitted when the fits ran, and the manifest records `code.dirty = true`. The gate is doing exactly its job. The numbers above are real and the comparison is sound, but promoting them as models of record requires committing the change and re-running from a clean tree (~3 h for the DS pool).

Two further blocks stand between here and a publishable set.

**The five TD models are flagged stale** — "Raw data inputs differ from those used for this fit" — because `source_data_hash` covers every CSV in `data/`, not only the ones a given model reads. uk_07 is a Down syndrome source and changes nothing about the TD analysis frame, but the hash cannot express that, so they need refitting to publish. VG11 is the obstacle: it peaked at 157 GB anon-RSS on the 251 GB VM in August and will not fit in 48 GB. The TD refits need the VM.

**VG11–VG13 carry pre-existing soft-tier caveats** (divergences, BFMI) unrelated to this change, and publish through `--allow-caveats`.

## 8. Smaller movements

All modest, and nothing below 24 months moves at all — uk_07 starts at 34.

- **Production ratio `q(a)` falls**: VG10 −7% at 24–36 months, −2% to −4% at 48–72; VG09 the same. Consistent with uk_07's children routing more expression through sign.
- **Comprehension rises at older ages**: VG02 words understood at 72 months 396 → 422 (+6.7%); VG07 +2.8%; VG15 +2.6%.
- **Spoken at 90 months falls about 5%**: VG01 411 → 390; VG15 434 → 411.
- **VG16** (cross-lag) barely moves: within ±1.4% everywhere.

## 9. Consequence for LOO

VG15's spoken and signed LOO components exclude rows represented by the four-cell likelihood, by design. That exclusion has grown from 56 rows to 138, which widens the pre-existing non-comparability of VG14-against-VG15 LOO. It is a documented property rather than a new defect, but the comparison book should say so where the two are placed side by side.

## 10. Outstanding

- [x] Commit the wiring and re-run the DS pool from a clean tree (§7) — committed on `feat/uk07-pactds`; the clean-tree refit follows in a second commit.
- [x] Decide `report_max_age_signed` — raised from 60 to **84** by the study owner on the §5 evidence. 84–96 stays out: 8 observations from one source, and 84 is the trend's high anchor.
- [x] `uk_06` signing construct — now tracked as [#211](https://github.com/dseinternational/vocabulary-growth/issues/211). See the appendix below for the evidence it carries.
- [ ] Refit the five TD models on the VM; VG11 will not fit locally (§7).
- [ ] Regenerate `output/comparisons/` — the DS/TD comparison artefacts still reflect the pre-uk_07 DS fits.
- [ ] Say in the comparison book that VG14-against-VG15 LOO is non-comparable, and by how much more (§9).

## Appendix: uk_06 {#appendix-uk_06}

Asked in passing during this work: where is the uk_06 sign-construct verification? It was nowhere; it is now [#211](https://github.com/dseinternational/vocabulary-growth/issues/211). The 15 June decision to include it ([202606151700](202606151700-vg14-signed-ratio-shape-and-p-any-bias.md) §3) was reversed on 16 July by `8ea6227`, which set `include_uk06 = False` "until its field dictionary confirms comparability". It is **not** on issue #190 §C, whose source-codebook list holds only the ie_01 Checklist 1 denominator and the uk_01 `understood` coding, so it has no owner and no tracking item.

Two observations sharpen the question. The upstream preparation record names the source column `CheckUnderAndSign` ("understands and signs") from a DSE checklist with four categories — understands, understands-and-signs, imitates, uses-spontaneously. In the data, `signed + spoken > understood` on **7 of 11 rows**, which is impossible if those categories were a mutually exclusive ladder. So they are overlapping per-word ticks, and a word both signed and said is counted in both columns — which is the total-sign construct, and argues _against_ the `uk_01` signed-only reading the mask implicitly fears.

Against that: `imitated` exceeds `understood` on 4 of 11 rows and reaches **822**, above the 810 the view assigns uk_06 as its form ceiling, and the upstream record states this instrument's vocabulary maximum is "not documented". So the 810 assignment is itself an assumption, and a column exceeding it means either the instrument is larger or the columns are mis-mapped.

The question for the source is now concrete: are the four checklist columns independent per-word ticks; how many items does the RLI checklist have; and why does `imitated` exceed both `understood` and 810?
