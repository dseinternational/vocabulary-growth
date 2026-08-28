# es_01 (Galeote): the gesture column measures symbolic gesture, not taught signing

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Investigation record, 27 August 2026, prompted by the study owner re-reading Galeote et al. (2011) and questioning whether es_01's non-vocal expressive column is "a pure measure of lexical signing". It re-examines the construct decision recorded in [202608121030 §2](202608121030-psi-heterogeneity-and-age-invariance.md), which reversed an earlier draft's treatment of es_01's gestures as a different construct from taught signs on the grounds that the coding is structurally identical to the sign sources'. **Conclusion: that reversal was right about the coding and wrong about the construct.** The CDI-Down's third column is a per-word non-vocal expressive marker, exactly as `uk_02`, `uk_07` and `nz_01` code signs — but what it marks is the child's _symbolic-gesture_ repertoire, explicitly including spontaneously learnt iconic gestures, undifferentiated from any taught signs. That is a related but broader construct than the taught lexical signing the UK and New Zealand sources record, and it is the most economical explanation of es_01 sitting at independence on the sign–speech association while every sign source sits far above it. Nothing in the data preparation or model machinery is wrong; several interpretive sentences are, and one measurement caveat in the source paper is not yet on record here.

## 1. What the source says it measured

Four things in Galeote et al. (2011), read in full for this note, settle the construct.

**The definition includes spontaneous gestures by design.** "Symbolic gestures are defined as those gestures that, when properly taught or spontaneously learnt, substitute specific lexical items (e.g., closing the hand with tight fingers and moving it towards the mouth to simulate 'to eat')." The exemplar is a natural iconic gesture, not an item from a sign lexicon, and parents were never asked to distinguish taught signs from home-grown gestures. The scoring instruction was that "Only words produced or signed referentially and spontaneously were marked", with parents shown examples and photographs of symbolic gestures during the initial interview. The DS children were recruited through early intervention units and parent associations and "All children received regular therapy from birth", so a taught component is surely present for many — but it is unidentifiable inside the total.

**The typically developing group scores substantially on the same column.** The 186 mental-age-matched TD children — recruited through ordinary nurseries, with no sign instruction reported and none plausible at scale — average 23.6 gestured words (maximum 105), non-zero at every mental-age level, gesturing 6.5 words at MA 9 months when they speak only 1.9. A column that required taught conventional signs would read near zero in this group. This is decisive: whatever the column captures is something typical Spanish toddlers do spontaneously, which is the symbolic/representational gesture repertoire of the developmental gesture literature.

**The trajectory is a gesture trajectory.** Gestural production rises early, plateaus at 18–21 months MA, and slightly declines at the oldest ages, in both groups — the classic pattern of gesture being supplanted by speech. The paper interprets it exactly that way, situating itself in the symbolic-gesture literature (Caselli et al., 1998, and related work), reading the DS group's larger gestural totals as compensation for the expressive-speech deficit, and noting that children "progressively give up the use of gestures" as oral vocabulary expands. Its clinical discussion is about parents' worries over gesture use — the authors themselves treat the measure as gesture throughout.

**Our data are the published scoring.** The four counts in `vocab_data_es_01.csv` reproduce the paper's Table 2 group means exactly, to the second decimal on both groups:

| group | n   | understood | spoken | gestured | union  |
| ----- | --- | ---------- | ------ | -------- | ------ |
| DS    | 186 | 272.91     | 104.56 | 40.60    | 132.66 |
| TD    | 186 | 244.53     | 114.52 | 23.56    | 127.65 |

So whatever measurement treatment the published analysis carries, we carry.

## 2. A measurement caveat not previously on record: eleven categories eliminated from the gesture data

The paper's Method section reports a validity exercise on 66 parents who were asked to describe each of their child's gestures. Categories whose gesture reports "frequently presented problems" were identified — words for people, body parts, food and drink, clothes, objects and places at home, objects and places away from home, questions, prepositions, auxiliary verbs, periphrasis, and sentence connectors — and the authors "decided to be conservative and eliminate these categories from the data". That is 11 of the instrument's 21 categories, leaving the gesture assessment spanning 10.

The elimination can only have applied to the gesture column: two DS children in our data reach the full 651 on comprehension, so the complete checklist stands behind `understood` (and `spoken`), while the DS `gestured` maximum is 214. It is possible in principle that the elimination describes only the validity analysis rather than the study data, but since our totals match Table 2 exactly we hold whichever treatment the published analysis applied — the question for the author is which that is.

If the totals do carry it, `gestured` is measured over a substantially reduced item universe relative to `understood` and `spoken` — plausibly around half, though the exact share depends on the CDI-Down's category sizes, which we do not hold — and the eliminated set includes several of the most gesture-dense categories (people, body parts, food and drink, clothes, household objects and places), where `uk_07`'s sign coding spans all 674 of its items. For the four-cell cross-tab this means items in eliminated categories can land only in the spoken-only or neither cells, never in both or gestured-only. The direction of the induced bias on the within-understood odds ratio depends on where the child's gestures in those categories would truly have fallen, so it is not determinable from totals; its scale is bounded by the proportional inflation of the affected cells, which at an eliminated share near half cannot plausibly exceed a factor of about two in either direction. It is therefore a genuine comparability artefact stacked on top of the construct difference, not an alternative explanation: no correction of that size bridges 0.90 to 6–14.

## 3. The data behave like a gesture measure

Every quantitative signature es_01 shows is the one a symbolic-gesture measure would produce and a taught-sign measure would not. The association table ([`scripts/psi_heterogeneity_audit.py`](../scripts/psi_heterogeneity_audit.py), re-run today against the current loader rules — it reproduces [202608121030](202608121030-psi-heterogeneity-and-age-invariance.md) exactly):

| source  | rows | MH odds ratio | reference set     | per-child OR < 1 | non-vocal words also spoken |
| ------- | ---- | ------------- | ----------------- | ---------------- | --------------------------- |
| `uk_02` | 56   | 6.09          | within understood | 4%               | 50.4%                       |
| `uk_07` | 82   | 13.90         | within understood | 11%              | 72.2%                       |
| `nz_01` | 111  | 14.63         | all 675 items     | 4%               | 44.8%                       |
| `es_01` | 185  | **0.90**      | within understood | **45%**          | **30.8%**                   |

Taught key-word signing is sign-_and_-say by construction, so the signed and spoken lexicons coincide — `uk_07`, the intervention-context source, has 72% of non-vocal words also spoken. A spontaneous gesture repertoire is deployed for words the child cannot yet say and abandoned as speech arrives, so the two lexicons diverge — es_01 sits at independence with the lowest overlap. The age-matched contrast (1.00 against 5.94 for `uk_02` at an identical median 41 months) shows this is not composition.

The non-vocal lexicon is also the wrong _size_ for taught signing. Mean `signed` in `vocab_combined` at 34–56 months, across the seven usable-total sources with rows in that window (`uk_01` is masked as sign-only, `uk_06` has no rows there):

| source  | n   | mean signed | max signed |
| ------- | --- | ----------- | ---------- |
| `es_01` | 62  | **50.7**    | 134        |
| `nz_01` | 74  | 81.0        | 232        |
| `uk_05` | 5   | 93.8        | 193        |
| `ie_02` | 25  | 102.8       | 301        |
| `uk_04` | 6   | 106.3       | 197        |
| `uk_07` | 37  | 131.7       | 290        |
| `uk_02` | 71  | 186.5       | 641        |

es_01 is the smallest, at roughly a third of `uk_02` — repertoires of a few dozen items, exactly where the gesture literature sits, against sign vocabularies running to hundreds. And within es_01 the gestural lexicon stalls while speech runs away: at 4–5 years its children average 300+ spoken words against 39–69 gestured. (Part of the level gap may be the §2 category truncation; the trajectory shape is not.)

The internal TD control reads naturally under the construct account. The identical instrument finds a positive association in the TD group (Mantel-Haenszel 2.13, climbing to 2.3–3.5 at MA levels 4–6) against independence in the DS group — TD toddlers gesture words at the edge of their spoken development, while children with DS use gesture to compensate for the words speech withholds, which is the paper's own account of its DS–TD contrast. The instrument can plainly detect coincidence; the DS independence is a property of how a compensatory gesture lexicon relates to speech, and under this reading it is _expected_, not anomalous. es_01's fitted per-study association (1.08, 90% HDI [0.87, 1.35] in the dev-config fit of [202608121030 §6](202608121030-psi-heterogeneity-and-age-invariance.md)) is then a **gesture–speech association** — a defensible finding in its own right and a useful bookend to the sign sources, not a discordant signing result.

## 4. What this changes, and what it does not

**Nothing in the data preparation or the model machinery.** The union semantics, the four-cell arithmetic and the 185-of-186 valid partitions all stand — they were verified against the source's own labels and are construct-independent. `delta_psi` remains exactly the right structure, and arguably more clearly so: the study-level term is what licenses pooling a gesture measure with sign measures at all, because each source keeps its own association and the population value is a shrunk centre. `include_es01_cells = True` remains defensible, but the stated _reason_ shifts — es_01 stays in as its own study measuring a related but broader construct, not because "it is the same measurement".

**The reading of the between-study spread.** [202608121030 §2](202608121030-psi-heterogeneity-and-age-invariance.md) concludes "The heterogeneity below is therefore a real difference between samples, not an artefact of what was being counted." The second half of that sentence no longer stands: the es_01-versus-sign-sources gap is plausibly first-order a difference in what was counted, with the §2 category truncation a smaller artefact on top, and the residual spread among the three sign sources (6–15) the part that is about samples and context. The two cannot be apportioned — no source records its children's sign instruction — and the §8 warning against a teaching interpretation stands unchanged, including the uk_07 control-arm evidence that cuts against any simple teaching story.

**The signed marginal in VG14/VG15.** es_01 is the single largest source of usable signed totals — 185 of the default pool's 686 signed observations — and supplies 52 of the 142 below 24 months (alongside `ie_02` 54, `uk_05` 16, `uk_04` 13, `uk_02` 7). So $r(a)$ is fitted to a construct mixture, roughly one-third gesture at the young end; `delta_sign` absorbs the level difference but not the construct. The four-cell side is sharper: below 20 months the $\psi$-informing rows are es_01 alone, so the young end of the association evidence is entirely gesture evidence.

**Sentences now wrong or overstated**, to be revised rather than rewritten around:

- [202608121030 §2](202608121030-psi-heterogeneity-and-age-invariance.md) — a decision record, so per the notes policy the correction should be **appended as a flagged block**, not edited in.
- `data/vocab_data_es_01.md` — "the terminology differs, the measurement does not" (and the same framing in its analysis-pool section); the §2 category elimination also belongs in its Known issues, marked pending author confirmation.
- The `include_es01_cells` docstring in `src/vocab_growth/models/definitions.py` — "It is the same measurement."
- `docs/report/_caveats-signing.qmd` and the es_01 study description in `docs/report/methods-data.qmd` (its open `[TODO: lexical signs or not?]` resolves to: _not purely_ — symbolic gestures including, but not limited to and probably not dominated by, taught signs). Mind the caveats file's freeze note: delete `_freeze/<chapter>/` for every chapter that includes it before rendering.

## 5. Recommended actions

- [ ] Append the correction block to 202608121030 §2.
- [ ] Revise `data/vocab_data_es_01.md` (construct wording; add the 11-category elimination to Known issues).
- [ ] Revise the `include_es01_cells` docstring; keep the default `True` with the corrected rationale.
- [ ] Revise the report: `methods-data.qmd` es_01 description, and a construct sentence in `_caveats-signing.qmd` beside "No source records how its children were taught to sign".
- [ ] Ask Galeote directly — the route that settled `uk_06`'s construct (issue #211): (a) confirm parents did not distinguish taught signs from spontaneous gestures; (b) confirm whether the supplied totals reflect the 11-category elimination; (c) confirm the supplied TOTAL GESTURES column counts gesture _production_ only, given the instrument's third column assessed "the comprehension and production" of gestures; (d) ask whether item-level or category-level data survive.
- [ ] Optional: cite an `include_es01_cells = False` sensitivity fit if the report wants to show the population $\psi$ is robust to excluding the gesture source.

No refit is required by this note: it changes interpretation and documentation, not the graph, the data rules or the priors.

## 6. Reproduction

The association and age-matched tables are printed by `uv run python scripts/psi_heterogeneity_audit.py` (re-run 2026-08-27; reproduces). The Table 2 comparison and the gestured maxima are direct pandas aggregations of `data/vocab_data_es_01.csv` by group; the cross-source signed table and pool counts are DuckDB aggregations of `vocab_combined` (`signed IS NOT NULL`, excluding `uk_01`; the 34–56 month window for the size table). Galeote et al. (2011), doi:10.3109/13668250.2011.599317, was read in full from the Zotero library; the quoted definitions are from its Predictions, Instruments and Procedure sections.
