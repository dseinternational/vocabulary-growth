# Steep within-child vocabulary gains: what is real and what is not

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

**2026-08-31.** After the uk_01 identifier fixes (`notes/202608311600-uk01-homonym-fusion.md`), the repeated-measures spaghetti plots still showed remarkable within-child gains in comprehension and production over short periods. This note records the investigation: how steep gains were ranked and benchmarked, which proved genuine, which proved defective, and the two data-preparation actions taken.

## Method

Within-child, within-form gain segments (consecutive observations of one child on one form) were ranked by words per month, each flagged child's full record inspected, study-level gain-rate distributions compared, and — the key yardstick — the same statistic computed for the typically-developing Wordbank pool's repeat-measures children.

## The typically-developing yardstick

In the TD pool (within-child, within-form, gaps of 1–8 months): comprehension gains of **40+ words/month occur in 21% of segments** (median 26, 99th percentile 90, maximum 297); production gains of 40+ words/month occur in 10% (99th percentile 75). Parent-report CDI vocabulary spurts of this size are simply normal instrument behaviour. The steepest Down syndrome gains (110, 76, 70, 53 words/month) sit between the TD 90th and 99th percentiles — DS children reach TD-sized spurts, at older ages.

## Gains that look genuine

- **Production explosions on large comprehension**: an it_01 cluster (gains of 203–240 words over 6 months, ages 28–42 months) in children already understanding 270–330 words — production catching up to receptive vocabulary; uk_02's 43→461 over 13 months (understood 242→510 alongside) is the same shape.
- **Sign-to-speech transitions**: nz_01's 155→333 spoken gain with signed falling 263→179 — words migrating between modalities while total production grows modestly.
- **Coherent multi-visit accelerations**: uk_05's child rising smoothly on both outcomes over three visits (understood 93→132→353, spoken 8→67→252); uk_07's (PACT-DS trial) rising on all three outcomes across 13 months (understood 298→586→655, spoken 111→346→514).

## Two defects found, two actions taken

**ie_02 `ID_62C63BE2B3B627E6` t2 (48 months).** Understood 111→442 (+331, the pool's steepest comprehension gain and beyond the TD 99th percentile) in the same administration where spoken collapses 72→3 and signed jumps 64→301. The spoken collapse is the "ie_02 record at 45 months" the longitudinal-collapse rule deliberately left for separate investigation; seen whole, the t2 administration is internally contradictory — the pattern of a checklist completed differently between waves (the DSE checklists carry separate per-word "understands and signs" and "says" columns). Which columns are trustworthy cannot be recovered from aggregate counts, so the administration is withheld whole at CSV load (`IE02_WITHHELD_ADMINISTRATIONS` in `data_utils.py`, applied in `prepare_data.py`), the same mechanism as uk_07's withheld administration. The child's t1 stands.

**us_01 same-day WG/WS contradictions.** The pool's steepest production gain (7→385 over 18→23 months, Words & Sentences) is contradicted by the child's same-day Words & Gestures administration recording 11 words spoken; a second child records 406 on WS against 50 on WG the same day. Both WS counts are impossible against the Berglund et al. (2001) benchmark, and the 406 record is the one the implausible-production docstring had long retained as "extreme but uncontradicted" — the same-day WG administration is the contradiction. A new masking rule, `mask_same_day_production_disagreements` (constants `SAME_DAY_DISAGREEMENT_*`, default on, reinstatement flag `include_same_day_disagreements`), masks the larger side of a same-day pair when it is at least 100 words and at least 5× the smaller. The smaller side is corroborated by the benchmark and by the child's independently measured WG comprehension-production gap, so it is retained — unlike the duplicated-outcome rule, which masks both sides because neither is defensible there. Scoped to us_01: uk_01's same-day pairs agree closely, and uk_02's dual-form pairs differ mechanically (Oxford 416 vs DSE 810 inventories), where a threshold rule would be wrong.

## The ie_02 wave pattern, noted but not masked

ie_02 dominates the steep-comprehension list (six of the top twelve) and its median within-child comprehension gain rate (12.7 words/month) is roughly double every other DS study's. Two mundane mechanisms cover it: the study's 3-month wave gap doubles the noise of a words-per-month rate relative to the 6-month studies, and t1 is a parent's first exposure to an 810-item checklist — under-report then recalibration at t2 is a familiar parent-report effect, and several of the big comprehension jumps co-occur with equally large signing jumps (the parent finding more of everything at t2). Worth remembering when interpreting ie_02 within-child slopes; not a defect class to mask.

## Consequences

- The prepared frame changes for models consuming ie_02 (one administration fewer) and us_01 spoken observations (two counts masked), so their fits are stale by design, on top of the uk_01 staleness from the same day's identifier fixes.
- The retained high-comprehension us_01 records (213/217 understood at 18 months; the 70→334 and 75→278 comprehension gains by 23 months) remain sensitivity targets per the study owner's earlier ruling — nothing new was found against them beyond their membership of the same cluster, and nothing here masks them.
