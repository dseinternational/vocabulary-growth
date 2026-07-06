# VG06 (TD bivariate): WS comprehension is production-only in Wordbank

Date: 2026-05-15

> **Note:** This document was generated with assistance from an AI model (Claude, Anthropic) and should be independently verified.

## Summary

While reviewing VG09B output against VG06 for a DS-vs-TD comparison, we discovered that the `comprehension` column in Wordbank's CDI: Words & Sentences (WS) data is a production proxy, not an independent measurement. Including WS in models that learn a comprehension trajectory (VG04, VG06) had been telling those models, by data convention, that "U = S" for the majority of TD rows above 18 months. This invalidated all prior VG06-based claims about TD comprehension trajectory and the comprehension–production gap above ~18 months.

VG06 has been refit with WS excluded; VG04 has _not_ yet been refit. A careful review of the data handling code and a higher-density VG06 fit are the planned next steps.

## Empirical evidence

Across the full English Wordbank export (the English subset of `data/wordbank_administration_data.csv`, 39,551 rows), we counted rows by form where `comprehension == production`:

| Form        | n          | U == S              | U > S         | U < S | Verdict             |
| ----------- | ---------- | ------------------- | ------------- | ----- | ------------------- |
| WG          | 5,123      | 112 (2.2%)          | 5,011 (97.8%) | 0     | **bivariate**       |
| Oxford CDI  | 1,210      | 7 (0.6%)            | 1,203 (99.4%) | 0     | **bivariate**       |
| WGShort     | 547        | 22 (4.0%)           | 525 (96.0%)   | 0     | **bivariate**       |
| **WS**      | **10,689** | **10,681 (99.93%)** | **8 (0.07%)** | 0     | **production-only** |
| WSShort     | 63         | 63 (100%)           | 0             | 0     | production-only     |
| TEDS Twos   | 11,129     | 11,129 (100%)       | 0             | 0     | production-only     |
| TEDS Threes | 10,790     | 10,790 (100%)       | 0             | 0     | production-only     |

The 8 WS exceptions are all from a single dataset (`Armon-Lotem`, a Hebrew/bilingual-acquisition lab) and look like a non-standard administration. They do not represent genuine WS bivariate measurement.

This is consistent with the underlying instrument design: CDI: Words & Sentences is canonically a _production-only_ form. The Wordbank schema appears to have populated `comprehension` with the production value as a placeholder so the column is non-null for every row.

## Impact on prior VG06 runs

VG06's `data_utils.load_data` (prior to the fix) returned rows from three forms: WG, WS, and Oxford CDI. After the 10% subsampling, ~1,655 rows were used to fit the model, of which roughly 63% were WS — i.e. roughly 1,000 rows in the 12–30 month window where the model was being told `U = S` exactly. This forced the fitted U trajectory to collapse onto S above ~18 months and produced the artefactual finding that the TD comprehension–production gap closes to a few words by 30 months.

The downstream consequences propagated into every DS-vs-TD comparison built on the VG06 output: joint trajectory, comprehension–production gap, learn-to-say latency, and the production-ratio q(U) overlap.

## Fix applied (2026-05-15)

Two changes:

1. **`src/vocab_growth/data_utils.py`**, line 98 — `load_data(Population.TYPICALLY_DEVELOPING, ...)` now filters `form IN ('WG', 'Oxford CDI')`. WS is dropped.
2. **`src/vocab_growth/models/definitions.py`** — `VG06` now uses `n_trials = 800` (was 690, matching the WS ceiling) and `sample_fraction = 0.25` (was 0.10, to preserve the previous effective training-set size of ~1,500 rows after the WS exclusion shrinks the pool from 16,552 to 6,134).

The 800-item reference inventory is justified because few WG/Oxford CDI observations are anywhere near their actual instrument ceilings (396 and 418 respectively): ~2.4% of WG and ~4.5% of Oxford CDI rows have U within 5% of ceiling, so treating the counts as samples from a notional 800-item reference produces an inventory-comparable scale with the DS bivariate models without meaningful distortion.

VG06 was refit on this corrected data with `--config rep` (6 chains × 6,000 tune × 6,000 draws, target_accept 0.95). Total wall time 1h 11m. Diagnostics clean: all r̂ ≤ 1.001, ESS bulk/tail well above 400. Pareto k all in (-∞, 0.7]. The refitted output replaces the previous VG06 outputs in `output/models/VG06-age-understood-spoken-td/`.

All downstream DS-vs-TD comparison scripts have been rerun (latency, q-overlap, joint trajectory, gap, q-vs-understood).

## Models requiring attention

| Model     | Outcome(s)        | Affected by WS issue?       | Status                                                                     |
| --------- | ----------------- | --------------------------- | -------------------------------------------------------------------------- |
| VG01      | DS spoken         | n/a (DS)                    | unaffected                                                                 |
| VG02      | DS understood     | n/a (DS)                    | unaffected                                                                 |
| VG03      | TD spoken         | no — WS production is valid | data filter now narrower; refit optional, not corrective                   |
| **VG04**  | **TD understood** | **yes**                     | **needs refit** — data filter is now correct but model has not been re-run |
| VG05      | DS bivariate      | n/a (DS)                    | unaffected                                                                 |
| **VG06**  | **TD bivariate**  | **yes**                     | **refit completed 2026-05-15**                                             |
| VG07–VG09 | DS bivariate      | n/a (DS)                    | unaffected                                                                 |

**Update (2026-07-06):** the "unaffected" verdicts for the DS models above were wrong. The us_01/Edgin block of the `vocab_combined` view had the same defect — WS `comprehension` passed through as `understood` unguarded — so VG02 and all DS joint models did train on WS proxy rows. See `notes/202607061200-us01-edgin-ws-comprehension-issue.md` for the DS-side fix and refit list.

## Reproducing the discovery

```python
import pandas as pd

from vocab_growth.data_utils import ENGLISH_LANGUAGES

# The export now contains all languages; restrict to English to reproduce the counts above.
df = pd.read_csv("data/wordbank_administration_data.csv", low_memory=False)
df = df[df["language"].isin(ENGLISH_LANGUAGES)]
for form in df["form"].unique():
    f = df[df["form"] == form].dropna(subset=["comprehension", "production"])
    eq = (f["comprehension"] == f["production"]).sum()
    print(f"{form:<15}  n={len(f):>6}  U==S: {eq:>5}  pct: {100*eq/len(f):>6.2f}%")
```

A form for which U==S is the norm rather than the exception cannot contribute information about comprehension as an independent measurement.

## Recommended follow-up

1. **Careful review of data-handling code.** The discovery here points to a class of bugs where the conventions of the source data don't match what the model assumes about the data. Worth a focused review of:
   - `src/vocab_growth/data_utils.py` (both `load_combined_data` for DS and `load_data` for TD).
   - `scripts/prepare_data.py`, especially the per-study transformations in the `vocab_combined` view. The `ie_01` study uses `GREATEST(says_total, understands_total)` to derive `understood`, which is a similar form of imputation worth understanding for any rows where this triggers.
   - Any per-source `survey_vocab_max` handling. The DS data has multiple instruments with different ceilings (DSE inventory at 800, Oxford CDI at 428, WG at 396, WS at 690) but the model uses a single `n_trials`; the same family of concerns applies.
2. **Refit VG04** with the same data correction. The data filter already excludes WS, but VG04's `n_trials` should also be reviewed.
3. **A higher-sample-fraction VG06 fit.** Current `sample_fraction = 0.25` was a conservative choice to preserve previous training-set size. With 6,134 bivariate rows available, going to `sample_fraction = 1.0` (or close to it) would tighten the posterior, especially in the 18–25 month tail where Oxford CDI is the sole contributor.
4. **Optional: include `WGShort`** (547 rows, Walle dataset, 16–25 months, ceiling 89). This is genuinely bivariate and would extend the bivariate window slightly. Including it would require switching the Beta-Binomial likelihood to a per-row `n_trials` tensor — a small code change in `common_bivariate.py` — since the ceiling (89) is much smaller than the others.

## Artefacts produced during this work

- `notes/202605151545-vg09b-reply-to-chris-draft.{md,docx}` — analysis write-up using corrected VG06 output.
- `output/comparisons/ds_td_joint_trajectory.{png,svg}` — Figure 22 side-by-side, regenerated.
- `output/comparisons/ds_td_comprehension_production_gap.{png,svg}` — Figure 27 side-by-side, regenerated.
- `output/comparisons/ds_td_learn_to_say_latency.{png,svg}` — latency analysis, regenerated.
- `output/comparisons/ds_td_q_overlap.{png,svg}` — q-overlap pilot, regenerated.
- `output/comparisons/ds_td_q_vs_understood_vg09b.{png,svg}` — direct overlay of model `production_rate_by_understood.csv` for both populations.
- `scripts/compare_ds_td_q_overlap.py`, `scripts/compare_ds_td_q_vs_understood.py` — new comparison scripts.
- `scripts/verify_ds_td_latency.py` — verification harness for the latency calculations (caught a separate hardcoded `n_trials` bug in `compare_ds_td_latency.py` during this work).

Earlier blob uploads:

- `019e2c2a-444f-75a3-89c4-...` — VG06 refit output, no rendered HTML (initial upload).
- `019e2c2c-d4f8-77e0-9b78-...` — VG06 refit output, with rendered Quarto HTML report.
