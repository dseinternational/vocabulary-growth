# us_01 (Edgin, DS): WS comprehension was passed through as understood

Date: 2026-07-06

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

## Summary

The us_01/Edgin block of the `vocab_combined` view in `scripts/prepare_data.py` selected `comprehension as understood` from `wordbank_child` with no filter on `form`. Wordbank's CDI: Words & Sentences (WS) form records `comprehension` as a production proxy (`comprehension == production` by data convention), so every WS row entered the DS models claiming the child understood exactly as many words as they produced. This is the same defect that retired VG06 on the TD side (see `notes/202605151630-vg06-ws-comprehension-issue.md`); the 2026-05-15 fix guarded the TD loader (`load_data`) but not the DS path, and that note's "unaffected" verdicts for the DS models were therefore wrong.

The view has now been fixed to null out `understood` for non-bivariate forms while keeping the (valid) production counts. **VG02 and all DS joint models (VG05, VG07–VG10, VG14, VG15, VG16) trained on the corrupted rows and need refitting** after `python scripts/prepare_data.py` rebuilds the local database.

## Empirical evidence

In the English Down syndrome subset of the Wordbank export (`dataset_name = 'Edgin'`, `lower(health_conditions) = 'down syndrome'`), counting rows where `comprehension == production`:

| Form | n   | U == S      | After `production <= 100` cap | Ages (used rows) | Median "understood" (used rows) |
| ---- | --- | ----------- | ----------------------------- | ---------------- | ------------------------------- |
| WG   | 87  | 5 (5.7%)    | 79                            | 11–18 months     | 33                              |
| WS   | 109 | 109 (100%)  | 85                            | 17–27 months     | 7 (== median spoken)            |

The WS rows are production-only measurements wearing a comprehension label. The contrast at the age overlap makes the distortion concrete: WG rows at 17–18 months have median understood 46.5 (n = 46), while the WS rows spanning 17–27 months report median "understood" 7 — because that is their median production.

## Impact on the DS models

- The 85 corrupted rows were 85/819 ≈ 10% of the DS `understood` pool, and 48 of them sit above 18 months, precisely the window where the DS comprehension–production gap is of interest.
- Every model consuming DS `understood` via `load_combined_data` trained on them: the univariate comprehension model (VG02) had its trajectory dragged down in the 17–27-month window, and the joint models (VG05, VG07–VG10, VG14–VG16) were additionally told `U = S` exactly on those rows, biasing the comprehension–production gap and the production ratio `q` toward 1 — the same artefact class that produced VG06's spurious "gap closes by 30 months" finding on the TD side.
- `data/vocab_data_merged.csv` is unaffected: the pandas merge never included us_01, which only enters via the DuckDB view.
- DS production observations are unaffected: WS production counts are valid and are retained (as `spoken`/`produced`).

## Fix applied (2026-07-06)

1. **`src/vocab_growth/data_utils.py`** — the `vocab_combined` view definition moved out of `scripts/prepare_data.py` into `vocab_combined_view_sql()` so it is importable and regression-tested. The us_01/Edgin block now takes `understood` only from forms in `WORDBANK_BIVARIATE_FORMS` (renamed from `TD_BIVARIATE_FORMS` — the bivariate property belongs to the instrument, not the population) and both the TD and DS guards are built from that one constant.
2. **`tests/test_data_utils.py`** — the fixture now builds the full `vocab_combined` view over a miniature `wordbank_child`, and four DS regression tests assert that WS rows contribute `spoken` but never `understood`, that WG rows keep independent comprehension, that no surviving us_01 row carries the `understood == spoken` proxy, and that the production cap behaves as documented.

Rebuilding the database changes no us_01 row counts (164 rows: 79 WG + 85 WS); the 85 WS rows simply have `understood` NULL instead of the production proxy.

## The `production <= 100` cap

While fixing the block we documented the `AND production <= 100` filter, which has been in place since the initial import with no recorded rationale. In the current export it drops the highest-production administrations: 8/87 WG and 24/109 WS English DS rows. It is retained as-is (removing it would confound the refit that this fix already requires) but should be reviewed: either record a rationale (e.g. an early-vocabulary scope decision) or remove it and let the Beta-Binomial ceiling handle the full range.

## Models requiring attention

| Model     | Population | Outcome(s)                   | Affected?                        | Action                              |
| --------- | ---------- | ---------------------------- | -------------------------------- | ----------------------------------- |
| VG01      | DS         | Spoken                       | no — WS production is valid      | none                                |
| **VG02**  | **DS**     | **Understood**               | **yes**                          | **refit**                           |
| VG03/VG04 | TD         | Spoken / understood          | no — TD loader guarded 2026-05-15 | none                                |
| **VG05**  | **DS**     | **Joint U + S**              | **yes**                          | **refit**                           |
| **VG07–VG10** | **DS** | **Joint U + S (+ REs)**      | **yes**                          | **refit**                           |
| VG11–VG13 | TD         | Various                      | no                               | none                                |
| **VG14**  | **DS**     | **U + S + signed**           | **yes** (us_01 U/S marginals)    | **refit**                           |
| **VG15**  | **DS**     | **Joint modality**           | **yes** (us_01 in merged marginals) | **refit**                        |
| **VG16**  | **DS**     | **Cross-lag (U → q)**        | **yes**                          | **refit**                           |

Downstream DS-vs-TD comparisons and report chapters that consume these posteriors (comprehension–production gap, joint trajectory, latency, `q` overlap) need regenerating after the refits.

## Reproducing the discovery

```python
import pandas as pd

from vocab_growth.data_utils import ENGLISH_LANGUAGES

df = pd.read_csv("data/wordbank_administration_data.csv", low_memory=False)
edgin_ds = df[
    (df["dataset_name"] == "Edgin")
    & (df["language"].isin(ENGLISH_LANGUAGES))
    & (df["health_conditions"].str.lower() == "down syndrome")
]
for form, rows in edgin_ds.groupby("form"):
    eq = (rows["comprehension"] == rows["production"]).sum()
    print(f"{form}: n={len(rows)}, U==S on {eq} ({100 * eq / len(rows):.1f}%)")
```
