# Handover: the eleven refits the comprehension cap change requires

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Written 2026-08-22 at the study owner's instruction to run the refits in a separate session. Everything needed to start is here. **Nothing is published and nothing should be until these land** — both uploads were held deliberately, so no incorrect artefact reached the public container. The change that made the fits stale is `ae04e5e`, which returned `report_max_age_understood` to 72; the reasoning is in [202608221200](202608221200-reporting-source-by-quantity.md) §4 and is not repeated here.

## 1. Why the fits are stale, and why that is correct

`report_max_age_understood` is a **model-definition field**, and `fit_artifacts.validate_fit_output` compares the definition as a whole object against `fit_manifest.json`. Changing it therefore marks every fit that carries it stale, even though the cap is pure post-processing and **cannot move a posterior**. Every refit will reproduce its own trace up to Monte Carlo error.

That is the design, stated in `reporting_ages`: "change it and every affected model of record is correctly marked stale." It is worth not fighting. The alternative — a reporting field outside the fingerprint — is how a published figure silently disagrees with the definition that produced it.

## 2. What to refit

Eleven models, all at `--config rep`:

| model | current trace | notes                                                               |
| ----- | ------------: | ------------------------------------------------------------------- |
| VG02  |          1.3G | DS understood, univariate                                           |
| VG05  |          2.9G | DS joint, no random effects                                         |
| VG07  |          9.5G |                                                                     |
| VG08  |          9.9G |                                                                     |
| VG09  |           14G | watch convergence — regressed once under the `kappa` block, see #55 |
| VG10  |           11G | VG19 and VG20 inherit the field from this definition                |
| VG14  |           12G | trivariate; `report_max_age_signed` stays 84 on its own field       |
| VG15  |          6.7G | joint sign/speech                                                   |
| VG16  |           11G | cross-lag                                                           |
| VG19  |           12G | not a model of record, but registered, so it blocks the sync        |
| VG20  |           11G | **model of record**                                                 |

`VG01`, `VG03`, `VG04`, `VG11`, `VG12` and `VG13` are **not** affected: VG01/VG03 carry no comprehension quantity, VG11/VG13 declare no cap, and VG04/VG12 declare 25 under the separate TD policy (#228), which this change does not touch.

**VG19 is not optional even though it is not a model of record.** `sync_report_figures` validates every directory that resolves to a registered model and one failure aborts the whole run — the trap recorded at `docs/runbooks/full-refit.md:464`.

## 3. Disk — plan this before starting

At the time of writing `/scratch2` has **33 G free** against **98 G** of current traces for these eleven. Refitting writes the new trace before the old is replaced, so the peak matters.

The clean source of room is the **three VG19 `rep` recovery replicate directories, 35 G in total**, which are fully scored — `recovery_matrix_vg19.csv` and the six per-replicate CSVs were written 2026-08-22 08:03 — and have no remaining consumer. Removing them gives roughly 68 G.

Then choose persistence deliberately:

- `--trace-persistence compact` takes the set to roughly 35 G and is **byte-identical for reporting**. It drops the observation-sized deterministics, which are recomputable. It does **block** later recovery scoring, `regenerate_plots.py` and `kfold_loso`-adjacent work on those fits without another refit.
- `full` needs the 98 G and leaves every downstream option open.

Given that recovery, k-fold and the parameter-recovery baselines have all just been run and are unaffected by this change (§5), `compact` is the reasonable default here, with `full` for **VG20** alone if any of its downstream work is expected.

## 4. Order of operations, and the traps

The canonical sequence is `docs/runbooks/full-refit.md:443-464`. The three that have bitten this project before:

1. **`sync_report_figures.py --config rep --allow-caveats`.** The flag is required: VG12 and VG13 carry recorded soft-tier caveats, and without it the sync fails on them with no mention of sampling configuration at all. (It was four models before the recent refits; on the 2026-08-22 run it was down to two.)
2. **`prepare_report_figures.py` AFTER the sync, every time.** The sync _atomically replaces_ `docs/report/figures/`, destroying the illustrative `bayes_update*.png` the introduction uses, and `quarto render docs/report` then fails on a missing file.
3. **Copy the comparisons immediately before rendering the comparison book, in the same shell.** `docs/comparison/` is gitignored, so a stale copy survives indefinitely and renders without complaint. Treat a comparison-book render as invalid unless the copy directly precedes it.

Then: comparisons, model reports (`fit_model.py <id> --config rep --render-only` per model, or `--render` on the fit), the report book, the comparison book, and only then the uploads.

**Uploads.** `upload.py` uses the Azure SDK with `DefaultAzureCredential`, **not** azcopy. On this VM the managed identity wins and has no write role, so `export AZURE_TOKEN_CREDENTIALS=dev` is required to resolve the developer identity — verified 2026-08-22. `DSERESEARCH_BLOB_CONTAINER_URL` is `https://dseresearch.blob.core.windows.net/public`. **Traces must not go there**: `--include-traces` is off by default and must stay off, because `trace.nc` carries per-observation ages, subject codes, study codes and counts. The internal container path for the trace archive was never supplied and is still outstanding.

## 5. What is NOT stale, and must not be re-run

None of these reads a trimmed reporting table, so the cap cannot touch them:

- **K-fold LOSO** ([202608212000](202608212000-vg19-gates-g2-g4-g5.md) §4e) — held-out predictive densities computed from the likelihood. `kfold_loso_*_ds_slope_corr.csv` in `comparisons/` stand.
- **G3 recovery** (§4d) — scores parameters against a known truth.
- **Gate 1 for the low-rank successor** ([202608221000](202608221000-four-by-four-gate1.md)) — maximum likelihood on raw residuals; no fitted model involved.
- **The individual-trajectory analysis** ([202608220748](202608220748-vg19-individual-trajectories.md)) — reads posterior parameters. Its §3 tables run to 84 months, above the new cap, and §6.6 records that they are internal-only.

What **is** stale and was produced this session before the cap changed: the figure sync (2,566 files), all seventeen model report renders, the report book, the comparison book, and the comparison suite outputs. Redo all of them after the refits.

## 6. Verification before publishing

- `python scripts/check_fit.py all --config rep --purpose publish --output-dir <root>` passes.
- `pytest` is clean. **39 tests currently fail** and all 39 are this staleness — `test_reporting_age_policy.py` (30) and `test_reported_age_range.py` (9) check persisted output against the current policy. 809 pass, 43 skip. If any of the 39 survives the refits, it is a real defect and not this.
- Spot-check that comprehension figures and tables now stop at 72 and that **spoken still runs to 90** and **signed to 84** — the whole point of the per-quantity policy is that they move independently.

## 7. Decisions that could be folded into the same session

Both are definition changes, so bundling them costs nothing extra in refits and avoids fragmenting the staleness across further commits. Both are the study owner's call.

1. **Register the low-rank factor successor** to VG19/VG20, per [202608221000](202608221000-four-by-four-gate1.md) §5 — `b = L z` with `k` a definition field and `k = 1, 2, 3` as a registered sensitivity family. Note the free 4x4 is **not** what to build; that note explains why.
2. **Register VG21**, the `window-22` promotion, whose prior gate passed on 2026-08-21 ([202608211545](202608211545-window-22-prior-gate-passed.md)) and which has been unblocked since.
