# Handover: the eleven refits the comprehension cap change requires

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!CAUTION]
> **Superseded as a work list on 2026-08-25 — read §0 first, then this note for the traps, the disk plan and the upload identity, which all still hold.** The comprehension cap is no longer the only thing making fits stale, four of the eleven models below have since been refit, and two models this note does not mention are stale in a way `check_fit.py` cannot see.

## 0. The current refit set (2026-08-25)

Established by running `scripts/check_fit.py all --config rep --purpose publish` and `uv run pytest` against `main` at `4bae7ea`. **Thirteen fits**, not eleven, and the reasons are now mixed:

| model                                                      | why stale                                                                                             | flagged by `check_fit`?    |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------- |
| VG05, VG07, VG08, VG09, VG10, VG13, VG16, VG19, VG20, VG22 | the `spoken_fallback` field on `BivariateModelDefinition` (PR #256)                                   | yes — "definition differs" |
| **VG14, VG15**                                             | **`posterior_summary_r` / `posterior_summary_p_any` written to 84 against a 72-month sign-ratio cap** | **no**                     |
| VG21                                                       | never fitted (registered in #249)                                                                     | yes — no trace             |

Already refit and **valid**, so leave them alone: VG01, VG02, VG03, VG04, VG11. VG12 carries a soft BFMI caveat (0.214) and no definition mismatch, so `--allow-caveats` covers it and it does not need a refit.

### VG14 and VG15 are the trap in this set

`check_fit.py` passes both, because it compares the manifest's definition object against the current one and those match. What it does not do is read the persisted tables. `tests/test_reporting_age_policy.py` does, and it is the **only** thing on `main` that is red:

```
FAILED tests/test_reporting_age_policy.py::test_every_table_respects_its_outcome_cap[vg14]
FAILED tests/test_reporting_age_policy.py::test_every_table_respects_its_outcome_cap[vg15]
```

The sequence that produced it:

1. `ae04e5e` (2026-08-22 10:47) returned `report_max_age_understood` to 72.
2. VG14 and VG15 were fitted at 15:55 and 15:57 the same day — with 72 in their definitions, and it is recorded in their manifests.
3. `reporting_ages.max_age_for_sign_ratio`, which makes `r` and `p_any` follow the _lower_ of the understood and signed caps, landed the **next day** in `565a769` (2026-08-23). At fit time those two artefacts were still capped at `report_max_age_signed` = 84.

So the definitions were current and the code was not. Verified that a refit fixes it: `max_age_for_sign_ratio` returns 72.0 for both models under `main` today. Verified that regeneration does **not** — both artefacts are summary-stage, per the runbook's own warning, and a `regenerate_plots.py` pass on 2026-08-24 13:22 refreshed VG14's plot-stage files while leaving `posterior_summary_r.csv` at 84. `KNOWN_STALE` was emptied in `fa9f836` on the assumption an earlier refit had cleared this class, so nothing excuses them.

Their traces are 12 G (VG14) and 6.7 G (VG15) at the current tier — add that to §3's disk plan.

### Decisions settled 2026-08-25, and what each does to this run

Four calls by the study owner. Two change the set, one changes the order, one changes nothing yet.

**1. The spoken fallback: arms first, then decide.** `product_marginal` stays the default. **This makes the run order load-bearing** rather than incidental: fit VG10 and VG20 with `paired-only`, `fallback-dispersion` and `marginal-moments` _early_, read them, and only then launch the remaining models. Promotion of `moment_matched` becomes a targeted follow-up if the arms say it matters. Launching the whole set first and reading the arms afterwards forfeits the point of the sequencing — if it matters you would refit everything a second time.

**2. A correlated typically-developing model, on VG13.** VG13 plus `rho_uq` and nothing else, as a **new entry in `MODEL_REGISTRY`** — not a sensitivity variant. `make_variant` is `dataclasses.replace`, which changes field values and cannot change the definition _class_, and the correlated block needs `BivariateCorrelatedSubjectREModelDefinition`. This is the VG10 → VG20 relationship repeated on the typically-developing side. VG13 is the parent rather than VG21 so the pair is a one-factor contrast against a model with a fitted history; VG21 can gain it once it has a fit of its own. **Registered as `VG23` on 2026-08-25** — definition, model module, report template, inventory row and tests, including a built-graph test showing it is VG13's graph plus `rho_uq` and `rho_uq_raw` and nothing else. **One extra fit.**

> [!NOTE]
> VG23's report template was written from scratch rather than copied from VG13's, and says so at the top. That is the whole lesson of #240: VG21's template _was_ copied, and inherited three claims that had already been withdrawn from its parent. VG23's narrative is written from the specification and will need revisiting once there are numbers — as VG21's still does.

**3. Comprehension below production: a sixth masked defect class.** Ten administrations across `ie_01` (7), `uk_01` (2) and `it_01` (1) record a comprehension count below the child's own production, which an inclusive field cannot do. `understood` is masked on those rows, with `include_comprehension_below_production=True` to reinstate. The Down syndrome pool's comprehension observations go **987 → 977**. Full reasoning, including why the denominator is the recorded `produced` union and not `spoken + signed`, in [202608251500](202608251500-comprehension-below-production.md).

> [!CAUTION]
> **This adds VG02 to the refit set, and nothing will tell you so.** The rule lives in the loader, so `data/*.csv` is untouched and `source_data_hash` does not move; `validate_fit_output` compares only that hash. Every affected fit keeps passing `check_fit.py --purpose publish` on a frame that no longer exists. Affected: **VG02**, VG05, VG07-VG10, VG14, VG15, VG16, VG19, VG20, VG22 — every Down syndrome model that observes comprehension. All but VG02 are already in the set above for other reasons, so **VG02 is the single model that would be silently missed**. VG01 observes only production; the typically-developing models never see these rows. Closing the provenance gap properly was considered and deferred for this run.

**4. The `uk_01` imputation branch is not merged.** `origin/claude/flamboyant-tu-e770df` substitutes `max(spoken, signed)` for 41 `uk_01` WG rows with no recorded comprehension. Under the `produced` rule those rows admit nothing — `max(spoken, signed) <= spoken + signed = produced` always — so the branch adds no data and should be left unmerged. It was also the wrong shape of fix: 30 of the 41 would have entered at `q` = 1.000 exactly, by construction, into a stratum whose measured median is 0.110.

**Revised total: fourteen fits** — the thirteen above plus VG02, plus the correlated VG13 (fifteen), plus the six fallback-arm fits on VG10 and VG20 if all three arms are run on both.

### What else moved since this note was written

- **`spoken_fallback` (PR #256)** is a model-graph field, so it re-staled every bivariate model. Its default is unchanged (`product_marginal`) and both engines reproduce `origin/main`'s initial log-probability to every printed digit, so this is a fingerprint change rather than a numerical one — but it is a refit either way.
- **VG21 and VG22 are registered** (`9994eab`). `fit_model.py all` derives its list from `MODEL_REGISTRY`, so both are picked up automatically; the three agent-instruction files were the only place that still listed the old set, corrected 2026-08-25.
- **The 39 failing tests §6 describes are down to 2**, and the 2 are the VG14/VG15 defect above rather than staleness that a regeneration clears.

---

_Everything below is the note as written on 2026-08-22. §2's table is the eleven-model set of that date; §§3–7 stand._

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
- `pytest` is clean. **30 tests currently fail**, all in `test_reporting_age_policy.py`, and all of them are this staleness: they read the persisted CSVs and check them against the current policy, so they pass once the output is regenerated at 72. If any survives the refits it is a real defect and not this.
  - A further 9 failures in `test_reported_age_range.py` were **not** staleness and are already fixed on this branch. That test pins the cap by name — `test_comprehension_reporting_stops_at_84_months` asserted `== 84` — so it is a specification test that had to move with the policy, not an artefact of stale output. It was missed on the first pass because it failed alongside the 30 and was counted with them; CI caught it, because the staleness tests are `@needs_fit` and skip on a runner with no fitted models while the pinned test does not. Worth remembering as the general shape: **a green local run and a red CI run can disagree here in either direction**, since CI sees the specification and not the output.
- Spot-check that comprehension figures and tables now stop at 72 and that **spoken still runs to 90** and **signed to 84** — the whole point of the per-quantity policy is that they move independently.

## 7. Decisions that could be folded into the same session

Both are definition changes, so bundling them costs nothing extra in refits and avoids fragmenting the staleness across further commits. Both are the study owner's call.

1. **Register the low-rank factor successor** to VG19/VG20, per [202608221000](202608221000-four-by-four-gate1.md) §5 — `b = L z` with `k` a definition field and `k = 1, 2, 3` as a registered sensitivity family. Note the free 4x4 is **not** what to build; that note explains why.
2. **Register VG21**, the `window-22` promotion, whose prior gate passed on 2026-08-21 ([202608211545](202608211545-window-22-prior-gate-passed.md)) and which has been unblocked since.
