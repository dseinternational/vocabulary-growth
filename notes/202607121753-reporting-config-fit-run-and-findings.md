# Full reporting-config fit run of VG01–VG16 (810 scale) — run record and findings

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> Live run record started 2026-07-12 ~17:53 UTC while the run was still in
> progress (TD tail + two convergence refits sampling). The convergence and
> methodological findings below are settled; the final per-model results table
> and wall-clock timings are completed once the run drains.

## Purpose

Fit every registered model (VG01–VG16) at the `rep` (reporting) sampling
configuration on the corrected **810-item** reference scale, render all reports,
and produce a results summary. This is the deferred refit work for
[#149](https://github.com/dseinternational/vocabulary-growth/issues/149) (the
810-scale correction from #148 was code+docs only).

## Environment and method

- VM: 32 cores, 251 GB RAM, no GPU, 410 GB scratch. Output root redirected to
  `/scratch/vg-output` to keep multi-GB traces off the checkout.
- `rep` config: 6 chains / 6 cores / 6000 tune / 6000 draws / `target_accept` 0.95.
- Parallelism: **DS models fitted concurrently** (5-slot pool, 30 of 32 cores);
  **TD models strictly sequential** (vg03 → vg04 → vg13 → vg11 → vg12) to avoid
  OOM on the full-data TD fits. Driver: `/scratch/vg-output/run_suite.sh`.
- Data confirmed current (merged DuckDB newer than all source CSVs; #148's 810
  correction present); no re-prep needed.

## Incidents and fixes

### 1. DuckDB lock conflict on concurrent DS fits (fixed)

The first wave of 5 concurrent DS fits collided opening the shared
`data/vocabulary.duckdb`: `load_combined_data()` used a default **read-write**
connection (exclusive lock), so vg02/vg05/vg07/vg10 died in ~2 s with
`IOException: Conflicting lock`. Both connect sites in
`src/vocab_growth/data_utils.py` are read-only `SELECT`s, so they were changed to
`duckdb.connect(..., read_only=True)` (the documented DuckDB concurrent-reader
pattern). The four failed models were re-fitted cleanly. **Working-tree change**,
flagged for review.

### 2. Convergence gate rounded R-hat to 2 s.f. (research issue #65; fixed)

The shared gate `write_diagnostics_summary` called `az.summary(..., round_to=None)`
believing it disabled rounding. In arviz-stats 1.2.0 `round_to=None` falls through
to `rcParams["stats.round_to"] = "2g"` (2 significant figures); only the string
`"none"` disables it. So `max_rhat`/`min_ess` and the `rhat_failing`/`ess_failing`
lists were gated on **rounded** values — a true R-hat up to ~1.045 rounds to `1.0`
and passes the ≤1.01 gate. Confirmed on our stack (same data: `None`→1.0,
`"none"`→1.0016).

Fixes: patched the installed gate to `round_to="none"` (env-local; identical to the
official fix); bumped `environment.yml` to `dse-research-utils` **v0.6.0**, which
ships exactly this fix (`round_to="none"`) and leaves the compiled core + sampling
config unchanged. Every fit launched after the patch records true diagnostics
natively; the six fitted before it were audited post-hoc from their saved traces
(`/scratch/vg-output/verify_diagnostics.py`).

## Finding A — real (masked) convergence failures on the understood-GP block

Recomputing true (unrounded) R-hat over the posterior revealed that the rounded
gate had **falsely passed** several DS joint/hierarchical models. All failures are
the **understood-trajectory GP block** (`g_u` / `g_unit_u` / `g_unit_u_hsgp_coeffs`
/ `slope_u` / `p_slope_low_u`) — the trend/GP/intercept redundancy the VG10 GP
anchor addresses for the _q_-GP, here on the _understood_ GP.

| Model | gate R-hat (rounded) | true R-hat (`rep`) | true min ESS | verdict                          |
| ----- | -------------------- | ------------------ | ------------ | -------------------------------- |
| vg01  | 1.0                  | 1.00120            | 5856         | PASS (genuine)                   |
| vg08  | 1.0                  | 1.00973            | 704          | PASS (borderline)                |
| vg14  | 1.0                  | 1.00262            | 3698         | PASS (genuine)                   |
| vg09  | 1.0                  | **1.01385**        | 417          | FALSE PASS → refit               |
| vg15  | 1.0                  | **1.01072**        | 600          | FALSE PASS → refit               |
| vg16  | 1.0                  | **1.02417**        | 379          | FALSE PASS (R-hat + ESS) → refit |
| vg10  | 1.0 (native)         | **1.01131**        | 640          | FAIL → refit                     |

### Remediation — `rep-hightune` refits (user-requested)

Refit the four failing models with heavier tuning (tune 6000→**12000**, draws
6000→**8000**, `target_accept` 0.95→**0.97**, 6 chains) via a monkey-patched
sampling config (`refit_hightune.py`, no source edits). Non-converged originals
preserved under `/scratch/vg-output/preconv-backup/`; converged refits become the
models of record.

| Model | `rep` true R-hat | `rep-hightune` R-hat | result    |
| ----- | ---------------- | -------------------- | --------- |
| vg09  | 1.01385          | **1.00643**          | CONVERGED |
| vg16  | 1.02417          | **1.00889**          | CONVERGED |
| vg15  | 1.01072          | _(refit sampling)_   | pending   |
| vg10  | 1.01131          | _(refit sampling)_   | pending   |

## Finding B — fixed-810 denominator is VALIDATED (not a bias); per-form over-corrects

Investigating [#149](https://github.com/dseinternational/vocabulary-growth/issues/149)
step 3 (per-row-denominator sensitivity), an initial VG02 experiment
(fixed-810 vs per-form `n = survey_vocab_max`) showed a large trajectory shift
(~18 pp), which I first misread as fixed-810 being biased. **That was wrong**, and
is corrected here: `notes/202607121200-statistical-model-review.md` §3A had already
investigated and defused this.

- The CDI forms are **nested**: short forms are the earlier/common words; the items
  absent from a short form are the rarer words an ability-matched child mostly does
  not know. So a short-form count `k` entered as `k/810` is close to the child's true
  full-inventory fraction — not a length-scaled deflation.
- Empirical proof (reproduced): uk_02 children given **both** the DSE-810 and
  Oxford-416 forms at the same age score about the **same count on each** — median
  ratio **1.136** (n=10 concurrent). The full crosswalk
  (`scripts/crosswalk_dse_oxford.py`, re-run: max R-hat 1.000, min ESS 1953) gives
  age-adjusted R = 0.85 / 0.96 / 1.10 / 1.25 / 1.40 at ages 25–49, and
  **P(R < 1.95) = 1.00**. Fixed-810 implies R ≈ 1; per-form implies R = 810/416 = 1.95,
  which is excluded everywhere.
- So the per-form denominator is the **over-correction** (the VG02 ~18 pp shift was
  that artifact, not a bias in fixed-810). Study REs cannot help either way; the
  small residual is near zero at the young ages where short forms are used and mild
  only in the non-hierarchical baselines (VG01/VG02/VG05/VG14).

**Outcome:** fixed-810 retained as a _validated_ approximation. No re-fit, no
likelihood change. `docs/report/methods-data.qmd` (§Measures) updated to state the
finding (replacing the "sensitivity follow-up" placeholder); #149 closes on this run
(step 4 doc), with the earlier "adopt per-row" issue comment retracted.

**Process lesson:** consult `notes/` (the existing model review) _before_ running a
new experiment — the answer was already on record.

## Final results

Run window: 2026-07-12 13:04 → 2026-07-13 10:47 UTC (~21 h 43 m driver wall: DS pool +
sequential TD), plus the convergence refits overlapping. Longest single fit vg11 9 h 17 m
(16,235 obs). Total trace footprint 187 GB (redirected to scratch).

Convergence of the models of record (true, unrounded diagnostics):

| Model | population / outcome | max R-hat | min ESS | div | verdict                    | config                 |
| ----- | -------------------- | --------- | ------- | --- | -------------------------- | ---------------------- |
| vg01  | DS spoken            | 1.00120   | 5856    | 0   | PASS                       | rep                    |
| vg02  | DS understood        | 1.00072   | 8536    | 0   | PASS                       | rep                    |
| vg03  | TD spoken            | 1.00044   | 8647    | 0   | PASS                       | rep                    |
| vg04  | TD understood        | 1.00069   | 10416   | 0   | PASS                       | rep                    |
| vg05  | DS joint             | 1.00104   | 5594    | 0   | PASS                       | rep                    |
| vg07  | DS joint + study RE  | 1.00209   | 4429    | 0   | PASS                       | rep                    |
| vg08  | DS joint + subj RE   | 1.00973   | 704     | 0   | PASS (borderline)          | rep                    |
| vg09  | DS joint + q RE      | 1.00643   | 1169    | 0   | PASS                       | rep-hightune           |
| vg10  | DS anchored joint    | 1.00563   | 1630    | 0   | PASS                       | rep-hightune           |
| vg11  | TD spoken (full)     | 1.00041   | 9851    | 1   | **KEEP (1 div, accepted)** | rep                    |
| vg12  | TD understood (full) | 1.00093   | 10619   | 0   | PASS                       | rep                    |
| vg13  | TD joint (young)     | 1.00225   | 2308    | 0   | PASS                       | rep-hightune (ta 0.99) |
| vg14  | DS + signing         | 1.00262   | 3698    | 0   | PASS                       | rep                    |
| vg15  | DS sign–speech       | 1.00339   | 1691    | 0   | PASS                       | rep-hightune           |
| vg16  | DS cross-lag         | 1.00889   | 957     | 0   | PASS                       | rep-hightune           |

**14 / 15 PASS.** Only **vg11** fails the strict 0-divergence gate — by **one**
divergence, with R-hat 1.00041 / ESS 9851 otherwise excellent; not auto-refit (~10 h for
1 div). **Decision (2026-07-13): KEEP** — accepted with caveat. A single divergence
against 16,000 post-warmup draws, with R-hat and ESS far inside the gates, does not
materially threaten the posterior; the ~10 h cost of a higher-`target_accept` refit is
not warranted. This `rep` fit is the model of record.
Five models needed `rep-hightune` refits: vg09/vg10/vg15/vg16 (understood-GP R-hat ridge),
vg13 (27→0 divergences at target_accept 0.99). Originals preserved under `preconv-backup/`.

**Render:** all 15 model reports produced valid `index.html`; the per-model quarto exit
was non-zero only from the `code-links: [repo]` post-processor (the scratch output dir is
outside the git checkout — cosmetic; clean in the normal in-repo `output/` workflow).
Consolidated `docs/report` rendered clean (all chapters). `docs/comparison` needed the
comparison artefacts staged into its own dir first (`cp <out>/comparisons/* docs/comparison/`,
gitignored) — `sync_report_figures` only targets `docs/report/figures/`, a workflow gap;
after staging it rendered exit 0.

**Side analyses (exploratory, not in the registry):** VG17 (spoken by sign-group) — null;
VG18 (total expressive `produced` by sign-group) — full-data OR 1.65 was an artifact of a
heterogeneous `produced` definition; the clean de-dup-union studies (uk_01/uk_02/nz_01)
give signer-vs-non-signer **OR 0.93, null**. uk_01 `produced` confirmed a de-dup union
(`signed` = signed-only); cross-study `signed`-definition inconsistency flagged for
VG14/VG15 (task). vg11 `rep-lite` vs `rep` validation **done (2026-07-13): validated** —
trajectories agree to ≤ 0.27 words (≤ 0.11%) across 9–30 mo, HDI widths unchanged (ratio
0.99), `rep-lite` min ESS 4,461 (0 divergences), wall time 5 h 59 m vs 9 h 17 m (~35%
saving). Runbook now recommends `rep-lite` for the big-data TD models.

## Follow-ups

- [#147](https://github.com/dseinternational/vocabulary-growth/issues/147) — Target 8
  young-age anchor prior-sensitivity checks, **`test` tier** (baselines re-fit at
  `test` to match the variants), vg10/vg11/vg12/vg15; priority `vg12 hi-anchor-broad`.
  To run after this run is written up.
- If a denominator correction is ever wanted (it is not, per Finding B), the
  instrument is the §3A age-dependent per-form _scaling_ crosswalk, not per-form
  `n_trials`.

## Reproduction / artefacts

- Driver + logs: `/scratch/vg-output/run_suite.sh`, `/scratch/vg-output/logs/`,
  `/scratch/vg-output/status.txt`; full run log `/scratch/vg-output/RUN_LOG.md`.
- Convergence audit: `/scratch/vg-output/verify_diagnostics.py` (writes
  `true_diagnostics.json` per model).
- Refit wrappers: `/scratch/vg-output/refit_hightune.py`,
  `/scratch/vg-output/refit_vg09_hightune.py`.
- #149 per-row experiment: `/scratch/vg-output/perrow_denominator_check.py`;
  crosswalk `scripts/crosswalk_dse_oxford.py`.
- Traces/figures under `/scratch/vg-output/models/<MODEL>/`.
