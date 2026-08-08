# Trace persistence tiers: cutting artifact size without touching inference

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

## 1. Why now

A full reporting output tree is ~257 GB, and ~256 GB of that is `.nc` traces: the figures, tables, diagnostics and rendered HTML together come to ~0.65 GB (1,251 PNG + 1,188 SVG + 546 CSV + assets). Individual rep traces run to VG11 53.6 GB, VG13 40.6 GB, VG12 21.3 GB, VG10 9.8 GB. This is not merely inconvenient: VG12's recovery replicate r08 died with `ENOSPC` on the VM's `/scratch` on 2026-08-06 partway through writing its own trace, and re-downloading the tree on 2026-08-08 took a 926 GB workstation to 94% full.

The alternative lever — subsampling the typically-developing pool — was considered and rejected in [`202608050900-td-hierarchical-geometry.md`](202608050900-td-hierarchical-geometry.md) §10. Within-child replication is the binding constraint on the TD hierarchical geometry, and any subsampling removes repeat-measured children; a row-wise draw reproduces the pathology outright. That note set the order of resort: plain `rep` alone; then stop storing the observation-sized deterministics, which are recomputable functions of the parameters and cost nothing statistically; and only then subsample. This note scopes the middle step.

## 2. What is actually in a trace

Measured directly from two completed reporting traces (`h5py`, uncompressed sizes).

**VG03** (TD spoken, no random effects) — 10.94 GB on disk, 4,075 observations, 6 chains × 6,000 draws:

| GB   | shape           | variable                         | class                                 |
| ---- | --------------- | -------------------------------- | ------------------------------------- |
| 1.23 | (6, 6000, 4583) | `f_all`                          | concatenated obs+plot+query predictor |
| 1.23 | (6, 6000, 4583) | `g`                              | GP, derived                           |
| 1.23 | (6, 6000, 4583) | `g_unit`                         | GP, raw                               |
| 1.09 | (6, 6000, 4075) | `f_obs`                          | obs deterministic                     |
| 1.09 | (6, 6000, 4075) | `kappa_obs`                      | obs deterministic                     |
| 1.09 | (6, 6000, 4075) | `p_obs`                          | obs deterministic                     |
| 1.09 | (6, 6000, 4075) | `posterior_predictive/y_obs`     | predictive                            |
| 1.09 | (6, 6000, 4075) | `log_likelihood/y_obs`           | likelihood                            |
| ~0.1 | —               | free parameters + `sample_stats` | **irreplaceable**                     |

**VG15** (DS joint sign+speech, subject REs on three trajectories) — 6.24 GB, 1,349 observations, 737 subjects:

| GB    | shape               | variable                                            | class                                      |
| ----- | ------------------- | --------------------------------------------------- | ------------------------------------------ |
| 1.50  | (6, 6000, 1864)     | `g_unit_u`, `g_unit_q`, `g_unit_sign`               | GP, raw                                    |
| 0.72  | (6, 6000, 1349)     | `kappa_sign_obs`, `z_obs`                           | obs deterministic                          |
| 1.20  | (6, 6000, 737)      | `delta_subj_{u,q,sign}` **and** `z_subj_{u,q,sign}` | subject RE, **scaled and raw both stored** |
| 1.12  | (6, 6000, 905/1179) | `log_likelihood` + `posterior_predictive`           |                                            |
| 0.101 | —                   | everything under 0.02 GB, i.e. the free parameters  | **irreplaceable**                          |

Three conclusions:

1. **The parameters are a rounding error.** ~0.1 GB in both models. Everything else is a deterministic function of them evaluated at observation, grid, or subject index.
2. **Subject-sized arrays matter too, and are stored twice.** `delta_subj_u = tau_subj_u * z_subj_u` is persisted alongside `z_subj_u`. That is the same information twice, 1.20 GB on a model with only 737 children. VG11 has 1,947 repeatedly-measured children out of a much larger pool across 18,522 observations, so its subject-sized block will be far bigger — VG03 has no random effects and therefore understates the VG11 profile.
3. **VG03 is the cheap case.** It is a no-RE model at 4,075 observations. VG11 is 18,522 observations with subject REs, which is most of why it is 53.6 GB.

## 3. Consumer audit

Who actually reads these variables back off disk, checked across `src/`, `scripts/` and `docs/`:

| Consumer                             | Reads                                                                                   | Affected?                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `docs/models/*/index.qmd`            | `chain`/`draw` **dimension sizes only**, via `h5netcdf` header                          | No                                                      |
| `vocab_growth.comparison`            | plot-grid vars (`p_plot`, `f_plot`, `kappa_plot`, `f_u_plot`, `h_plot`) + scalar scales | No                                                      |
| `recovery/simulate.py`               | free RV names                                                                           | No                                                      |
| `fit_artifacts.validate_fit`         | existence of `trace.nc` only — no variable-level checks                                 | No                                                      |
| `scripts/kfold_loso.py`              | `p_u_obs`, `p_s_obs`, `q_obs`                                                           | **Yes**                                                 |
| `scripts/loso_compare.py`            | `f_u_obs`, `h_obs`                                                                      | **Yes**                                                 |
| `extract_model_samples` (in-process) | `f_obs`, `f_plot`                                                                       | No — reads the in-memory trace, and runs after the save |

So exactly two post-hoc consumers depend on observation-sized posterior variables, both cross-validation tools. Everything on the reporting path — render, comparison, recovery, validation — is indifferent.

Note that the newly added between-child estimand (`comparison.subject_heterogeneity`) reads only the scalar subject scales `tau_subject` / `tau_subj_u` / `tau_subj_q`, never the per-child arrays. The heterogeneity contrast is therefore compatible with dropping `delta_subj_*` entirely.

## 4. Design

**Tiers**, selected at fit time, applied only at save:

- `full` — today's behaviour. Default initially.
- `compact` — drop observation-sized posterior deterministics and the redundant scaled random effects. Keeps `log_likelihood` and `posterior_predictive`.
- `minimal` — additionally drop `log_likelihood` and `posterior_predictive/y_*_obs`.

Expected effect on VG03: `compact` ≈ 10.9 → 3.9 GB (−64%), `minimal` ≈ 1.7 GB (−84%). On VG11, proportionally more, because its obs and subject blocks dominate a larger share.

**Select by dimension, not by name.** A hardcoded name list would need maintaining across six engine modules and every future model. The rule should be: drop posterior variables whose dims intersect `{obs_id}`, plus the concatenated `*_all` grids, plus scaled random-effect deterministics whose raw counterpart and scale are both retained. Keep an explicit allowlist for anything grid-shaped (`*_plot`, `*_query`) that the reporting path needs. Dimension-based selection is robust to new models in a way a name list is not.

**Where the setting lives — not in the model definition.** This is the one design point that must not be got wrong. `ModelDefinition` fields are part of the model graph and its fingerprint: `td_languages` is documented as a definition field precisely because changing it requires a refit. Trace persistence changes nothing about the posterior, so putting it in the definition would invalidate every fitted model for a storage decision. It belongs on the CLI (`fit_model.py --trace-persistence=`) with an environment-variable default, alongside `--output-dir`, and must be excluded from the fingerprint.

**Record it in the manifest.** `fit_manifest.json` should carry the tier and the list of dropped variable names, so a later reader can tell "absent by policy" from "corrupt or truncated trace" — the failure mode we have already seen once this week.

## 5. Risks and guardrails

- **The two LOSO scripts must fail loudly.** On a compacted trace they would currently raise a bare `KeyError` on `p_u_obs`. They need an explicit check that reads the manifest tier and says "this fit was saved at `compact`; refit at `full` to run leave-one-study-out".
- **`log_likelihood` and `posterior_predictive` are a genuine trade, unlike the deterministics.** LOO and the PPC plots are computed during the fit and their outputs persisted as CSVs and figures, so `minimal` costs nothing for the _current_ report — but it forecloses recomputing LOO, running new model comparison, or plotting a new predictive view without a refit. `compact` is free; `minimal` is a decision.
- **Dropping `delta_subj_*` forecloses per-child prediction.** The reported estimands (τ, and the derived σ_child) need only the scales. Anything wanting a specific child's intercept would need `full`.
- **Do not mutate the in-memory trace.** Later pipeline stages — `extract_model_samples` reads `f_obs` — run against `context.trace` after the save call. The filter must build a copy.
- **Compression is a separate, orthogonal lever.** These are stored uncompressed; netCDF4/HDF5 chunked deflate on the large float arrays is worth measuring independently, and would help `full` too.

## 6. Files touched

- **Prerequisite refactor.** Six modules each call `trace.to_netcdf(...)` directly: `common.py`, `common_bivariate.py`, `common_joint_modality.py`, `common_trivariate.py`, `common_univariate_re.py`, `model_vg17.py`. (`common_bivariate_re.py` reuses `common_bivariate`'s.) These should route through one helper — otherwise the policy has to be repeated six times and will drift.
- New: the tier enum, the dimension-based filter, and its unit tests.
- `scripts/fit_model.py`: flag, env-var default, plumbed to the engines.
- `fit_artifacts.py`: manifest field; keep `validate_fit` indifferent to it.
- `scripts/kfold_loso.py`, `scripts/loso_compare.py`: tier guard with a clear message.
- Docs: `CLAUDE.md` / `AGENTS.md` / `.github/copilot-instructions.md` command section, and the full-refit runbook.

Roughly a day, with the shared-save refactor the bulk of it and the highest-risk part, since it touches every engine's fit path.

## 7. Rollout

Ship with `full` as the default so nothing about existing artifacts or reproduction changes. Fit one model at `compact` and one at `minimal`, and confirm byte-for-byte that the reporting path — `--render-only`, `sync_report_figures.py`, the comparison suite — produces identical output to the `full` fit. Flip the default at the next full refit, not before.

## 8. Reducing draws: the other half of `n_draws × n_obs`

Raised 2026-08-08 alongside this note. Trace size is linear in draws, so cutting them shrinks everything — including the ~0.1 GB of parameters that §2 calls irreplaceable — and speeds every downstream load. Unlike §4 it is not free: it costs precision in every reported interval. The question is whether there is headroom.

**There is, but it is thin, and it is thinnest where it would help most.** Effective sample sizes at the current `rep` (6 chains × 6,000 = 36,000 draws), from each model's `diagnostics.csv`:

| model    | worst `ess_bulk` | worst `ess_tail` | efficiency | worst parameters                |
| -------- | ---------------- | ---------------- | ---------- | ------------------------------- |
| **VG11** | **1,008**        | 1,718            | **2.8%**   | `eta`, `ell_unit`, `ell`        |
| VG13     | 1,432            | 2,556            | 4.0%       | `tau_u`, `tau_q`                |
| VG10     | 2,819            | 2,456            | 7.8%       | `kappa_min_s`, `a_kappa_s`      |
| VG12     | 2,901            | 4,504            | 8.1%       | `eta`, `subject_variance_share` |

Nothing sits below 400. But the binding constraint is VG11's GP hyperparameters at 2.8% efficiency, and VG11 is the 53.6 GB trace — the model most worth shrinking has the least room to shrink. Scaling ESS roughly linearly in draws:

| config            | draws  | VG11 worst `ess_bulk` | verdict          |
| ----------------- | ------ | --------------------- | ---------------- |
| `rep` 6×6000      | 36,000 | 1,008                 | current          |
| 6×4000            | 24,000 | ~672                  | comfortable      |
| 6×3000            | 18,000 | ~504                  | marginal         |
| `rep-lite` 4×4000 | 16,000 | ~448                  | at the 400 floor |
| 6×1500            | 9,000  | ~252                  | below the floor  |

**The sensitivity test is nearly free, and it is exact rather than approximate.** Truncating each chain to its first _N_ post-warmup draws reproduces exactly what a shorter sampling run would have produced: post-warmup NUTS draws are a deterministic function of the RNG stream, so holding warmup and adaptation fixed and cutting sampling iterations does not change the first _N_. What 6×3000 would have given is already sitting in the existing traces. This is a stronger guarantee than a refit-and-compare, which would confound the draw count with a different random seed.

Note the contrast with thinning by stride: taking every _k_-th draw estimates the information content of a thinned chain, which is **not** what a shorter run produces. Truncation is the correct operation here, and the exactness depends on it.

The test:

1. For each model and each candidate _N_, truncate chains to the first _N_ draws.
2. Re-run the existing convergence gate — R-hat, ESS bulk/tail, divergences, energy BFMI. Does it still pass?
3. Recompute every reported estimand and every DS/TD contrast; compare against the full-draw values relative to MCSE and to reported interval width.
4. Report the smallest _N_ at which all models clear the gate and no headline moves by more than a stated fraction of its interval.

No sampling at all. The cost is I/O — reading the traces once — plus perhaps half a day to write it against the existing diagnostics code, and it should reuse the gate rather than reimplement it.

**The one caveat, and it bites immediately.** Exactness holds only if warmup is unchanged — and **no existing named configuration isolates the draw count.** `tune` scales with `draws` in every one: `dev` 500/500 (2 chains), `test` 2000/2000 (4), `rep` 6000/6000 (6), `rep-lite` 4000/4000 (4). So a `rep` → `rep-lite` comparison changes chains, tuning and draws together, so truncation cannot reproduce it; only a "`rep` tuning, fewer draws" shape can be reproduced that way, and no such config exists today. The test therefore needs one new config shape, and any recommendation from it must state the warmup it assumes. Comparing a genuinely shorter _warmup_ still requires a refit.

### 8.1 Per-model draws, or overrides for a few models?

Raised 2026-08-08: if efficiency varies this much between models, should draws be set per model at each of `dev` / `test` / `rep`, rather than one blanket level?

**Overrides for specific models, yes; a full per-model matrix, no** — and the number should be derived from an ESS target rather than chosen. Against a ~1,000 working target for the worst parameter (2.5× the gate's 400 floor, since 400 is the minimum for R-hat reliability, not for stable 89% interval bounds):

| model    | worst `ess_bulk` at 36,000 | draws for ESS ≈ 1,000 | trace          |
| -------- | -------------------------- | --------------------- | -------------- |
| **VG11** | 1,008                      | **~35,700 — no room** | 53.6 GB        |
| VG13     | 1,432                      | ~25,100 (70%)         | 40.6 → ~28 GB  |
| VG10     | 2,819                      | ~12,800 (35%)         | 9.8 → ~3.5 GB  |
| VG12     | 2,901                      | ~12,400 (34%)         | 21.3 → ~7.3 GB |

**It saves least where it matters most.** Equalising ESS takes those four from ~125 GB to ~93 GB, a 26% cut, while VG11 — 43% of the total on its own — is untouched or needs _more_. The worst-mixing model is also the largest. So per-model draws is a defensible way to equalise inferential precision across the model set; it is a weak lever for artifact size, where §4 still dominates.

It is also aimed at the wrong parameter. VG11's 2.8% efficiency is a geometry symptom — the worst movers are `eta`, `ell` and `ell_unit`, the GP hyperparameters — and its 22 divergences, like VG12's and VG13's BFMI failures, point at `target_accept` and `tune` rather than at draw count. If per-model overrides are introduced, those are the parameters more likely to earn one; spending draws instead institutionalises a workaround for something reparameterisation or a tighter length-scale prior should address, exactly as the variance partition was handled.

Four mechanical constraints, all verified against the current code:

1. **The configurations are shared.** They come from `dse_research_utils.statistics.models.sampling` and are used across DSE research repos, so per-model overrides must live in this repository rather than by editing those definitions.
2. **The model reports would mark every override "approximate".** `docs/models/*/index.qmd` maps `(chains, draws)` to a configuration name, with `(6, 6000)` alone meaning reporting-grade. Any override renders as a bare "N chains × M draws" flagged approximate — which is already why VG08 and VG09, fitted at 6×8000, misreport. The companion change is to read the configuration name from `fit_manifest.json`, which records it, instead of inferring it from trace dimensions.
3. **Validation has to learn the override too.** `validate_fit_output` compares `expected_sampling_parameters` against the manifest; an override that feeds the fit but not the check makes every overridden fit fail `sync_report_figures.py`.
4. **Overrides already happen, unmanaged.** VG08 and VG09 at 6×8000 match no named configuration. The question is not whether exceptions exist but whether they are explicit, validated and visible in the report.

Sequence: run the §8 truncation test at fixed `tune = 6000` to establish each model's actual draw requirement; adopt §4 for storage regardless of the outcome; then add overrides only where the test shows a model over- or under-sampled, with manifest-based configuration detection as a prerequisite for the report not to misdescribe them.

**Order of resort.** For both stated goals — smaller artifacts and faster analysis — draws is the weaker lever. On VG11, 6×4000 gives 53.6 → ~36 GB while §4's `compact` gives roughly 10 GB, because the deterministics dominate. The persistence change is free and dominates on both axes; draws costs precision in every interval. Take §4 first, and treat draws as a separate decision on its own evidence. The two compose: 6×4000 plus `compact` is roughly −76%.

## 9. What this does not solve

Trace size scales with `n_draws × n_obs`, and §4 changes neither. It buys roughly a 3–6× reduction in stored bytes; it does not make VG11 a small model, and it does not touch the underlying geometry problem that §10 of the hierarchical-geometry note describes. It is a storage fix, deliberately chosen because it is the one lever in this area with **zero statistical cost**.
