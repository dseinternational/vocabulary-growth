# Test-config refit of VG01–VG16 after the data and prior changes — run record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!WARNING]
> Run record, 2026-08-03. **Nothing in this run is reporting quality.** The `test` configuration is 4 chains x 2,000 draws (tune 2,000, target accept 0.90) against the reporting configuration's 6 x 6,000 at 0.95, and no fit here has been promoted to the report or uploaded. Its purpose is to validate two data changes and one prior recalibration against each other for the first time, and to give the reporting-quality run a set of expectations to be checked against.

## 1. Why this run

Three changes have landed since the last complete set of fits, and none of them has been fitted alongside the others.

1. **The `us_01` (Edgin) source was rebuilt from the item-level contributor files** ([202608031500a](202608031500-edgin-out-of-window-administrations.md)). The Down syndrome pool goes from 1,404 to 1,439 rows and its subject index moves, so every Down syndrome fit is stale — not merely out of date, but keyed to a subject set that no longer exists.
2. **The typically-developing reference pool was widened to Italian and Spanish (European)** ([202608031500b](202608031500-td-romance-extension.md)). VG11, VG12 and VG13 draw on `ENGLISH_AND_ROMANCE_LANGUAGES`; VG03 and VG04 stay English-only.
3. **The dispersion and subject random-effect priors were recalibrated** ([202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §§18–23). Eleven models moved to the two-anchor `kappa` parameterisation and every subject random-effect scale in the registry went from `HalfNormal(0.5)` to `HalfNormal(1.5)`.

The recalibration's own validation fits (§§18–23) were made on the **pre-change data**. The data changes were shipped without refitting. So the current position is that every recorded estimate in the project is stale for one reason or the other, and no fit exists in which both are true at once. That is what this run establishes.

It also discharges the largest item on §23's open list. "Rerun `scripts/prior_predictive_audit.py` for the whole family … the largest single piece of unfinished validation and should come before any reporting-quality run." Prior-predictive checks are a stage of every `fit()`, so a complete family refit regenerates the audit for all fifteen models as a by-product; the standalone script was not needed.

## 2. Configuration

|                        |                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------- |
| Sampling configuration | `test` — 4 chains, 2,000 tune, 2,000 draws, target accept 0.90, seed 47          |
| Models                 | all fifteen registered (VG01–VG16, retired VG06 omitted)                         |
| Model definitions      | as registered at `245e524`; **no overrides of any kind**                         |
| Output root            | repository-local `output/`                                                       |
| Data                   | `data/vocabulary.duckdb` rebuilt from the source CSVs immediately before the run |
| Host                   | 16-core arm64 (Apple silicon), 48 GB                                             |
| Concurrency            | two serial workers in parallel, 4 sampling cores each                            |

**Every model was fitted at its registered `sample_fraction`.** That means VG11, VG12 and VG13 ran on their **full** pools, not a subsample. `sample_fraction` is a model-definition field and therefore part of the fit's identity: a subsampled `test` fit would not validate against the full-pool reporting definition, and so could not serve as preparation for it. VG03 and VG04 keep their registered `sample_fraction=0.25`, which is a property of those models rather than a concession to this run.

Each model was fitted by its own `fit_model.py … --render` invocation rather than through `fit_model.py all --render`, because the batch form renders only after every fit in the batch has finished. Per-model invocation produces each report as its own fit completes.

### De-risking

Two `dev`-config smoke fits to a scratch output root first, chosen to cover the two new data paths rather than to sample the family: **VG02** (Down syndrome understood — the rebuilt Edgin rows) at 1m 26s, and **VG13** (young typically-developing joint, hierarchical — the widened language scope, the narrowest age window, and the model most exposed to a Gaussian-process domain violation) at 8m 44s. Both cleared the full pipeline. Smoke output discarded.

This was worth doing: the Romance extension had already produced one domain violation during implementation, when Italian Words & Gestures administrations registered from 7 months fell outside `_TD_GP_DOMAIN_MONTHS`. That was fixed by bounding the pool (`TD_POOL_AGE_MONTHS`) rather than widening the shared domain, and the smoke fit confirms the bound holds on the real frame.

### Deviation: rendering was retried out of band

The driver invoked the environment's interpreter by absolute path without putting the environment's `bin` directory on `PATH`. `fit_model.py` spawns `quarto render` as a subprocess, and Quarto resolves its own Jupyter kernel from `PATH`, so every in-run render executed against the _system_ framework python — which has no `h5netcdf` and therefore cannot open the trace the report cells load. All fifteen fits are unaffected: promotion precedes rendering, so each fit completed, passed its gate and promoted normally, and only the render step failed.

Rather than discard sampling already in flight, a watcher retried each render with the correct interpreter as soon as its model reached a terminal state, so reports still appeared per model as their fits landed (typically within 40 seconds). Renders take about 7 seconds each. The driver script now exports both `PATH` and `QUARTO_PYTHON`.

This is an operator error, not a repository defect — activating the environment would have avoided it — but it is a sharp edge worth removing: `_render_output` could pin `QUARTO_PYTHON` to `sys.executable` so a report is always rendered by the interpreter that produced the fit. Filed separately; deliberately not changed mid-run, so that every fit in this record comes from an unmodified `245e524`.

## 3. What the language scope did to the study random effect

Measured before the run, on the rebuilt database. The engines drop studies below 200 observations before building the study random effect, so the counts that matter are the retained ones.

| outcome    | scope             |   rows | subjects | studies | retained (>= 200 obs) |
| ---------- | ----------------- | -----: | -------: | ------: | --------------------: |
| spoken     | English only      | 16,550 |   12,488 |      12 |                     7 |
| spoken     | English + Romance | 18,837 |   14,775 |      15 |                **10** |
| understood | English only      |  6,133 |    4,899 |       7 |                     4 |
| understood | English + Romance |  7,188 |    5,954 |       9 |                 **6** |

Newly retained groups: `Caselli` (Italian), `Karousou` (Spanish (European)) and — spoken only — `CLEX` (Italian).

**This is the run's most specific prediction, and it was not the reason the scope was widened.** §23 found the study random-effect scale to be the worst-mixing parameter in all three hierarchical typically-developing models and in none of the four Down syndrome ones, and attributed the difference to group count: four or seven groups against twelve. It named the remedies as structural — a prior justified by the group count, pooling across outcomes, or fixed study effects — and none of them has been implemented. The Romance extension, adopted for cross-linguistic defensibility, incidentally raises the group counts to ten and six. If §23's diagnosis is right, the study scales should mix better in this run with no change to their prior.

### A latent trap worth recording

The study random effect is keyed on Wordbank's `dataset_name`, and **`CLEX` is not a single-language dataset**. It is a cross-linguistic norming project spanning Croatian, Danish, French (Quebecois), Italian, Russian, Swedish and Turkish. Only its Italian rows are in scope here, so the retained `CLEX` group is single-language _in this analysis_ — but it is single-language by accident of the language filter, not by construction. A future scope widening that admitted Swedish or Danish would silently pool them with the Italian children into one study group. `Caselli` has the same shape: its Italian Words & Gestures rows are admitted and its American Sign Language `CDITwo` rows are not.

Nothing needs changing now. It should be checked whenever `ROMANCE_LANGUAGES` or `KNOWN_TD_LANGUAGES` grows.

## 4. Results

Prior CDF is evaluated at the posterior median; contraction is `1 - sd_post/sd_prior`. Neither is produced by the fit pipeline, so both are computed from `trace.nc` and the registered prior parameters. Medians are used rather than the means in `diagnostics.csv` because most of these parameters have LogNormal priors, where the prior CDF at the mean and at the median are materially different numbers.

### 4.1 Convergence

All fifteen models fitted, promoted and rendered. Run window 2026-08-03 13:41 → 15:00 UTC, about 1 h 19 m of wall time for the family.

The convergence gate is **advisory at `test`**: `enforce_convergence_gate` returns early for any non-reporting configuration, so the R-hat/ESS tier that is fail-closed at `rep` only records a verdict here. Every model below therefore promoted, including those with R-hat and ESS failures. `passed` is the gate's own clean/not-clean flag; `soft` lists the checks it marked failing.

| model | passed  | div | max R-hat | min ESS | min BFMI | failing checks | R-hat fails | ESS fails |    time |
| ----- | ------- | --: | --------: | ------: | -------: | -------------- | ----------: | --------: | ------: |
| VG01  | **yes** |   0 |    1.0041 |   1,102 |    0.859 | —              |           0 |         0 |  3m 12s |
| VG02  | no      |   5 |    1.0031 |   1,197 |    0.844 | div            |           0 |         0 |  2m 34s |
| VG03  | **yes** |   0 |    1.0017 |   2,611 |    0.887 | —              |           0 |         0 | 22m 21s |
| VG04  | no      |   2 |    1.0018 |   1,828 |    1.001 | div            |           0 |         0 |  6m 13s |
| VG05  | no      |  33 |    1.0076 |     420 |    0.845 | div            |           0 |         0 |  5m 52s |
| VG07  | no      |   4 |    1.0058 |   1,014 |    0.781 | div            |           0 |         0 |  5m 54s |
| VG08  | no      |   1 |    1.0217 |     208 |    0.423 | rhat, ess, div |           5 |         8 |  5m 07s |
| VG09  | no      |  23 |    1.0132 |     200 |    0.421 | rhat, ess, div |           2 |        11 |  5m 33s |
| VG10  | no      |   2 |    1.0177 |     387 |    0.446 | rhat, ess, div |           6 |         3 |  5m 35s |
| VG11  | no      |  25 |    1.0366 |     168 |    0.335 | rhat, ess, div |          16 |        12 | 49m 38s |
| VG12  | no      |  25 |    1.0059 |     555 |    0.188 | div, bfmi      |           0 |         0 | 17m 56s |
| VG13  | no      |   3 |    1.0061 |     372 |    0.229 | ess, div, bfmi |           0 |         2 | 24m 34s |
| VG14  | no      |   7 |    1.0048 |   1,241 |    0.860 | div            |           0 |         0 |  7m 54s |
| VG15  | no      |   2 |    1.0248 |     379 |    0.482 | rhat, ess, div |           6 |         1 |  9m 06s |
| VG16  | no      |   1 |    1.0113 |     402 |    0.433 | rhat, div      |           2 |         0 |  6m 09s |

**Two models pass the gate outright: VG01 and VG03.** That is one better than §23's position, where nothing in the family cleared both the divergence and R-hat requirements.

Timings are not comparable across models: two to three fits ran concurrently throughout, and the concurrency changed mid-run when the workers were re-sharded. They are recorded only for planning the reporting run. On that basis, the family at `test` is roughly 80 minutes of wall time at concurrency 2–3, and VG11 alone is 50 minutes of it.

### 4.2 The two-anchor dispersion prior holds on the rebuilt data

VG01 is the model this matters most for. Its `b_kappa_mag` was the thread running through §§10–18 of the recalibration note: censored at prior CDF 1.00 with contraction 0.07, and moving _further_ into the tail when the prior was widened — the observation that motivated the two-anchor reparameterisation in the first place. §18 validated the new form on the pre-change data. This is its first fit on the rebuilt Edgin pool.

| VG01                 | median | prior CDF | contraction | R-hat |    ESS |
| -------------------- | -----: | --------: | ----------: | ----: | -----: |
| `kappa_min`          |  3.021 |     0.503 |       0.940 | 1.000 |  9,532 |
| `kappa_excess_young` |  54.27 |     0.606 |       0.855 | 1.001 | 10,717 |
| `kappa_excess_old`   |  4.136 |     0.519 |       0.841 | 1.000 |  9,136 |

Every parameter inside the central quarter of its prior, with contraction above 0.84. The censoring is gone, and the 35 new Edgin rows did not disturb it.

All four univariate models now carry the two-anchor form, and all four come back centred. Every dispersion parameter in the group sits between prior CDF 0.17 and 0.75, and nine of the twelve between 0.46 and 0.63.

| model | parameter            | median | prior CDF | contraction |
| ----- | -------------------- | -----: | --------: | ----------: |
| VG01  | `kappa_min`          |  3.021 |     0.503 |       0.940 |
| VG01  | `kappa_excess_young` |  54.27 |     0.606 |       0.855 |
| VG01  | `kappa_excess_old`   |  4.136 |     0.519 |       0.841 |
| VG02  | `kappa_min`          |  1.409 |     0.173 |       0.861 |
| VG02  | `kappa_excess_young` |  12.31 |     0.556 |       0.924 |
| VG02  | `kappa_excess_old`   |  5.496 |     0.751 |       0.822 |
| VG03  | `kappa_min`          |  2.940 |     0.490 |       0.958 |
| VG03  | `kappa_excess_young` |  33.23 |     0.558 |       0.927 |
| VG03  | `kappa_excess_old`   |  3.158 |     0.529 |       0.899 |
| VG04  | `kappa_min`          |  3.924 |     0.632 |       0.465 |
| VG04  | `kappa_excess_young` |  7.459 |     0.489 |       0.724 |
| VG04  | `kappa_excess_old`   |  7.254 |     0.504 |       0.707 |

VG02's `kappa_min` at 0.173 is the one off-centre value, and it is the lowest in the family. That is §23's open item 5 — `kappa_min` meaning different things on different outcomes — appearing unchanged rather than as a new problem.

### 4.3 VG04 settles the sign-constraint question

This is the run's cleanest single result, and it needs no comparison with anything outside it.

§3 found that typically-developing comprehension dispersion is flat to slightly _rising_ with age, which the legacy form's `b_kappa_mag >= 0` cannot represent at all. §13 confirmed it on better evidence in the way that a constrained posterior does: given 2.5 times the data, VG04's `b_kappa_mag` moved **closer** to the boundary, 0.073 to 0.031 with a 5th percentile of 0.002. The note's reading was that VG04 was "answering 'as close to flat as you will let me'".

The two-anchor form has no sign constraint. Asked the question in a way that admits either answer, VG04 gives **7.459 at the young anchor and 7.254 at the old** — flat to within 3% across the age range, with both anchors at prior CDF 0.489 and 0.504. The boundary pile-up is gone because the boundary is gone, and what replaces it is the flat curve §3 predicted from the per-age series.

VG04 is also a **control on the whole run**. It is English-only, so neither data change touches it, and its frame came back at 1,555 rows — identical to the post-§17 baseline. Its trajectory parameters are accordingly unmoved: `p_slope_low` 0.0766 against §13's 0.087, `p_slope_hi` 0.442 against 0.416, `eta` 0.683 against 0.661. VG03 is the same kind of control at 4,075 rows, also matching, and also unmoved. Two frames provably unchanged, two sets of estimates that did not move: whatever else this run shows, it is not showing drift from the pipeline itself.

### 4.4 The four unmigrated models are worse than §22 recorded, and now measurably so

§23's open item 3 flagged VG05, VG07, VG08 and VG14 as still carrying the legacy `KappaPriorParams`, put their `b_kappa_mag_u` at prior CDF 0.993–0.9998, and called the position "a decision to revisit rather than a settled state". This run measures them on current data, alongside migrated models fitted in the same batch — which is the comparison the earlier note could not make.

| model | parameter       | median |  prior CDF | contraction |
| ----- | --------------- | -----: | ---------: | ----------: |
| VG05  | `kappa_min_u`   |  1.407 |  **0.017** |       0.894 |
| VG05  | `a_kappa_u`     |  1.555 |      0.300 |       0.883 |
| VG05  | `b_kappa_mag_u` |  0.768 |  **0.990** |       0.446 |
| VG05  | `kappa_min_s`   |  2.157 |  **0.081** |       0.947 |
| VG05  | `a_kappa_s`     |  0.412 |  **0.048** |       0.825 |
| VG05  | `b_kappa_mag_s` |  1.188 | **1.0000** |  **−0.109** |
| VG07  | `kappa_min_u`   |  1.512 |  **0.023** |       0.878 |
| VG07  | `a_kappa_u`     |  1.674 |      0.343 |       0.879 |
| VG07  | `b_kappa_mag_u` |  0.734 |  **0.986** |       0.447 |
| VG07  | `kappa_min_s`   |  2.191 |  **0.085** |       0.944 |
| VG07  | `a_kappa_s`     |  0.541 |  **0.062** |       0.832 |
| VG07  | `b_kappa_mag_s` |  1.222 | **1.0000** |  **−0.056** |

Five of six dispersion parameters in a prior tail, in both models, and the spoken slope has **negative contraction** in both — the posterior is _wider_ than the prior, which is what a parameter does when the prior is fighting it rather than informing it. Set against VG01's 0.503 / 0.606 / 0.519 from the same run and the same data-generating process, this is as clean a demonstration as the family admits that the difference is the parameterisation and not the pool.

All four unmigrated models behave identically. The spoken slope is at prior CDF 0.9999–1.0000 with **negative contraction in every one**:

| parameter                   |   VG05 |   VG07 |       VG08 |   VG14 |
| --------------------------- | -----: | -----: | ---------: | -----: |
| `b_kappa_mag_s` prior CDF   | 1.0000 | 1.0000 |     0.9999 | 0.9999 |
| `b_kappa_mag_s` contraction | −0.109 | −0.056 | **−0.165** | −0.089 |
| `b_kappa_mag_u` prior CDF   |  0.990 |  0.986 |      0.999 |  0.989 |
| `kappa_min_u` prior CDF     |  0.017 |  0.023 |      0.139 |  0.017 |
| `a_kappa_s` prior CDF       |  0.048 |  0.062 |      0.058 |  0.047 |

Four models, four independent fits, the same five parameters in the same tails. §23 recorded prior CDF 0.993–0.9998 for `b_kappa_mag_u` and negative contraction on the spoken slope; this run reproduces both on current data and puts a migrated comparison beside them in the same batch.

### The signed outcome is the family's one uncalibrated dispersion block — in both models that have one

This was not on any open list, and the run surfaces it because VG14 and VG15 were fitted alongside migrated models for the first time.

Both models carrying a signed outcome keep the **legacy** dispersion form on it. VG15 is a partially migrated model: its understood and spoken-ratio blocks are two-anchor, and its signed block is not. §23 records that `tau_subj_sign` has no calibration because nothing estimates a signing subject scale; the same is true of signed _dispersion_, and that was never stated.

| parameter                    |      VG14 |      VG15 |
| ---------------------------- | --------: | --------: |
| `kappa_min_sign` median      |     2.720 |     7.056 |
| `kappa_min_sign` prior CDF   |     0.155 |     0.717 |
| `kappa_min_sign` contraction |     0.821 | **0.180** |
| `a_kappa_sign` prior CDF     | **0.040** |     0.418 |
| `a_kappa_sign` contraction   |     0.455 |     0.358 |
| `b_kappa_mag_sign` median    |     0.064 |     0.076 |
| `b_kappa_mag_sign` prior CDF |     0.168 |     0.199 |

`b_kappa_mag_sign` at 0.064 and 0.076 is the **VG04 signature** — a slope collapsed towards the boundary in both models independently, which is what the constrained form does when signed dispersion is close to flat with age. VG04's version of this was resolved by migration, which removes the constraint as a side effect. Nothing has been done for the signed outcome, and the two models disagree by a factor of 2.6 on `kappa_min_sign` (2.72 against 7.06) with VG15's barely informed (contraction 0.180) — so neither is currently measuring the same thing with any confidence.

This belongs on the open list alongside §23's item 3 rather than being treated as a separate problem: it is the same unmigrated legacy form, on the one outcome nobody has calibrated.

### 4.5 On the Down syndrome joint models the binding parameter has moved to `kappa_min_s`

VG09 and VG10 both carry §22's lower-bound dispersion calibration, and both put the **spoken-ratio dispersion floor** in the top few per cent of its prior with almost no contraction:

| model | parameter     | median | prior CDF | contraction |     R-hat |     ESS |
| ----- | ------------- | -----: | --------: | ----------: | --------: | ------: |
| VG09  | `kappa_min_s` |  10.74 |     0.945 |       0.205 |     1.002 |     982 |
| VG10  | `kappa_min_s` |  10.60 |     0.943 |       0.163 | **1.018** | **474** |

This is §23's open item 5 arriving with sharper numbers rather than a new problem: §22 measured the same parameter at 9.2 against a prior median of 3 with contraction −0.05. It is still the loudest prior-data conflict in the joint frame, and contraction of 0.16–0.21 means the posterior is barely narrower than the prior — the prior is doing most of the work.

What is new is that in VG10 **`kappa_min_s` is now the worst-mixing parameter in the model** (R-hat 1.018, ESS 474). In §23 that position was held by `p_slope_low_u` at R-hat 1.006. The understood-trajectory ridge that dominated VG08 and VG09's diagnostics through §15 is, in the anchored model, no longer the binding constraint; the dispersion floor is. For a reporting-quality run that is where the attention should go.

### 4.6 Every subject random-effect scale in the family moved down

Not one of the thirteen went up. This is the most systematic single pattern in the run.

| model | parameter       | this run |   §23 |     change | prior CDF now |   §23 |
| ----- | --------------- | -------: | ----: | ---------: | ------------: | ----: |
| VG08  | `tau_subj_u`    |    0.778 | 0.824 |      −5.6% |         0.396 | 0.901 |
| VG09  | `tau_subj_u`    |    0.784 | 0.831 |      −5.7% |         0.399 | 0.420 |
| VG09  | `tau_subj_q`    |    1.257 | 1.380 |      −8.9% |         0.598 | 0.642 |
| VG10  | `tau_subj_u`    |    0.785 | 0.831 |      −5.5% |         0.399 | 0.420 |
| VG10  | `tau_subj_q`    |    1.259 | 1.382 |      −8.9% |         0.599 | 0.643 |
| VG11  | `tau_subject`   |    1.039 | 1.061 |      −2.1% |         0.511 | 0.521 |
| VG12  | `tau_subject`   |    0.686 | 0.736 |      −6.8% |         0.353 | 0.376 |
| VG13  | `tau_subj_u`    |    0.735 | 0.769 |      −4.4% |         0.376 | 0.392 |
| VG13  | `tau_subj_q`    |    1.107 | 1.118 |      −1.0% |         0.540 | 0.544 |
| VG15  | `tau_subj_u`    |    0.784 | 0.828 |      −5.4% |         0.399 | 0.419 |
| VG15  | `tau_subj_q`    |    1.151 | 1.275 |      −9.7% |         0.557 | 0.605 |
| VG15  | `tau_subj_sign` |    0.885 | 1.128 | **−21.5%** |         0.445 | 0.548 |
| VG16  | `tau_subj_u`    |    0.785 | 0.834 |      −5.9% |         0.399 | 0.422 |
| VG16  | `tau_subj_q`    |    1.257 | 1.381 |      −9.0% |         0.599 | 0.643 |

**This is not a prior problem, and §23's widening is vindicated.** Every scale still sits in the central half of `HalfNormal(1.5)` (prior CDF 0.35–0.60) with contraction between 0.929 and 0.990. These are data-determined parameters, and they moved because the data moved.

The two populations have different mechanisms, and both are consequences of the same commit.

For the **Down syndrome** models the likelier cause is not the extra rows but the two new exclusion rules. `exclude_ceiling_only_children` drops children with no non-ceiling record anywhere — that is, the most extreme high performers — and the Edgin re-linking collapsed 119 apparent children to 71 real ones. The pool went from 812 children to 751 while gaining 35 rows. Removing extreme children and merging duplicated ones both reduce apparent between-child spread.

For the **typically-developing** models the cause is the language scope: the new datasets contribute 2,065 additional children and no additional replication, so the pool is diluted towards single-observation children, whose contribution to a subject scale is weakest.

**Neither is isolated.** Separating them would need a fit with the exclusion rules reinstated and the language scope reverted, which the reinstatement flags make possible and this run did not do. The consistency of sign across thirteen parameters and two populations is what makes it worth recording; the size (mostly 5–9%) does not change any reported conclusion.

### 4.7 VG16's cross-lag coefficient roughly halved, and the mechanism is the child linkage

This is the largest substantive movement in the run and the one with the clearest cause.

| VG16 `beta_lag`            |    median | 89% ETI            | R-hat |   ESS |
| -------------------------- | --------: | ------------------ | ----: | ----: |
| §16 / §23, pre-change data |     0.308 | [0.182, 0.438]     |     — |   331 |
| **this run**               | **0.167** | **[0.053, 0.283]** | 1.000 | 2,656 |

The point estimate has roughly halved. The intervals still overlap on [0.182, 0.283], so this is not a reversal, but it moves the reported strength of the within-child comprehension-to-production lag by a factor of about two.

The mechanism is specific to the Edgin rebuild, and it is not the extra rows. Wordbank's by-child export assigns `child_id` **per form**, so an Edgin child seen on Words & Gestures at 14 months and on Words & Sentences at 22 months entered the pool as _two unrelated children_ — 119 apparent children resolving to 71 real ones, with 46 appearing on both forms ([202608031500a](202608031500-edgin-out-of-window-administrations.md)). The rebuild links them.

VG16 is the one model whose headline parameter depends on exactly that. Its cross-lag is identified only by observations that have a prior-wave understood source — 360 of 1,349 in this frame — and re-linking a child across the two forms is precisely what creates such a pair, in precisely the age range where a comprehension-to-production lag is expected to operate. So the linkage fix should add cross-lag pairs, and the parameter most exposed to it is the one that moved most.

**Not isolated.** Attributing the shift would need VG16 refitted on the old linkage with everything else current, which this run does not include. The direction and the magnitude are both consistent with the correction, and the prior fits' `beta_lag` should be treated as superseded rather than as a discrepancy to be explained away.

### 4.8 The study-scale prediction holds in all three models

§3 set this up. §23 found the study random-effect scale to be the worst-mixing parameter in all three hierarchical typically-developing models, attributed it to group count rather than to its prior, and named only structural remedies — a prior justified by the group count, pooling across outcomes, or fixed study effects. None has been implemented. The Romance extension raised the retained group counts for reasons that had nothing to do with sampling.

| model | retained groups | `tau` R-hat |           | `tau` ESS |         | model max R-hat |           |
| ----- | --------------- | ----------: | --------: | --------: | ------: | --------------: | --------: |
|       | before → after  |         §23 |       now |       §23 |     now |             §23 |       now |
| VG11  | 7 → **10**      |       1.072 | **1.031** |        63 | **167** |           1.072 | **1.037** |
| VG12  | 4 → **6**       |       1.009 | **1.005** |       299 | **554** |           1.011 | **1.006** |
| VG13  | 4 → **6**       |       1.023 | **1.004** |       164 | **374** |           1.023 | **1.006** |

The study scale's effective sample size roughly doubles or better in every model, its R-hat falls in every model, and the model-level max R-hat falls in every model — with **no change to the prior**, which is still the `HalfNormal(0.5)` it has carried since before §23. More groups is what changed. §23's diagnosis is confirmed on the strongest evidence available, and its structural remedies are correspondingly less urgent than they looked.

The study scale is nonetheless _still_ the worst-mixing parameter in VG11 (R-hat 1.031, ESS 167), so ten groups improves the problem without dissolving it.

Two things temper the result. Divergences did not improve — VG11 18 → 25, VG12 23 → 25, VG13 4 → 3 — and **BFMI is low across the trio**: VG11 0.335, VG12 0.188, VG13 0.229, against the ≈0.28 the July reporting run recorded and described as intrinsic to the age-varying dispersion posterior over a narrow young-age window. VG12's 0.188 is worse than that baseline. The geometry is not better overall; one specific weakly-identified parameter is better identified.

### The extension slightly worsens within-child replication

Worth recording because it cuts the other way. VG11's subjects went from 12,488 to 14,553, but the count with more than one observation stayed at **exactly 1,947** — the Italian and Spanish datasets contribute no within-child replication at all. Observations per child therefore fell from 1.326 to 1.294.

§19's open item 9 asks whether subject random effects earn their place at about 1.32 administrations per child. Widening the pool makes that ratio marginally worse, not better, even as it improves the study level. VG11's `tau_subject` is nonetheless 1.0385 with contraction 0.990, ESS 1,044 and R-hat 1.001 — the best-determined scale in the model — so the answer for now is still that they do.

### 4.9 The widened pool moved VG12's young dispersion anchor by 20%

| VG12                 |  this run |  §23 |
| -------------------- | --------: | ---: |
| `kappa_excess_young` | **34.09** | 42.5 |
| `kappa_excess_old`   |     63.99 | 65.8 |

The old anchor is unmoved; the young one fell by a fifth. Italian and Spanish (European) contribute Words & Gestures administrations concentrated at the young end (`Caselli` from 8 months after the pool bound, `Karousou` 8–15 on WG), so the young anchor is where the new data land and the old anchor is where they do not. Both parameters remain centred (prior CDF 0.430 and 0.507) with contraction above 0.95.

Dispersion still **rises** with age on this outcome — 34 at the young anchor against 64 at the old — which is §19's central finding about typically-developing comprehension, and the direction the retired `b_kappa_mag >= 0` form forbade outright. Widening the pool across three languages did not change it.

### 4.10 `kappa_min_s` contraction is worst in VG16

VG16 puts the spoken-ratio floor at prior CDF 0.933 with contraction **0.076** — the lowest contraction anywhere in the run. With VG09 (0.205) and VG10 (0.163) this makes three of the four §22-calibrated joint models where the prior is carrying over 80% of the posterior width on this one parameter.

### 4.11 VG11 and VG13 dispersion, for the record

| model | parameter              | median | prior CDF | contraction |
| ----- | ---------------------- | -----: | --------: | ----------: |
| VG11  | `kappa_min`            |  5.874 |     0.489 |       0.868 |
| VG11  | `kappa_excess_young`   |  282.4 |     0.445 |       0.950 |
| VG11  | `kappa_excess_old`     |  44.52 |     0.507 |       0.944 |
| VG13  | `kappa_min_u`          |  26.73 |     0.424 |       0.767 |
| VG13  | `kappa_excess_young_u` |  12.45 |     0.596 |       0.636 |
| VG13  | `kappa_excess_old_u`   |  83.84 |     0.469 |       0.913 |
| VG13  | `kappa_min_s`          |  3.197 |     0.532 |   **0.180** |
| VG13  | `kappa_excess_young_s` |  30.30 |     0.452 |       0.868 |
| VG13  | `kappa_excess_old_s`   |  25.88 |     0.476 |       0.857 |

VG11's anchors came down about 11% on the widened pool (282.4 against §23's 317.2; 44.5 against 50.4), both still centred. VG13's understood block again shows dispersion rising steeply with age (12.4 young against 83.8 old) with `kappa_min_u` acting as a young-age asymptote at 26.7 rather than a floor — which is exactly how §23's open item 5 describes it, and a reminder that `kappa_min` does not mean the same thing across this family.

## 5. Findings, and how firm each is

- **The two-anchor dispersion prior works, and works on data it was not calibrated on.** Firm. Across the eleven migrated models there are 49 two-anchor dispersion parameters; all 49 fall between prior CDF 0.119 and 0.945, and 40 of them between 0.28 and 0.75. VG01 — the model whose censored `b_kappa_mag` motivated the reparameterisation, at prior CDF 1.00 with contraction 0.07 through §§10–18 — is now at 0.503 / 0.606 / 0.519 with contraction above 0.84. The priors were calibrated on the pre-change pools and are still centred on the post-change ones, which is the property a scale prior is supposed to have. The two extremes are both `kappa_min` on a spoken-ratio block (VG15's `kappa_excess_old_s` at 0.119, VG09's `kappa_min_s` at 0.945), which is the conflict picked out separately below.
- **The sign constraint was the problem, and removing it resolved VG04.** Firm, and self-contained. Given a form that can express either direction, VG04 answers "flat" — 7.459 at the young anchor, 7.254 at the old, both at prior CDF ≈0.50 — where the constrained form could only pin against zero and tighten as data grew. §3 predicted this from the per-age series and §13 inferred it from a boundary collapse; it is now measured directly.
- **§23's widening of the subject random-effect scales is vindicated.** Firm. All thirteen sit at prior CDF 0.35–0.60 with contraction 0.93–0.99, on two populations whose data both changed.
- **The four unmigrated models are demonstrably mis-specified, not merely un-migrated.** Firm. VG05, VG07, VG08 and VG14 all put `b_kappa_mag_s` at prior CDF 0.9999–1.0000 with **negative contraction**, `b_kappa_mag_u` at 0.986–0.999, and `kappa_min_u` at 0.017–0.139. Four independent fits, the same five parameters, the same tails, with migrated models in the same batch for comparison.
- **The signed outcome's dispersion has never been calibrated, in either model that has one.** Firm as an observation, new to this run. VG14 and VG15 both keep the legacy form on the signed block; `b_kappa_mag_sign` collapses towards the boundary in both (0.064 and 0.076), and the two models disagree by a factor of 2.6 on `kappa_min_sign` with VG15's barely informed (contraction 0.180).
- **The study random-effect scale's poor mixing was a group-count problem, as §23 diagnosed.** Firm. Raising retained groups from 7 → 10 and 4 → 6 roughly doubled the study scale's ESS and lowered its R-hat in all three hierarchical typically-developing models, with no prior change. It remains the worst-mixing parameter in VG11.
- **`kappa_min_s` is now the binding prior-data conflict in the Down syndrome joint frame.** Firm. Prior CDF 0.93–0.95 with contraction 0.076–0.205 in VG09, VG10 and VG16, and in VG10 it is the worst-mixing parameter in the model — a position previously held by the understood-trajectory ridge. §22 already measured this parameter at 9.2 against a prior median of 3; nothing has been done about it.
- **VG16's cross-lag coefficient roughly halved, from 0.308 to 0.167.** The movement is firm; the attribution is reasoned but not isolated. The Edgin rebuild fixed per-form `child_id` splitting, re-linking 46 children who appear on both CDI forms, and that is exactly what creates the prior-wave comprehension pairs identifying the cross-lag. Earlier `beta_lag` figures should be treated as superseded.
- **Every subject scale in the family moved down 1–22%.** Firm as a pattern, unattributed. Down syndrome: plausibly the ceiling-only-children exclusion and the Edgin re-linking, which together removed 61 children including the most extreme performers. Typically developing: plausibly the language scope, which added 2,065 children and no replication.
- **Nothing changed about the geometry.** Firm and negative. Divergences are unimproved or slightly worse across the board (VG05 33, VG09 23, VG11 25, VG12 25), and the typically-developing trio's BFMI is 0.19–0.34 against the ≈0.28 the July reporting run called intrinsic. The improvements in this run are to _identification_, not to sampler behaviour.
- **Two models now pass the convergence gate outright** — VG01 and VG03 — against none in §23. Firm, but note that the gate is advisory at `test` and that these are the two simplest models in the family.

### What this run does not establish

- **Nothing here is reporting quality.** The `test` configuration is a third of the reporting draws at two thirds the chains, and the gate does not block. R-hat and ESS at this width are not the numbers a reporting run will produce: VG08's max R-hat of 1.0217 here sat at 1.0026 at plain `rep` in the July run, so `test`-level failures are not predictions of `rep`-level failures.
- **No attribution is isolated.** Two data changes and one prior recalibration landed together, and this run fits them together. Every before/after comparison in §4 is against §23's fits, which differ in both data and — for VG02, VG04, VG05, VG07, VG08, VG14 — in having no `test` baseline at all.
- **No sensitivity analysis was run.** The reinstatement flags for the new exclusion rules exist and were not exercised; the subject-scale and cross-lag movements above are the two places where that would pay.

## 6. What should happen before a reporting-quality run

In priority order.

1. **Migrate the four legacy models, or record why not.** §23's open item 3 said this was "a decision to revisit rather than a settled state". This run makes the cost concrete: five dispersion parameters per model in prior tails, with negative contraction on the spoken slope in all four. The stated reason for holding off — that VG07 and VG08 are lineage steps whose contrast a mid-sequence prior change would confound — is still valid, and is now a reason to decide deliberately rather than to defer again.
2. **Calibrate the signed dispersion block** (§4.4). It is the one dispersion block in the family with no calibration at all, it shows the boundary pathology in both models that carry it, and the two models disagree about it.
3. **Address `kappa_min_s` on the Down syndrome joint frame** (§4.5, §4.10). Prior CDF 0.93–0.95, contraction as low as 0.076, and the worst-mixing parameter in VG10. This is the loudest remaining prior-data conflict and it now costs sampling quality as well as calibration.
4. **Re-read §23's open item 2 in the light of §4.8.** The study-scale remedies it proposed — a group-count-justified prior, cross-outcome pooling, fixed study effects — should be re-priced now that the group counts have risen and the ESS has roughly doubled without any of them. VG11 at ten groups is still the worst case.
5. **Decide whether the `beta_lag` change needs isolating** (§4.7). A single VG16 fit with the Edgin linkage reverted would settle it. Whether that is worth a fit depends on how much weight the report puts on the cross-lag.
6. **Do not sync report figures from this run.** `sync_report_figures.py` requires reporting quality for good reason; `--allow-provisional` exists for local work and should not be used to move `test` numbers into the report. The staleness warning in `docs/models/README.md` stands.

### 6.1 The reporting run will not fit on this machine

Measured, not projected: this `test` run produced **48 GB** of output for fifteen models. VG11 alone is 12 GB and VG13 is 9.1 GB.

The reporting configuration is 6 chains x 6,000 draws against `test`'s 4 x 2,000 — 4.5 times the draws — and trace size scales with chains x draws. That projects to roughly **215 GB**, and `fit_model.py`'s own preflight is stricter still: it requires 20 GB per fit at reporting quality, so `all --config rep` demands **300 GB** before it will start.

This volume currently has **285 GB free**, with 48 GB of that consumed by this run. The reporting run therefore fails its own preflight as things stand. Three ways round it, in order of preference:

- **`--output-dir` to a scratch volume.** This is what the option exists for, and `vocab_growth.environment.output_root` is honoured consistently by `fit_model.py`, `fit_sensitivity.py`, `sync_report_figures.py` and `upload.py`, so nothing else needs changing. The report figure cache stays in the checkout either way.
- **A VM**, as the July run used ([202607170935](202607170935-full-refit-vm-run-147-163.md)).
- **Delete this run's traces first.** Legitimate — every number in §4 has been extracted, and traces are excluded from upload by default — but it forecloses any re-examination of these fits, including the `--render-only` retries and the isolation fits suggested above. Do the other two first.

Note also that the July run recorded roughly 21 h 43 m of driver wall time for the family at `rep`, against about 1 h 19 m here at concurrency 2–3. Plan the reporting run as an overnight job with the output redirected, not as an afternoon's work in the checkout.
