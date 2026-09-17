# Parameter recovery

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

A parameter-recovery check simulates data at known parameter values, fits a model to those data and compares its estimates with the known values. It asks how well this model and fitting procedure can recover quantities from the study's design, sample size and missingness pattern. It does not establish that the model represents the real data adequately.

A poorly recovered value can reflect limited information, prior influence, a model constraint, simulation error or sampling failure. Diagnose these possibilities before drawing a conclusion about identification.

## How the simulation works

The harness draws outcomes from the engine's own likelihood nodes with one fixed parameter draw. It does not maintain a second implementation of the probability model.

Comprehension is drawn before outcomes that use it as a denominator. The engine then rebuilds the graph from the simulated parent counts. Lagged predictors are also rebuilt in the required order. `recovery.spec.single_pass_is_sound` determines whether stages suffice or a wave-by-wave simulation is needed. VG16's comprehension lag can use the staged path; VG25's signing lag requires the wave loop. Checks compare the predictors, denominators and row membership used during simulation with those reconstructed from the finished frame.

The simulation keeps ages, studies, child identifiers and the pattern of missing outcomes. It replaces the observed counts. Model-generated nested counts cannot reproduce source-data violations such as speech exceeding comprehension. A recovery result therefore does not test every data-quality problem the real loader handles.

## Run a study

```bash
uv run python scripts/fit_recovery.py headline --config test
```

`headline` selects VG20, VG12 and VG15. A single model key selects one model; `all` selects every supported model. Each replicate simulates, refits and scores. `test` is useful for an initial run, but its name does not establish adequate sampling. Only converged recovery fits can support an interpretation; increase the sampling effort when needed.

| Option                     | Effect                                                            |
| -------------------------- | ----------------------------------------------------------------- |
| `--replicates N`           | Number of replicates; default 3.                                  |
| `--replicate N`            | Select one replicate; repeatable.                                 |
| `--truth posterior\|prior` | Source of the parameter draw; default `posterior`.                |
| `--config NAME`            | Sampling configuration.                                           |
| `--simulate-only`          | Write simulated data and truth without fitting.                   |
| `--fit-only`               | Fit existing simulated data.                                      |
| `--compare-only`           | Score existing recovery fits.                                     |
| `--variant NAME`           | Simulate and fit a registered sensitivity variant.                |
| `--fit-variant NAME`       | Fit a different definition; `record` selects the model of record. |
| `--set-truth NAME=VALUE`   | Set a free parameter in the truth draw; repeatable.               |
| `--output-dir PATH`        | Select the output root.                                           |

Separate stages allow inspection and resumption:

```bash
uv run python scripts/fit_recovery.py headline --config test --simulate-only
uv run python scripts/fit_recovery.py headline --config test --fit-only
uv run python scripts/fit_recovery.py headline --config test --compare-only
```

Before taking truth from a stored posterior, the harness validates that fit against its definition and data. Later stages also compare the simulation's recorded definition with the one requested. A mismatch is refused. Regenerate the simulation or use the revision that produced it; do not combine old free-parameter draws with a changed graph's derived quantities.

Recovery fits are named `<model_id>-<config_name>-recovery-rNN` under the output root's `models/` directory. They do not replace models of record and are not selected by the report figure sync. Inputs are stored separately under `recovery/`, because completing a fit atomically replaces its model directory:

- `synthetic_analysis_frame.parquet` contains the simulated data. A checked round trip preserves values, data types and missingness.
- `truth.nc` contains the parameter draw and its derived quantities.
- `simulation.json` records the source draw, definition, simulation order and coherence checks.

## Choose the truth

`--truth posterior` selects draws spread across the fitted posterior. This examines recovery in the parameter region used by the report. It requires a compatible full trace at the same output root. A variant uses its own fitted posterior.

`--truth prior` draws from the prior and requires no stored fit. This helps test the harness and recovery across the prior's range. Broad priors can produce settings far from the reported estimates, so label the truth source whenever quoting results. Neither truth source covers every setting of scientific interest.

### Set parameters for a designed check

`--set-truth` replaces a named free parameter after choosing the draw. This is useful for separating a lag effect from a persistent child correlation, as requested in [#297](https://github.com/dseinternational/vocabulary-growth/issues/297) for VG25 and [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) for VG16.

```bash
uv run python scripts/fit_recovery.py vg25 --config test --truth prior --set-truth beta_sign_lag=0
uv run python scripts/fit_recovery.py vg25 --config test --truth prior --set-truth subject_re=independent
uv run python scripts/fit_recovery.py vg25 --config test --truth prior
```

The first run removes the lag coefficient, the second removes child correlations, and the third retains the draw. Inspect the resulting truths rather than assuming an unmodified draw supplies a particular nonzero effect size.

Only sampled free variables can be set. Trajectories, correlations and other derived quantities are recomputed from the changed draw. The correlated child block samples a packed Cholesky factor, `subject_re`, which encodes its covariance. `subject_re=independent` removes the correlations while retaining the scales. It does not set the child scales to zero.

Settings appear in simulation paths, fit names and score labels so distinct checks cannot overwrite each other. The harness rejects settings that produce non-finite reported quantities. The user must still choose scientifically meaningful values. Long Windows paths may prevent the optional Graphviz diagram from being written.

## Recover a sensitivity variant

```bash
uv run python scripts/fit_recovery.py vg10 --variant a1-tau-age-varying --config test
```

This uses the variant for both simulation and refitting, with the base model's engine. Labels include the variant, for example `recovery_matrix_vg10-a1-tau-age-varying.csv`.

Structural changes need checks of their added quantities. Prior changes can also alter recovery through shrinkage, even when the likelihood structure stays the same. Choose checks around the actual concern, including parameter settings or age regions with little information. Success where the data are plentiful does not establish recovery elsewhere.

### Compare fits to the same simulated data

`--fit-variant` separates the generating definition from the fitted definition:

```bash
# Simulate from VG15 and refit with a narrower prior on tau_psi.
uv run python scripts/fit_recovery.py vg15 --fit-variant tau-psi-narrow --config test

# Simulate from the variant and refit the model of record.
uv run python scripts/fit_recovery.py vg15 --variant tau-psi-narrow --fit-variant record --config test
```

Compare the first run with a plain VG15 recovery using the same truth, replicate settings and simulated data. Differences can then assess the prior's effect, subject to Monte Carlo error. In contrast, changing both the generating prior and fitted prior changes the question.

The simulation directory remains keyed to the generating definition, and its provenance checks still apply. Fit names and scored labels contain `-under-<tag>` for cross-definition runs. The harness scores quantities shared by truth and fit; read the target list when the definitions differ.

## Read the results

Tables are written under `<output-root>/comparisons/recovery/`:

| File                                  | Contents                                                                                                                                     |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `recovery_<model>_rNN.csv`            | Truth, posterior summaries, standardised error `z`, 50% and 89% intervals, containment flags and the truth's quantile among posterior draws. |
| `recovery_<model>_rNN_aggregates.csv` | Summaries across high-dimensional child effects, including interval containment and correlation between true effects and posterior means.    |
| `recovery_matrix_<model>.csv`         | One row per replicate and an indicative pooled row.                                                                                          |

Targets are selected from the graph by dimension. Scalars, query-age trajectories and study effects are scored separately. The rules in `recovery/compare.py` exclude design grids, duplicate logit-scale quantities and non-centred raw offsets. The many trajectory rows are correlated; they are not independent recovery experiments.

The harness also derives the standard deviation, in words, of one assessment for a new child. It uses `comparison.total_spread_from_values`, the same calculation as the population comparison. This target is available for the single-outcome and supported bivariate models, but not the joint signing engine, VG22's factor structure or cross-definition runs with incompatible query grids. Read its rows separately when assessing the between-population spread claim. Its addition on 2026-09-14 changed target counts, so older and newer matrices are not directly comparable row for row. See the [total-spread decision](../../notes/202609141600-total-spread-estimand.md).

### Check convergence before interpreting recovery

A replicate is assessed only when its diagnostics confirm convergence. The thresholds include maximum R-hat of 1.01 and minimum effective sample size of 400. Missing or failed diagnostics yield `UNVERIFIED` or `NON-CONVERGED`, with no contribution to the pooled result. Read any recorded diagnostic caveats as well.

A `dev` or `test` fit can finish without meeting the reporting pipeline's hard gate. Completion is therefore insufficient. If a recovery fit has not converged, the apparent error cannot yet distinguish a sampling problem from poor recoverability.

### Interpret a small study cautiously

`coverage_ci89` is the fraction of selected quantities whose 89% intervals contain the truth in these runs. It is not an estimate of repeated-sample coverage for each quantity. Quantities within a replicate share one dataset and are strongly dependent. Also, some interval misses are expected even from a well-calibrated procedure.

A large standardised error or repeated errors in one direction warrant investigation. Central, precise recovery is reassuring at the tested truths, but does not prove general identification. Review error size, direction, interval width and sampling diagnostics together. Do not treat every interval miss as a defect or every hit as success.

A larger calibration study needs a design and replicate count matched to the precision required. There is no universal replicate threshold. Prior-based simulation calibration and recovery at selected posterior truths answer different questions. The recorded `truth_quantile` supports later calibration work, provided the generating design is appropriate.

## Supported models and cost

The supported set is VG07–VG13, VG15, VG16 and VG19–VG26. `recovery.spec.supported_models()` is authoritative. VG01–VG05 and VG14 are excluded descriptive baselines; `UNSUPPORTED_REASONS` records why. A test checks that the supported and excluded sets cover the registry.

Simulation builds the graph several times but does not run posterior MCMC. Refitting is the main cost and depends on the model, sampling effort and hardware. Run `--simulate-only` first to check the design before committing to a long study.

Recovery complements sensitivity and predictive checks. Sensitivity asks how a modelling choice changes the estimate. Posterior predictive checks compare model-generated outcomes with observed data. Recovery asks how the fitted procedure behaves when the generating truth is known. None replaces the others.
