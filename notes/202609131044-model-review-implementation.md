# Implementing the model-code review

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-6).

This records the implementation of the accepted [13 September review](202609131002-model-code-correctness-and-teaching-review.md). Implementation began after rebasing to `9adbfcb`, with a final rebase to `e95288a` described below. The source changes preserve the registered multi-study probability models at the saved graph checks. They correct non-default edge cases and convergence reporting, expose the existing statistical decisions through smaller helpers, and add a tested route for students. Passing graph checks is evidence about those checks, not a proof of equality at every parameter value or evidence from a new posterior fit.

## Correctness and modelling choices

| Review finding                                                            | Implemented behaviour                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Zero child-slope reference age was replaced with 36 months during fitting | Fitting, prediction, prior checks and comparisons use `slope_reference_age`. Only an absent value selects the historical 36-month default. Tests compare fitted and predictive multipliers at reference ages zero, 24 and 36 months.                                                                                                                                   |
| Study constraints failed with one study                                   | `zero_sum_study_offsets` returns a deterministic zero vector when fewer than two studies enter the constraint. No between-study contrast is estimable in this case; any retained scale has no likelihood information and retains its prior.                                                                                                                            |
| The bivariate study-constraint explanation contradicted outcome coverage  | The helper makes axis membership explicit. The bivariate engine retains its reference over every retained study. The joint engine retains its reference over studies involved in each outcome's likelihood. Tests include comprehension-only and speech-only studies, under both default and paired-only treatments.                                                   |
| A missing hard diagnostic result could be reported as a pass              | Both report blocks use `hard_tier_status`, which distinguishes passed, failed and unknown. It checks explicit failures, incomplete or unassessable scans, and finite numerical R-hat and effective-sample-size results against the recorded thresholds. Unknown evidence produces a provisional report. An accepted convergence exception remains a separate decision. |

The choice to retain the bivariate study reference is deliberate. Changing it to outcome-informed studies would alter the prior and the interpretation of the population curve. The review established that its explanation was wrong, not that the probability model was invalid. Such a change needs a separate statistical comparison. A study without a direct observation of an outcome can still acquire indirect information through the zero-sum constraint or another likelihood involving that outcome.

## Refactoring and comments

The bivariate and joint builders now call shared study-offset and child-effect builders. Their returned records name the comprehension, spoken-ratio and signed-ratio quantities. Joint observation preparation returns a `JointObservations` record with the likelihood indices and count arrays. The bivariate builder reads its existing observation record directly. The Plackett probabilities and both composition likelihoods now have a separate module, with the original public aliases retained for compatibility.

Counting the graph-building function itself, the joint core is 638 physical lines, down from 946, and the bivariate random-effect core is 537, down from 753. Some code moved to the named helpers; the reduction is not all deletion. The small reporting wrapper is excluded from these counts.

All six engine families use the bounds already held by `AgeGrids`. Each exposes `build_model_graph`, which returns build-report values without printing tables, drawing figures or writing report files. The existing pipeline entry point still calls the report function after construction. `report_build=False` also suppresses the earlier prior-configuration output. The quiet-build tests exercise each engine with the suite's usual reporting suppression disabled.

The bivariate predictive code now selects the child structure from the same `SubjectEffectPlan` used during fitting. It checks that the variables required by the selected structure exist. A missing slope variable raises an error instead of selecting a simpler offset model. Plot and query offsets for one new child are returned together so they share the same child draw.

Comments now distinguish an independence calculation from a bound, a logit midpoint from an average probability, a middle knot from a guaranteed peak, and an unsupported correlated age-varying implementation from a mathematical impossibility. GP documentation states that the curves need not increase monotonically. The variance-partition documentation describes an approximate allocation at a fixed reference proportion and makes clear that its priors change the model. Count-validation and covariance-factor explanations remain beside the code they justify. Repeated headings, descriptions of assignments, and long development histories have been shortened or replaced by links to their evidence.

The retained historical evidence includes:

- The earlier VG10 high-age mean and GP compensation, documented in [the mean-extrapolation note](202608042030-q-mean-extrapolation.md). Those fitted values explain the clamp's origin; they are not current numerical guarantees or a monotonicity constraint.
- The study of sampling geometry in [the TD hierarchical note](202608050900-td-hierarchical-geometry.md) and the replicate results in [the recovery baseline](202608161700-recovery-baseline-215.md). Low child-scale estimates appeared with and without the variance partition. Those runs do not establish that the partition caused the discrepancy or that every reported contrast is unaffected.
- The earlier fixed signing-knot comparison in [the prior-conflict note](202608060900-three-prior-conflicts.md). The adjustable knot addresses that modelling question without guaranteeing a peak there.
- The prior VG16 and VG10 fitted-child correlations, previously quoted as `0.135 [0.087, 0.180]` and `0.152 [0.105, 0.195]` in VG20's introduction. These were descriptions of overlapping data under earlier models, not independent measurements or guaranteed lower bounds. Their context is [the VG16 analysis](202608151120-vg16-cross-lag-quantified.md).
- The locked-library distinction between a correlation matrix and a Cholesky factor, recorded in [VG24's registration note](202609061530-vg24-registration-and-the-lkj-primitive.md). The joint child builder still uses `LKJCholeskyCov`; its returned covariance factor already contains the child scales. Distribution and density checks remain in the tests.

The old comments above VG25 predicted a small coefficient from the earlier descriptive interval and VG16's recovery result. Those observations concerned different models and a different subset of observations. They may motivate a sensitivity or recovery analysis, but do not specify a valid expected result for VG25. The new comments describe the conditional association being modelled and link to its history.

## Form audit and a VG25 sensitivity

Dividing signed words by understood words cancels a common numerical inventory denominator. It does not show that changing the words on the questionnaire preserves the measured ratio. I reconstructed form membership from source metadata and the retained counts, including the DSE and Oxford CDI administrations within `uk_02`. An ambiguous or absent source match remains unknown.

On the prepared frame at this revision, VG15, VG24 and default VG25 each have 1,708 observations and the same frame hash:

```text
sha256:c87e06af0ba30b262319feadbabd7debd1aebe5837ed5b6bfd77a1bbe70ac7f1
```

VG25 has 191 lagged rows. Seven use a DSE source administration with an Oxford CDI target, all in `uk_02`. All seven enter the conditional spoken-count likelihood. The other 34 `uk_02` lags connect two DSE administrations. None of the 191 source or target inventories is unknown, although 184 rows elsewhere in the full frame lack resolved inventory metadata. There are no same-child, same-age observations with usable signed shares on different forms in this frame, so it cannot directly estimate whether those shares agree across forms.

The new `sign-lag-same-form` sensitivity keeps all 1,708 observations and retains 184 lags. It drops the seven cross-form lags. Selection first finds the latest earlier wave, then applies the restriction; it does not search further back for an older matching form. Unknown source or target inventory sizes also fail the restriction. Inventory size distinguishes the current within-study forms here; it does not certify measurement equivalence between studies or between arbitrary forms of equal size.

To reproduce the audit or fit the new arm:

```bash
uv run python scripts/audit_sign_lag_forms.py
uv run python scripts/fit_sensitivity.py vg25 sign-lag-same-form --config dev
```

The audit writes row membership, form-transition counts and a summary under `output/audits/sign-lag-forms/`, subject to the configured output root. It reads prepared data, not a posterior. Its rows also report which likelihood terms the lag can reach. The same-form frame adds `survey_vocab_max` for this selection and has hash `sha256:4798a7d45839aa50aaeaf232a3ecc9337f0fae418858a3e1e066fe37ed3d198d`. The default frame and its schema remain unchanged.

`sign_lag_same_form_only` defaults to false. The historical-definition backfill records exactly that value, so an old default VG25 definition remains equivalent on this field, while the sensitivity is a graph-affecting change. A regression check also changes the lag coefficient in built models and verifies that cross-form likelihood terms lose their dependence on it when restricted. No sensitivity posterior was fitted in this implementation, so the effect on the estimated coefficient remains unmeasured.

## Teaching and verification

The [student guide](../docs/tutorials/model-code-walkthrough.md) follows counts, logits, the Beta-Binomial, conditional speech, study and child offsets, correlation, and overlapping sign-speech categories. Its [executable example](../examples/model_walkthrough.py) uses the registered builders on invented data. Tests run the complete example, including missing comprehension, one study, a zero reference age and the independence counterexample. It builds graphs and takes eight prior draws; it does not fit a posterior or assess convergence.

Before changing statistical code, I saved two parameter points and their log probabilities for each of the 22 registered graphs, together with every coordinate value. The original `graph_baseline.json` was not regenerated. The expanded checks compare the refactored graph with both saved values at the same transformed parameters. A separate HalfNormal example verifies that a prior-scale change is detected when a fixed common point is used; deriving a fresh point from each changed prior could conceal that change. These checks accompany tests of likelihood membership, conditional composition, correlation distributions, reference-age handling and reporting-grid invariance.

The full suite passed with **2,575 tests passed and 11 skipped** in 351.64 seconds. Both fast and slow tests were selected, using `python -m pytest -m "slow or not slow" -n 4 --dist loadgroup --tb=short -r s`. Six skips concern models without a clamp field; five require a stored model-of-record fit, which this worktree does not contain. The run includes numerical sampling tests, all 22 registered graph comparisons, and the teaching example. It is not just the default fast selection.

The first full run exposed fixtures that patched the old location of the graph renderer. Those now use `report_build=False`, without changing their statistical assertions. The sensitivity registry assertion now includes the new arm. The real-data DSE-native preparation test also now writes to `tmp_path`, so a user's configured output directory no longer controls where it writes. These corrections were checked in the successful full run. Separate tests confirm that the normal pipeline still creates reporting artefacts.

Ruff passed over `src/`, `scripts/`, `tests/` and `examples/`. Mypy passed its four configured source modules. Prettier passed the changed Markdown files, and CSpell passed the changed Markdown and VG25 Quarto template. The local links in the new guide and implementation note resolve. Formatting the changed Python sections preserved their executable ASTs.

Tests used the main checkout's existing Python 3.14.4 environment with imports explicitly pointed at this worktree. Its installed PyMC, PyTensor, NumPy, SciPy, pandas, ArviZ, nutpie and `dse-research-utils` versions match this worktree's lockfile. Compilation caches and test outputs used temporary directories. Windows had no `g++`; PyTensor reported loop-fusion limits and Numba object-mode fallbacks, with no test failures in the final run.

During the last repository check, `origin/main` had advanced to `e95288a`. Its two new commits are [PR #343](https://github.com/dseinternational/vocabulary-growth/pull/343), which restores VG15's association-support table and adds checks for static package references in Quarto templates. They change the VG15 template and a test module, not model calculations. I rebased with Git's automatic stash and checked all 57 implementation files afterwards. The 42 tracked edits were preserved apart from line-ending normalisation; all 15 untracked files retained their exact hashes. There were no conflicts or remaining stash entries. The new template checks, both convergence-report test modules and the notes-index checks then passed, **235 tests in total**. These overlap the earlier suite and should not be added as unique tests. The final rebase does not resolve or alter the statistical findings addressed here; it adds a guard against another kind of broken report reference.

## Final review against main

The PR review rebased onto `6eb4579`, which adds study exclusion to the joint definitions and registers VG25's `no-uk07` and `no-ie02` sensitivities in [PR #344](https://github.com/dseinternational/vocabulary-growth/pull/344). The same-form arm remains useful alongside those checks. All ten VG25 sensitivities are now registered. The overlap resolution preserves the new exclusion field, its historical-definition backfill, both study-exclusion arms and their documentation.

Three new integration cases check the study exclusions together with the form restriction. They verify that study and child codes remain consecutive, the retained frame differs only by its inventory metadata, and every likelihood keeps the same observation indices. The form restriction removes a cross-form lag without selecting an older source. The real-data check gives:

| Excluded study | Remaining rows | Lags before the form restriction | Same-form lags |
| -------------- | -------------- | -------------------------------- | -------------- |
| None           | 1,708          | 191                              | 184            |
| `uk_07`        | 1,626          | 139                              | 132            |
| `ie_02`        | 1,597          | 148                              | 141            |
| `uk_02`        | 1,572          | 150                              | 150            |

The default and same-form analysis-frame hashes still match the audit above. The default graph baseline remains unchanged. The review also clarified that the study offsets, rather than the child offsets, sum to zero, and that a midpoint on the logit scale need not give the average probability after transformation. Formatting changes unrelated to the new sensitivity were removed from the registry.

The complete suite on this rebased code passed with **2,644 tests passed, 11 skipped and 87 warnings** in 402.56 seconds. The command again selected both fast and slow tests with four workers. The skip reasons remain six inapplicable clamp settings and five checks requiring stored fits. Ruff, the four-module mypy check, Prettier, CSpell and the local documentation links also passed. The combined sensitivity checks ran against both fixture data and the prepared study data. No further correctness issue was found in this final review.

The executable refactoring changes the implementation signature even where the probability model is preserved. Publication and resume checks continue to enforce that signature. No reporting-quality posterior refits, rendered fit reports or live publication checks were performed, and no provenance rule was relaxed.
