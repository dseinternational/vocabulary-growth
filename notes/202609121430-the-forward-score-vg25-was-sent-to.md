# The forward score VG25 was sent to, and could not run

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-12. **Question:** VG25's report page tells the reader its out-of-sample figures leak across the lag and that `scripts/wave_forward_score.py` is "the comparison to trust over the LOO table above". Does it? **Evidence:** the script's own model gate, the joint engine's diagnostics stage, and what a joint trace exposes — each measured rather than read. **Change:** the joint engine suppresses the scores that leak, as the bivariate engine has since [#242](https://github.com/dseinternational/vocabulary-growth/issues/242); the script takes VG25; and the per-row quantities a score needs are named on the joint engine, which they were not. No fit of record is touched and no posterior moves.

## 1. The remedy the page named could not be run

`wave_forward_score.py` chose its models with

```python
CROSS_LAG_MODELS = tuple(
    key for key, d in MODEL_REGISTRY.items() if getattr(d, "use_cross_lag", False)
)
```

`use_cross_lag` is a `BivariateModelDefinition` field. VG25 does not have it — its coefficient is switched on by `use_sign_cross_lag` — so `CROSS_LAG_MODELS` was `('vg16',)` and `--model vg25` was an argparse error. The issue and the registration note both recorded this as "has not been run for this model", which reads as a scheduling matter. It was not.

## 2. The joint engine computed the scores that leak

Worse, and the reason the page needed that pointer at all. The bivariate engine has suppressed VG16's understood LOO since #242: the lag predictor holds an earlier wave's observed comprehension as a fixed covariate of every later row it feeds, so leaving that likelihood term out does not remove the count from the model, and Pareto-$k$ checks the importance-sampling approximation rather than the leakage.

VG25's predictor reads an earlier wave's `signed` **and** `understood`. The joint engine's `diagnostics` took no notice: it computed all three marginal scores plus the administration-level score, unconditionally. The leak was flagged in the report's prose and nowhere in the code.

It is now suppressed on the same terms. `y_u_obs` and `y_sign_obs` are not computed; neither is the administration-level score, which bundles both with the two composition terms the lag also enters. What is kept is `y_s_obs` — no predictor reads a spoken count — labelled as prediction conditional on the child's observed sign history. `tests/test_cross_lag_loo_suppression.py` pins both engines, including the fail-closed case: a cross-lag registered on an engine with no suppression fails and names the engine. Neither engine's suppression had a test before.

## 3. The obstacle was not the script

The port looked like a day's renaming. It was not, and the reason is worth recording because it is invisible from the script.

**The joint engine named almost nothing per row.** The bivariate random-effect engine has exposed `p_u_obs`, `p_s_obs`, `q_obs`, `kappa_u_obs` and `kappa_s_obs` as deterministics since it was written, and the forward score reads a held-out row's predictive density straight off them. Built on the same synthetic frame:

| engine           | observation-level deterministics                                            |
| ---------------- | --------------------------------------------------------------------------- |
| `bivariate_re`   | `f_s_obs f_u_obs h_obs kappa_s_obs kappa_u_obs p_s_obs p_u_obs q_obs z_obs` |
| `joint` (before) | `kappa_sign_obs z_obs`                                                      |

`p_u_obs`, `q_obs`, `r_obs` and the two kappas existed only as plain PyTensor expressions consumed by a likelihood. There was nothing in the trace to score from, and no amount of work on the script would have produced any.

Five are named now — `p_u_obs`, `q_obs`, `r_obs`, `kappa_u_obs`, `kappa_s_obs` — with the four-cell composition beside them. **Naming them changes no draw**: a deterministic is a function of the free variables, so the log density is unchanged — `tests/test_graph_equivalence.py` records a fixed-point log probability beside the names, and regenerating its baseline moved the names and the dims and left VG15's, VG24's and VG25's log probabilities where they were (VG15's by 4e-12, one ulp on 7201). **Naming them costs nothing in a stored trace**: they carry `obs_id`, so `fit_artifacts.unsampled_deterministic_names` keeps them out of `pm.sample`'s `var_names` and nutpie never evaluates them. Only a caller that asks — `fold_fits.fit_holdout_fold`, with `store_observation_deterministics` — pays.

It is not free everywhere, and the exception decided the length of that list. The prior-checks stage calls `sample_prior_predictive` with no `var_names`, so it evaluates and holds **every** deterministic: on the real frame each obs-sized column is 6.8 MB at 500 draws, and the six added here are about 61 MB on top of that stage's existing 82 MB. `q_obs_pop`, `r_obs_pop` and `log_psi_obs` were named too until that was measured — they are the inputs to the composition and nothing reads them separately, so naming the composition alone carries the same information for a quarter less. Giving the prior-predictive stage the `var_names` treatment `pm.sample` already has would remove the rest, and is a change to three fitted models' prior-checks behaviour rather than a detail of this one.

**And the fold fit itself was hard-wired to the wrong engine.** `fit_holdout_fold` called `configure_bivariate_priors` and `build_model_re` by name. A joint definition passed to it would have been built by the bivariate builder, producing a graph that is not the model, then fitted and scored without complaint. It resolves the stages from the definition's own engine now, through a new `catalogue.engine_for_definition` — by `model_id`, because a fold arm is a `dataclasses.replace` with a different `config_name` and the key is not recoverable from it.

## 4. What is scored, and the one decision in it

The headline stays the spoken elpd difference, on both engines, because that is where the coefficient enters the logit. VG25 adds two things.

**Signed is a second control.** The lag enters neither comprehension nor signing, so both should show no difference between the arms.

**The composition is not a control.** With `sign_lag_in_cells` the lag is added to the population production logit the Dirichlet-Multinomials are built on — the scope decision the registration took, worth 191 supporting observations against 111 without it. Scoring the marginals alone would score the coefficient on the evidence that decision chose against, so `elpd_cells` is computed and reported beside the spoken difference. nz_01's three-cell produced composition goes in the same column: its rows are disjoint from the four-cell rows.

## 5. The first end-to-end run found two defects the unit tests did not

Both in which rows get scored, and both invisible to anything short of running it.

**`obs_cells_mask` marks the rows in the likelihood.** A fold's held-out rows are excluded from the likelihood by construction, so every row the score evaluates is absent from that mask — reading it scored no composition at all. The run pivoted an all-missing column away and then failed on its absence, fifteen minutes of fold fitting later. The criterion is the frame's own `signed_spoken` column, and no inclusion flag has to be reconstructed to use it: `include_uk07_cells` and `include_es01_cells` act at data preparation, moving a study's rows between the frame's cross-tab and marginal branches.

**A four-cell row carries no spoken or signed marginal.** The engine's `marginal_outcome_eligible` excludes it, because its production information is in the composition. Scoring a marginal density there would have scored a density the model does not hold. The same exclusion now reaches the scorer through `nested_outcome_spec`'s own `eligible_mask` rather than being restated.

Both are pinned by unit tests against a fabricated one-draw trace, which is where they should have been caught: the composition density is checked against `pm.DirichletMultinomial`'s own logp rather than against the formula that produced it.

## 6. What has not been done

**The score has not been run for record.** What has run is a two-fold `dev` smoke test to prove the path works end to end, and `dev` does not converge these models — every fold missed the R-hat gate, which is what the script's own convergence warning is for. A result needs `test` at least, and VG16's experience says even that may not be enough. **The elpd numbers it printed are not quoted here, and should not be quoted anywhere.**

What the smoke test does establish is that the right rows are scored, and the counts reconcile against the frame exactly rather than approximately:

| scored                                                | run | frame |
| ----------------------------------------------------- | --: | ----: |
| rows (all later waves of the fold children)           | 733 |   733 |
| spoken marginal                                       | 468 |   468 |
| signed marginal                                       | 125 |   125 |
| composition (80 four-cell + 78 produced)              | 158 |   158 |
| rows carrying both a spoken density and a composition |   0 |     0 |

The last row is the one that matters: a cross-tab row's production information is in its composition, and on this frame the preparation already leaves such a row's `spoken` missing — so the `eligible_mask` exclusion is belt-and-braces here rather than load-bearing, and the two were checked separately for that reason. The 733 is the other: it is the count of distinct scored rows, and the first attempt at keeping an empty outcome column would have inflated it to a cartesian product of the index levels.

VG16's own forward score has still not been run either ([#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 3.8, registered 2026-09-07), so neither cross-lag model has generalisation evidence. What has changed is that both now can: before this, one of them could not, and its report page said otherwise.
