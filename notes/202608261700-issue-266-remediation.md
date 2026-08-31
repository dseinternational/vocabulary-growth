# Issue #266 remediation: what was verified, what was fixed, what still needs a refit

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

## 1. What this records

[#266](https://github.com/dseinternational/vocabulary-growth/issues/266) is a read-only audit of every registered model, the shared engines and the data/fit/report scripts. This note records the independent verification of its findings, the remediation that landed, and — as importantly — the parts of the required correction that a code change cannot discharge, so that nobody reads a green test suite as meaning the fitted output is now publishable. **It is not.** Every registered model needs a reporting-quality refit before anything is published, and this note explains why that is now enforced rather than merely advised.

## 2. Verification

Every finding was checked against the code before anything was changed; all were confirmed, several with a larger blast radius than the audit stated. Three were reproduced end to end rather than read:

- **VG08 could not complete a fit.** A VG08-shaped prior raises `KeyError: tau_subj_q` in `prior_child_checks`, in the stage immediately before posterior sampling. VG08 is the only registered model with a one-sided child effect, and the `"slope"` branch carried the same latent defect.
- **Every `paired-only` sensitivity would have died after sampling.** The audit named `common_bivariate.py`'s extraction guard as the failure point; the actual first failure is earlier, in `write_trace_calibration`, and lands _after_ NUTS has finished and _before_ `save_trace` — so the whole posterior is lost. The same defect exists a second time in `common_bivariate.py`'s own build path, which the audit did not report; it is unreachable today only because VG05 has no registered fallback variant.
- **The six fits the audit said were wrongly accepted are exactly the six.** Re-running the publication check after the fix invalidates 19 of 19 registered fits, and the six whose _only_ failure is the new prepared-frame check — VG01, VG02, VG03, VG04, VG11, VG15 — are precisely those the audit listed as having passed.

The VG17/VG18 row counts were reproduced to the digit (1,338 → 1,232 spoken; 1,305 → 1,199 produced, with all 106 excess rows in the `unknown` sign group), and VG22's induced prior on `rho_uq` was confirmed to be exactly the arcsine distribution: `P(|rho| > 0.8) = 0.410` against `0.056` under the LKJ(2) prior VG20 places on the same quantity.

## 3. The central fix: fitted output can now go stale

The manifest has always recorded `data.analysis_frame_hash`, an exact hash of the prepared frame. Nothing ever read it back — validation compared only the raw-CSV fingerprint, and the masking and exclusion rules run in Python _after_ the CSVs are read, so a loader-rule change left every stale posterior accepted as current.

Three things were needed to close that.

**A deterministic row order.** The loader queries carry no `ORDER BY`, so the frame's order followed the DuckDB scan and the recorded hash could not be recomputed. `load_combined_data` and the typically-developing query now sort on every column before the rules run. This was checked not to change the frame's _content_: the rules are order-independent, and the prepared frame is identical as a multiset before and after. It also removes a latent irreproducibility in the three registered `single-administration` sensitivity variants, whose retained row per child depended on the incoming order.

**A way to rebuild a frame outside a fit.** Each engine now exposes a pure `build_*_analysis_frame(definition)` holding exactly the frame construction its `prepare_*_data` stage runs, and `vocab_growth.analysis_frames` maps every registered model to its engine's builder. The mapping is keyed by model rather than definition class, because the engine choice lives in each `model_vgNN` module — VG05 and VG07 share a definition class on different engines. Two tests defend the split that this buys: a drift guard asserting that each engine's builder produces exactly the frame its prepare stage sets on the fit context (the manifest hashes that frame immediately after the data stage, so it is the frame a fit records), and a static check that each model's mapped engine is the one its module actually imports.

**Consumers that compare it.** `fit_model.py` (render and publish), `sync_report_figures.py`, `check_fit.py`, `regenerate_plots.py`, `upload.py`, `loso_compare.py`, `compare_models.py` and the recovery truth-draw reader all now pass the recomputed hash. `loso_compare.py` is the sharpest case: its docstring already promised the rebuilt graph matched the posterior "on the same rows in the same order", which is what this checks and the raw fingerprint cannot.

Comparison outputs got the provenance they lacked. A generating script records a `comparison_manifest.json` naming its outputs and fingerprinting every contributing fit's manifest; the sync validates it, and a contributing fit that has been refitted since fails. Coverage is ratcheted deliberately — files no entry claims are reported as warnings, so scripts can adopt the manifest one at a time instead of the first adopter breaking the sync for the rest.

## 4. The rest of the remediation

| Finding                               | What landed                                                                                                                                                                                                                                                                                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2 — VG08 blocked                      | A missing outcome's child effect is now an exact zero offset at the expected shape, matching the graph, which sets `subject_shift_q = 0.0`. Parametrised tests cover U-only and q-only, constant and slope. The trajectory panel says "no child effect on this outcome" rather than drawing 200 identical curves. |
| 3 — `paired-only` broken              | `obs_s_mask` is derived from the filtered `spoken_spec.indices` in both bivariate engines, so it marks the likelihood rows every consumer reads it as. Regression test runs a real build through the genuine predictive/calibration/extraction path.                                                              |
| 4 — LOO mislabelled                   | Relabelled, not re-computed. The multi-outcome branch now states that each row is leave-one-likelihood-term-out, that expressive rows are conditional on the same administration's observed comprehension, and that VG15's composition terms — the ones identifying psi — are excluded from every row.            |
| 5 — VG22 report describes VG20        | `docs/models/vg22/index.qmd` rewritten for the implemented rank-3 factor model, with the induced arcsine prior documented quantitatively as a known limitation and VG20's provenance and posterior removed from it.                                                                                               |
| 6 — VG17/VG18 cleaning and clustering | Both route through the canonical loader (`include_produced=True` is the new opt-in that made this possible), gained a child random intercept, and VG18 carries a prominent caution that its contrast is partly mechanical.                                                                                        |
| 7 — scripts mixing estimands          | The VG16 experiment scripts validate their traces and now refuse the stale pre-#242 fit; `compute_q_at_U` is relabelled a population transformation; K-fold folds run and record the canonical diagnostics and flag non-converged models; sensitivity and recovery scoring read the full gate, not R-hat and ESS. |
| Additional defects                    | nz_01's missing source fails closed; VG15's per-study psi is selected by an explicit informed-study indicator instead of equality with the population value; `compact_traces.py` uses a portable liveness probe; `prepare_data.py` builds into a temporary file and promotes atomically.                          |

## 5. What a code change cannot fix

**Every registered model needs a reporting-quality refit.** All 19 existing fits now fail publication validation, and that is the correct state, not a regression: the frame they were fitted to no longer exists. Until the refit run lands, `sync_report_figures.py` will refuse, which is the intended behaviour.

**VG14's and VG15's 84-month sign-ratio and `p_any` tables are recorded as known-stale.** The writers apply the 72-month cap, so a fresh fit cannot produce those rows; only a refit clears the artefacts. The exemptions are self-expiring — `test_known_stale_entries_are_still_needed` fails the moment they become unnecessary.

**Administration-level LOO is not implemented.** The per-outcome scores are relabelled accurately, which is the honest position given the stored traces, but aggregating the likelihood factors belonging to one administration — and including VG15's composition terms — is a separate change that regenerates every model page. `scripts/loo_compare.py` already contains a working reference implementation for the bivariate case.

**VG22's prior needs designing, not documenting.** The arcsine prior on `rho_uq` is now stated with its numbers and its anchor-order dependence, and the four prior-only loading magnitudes are named. Correcting it requires an explicit prior over the implied correlations, a refit of the rank family, parameter recovery and a whole-child predictive comparison before the default rank is re-selected.

**The marginal spoken fallback sensitivity is still outstanding.** Finding 8 is a real methodological exposure — roughly 455 of 1,428 Down syndrome spoken rows use an approximation that preserves the mean but not the variance — and the registered `paired-only` arms could not previously run at all. They can now, but running them is a fitting task, and VG14/VG15 still expose no equivalent choice.

## 6. Discovered in passing

VG17 and VG18 cannot be fitted at all, on this branch or on `main`: `_build` reuses VG01's `ages_query`, which runs to 90 months, against VG17's 12–66 month GP domain, and the domain check rejects it. This is pre-existing and unrelated to #266; it wants its own issue.

## 7. Second remediation pass (2026-08-31)

Three of the five items in §5 were code changes waiting on nothing, and this pass took them. The two that remain are refits and a statistical design; both are named again at the end.

### Administration-level LOO is now computed, not only relabelled

§5 recorded this as a separate change that "regenerates every model page", and deferred it. It is taken now for a reason that has a deadline attached: **every registered fit already needs re-running**, so the cost of this change is currently zero and rises the moment the refit run starts. Landing it after that run would waste the run.

`vocab_growth.administration_loo` sums every likelihood factor belonging to one row of the analysis frame into a single pointwise entry, and the diagnostics stage computes a LOO over it **alongside** the per-outcome scores rather than instead of them. The per-outcome numbers stay comparable across models on the same conditional units; what they never were is whole-administration predictive accuracy, and the reports said they were.

Three things follow. A paired administration is one held-out case with one importance weight rather than two. Nothing of a held-out row remains in the conditioning set — previously, holding out the comprehension factor left its own observed value in the spoken factor's denominator. And **VG15's two composition terms are included**, so the model's headline association `psi` is scored by leave-one-out at all for the first time; the per-outcome scores omit those terms entirely.

The mechanics are `scripts/loo_compare.py`'s `_attach_joint_log_likelihood`, which has computed this correctly for the bivariate case since #236, generalised to any number of factors including the matrix-valued composition ones. Verified on a real two-chain VG10 fit while it was written: 96 per-outcome terms collapse to 48 administration cases with the total log-likelihood conserved to 0.0 exactly.

**This depended on finding 3 having been fixed first.** The mapping from a factor's likelihood rows back to administration rows is the `obs_*_mask` constant data. A mask marking recorded rows rather than likelihood rows would sum the wrong factors onto the wrong administrations — silently, and plausibly. The combiner refuses a mask that does not match its factor rather than proceeding.

### VG14 and VG15 can run the marginal fallback sensitivity

Finding 8's exposure — roughly 455 of 1,428 Down syndrome spoken rows take an approximation that preserves the mean but not the variance, on rows that are older and clustered by study — could not be measured on either signing model, because both hard-coded the default. Both now carry `spoken_fallback`, route their child-outcome likelihoods through the shared `nested_outcome_alpha_beta`, and have the three arms registered. The treatment applies to the **signed** rows as well as the spoken ones: signing is nested inside comprehension exactly as speech is, so varying one and not the other would leave half the exposure in place.

Two things were found in doing it.

**The trivariate engine carried finding 3's defect, unreachably.** It stored `has_s` and `has_sign` — the _recorded_ masks — where the bivariate engines store the likelihood ones. With no fallback choice to select, no row was ever dropped and the two coincided; exposing the choice would have made it reachable, and the first `paired_only` run would have died at calibration after sampling, exactly as the bivariate fits did. Fixed before it could be discovered that way.

**The joint engine read its indices before applying the drop**, leaving the `obs_s_id` coordinate sized by the unfiltered rows and the trial counts by the filtered ones. That fails at logp evaluation with a bare shape error and no indication of the cause. Caught because the test builds each treatment on a frame that _has_ marginal rows — the graph-equivalence harness's frame is fully paired, so `paired_only` and `moment_matched` are no-ops on it and a test using it would have passed while proving nothing.

**Adding the field invalidated no existing fit**, which is the first real use of the versioned manifest payload from #273 step 5. Under the raw dictionary equality this replaced, adding `spoken_fallback` would have invalidated every VG14 and VG15 fit ever made, for a field whose default is what those fits already did. `BACKFILL_DEFAULTS` records that claim, and it is checkable rather than asserted: `resolve_fallback_treatment` reads the field through `getattr` with exactly this default, so a definition that predates it resolved to `product_marginal`.

Both models also became reachable from `fit_sensitivity.py` and `refit_hightune.py` with no further change, because #273 made those scripts derive their runner map from the catalogue and the variant registry.

### Still outstanding, and still not a code change

**Every registered model needs a reporting-quality refit.** Unchanged from §5, and now also the only way to get an administration-level LOO row onto any model page — the score is computed during the fit.

**VG22's induced prior needs designing.** Unchanged: an explicit prior over the implied correlations, a refit of the rank family, parameter recovery and a whole-child predictive comparison before the default rank is re-selected.

**The fallback sensitivity arms need running.** They can now be run on all four affected engines rather than two, which is what this pass changed; running them is a fitting task.
