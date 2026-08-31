# One model catalogue, and the four dispatch tables that had drifted from it

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Work on [#273](https://github.com/dseinternational/vocabulary-growth/issues/273), steps 1–4. Records what was wrong, what the catalogue is, and the two design choices in it that are not obvious.

## 1. What was wrong

A registered model's **engine** — which of the six shared fitting modules builds its graph — was declared in seven places: the `model_vgNN` wrapper's import, `analysis_frames.FRAME_BUILDERS`, `scripts/regenerate_plots.py`, `scripts/prior_predictive_audit.py`, `scripts/fit_sensitivity.py`, `scripts/refit_hightune.py` and `recovery/spec.py`. Two of the seven had drift guards. Three of the remaining five were wrong at the same time.

**The prior-predictive audit built the wrong graph for six models.** Its `_BIVARIATE_RE` set listed VG07–VG10 and VG13; VG16 and VG19–VG23 also run on `common_bivariate_re`, and the fallback sent them through the plain `common_bivariate`. So an audit of the cross-lag, child-slope, correlated-effect and factor models built a graph without the structure that distinguishes them — and still produced plots, which is how the mismatch reads as a valid prior check rather than as a failure. (The issue says seven models; it is six. VG05 is genuinely on the plain engine and the fallback was right for it.)

**The same script dropped the definition for every random-effect model.** `common_bivariate_re`'s fit pipeline calls `prior_predictive_checks(ctx, definition)`; the plain engine's calls `prior_predictive_checks(ctx)`. The audit used the second form for both, so `prior_child_checks` never ran — the unseen-child trajectories, the nested Beta-Binomial counts and the induced joint association, which are exactly the figures [#233](https://github.com/dseinternational/vocabulary-growth/issues/233) added because a child-effect model's prior figures contained no child.

**Seven registered sensitivity variants were unreachable.** `_RUNNER_BY_KEY` in `fit_sensitivity.py` (copied into `refit_hightune.py`) had no entry for VG16, VG21 or VG23, while the variant registry holds five for VG16 and one each for VG21 and VG23. `fit_sensitivity.py vg16 lag-gap-12` exited 1 with "No sensitivity variants for model: vg16". Not in the issue; found by deriving the map.

**VG14 could not finish a fresh fit.** `TrivariateModelSamples` declared `g_sign_obs` and `r_obs`, and `extract_model_samples` read them from `trace.posterior`. Both are `obs_id`-dimensioned deterministics, so since 2026-08-23 the sampler is told not to store them and the read raises `KeyError` — after sampling and after posterior prediction, which on a reporting fit is hours in. Nothing consumed either field. `r_obs` stays in the graph because the likelihood uses it; that needs no stored draws.

**VG22's priors table omitted its whole factor block.** `_PRIOR_SPECS` maps one fitted parameter to one definition field, and the factor block fits neither half of that shape: the loading entries are a family whose size depends on `rank`, and the two rate scales live on the factor spec while the two level scales are inherited from the parent's own scalar fields. The model page recorded the gap in prose, which is better than hiding it and worse than fixing it.

**And three more priors had no row either.** None in the issue; all found immediately by the coverage check written for VG22. VG15's Dirichlet-Multinomial concentration `log_conc` has its own prior figure on the VG15 page and is named there as a prior-sensitivity target, and the table beside it did not mention it. Among the registered sensitivity variants — which render the model of record's own template, so a gap shows up on a real page — `fallback-dispersion` samples `log_kappa_s_fallback` for which there was no row at all, and `a1-tau-age-varying` names its ratio parameter `log_tau_subj_u_ratio`, a shape the block row's coverage rule did not recognise. Same class, four different places, noticed by nobody — which is the argument for a contract over a corrected list.

## 2. The catalogue

`vocab_growth.models.catalogue` holds one `RegisteredModel` per key in `MODEL_REGISTRY`, each carrying an `EngineAdapter`: the engine module plus the names of its prepare, priors, build, prior-check, plot, frame-builder, stage-factory and fit hooks, and the calling convention for the two that vary. Everything downstream is derived — `FRAME_BUILDERS`, `regenerate_plots`'s `ENGINES`/`ENGINE_BY_MODEL`, the prior audit's dispatch, both sensitivity scripts' runner map, `recovery/spec.py`'s stage factory, and `fit_model.py`'s wrapper lookup. Registering a model is now one catalogue entry plus its definition, wrapper and report prose.

`FRAME_BUILDERS` was checked byte-for-byte against the table it replaced before anything else changed, because every fit's `data.analysis_frame_hash` is validated against what that mapping produces. The derived table is identical, so no fitted output moved.

### It is outside the statistical definition, deliberately

A fit is validated by comparing the manifest's recorded definition field for field, so adding a field to a definition dataclass invalidates every existing fit of that class. Engine identity, plot hooks and report templates are implementation facts that must be free to change without a refit. They therefore live in a record keyed by model id, not in the dataclass.

### Engine identity is declared, not inferred

VG05 and VG07 share `BivariateModelDefinition` and run on different engines, so the class cannot decide. `tests/test_model_catalogue.py` pins each declaration against what the model's own wrapper module imports, and separately substitutes the engine's fit function and calls `model_vgNN.fit("test")` to check the wrapper reaches _that_ engine with _that_ definition — exercised rather than read, because the wrapper binds the function at import time.

### `univariate_re` declares no plot hook, with a reason

The engine re-exports `run_standard_plots` and `extract_model_samples` from `common`, so a replot path for VG11/VG12 is within reach. It is not claimed, because the posterior-predictive stage is this engine's own (`sample_posterior_predictive_re`) and redrawing either model out of a fit has never been exercised. Claiming support that has not been run would turn a clear refusal into a wrong figure. The adapter carries `replot_note` instead, and `regenerate_plots` prints it — the exemption moved out of a test's hard-coded set into the record, where the script can read it.

## 3. Contracts

Nothing here samples; the heaviest thing is importing the engine modules.

- Every declared hook resolves; the optional ones are all-or-nothing; a missing hook raises naming both the engine and the field.
- The prior-check and plot calling conventions match the stages' signatures. Passing the wrong one raises only when that stage is reached, which for a script that runs a single stage is at the point of use.
- Frame builders take the definition and nothing else, so they stay runnable outside a fit.
- Every model has a report template, and every template has a model.
- **No engine may read a posterior variable on `obs_id` or `all_id`.** Checked by an AST walk over the engine sources rather than as a list of names, so it covers every engine including the ones no CI job samples. This is the VG14 defect stated as the rule it violates.
- Every reported parameter is either rendered in the priors table or exempt with a reason (`report_cells.prior_coverage`). Checked against each model's **real graph** — the same variable set `common.diagnostics_var_names` writes into `diagnostics.csv`, which is what the table gates on — so it is twenty graph builds and no sampling. All twenty are clean; getting there took the two missing rows above plus three corrections to the coverage rules, each of which had been hiding a family rather than covering it: the dispersion rows carry the outcome as a **suffix** on several stems (`kappa_min_u`, `a_kappa_s`, `b_kappa_mag_sign`), not as the `kappa_u` prefix the field is named with; a subject-scale field holding a block renders one row for the block's own `_0`/`_1`/`_rho` parameters; and the trend deterministics (`slope_q`, `intercept_u`, `ell_sign`) are functions of priors that carry their own rows.

  Variants are checked too, for the reason above. The exemptions stay short and are all structural: the non-centred offsets (`*_raw`, `*_z`, `z_*`), the trend deterministics, and `psi`/`conc`, whose priors are placed and reported on the log scale. Four names are deliberately **not** exempt — `tau_subject`, `tau_subj_u`, `tau_subj_q`, `rho_uq` — because each is sampled in some models and derived in others, and an unconditional exemption would absorb the loss of a row that VG10, VG19 or VG20 does need. They are resolved per model from the definition instead. A test asserts that nothing `_PRIOR_SPECS` renders is also exempt, which is what stops that shortcut being taken again.

- The table also **says so on the page** when it is incomplete, so the next gap discloses itself where a reader will see it rather than waiting for someone to write a sentence about it by hand.

The observation-deterministic module was entirely marked `slow`, so its struct contract — the one that should have caught VG14 — was not running on any pull request. The mark now sits on the tests that draw the two-real-fits fixture; the pure rule tests run in the fast job.

## 4. Not done here

VG17/VG18 (issue finding 4) are untouched and deliberately absent from the catalogue: an entry would assert a supported lifecycle they do not have. Their query grid runs to 90 months against a 12–66 month observation window with no explicit GP domain, and their custom fit path bypasses the shared manifest, staged promotion, calibration, LOO and the convergence gate. Clipping the grid and widening the domain are different statistical choices, so it is a model decision, to be coordinated with [#266](https://github.com/dseinternational/vocabulary-growth/issues/266).

Steps 5–8 of the issue — versioned semantic manifest payloads, the typed `SubjectEffectPlan`, targeted static type checking, builder seams and frozen definitions — are also not done.
