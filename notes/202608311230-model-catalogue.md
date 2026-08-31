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

## 5. Steps 5–8 (added 2026-08-31)

Steps 1–4 above landed first. What follows is the rest of the issue's structural programme, minus step 9 (VG17/VG18), each verified against the graph baseline described below.

### The prerequisite: a graph-equivalence harness

The issue's constraints say what a structural refactor must preserve — free random-variable names **and order**, deterministic names, dimensions, coordinates, likelihood factorisation, and a fixed-point log probability to numerical tolerance — and none of it was checked. `tests/test_graph_equivalence.py` builds all twenty models on one small deterministic synthetic frame and compares against a committed baseline.

Synthetic on purpose. Building against the prepared DuckDB would tie the recorded fingerprint to the data as well, so a legitimate data change would present as a refactor failure and a real refactor failure could hide inside one; data changes are already guarded exactly by `data.analysis_frame_hash`. All twenty models build on the one frame, across all six engines, at about a second each.

**The log probability is read away from the model's own initial point, and that is the whole design.** PyMC initialises a positive parameter at its moment, which for the `HalfNormal` scales this family is built from _is_ the scale; on the log transform the Jacobian contributes `+log(sigma)` while the density contributes `-log(sigma)`, and they cancel exactly. A log probability read at the initial point is therefore **invariant to every prior scale in the model** — a 1% change to VG05's `eta_u_sigma` moved it by exactly zero, and VG09 and VG10, which differ, recorded the identical value. Measured, not reasoned about. Offsetting each coordinate by a fixed amount on the unconstrained scale breaks the cancellation, and the offsets vary along each vector so a permutation within one array is visible too.

Mutation-checked on the shapes of error the refactors could introduce: a reordered variable creation with the same names, a 1% prior-scale change, and a swap of the two whitened coordinates in VG20's Cholesky — same names, same dims, same order, caught by the log probability alone. A negligible epsilon change inside a norm correctly passes, which is what the tolerance is for.

### Step 8: immutable definitions

Twenty definitions are module-level singletons shared by every fit, sensitivity variant, recovery replicate and validator in the same process, and they were mutable dataclasses holding mutable lists. `_as_definition_subclass` shares nested prior blocks with its base **by reference** — VG20, VG22 and VG23 all carry VG10's or VG13's kappa objects — so one edit could have moved several models at once. The sensitivity override code has carried a comment about exactly that aliasing risk since it was written; freezing the blocks is what makes the sharing safe rather than merely untested.

No fitted output is invalidated, and that is checked rather than asserted: `normalise_for_json` renders a tuple and a list as the same JSON array, so every registered model's serialised definition is byte-identical before and after.

### Step 6: one child-effect plan, and targeted typing

Five structures can occupy the same definition seam — constant offset, variance partition, age-varying scale, child slope, low-rank factor — and three arrive as a scalar field holding an object, because a new field on a shared base class would invalidate every existing fit of it. "What child structure does this model have?" was answered by four selector calls, two `getattr` reads and five rejection rules interleaved with graph construction _inside_ the PyMC context, so a refusal fired part-way through a half-built model and the rules could only be tested by building one.

`models/subject_effects.resolve` answers it once, before the context is entered, as a pure PyMC-free function of the definition. Writing it found two things the old scattered form had hidden: a plan assuming two outcomes silently drops VG15's third (signing) block, and VG14 has no child-effect seam at all rather than an inactive one. The outcome set is therefore read from the definition.

The tests that pinned the old correlation resolver moved with the rules and are stronger for it: they passed argument combinations no caller could produce — `build_model_re` always derived the booleans and specs from the same definition it passed — so they exercised a state the code could not reach.

mypy now covers the four modules that _declare_ things. Narrow on purpose: the PyMC graph code is excluded and should stay excluded until these are stable, because PyTensor's tensor algebra is not usefully typed and the noise would bury real findings. It found two immediately, both annotations disagreeing with every value in the registry — `tau_subj_{u,q}_sigma` said `float` while VG19 and Proposal A1 put objects there, and the trivariate and joint `kappa_u`/`kappa_s` said `KappaPriorParams` while VG14 and VG15 both pass the two-anchor form.

### Step 5: a versioned, classified manifest payload

Raw dictionary equality over `dataclasses.asdict` has one consequence that has shaped the model API more than any statistical consideration: **adding a field with a default invalidates every historical fit of that dataclass**, even when the default reproduces exactly what those fits did. That is why VG19's child slope and Proposal A1's age-varying scale arrive through an overloaded scalar field, why VG20's correlation and VG22's factor live on sibling subclasses, and why `CLAMP_Q_ONLY` rides on `clamp_mean_above_hi_anchor`.

`models/fit_identity` classifies every field of every registered definition class as graph-affecting, data-affecting, reporting or identity. The classification is complete (checked against the registry) and fails closed (an unclassified field is graph-affecting). `BACKFILL_DEFAULTS` names the fields whose _absence_ from an older manifest is equivalent to a stated value — a claim that every fit made before the field existed behaved exactly as a fit with the field set to it, and the only thing that excuses a difference. It starts empty on purpose: nothing needs backfilling yet, and the first entry belongs with the change that adds the field it excuses.

**Every difference remains fatal, reporting and identity ones included.** The classification's job here is to say what kind of thing moved. Whether a reporting-only difference should stop a fit being published is a separate decision with a real consequence — a changed `ages_query` leaves the stored query outputs describing ages the report no longer asks for — and it is not made here.

The payload is written _alongside_ the raw `model.definition` dictionary rather than replacing it: every fit on disk carries the raw form, several readers index it directly, and the report layer reads its numbers out of it.

### Step 7: the first builder seam

`observation_arrays.prepare_bivariate_observations` takes the seventy lines at the top of `build_model_re` that derive the likelihood's arrays and masks from the frame, and makes them a pure function returning a frozen record. Three things in there have a specific past failure behind them and none could be exercised without building a model on real data: the spoken likelihood mask (#266 finding 3), the count validation that must precede the integer cast (#236, #240), and the held-out mask that keeps rows in observation space while removing them from every likelihood. `tests/test_observation_arrays.py` is those three failures, stated directly.

With the plan and this seam, `build_model_re` is 702 lines, from 772. The remaining reduction is in the trajectory, dispersion and likelihood blocks, which are not extracted here.
