# VG01-VG04 statistical-model review

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

## Status and scope

This is a read-only review of the four univariate baseline models: VG01 (Down syndrome, spoken), VG02 (Down syndrome, understood), VG03 (typically developing, spoken), and VG04 (typically developing, understood). It covers their definitions, shared PyMC engine, data frames, priors, likelihood, posterior-predictive path, statistical interpretation, calibration and reporting helpers, and relevant tests. It does not replace a reporting-quality refit: no compatible reporting-quality traces were available in this worktree, so final convergence, posterior predictive fit, Pareto-$k$ reliability and substantive posterior estimates were not independently recomputed.

Remediation is tracked in [#234](https://github.com/dseinternational/vocabulary-growth/issues/234). The issue cross-references existing work where the scope overlaps rather than duplicating it: anchor sensitivity in #147, VG04/VG12 reporting scope in #228, child-level variance separation in #229, and shared reporting corrections found independently in VG19/VG20 in #233.

The core implementation is internally coherent. I found no arithmetic, support, shape, indexing, likelihood-parameterisation, posterior-predictive or derivative error. All four production graphs built against a fresh temporary reconstruction of the current database with finite initial log densities. The important findings are instead one shared mean-anchor semantics defect, a shared discrete-calibration error, and interpretation and measurement assumptions that make the models valid only as pooled descriptive baselines rather than child-level inferential models.

## Current analysis frames

"Repeat rows" are administrations belonging to children observed more than once.

| Model | Current frame                           | Important composition                                                           |
| ----- | --------------------------------------- | ------------------------------------------------------------------------------- |
| VG01  | 1,428 rows; 767 children; 14 studies    | 995 repeat rows; 264 rows natively measured on the 810-item inventory           |
| VG02  | 987 rows; 610 children; 13 studies      | 630 repeat rows; 259 rows natively measured on the 810-item inventory           |
| VG03  | 4,075 rows; 3,122 children; 12 datasets | 1,439 repeat rows; one dataset contributes 49.3%                                |
| VG04  | 1,555 rows; 1,225 children; 7 datasets  | 589 repeat rows; one dataset contributes 46.8%; no observations above 25 months |

## Findings and proposed fixes

### 1. The advertised mean anchors do not anchor the fitted trajectory

All four models describe `p_slope_low` and `p_slope_hi` as expected proportions at their reference ages. They anchor only the straight-line component. The univariate engine calls `trend_and_gp` without `anchor_idx`; the helper constructs the line from the two proportions, then adds an unconstrained HSGP deviation. At the lower anchor,

$$
p(a_{\mathrm{low}}) = \operatorname{logit}^{-1}\!\left[\operatorname{logit}(p_{\mathrm{slope,low}}) + \eta g_{\mathrm{unit}}(a_{\mathrm{low}})\right],
$$

not `p_slope_low`, with the analogous result at the upper anchor. Consequently externally or empirically calibrated anchor priors are not priors on the full trajectory at those ages, and the linear component can trade off against the GP's constant and low-frequency directions. The priors make the posterior proper, but `p_slope_low`, `p_slope_hi`, `slope` and `intercept` are not interpretable as currently documented. The combined trajectory may remain useful if sampling is satisfactory.

- [ ] Decide whether these parameters are intended to be coordinates of the linear trend or exact anchors of the full trajectory.
- [ ] If they are trend coordinates, relabel the definitions, prior table, figures and reports, and report the induced prior on full `p(age)` at the reference ages.
- [ ] If exact trajectory anchors are intended, remove the GP line through both endpoints, add a regression test for both equalities, recalibrate the priors and refit. Pinning one reference point does not make both endpoints exact.
- [ ] Add simulation recovery for the full trajectory and its derivative; VG01-VG04 currently have no registered parameter-recovery checks.

### 2. The independent-row likelihood cannot support child-level interpretations

Data preparation retains only age and outcome, and each row receives an independent Beta-Binomial likelihood contribution. The Beta-Binomial represents marginal extra-Binomial spread but does not model covariance among repeated observations or separate persistent child differences, study or dataset composition, form effects, occasion variation and measurement error. The fitted mean is therefore an administration-weighted mixture, not a population mean giving each child or study equal influence. Positive within-child or within-study dependence generally makes trajectory uncertainty too optimistic, and age-dependent contributor composition can appear as developmental curvature.

The reports disclose the missing hierarchy, but later text still describes `kappa` as between-child variability and `Y` as where a single child might fall. `kappa` is marginal count dispersion, and `Y` is a new exchangeable administration from the fitted row-weighted mixture. Pointwise PSIS-LOO similarly assesses another row, potentially from a child or study still represented in training, rather than a new child or study.

- [ ] State the estimand as an administration-weighted pooled trajectory everywhere these baselines are reported.
- [ ] Replace "between-child spread" with "marginal extra-Binomial dispersion" and narrow the interpretation of posterior-predictive `Y`.
- [ ] Label pointwise LOO explicitly as leave-one-administration-out.
- [ ] Use VG20 for Down syndrome child-level expectations, VG11 for typically developing production and VG12 for typically developing comprehension; coordinate structural work with #229 rather than extending these baselines into new models of record.

### 3. The common 810-item scale leaves material form-age confounding

Every observation is modelled with `n_trials = 810`, although most source instruments are shorter. The project has a useful dual-form crosswalk supporting fixed 810 over naive form-length rescaling, so this is not a denominator coding error. It remains a measurement-model assumption: the shorter forms must contain the easier words, and aggregate counts cannot test that ordering.

The residual risk is largest where form and age are confounded or a form approaches its ceiling. In VG01 and VG02 only 264/1,428 and 259/987 observations, respectively, are native to the 810-item inventory, and these non-hierarchical models cannot absorb residual form offsets. VG03 has only WS observations from 26 to 30 months; 17-20% of its fitted rows at ages 28-30 are within 90% of the WS ceiling. VG04 has no comprehension observations above 25 months while its high trend anchor is 26 months; in the fitted subsample 12/25 observations at 24 months and 3/8 at 25 months are within 90% of their native form ceiling. Those high-age counts are effectively right-censored relative to the 810-word scale, but the likelihood treats them as ordinary exact counts.

- [ ] Run and report native-810 and form-restricted sensitivities for these baselines, or explicitly rely on the corresponding hierarchical models' sensitivities and state that transfer of evidence.
- [ ] Add a censored or form-linked sensitivity for the VG03/VG04 high-age tails.
- [ ] Do not interpret VG04's 26-month trend anchor as data-supported; coordinate its broad-anchor sensitivity with #147 and its reporting cap with #228.

### 4. Dispersion priors are empirical-Bayes priors fitted to the analysed outcomes

The four marginal `kappa` priors are explicitly centred on dispersion curves fitted to the models' own data. VG02's lower mean anchor is also explicitly data-informed. Calibration uncertainty is not propagated, prior predictive checks are not epistemically "before the data were seen", and a posterior direction matching the calibrated prior is not independent confirmation of that direction. The broad scales mitigate but do not remove this issue.

- [ ] Label these priors and their checks as empirical Bayes.
- [ ] Report the induced prior probability of increasing, flat and decreasing dispersion before quoting a fitted direction.
- [ ] Add sign-neutral or constant-dispersion sensitivities for scientific claims about change in dispersion.
- [ ] Coordinate mean-anchor sensitivity work with #147 rather than duplicating its registered model-of-record fits.

### 5. Deterministic mid-PIT is compared with the wrong reference distribution

The calibration helper computes deterministic mid-PIT, then compares its variance with the continuous Uniform variance $1/12$ and interprets a lower value as excessive predictive width. Deterministic mid-PIT is not Uniform for a discrete outcome even under perfect calibration: calibrated Bernoulli(0.5) observations have mid-PIT values 0.25 and 0.75, hence variance 0.0625 rather than $1/12 \approx 0.0833$. Discrete equal-tailed predictive intervals can also exceed nominal coverage solely because of discreteness.

- [ ] Replace the comparison with randomized PIT, preferably out of sample or leave one out, or use a model-simulated discrete reference distribution.
- [ ] Calibrate the expected coverage of discrete intervals rather than treating nominal continuous coverage as exact.
- [ ] Coordinate the shared reporting correction with #233, which reaches the same helper from VG19 and VG20.

### 6. Headline dispersion and growth summaries overstate posterior evidence

The shared headline table declares that spread "widens" or "narrows" by comparing two endpoint medians, supplies no interval or sign probability, and calls the result spread between children. Its "Fastest growth" row selects the maximum of the median derivative curve, including the endpoints. A boundary maximum says only that the peak was not located within the declared range, and the pointwise interval at the selected age does not represent uncertainty in the peak age.

- [ ] Calculate the endpoint dispersion contrast draw by draw and report its interval or posterior sign probability.
- [ ] Calculate peak age draw by draw over a predeclared evidence-supported range.
- [ ] When the median maximum is at a boundary, report "highest estimated rate occurs at the boundary; peak not located" rather than "fastest growth".

### 7. Lower-priority code, provenance and documentation defects

- [ ] Reconcile the active comprehension caps with the documentation. VG02 now stops at 72 months and VG04 at 25, but `docs/models/README.md` still says 84 and the VG04 report still says there is no cap. #228 already tracks the VG04/VG12 scope decision.
- [ ] Correct the general methods claim that every concentration slope is constrained negative; VG01-VG04 now use free-sign two-anchor concentration priors.
- [ ] Correct VG03's sampling rationale. The comment says 25% preserves about 1,500 rows after excluding WS, but spoken-only loading intentionally includes WS and the actual frame has 4,075 rows.
- [ ] Preserve study, child, language, form and native ceiling in the analysis frame or manifest. The current manifest's source counts are empty because univariate preparation discards `study`.
- [ ] Decouple the HSGP basis centre from reporting queries. Under locked PyMC 6.3.1, removing VG04's now-unreported 27- and 30-month queries shifts the verified basis centre from 19 to 16.5 months despite the explicit 8-30-month domain. Current configurations happen to align correctly, so this is latent regression debt rather than a current-fit defect.
- [ ] Validate count integrality before casting to integer; current source counts are integral, but a future fractional value would be silently truncated.
- [ ] Use exact `y_query` draws in selected-age PMF and CDF figures instead of the nearest point on the 500-point plot grid.
- [ ] Apply `requires_real_db` to `test_dse_native_restriction_on_the_real_pool`; a clean checkout currently fails instead of skipping when the generated DuckDB is absent.

## Checks completed

- The four thin `model_vgNN.py` wrappers dispatch to the intended definitions and shared engine.
- Model-definition validation, anchor ordering, explicit domains, reporting ages and count support checks are correct.
- The Beta-Binomial parameterisation is correct: $\alpha = p\kappa$, $\beta = (1-p)\kappa$, $E[Y] = 810p$, and the variance inflation is $(810+\kappa)/(1+\kappa)$.
- The two-age concentration algebra reproduces total $\kappa$ exactly at both concentration anchors and permits either slope sign.
- Age standardisation and observation, plot and query slicing are aligned.
- Posterior-predictive draws use the same $p$, $\kappa$ and $N=810$ as the likelihood.
- The learning-rate transformation $810p(1-p)\,df/da$ is correct.
- Reporting-quality convergence screening covers all free parameters.
- All four production graphs built against a temporary reconstruction of the current database with finite initial log densities.
- Consolidated targeted regression run: 376 passed and 5 fitted-output-dependent tests skipped.
- Ruff passed on all reviewed Python files, and the review left the worktree unchanged.

## Proposed sequence

1. Resolve the full-trajectory anchor semantics and add recovery tests before interpreting or refitting the four baselines.
2. Correct the pooled-administration, `kappa`, posterior-predictive and LOO language.
3. Repair discrete calibration and draw-wise headline calculations.
4. Run or explicitly inherit empirical-prior and native-form sensitivities.
5. Reconcile reporting caps, methods, sampling rationale, provenance and defensive validation.

This sequence separates changes that alter the model graph and require refits from reporting-only corrections. Documentation-only corrections should not be used to close the implementation issue.
