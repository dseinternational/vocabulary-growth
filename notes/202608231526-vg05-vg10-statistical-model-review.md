# Statistical review of VG05–VG10

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

## Status and scope

This is a read-only code and statistical review of VG05, the retired VG06, and VG07–VG10 at commit `0271871298f226591a2f90ff7ce4349dec856c47`. It covers the registered definitions, shared PyMC engines, current data frame, priors, likelihoods, hierarchical structure, posterior-predictive path, model-comparison code, statistical interpretation, reports and relevant tests. It does not replace a reporting-quality refit: no compatible traces are present in this worktree, and the 22 August comprehension-cap change deliberately marks the existing VG05 and VG07–VG10 artefacts stale. Remediation is tracked in [#236](https://github.com/dseinternational/vocabulary-growth/issues/236).

The five active graphs are internally coherent with their declared specifications. All build against a fresh reconstruction of the current database, and I found no current indexing, coordinate, random-effect scaling, HSGP, valid-pair likelihood or posterior-predictive wiring error. That code-level result is narrower than statistical adequacy. The most important shared concern is that almost one third of spoken observations use a fallback distribution that is not the marginal distribution implied by the paired model. The model lineage then moves from no hierarchy to independent child effects, leaving each rung with known limitations; VG20 has already replaced VG10 for child-level inference.

## Current model and data map

| Model | Current structure                                                                                             | Review status                                                                                                      |
| ----- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| VG05  | Joint understood and spoken; no study or child effects; unanchored GPs; legacy concentration priors           | Development step only; not adequate for substantive study-, child- or dispersion-level inference                   |
| VG06  | No current definition, module, registry entry, report or fit                                                  | Retired after the Wordbank WS comprehension defect; superseded by VG13                                             |
| VG07  | VG05 plus sum-to-zero study intercepts                                                                        | Study implementation is correct; repeated administrations remain independent                                       |
| VG08  | VG07 plus a child intercept on understood                                                                     | Child indexing is correct; persistent child variation in the production ratio remains omitted                      |
| VG09  | VG08 plus an independent child intercept on `q`, and both concentration blocks migrate to the two-anchor form | Development step; the two simultaneous changes confound the advertised one-factor lineage comparison               |
| VG10  | VG09 plus observed-row GP orthogonalisation and a 54-month GP anchor                                          | Stabilisation is correct; independent child effects and constant developmental tracking remain; superseded by VG20 |

The active bivariate frame contains 1,431 administrations, 987 understood observations, 1,428 spoken observations, 984 rows with both outcomes and 14 studies. VG08–VG10 identify 767 study-namespaced children: 432 have one administration and 335 have repeated administrations; 999 of the 1,431 rows come from repeat-measured children. All present ages and counts are finite, counts are integral and in range, and the observed ages span 8–115 months.

## Findings and proposed actions

### 1. The spoken fallback is not the marginal distribution of the paired model

For 973 valid paired observations the implementation correctly uses

$$
U_i\sim\operatorname{BetaBinomial}(810,p_{U,i},\kappa_{U,i}),\qquad
S_i\mid U_i\sim\operatorname{BetaBinomial}(U_i,q_i,\kappa_{S,i}).
$$

For a missing understood count or a recorded $S_i>U_i$, it instead uses

$$
S_i\sim\operatorname{BetaBinomial}(810,p_{U,i}q_i,\kappa_{S,i}).
$$

The routing is deliberate and robustly implemented in [`likelihood_utils.py`](../src/vocab_growth/models/likelihood_utils.py), [`common_bivariate.py`](../src/vocab_growth/models/common_bivariate.py) and [`common_bivariate_re.py`](../src/vocab_growth/models/common_bivariate_re.py). The statistical approximation is the problem. If $P$ and $Q$ are the independent latent Beta proportions underlying the paired hierarchy, then after integrating out $U$, $S\mid P,Q\sim\operatorname{Binomial}(810,PQ)$ and

$$
\operatorname{Var}(S)=Nm(1-m)+N(N-1)\left[E(P^2)E(Q^2)-m^2\right],\qquad m=p_Uq.
$$

This depends on both $\kappa_U$ and $\kappa_S$. The implemented fallback depends only on $\kappa_S$. The same parameter therefore describes residual variation in a conditional proportion on paired rows and variation in the complete product $p_Uq$ on fallback rows. The methods correctly disclose it as a pragmatic approximation, but the fitted family is a pattern-specific composite model rather than one joint generative model.

The approximation is material and not missing at random:

- 455 of 1,428 spoken rows use it, or 31.9%; 444 lack understood and 11 record $S>U$.
- The conditional rows have median age 32 months; fallback rows have median age 50 months.
- All 111 `nz_01` spoken rows and 191 of 218 `uk_01` spoken rows use the fallback.
- Reported plot/query counts use the coherent nested generator, so they target a hypothetical fully paired administration rather than the observation process used by these 455 records.

For $S>U$, either the source construct is not genuinely nested or one component is defective. Multiplying a separate understood likelihood by a marginal spoken likelihood does not reconcile the observed pair with a model whose scientific interpretation requires $S\le U$.

- [ ] Add paired-only and branch-specific-dispersion sensitivities, coordinating with #233 rather than duplicating its shared-engine work.
- [ ] Evaluate exact log-sum-exp marginalisation over latent $U$, a coherent categorical/item-state model, or a separately parameterised marginal spoken component.
- [ ] Stratify predictive calibration by paired and fallback branches and label query predictions as fully paired-administration predictions.
- [ ] Resolve the eleven $S>U$ records against source codebooks or model them as a distinct measurement construct; do not silently call them nested.

### 2. The model lineage does not support child-level inference until after VG10, and VG10 is superseded

VG05 discards study and child identity. Its Beta-Binomial can represent marginal extra-Binomial spread but cannot model covariance between repeat administrations, separate persistent child variation from occasion variation, or prevent study composition from appearing as age curvature. The estimand is an administration-weighted pooled trajectory, and positive within-child or within-study dependence generally makes effective information and trajectory precision optimistic.

VG07 correctly adds non-centred sum-to-zero study effects. Its $\sqrt{K/(K-1)}$ scaling restores each constrained study effect's marginal prior variance to $\tau^2$. It still treats repeat administrations as independent.

VG08 correctly adds a child effect to understood. Because `q` has no child effect, persistent differences in how much of understood vocabulary a child says remain in $\kappa_S$ and the spoken predictive distribution. VG09 and VG10 add both child effects but force them independent. That assumption is contradicted by the recorded fits: VG10's fitted deviations correlate $+0.151$ [0.106, 0.195], while VG20 estimates $\rho_{Uq}=+0.368$ [0.287, 0.447]. VG20 widens VG10's spoken subject-marginal intervals by 9–33% and increases the Down syndrome child-scale spoken contrast by about 15% on the logit scale. Population means remain stable; child-level results do not.

There is further weak separation in VG09/VG10. The current frame contains 157 children, contributing 307 rows, with spoken observations but no understood observation at any wave. Their likelihood chiefly identifies the product

$$
\operatorname{logit}^{-1}(f+u_i)\operatorname{logit}^{-1}(h+v_i),
$$

not $u_i$ and $v_i$ separately. Population hyperparameters and the paired children help, but the individual decomposition for spoken-only children is substantially prior- and hierarchy-driven.

- [ ] Keep VG05, VG07, VG08, VG09 and VG10 explicitly classified as development steps and prevent them from supplying reported child-level numbers.
- [ ] Use VG20 for current Down syndrome child-level understood/spoken inference; coordinate its remaining shared likelihood and validation work through #233.
- [ ] State VG05's estimand as an administration-weighted pooled trajectory and VG07's as conditional on the retained study hierarchy, not a child-level population estimate.
- [ ] Add a paired-child/repeated-child sensitivity when interpreting the two child scales or their correlation.

### 3. Constant child intercepts assume perfect developmental tracking

VG08–VG10 reuse one scalar child offset at every age. Their latent child trajectories are parallel: a child's standing is fixed, no rank changes are possible, and developmental tracking is perfect by construction rather than estimated. The direct repeated-measures investigation in [`202608141600-rank-stability-tracking.md`](202608141600-rank-stability-tracking.md) establishes this distinction. The registered random-slope successor VG19 was motivated by residual evidence: a random slope improved spoken residual fit by $2\Delta\log L=36.05$, including 20.81 among repeat-measured children.

This structure can move changing child spread, developmental drift, reporter/occasion effects and model misspecification into $\kappa(a)$. VG10's recovery evidence does not certify the split: three assessable replicates cover 0.72 of targets at the nominal 0.89 level, and `tau_subj_u` returns below its truth in every replicate. A clean convergence diagnostic does not answer this identification question.

- [ ] Describe VG08–VG10's child effects as constant-offset models that assume perfect tracking.
- [ ] Do not interpret age variation in $\kappa$ as changing between-child spread once child effects are present.
- [ ] Treat VG19 and VG20 as parallel refinements, and coordinate any combined low-rank longitudinal/cross-outcome successor with the structural work already recorded in #233.

### 4. VG05, VG07 and VG08 have severe concentration-prior conflict

These three inherit the legacy form

$$
\kappa(z)=\kappa_{\min}+\exp(a_\kappa-b_{\kappa,\mathrm{mag}}z),\qquad
b_{\kappa,\mathrm{mag}}\sim\operatorname{HalfNormal}(0.3).
$$

The recorded spoken posterior means are 1.213–1.242, beyond essentially all prior mass, with negative contraction on all three models. That prior is severely too narrow for this dispersion slope, which is also weakly identified. It is not literally a parameter "pinned at a boundary": a HalfNormal is unbounded above, and these posteriors are far from the boundary at zero. The repository wording should say extreme prior-tail conflict and negative contraction.

VG08 additionally has a child effect on understood but uses the inherited marginal concentration calibration, while spoken has no equivalent child effect. Its two $\kappa$ blocks therefore do not share a clean conditional or marginal interpretation.

- [ ] Do not quote VG05/VG07/VG08 concentration slopes, dispersion trends or variance decompositions as estimates.
- [ ] Correct the boundary wording in the model inventory and reports.
- [ ] If these historical rungs are ever made substantively usable, fit a grouping-appropriate free-sign/two-anchor concentration form and validate predictive calibration rather than migrating their stored historical comparisons silently.

### 5. VG08 to VG09 is not a one-factor comparison

VG09 is documented as VG08 plus a child intercept on `q`. The definition simultaneously replaces both legacy concentration blocks with `_DS_JOINT_UNDERSTOOD_KAPPA_RE` and `_DS_JOINT_Q_KAPPA_RE`. Changes in $\kappa$, predictive widths or variance allocation between the two fits therefore cannot be attributed solely to the new child effect.

This contradicts the stated purpose of the lineage and the commentary in `tests/test_model_definitions.py`, which says changing a prior partway through the sequence would confound the contrast while pinning exactly that transition. The model graphs themselves are valid; the causal interpretation of their difference is not.

- [ ] Amend the inventory, VG09 report and lineage comparison to state that both the `q` child effect and concentration parameterisation change.
- [ ] Do not interpret VG07/VG08/VG09 concentration differences as the isolated effect of adding each child block.

### 6. The mean anchors do not anchor the fitted curve before VG10

VG05, VG07, VG08 and VG09 add an unconstrained HSGP deviation to a logit-linear mean. Constant and linear GP directions can trade against the intercept and slope. `p_slope_low_*` and `p_slope_hi_*` are therefore coordinates of the straight-line component, not exact priors on the realised expected curve at those ages. The combined trajectory can remain well estimated while its trend/GP decomposition and individual hyperparameters are weakly identified.

VG10's observed-row projection and 54-month GP anchor correctly remove the constant and linear redundancy. The projection does not depend on reporting/query rows, and the GP equals zero at the declared reference age. VG09's earlier sampler problems are consistent with its unanchored geometry, although its last recorded fit passed the diagnostic tier and should not be described as currently non-converged without a compatible refit.

The mean anchors are also calibrated against this project frame rather than independent Down syndrome norms. This is data-informed regularisation, not external prior evidence. Its uncertainty is not propagated, so broad-anchor sensitivities remain necessary where sparse older-age conclusions depend on it.

- [ ] Relabel pre-VG10 anchors as trend-component coordinates, or apply an explicit identifiability constraint and refit if exact trajectory anchors are intended.
- [ ] Label the Down syndrome mean-anchor priors as data-informed/empirical-Bayes regularisation and do not treat posterior agreement as independent confirmation.
- [ ] Correct the claim that the soft high-age clamp guarantees a monotone fitted curve: it guarantees a differentiable mean transition, while the unconstrained GP can still create non-monotonicity.

### 7. The common 810-word scale remains a measurement assumption

Of the 1,431 active bivariate rows, 1,166 were collected on forms whose recorded ceiling is below 810; only 265 are native 810-item administrations. The dual-form crosswalk gives useful evidence that the shorter forms contain earlier-acquired items and that naive per-form rescaling would over-correct. Fixed 810 is therefore not a demonstrated denominator error.

It remains load-bearing. Near a shorter form's ceiling, the count is effectively right-censored relative to the reference inventory; study intercepts can absorb a constant level difference but not age- or ability-dependent compression, and VG05 has no study effects. Form, study, age and fallback branch are substantially confounded.

- [ ] Regenerate and foreground the registered DSE-native-only sensitivity before publication.
- [ ] Describe fixed 810 as a crosswalk-supported measurement approximation rather than a literal administered denominator.
- [ ] Consider a censored or form-linked measurement sensitivity if older/high-count conclusions depend on short-form rows.

### 8. Joint LOO uses the wrong pointwise observational unit

[`scripts/loo_compare.py`](../scripts/loo_compare.py) creates `y_joint` by concatenating understood and spoken log-likelihood factors. For a paired administration, the coherent pointwise contribution is

$$
\log p(U_i\mid\theta)+\log p(S_i\mid U_i,\theta).
$$

Concatenation makes them two held-out cases. When the understood contribution is held out, the spoken term still conditions on its observed count, leaking information relative to leave-one-administration-out; paired administrations also receive two PSIS weights. The resulting joint VG05-versus-VG07 Pareto-$k$, standard errors and comparison weights are not joint-administration LOO. Outcome-specific comparisons are unaffected.

- [ ] Map both likelihood masks back to original administration IDs and sum paired factors before calling ArviZ.
- [ ] Label outcome-specific checks precisely, and use grouped LOSO for new-child questions.

### 9. Defensive count validation is in the wrong order

Both bivariate engines cast understood counts to integer before validating their bounds. Values such as `1.9`, `-0.1` or `810.9` can become `1`, `0` or `810` and pass. Spoken counts correctly validate numeric, finite and integral values before conversion. Current understood data are integral, so no present fit changes.

- [ ] Validate understood counts for numeric type, finiteness, integrality and bounds before casting, in both engines and any LOSO/recovery consumers.
- [ ] Add fractional and just-outside-bound regression tests. Coordinate the shared-engine fix with #233 and #234.

### 10. Prior and posterior plots are described as different estimands from those implemented

The bivariate prior check samples the full graph but plots only `p_u_plot`, `p_s_plot` and `q_plot`. In the hierarchical engine these are explicitly zero-study, zero-child population mean functions; no study effects, child effects, Beta-Binomial count noise or concentration enter the displayed curves. VG08 and VG09 nevertheless call them simulated children and claim that they assess child spread. They cannot validate `tau`, $\kappa$ or count-level prior plausibility.

VG05's joint posterior trajectory plots have the opposite error. They use `y_u_plot` and `y_s_plot`, which are posterior-predictive count draws, but the captions call them median expected trajectories and explicitly say the intervals are not where a child would fall. The report's later explanation of expected count versus predictive count is correct and contradicts those captions.

The reports also call zero-random-effect inverse-logit curves the "average child". Because inverse-logit is nonlinear, these are conditional zero-effect trajectories, approximately median latent-effect trajectories, not response-scale marginal averages. After child effects are included, $\kappa$ is residual conditional overdispersion rather than persistent between-child heterogeneity.

- [ ] Relabel the current prior figures as population mean-function prior draws or add full unseen-child/unseen-study nested count draws.
- [ ] Correct VG05's joint-trajectory captions or plot `810*p_*_plot` if mean-trajectory uncertainty is intended.
- [ ] Use "conditional zero-effect trajectory" unless random effects are numerically integrated for a marginal response-scale mean.
- [ ] Describe $\kappa$ as residual concentration/overdispersion at the hierarchy level each model actually fits.

### 11. Documentation and artefact state are inconsistent

VG05 and VG07 reports describe their legacy concentration priors as priors at two reference ages; they are priors on `kappa_min`, `a_kappa` and `b_kappa_mag`. VG08 says two random-effect scale priors are new relative to VG07 although it adds only `tau_subj_u`. VG09 says its concentration calibration is bias-corrected although the current definition explicitly records that the correction was deliberately dropped. The model-role source of truth omits VG09 even though VG09's report calls it a development step.

The active definitions cap understood and `q` reporting at 72 months, while the model inventory, prior guide and parts of VG10's rationale still say 84 months or say the cap equals the high mean anchor. The 22 August handover correctly records that this reporting-only change cannot move a posterior but makes the existing artefacts incompatible with the current definitions. Nothing should be published until the eleven queued refits and report regeneration complete.

- [ ] Correct the concentration, random-effect, role, cap and bias-correction documentation.
- [ ] Complete the already-planned VG05 and VG07–VG10 refits and regenerate reports after graph-changing decisions above have been separated from reporting-only fixes.
- [ ] Do not use an old compatible posterior as evidence that the exact current artefact/report pair has been verified.

## VG06: retired, not a current model to certify

VG06 was the typically-developing counterpart to VG05. Its original Wordbank frame consumed the Words & Sentences `comprehension` field, which is a production proxy rather than an independent comprehension measurement. About 63% of that historical frame consequently asserted $U=S$, creating an artificial closing comprehension-production gap and invalidating its comprehension, production-ratio and gap conclusions above roughly 18 months. [`202605151630-vg06-ws-comprehension-issue.md`](202605151630-vg06-ws-comprehension-issue.md) records the defect.

A corrected May 2026 fit excluded the proxy and reportedly passed its contemporary diagnostics, but it used the older non-hierarchical, non-nested engine and was later removed rather than revalidated. There is now no VG06 definition, module, registry entry, report or trace. Restoring the identifier would require a new current definition, form/age/language policy, present nested-likelihood decision, contemporary priors and hierarchy, recovery and a fresh reporting-quality fit. Unless there is a new scientific reason to do that, VG13 remains the applicable typically-developing joint model.

- [ ] Keep VG06 retired and prevent historical output from being treated as current evidence.
- [ ] Use VG13 for the current typically-developing joint comparison.

## Checks completed

- The active thin `model_vgNN.py` wrappers dispatch to the intended definitions and shared engines; VG06 is correctly absent from the registry.
- All five active production graphs built against a temporary reconstruction of the current database with the expected masks, coordinates and observed likelihoods.
- Valid paired rows use $n=810,p_U,\kappa_U$ for understood and $n=U,q,\kappa_S$ for spoken.
- Study effects are non-centred, exactly sum to zero and preserve marginal prior variance; child identifiers are namespaced by study and child effects are correctly non-centred.
- The fixed HSGP domain, age standardisation, observation/plot/query slicing, q-only clamp, 72-month understood/`q` reporting cap and VG10 GP projection are internally consistent.
- Posterior prediction draws understood before spoken, guarantees $S\le U$, draws one coherent unseen child per posterior draw and holds that child across the age grid.
- The clean-checkout suite produced 745 passes and 64 expected skips. Its two failures required the generated DuckDB; both and the additional DB-dependent study-effect tests passed when pointed at the temporary prepared database.
- `ruff check src/ scripts/` passed.
- The review made no model, data or report changes.

## Proposed sequence

1. Correct the likelihood/evaluation defects: quantify or replace the spoken fallback, repair joint LOO and harden understood-count validation.
2. Correct the reports and inventory without changing historical development models: estimands, prior-predictive labels, concentration interpretation, VG08→VG09 two-factor transition, VG09 role and 72-month caps.
3. Keep VG05/VG07–VG10 as development steps and route current child-level inference through VG20, while coordinating inherited-engine work with #233.
4. Complete paired-only, fallback, repeated-child and native-810 sensitivities before interpreting $q$, $\kappa$ or child-scale quantities.
5. Complete the already-planned refits only after deciding which changes are graph-changing and which are documentation-only.

This sequence avoids refitting superseded structures merely to repair their prose, while ensuring that shared defects inherited by current models are fixed once in the common engine.
