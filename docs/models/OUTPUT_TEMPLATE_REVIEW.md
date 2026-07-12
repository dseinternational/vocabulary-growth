# Model Output Template Review

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

> [!WARNING]
> Working review note. The recommendations below are based on the checked-in
> `docs/models/vgNN/index.qmd` templates, the model inventory in
> `docs/models/README.md`, the prior ledger in `docs/models/PRIORS.md`, and the
> current shared fitting engines.

## Scope

This note reviews each model output template against five questions:

- Does the template describe the model accurately and sufficiently?
- Does it describe all priors accurately and sufficiently?
- Does it show sufficient prior predictive checks?
- Does it show sufficient diagnostics?
- Does it thoroughly explore findings and interesting posterior predictions?

The report generation path copies `docs/models/vgNN/index.qmd` into each model
output directory after fitting, so these checked-in `.qmd` files are the source
templates for rendered model outputs.

## Cross-model recommendations

1. Add a consistent "Model at a glance" block to every template: population,
   outcomes, age range, observation scale, anchor ages, data inclusion/exclusion
   rules, random effects, GP anchoring, and primary derived quantities.
2. Keep the prior plots, but add plain-language prior interpretation next to
   them: probability scale, approximate word-count scale when applicable, and a
   note when a prior is data-informed or computationally stabilising.
3. Make prior predictive checks evaluative, not just visual: include a short
   checklist for floor behaviour, ceiling behaviour, smoothness, count spread,
   `q(a)` and `r(a)` plausibility, random-effect heterogeneity, and four-cell
   composition where relevant.
4. Standardise diagnostics: styled R-hat/ESS table, divergences, energy
   diagnostics, max tree depth if available, trace plot, pair plot for the most
   geometry-prone parameters, and a short "diagnostic read" paragraph.
5. Add a "Main findings to inspect" section to every report. The current
   templates often show the right artefacts but rarely say what substantive
   claims the reader should or should not take from them.
6. Add a "Limits and sensitivity targets" block for models with sparse data,
   posterior-informed priors, random effects, GP anchoring, signing, or the
   VG15 association parameter.

## Model-by-model recommendations

### VG01: DS spoken baseline

- Model description is accurate for a single-outcome, no-hierarchy baseline.
  Add explicit statements that repeated observations and study differences are
  not modelled here, so this is a baseline rather than the preferred DS spoken
  model.
- Priors are displayed but not interpreted. Add the low/high anchor
  distributions on the word-count scale, and explain the HSGP lengthscale,
  amplitude, and age-varying `kappa` roles.
- Prior predictive coverage is adequate in artefact count (`prior_samples`,
  `prior_predictions`, `prior_predictive_checks`) but needs text saying whether
  young-age floor, later-age upper tail, and count spread are plausible.
- Diagnostics are visually sufficient but need a pass/fail paragraph, especially
  for GP hyperparameters and `kappa`.
- Posterior findings should highlight spoken vocabulary trajectory, peak
  learning-rate age, uncertainty at older ages, and how this baseline compares
  to joint/hierarchical DS models.

### VG02: DS understood baseline

- Model description is accurate. Add the same no-hierarchy caveat as VG01.
- Priors need comprehension-specific interpretation: `Beta(1, 10)` at 24 months
  and broad `Beta(1.1, 1.1)` at 84 months imply intentionally weak later-age
  knowledge.
- Prior predictive text should call out whether comprehension can reach
  implausible ceilings or remain implausibly low at later ages before seeing
  data.
- Diagnostics should report whether the understood GP and age-varying
  dispersion are well behaved.
- Findings should focus on comprehension growth and uncertainty, then point to
  VG05/VG10 for comprehension-production gaps.

### VG03: TD spoken baseline

- Model description is mostly sufficient, but add that this model uses a 25%
  TD subsample and that VG11 is the full-data, study-random-effect successor.
- Priors should be interpreted at 12 and 26 months and contrasted with the DS
  anchor ages.
- Prior predictive checks should include whether the TD prior reaches plausible
  spoken counts by 24-30 months under the 810-word reference scale.
- Diagnostics should note whether subsampling creates stable estimates across
  the random seed or whether seed sensitivity should be checked.
- Findings should compare the baseline TD spoken trajectory with VG11, not only
  report it in isolation.

### VG04: TD understood baseline

- Model description accurately notes that Words & Sentences comprehension is
  excluded because it is a production proxy. Add the 25% subsampling caveat and
  reference VG12 as the full-data successor.
- Priors should explain why the early TD understood anchor is tighter
  (`Beta(1, 20)`) than the DS understood anchor.
- Prior predictive checks should assess ceiling pressure because TD
  comprehension can approach checklist limits quickly.
- Diagnostics should include a short readout as in VG03.
- Findings should report understood vocabulary growth and flag that TD
  understood/spoken relationships are better handled in VG06/VG13.

### VG05: DS baseline joint understood + spoken

- Model description is accurate and sufficiently explains `p_S = p_U * q`.
  Add a one-line statement that no study or subject random effects are included.
- Priors show the right U, `q`, and dispersion plots. Add plain-scale summaries,
  and explicitly label the baseline `q` anchors as broad regularisation.
- Prior predictive checks should add an evaluative note for the induced spoken
  trajectory, the comprehension-production gap, and whether `q(a)` is plausible
  over 12-90 months.
- Diagnostics are adequate but should focus on the bivariate geometry:
  U anchors, `q` anchors, GP hyperparameters, and both `kappa` curves.
- Posterior exploration is strong. Add a concise interpretation of `q(a)` versus
  posterior predictive spoken/understood count ratios, since the template already
  shows both.

### VG06: TD baseline joint understood + spoken

- Model description is accurate but should explicitly describe TD data handling:
  WG/Oxford bivariate rows, WS spoken-only rows, 25% sampling, and the 810-word
  reference scale.
- Priors should explain that the `q` priors are broad defaults shared with VG05,
  while U anchor ages/distributions are TD-specific.
- Prior predictive checks should focus on the younger TD age range and
  plausibility of `q(a)` by 24-30 months.
- Diagnostics should include a short readout and any seed/subsampling caveat.
- Findings should compare TD comprehension-production timing with DS joint
  models and state that VG13 is the denser young-TD bivariate refinement.

### VG07: DS joint with study random intercepts

- Model description should add a compact equation or paragraph for study random
  intercepts on U and `q`, and state that reported curves are population-level
  with study effects set to zero.
- Priors are incomplete in prose because the HalfNormal study-effect scale
  priors (`tau_u`, `tau_q`) are not interpreted. Add their logit-scale and odds
  multiplier meanings.
- Prior predictive checks should include the implied between-study variation,
  not only population trajectories.
- Diagnostics should monitor `tau_u`, `tau_q`, study intercepts, and any
  confounding with GP level.
- Findings should show what study effects change relative to VG05: population
  trajectory, uncertainty, `kappa`, and the comprehension-production gap.

### VG08: DS joint with study REs and subject RE on understood

- Model description is useful but should state the random-effect structure in a
  standard at-a-glance block.
- Priors should include `tau_subj_u` and clarify that subject-marginal posterior
  predictive summaries integrate over a new child-level effect.
- Prior predictive checks should include the implied child-to-child variation in
  understood vocabulary.
- Diagnostics should focus on `tau_subj_u`, `tau_u`, GP level, and whether the
  subject RE reduces pressure on `kappa_u`.
- Findings should compare population-level and subject-marginal predictions, and
  explain how repeated observations change uncertainty versus VG07.

### VG09: DS joint with subject REs on understood and `q`

- Model description is accurate, including the motivation for adding a subject
  RE on `q`. Add a short limitation note that this model motivated VG10 because
  the extra hierarchy can expose a GP/trend/intercept ridge.
- Priors should include `tau_subj_q` and make clear that the `q` anchors are
  still the broad baseline anchors.
- Prior predictive checks should inspect both subject-marginal U and
  subject-marginal spoken predictions.
- Diagnostics are especially important here. Add text for the known ridge-prone
  parameters: `q` GP lengthscale/amplitude, `tau_subj_q`, `tau_q`, and `kappa_s`.
- Findings should be framed as a structural stepping stone to VG10 unless
  diagnostics are clean enough for substantive use.

### VG10: DS joint with VG09 hierarchy plus stabilisation

- Model description is good. Add a clear "what changed from VG09" block: the
  per-draw GP anchor at 54 months (the `q` anchors now match VG09).
- Priors need accurate labelling: the `q` anchors are the shared
  weakly-informative DS-joint priors, not posterior-informed regularisation.
  #155 broadened the earlier VG07-posterior-derived anchors to remove the
  prior-data double-dipping; stabilisation now comes from the tightened GP
  amplitude `eta_q` and the GP anchor.
- Prior predictive checks should demonstrate that the weakly-informative `q`
  anchors with the tightened `eta_q` still allow plausible spoken trajectories
  and gaps over 12-90 months.
- Diagnostics should explicitly report whether the VG09 geometry issue is
  resolved: R-hat/ESS for `q` hyperparameters, random-effect scales, and pair
  plots around the anchored GP components.
- Findings should be the main DS understood/spoken interpretive template:
  population trajectories, subject-marginal predictions, `q(a)`, gap, learning
  rates, and dispersion.

### VG11: TD spoken with study random intercepts

- Model description should be expanded. The current univariate wording hides
  the important differences from VG03: full TD data, study random intercepts,
  minimum study-size filter, and GP anchor at 19 months.
- Priors should add `tau` for study effects and explain the GP anchor as a
  stabilisation/identifiability constraint, not a substantive prior.
- Prior predictive checks should include between-study heterogeneity and the
  retained/dropped study count table.
- Diagnostics should monitor `tau`, study intercepts, and GP anchor geometry.
- Findings should compare with VG03 and show how using all TD data changes the
  spoken trajectory and uncertainty.

### VG12: TD understood with study random intercepts

- Same structural improvements as VG11, with comprehension-specific data rules:
  WG/Oxford only, WS comprehension excluded, full data, minimum study-size
  filter, GP anchor at 19 months.
- Priors should explain the `Beta(1, 20)` low anchor and `Beta(1.5, 1.1)` high
  anchor on the word-count scale.
- Prior predictive checks should assess rapid TD comprehension growth and
  ceiling behaviour.
- Diagnostics should focus on study-effect scales and GP hyperparameters.
- Findings should compare with VG04 and then with VG13 for young bivariate TD
  comprehension-production behaviour.

### VG13: young TD joint model, ages 8-18 months

- This template needs the largest expansion. The statistical description is
  mostly accurate, but it says the study effects are "centred Normal"; the engine
  implements them with a non-centred parameterisation (`delta = tau * z`) with
  the same Normal marginal distribution.
- Add a full priors section: U anchors at 10/16 months, broad baseline `q`
  anchors, U/`q` GP lengthscale and amplitude, U/S `kappa`, `tau_u`, `tau_q`,
  study-size filtering, and GP anchor at 13 months.
- Add prior predictive checks. The bivariate engine already creates
  `prior_samples_u.png`, `prior_samples_s.png`, and `prior_samples_q.png`; embed
  and interpret them.
- Diagnostics are present but should add a diagnostic readout for study-effect
  scales and anchored GP parameters.
- Posterior exploration is currently too thin. The same bivariate plotting
  pipeline should provide many existing artefacts: `joint_trajectory_hdi`,
  `posterior_summary_s.csv`, `posterior_summary_q.csv`,
  `production_rate_by_understood`, `production_rate_predictive`,
  `understood_vs_spoken`, `understood_vs_spoken_predictive`, outcome PMFs/CDFs,
  count distributions, median trends, learning rates, and kappa plots. Add these
  so VG13 is comparable to VG05-VG10.
- Add a limitation callout: inference is intentionally restricted to 8-18
  months, and outside that window it should not be used as a general TD
  trajectory.

### VG14: DS trivariate understood + spoken + signed

- The template is one of the strongest overall. One accuracy fix is important:
  the statistical model section says each of `f_U`, `h`, and `g_sign` has a
  linear trend plus GP, but the signed ratio currently has an intercept-only mean
  plus GP. The later signed-prior section says this correctly; make the model
  section match.
- Priors are well covered, including the signed-ratio rationale. Add a compact
  prior table for signed intercept, signed lengthscale, signed amplitude, and all
  three `kappa` curves.
- Prior predictive checks are broad. Add an explicit interpretation of whether
  the signed prior permits a plausible rise-and-fall pattern without implausible
  pre-data extrapolation.
- Diagnostics should focus on the sparse signing components: signed GP
  hyperparameters, `kappa_sign`, and any pair-plot evidence of confounding.
- Posterior findings are strong. Add a clear "upper-bound" warning near every
  `p_any` section, and make the uk_02 validation gap a headline limitation rather
  than a small subsidiary table.

### VG15: DS joint sign/speech association model

- Model description is scientifically rich, but the opening paragraph should
  mention both study and subject random intercepts because the fitted definition
  includes subject REs on U, `q`, and signing.
- Priors are under-displayed relative to model complexity. Add all trajectory
  priors, not just a subset: U low/high anchors, `q` low/high anchors, U/`q`/sign
  GP length-scale parameters and amplitudes, signed intercept, all three `kappa` priors,
  `tau_u`, `tau_q`, `tau_sign`, all subject `tau` priors, `log_psi`, and
  `log_conc`.
- Label the tightened `q` priors and `log_psi` prior as data-informed
  regularisation. Add a sensitivity callout for neutral `psi`, wider `psi`, and
  alternative concentration priors.
- Prior predictive checks should include more than `r(a)` and `q(a)`: add
  `p_U`, spoken, signed, `p_any`, prior four-cell composition, and prior
  probability of `psi > 1` if the engine can generate them.
- Diagnostics should use the styled table used elsewhere, add a pair plot for
  `psi`, `conc`, signed GP hyperparameters, and random-effect scales, and
  explicitly report whether `tau_subj_sign` is data-identified.
- Posterior exploration should embed the CSVs already produced for U, spoken,
  signed, `q`, `r`, `p_any`, and `psi`, not only `r`, `p_any`, and `psi`. Add a
  section that separates population-level curves from subject-marginal
  prediction, because the four-cell likelihood deliberately excludes subject
  shifts from the `psi` composition.
- Add an extrapolation warning to the composition and `p_any` plots: the scalar
  `psi` is identified mostly by uk_02 four-cell rows from roughly 19-56 months,
  so four-cell composition outside that age support is exploratory.

## Suggested implementation order

1. Fix accuracy issues first: VG14 signed mean description, VG13 non-centred
   study-effect wording, and VG15 opening paragraph.
2. Bring VG13 up to the same bivariate-report coverage as VG05-VG10, using
   existing artefacts from the shared engine.
3. Expand VG15 priors, diagnostics, and posterior summaries to match its model
   complexity.
4. Add random-effect prior and interpretation text to VG07-VG12.
5. Add plain-language prior interpretation and diagnostic readout paragraphs to
   VG01-VG06.
6. Add cross-model sensitivity and limitation callouts for VG10, VG14, and VG15.
