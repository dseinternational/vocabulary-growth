# VG07: Bivariate model with study-level random intercepts (DS)

Date: 2026-04-12

> **Note:** This document was generated with assistance from an AI model (Claude, Anthropic) and should be independently verified.

## Motivation

Models VG02 and VG05 both show an apparent decline in expected words understood between approximately 40 and 60 months for children with Down syndrome. Investigation (see `notes/202604121055-understood-ds-decline.md`) established that this is a data composition artefact — a form of Simpson's paradox — rather than a real developmental phenomenon. Higher-scoring studies (1, 2, 6, 7) stop contributing understood data after ~50 months, while lower-scoring studies (3, 5) continue, pulling the observed distribution downward. The GP has sufficient flexibility to fit a localised dip to track this compositional shift.

A model with study-level random intercepts should absorb systematic differences between studies and produce a population-level trajectory that is less susceptible to this artefact.

## What VG07 is

VG07 is a new bivariate model (understood + spoken, Down syndrome) that extends VG05 with study-level random intercepts on both the understood trajectory and the production ratio.

The model specification adds:

$$f_U(a, s) = \text{intercept}_U + \text{slope}_U \cdot a_z + g_U(a) + \delta_U[s]$$
$$h(a, s) = \text{intercept}_q + \text{slope}_q \cdot a_z + g_q(a) + \delta_q[s]$$

where:

- $\delta_U[s] \sim \text{Normal}(0, \tau_U)$ — study-level intercept shift for understood (logit scale)
- $\delta_q[s] \sim \text{Normal}(0, \tau_q)$ — study-level intercept shift for production ratio (logit scale)
- $\tau_U \sim \text{HalfNormal}(0.5)$ — SD of study intercepts for understood
- $\tau_q \sim \text{HalfNormal}(0.5)$ — SD of study intercepts for production ratio

All other components (GP, kappa, likelihood) are identical to VG05.

## Key design decision: population-level predictions

Plot and query predictions use the population-level trajectory ($\delta = 0$), not any specific study. The random intercepts shift individual study observations up/down on the logit scale during fitting, but the reported trajectory, HDI bands, and word counts reflect the population mean. This is standard practice for mixed-effects models — condition on the random effects for fitting, marginalise over them for prediction.

## What was implemented

### New files

| File                                             | Purpose                               |
| ------------------------------------------------ | ------------------------------------- |
| `src/vocab_growth/models/model_vg07.py`          | Model module (thin wrapper)           |
| `src/vocab_growth/models/common_bivariate_re.py` | Pipeline with study random intercepts |
| `docs/models/vg07/index.qmd`                     | Quarto report template                |

### Modified files

| File                                     | Change                                                                                                                                                                                |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/vocab_growth/models/definitions.py` | Added `include_study_re`, `tau_u_sigma`, `tau_q_sigma` fields to `BivariateModelDefinition` (with defaults that preserve VG05/VG06 behaviour); added VG07 instance and registry entry |
| `scripts/fit_model.py`                   | Added VG07 import and dispatch                                                                                                                                                        |
| `scripts/upload.py`                      | Added VG07 config                                                                                                                                                                     |

### Architecture

Rather than modifying `common_bivariate.py` (which would risk affecting VG05/VG06), VG07 has a dedicated `common_bivariate_re.py` module. This module:

- Imports and reuses from `common_bivariate.py`: prior configuration, sampling, diagnostics, sample extraction, all plotting functions, posterior summary, and reporting.
- Defines a new `prepare_bivariate_re_data` that loads the `study` column and creates integer study codes.
- Defines a new `build_model_re` that extends the model with study random intercepts.
- Defines `fit_bivariate_re_model` orchestrating the full pipeline.

### Model build changes (relative to VG05)

1. **Data loading**: `["age", "understood", "spoken", "study"]` — study column now included.
2. **Study encoding**: Unique studies mapped to integer codes (0 to n_studies-1).
3. **Coordinates**: `"study_id"` added.
4. **Random effects**: `tau_u`, `delta_u`, `tau_q`, `delta_q` added as new model parameters.
5. **Observation-level**: `p_u_obs` and `q_obs` computed from `f_u + delta_u[study]` and `h + delta_q[study]` respectively, so the likelihood sees study-adjusted probabilities.
6. **Population-level**: `p_u_plot`, `p_u_query`, `q_plot`, `q_query` computed from `f_u_all` and `h_all` without study effects, so predictions reflect the population mean.

## How to fit

```bash
python scripts/fit_model.py vg07 --config dev
```

Output is written to `output/models/VG07-age-understood-spoken-ds-re/`.

## What to look for

1. Compare `posterior_predictive_median_trend_u_smoothed` between VG05 and VG07 — does the dip at 40–60 months reduce or disappear?
2. Compare `expected_learning_rate_u` — does the learning rate remain positive throughout?
3. Check `understood_vs_spoken` — does the "leftward hook" resolve?
4. Check `diagnostics.csv` — do `tau_u` and `tau_q` have reasonable posteriors? Large `tau_u` would confirm substantial between-study variation.
5. Compare `posterior_summary_u.csv` at ages 42, 48, 54, 60 months between VG05 and VG07.

## Results (dev config)

Models VG06 (TD, bivariate) and VG07 (DS with study RE, bivariate) were fitted using the `dev` sampling configuration and compared. VG05 (DS, bivariate, no RE) is included as baseline.

### Production ratio over age

**VG06 (typically developing, 9--30 months):** The production ratio follows a steep, smooth sigmoid. Children go from speaking ~5% of words understood at 9 months to ~99% by 30 months. The transition from comprehension-dominant to near-parity happens rapidly between 12 and 24 months. By 24 months, TD children speak 95% of words they understand. HDI bands are narrow throughout.

**VG07 (Down syndrome with study RE, 12--90 months):** The production ratio rises much more slowly. At 24 months DS children speak only ~14% of words understood (vs 95% for TD). The curve reaches 50% around 45 months and does not approach 90% until ~72 months. HDI bands are substantially wider, especially beyond 60 months. The trajectory is smoother than VG05 -- the study random intercepts have absorbed between-study variation.

| Age (months) | VG06 q_median | VG07 q_median | VG05 q_median |
| :----------: | :-----------: | :-----------: | :-----------: |
|      12      |     0.14      |     0.04      |     0.09      |
|      18      |     0.63      |     0.08      |     0.16      |
|      24      |     0.95      |     0.14      |     0.20      |
|      30      |     0.99      |     0.20      |     0.19      |
|      42      |      --       |     0.43      |     0.36      |
|      54      |      --       |     0.73      |     0.69      |
|      72      |      --       |     0.90      |     0.80      |
|      90      |      --       |     0.97      |     0.88      |

### Production ratio by words understood

**VG06 (TD):** Smooth, monotonically accelerating curve. At ~100 words understood, TD children speak about 20%. At ~150 words, ~50%. By ~250 words, ~75%. Approaches ceiling around 350 words understood. Very tight HDI.

**VG07 (DS):** Qualitatively different shape. The curve rises more gradually, with an inflection/steepening around 200--300 words understood. Below ~200 words, DS children speak a smaller fraction than TD children at the same comprehension level. The catch-up accelerates between 200--350 words. Beyond ~400 words understood, DS and TD children converge toward similar production ratios (~90%+). HDI is much wider, particularly in the 200--400 word range.

### Effect of study random intercepts on the understood trajectory

The comprehension dip at 40--60 months identified in VG05 is substantially reduced in VG07. The understood trajectory is now monotonically increasing:

| Age (months) | VG05 Ey_median | VG07 Ey_median | Change |
| :----------: | :------------: | :------------: | :----: |
|      42      |      290       |      235       |  -55   |
|      48      |      284       |      246       |  -38   |
|      54      |      273       |      255       |  -18   |
|      60      |      285       |      279       |   -6   |

VG07 estimates are lower overall -- the random intercepts absorb the fact that higher-scoring studies dominate earlier ages -- but the trajectory no longer shows a decline. The learning rate stays positive throughout, though it does slow between 40--55 months.

The understood-vs-spoken plot for VG07 no longer shows the pathological "leftward hook" present in VG05. The median curve is now monotonically increasing in both dimensions.

### Study random intercept estimates

- $\tau_U$ = 0.47 (SD 0.13) -- substantial between-study variation in understood scores on the logit scale
- $\tau_q$ = 0.68 (SD 0.17) -- even larger between-study variation in the production ratio

Both are well-identified with good ESS (>7000) and R-hat = 1.0. The large $\tau_q$ is consistent with different studies using different assessment instruments or protocols for spoken production.

### Summary

1. The comprehension-production gap is much larger and longer-lasting in DS than in TD. TD children close the gap by ~24 months (~250 words understood). DS children do not reach comparable production ratios until ~72 months (~400 words understood).
2. The study random intercepts resolve the Simpson's paradox artefact in the understood trajectory. The dip at 40--60 months is gone.
3. VG07 provides a more biologically plausible trajectory and should be preferred over VG05 for reporting understood word counts in the 40--70 month range.
4. These results were obtained with `test` config and should be confirmed with `rep` quality sampling.
