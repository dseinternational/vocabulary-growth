# Read and build a vocabulary model

> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-6).

This guide follows the code that builds the study's models. Start with the executable [model walkthrough](../../examples/model_walkthrough.py). It supplies twelve invented assessments from six children in three studies. It builds models and draws from a prior, but does not estimate a posterior from those counts or write report files.

A prior describes uncertainty about the model's parameters before using these assessments. The likelihood assigns probabilities to observed counts at specified parameter values. Bayesian inference combines the prior and likelihood to obtain the posterior, which describes uncertainty after using the counts. Sampling approximates that distribution with draws.

From the repository root, run:

```bash
uv sync --locked
uv run python examples/model_walkthrough.py
```

On Windows, set `$env:PYTHONUTF8='1'` in PowerShell first. The example uses percent-format cells, which Jupytext can open as a notebook. Read the cells in order, or import the file and call its functions in an interactive Python session.

## Begin with the quantities being measured

An **observation row** is one assessment at one age. A child can have several rows, and a study can contain many children. `age` is in months. `subject_code` and `study_code` are integer positions that connect each observation to the corresponding child and study. They are identifiers, not measured scores.

Let `N` be an inventory size and `U` the number of its words understood. The probability `p_U` describes the expected understood fraction at specified model parameters. The expected count is `N * p_U`. A mean of 0.2 therefore corresponds to 20 words on a 100-word checklist.

The distribution exercise uses `N = 100` for easy arithmetic. The model builds retain the registered 810-word reference and use invented counts on that scale. Preparing real study data also requires the source, measurement and exclusion rules in the analysis-frame builders. Do not use the example's direct frame insertion as a substitute for those rules.

## Allow counts to vary around their mean

A Binomial distribution with parameters `N` and `p` has mean `N*p` and variance `N*p*(1-p)`. Variance describes the expected squared distance from the mean; its square root is the standard deviation, in words.

The Beta-Binomial adds a positive **concentration**, `kappa`. Its mean is still `N*p`, while its variance is:

```text
N * p * (1 - p) * (N + kappa) / (1 + kappa)
```

For `N > 1` and `0 < p < 1`, smaller concentration permits more count variation at the same mean. As concentration grows, this variance approaches the Binomial variance. In PyMC, supply `n=N`, `alpha=p*kappa` and `beta=(1-p)*kappa`. These results follow by substitution into the [PyMC Beta-Binomial formulas](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.BetaBinomial.html).

`count_variation()` calculates the moments independently with SciPy. At `N = 100` and `p = 0.2`, all four distributions have mean 20. The Binomial variance is 16; Beta-Binomial concentrations 2, 20 and 200 give variances about 544, 91.43 and 23.88. The tests compare these calculations with the formula above.

The fitted models allow concentration to depend on age. This describes changes in variation around the expected count. It does not itself change that expected count.

## Let the expected proportion change with age

The **logit** transforms a probability into a number without finite bounds:

```text
logit(p) = log(p / (1 - p))
inverse_logit(x) = 1 / (1 + exp(-x))
```

The code can therefore add age, study and child terms on the logit scale, then apply the inverse transformation to obtain a probability between zero and one. The same shift on the logit scale has different effects on probability depending on the starting value.

In VG01, the mean combines a logit-linear age trend with a Gaussian process departure. A Gaussian process defines a probability distribution over functions. Here it lets the age curve bend around the simpler trend. The implementation uses an HSGP, a finite basis approximation to that process. Begin with the roles of the trend and departure; the basis calculation can wait until you have traced the likelihood.

The code standardises age for its calculations. `AgeGrids` records where observation ages, plot ages, query ages and the optional anchor sit in the combined array. Plot and query ages are places to evaluate the model, not extra observations. Adding plot points must not change the prior at the observed ages. The regression checks test that property.

The anchor priors can favour increasing vocabulary. They do not force it, and the GP can produce local decreases. The signing mean uses three anchors joined by two straight segments on the logit scale. Its middle anchor can move, and it is not necessarily the highest point of the complete curve.

## Add speech conditional on comprehension

VG05 introduces a second count, `S`, for spoken words. The model treats spoken words as a subset of understood words. In mean-and-concentration shorthand:

```text
U     ~ BetaBinomial(N, p_U, kappa_U)
S | U ~ BetaBinomial(U, q,   kappa_S)
```

This is mathematical shorthand, not PyMC's positional argument order. The corresponding PyMC call for speech uses `n=U`, `alpha=q*kappa_S` and `beta=(1-q)*kappa_S`.

Here `q` is the expected fraction of understood words that are spoken. At fixed parameter values, `p_U*q` is the expected spoken fraction of the full inventory. A child who understands 200 words and has `q = 0.25` has an expected spoken count of 50. The distribution still allows their realised count to differ from 50.

`prepare_bivariate_observations` checks counts before converting them to integers and records which likelihood each row enters. A missing count is different from an observed zero. The example removes one comprehension count while retaining its spoken count. The default treatment keeps that speech observation through a separate full-inventory Beta-Binomial likelihood with mean proportion `p_U*q`. This is a specified fallback distribution; it is not the exact distribution obtained by integrating out an unobserved `U` from the nested pair. The `paired_only` sensitivity omits that spoken likelihood term.

## Separate study differences from persistent child differences

VG07 adds study offsets. VG10 also adds child offsets. For example, its comprehension logit has the form:

```text
age function + study offset for this row + child offset for this row
```

A child's offset is shared across that child's assessments. It represents a persistent difference from the age curve, while the count distribution permits variation on each assessment. These are different sources of variation. Repeated assessments help distinguish them.

Study offsets sum to zero over a chosen set of studies. For `K >= 2` studies, the shared helper scales PyMC's zero-sum Normal so each offset has conditional prior standard deviation `tau_study`. With one study, its offset must be zero and the data cannot estimate a between-study contrast. The scale then retains its prior. See the covariance definition in [PyMC's ZeroSumNormal documentation](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ZeroSumNormal.html).

The bivariate models define this reference over all retained studies, including studies without a direct observation of one outcome. The joint signing engine instead uses the studies whose retained likelihood terms involve each outcome. These are existing, different reference choices. The shared helper makes the choice explicit; it does not make the choices statistically equivalent.

VG20 allows the comprehension and spoken-ratio child offsets to be correlated. Its `rho_uq` ranges from -1 to 1. Positive values mean that children above their age curve for comprehension also tend to speak a larger share of what they understand; negative values describe the opposite tendency. Within this Normal model for child offsets, correlation zero gives independence. Neither the definition nor a correlation between previously fitted child estimates establishes the sign or size that this parameter should have.

## Count overlapping spoken and signed words

Adding spoken and signed counts double-counts words a child both speaks and signs. The joint engine instead uses four categories among understood words, in this order:

1. Neither signed nor spoken.
2. Signed only.
3. Spoken only.
4. Both signed and spoken.

The example evaluates `composition_probabilities` with signed share `r = 0.4`, spoken share `q = 0.5` and odds ratio `psi = 1/6`. It returns `[0.2, 0.3, 0.4, 0.1]`. The categories sum to one, the signed categories sum to 0.4 and the spoken categories sum to 0.5. The total produced fraction among understood words is 0.8.

Here the odds ratio is `P(neither)*P(both) / (P(sign only)*P(speech only))`. A value of one gives independence. At fixed marginal shares, values above one imply more overlap than independence, and values below one imply less.

Independence gives overlap `r*q = 0.2` and produced fraction 0.7 for these same margins. Independence is therefore a comparison assumption, not a general upper bound on production. Multiplying either produced fraction by `N*p_U` gives the corresponding expected total count at those parameter values.

`build_composition_likelihood` links those probabilities to the observed category counts through a Dirichlet-Multinomial distribution. That distribution allows variation beyond a Multinomial at fixed probabilities. A source that records only produced words has three categories. Its likelihood conditions on production and retains the concentration belonging to those categories, as the distribution requires.

## Keep building, fitting and prediction distinct

`build_example()` configures the real registered priors, creates a `ModelFitContext` and calls the engine's `build_model_graph`. A graph records the probability model. Its existence is not evidence that sampling has succeeded. `details.tables` contains the build settings for inspection. Setting `report_build=False` also suppresses the prior-configuration figures and headings.

The example draws eight sets of prior values for VG01's query probabilities and concentrations. These express assumptions before the counts update them. Eight draws are an illustration, not a precision check. Posterior sampling and its diagnostics belong to the fitting pipeline described in the [model inventory](../models/README.md).

When reading later predictions, distinguish these quantities:

| Quantity                         | What is being calculated                                                                          |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| Curve at zero offsets            | The age function evaluated with study and child offsets set to zero.                              |
| Average over children            | Expected probabilities or counts averaged over a specified distribution of child effects.         |
| Expected curve for one new child | One draw of that child's persistent effects, used across the queried ages.                        |
| Future observed count            | A count drawn from the observation distribution as well as the uncertainty in the expected curve. |

Applying the inverse logit after averaging offsets generally differs from averaging probabilities. Thus a zero-offset curve need not be the population average. Study membership and available history also need to be specified when making a prediction.

## Follow the source and try changes

| Question                              | Source to read                                                                                                                                                                                                   |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Which model and priors are selected?  | [definitions.py](../../src/vocab_growth/models/definitions.py), [catalogue.py](../../src/vocab_growth/models/catalogue.py)                                                                                       |
| Which counts enter which likelihood?  | [observation_arrays.py](../../src/vocab_growth/models/observation_arrays.py), [likelihood_utils.py](../../src/vocab_growth/models/likelihood_utils.py)                                                           |
| How do age functions and grids work?  | [gp_utils.py](../../src/vocab_growth/models/gp_utils.py), [build_utils.py](../../src/vocab_growth/models/build_utils.py)                                                                                         |
| How are study and child offsets made? | [study_effects.py](../../src/vocab_growth/models/study_effects.py), [subject_effects.py](../../src/vocab_growth/models/subject_effects.py), [subject_graphs.py](../../src/vocab_growth/models/subject_graphs.py) |
| How does the joint composition work?  | [composition.py](../../src/vocab_growth/models/composition.py)                                                                                                                                                   |
| How does the pipeline report a build? | [build_reporting.py](../../src/vocab_growth/models/build_reporting.py), [common.py](../../src/vocab_growth/models/common.py)                                                                                     |

Try these exercises in the executable example:

- Change concentration while keeping `N` and `p` fixed. Predict which moments change before evaluating them.
- Compare a missing comprehension count with an observed zero. Inspect `spoken_spec` and explain the different likelihood membership.
- Restrict the frame to one study. Inspect `delta_u` and explain why estimating a study contrast is impossible.
- Change VG19's child-slope reference age to zero, 24 and 36 months. The slope multiplier is `(age - reference_age) / 12`, in years. Moving the reference age while keeping the same intercept value changes the curve; preserving a particular curve also requires changing its intercept. Priors must be considered when comparing differently centred model fits.
- Change the Plackett odds ratio while holding the two marginal shares fixed. Check all four probabilities and the fraction produced.

The executable checks are in [test_model_walkthrough.py](../../tests/test_model_walkthrough.py). The broader regression checks compare the registered graphs at saved parameter values and test the non-default cases separately.
