# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared definitions for standalone model reports and the technical report.

``scripts/build_glossary.py`` generates the glossary chapter from these entries.
``render_glossary`` prints the subset needed by each model report. Definitions
assume familiarity with probability and calculus, but not Bayesian modelling.
"""

from __future__ import annotations

# Render terms in conceptual order, from model structure to interpretation and
# sampling checks, regardless of the caller's requested order.
GLOSSARY: dict[str, str] = {
    # -- The modelling framework --
    "Bayesian inference": (
        "A framework that combines prior information with the likelihood of the "
        "observed data to obtain a posterior distribution."
    ),
    "Prior distribution": (
        "What the model assumes about a parameter before seeing the data. Priors "
        "here are mostly weak -- wide enough to let the data decide -- but a few "
        "are deliberately informative, and those are labelled where they are used."
    ),
    "Posterior distribution": (
        "What the model believes about a parameter after seeing the data. Every "
        "number in these reports is a summary of a posterior: a median, and an "
        "interval around it."
    ),
    "Likelihood": (
        "The probability the model assigns to the observed data for a given set "
        "of parameter values."
    ),
    "Estimand": (
        "The quantity a number is meant to estimate. Naming it matters here "
        "because the same figure can show a population average or a prediction "
        "for one child, and the two differ by a factor of ten in width."
    ),
    # -- How vocabulary is modelled --
    "Reference inventory": (
        "The common 810-word scale every vocabulary count is expressed against. "
        "Studies used different checklists, so counts are harmonised onto this "
        "scale to be comparable. Some children completed the full DSE checklists; "
        "other studies used shorter forms whose counts are treated on the same "
        "reference scale."
    ),
    "Logit": (
        "The transformation $\\operatorname{logit}(p) = \\log\\!\\big(p/(1-p)\\big)$, "
        "which stretches a proportion confined to $(0,1)$ onto the whole real "
        "line. Trends are modelled on this scale so that a straight line can "
        "never predict a proportion below 0 or above 1. A difference of 1 on the "
        "logit scale multiplies the odds by $e \\approx 2.7$."
    ),
    "Odds and odds ratio": (
        "The odds of an outcome with probability $p$ are $p/(1-p)$. An odds "
        "ratio compares two sets of odds; it equals 1 when the two are the same. "
        "Random-effect scales and the sign--speech association are reported this "
        "way because it is the natural scale of a logit model."
    ),
    "Beta-Binomial distribution": (
        "A distribution for bounded counts that permits more variation than an "
        "ordinary Binomial distribution. In this study it represents "
        "heterogeneity in vocabulary scores among observations with similar "
        "expected proportions."
    ),
    "Concentration ($\\kappa$)": (
        "The Beta-Binomial parameter controlling extra variation around the "
        "mean. **Larger $\\kappa$ means *less* variability** (approaching an "
        "ordinary Binomial); smaller $\\kappa$ allows more. The direction is "
        "counter-intuitive, so read the $\\kappa$ figures as *falling $\\kappa$ = "
        "same-age administrations becoming more spread out*. It is marginal "
        "count dispersion, not a between-child quantity: in models without "
        "child or study effects it mixes between-child, between-source and "
        "repeat-visit variation, and in models with subject random intercepts "
        "it is an observation-level residual — in neither case a measure of "
        "how much children differ."
    ),
    "Overdispersion and variance inflation": (
        "How much more variable the observed counts are than an ordinary "
        "Binomial would predict. The variance inflation factor (VIF) reported "
        "alongside $\\kappa$ states this as a multiple: VIF 50 means the observed "
        "spread is fifty times the Binomial variance at the same mean."
    ),
    "Conditional probability": (
        "The probability of one outcome given another. In the joint models, "
        "$q(a)=P(\\text{spoken}\\mid\\text{understood},a)$ is the fraction of "
        "understood words expected also to be spoken at age $a$."
    ),
    "Production ratio ($q$)": (
        "The fraction of the words a child understands that the same child also "
        "says, at a given age. Spoken vocabulary is modelled as understood "
        "vocabulary times this ratio, so $q$ is what separates *knowing more "
        "words* from *saying more of the words you know*."
    ),
    "Age anchor": (
        "A reference age at which a prior is placed on an expected vocabulary "
        "proportion. Two anchors define the broad linear trend on the logit "
        "scale in a more interpretable way than an intercept at age zero. The "
        "anchors parameterise the linear component: unless the model pins its "
        "GP at a reference age, the fitted trajectory at an anchor age is the "
        "anchor plus the GP's deviation there."
    ),
    # -- The flexible part of the trend --
    "Gaussian process (GP)": (
        "A prior over smooth functions, letting the trend bend away from a "
        "straight line by as much as the data support, without the analyst "
        "choosing the shape in advance."
    ),
    "HSGP (Hilbert-space Gaussian process)": (
        "A finite basis-function approximation to a Gaussian process. It gives "
        "nearly the same fit at a small fraction of the computational cost, "
        "which is what makes these models feasible to sample."
    ),
    "GP length-scale ($\\ell$)": (
        "How far apart in age two points must be before the GP lets them move "
        "independently. Short length-scales permit wiggly trends; long ones "
        "force near-linearity. It is given a prior over a stated window in "
        "months, so the fitted value should be read against that window."
    ),
    "GP amplitude ($\\eta$)": (
        "How far the GP may depart from the anchor-defined straight line, on the "
        "logit scale. Small $\\eta$ keeps the trend close to linear."
    ),
    "GP anchor": (
        "A constraint setting the GP contribution to zero at a reference age. "
        "Anchored models first project out mean-like components, then subtract "
        "the GP value at that age. The subtraction can restore a constant "
        "component, so the result need not remain orthogonal to the trend basis. "
        "The constraint helps distinguish model components; it is not a claim "
        "about development."
    ),
    "Mean clamp": (
        "Holding a fitted mean level above the highest age anchor rather than "
        "letting the trend continue to extrapolate. Applied where the data run "
        "out but the reporting grid does not, so that a curve does not imply "
        "evidence that no observation supports."
    ),
    # -- Hierarchy --
    "Random intercept": (
        "A per-group offset -- one per study, or one per child -- added to the "
        "trend on the logit scale, letting groups sit systematically above or "
        "below the population curve."
    ),
    "Partial pooling": (
        "The compromise a hierarchical model strikes between treating every "
        "group separately and ignoring groups entirely. Small groups are pulled "
        "towards the population mean; large ones are left closer to their own "
        "data. This is why a study contributing few children moves the fitted "
        "curve less than its raw average would."
    ),
    "Between-child heterogeneity ($\\tau$)": (
        "The standard deviation, across children of the same age, of a child's "
        "own value on the logit scale -- how much children within a population "
        "differ from one another. It is the scale of the subject random "
        "intercept, and it is a different quantity from the concentration "
        "$\\kappa$: in a model carrying subject random effects, persistent "
        "between-child differences are absorbed by $\\tau$ and $\\kappa$ "
        "describes what remains. As an odds multiplier, a $\\tau$ of 0.8 means a "
        "child one standard deviation above the centre has about $e^{0.8} "
        "\\approx 2.2$ times the odds of a child at the centre. That centre is "
        "the *median* child -- the effects are symmetric around zero on the "
        "logit scale and the inverse-logit is monotone -- not the arithmetic "
        "average of children's counts."
    ),
    "Non-centred parameterisation": (
        "Writing a random effect as $\\delta = \\tau z$ with $z \\sim "
        "\\mathcal{N}(0,1)$ rather than drawing $\\delta$ directly from "
        "$\\mathcal{N}(0,\\tau)$. The two describe the same distribution, but the "
        "first is far easier for the sampler to explore when $\\tau$ is small."
    ),
    "Sum-to-zero study effects": (
        "A constraint making the study offsets add to zero, so that the "
        "population curve is the *average study* rather than an arbitrary "
        "baseline. Without it the overall level and the study offsets are not "
        "separately identified."
    ),
    "Population-level and subject-marginal": (
        "Two different predictions. A **population-level** curve sets all random "
        "effects to zero. It describes a reference profile. For a single "
        "inverse-logit with a symmetric effect this gives the median latent "
        "proportion, but a product of two such probabilities need not have its "
        "median at zero effects. Neither is generally a mean across children. Its interval "
        "reflects uncertainty in that curve. A **subject-marginal** prediction "
        "draws a new child's random effect too: it answers *where would one more "
        "child fall?* Include observation noise too when predicting a questionnaire "
        "count. The width and target must be read from the reported quantity."
    ),
    # -- Reading the numbers --
    "Credible interval": (
        "An interval containing a stated proportion of the posterior "
        "probability for a parameter or derived quantity, conditional on the "
        "model, priors, and data. Unlike a confidence interval, it *is* a "
        "probability statement about the quantity."
    ),
    "Equal-tailed interval (ETI)": (
        "A credible interval with equal probability excluded from each tail -- "
        "an 89% ETI runs from the 5.5th to the 94.5th percentile. **This study "
        "reports an 89% outer and a 50% inner ETI by default.** 89% rather than "
        "95% is a convention that avoids implying a decision threshold."
    ),
    "Highest-density interval (HDI)": (
        "The narrowest interval containing the stated probability. It differs "
        "from an ETI for skewed posteriors, and is used here for a short list of "
        "quantities where the skew matters."
    ),
    "Cross-sectional age derivative": (
        "The slope of the fitted age trajectory, in words per month. It "
        "describes how expected vocabulary differs between children of "
        "different ages, **not** how fast any individual child learns. Because "
        "most of these data are cross-sectional, the within-child learning rate "
        "is not what is being estimated."
    ),
    "Posterior predictive check": (
        "Simulating new observations from the fitted model and comparing them "
        "with the real ones. Systematic mismatch is evidence the model is "
        "missing something."
    ),
    "Prior predictive check": (
        "The same idea run *before* the data: simulating from the priors alone "
        "to confirm they permit plausible vocabularies and exclude absurd ones."
    ),
    "PMF and CDF": (
        "The probability mass function gives the probability of each exact word "
        "count; the cumulative distribution function gives the probability of "
        "that count or fewer. The CDF is the more useful of the two for "
        "questions like *what fraction of children say 50 words or fewer at this "
        "age?*"
    ),
    "PIT (probability integral transform)": (
        "Where each observation falls within its own predictive distribution. If "
        "the model is well calibrated these values are spread uniformly between "
        "0 and 1; clustering in the middle means the predictions are wider than "
        "the data warrant."
    ),
    "LOO and ELPD": (
        "Leave-one-out cross-validation, summarised by the expected log "
        "predictive density. It estimates how well the model would predict an "
        "observation it had not seen, and is used to compare models rather than "
        "to judge one in isolation."
    ),
    # -- Whether to believe it --
    "MCMC (Markov chain Monte Carlo)": (
        "The family of algorithms used to draw samples from a posterior that "
        "cannot be written down in closed form."
    ),
    "NUTS (No-U-Turn Sampler)": (
        "The gradient-based MCMC algorithm used here. It adapts its own step "
        "size and trajectory length rather than requiring them to be tuned."
    ),
    "Chain": (
        "One sequence of samples generated by MCMC. Several chains started from "
        "different initial values help diagnose whether sampling reached the "
        "same posterior distribution."
    ),
    "R-hat ($\\hat{R}$)": (
        "A convergence diagnostic comparing variation within each chain to "
        "variation between chains. Values near 1 indicate agreement; this "
        "project's usual reporting limit is 1.01. A narrowly registered exception "
        "can permit a named failure with disclosure; passing the limit alone "
        "does not prove convergence."
    ),
    "Effective sample size (ESS)": (
        "The number of independent samples the correlated MCMC draws are worth. "
        "This project requires at least 400 for every parameter."
    ),
    "Divergent transition": (
        "A warning that the NUTS sampler could not accurately follow part of the "
        "posterior geometry. Divergences are one of the two soft-tier "
        "convergence checks: a reporting fit that records any is marked with a "
        "sampling caveat and its interval bounds read with extra caution, rather "
        "than being discarded."
    ),
    "BFMI (Bayesian fraction of missing information)": (
        "A diagnostic of how well the sampler explores the posterior's energy "
        "distribution. Low values -- below 0.3 here -- can signal inefficient "
        "exploration, and are recorded as a sampling caveat."
    ),
    "Sensitivity analysis": (
        "Refitting the model with one deliberate change -- a different prior, a "
        "different inclusion rule -- to establish whether a conclusion depends "
        "on that choice."
    ),
    # -- Signing models --
    "Copula": (
        "A construction that joins two outcomes into a joint distribution while "
        "preserving the marginal distributions at fixed parameter values. "
        "Refitting a joint model can change the estimated marginal trajectories "
        "because its parameters share information across likelihood terms."
    ),
    "Association parameter ($\\psi$)": (
        "The odds ratio measuring how much signing and speaking a given "
        "understood word go together. $\\psi = 1$ means independence; $\\psi > 1$ "
        "means a word a child signs is *more* likely to be a word they also say. "
        "At fixed spoken and signed shares, $\\psi > 1$ increases overlap and "
        "reduces the expected expressive union relative to independence; "
        "$\\psi < 1$ has the reverse effect."
    ),
}


def render_glossary(terms: list[str] | None = None, *, title: str = "Terms used in this report") -> None:
    """Print a collapsible glossary for a report cell with ``#| output: asis``.

    ``terms`` selects the subset this model needs; ``None`` prints all of them.
    An unknown term raises rather than being skipped, so a typo in a template
    fails the render instead of silently dropping the definition the reader
    needed.
    """
    if terms is None:
        selected = list(GLOSSARY)
    else:
        unknown = [term for term in terms if term not in GLOSSARY]
        if unknown:
            raise KeyError(
                f"Not in the glossary: {', '.join(unknown)}. "
                f"Add them to vocab_growth.glossary.GLOSSARY."
            )
        # Definition order, not call order, so every report reads the same way.
        wanted = set(terms)
        selected = [term for term in GLOSSARY if term in wanted]

    print(f'::: {{.callout-note collapse="true" title="{title}"}}')
    print()
    for term in selected:
        print(f"**{term}**")
        print()
        print(f": {GLOSSARY[term]}")
        print()
    print(":::")
