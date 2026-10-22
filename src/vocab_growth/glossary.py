# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared definitions for standalone model reports and the technical report.

``render_glossary`` prints the subset needed by each model report. The technical
report maintains its glossary in ``docs/report/glossary.qmd``; review both when
changing a shared term. Definitions introduce the terminology used in reports.

Revised with assistance from OpenAI Codex/GPT-6.
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
        "A distribution for parameter values specified before the likelihood "
        "updates them. Some project priors use external evidence; others were "
        "calibrated on the fitted data. That reuse needs disclosure and "
        "sensitivity checks."
    ),
    "Posterior distribution": (
        "The distribution for parameters or derived quantities after combining "
        "the prior and likelihood for the observed data. Reports summarise it "
        "with estimates and intervals, conditional on the model assumptions."
    ),
    "Likelihood": (
        "The probability the model assigns to the observed data for a given set "
        "of parameter values."
    ),
    "Estimand": (
        "The quantity the analysis aims to estimate. A reference curve, an "
        "average over children and a prediction for one child are different "
        "targets, with different sources of variation."
    ),
    # -- How vocabulary is modelled --
    "Reference inventory": (
        "The common denominator of 810 items against which raw counts are "
        "modelled. Counts from shorter forms are not multiplied by a checklist "
        "length ratio. This approximation does not make the lists equivalent "
        "or measure a child's entire vocabulary."
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
        "counter-intuitive: compare concentrations at a fixed mean and "
        "denominator. Changes in either can change count spread too. With "
        "child and study effects, concentration describes residual assessment "
        "variation. Without them it can also absorb persistent differences. "
        "Concentration alone does not measure total variation between children."
    ),
    "Overdispersion and variance inflation": (
        "How much more variable the observed counts are than an ordinary "
        "Binomial would predict. The variance inflation factor (VIF) reported "
        "alongside $\\kappa$ states this as a multiple: VIF 50 means the model's "
        "conditional count variance is fifty times the Binomial variance at the "
        "same mean and denominator. It does not include additional variation "
        "from child or study effects."
    ),
    "Conditional probability": (
        "The probability of an outcome given specified information. In the "
        "joint count models, $q$ is the expected spoken share of understood "
        "words at the stated age and model profile. It does not identify the "
        "probability of producing a particular word."
    ),
    "Production ratio ($q$)": (
        "The expected fraction of understood words that a child also says, "
        "at a specified age and model profile. Spoken vocabulary is modelled as understood "
        "vocabulary times this ratio, so $q$ is what separates *knowing more "
        "words* from *saying more of the words you know*."
    ),
    "Age anchor": (
        "A reference age at which a prior is placed on an expected vocabulary "
        "proportion. Two anchors define the broad linear trend on the logit "
        "scale in a more interpretable way than an intercept at age zero. The "
        "anchors parameterise the trend component. A GP adds departures on the "
        "logit scale, even at a trend anchor unless the GP is pinned there. "
        "A smooth mean clamp can also change the trend at the upper anchor."
    ),
    # -- The flexible part of the trend --
    "Gaussian process (GP)": (
        "A distribution over functions. Here it allows smooth departures from "
        "the age trend. The kernel, amplitude, length scale and constraints "
        "specify which departures the prior permits."
    ),
    "HSGP (Hilbert-space Gaussian process)": (
        "A finite basis-function approximation to a Gaussian process. It reduces "
        "computation; its accuracy depends on the basis size, domain and "
        "length-scale range."
    ),
    "GP length-scale ($\\ell$)": (
        "The age scale over which GP departures remain associated. Shorter "
        "length scales permit faster changes; longer ones favour smoother "
        "changes. The stored unit parameter is mapped through the model's "
        "age-domain geometry and is not itself a length in months."
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
        "Smoothly levelling a trend beyond its upper age anchor. Registered "
        "Down syndrome joint defaults apply it to the spoken-share trend alone; "
        "sensitivity variants can clamp other means. The GP can still vary. "
        "The rule does not establish support for older-age predictions."
    ),
    # -- Hierarchy --
    "Random intercept": (
        "A per-group offset -- one per study, or one per child -- added to the "
        "trend on the logit scale, letting groups sit systematically above or "
        "below the population curve."
    ),
    "Partial pooling": (
        "Group estimates share a population distribution. Groups with less "
        "information generally borrow more from that distribution. The amount "
        "depends on data precision and the estimated group variation, not "
        "group size alone. Pooling does not establish measurement equivalence."
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
        "\\approx 2.2$ times the odds for that modelled proportion. Zero effects "
        "give the median of a single inverse-logit proportion with symmetric "
        "offsets; a product of proportions need not have its median there. "
        "Models with child slopes can imply an age-dependent scale."
    ),
    "Non-centred parameterisation": (
        "Writing a random effect as $\\delta = \\tau z$ with $z \\sim "
        "\\mathcal{N}(0,1)$ rather than drawing $\\delta$ directly from "
        "$\\mathcal{N}(0,\\tau)$. The two describe the same distribution, but the "
        "first can improve exploration when group effects are weakly informed. "
        "Its benefit depends on the likelihood and must be checked."
    ),
    "Sum-to-zero study effects": (
        "A constraint making the study offsets add to zero over a stated set "
        "of studies. Zero study effects define their centre on the logit scale, "
        "not generally the average of their expected counts. This chooses a "
        "reference; it does not give every study equal influence on the fit."
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
        "Simulating from the combined priors before fitting to inspect whether "
        "they imply plausible counts and shapes. Agreement with data used to "
        "calibrate those priors is not independent validation."
    ),
    "PMF and CDF": (
        "The probability mass function gives the probability of each exact word "
        "count; the cumulative distribution function gives the probability of "
        "that count or fewer. The CDF is the more useful of the two for "
        "questions like *what fraction of children say 50 words or fewer at this "
        "age?*"
    ),
    "PIT (probability integral transform)": (
        "Where each observation falls within its predictive distribution. "
        "For discrete counts, randomisation within the count's probability "
        "mass permits a uniform reference under calibration. Systematic "
        "departures can reveal bias or inappropriate predictive spread."
    ),
    "LOO and ELPD": (
        "Leave-one-out cross-validation and expected log predictive density. "
        "The prediction question depends on the likelihood term held out and "
        "the information retained. A conditional outcome score is not a "
        "forecast for a new child. Check Pareto diagnostics before interpreting "
        "the importance-sampling approximation."
    ),
    # -- Whether to believe it --
    "MCMC (Markov chain Monte Carlo)": (
        "Algorithms that draw correlated samples to approximate a probability "
        "distribution, here the posterior. Diagnostics assess the approximation."
    ),
    "NUTS (No-U-Turn Sampler)": (
        "The gradient-based MCMC algorithm used here. It adapts a step size "
        "during tuning and selects trajectory lengths while sampling. The "
        "analyst still chooses sampling effort and target acceptance and "
        "checks the diagnostics."
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
        "The approximate number of independent draws containing the same "
        "information as the correlated MCMC draws for a specified summary. "
        "The usual reporting gate requires bulk and tail ESS of at least 400 "
        "for the screened parameter elements."
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
