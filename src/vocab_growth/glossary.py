# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Canonical definitions of the terms the reports use.

The technical report carries a glossary chapter, but every model report renders
as a standalone page inside its own fitted output directory, so a reader who
lands on one of them gets none of it. Ten independent reviews of the fifteen
model reports each raised the same finding: terms that decide whether a figure
can be read at all -- which direction of :math:`\\kappa` means more variability,
whether a band is a population mean or one child, what an 89% interval is --
appear undefined on pages that then ask the reader to interpret them.

Holding the definitions here rather than in each ``.qmd`` means a term is
defined once and every report that uses it gets the same words. The report
chapter is generated from this module (``scripts/build_glossary.py``), and
:func:`vocab_growth.glossary.render_glossary` prints the subset a given model
needs into a report cell, so the two cannot drift.

Definitions are written for a reader who has met probability and calculus but
not Bayesian workflow -- an undergraduate science or maths student -- because
that is the audience these reports are meant to serve.
"""

from __future__ import annotations

# Ordered so the rendered list reads as a progression rather than an alphabet:
# what the model is, what its parts mean, what the fitted numbers mean, and how
# to tell whether to believe them. `render_glossary` preserves this order
# regardless of the order a caller asks for terms in.
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
        "scale to be comparable; it is a unit of measurement, not a checklist "
        "any child was actually given."
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
        "A constraint pinning the GP's contribution to zero at one reference age "
        "(and, in the anchored models, removing its linear component as well). "
        "It is an identifiability device, not a claim about development: without "
        "it the GP and the linear trend can trade off against each other freely, "
        "which is what made earlier models in this family sample badly."
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
        "child one standard deviation above average has about $e^{0.8} \\approx "
        "2.2$ times the odds of the average child."
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
        "effects to zero. Because the effects are symmetric around zero on the "
        "logit scale, that is the *typical* (median) child in the typical study "
        "-- not the arithmetic average of children's counts, which the nonlinear "
        "link shifts away from the median wherever children differ. Its interval "
        "reflects uncertainty in that curve. A **subject-marginal** prediction "
        "draws a new child's random effect too: it answers *where would one more "
        "child fall?* and is much wider. Confusing them is the single easiest "
        "way to misread these reports."
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
        "project requires $\\hat{R} \\le 1.01$ for every parameter before a fit "
        "may be reported."
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
        "leaving each one's own distribution unchanged. It is what lets the "
        "sign/speech model add a single association parameter without altering "
        "the separately estimated signing and speaking trajectories."
    ),
    "Association parameter ($\\psi$)": (
        "The odds ratio measuring how much signing and speaking a given "
        "understood word go together. $\\psi = 1$ means independence; $\\psi > 1$ "
        "means a word a child signs is *more* likely to be a word they also say. "
        "Because the two overlap, the total number of words expressed either way "
        "is **smaller** than independence would predict."
    ),
}


def render_glossary(terms: list[str] | None = None, *, title: str = "Terms used in this report") -> None:
    """Print a collapsible glossary for a report cell with ``#| output: asis``.

    Mirrors :func:`vocab_growth.models.calibration.render_calibration_section`
    and :func:`vocab_growth.plotting.ppc_count_distribution_gallery`, which is
    how every other shared report block reaches the page.

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
