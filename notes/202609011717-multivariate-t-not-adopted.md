# Multivariate t on the child effects: measured, and not adopted

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Records a decision made 2026-09-01, companion to [202609011709](202609011709-shared-child-effect-row-primitive.md): whether any of the model family's multivariate normals should become multivariate Student-t distributions. The tails were measured on the fit of record before deciding, and the answer is no — with the measurement, the reasoning, and the reopen conditions recorded here so the negative is quotable. No model code changed; the diagnostic harness is promoted to `scripts/experiments/vg20_child_effect_tails.py`.

## 1. Where a multivariate t could even apply

The family's only multivariate normals are the three child-effect constructions — VG20's and VG23's correlated intercepts, VG19's per-outcome intercept-and-slope blocks, and VG22's low-rank factor — and there is no Student-t of any dimension anywhere in `src/`. The one defensible site is the child-effect block of the Down syndrome model of record, because the robustness argument for a multivariate t is that correlations are the quantity most exposed to tail mis-specification, and `rho_uq` = +0.368 is a headline number.

Everywhere else the answer is structural rather than empirical. The likelihood is Beta-Binomial — already the heavy-tailed robustification of the Binomial, with the age-varying `kappa` carrying the dispersion — and a t is inapplicable to counts. The study effects are `ZeroSumNormal` over 14 studies: far too few groups to identify a tail parameter, and the zero-sum construction has no t counterpart. The HSGP population curves have no outlier exposure. And the TD pool averages 1.16 administrations per child, so a tail degree of freedom on its child effects would be pure prior, on top of the between-child identification that is already contested there ([#225](https://github.com/dseinternational/vocabulary-growth/issues/225), VG23's registration).

## 2. The measurement

Against VG20's fit of record (the 2026-08-22 `rep` refit, commit `d7ee170`, `full` trace tier; 767 children), using 9,000 of the 36,000 posterior draws (thinned 4x — these are cross-child summaries, not MCMC estimands). Per draw, each child's two effects are standardised by that draw's own `tau_subj_u` / `tau_subj_q`, and three quantities are read across children: the excess kurtosis of each margin, the count of children beyond 3 SD, and the cross-child Pearson correlation of the standardised pair, full versus with the top-1% Mahalanobis children (under that draw's `rho_uq`) removed. The check is conservative in one direction only: the normal prior itself pulls the fitted effects toward normality, so real heavy tails are understated, never overstated.

- **The tails are genuinely heavier than normal, but mildly.** Excess kurtosis is +0.466 (89% ETI [+0.144, +0.852]) on comprehension and +0.336 [+0.074, +0.645] on the production ratio, against a normal reference band for n = 767 of [-0.268, +0.282]. On average 4.60 (u) and 3.41 (q) children sit beyond 3 SD against a normal expectation of 2.07. As an equivalent Student-t (excess kurtosis `6 / (nu - 4)`), that is roughly nu = 17 on comprehension and nu = 22 on the production ratio — and by the conservatism above, possibly heavier.
- **The quantity a multivariate t exists to protect does not need protecting.** Removing the top eight Mahalanobis children per draw moves the cross-child correlation by -0.0027, 89% ETI [-0.0245, +0.0184] — noise against `rho_uq`'s own interval width of ~0.16. The full cross-child correlation (+0.370) tracks the sampled `rho_uq` (+0.367) closely, so the diagnostic is reading the right quantity.

## 3. The decision, and why the positive kurtosis does not carry it

Not adopted, for three reasons that stand together.

First, the extreme children are not driving the headline: the trim effect on the correlation spans zero at a magnitude an order below the estimand's own uncertainty. Second, at nu = 17-22 the 89% intervals this project reports differ from their normal counterparts by about 2-3%; only far-tail child-level statements — around 3 SD, which nothing currently reports — would move materially (roughly 10-15% wider at the 99.5th percentile). Third, and decisive as policy: a t distribution downweights extreme children silently, and this project's stance on aberrant observations is the opposite — named, documented masking rules with reinstatement flags. The kurtosis finding is a reason to _identify_ the four-to-five beyond-3-SD children, not to absorb them: if they are a defect class the current rules miss, the fix is a rule; if they are a real subgroup, the honest structure is a covariate or a mixture, which a t would blur into a tail parameter. That identification is running as its own investigation.

## 4. What would reopen it, and the shape if it is ever built

Two triggers: a reporting need for far-tail child-level quantiles (a "1st-percentile child" statement), or the beyond-3-SD investigation finding neither a defect class nor an identifiable subgroup _and_ a reported quantity shown sensitive to those children. If either lands, the shape is a per-child Gamma scale mixture shared across the pair — the sharing is what makes it multivariate-t rather than independent heavy margins — parameterised by the **marginal** SD (Cholesky scale times `sqrt((nu - 2) / nu)`) so `tau_subj_*` keeps the cross-model meaning every comparison depends on, with nu fixed rather than estimated (or given the Juárez-Steel Gamma(2, 0.1) if estimation is insisted on). Governance-wise it is a new field on a definition subclass, per the VG20 precedent, invalidating nothing existing.

Reproduced by `scripts/experiments/vg20_child_effect_tails.py`.
