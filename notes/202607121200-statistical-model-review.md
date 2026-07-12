# Statistical review of the vocabulary-growth model family (VG01–VG16)

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Independent code-and-data review. Conclusions were formed by reading the model
> engines and querying the fitted data directly, not by trusting the existing
> documentation. Where this review and the checked-in prose disagree, the
> disagreement is flagged explicitly.

## Scope and method

This note reviews every statistical model in `src/vocab_growth/models/`. For each
I read the actual PyMC graph in the shared engines (`common.py`,
`common_univariate_re.py`, `common_bivariate.py`, `common_bivariate_re.py`,
`common_trivariate.py`, `common_joint_modality.py`, plus `gp_utils.py` /
`build_utils.py`), the data layer (`data_utils.py`), and the post-processing
(`posterior_analysis.py`, `comparison.py`). I cross-checked the priors and the
structural assumptions against the actual DS analysis data in
`data/vocabulary.duckdb` (1,189 DS observations, 12 studies, ages 8–115 months)
and the TD Wordbank export.

**Headline verdict.** The methodology is sound and the implementations are
correct. The core design — a Beta-Binomial likelihood with an age-varying
dispersion, a logit-scale linear trend reparameterised through two interpretable
anchor ages, an HSGP for smooth nonlinear departures, and the
`p_S = p_U · q` production-ratio decomposition — is a well-judged, coherent
framework, and the harder pieces (non-centred random effects, GP anchoring, the
Plackett association, the Dirichlet-Multinomial cross-tab, the disjoint-posterior
DS-vs-TD contrasts) are implemented correctly. The substantive risks are not
coding errors; they are **modelling-assumption** risks, and the most important one
is the same for almost every model: **pooling raw counts from checklists of
different lengths onto a single 810-item scale.** Details below.

---

## 1. Shared architecture (what every model is built from)

### 1.1 Likelihood — Beta-Binomial with age-varying dispersion

Every vocabulary count `y` is modelled as `BetaBinomial(n = 810, α = p·κ, β = (1−p)·κ)`.
This is the mean/precision parameterisation: `E[y] = 810·p` and the concentration
`κ = α+β` controls overdispersion. The implied count variance is
`810·p(1−p)·(κ+n)/(κ+1)` (verified in `comparison.implied_sd_y`), so the
variance-inflation factor over a Binomial is `(κ+n)/(κ+1)`. This is the right
tool: CDI counts are bounded, heavily overdispersed relative to a Binomial, and
show floor/ceiling behaviour, all of which the Beta-Binomial handles.

The dispersion is itself age-varying: `κ(z) = κ_min + exp(a_κ + b_κ·z)` where
`z` is standardised age. Note `b_κ = −b_κ_mag` with `b_κ_mag ~ HalfNormal`, so
**`b_κ ≤ 0` is a hard constraint**: precision is forced to be monotonically
non-increasing in age, i.e. overdispersion is forced to be non-decreasing with
age. This is plausible for Down syndrome (between-child spread widens as children
develop) and the data agree, but it is a structural assumption the data cannot
override — see §3(C).

### 1.2 Mean trajectory — anchored logit-linear trend + HSGP

On the logit scale the expected proportion is `f(a) = intercept + slope·z + g(a)`,
where `g` is a Hilbert-Space GP (`pm.gp.HSGP`, ExpQuad kernel). The linear part is
not parameterised by an intercept/slope directly; instead priors are placed on the
**expected proportion at two reference ages** (`slope_anchors`), and slope/intercept
are derived by `slope = (logit(p_hi) − logit(p_lo))/(z_b − z_a)`. This is a genuinely
good design choice: priors live on an interpretable scale ("what fraction of the
810 words does a typical child know at 24 and 84 months") rather than on
uninterpretable logit coefficients. The algebra is correct (`gp_utils.trend_and_gp`).

The HSGP length-scale is given a **bounded** prior — a Beta on `[0,1]` mapped onto
`[ell_low, ell_high]` months — which is the right way to keep an HSGP well-behaved
(it avoids the pathological very-short and very-long length-scales that break the
basis approximation). The basis size `m` and boundary `L` are sized from the
grid half-range via `approx_hsgp_hyperparams`, and the basis is evaluated on the
full stacked grid (observed + plot + query + optional anchor), which correctly
covers query ages that extend beyond the observed range. Verified in
`common.get_hsgp_hyperparams`.

The mean trajectory is **not** constrained to be monotone. That is fine for a
descriptive model but means extrapolation beyond the data can wander; see §3(I).

### 1.3 Production-ratio decomposition (joint models)

Rather than modelling spoken independently, production is a fraction of
comprehension: `p_U = σ(f_U)`, `q = σ(h)`, `p_S = p_U · q`. This enforces
`p_S ≤ p_U` (a child cannot say more of the inventory than it understands) **by
construction**, which is exactly the right structural prior for this domain. I
checked the raw data: of 730 DS rows with both outcomes, only 4 (0.5%) have
spoken > understood, and 0 of 331 sign/understood rows have signed > understood.
So the hard constraint is almost perfectly consistent with the data — a strong
endorsement of the decomposition. The trivariate/joint models extend this with a
signing ratio `r = σ(g_sign)` and `p_Sign = p_U · r`.

### 1.4 Composite marginal (pseudo-)likelihood

This is the most important thing to understand about the joint models, and the
code comments it honestly. When a child has both understood and spoken recorded,
the two counts enter as **two separate Beta-Binomial marginals** coupled only
through the shared latent means (`p_S = p_U·q`); they are treated as
conditionally independent given those means. The item-level joint — *which* of the
understood words the child also speaks — is not modelled (most sources report only
marginal totals). This is a composite/pseudo-likelihood. Consequence: the
posterior for the **means** (`p_U`, `q`, `p_S`) is well-behaved and interpretable,
but uncertainty for joint functionals is not fully propagated, and the composite
likelihood's absolute scale should not be over-interpreted as a proper joint
likelihood (this matters for LOO comparisons — see §3(J)). VG15 is the one model
that adds a genuine item-level joint term (the uk_02 four-cell Dirichlet-Multinomial).

### 1.5 Hierarchy: random effects and GP anchoring

Later models add non-centred (`δ = τ·z`, `z ~ N(0,1)`) random intercepts — study
level (absorbing between-lab level differences) and subject level (absorbing stable
between-child differences for the many repeated-measures children). Non-centring is
the correct HMC parameterisation here. Subject IDs are namespaced `study::subject_id`
before coding, which correctly prevents cross-study ID collisions (verified in
`common_bivariate_re.prepare_bivariate_re_data`).

"GP anchoring" (`anchor_idx`) subtracts `g(a_ref)` from the GP for every draw so the
GP passes through zero at a reference age. This is a legitimate device to remove the
statistical redundancy between the linear trend, the GP, and the intercepts (a
"ridge" in the posterior) that otherwise degrades sampling once random intercepts add
another level-carrying term. It changes the *parameterisation*, not the model
family. Correctly implemented in `gp_utils._gp_from_mean`.

**Estimand nuance (applies to every RE model).** Plot/query trajectories are reported
at the population level (`δ_study = 0`). The posterior-predictive count bands
marginalise over a fresh **subject** draw (giving a coherent unseen-child trajectory
across age), but they do **not** marginalise over the **study** random effect. So the
predictive intervals mean "a new child in an average/typical study" and exclude
between-study heterogeneity. This is a defensible, clearly-scoped estimand, and
`posterior_analysis.add_probability_estimand_columns` helpfully reports both the
population-mean (`p_population`) and new-child (`p_subject_marginal`) estimands — but
readers must not read the bands as full population-predictive intervals.

---

## 2. Data and harmonisation

Raw study CSVs are merged by `scripts/prepare_data.py` into a DuckDB `vocab_combined`
view (DS) plus the Wordbank export (TD). The key facts I verified:

- **DS:** 1,189 observations, 12 studies, ~609 children, ages 8–115 months.
  Understood observed in 731 rows, spoken in 1,173, signed in 638. Repeated
  measures are heavy but short: 232 children have 1 observation, 262 have exactly
  2, and only ~52 have ≥4. **The panel is 2-wave-dominated** — decisive for VG16.
- **Instrument ceilings differ** and are recorded in `survey_vocab_max`: DSE
  checklist = 810, Oxford CDI = 416, MB-CDI WG = 396, WS = 680, NZCDI = 675.
  A form-ceiling guard drops only counts strictly above their form's native
  ceiling (data-entry errors); legitimate ceiling observations are kept.
- **WS comprehension is excluded** (both TD and the us_01/Edgin DS subset) because
  on WS/TEDS forms `comprehension == production` by data convention — it is a
  production proxy, not an independent measurement. WS still contributes production.
  This is a correct and important fix (it is what retired VG06).
- Harmonisation choices worth knowing: `ie_01` sets understood = `max(says, understands)`
  (production implies comprehension); `nz_01` is production-only with
  modality-exclusive columns recombined as any-modality (`spoken = word-only + both`,
  `signed = sign-only + both`).

Empirical trajectories by age band (verified, understood/spoken as fraction of 810):

| Age band (mo) | n | U/810 | S/810 | q = S/U (median) | sign/U (median, n) |
|---|---|---|---|---|---|
| 8–18  | 127 | 0.04 | 0.00 | 0.00 | 0.00 (32) |
| 18–24 | 140 | 0.12 | 0.01 | 0.03 | 0.07 (60) |
| 24–36 | 267 | 0.25 | 0.04 | 0.08 | 0.29 (128) |
| 36–48 | 245 | 0.36 | 0.14 | 0.24 | 0.40 (72) |
| 48–60 | 167 | 0.39 | 0.23 | 0.27 | 0.53 (21) |
| 60–72 | 132 | 0.45 | 0.34 | 0.65 | 0.59 (5) |
| 72–115| 109 | 0.58 | 0.46 | 0.65 | 0.04 (7) |

Two things jump out and drive interpretation: (a) understood observations thin
dramatically past 60 months (only 18 understood counts above 72 mo), so the
old-age understood curve is prior/GP-driven; (b) the signed ratio shows the
expected preschool **hump** but its descending limb rests on 5–7 observations — the
post-peak collapse is essentially unidentified from data.

I also confirmed the anchor priors are sensibly calibrated to these data rather than
distorting: e.g. VG02's 84-month understood anchor `Beta(2,1.5)` has mean 463 words
vs an empirical ~471; VG01's 24-month spoken anchor `Beta(1,25)` has mean 31 words vs
an empirical ~30. They are weakly informative and roughly data-consistent.

---

## 3. Cross-cutting soundness findings (my independent assessment)

**(A) Cross-instrument pooling onto a fixed n = 810 — the single biggest caveat.**
Every likelihood uses `n_trials = 810`, but the raw counts come from checklists that
can be much shorter. A child assessed on the 416-item Oxford CDI is modelled as
"successes out of 810," so that study is *structurally censored* at 416/810 ≈ 0.51 of
the reference scale, and MB-CDI WG at 396/810 ≈ 0.49. Study random intercepts
(VG07+) absorb a **constant logit-level shift** between forms, but a fixed
multiplicative ceiling is not a constant logit shift — it bites only at high
proportions and older ages, exactly where DS children on the 810-item DSE form keep
rising while a shorter form cannot follow. The single-outcome baselines (VG01–VG04)
and the non-hierarchical joint models (VG05, VG14) have **no** study effects, so this
heterogeneity is only soaked up by overdispersion there. This is the deepest threat
to cross-study and DS-vs-TD comparisons. It is partially mitigated (ceiling guard +
study REs) and partially acknowledged in the code, but the residual
scale-incommensurability remains and should be stated prominently in any headline. A
cleaner future fix would rescale each form's count to the 810 frame (or use a
per-form `n_trials`) rather than treating a 416-count as 416/810.

**(B) The `us_01` `production ≤ 100` cap.** An undocumented legacy filter drops the
highest-production rows on the Edgin/Wordbank DS subset (8/87 WG, 24/109 WS rows in
the current export). This biases DS spoken *downward* precisely at the ages where the
strongest talkers sit. The code flags it as pending review; I agree it is a real, if
localised, bias and should be resolved or justified.

**(C) The forced-sign dispersion trend.** `b_κ ≤ 0` forces overdispersion to be
non-decreasing in age. Plausible and data-consistent, but it cannot represent a
non-monotone dispersion profile. Worth a one-line sensitivity check allowing either
sign; minor.

**(D) Conditional vs marginal reporting.** As in §1.5 — reported trajectories are
population/typical-study level; predictive bands marginalise subject but not study
REs. Correctly scoped and dual-reported, but easy to over-read.

**(E) No-hierarchy models understate uncertainty.** VG01, VG02, VG05 and **VG14**
pool multiple studies and repeated measures with no random effects. Overdispersion
partly compensates, but within-child/within-study correlation is unmodelled, so
their intervals are optimistic and their levels can be biased by whichever study
dominates a given age. This is by design for the VG01/VG02/VG05 baselines, but
**VG14 is a live signing model, not just a baseline** — its uncertainty should be
read as a floor, with VG15 the model to trust.

**(F) VG16 is the least clean and the team says so.** See §4. I concur with the
authors' own diagnosis: with 2-wave-dominated data the within-child cross-lag is
biased (short-T / dynamic-panel / errors-in-variables), and only the
population-relative (≈ null) estimate should be reported as headline.

**(G) VG14's `p_any` is an upper bound only under positive item-level association.**
`p_any = p_U·(r + q − r·q)` assumes sign ⟂ speak given understood. If DS children
*substitute* signs for not-yet-spoken words (negative item-level correlation),
independence would *underestimate* total expressive vocabulary. The uk_02 four-cell
data show a positive association (`psi > 1`), so "upper bound" holds empirically —
but it is data-contingent, and VG15 replaces the assumption with a directly estimated
association. Trust VG15's `p_any`, not VG14's.

**(H) `ie_01` understood = max(says, understands)** is a reasonable harmonisation
(production implies comprehension) but can inflate understood where says > understands.

**(I) Old-age and signed-tail estimates are prior/GP-driven.** With only 18 understood
counts past 72 months and 5–7 signed-ratio points on the descending limb, HDIs there
should be wide and read as regularised extrapolation, not evidence. This is a
data-density fact, not a model flaw, but it must be stated wherever those regions are
plotted (the comparison code's `shade_unsupported` does exactly this — use it).

**(J) LOO across models with different likelihood structures.** LOO/ELPD is reported
per-outcome, which is right, but comparing a composite-marginal model to one with an
extra Dirichlet-Multinomial term (VG15) or a different observation set is not an
apples-to-apples predictive comparison. Compare within a lineage and outcome, not
across structurally different likelihoods.

None of A–J is a correctness bug. They are the assumptions a reader must hold in mind.

---

## 4. The models, one by one

Each entry: **purpose / questions**, **methodology specifics** (beyond the shared
core), **interpretation**, and **model-specific assessment**.

### VG01 — DS, spoken (baseline)
- **Purpose.** Establish the baseline chronological-age → words-spoken trajectory for
  children with Down syndrome. Questions: how does expressive vocabulary grow with
  age; when is growth fastest; how uncertain is it, especially at older ages?
- **Methodology.** Single Beta-Binomial outcome, anchors at 24 & 84 months, no
  hierarchy. Spoken low anchor `Beta(1,25)` (~31 words at 24 mo) respects the
  near-zero early-speech floor; high anchor `Beta(2,1.5)` is broad regularisation
  (no independent DS norm beyond 60 mo).
- **Interpretation.** Read `posterior_summary` at query ages: `Ey_median` (expected
  words) with HDI, and the `P(Y≤k)` columns (e.g. `P(Y≤10)`) which are directly
  clinically legible. The `expected_learning_rate` plot locates the fastest-growth
  age.
- **Assessment.** Correct and appropriately humble as a baseline. Its intervals are
  optimistic (§3E) and it inherits §3A. Not the preferred DS spoken model — VG07–VG10
  supersede it.

### VG02 — DS, understood (baseline)
- **Purpose.** Baseline age → words-understood trajectory for DS. Comprehension is the
  scaffolding for the joint models.
- **Methodology.** As VG01 but understood; low anchor `Beta(1,7)` (~101 words at 24 mo)
  and wider `eta_u`/`eta`. The comment correctly flags that DS comprehension has **no
  independent normative source** in the library, so these anchors are data-informed
  regularisation, not external anchors — a sensitivity target.
- **Interpretation.** Same as VG01; the understood curve rises to ~0.58 of 810 by
  6+ years in the data.
- **Assessment.** Correct; same baseline caveats. The lack of an independent
  comprehension norm is the honest weak point and is disclosed.

### VG03 — TD, spoken (baseline)
- **Purpose.** Typically-developing counterpart to VG01, for DS-vs-TD contrasts.
- **Methodology.** Anchors at 12 & 26 months (the TD-relevant window), anchored to
  published Wordbank deciles (spoken median ~11/810 at 12 mo). Uses WG + Oxford CDI +
  WS production rows, subsampled to 25% (`sample_fraction=0.25`) after the WS
  comprehension exclusion shrank the pool.
- **Interpretation.** TD spoken trajectory over 8–30 months; the natural comparator
  is DS at matched comprehension, not matched age (see `comparison.py`).
- **Assessment.** Correct. The 25% subsample is a pragmatic speed choice; VG11 is the
  full-data hierarchical replacement, so VG03 is a baseline.

### VG04 — TD, understood (baseline)
- **Purpose.** TD counterpart to VG02.
- **Methodology.** 12-month understood anchored to the Wordbank comprehension norm
  (`Beta(1.2,8)`, ~84/810); the 26-month high anchor has **no** independent CDI
  comprehension norm (WS is production-only) and is broad regularisation. WG + Oxford
  CDI only (WS comprehension excluded). Subsampled 25%.
- **Assessment.** Correct; VG12 is the full-data hierarchical replacement.

### VG05 — DS, understood + spoken (joint baseline)
- **Purpose.** First joint model: estimate comprehension, production, and the
  **production ratio `q(a) = P(speak | understood)`** together. Questions: what
  fraction of understood words can a DS child say, and how does that fraction grow?
- **Methodology.** The `p_S = p_U·q` decomposition (§1.3), composite marginal
  likelihood (§1.4), no hierarchy. `q` gets its own anchored trend + GP.
- **Interpretation.** The `production_rate` and `production_rate_by_understood` plots
  are the payoff: `q` rises from ~0 to ~0.65 in the data. `comprehension_production_gap`
  shows the widening understood-minus-spoken gap.
- **Assessment.** The decomposition is the right structural choice (validated by the
  0.5% violation rate). No-hierarchy caveats (§3E) apply; superseded by VG07–VG10.

### VG07 — VG05 + study random intercepts
- **Purpose.** Absorb systematic level differences between the pooled source studies.
- **Methodology.** Non-centred study REs `δ_u`, `δ_q` on both latent trajectories.
- **Interpretation.** `q` and the trajectories now represent a "typical study"
  (`δ = 0`); `τ_u`, `τ_q` quantify between-study spread. First model where the §3A
  form heterogeneity is partly (level-shift) absorbed.
- **Assessment.** Correct; the natural first hierarchical step.

### VG08 — VG07 + subject REs on understood
- **Purpose.** Separate stable between-child differences from within-child change for
  the many repeated-measures children.
- **Methodology.** Adds non-centred subject REs on `f_U`. Posterior-predictive uses a
  fresh subject draw (coherent unseen-child trajectory).
- **Assessment.** Correct and important given 377/609 children have repeats.

### VG09 — VG08 + subject REs on `q`
- **Purpose.** Allow stable between-child differences in the *production ratio*, not
  just comprehension.
- **Methodology.** Adds subject REs on `h` (the `q` logit). Note `eta_q_sigma` is
  tightened to 0.20 to curb the GP-vs-slope/intercept competition on `q`.
- **Assessment.** Correct. The team's own notes record marginal R-hat/ESS strain on
  the `q` hyperparameters here — a genuine identifiability tension between the `q`
  trend, `q` GP, and the `q` intercepts, which motivated VG10.

### VG10 — VG09 + tighter `q` anchors + GP anchoring at 54 mo
- **Purpose.** The **preferred** DS joint (understood+spoken) model. Same structure as
  VG09 but with the sampler pathology resolved.
- **Methodology.** GP anchored to zero at 54 months on both trajectories (§1.5),
  removing the trend/GP/intercept ridge; `q` anchor priors broadened but kept
  weakly-informative (the code notes they are deliberately *not* double-dipped from a
  posterior). This is a parameterisation change, not a different model.
- **Interpretation.** Use VG10 for the headline DS understood/spoken/`q` story. Report
  both the population and new-child estimand columns.
- **Assessment.** Sound. GP anchoring is the correct remedy for the ridge, and the
  care taken to avoid double-dipping the priors is commendable.

### VG11 — TD, spoken, with study REs + GP anchor at 19 mo
- **Purpose.** Full-data hierarchical replacement for VG03.
- **Methodology.** Study (dataset/lab) REs, no subsampling, drop studies with < 200
  observations (`min_study_observations`), GP anchored at 19 months. Population
  (`δ=0`) trajectory reported.
- **Assessment.** Correct. Using study REs instead of subsampling is the right call —
  it lets all qualifying observations contribute while absorbing between-lab level.

### VG12 — TD, understood, with study REs + GP anchor at 19 mo
- **Purpose.** Full-data hierarchical replacement for VG04.
- **Methodology.** As VG11 for comprehension (WG + Oxford CDI). The 26-month high
  anchor remains a named sensitivity target (no independent comprehension norm).
- **Assessment.** Correct; the disclosed weak point is the un-normed high anchor.

### VG13 — TD, understood + spoken (joint), 8–18 months
- **Purpose.** The TD joint model used for DS-vs-TD comparison, replacing retired VG06.
- **Methodology.** Restricted to 8–18 months, where WG (8–18) and Oxford CDI (12–25)
  are dense and the WS production-proxy bias is avoided entirely. Study REs, GP anchor
  at 13 months. Window-appropriate `q` anchors (~0.07–0.20, from Wordbank
  production/comprehension ratios) — crucially, **not** the DS-tuned bivariate
  defaults, which would overshoot young-TD production several-fold.
- **Interpretation.** Valid only over 8–18 months; above that only Oxford CDI provides
  bivariate rows (single study → unreliable), which I confirmed in the Wordbank age
  ranges. The retirement of VG06 (WS comprehension = production proxy) is correct.
- **Assessment.** The age restriction and the window-specific `q` anchors are exactly
  right; this is a careful model. The 810-scale caveat (§3A) is mild here because TD
  proportions are low in this window.

### VG14 — DS, understood + spoken + signed (trivariate)
- **Purpose.** Add signing as a third modality; estimate the signed ratio
  `r(a) = P(sign | understood)` and a **total expressive** vocabulary `p_any`.
- **Methodology.** Adds `r = σ(g_sign)` with an **intercept-only mean + GP** (no age
  slope) — a deliberate, well-reasoned choice: the empirical signed ratio is a hump
  (near zero < 24 mo, peaks in preschool, recedes), and a free slope would extrapolate
  a spurious high signed level at 12 months. A short GP length-scale lets the hump
  form. `p_any = p_U·(1 − (1−r)(1−q))` assumes sign ⟂ speak given understood (§3G).
  **No random effects** (mirrors VG05).
- **Interpretation.** `r(a)` peaks in the preschool years then declines; but see §3(I)
  — the declining limb rests on 5–7 observations, so treat the peak location as
  better-identified than the post-peak collapse. `p_any` is an independence-based
  **upper bound** on combined production.
- **Assessment.** The intercept-only signed mean is a smart structural fix. The two
  live weaknesses are the independence assumption (relaxed by VG15) and the absence of
  hierarchy on strongly repeated, multi-study signing data (VG15 adds it). Read VG14 as
  the transparent starting point and VG15 as the trustworthy version.

### VG15 — DS joint sign/speech with a data-identified association (the flagship)
- **Purpose.** Replace VG14's independence assumption with a **directly estimated
  within-understood sign–speech association**, yielding a data-identified `p_any`, and
  add full hierarchy. Question: given a word is understood, how do signing and
  speaking co-occur, and what is total expressive vocabulary once that is estimated?
- **Methodology.** A scalar **Plackett odds ratio `psi`** identifies the 2×2
  within-understood cross-tab {neither, sign-only, speak-only, both} from the given
  margins `r`, `q`. I verified the Plackett root: the code uses the rationalised,
  branch-free form `pi_both = 2·psi·r·q / (S + disc)` with `S = 1 + (r+q)(psi−1)` and
  `disc = sqrt(S² − 4·psi(psi−1)rq)`, which is algebraically identical to the textbook
  `(S − disc)/(2(psi−1))` but numerically stable and continuous at `psi = 1` (returns
  `r·q`, i.e. independence). Correct. The cross-tab is fit with a
  **Dirichlet-Multinomial** (concentration `conc`) on the uk_02 four-cell counts, and a
  three-cell (within-*produced*) Dirichlet-Multinomial on nz_01 (which has no
  comprehension, so the "neither" cell is dropped and the composition renormalised over
  {sign-only, speak-only, both}). Both cross-tab terms share `psi`, so the two sources
  jointly identify it. Study REs on all three trajectories, optional subject REs (on by
  default), and VG10's GP anchoring at 54 mo. `p_any = p_U·(r + q − pi_both)` (data-
  identified) is reported **alongside** the independence bound `p_any_indep` — excellent
  practice, letting the reader see how much the estimated association moves the total.
- **Key identification decision (correct and worth stating).** `psi` is fed the
  **population+study** marginals, deliberately *excluding* the subject sign-RE. The
  reason (recorded in a note and verified in code): the per-child sign offset is co-
  identified with `psi` from the same ~62 uk_02 rows / 34 children, and letting it into
  the cross-tab composition makes `psi` pivot on a thinly-identified random effect
  (dev `psi` 1.78 → ~2.8). Holding `psi` to a population-conditioned association is the
  right conservative choice. Subject REs still enter every marginal likelihood.
- **Interpretation.** `psi > 1` means a word is more likely to be both signed and
  spoken than independence predicts (positive within-understood association), which is
  why VG14's independence `p_any` is an over-estimate (upper bound). Report `psi` with
  its HDI and whether it excludes 1; report the data-identified `p_any` vs the
  independence bound.
- **Assessment.** This is the most sophisticated and the most carefully implemented
  model in the family, and it is correct. Its honest limits: the association rests on a
  small cross-tab (uk_02 + nz_01), and the signed ratio's post-peak tail is data-thin
  (§3I). The nz_01 three-cell term assumes its produced cross-tab is a truncation of
  the same within-understood Plackett structure — a reasonable but unfalsifiable
  bridging assumption (the `include_nz01_cells` flag lets you test its pull on `psi`).

### VG16 — VG09 + a within-child receptive→expressive cross-lag
- **Purpose.** Ask a dynamic question: does a child's *earlier* comprehension standing
  predict their *later* production conversion `q` (earlier receptive → later
  expressive)?
- **Methodology.** For each observation the child's immediately-earlier understood wave
  is the lag source; `x_lag = has_lag · (observed prior understood logit − model-expected
  baseline)` enters the current `q` logit as `beta_lag · x_lag`. The baseline is
  configurable: **within-child** (subtract the child's own subject intercept → RI-CLPM
  within effect) or **population-relative** (subtract only population+study → blends
  within/between). The headline VG16 uses the **population-relative** baseline.
- **Interpretation and assessment — the least clean model, correctly flagged.** With
  the DS panel being 2-wave-dominated (I verified: 262 children have exactly 2 waves,
  and repeats > 3 are rare), the within-child estimator suffers the classic short-T
  dynamic-panel (Nickell) bias plus errors-in-variables/regression-to-the-mean, because
  the lag regressor is a single noisy observed proportion and the baseline is built from
  the same child's estimated intercept. The team's own dev results bear this out (within:
  `beta ≈ −0.60`, an artifact; population-relative: `≈ +0.05`, null), and they correctly
  demote the within-child variant to a "cautionary contrast." My independent read agrees:
  **these data cannot cleanly identify a within-child receptive→expressive cross-lag**,
  and the honest conclusion is the near-null population-relative estimate with the
  within-child number presented only to illustrate the bias. The implementation itself
  (the lag construction, the baseline switch, the validation that `use_subject_re_u` is
  on) is correct; the limitation is the data, not the code.

---

## 5. Posterior-predictive plots and reports that are (or would be) most instructive

The engines already produce a strong set (`plot_posterior_predictive_*`,
`production_rate*`, `comprehension_production_gap`, `understood_vs_spoken`, the
`comparison.py` milestone/latency/expressive-delay panels, `posterior_kappa`). What
is most instructive to lead with, and a few additions I would make:

**Most instructive existing artefacts:**
- **`posterior_summary` `P(Y≤k)` columns** at query ages — the most clinically legible
  output. "At 36 months, P(spoken ≤ 10 words) = …" communicates far more than a mean.
- **Production ratio `q(a)` and `q` vs words-understood** — the core scientific quantity
  (how much of comprehension converts to speech, on both an age and a comprehension
  axis).
- **Milestone-age tables** (`comparison.milestone_table`) — correctly computed as
  median-of-crossings, with `prop_reaching` flagging when a target is unreached by many
  draws (read those medians as conditional).
- **Comprehension-matched DS-vs-TD contrasts** (`expressive_specific_delay`,
  `comprehension_equivalent_age`) — the right way to ask "is DS production delayed
  *beyond* its comprehension delay," removing the timescale confound. This is arguably
  the study's most important comparative output.
- **VG15 four-cell composition and `p_any` vs `p_any_indep`** — shows the association's
  practical effect on total expressive vocabulary.
- **`posterior_kappa` / overdispersion factor** — communicates the widening between-child
  spread with age (the clinical "children fan out" message).

**Additions worth making:**
1. **Always shade unsupported age/level regions** (`comparison.shade_unsupported`) on
   every trajectory plot — especially DS understood past ~60 mo and the signed-ratio
   tail, where estimates are prior/GP-driven (§3I). This is the single highest-value
   presentational fix.
2. **Posterior-predictive coverage / calibration check:** overlay observed
   per-age-band empirical quantiles on the predictive bands (a PIT/coverage panel) to
   demonstrate the Beta-Binomial is calibrated, not just plausible.
3. **Both estimands side by side:** plot the population-mean band and the new-child
   (subject-marginal) band together so the conditional-vs-marginal distinction (§3D) is
   visible rather than implicit.
4. **Prior-sensitivity overlays:** the `sensitivity/` framework already fits
   alternative-prior variants for VG10/11/12/15 — surface those as overlay plots in the
   reports so the reader sees which conclusions move under the un-normed anchors (VG02/
   VG12 high anchors) and the signed GP priors (VG15).
5. **A form/instrument diagnostic:** a plot of study-RE posteriors `δ` coloured by
   checklist (810 vs 416 vs 396) would make the §3A scale-heterogeneity visible and
   show how much work the study REs are doing.

---

## 7. Documentation vs code — verified discrepancies

The task was explicit that comments, notes, and docs may not be correct. I
cross-checked the prose claims against the code. Most check out — the
composite-likelihood caveat, the intercept-only signed mean, the per-draw GP
anchoring, the `psi` decoupling from subject REs, and the fixed n = 810 are all
faithfully described. Two concrete discrepancies, both verified against the code:

1. **The `q`-anchor priors in the VG10/VG15 report templates are stale.** The
   per-model `.qmd` templates (`docs/models/vg10/index.qmd`, `docs/models/vg15/index.qmd`)
   and the stabilisation notes (`notes/202605131500-...`, `notes/202606171200-...`)
   describe the current DS-joint `q` anchors as `Beta(3,22)` / `Beta(20,4)`
   ("Option A", read off the VG07 posterior). **The code no longer uses those.**
   `definitions.py` (VG10 and VG15) sets `p_slope_low_q ~ Beta(2,12)` and
   `p_slope_hi_q ~ Beta(3,2)`, with `eta_q ~ HalfNormal(0.20)` compensating — the
   broadened, non-double-dipping anchors that commit `b440cb4` (#155, "broaden the
   double-dipped DS-joint q-anchors, tame the q-GP ridge") introduced. `PRIORS.md`
   is the one that matches the code. This matters because those templates are
   **copied into the rendered model outputs**, so a rendered VG10/VG15 report
   currently misstates its own priors (and repeats the "informed by the VG07
   posterior" framing that the code deliberately abandoned to avoid prior-data
   double-dipping). This is a documentation bug, not a model bug — the code is the
   good state — but it should be fixed so the reports do not describe a superseded
   prior.

2. **TD understood anchors: `OUTPUT_TEMPLATE_REVIEW.md` is stale, `PRIORS.md` is right.**
   That review note quotes VG04/VG12 low/high understood anchors as
   `Beta(1,20)` / `Beta(1.5,1.1)`. The code (`definitions.py`) uses
   `Beta(1.2,8)` / `Beta(1.3,1.3)`, matching `PRIORS.md`. `OUTPUT_TEMPLATE_REVIEW.md`
   is a self-described working note, so this is low-stakes, but it is stale.

Two documentation-currency issues I could not fully resolve from code alone and
that the maintainers should confirm:

3. **Refit currency after the WS/us_01 fix.** The 2026-07-06 note records that the
   `us_01`/Edgin comprehension-proxy defect (WS rows claiming U = S) affected the DS
   understood pool and lists VG02, VG05, VG07–VG10, VG14, VG15, VG16 as needing
   refit. Confirm the posteriors feeding the report and the `docs/report/figures/`
   cache are the post-fix fits, since the defect biased exactly the
   comprehension–production gap and `q` that the headline findings report.

4. **Config-quality of reported findings.** `methods-workflow.qmd` says the reported
   fits clear the convergence gate at `rep` quality, but the DS-vs-TD findings
   chapter calls the VG10 and VG13 fits behind the Δq contrast **development-tier**
   ("magnitudes await reporting-quality re-fits"), and the VG16 numbers are
   explicitly dev-tier. The direction of results is likely robust; the magnitudes
   should be regenerated at `rep` and the two chapters reconciled.

For reference, the headline posterior numbers the reports quote (to be read with
the currency caveats above): VG15 `psi` ≈ 1.72, 90% HDI [1.17, 2.35],
P(psi > 1) ≈ 0.996 (positive within-understood sign–speech association, so VG14's
independence `p_any` is an upper bound, sitting a few words above the
data-identified `p_any`); the DS-vs-TD comprehension-matched contrast
Δq = q_TD − q_DS is credibly positive from ~50 to ~150 understood words (peak
≈ +0.13, P(TD > DS) ≥ 0.96), i.e. the comprehension–production gap is
disproportionately larger in DS at low vocabulary and converges toward the TD
pattern as vocabulary grows; VG16's within-child cross-lag is a spurious
β ≈ −0.60 while the bias-robust population-relative β ≈ +0.05 is null.

## 8. Bottom line

- The statistical methodology is **sound** and the implementations are **correct**. I
  found no correctness bug in the model graphs, the Plackett/Dirichlet-Multinomial
  machinery, the random-effects parameterisation, or the cross-model contrast code. The
  Plackett root, the Beta-Binomial variance, the anchor reparameterisation, the
  subject-ID namespacing, and the `p_S ≤ p_U` constraint were all verified against the
  code and (where possible) the data.
- The residual risks are **assumptions, not errors**, and the codebase is unusually
  honest about most of them in comments and notes. The ones to keep at the front of the
  mind: (A) pooling different-length checklists onto n = 810; (B) the undocumented
  `us_01` production cap; (E) optimistic uncertainty in the no-hierarchy models
  (VG01/02/05/**14**); (I) prior-driven old-age and signed-tail regions; and (F) VG16's
  unidentified within-child cross-lag.
- Preferred models for headline claims: **VG10** (DS understood + spoken), **VG12/VG11**
  (TD understood/spoken), **VG13** (TD joint, 8–18 mo), **VG15** (DS sign/speech and
  total expressive). VG01–VG05, VG07–VG09 and VG14 are best framed as the development
  lineage leading to those; VG16 as a deliberately cautionary dynamic analysis.
