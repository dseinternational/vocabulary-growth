# VG16 scoping note: within-child cross-lagged receptive → expressive vocabulary

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

Status: scoping / pre-build, 2026-07-03. Addresses issue #113 (Q2). Builds on the
VG09/VG10 bivariate + subject-random-intercept foundation. The **cross-lagged
coupling** spec family was chosen by the modelling lead; this note pins the
precise formulation for sign-off before implementation.

## 1. The question (Q2)

Within children who have repeated measures, does a child's **earlier** receptive
vocabulary (words understood) predict their **later** expressive vocabulary
(words spoken), beyond the population age trajectory and the child's own stable
level? I.e. is there a within-child *lead* of comprehension over production?

## 2. What the foundation already provides (VG09/VG10)

Per observation `i` of child `c` at age `a`:

- `understood_i ~ BetaBinomial(N, p_U(a_i))`, `p_U = sigmoid(f_U(a) + δ_study + δ_subj_u[c])`
- `spoken_i ~ BetaBinomial(N, p_U(a_i)·q_i)`, `q_i = sigmoid(h(a) + δ_study + δ_subj_q[c])`

Subject effects are **intercepts only** — constant within a child, no temporal
structure. The *concurrent* U→S link is built in (`p_S = p_U·q`), but nothing
couples a child's **earlier** understood to their **later** spoken.

## 3. Data — the within-child signal

576 DS children / 1,078 observations. **350 children have ≥2 observations** (257
exactly 2, 57 with 3, 16 with 4, 19 with 5, 1 with 8); **279 have understood AND
spoken at ≥2 ages** — the pool that identifies a cross-lag. Age is the time axis;
spacing is irregular. Because most repeated-measures children contribute only 2
waves, the design must extract a *population* lead-lag from many short series, not
long per-child series.

## 4. Proposed VG16 structure (cross-lagged)

Add **one population lead-lag coefficient `β`** to the VG09 foundation. For each
observation `i` of child `c` at age `a_i`, build a lagged predictor from the
child's **immediately preceding observed wave** (age `a_prev`).

**Primary (within-child, RI-CLPM):** baseline against the child's *own* expected
comprehension, so `x_lag` is the within-child *fluctuation* (not their stable standing):

```
x_lag,i = logit(understood_prev / N) − ( f_U(a_prev) + δ_study + δ_subj_u[c] )   # within-child residual
x_lag,i = 0                                                                       # first wave / no prior understood
```

Subtracting `δ_subj_u[c]` is what separates the *within-child lead* ("a child
temporarily ahead of their own comprehension trend then produces more" — Q2) from
the *between-child* association ("high-comprehension children just talk more",
which the correlated-RE alternative already captures). This is the modern
random-intercept cross-lagged panel (RI-CLPM) logic: the subject REs are the
random intercepts, so `β` on top of them is the within-child cross-lag rather
than the classic-CLPM between/within blend. The production ratio gains the lag term:

```
q_i = sigmoid( h(a_i) + δ_study[study] + δ_subj_q[c] + β · x_lag,i )
```

with `spoken_i ~ BetaBinomial(N, p_U(a_i)·q_i)` as before. `δ_subj_q` absorbs the
child's stable conversion level; **`β` isolates the effect of the child's *prior*
receptive standing on their *current* production** — the within-child lead of
receptive over expressive. `β > 0` with a 90% HDI excluding 0 is the Q2-positive
result.

Prior: `β ~ Normal(0, σ_β)` weakly-informative, `σ_β` calibrated by prior
predictive so a ±1-SD prior-understood deviation moves `q` by a plausible amount.

### DAG

`A → U(t)`, `A → S(t)`; `U(t) → S(t)` concurrent (structural, via `q`);
**`U(t−1) → S(t)` cross-lag (via `β`)** — the new within-child term.

### Why this encoding

- Directly answers "earlier receptive → later expressive": predictor = prior
  wave's understood, outcome = current spoken.
- Identifiable with 2-wave children — each contributes one early→late pair; `β`
  pools across the 279.
- Age-adjusted, so `β` is a within-child effect, not the shared age trend.
- `δ_subj_q` controls for the child's stable production level (partial cross-lag control).
- Irregular spacing handled naturally ("prior wave" = previous observed age).

### Alternatives / companions

- **Population-relative baseline** (`x_lag` vs `f_U(a_prev)+δ_study` only, *not*
  subtracting `δ_subj_u`) — more strongly identified with 2-wave data and free of
  the own-intercept short-T bias, but blends within- and between-child comprehension
  on the predictor side. **Fitted as a robustness companion** (a `lag_baseline` flag):
  agreement with the primary is reassuring; divergence says whether the lead is a
  within-child process or a between-child trait.
- **Correlated subject REs / random q-slope** — level coupling, not temporal;
  doesn't answer "earlier→later". Deferred.
- **Latent change score / bivariate latent growth** — richer but data-limited
  (mostly 2 waves) and a much larger build. Deferred.
- **Latent-AR understood** (propagate measurement error into `x_lag`) — de-attenuates
  `β`; v2. v1 uses the observed prior-wave understood.
- **Gap-decay** `β·x_lag·exp(−(a_i−a_prev)/τ)` — lightweight continuous-time
  approximation; v2. v1 keeps a single `β`.

## 5. Caveats

- **Errors-in-variables → `β` is conservative.** `x_lag` uses the *observed* prior-wave understood (Beta-Binomial sampling noise), which attenuates `β` toward 0. A v1 `β>0` is a lower bound; the latent-AR-understood v2 de-attenuates.
- **Short-T / dynamic-panel bias.** With ~73% of repeated-measures children at exactly 2 waves, the within/between split leans on partial pooling, the ~93 children with ≥3 waves, and the prior; regressing on a lag tied to the child's own intercept carries a Nickell-type bias — mitigated (not eliminated) by the Bayesian pooled single-`β` setting. Check with prior/posterior predictive and a simulate-and-recover before interpreting.
- **Continuous time.** Age-adjusting the lag handles the trend; discrete "previous wave" approximates a continuous-time cross-effect (CT-SEM). Gap-decay (v2) refines it.
- **First wave / missing prior understood → `x_lag = 0`** (does not inform `β`); the report states how many observations inform the lag.
- **Not a full autoregressive CLPM** (no explicit `S(t−1)→S(t)` beyond `δ_subj_q`); `β` reads as "prior within-child comprehension → current production beyond stable level".
- **Direction not baked in.** `β ~ Normal(0, σ_β)` centered at 0; the DS receptive-advantage / expressive-delay literature predicts `β>0` (and its intervention relevance), but that is the hypothesis under test.

## 6. Decision gate

**GO** if the modelling lead approves this encoding (the `x_lag` definition, the
observed-vs-latent choice, single `β`, no gap-decay in v1). Then: implement VG16
as a cross-lag variant of `common_bivariate_re`, register
(`definitions.VG16` / `model_vg16.py` / `fit_model.py` / `MODEL_REGISTRY`),
prior-predictive-check `β`, fit dev→rep with nutpie, diagnostics (R̂/ESS/divergences),
report `β` + HDI, add `docs/models/vg16/index.qmd` + a `docs/models/README.md` row.

## 7. Implementation outline

- **Data prep**: sort observations by `(subject, age)`; per child, shift the
  understood-deviation to form `x_lag` (0 for first wave / missing prior understood).
- **Definition**: extend `BivariateModelDefinition` (or a small subclass) with
  `use_cross_lag_q: bool` + a `β` prior; `VG16` = VG09 settings + cross-lag on.
- **Build**: add `β · x_lag_obs` to the `q` logit in the RE engine (guarded by the flag).
- **Report**: a "Within-child receptive → expressive (cross-lag)" section reporting `β`, its HDI, and P(β>0).
