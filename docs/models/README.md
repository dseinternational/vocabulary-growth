# Model inventory

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

> [!WARNING]
> This is work in progress. All models and their output are preliminary and likely
> to change as the models evolve and further data are received.

This document lists every model in the `vocab_growth` family, what each one
targets, and how the models relate to one another. It is a map, not a
specification: for the full statistical detail of any single model, read its
report under `docs/models/vgNN/` (e.g. [`vg01/`](vg01/)) or the consolidated
[technical report](../report/index.qmd).

Each model is a thin module in
[`src/vocab_growth/models/`](../../src/vocab_growth/models) (e.g.
[`model_vg01.py`](../../src/vocab_growth/models/model_vg01.py)) that
selects a definition from
[`definitions.py`](../../src/vocab_growth/models/definitions.py) and dispatches to
a shared fitting engine. All procedural code (model building, sampling,
diagnostics, plotting, reporting) lives in the `common_*.py` engines; the models
themselves differ only by **population**, **outcome(s)**, **structure**, and
**priors**.

## How the models are named

Models are numbered `VG01`–`VG16` in roughly the order they were developed. The
numbering is historical rather than hierarchical — a higher number means "added
later", not "supersedes". Models build on one another along two main lineages
(joint understood + spoken for Down syndrome, and the signing models), with
several typically-developing counterparts fitted in parallel for comparison.

The three axes that distinguish the models:

- **Population** — children with **Down syndrome (DS)** or **typically
  developing (TD)** children.
- **Outcome** — words **understood**, words **spoken**, words **signed**, or a
  **joint** model of more than one outcome.
- **Structure** — whether the model adds study-level and/or subject-level random
  intercepts, GP anchoring, or a sign–speech association term on top of the
  baseline trajectory model.

## Inventory

| Model                  | Population | Outcome(s)                   | What it adds / its purpose                                                                                                                                                                                                |
| ---------------------- | ---------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [VG01](vg01/index.qmd) | DS         | Spoken                       | Baseline age → words spoken trajectory.                                                                                                                                                                                   |
| [VG02](vg02/index.qmd) | DS         | Understood                   | Baseline age → words understood trajectory.                                                                                                                                                                               |
| [VG03](vg03/index.qmd) | TD         | Spoken                       | TD counterpart to VG01.                                                                                                                                                                                                   |
| [VG04](vg04/index.qmd) | TD         | Understood                   | TD counterpart to VG02.                                                                                                                                                                                                   |
| [VG05](vg05/index.qmd) | DS         | Understood + spoken (joint)  | Baseline joint model; spoken modelled as a fraction `q(a)` of understood.                                                                                                                                                 |
| VG06 _(retired)_       | TD         | Understood + spoken (joint)  | TD counterpart to VG05; retired after the WS-comprehension data issue (see below) — superseded by VG13.                                                                                                                   |
| [VG07](vg07/index.qmd) | DS         | Understood + spoken (joint)  | VG05 + study random intercepts.                                                                                                                                                                                           |
| [VG08](vg08/index.qmd) | DS         | Understood + spoken (joint)  | VG07 + subject random intercepts on understood.                                                                                                                                                                           |
| [VG09](vg09/index.qmd) | DS         | Understood + spoken (joint)  | VG08 + subject random intercepts on the production ratio `q`.                                                                                                                                                             |
| [VG10](vg10/index.qmd) | DS         | Understood + spoken (joint)  | VG09 + per-draw GP anchor at 54 months (stabilisation); `q` anchors match VG09.                                                                                                                                           |
| [VG11](vg11/index.qmd) | TD         | Spoken                       | VG03 + dataset-level study random intercepts + GP anchor at 19 months (uses full data instead of subsampling).                                                                                                            |
| [VG12](vg12/index.qmd) | TD         | Understood                   | VG04 + dataset-level study random intercepts + GP anchor at 19 months (uses full data instead of subsampling).                                                                                                            |
| [VG13](vg13/index.qmd) | TD         | Understood + spoken (joint)  | Young TD joint model, ages 8–18 months only; study random intercepts + GP anchor at 13 months.                                                                                                                            |
| [VG14](vg14/index.qmd) | DS         | Understood + spoken + signed | Adds signing as a third ratio `r(a)`; total expressive vocabulary derived assuming sign/speech independence given age.                                                                                                    |
| [VG15](vg15/index.qmd) | DS         | Understood + spoken + signed | VG14 + within-understood sign–speech association `psi` + study & subject random intercepts + VG10 stabilisation.                                                                                                          |
| [VG16](vg16/index.qmd) | DS         | Understood + spoken          | VG09 + a within-child cross-lag (prior understood → current `q`): earlier receptive → later expressive. Population-relative headline (≈ null); the within-child (RI-CLPM) contrast is biased by short-T with 2-wave data. |

## Shared architecture

Despite their differences, every model is built from the same components:

- **Outcome** — each vocabulary count is treated as a bounded count out of an
  **810-item common reference inventory**, so DS and TD counts sit on a
  comparable scale regardless of which checklist (MB-CDI Words & Gestures, Words
  & Sentences, Oxford CDI, etc.) produced them.
- **Likelihood** — a **Beta-Binomial** with **age-varying dispersion**, so the
  degree of between-child heterogeneity can change across development rather than
  being fixed.
- **Mean trajectory** — on the logit scale, the expected proportion is a
  **linear developmental trend plus a Hilbert-Space Gaussian Process (HSGP)**
  term that captures smooth nonlinear departures from the trend. The linear trend
  is parameterised indirectly through the expected proportion at two **reference
  (anchor) ages**, which makes the priors interpretable.
- **Inference** — fitted with **PyMC** using the **nutpie** NUTS sampler, in an
  iterative Bayesian workflow (prior predictive checks → sampling → diagnostics →
  posterior predictive checks → summaries and plots).

### Structural decomposition (joint models)

The joint models do not model each outcome independently. Instead, production is
expressed as an age-varying fraction of comprehension:

- `p_U(a)` — proportion of the inventory **understood** at age `a`.
- `q(a) = P(speak | understood)` — the **production ratio**, so words spoken =
  `p_U(a) · q(a)`.
- `r(a) = P(sign | understood)` — the **signing ratio** (VG14/VG15 only), so
  words signed = `p_U(a) · r(a)`.

This decomposition keeps spoken and signed vocabulary bounded by comprehension by
construction, and lets the models report directly on quantities of practical
interest (how much of what a child understands they can also say or sign).

### Random intercepts and GP anchoring

The DS data are pooled across multiple studies, and several contain repeated
observations of the same children. Later models add hierarchy to account for
this:

- **Study (dataset) random intercepts** absorb systematic level differences
  between source studies.
- **Subject random intercepts** capture stable between-child differences for
  children observed more than once.
- **GP anchoring** constrains the HSGP correction to be zero at a reference age
  for every posterior draw. This removes a redundancy ("ridge") between the
  linear trend, the GP, and the intercepts that otherwise degrades sampling once
  random intercepts are present.

## Model lineages

### Single-outcome baselines (VG01–VG04)

The simplest models: one population, one outcome, no hierarchy. VG01/VG02 cover
spoken and understood vocabulary for children with Down syndrome; VG03/VG04 are
the typically-developing counterparts. These establish the baseline trajectory
shape that the joint and hierarchical models build on.

### Joint understood + spoken, Down syndrome (VG05 → VG07 → VG08 → VG09 → VG10)

This is the main DS development lineage. Each step adds structure to the previous
model:

- **VG05** — baseline joint model of understood and spoken via the `q(a)`
  decomposition.
- **VG07** — adds **study random intercepts** to absorb between-study level
  differences.
- **VG08** — adds **subject random intercepts on understood**.
- **VG09** — extends subject random intercepts to the **production ratio `q`** as
  well.
- **VG10** — the same structure as VG09, plus a **per-draw GP anchor at 54
  months**, introduced to resolve sampling diagnostics on the `q`-trajectory
  hyperparameters by removing the trend/GP/intercept redundancy. (VG10 was
  originally also given tighter, VG07-posterior-informed `q` anchors; #155
  broadened those back to the shared weakly-informative DS-joint values to remove
  the prior-data double-dipping, so the `q` anchors now match VG09 and the GP
  anchor is the only structural difference.)

### Joint understood + spoken, typically developing (VG13; VG06 retired)

- **VG06** _(retired)_ — was the TD counterpart to VG05, used for
  DS-versus-TD comparison. Wordbank's CDI: Words & Sentences (WS) form records
  `comprehension` as a production proxy rather than an independent measurement,
  which had been telling VG06 that `U = S` for most TD rows above 18 months
  (see `notes/202605151630-vg06-ws-comprehension-issue.md`). VG06 was removed
  from the model family rather than re-validated; **VG13** is now the joint TD
  model used for DS-versus-TD comparison.
- **VG13** — a TD joint model restricted to **ages 8–18 months**, where the
  Wordbank Words & Gestures and Oxford CDI data are dense and the WS production
  proxy bias is avoided entirely (WS is excluded, not merely down-weighted).
  Uses dataset-level study random intercepts and a GP anchor at 13 months.

### Typically developing single-outcome, with hierarchy (VG11, VG12)

Re-fits of VG03/VG04 that use the **full** TD data pool (rather than
subsampling) together with **dataset-level study random intercepts** and a GP
anchor at 19 months. The study intercepts absorb between-lab variation, so all
qualifying observations can contribute.

### Signing models, Down syndrome (VG14 → VG15)

- **VG14** — extends the joint model with a third production ratio for
  **signing**, `r(a)`. Total expressive vocabulary (`p_any`) is derived by
  assuming sign and speech are **independent given age** — a deliberately simple
  starting point that gives an upper bound on combined production.
- **VG15** — replaces that independence assumption with a directly estimated
  **within-understood sign–speech association** (`psi`, a scalar Plackett odds
  ratio identified from the four-cell sign/speak cross-tabulation in the `uk_02`
  dataset). It adds **study and subject random intercepts on all three
  trajectories** and carries VG10's stabilisation (per-draw GP anchor at 54
  months with the tightened `q`-GP amplitude `eta_q`), yielding a
  **data-identified** total expressive vocabulary rather than VG14's
  independence-based bound.
