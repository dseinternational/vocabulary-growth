# Does early signing predict later speech? What these data can and cannot answer

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **One of the two contrasts is identifiable and one is not, and the design table in §3 is why.** The dose measure — how much a child signs relative to what they understand — predicts later spoken vocabulary and survives every robustness check applied. The binary "does this child sign at all" does not separate from study and flips sign under single-study deletion. Reproduced by `scripts/experiments/early_signing_and_later_speech.py`; figures are the 2026-08-16 run.

## 1. The question, and why it is worth asking carefully

The signing results say that at two years about three-quarters of what a child with Down syndrome can express is available only in sign ([the comparison report](../docs/comparison/index.qmd), "What signing is worth to the child"). The immediate practice question is whether signing is only a _substitute_ for speech that is not yet there, or whether children who sign more go on to _say_ more.

This is the same shape of question as [202608151500](202608151500-within-child-crosslag-feasibility.md) §1, and it inherits that note's first obstacle: an association across children is not an intervention effect. It adds a second, worse one. Signing is taught **because** a child is not talking. So early signing is a marker of low speech by construction, and any naive comparison of signers with non-signers measures that selection.

## 2. Design

One row per child. At the **early wave** `t0` — the first administration carrying signed, understood _and_ spoken — record signing, comprehension and existing speech. At the **latest later wave** `t1` record spoken vocabulary. Every measure is an age- and study-adjusted logit residual, the scoring [202608141600](202608141600-rank-stability-tracking.md) uses, so "standing" means the same thing across the two notes.

$$\text{resid\_spoken}(t_1) \sim \text{resid\_spoken}(t_0) + \text{resid\_understood}(t_0) + \text{signing}(t_0) + \text{gap} + \text{age}_0$$

**Conditioning on `resid_spoken(t0)` is the analytical heart of it.** Without it the coefficient is dominated by the indication: children are taught to sign when they are not talking, so signers start lower and finish lower. With prior speech held, the question becomes the answerable one — among children at the same starting speech level _and_ the same comprehension standing, does signing more predict more speech later?

Two signing measures, because they are not the same question:

- **`signs`** — binary, does the child sign at all. The practitioner framing.
- **`sign_dose`** — the adjusted residual of the signed fraction of comprehension. Keeps within-study contrast where the binary has almost none.

Inference is a child-level bootstrap (3,000 resamples, 89% interval), plus leave-one-study-out, which matters more than the bootstrap here for the reason §3 gives.

## 3. The design table, which decides what is identifiable

147 children qualify, from six studies. Median age at `t0` is 28 months, median gap 13 months.

| study | children | signers | signer rate | identifying? |
| ----- | -------: | ------: | ----------: | ------------ |
| ie_02 |       47 |      33 |        0.70 | ✅           |
| uk_02 |       29 |      29 |    **1.00** | ❌           |
| uk_07 |       27 |      26 |    **0.96** | ❌           |
| uk_04 |       16 |       9 |        0.56 | ✅           |
| uk_05 |       15 |       9 |        0.60 | ✅           |
| uk_01 |       13 |       0 |    **0.00** | ❌           |

**Signing status is very nearly a function of study.** Three of six studies are all-or-nothing, carrying 69 of the 147 children; they contribute to the binary contrast only through a between-study comparison. Only `ie_02`, `uk_04` and `uk_05` — 78 children — carry within-study variation in whether a child signs.

This is not a nuisance to be adjusted away. It is the reason the binary result below cannot be believed, and it would have been invisible in a table of coefficients.

## 4. Result

Per SD of the predictor, on the logit-residual scale:

| model                      |   β per SD | 89% ETI              |    P(>0) |
| -------------------------- | ---------: | -------------------- | -------: |
| `signs`                    |     +0.076 | [−0.104, +0.245]     |     0.75 |
| `signs` + study FE         |     +0.096 | [−0.131, +0.325]     |     0.76 |
| **`sign_dose`**            | **+0.194** | **[+0.028, +0.363]** | **0.97** |
| **`sign_dose` + study FE** | **+0.189** | **[+0.028, +0.358]** | **0.97** |

Leave-one-study-out, the check that matters:

| dropped |                         `signs` |                 `sign_dose` |
| ------- | ------------------------------: | --------------------------: |
| ie_02   |                      **−0.078** |                      +0.127 |
| uk_01   |                          +0.073 |                      +0.170 |
| uk_02   |                          +0.107 |                      +0.210 |
| uk_04   |                          +0.131 |                      +0.213 |
| uk_05   |                          +0.176 |                      +0.176 |
| uk_07   |                          +0.202 |                      +0.241 |
| range   | **−0.08 to +0.20 — sign flips** | **+0.13 to +0.24 — stable** |

**The dose measure is the one that stands.** It is essentially unmoved by study fixed effects (+0.194 → +0.189), which says it is not a between-study artefact, and it keeps its sign and rough magnitude when any single study is removed. **The binary measure does not stand**: its interval spans zero, and dropping `ie_02` — the only large study with real internal variation — reverses it.

So: _how much_ a child signs relative to what they understand carries information about their later speech. _Whether_ they sign at all cannot be assessed in these data, and the honest reason is that in five of six studies it was effectively decided by the study.

## 5. What this does and does not license

**It survives a bias running the other way.** The residual indication — signing taught to children who are not talking — pushes the coefficient down. A positive estimate is therefore harder to manufacture than a negative one, and +0.19 should be read as attenuated rather than inflated. That is the single strongest thing that can be said for it.

**It is still not causal, and the confounding is severe rather than pro forma.** Signing is a decision taken by families and by the programmes these studies recruit from. Families who sign differ in ways no covariate here captures — engagement, access to intervention, therapist contact. Conditioning on prior speech and comprehension removes the crudest version of the selection, not the family-level version.

**And [202608151500](202608151500-within-child-crosslag-feasibility.md) §1's second obstacle is untouched.** Even a clean within-child estimate would not answer the intervention question, because an intervention perturbs one input and nothing guarantees it lands on the observed curve.

**What would answer it:** a randomised or quasi-randomised comparison, or at minimum a study where signing was introduced at a time not chosen by the child's speech. Nothing in this pool has that structure.

## 6. Caveats

- **The `t0` wave is chosen by data availability**, not by design: it is the first administration that happens to carry all three outcomes. Its median age of 28 months is later than the ages at which signing is typically introduced, so this is not "early signing" in the intervention sense — it is signing at first joint measurement.
- **`sign_dose` is a residual of a ratio** (signed ÷ understood), so it inherits comprehension measurement error in its denominator. That attenuates it further, in the same direction as the indication bias.
- **The gap between waves varies widely** (IQR 3–14 months) and is included as a covariate rather than matched on. A per-gap breakdown is not attempted at n = 147.
- **Six studies, one bootstrap.** The child-level bootstrap treats children as exchangeable within the fitted design; with signing this close to a study-level property, the leave-one-study-out range is the more honest uncertainty statement, and it is wider than the bootstrap interval for the binary measure.
- **Signed vocabulary is recorded differently across studies** — the signing-source harmonisation of #163 applies here as everywhere.
