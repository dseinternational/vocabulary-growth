# Modelling the sign–speech link: a correlated subject block on VG15 first, a sign → speech cross-lag deferred

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

> [!IMPORTANT]
> Proposal record, 2026-09-04. Two issues were opened and nothing was built: [#296](https://github.com/dseinternational/vocabulary-growth/issues/296) proposes **VG24**, VG15 with a correlated 3×3 subject random-effect block, and [#297](https://github.com/dseinternational/vocabulary-growth/issues/297) proposes **VG25**, VG15 with a sign → speech cross-lag, explicitly deferred on [#242](https://github.com/dseinternational/vocabulary-growth/issues/242). Every count below is read from VG15's registered prepared frame at commit `db4b298` through `vocab_growth.analysis_frames`; every estimate is quoted from the two descriptive notes it cites. No fit was run and no VG15 trace was available locally.

## 1. The question

Asked on 2026-09-04: have we looked at whether early signing predicts later spoken language, and can it be modelled properly? The practice question behind it is whether signing is only a substitute for speech that is not yet there, or whether children who sign more go on to say more.

## 2. What already exists

Two descriptive analyses, neither a fitted model:

- [202608160930](202608160930-early-signing-and-later-speech.md): among 147 children with signing, comprehension and speech at a first joint wave and speech at a later one, and holding earlier speech and comprehension standing fixed, the signed share of comprehension predicts later spoken vocabulary at **+0.19 SD per SD, 89% ETI [0.03, 0.36]**, unmoved by study fixed effects and stable under leaving any one study out (+0.13 to +0.24). The binary signer/non-signer contrast is **not identifiable**: two studies are almost all signers, one has none, and only three carry within-study variation, so the contrast spans zero and flips sign when `ie_02` is dropped. The comparison report quotes this in the callout "Does early signing predict later speech?" in `docs/comparison/index.qmd`.
- [202608141500](202608141500-sign-speech-additivity-and-cross-lag.md) §3: the **concurrent** between-child association at matched comprehension and age is zero to negative, bracketed at [−0.44, −0.05] by two specifications with opposing mechanical biases. Its §5 fitted a descriptive sign → speech cross-lag on 80 wave pairs from 55 children and found it null but uninformative, with a detectable-effect floor near 0.38.

The two are compatible. At a given moment children who sign more do not speak more: substitution. Among children at the same speech and comprehension level, the ones signing a larger share of what they understand gain more speech afterwards. Those are different quantities, and nothing in the family estimates either one.

## 3. Why no fitted model answers it

VG15's three subject random intercepts (understood, `q`, signed) are independent scales with no correlation structure, as [202608141500](202608141500-sign-speech-additivity-and-cross-lag.md) §1 recorded. VG16's cross-lag runs understood → `q`, not sign → speech. VG17 and VG18 are sign-group contrasts on the level of spoken and total production, exploratory and unpublishable. The 16 August analysis is a two-wave residual regression outside the model family.

## 4. Two proposals, in this order

**VG24 first** ([#296](https://github.com/dseinternational/vocabulary-growth/issues/296)). Replace VG15's three independent subject blocks with a joint prior carrying a free 3×3 correlation matrix, nested at the identity, and read `rho_sign_q`: do children who persistently sign a larger share of what they understand also persistently say a larger share of it? This is the step [202608151140](202608151140-cross-lag-not-for-models-of-record.md) §4 argued for on VG10 and which became VG20, the model of record since 2026-08-19. It applies to every child carrying both scales rather than the longitudinal subset, does not depend on follow-up design (though it does depend on which studies record a signed marginal, §6), and has no attenuation to argue about.

**VG25 deferred** ([#297](https://github.com/dseinternational/vocabulary-growth/issues/297)). The joint-engine analogue of VG16: a child's prior-wave signed share of comprehension, relative to a baseline, shifts the logit of their current `q` through one coefficient. It is deferred because every open item on #242 maps one-to-one onto it (the administration-wave definition across forms, which the complete-wave grouping now in `cross_lag.py` does not settle; the LOO leakage a lag creates; wave-sequential recovery; marginal predictions that condition on no lag history; the gap and leave-one-study-out sensitivities), and because the cross-lag is implemented only in the bivariate engine and would be built from scratch in `common_joint_modality`. Building a second lag model before those are settled on VG16 doubles the debt.

The ordering is also a matter of interpretability. Under VG16's population baseline the lag mostly absorbed the covariance between persistent standings, because within-child deviations have no memory beyond the occasion ([202608151140](202608151140-cross-lag-not-for-models-of-record.md) §3). A population-baseline sign lag would do the same, and what it would absorb is `rho_sign_q`. The lag cannot be read until the correlation is known.

## 5. Three quantities, each with a name

| quantity                                                                  | instrument                        | expected sign                                                                                       |
| ------------------------------------------------------------------------- | --------------------------------- | --------------------------------------------------------------------------------------------------- |
| persistent between-child correlation of signed share and production ratio | VG24's `rho_sign_q`               | zero to negative, the substitution result of §2                                                     |
| prospective within-child effect of prior-wave signed share on current `q` | VG25 with a within-child baseline | unknown; VG16's within variant is positive without excluding zero, and joint estimation inflates it |
| later speech given earlier signed share, conditional on earlier speech    | the 16 August regression          | positive, +0.19 [0.03, 0.36]                                                                        |

The third is neither of the first two: it conditions on speech at the earlier wave, which a `q` subject intercept (the child's average across all waves) does not do. Reproducing it inside a model needs prior-wave spoken standing as a second lagged predictor or a first-wave conditioning structure, which is a wave-definition question for #242. The report callout should end up stating which of the three it is quoting.

## 6. Support in VG15's prepared frame

VG15's frame has 763 children, 416 of them with signing information of some kind. Two structural facts decide what each proposal can see.

**The subject shifts enter only the marginal likelihoods.** The four-cell and produced-cell Dirichlet-Multinomial likelihoods are fed population + study marginals with no subject shift, deliberately, so the subject block cannot pull `psi`. Under that design `rho_sign_q` is identified by the children carrying both a signed and a spoken marginal:

| study   | children with both marginals |
| ------- | ---------------------------: |
| `ie_02` |                           65 |
| `uk_02` |                           36 |
| `uk_04` |                           18 |
| `uk_05` |                           16 |
| `uk_06` |                           11 |
| total   |    **146** (110 seen twice+) |

Extending the subject shifts into the cell likelihoods would bring the `es_01`, `uk_07` and `nz_01` children in, but it changes what `psi` means and is a separate proposal.

**A lag needs a signed-share source at one wave and a spoken outcome at a later one**, and its support depends on two design choices, the denominator for `nz_01` (whose cells partition produced words, not understood) and which likelihoods the lag enters:

| lag design                                                                         | children | later spoken obs | median gap (IQR) | by study                                                   |
| ---------------------------------------------------------------------------------- | -------: | ---------------: | ---------------- | ---------------------------------------------------------- |
| understood-denominator source, lag in every spoken likelihood                      |      136 |              201 | 7 mo (3–11)      | `ie_02` 46, `uk_02` 32, `uk_07` 27, `uk_04` 16, `uk_05` 15 |
| as above plus `nz_01` on a produced denominator                                    |      164 |              279 | 6 mo (3–11)      | + `nz_01` 28                                               |
| understood-denominator source, lag in the spoken **marginal** only, as the REs are |       87 |              121 | 3 mo (3–7)       | `ie_02` 46, `uk_04` 16, `uk_05` 15, `uk_02` 10             |

The VG16 fit of 2026-08-14 rested on 250 children and 412 observations at a 6-month median gap; the corrected wave-grouped builder gives 248 children and 473 spoken observations on today's frame. The dose predictor varies within every study here, so the identification failure that sinks the binary contrast does not apply, but with an interval that only just cleared zero at n = 147 in the descriptive analysis, VG25 should be expected to produce a coefficient near zero rather than a headline.

## 7. Design decisions recorded on the issues

- **VG24:** correlations as named deterministics rather than a packed Cholesky vector, so summaries and the recovery scorer read them by name; `eta = 2` to match VG20 and VG23, noting that for n = 3 the LKJ marginal is `(rho + 1)/2 ~ Beta(2.5, 2.5)` (SD ≈ 0.41) rather than VG20's Beta(2, 2) (SD 0.45); a definition subclass so VG15 keeps its fingerprint; reject any flag combination short of all three subject effects; report `tau_subj_sign`'s prior-to-posterior contraction beside the estimate, since that scale has no calibration of its own.
- **VG25:** a port of `cross_lag.prev_wave_lag` with signed over understood as the source, which needs a per-row denominator in place of its scalar `n_trials` and a source-selection rule for a ratio; `nz_01` carries no lag by default; the within-child baseline recommended with the population baseline as the registered sensitivity; the lag's entry into the cell likelihoods decided with the same argument the subject-shift decision had; recovery unsupported until #242's wave-sequential simulation exists; if VG24 is adopted, VG25 inherits its definition so the lag is estimated with the correlated block present.

## 8. What would change the sequencing

- **A near-zero realised correlation closes #296.** The first check needs no code: the correlation of VG15's fitted subject intercepts, `delta_subj_sign` against `delta_subj_q`, from the model-of-record trace, as [202608151120](202608151120-vg16-cross-lag-quantified.md) did for VG16 (+0.135). No VG15 trace was on either local output root on 2026-09-04, so this waits for the next fit or a fetch from blob storage.
- **#242 closing unblocks #297**, or its wave definition, LOO replacement and wave-sequential recovery being accepted as the standard a second lag model is held to.
- **Neither is fitted before the [#281](https://github.com/dseinternational/vocabulary-growth/issues/281) / [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) refit sequence is decided**; the `us_03` ingestion changes VG15's frame anyway.
- **VG15's recovery bias on `psi`** ([#226](https://github.com/dseinternational/vocabulary-growth/issues/226)) is inherited by both; its mechanism should be understood before either model's recovery is read.

None of this changes the causal position. Signing is taught because a child is not talking, and no model in the family can remove that selection. The one thing in the descriptive result's favour is that the residual selection runs against a positive finding, so +0.19 is more likely attenuated than inflated.

## 9. Where this is recorded

- [#296](https://github.com/dseinternational/vocabulary-growth/issues/296) (VG24) and [#297](https://github.com/dseinternational/vocabulary-growth/issues/297) (VG25), both unlabelled, matching [#224](https://github.com/dseinternational/vocabulary-growth/issues/224).
- This note, indexed in [README.md](README.md).
- The comparison-report callout and the two descriptive notes are the reporting homes to update when either model is fitted.
