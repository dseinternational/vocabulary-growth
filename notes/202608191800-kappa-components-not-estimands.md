> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

# The Beta-Binomial dispersion's components are coordinates, not estimands

**Status:** measured, 2026-08-19. Bears on the DS predictive intervals above roughly 60 months and on the 84/90 reporting cap. Does **not** bear on the between-child scale bias in [#229](https://github.com/dseinternational/vocabulary-growth/issues/229) / [#225](https://github.com/dseinternational/vocabulary-growth/issues/225), for reasons in §6.

## 1. The rule

`build_kappa_of_z_anchored` parameterises dispersion as `kappa(z) = kappa_min + exp(a + b z)`, with priors on an asymptote and on the _excess_ over it at each of two anchor ages. Three free parameters describe a curve the data pin at about two places.

**The anchor totals are identified. Their split into floor-plus-excess is not.** Only totals at ages should be reported or interpreted. `kappa_min`, `kappa_excess_young` and `kappa_excess_old` are sampling coordinates; quoting one, or reading a trend into it, is reading noise.

This is not a new claim in kind — the two-anchor form exists precisely so the _priors_ sit on interpretable quantities — but the size of the non-identification was not on the record, and the docstring's own recovery discussion invited exactly the component-level reading this note forbids.

## 2. The measurement

Parameter recovery, three replicates each. Components:

|                      | VG10                   | VG20                   | VG12                 |
| -------------------- | ---------------------- | ---------------------- | -------------------- |
| `kappa_min`          | −60.1%, −55.7%, −53.9% | +34.4%, −45.8%, −63.3% | +283%, +235%, −11.4% |
| `kappa_excess_old_s` | —                      | +263%, +155%, +112%    | —                    |

The totals containing them, on the same fits:

|               | VG20                 |
| ------------- | -------------------- |
| `kappa_old_u` | −1.4%, +9.6%, +2.8%  |
| `kappa_old_s` | −0.8%, −9.1%, −14.1% |

A sum recovering to within a few percent while its addends are out by factors of two to four is the signature of a redundant coordinate, not of a broken model.

The floor's _share_ of reported kappa, computed from VG20's own posterior (reconstruction validated against the reported trajectory: κ_u at 84 months 8.73 against 8.75):

|              | median share | 89% interval   |
| ------------ | ------------ | -------------- |
| κ_u at 48 mo | 14.2%        | [4.1%, 35.3%]  |
| κ_u at 84 mo | 42.5%        | [13.2%, 78.3%] |
| κ_s at 48 mo | 77.8%        | [16.0%, 94.0%] |
| κ_s at 84 mo | 95.2%        | [21.0%, 99.8%] |

Those intervals are the point. The share is barely determined at all, so any statement of the form "most of the reported dispersion at old ages is the floor" is true of the median and unsupported as a fact.

## 3. The reported totals are data-driven — a claim of mine that did not survive checking

Earlier on 2026-08-19 I wrote, in conversation, that "above roughly 60 months neither parameterisation estimates dispersion from data — both report a prior, and the two priors disagree by 7–14%". **That is wrong**, and it is recorded here because it was nearly written into the reporting caveat.

The `kappa-floor-recentred` variant moves the floor prior's median from 3.0 to 7.8, a **+160%** move. What the posterior does:

| quantity             | move   | elasticity |
| -------------------- | ------ | ---------- |
| κ_u at 84 mo         | +14.2% | **0.09**   |
| κ_u at 72 mo         | +6.6%  | 0.04       |
| κ_s at 84 mo         | +7.4%  | **0.05**   |
| κ_s at 72 mo         | +4.7%  | 0.03       |
| `kappa_min_u` itself | +60.8% | 0.38       |
| `kappa_min_s` itself | +27.3% | 0.17       |

An elasticity of 1.0 would mean the posterior follows the prior; 0.0 that the prior is irrelevant. At **0.03–0.09 on the reported quantities the data are doing nearly all the work**, and even the "unidentified" floor only moves 0.38 of the way. The 7–14% figure was real — tier noise is about 1%, measured on the anchor variant, which moved κ_u at 84 months by −0.9% — but it is evidence _for_ the data dominating, not against.

The intervals say the same thing. At 84 months κ_u is 8.75 [5.75, 12.07] on the baseline and 9.99 [6.39, 13.77] re-centred: the shift is about a fifth of the interval width, and the two are not distinguishable.

I had also quoted floor shares as point values computed from a posterior _mean_ floor over a posterior _median_ kappa. The corrected posteriors are in §2; the point estimates were roughly right and wildly over-precise.

## 4. The caveat that should travel

> The Beta-Binomial dispersion's components are not separately identified; only totals at ages are estimands. The reported totals are data-driven (prior elasticity 0.03–0.09), but at 72–90 months they carry a wide posterior interval — κ_u = 8.75 [5.75, 12.07] at 84 months — together with a residual prior sensitivity of order 10%.

That supports the 84/90 reporting cap on the ground that the interval is wide and mildly prior-sensitive out there. It does **not** support any claim that the number is invented, and the cap's justification should not be written that way.

## 5. The three variants

Registered on VG20 (`2a3cc59`), fitted at `test`. All three leave the between-child scales alone: `tau_subj_u` within ±0.3%, `tau_subj_q` within ±0.2%, `rho_uq` within ±0.8%.

| variant                    | divergences | max R-hat  | min ESS |
| -------------------------- | ----------- | ---------- | ------- |
| `kappa-anchor-18-72`       | 3           | 1.0103     | —       |
| `kappa-floor-recentred`    | 2           | 1.0118     | —       |
| `kappa-anchor-18-72-floor` | **0**       | **1.0065** | 342     |

All three miss the hard gate at `test` tier (R-hat ≤ 1.01, ESS ≥ 400), so none of this is publication-grade yet. The combination samples better than either change alone, which is the one thing the anchor move demonstrably buys.

Because the three form a 2 × 2 with the baseline, the two effects can be separated, and they do not combine:

| change at 84 months | κ_u    | κ_s   |
| ------------------- | ------ | ----- |
| anchors only        | −0.9%  | −1.4% |
| floor only          | +14.2% | +7.4% |
| both                | +13.8% | +6.4% |

**Both ≈ floor alone.** Moving the anchors 24 months contributes nothing to the reported curve on top of the floor prior, and nothing on its own: κ_u within 1% at every age above 24 months, κ_s within 3%. Its only estimate-side effect is to lower the floor's median share on q (82% → 73% at 84 months) and none at all on comprehension.

So the reported dispersion is **not** an artefact of anchor placement, and the single lever on it is where `kappa_min`'s prior is centred — which §3 shows is a weak lever. The anchor move is worth having for sampling geometry and for placing priors inside the reporting range; it is not worth having as a correction to any number.

## 6. What this does not explain

Six kappa treatments — baseline, `kappa-const`, `kappa-flat`, `kappa-anchor-18-72`, `kappa-floor-recentred`, `kappa-anchor-18-72-floor` — leave `tau_subj_u` between 0.784 and 0.798, a spread of one tenth of its own posterior SD (0.031). **The kappa parameterisation is not the route to #229's between-child scale bias.** Twelve conditions of the stripped replication study say the same from the other side. That line of attack should be closed.

## 7. The typically-developing side

The same diagnostic across the TD models, with the floor's share at whichever end the slope points away from:

| model           | anchors | reports to | floor share      | floor prior | conditional calibration |
| --------------- | ------- | ---------- | ---------------- | ----------- | ----------------------- |
| VG03 spoken     | 12, 20  | 30         | **95% at 30 mo** | 3.0         | —                       |
| VG04 understood | 12, 18  | 25         | 37%, flat        | 3.0         | 11.28                   |
| VG11 spoken     | 12, 20  | 30         | 56% at 30 mo     | 6.0         | **6.00**                |
| VG12 understood | 12, 20  | 25         | 3–9%             | 3.0         | unidentified (0.00)     |
| VG13 understood | 12, 17  | 18         | **85% at 9 mo**  | 30.0        | 36.84                   |
| VG13 q          | 12, 17  | 18         | 11–15%           | 3.0         | unidentified (0.00)     |

Three things follow. **VG03 is the family's worst case** — 95% at 30 months, and it carries no random effects, so kappa is its only device for scatter. **VG11's floor prior is already conditionally calibrated** (6.0 against 6.00), which is why the DS joint blocks stand out: they never had that treatment and still sit at the generic `log(3.0)`. And unlike DS, the TD spoken frames have **3,000–4,000 administrations at any anchor one might move to**, against 51 at the DS pool's 72 months — so if anchor placement ever does matter, TD is where it can be done properly.

## 8. Open

- Score `kappa` **at reporting ages** in the recovery harness. It currently scores components and anchor totals, neither of which is what the 60–90 month intervals are made of. This is the check that would settle §4 rather than leave it as a caveat.
- Decide whether `kappa-anchor-18-72-floor` is promoted; it needs a `rep` fit and a `test`-tier baseline for a like-for-like comparison, since the pairing above is `test` against a `rep` record.
- The `_DS_JOINT_*_KAPPA_RE` comments cite totals of 110 at 24 months and 33 at 48; the conditional calibration returns 71.2 and 27.7 on the same frame. That discrepancy is unexplained and predates this work.
