# A repeaters-calibrated prior does not repair VG12's geometry; pinning the split does

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

Date: 2026-09-07. Five `test`-tier fits of VG12 on the project's Windows workstation, about an hour of compute. The pre-refit test that [202609072200](202609072200-vg08-replication-thinning.md) recommended for [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.6, run the same day. It refines that note's recommendation: [#229](https://github.com/dseinternational/vocabulary-growth/issues/229) option 2 works as a **cut**, not as a prior, and the reason it does not work as a prior is itself the finding.

## The question

The thinning experiment established that the typically developing (TD) models' low energy BFMI and their τ–κ ridge are what a child effect looks like when most children are observed once, and that no coordinate change supplies the information a second visit carries. The cheapest remedy on #229's list was option 2, generalised: let the children who _can_ identify the between-child / within-child split — VG12's 998 children with two or more visits — set the prior on `subject_variance_share` for the full fit, so the 4,821 singletons contribute to the trajectory and the dispersion under a split the repeaters have already determined. This tests that on VG12, the model with the worst BFMI (0.208 at `rep`) and the strongest ridge (+0.76), at the tier the project treats as honest for a hierarchical claim.

## What was fitted

`scripts/experiments/vg12_repeaters_prior.py`, five arms, all VG12 as registered except where stated, all at `test` (4 chains × 2,000 draws after 2,000 tuning, target accept 0.9), fitted through the univariate RE engine's own stages on a supplied frame:

- **baseline** — the registered analysis frame (7,049 rows, 5,819 children, six studies, 17.2% repeaters) and the registered share prior Beta(3.9, 2.1).
- **repeaters** — only the children with two or more visits: 2,228 rows, 998 children, and — this matters below — **three** of the six studies. Study and child codes re-issued densely.
- **calibrated** — the full frame, with the share prior's Beta moment-matched to the repeaters arm's posterior on `subject_variance_share`: Beta(118.1, 68.4).
- **calibrated-half** — the same at half the concentration, Beta(59.0, 34.2), to separate the prior's location from its tightness.
- **calibrated-fixed** — the same at a thousand times the concentration, Beta(118,092, 68,434), sd 0.001: the split is pinned at the repeaters' value, a cut rather than a prior. Added after the first four arms ran, for the reason the next section gives.

Everything is scored on the trace: min BFMI per chain, corr(`tau_subject`, `kappa_young`), the energy correlations, the energy SD against √(d/2), and the posteriors on the pair and on the budget. Traces and per-arm JSON under `output/experiments/vg12-repeaters-prior/` on the workstation; `summary.csv` beside them.

## Result

| arm              | share prior           |  min BFMI | corr(τ, κ young) | corr(τ, energy) | energy SD ÷ √(d/2) | div | τ_subject (sd)    | share (sd)    | v_total (sd)  | κ young (sd) |
| ---------------- | --------------------- | --------: | ---------------: | --------------: | -----------------: | --: | ----------------- | ------------- | ------------- | ------------ |
| baseline         | Beta(3.9, 2.1)        |     0.201 |       **+0.750** |          −0.805 |               3.01 |  14 | 0.686 (0.014)     | 0.598 (0.028) | 0.789 (0.026) | 36.9 (2.2)   |
| repeaters        | Beta(3.9, 2.1)        |     0.510 |           +0.255 |          −0.585 |               1.98 |  10 | 0.591 (0.017)     | 0.633 (0.035) | 0.552 (0.034) | 58.3 (4.2)   |
| calibrated       | Beta(118.1, 68.4)     |     0.237 |           +0.689 |          −0.762 |               2.76 |  29 | 0.691 (0.013)     | 0.611 (0.021) | 0.782 (0.019) | 37.6 (2.0)   |
| calibrated-half  | Beta(59.0, 34.2)      |     0.242 |           +0.708 |          −0.779 |               2.82 |  46 | 0.689 (0.013)     | 0.606 (0.024) | 0.785 (0.021) | 37.4 (2.1)   |
| calibrated-fixed | Beta(118,092, 68,434) | **0.589** |       **−0.158** |          −0.375 |               1.81 |  22 | **0.702 (0.007)** | 0.633 (0.001) | 0.779 (0.016) | 39.5 (1.0)   |

Every arm converges (max R-hat 1.005–1.009, min ESS 594–763). The baseline at `test` reproduces the fit of record's geometry — BFMI 0.201 against 0.208, ridge +0.750 against +0.757 — so the comparison is like for like.

### A prior does nothing, and the reason is the finding

The calibrated prior moved the share posterior from 0.598 to 0.611 and left the ridge at +0.69 and the BFMI at 0.24. Halving the concentration made no difference. The reason is in the two sd columns: the **full frame's own posterior on the share (sd 0.028) is tighter than the repeaters' (sd 0.035)**. Combined as independent sources they give sd 0.021, which is exactly what the calibrated arm shows — the prior added information at about the strength the data already had, and could not dominate the split. But the repeaters are the only children with the replication that identifies a between-child / within-child split at all. The 4,821 singletons are tightening it through the Beta-Binomial's shape, and doing so more strongly than the replication does — which is the precise content of #229's reframe ("living on the assumption") and of `202608050900` §8's "strikingly tight for a parameter whose identification rests on 17% of children", measured against the alternative for the first time. The two sources also disagree: the singleton-dominated full fit puts the share at 0.598, the repeaters at 0.633, a difference of about one repeaters-posterior sd in the direction #225's recovery bias predicts (the full fit's `tau_subject` low).

### Pinning the split repairs the geometry

With the share held at 0.633, VG12's min BFMI is **0.589** — above the 0.3 threshold by a margin, and above every Down syndrome model with a child effect whose BFMI [202609061900](202609061900-td-bfmi-is-the-tau-kappa-ridge.md) reports (VG08 0.421, VG10 0.470, VG16 0.486). The τ–κ ridge is gone (−0.16, the mild negative correlation the shared budget induces once the split cannot move). The energy is no longer made of one direction: the strongest correlate is `kappa_min` at −0.43, then `kappa_old`, `tau_subject`, `b_kappa` and `v_total` all between −0.37 and −0.40, and the energy SD is 1.81 times the reference against 3.01 in the baseline. So the sampling pathology is the split direction and only the split direction; remove it and the rest of the posterior explores normally. Divergences stay at the `test` tier's level (22, against 14 in the baseline and 46 in the tempered arm) and are a `target_accept = 0.9` matter rather than a geometry one.

### What it does to the reported quantity

`tau_subject` under the pinned split is 0.702 with sd 0.007, against 0.686 with sd 0.014 in the baseline. The shift is the 0.598 → 0.633 disagreement passed through √(share · v_total); the halving of the sd is because the split's uncertainty no longer enters. Neither number is the honest one. The repeaters' share carries sd 0.035, and propagating that alone into τ gives ½ · τ · (0.035 / 0.633) ≈ 0.019 — **wider than the fit of record's interval**, not narrower. The current VG12 interval on τ_TD is overconfident, its tightness coming from functional form, and a repeaters-determined τ_TD would be about 0.70 ± 0.02 rather than 0.686 ± 0.014.

## What it means for task 4.6

The remedy for the TD refit is a **cut**: estimate the between-child / within-child split from the children who have replication, then fit the full frame with the split held at that estimate and its uncertainty propagated, rather than letting the singletons' likelihood update it. Concretely:

1. **VG11 and VG12**, which carry the variance partition: fix `subject_variance_share` from a repeaters-only fit. Propagate the repeaters' uncertainty by fitting at a small number of quantiles of its posterior (three to five fits, mixed with equal weight) or by drawing the pinned value per chain. Either is a definition-level change — a field holding the pinned value or the repeaters posterior's moments — so it belongs in the refit window.
2. **VG13, VG21 and VG23** have two free child scales and no partition. The same cut is to fix `tau_subj_u` and `tau_subj_q` from a repeaters-only fit, or to add the partition first. **This has not been tested**; the analogue of this experiment on VG13 is the next check, and it is bivariate, so about twice the cost.
3. The **reported contrast** then compares a Down syndrome τ identified by real replication with a TD τ identified by the same kind of replication, which is more comparable than the current pairing, not less. Option 4 (total scatter at matched age) remains available and `v_total` is unaffected by any of this (0.78–0.79 in every full-frame arm).

The recommendation of [202609072200](202609072200-vg08-replication-thinning.md) is therefore refined rather than reversed: option 2 is the right lever, but as a cut with propagated uncertainty, not as a prior — a prior at any plausible strength is outweighed by the singletons.

## Caveats

- **The repeaters are three studies, not six.** Thal, Floccia and Byers-Heinlein supply every repeated child, and their slice of the pool has a smaller budget (v_total 0.55 against 0.79) and a higher young-age concentration (58 against 37). Transplanting their _share_ to the whole frame assumes the between/within proportion travels even though the total does not. That is an assumption to state, and to check against the other three studies' data where a within-child quantity can be read at all.
- **`test` tier.** The geometry is legible at this tier — the baseline reproduces the fit of record — but the numbers are not the report's, and a `rep` confirmation belongs with the refit.
- **The pinned arm is a Beta with sd 0.001, not a constant.** Close enough for the geometry; a real implementation would remove the parameter.
- **Double counting is moot in the pinned arm and present in the others.** The calibrated arms fit the repeaters twice, once for the prior and once inside the full frame; since those arms show no effect, the double counting did not manufacture one.
- **One model.** VG12 is univariate with the partition. The bivariate TD models are untested, and their two-scale structure means the cut has to be stated on the scales rather than the share.
- **Not fits of record; nothing here is publishable.**

## Reproduction

```bash
uv run python scripts/experiments/vg12_repeaters_prior.py baseline repeaters --output-dir output/experiments/vg12-repeaters-prior
uv run python scripts/experiments/vg12_repeaters_prior.py calibrated calibrated-half calibrated-fixed --output-dir output/experiments/vg12-repeaters-prior
```

Baseline 14.6 min, repeaters 2.7, each calibrated arm 13–15, run two at a time on the 16-core workstation.
