# The `window-22` closure is the data's, not the prior's

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> Gate result, 2026-08-21. `window-22-vague-anchors` was fitted to test whether [202608211100](202608211100-window-22-adopted.md)'s central finding — the DS/TD gap in `q` closing by 300 understood words — was held up by the slope-anchor priors rather than by the data. **It is not.** The decision rule was fixed in advance of the fit and is quoted verbatim in §1. VG21 registration proceeds.

## 1. The pre-committed rule

Stated before the fit began, so the result could not be read to taste:

> **Closure survives** (Δq ≈ 0 at 300–320, P(TD>DS) ≈ 0.5) → the finding is the data's; the prior provenance becomes a disclosure, and VG21 registration proceeds. **Gap reopens** toward `window-25`'s +0.09 → the closure was prior-held and must not be published, and the promotion stops.

The variant restates `window-22` exactly except for the two slope-anchor priors, deliberately widened in the direction that threatens the finding: `p_slope_hi_u ~ Beta(1.2, 2.0)` and `p_slope_hi_q ~ Beta(1.3, 1.3)`. Upward displacement of the TD `q` curve is what would reopen the gap, so the prior was made permissive of exactly that.

## 2. The result

Δq = q_TD − q_DS, 89% intervals, both runs through the identical code path with the identical DS (`VG20`) trace:

| understood | `window-22` (baseline)         | P(TD>DS) | `window-22-vague-anchors`      | P(TD>DS) |
| ---------: | ------------------------------ | -------: | ------------------------------ | -------: |
|        175 | +0.0915 [+0.0662, +0.1116]     |    1.000 | +0.0914 [+0.0661, +0.1114]     |    1.000 |
|        200 | +0.0885 [+0.0436, +0.1209]     |    0.997 | +0.0885 [+0.0438, +0.1206]     |    0.998 |
|        250 | +0.0255 [−0.0490, +0.0937]     |    0.709 | +0.0252 [−0.0487, +0.0933]     |    0.710 |
|        300 | **−0.0014** [−0.0765, +0.0740] |    0.488 | **−0.0013** [−0.0770, +0.0735] |    0.489 |
|        320 | **+0.0011** [−0.0752, +0.0767] |    0.509 | **+0.0016** [−0.0767, +0.0763] |    0.513 |

Across the whole grid the largest difference in P(TD>DS) is **0.0085**, and `q_TD` medians differ by at most 2.7e-04. The levels whose 89% interval crosses zero are the same set in both — 250, 300 and 320. The closure is reproduced with the priors that were built to break it.

Two internal checks confirm the two runs really are different fits compared like with like. The **DS columns are bit-identical** (max |Δ| = 0.000e+00 on every one), as they must be when the same `VG20` trace is read twice — which rules out a stray difference in the pipeline. And the **TD columns are not** identical (max |Δ| = 9.5e-04 on the interval bounds), which rules out the opposite failure, a silent fallback to the baseline trace. The result is a genuine agreement between two fits, not an artefact of comparing one fit with itself.

## 3. Convergence, stated plainly

The gate fit clears the hard gate — R-hat 1.0088, minimum ESS 1194, **zero divergences** — and fails BFMI at 0.276–0.292 against the 0.30 threshold, so it carries `CONVERGENCE_CAVEATS.txt` and is not publishable as a clean fit.

That caveat does not discriminate here, because it is a property of the family rather than of this variant:

| fit                       | max R-hat | min ESS | divergences | BFMI range  |
| ------------------------- | --------: | ------: | ----------: | ----------- |
| `window-22`               |    1.0068 |    1205 |           0 | 0.273–0.300 |
| `window-22-vague-anchors` |    1.0088 |    1194 |           0 | 0.276–0.292 |
| `window-25`               |    1.0048 |    1101 |           0 | 0.255–0.279 |

All three sit in the same band. The vaguer priors did **not** worsen the geometry, which had been the expectation going in. BFMI below threshold degrades tail exploration, so interval bounds are less reliable than point estimates — but baseline and gate carry the same caveat in the same degree, so the _comparison between them_ is unaffected even where the absolute bounds are flagged.

## 4. Provenance, disclosed

The gate fit's manifest records `dirty: true`: it was launched at 09:27 from branch `refit/215-clamp-q-a1-and-disk-recovery` at commit `c3644af` with uncommitted changes in the tree — the registry entry that was committed later the same day as `a70daba`.

This was checked rather than waved through. `c3644af` is an ancestor of `main`. Comparing the manifest's recorded definition against what `build_variant("vg13", "window-22-vague-anchors")` produces from committed code today gives **50 fields with no substantive difference**; the only two that compare unequal are `kappa_u` and `kappa_s`, where `anchor_ages` was serialised as a JSON list against the dataclass's tuple, with every value identical. The fit ran the definition the registry now holds.

## 5. What follows

The prior provenance of the `window-22` anchors becomes a **disclosure**, not a limitation on the finding: it should be stated that the anchors were chosen before the window was extended, and that widening them in the direction that would reopen the gap does not reopen it.

VG21 registration — promoting `window-22` from sensitivity variant to registered model of record — proceeds. Until that happens §6 of [202608211100](202608211100-window-22-adopted.md) still applies: the variant is not a registered model, and nothing published rests on it.

## 6. Reproducing this

```bash
python scripts/compare_ds_td_re.py comprehension --td-joint=vg13:window-22
python scripts/compare_ds_td_re.py comprehension --td-joint=vg13:window-22-vague-anchors
```

Both write to `output/comparisons/` under the canonical filenames, so **the canonical files must be backed up and restored around them** — they were for this run, md5-verified identical afterwards, and no published artefact moved.
