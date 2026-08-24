# VG22's rep fits failed on a dead identification anchor

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!WARNING]
> Diagnostic note, 2026-08-23, written from the retained failed fits of the VG22 rank family's first `rep` attempt (run `output/replication-logs/vg22-ranks-20260823/`, code at `2412aca`). The fix described in §4 was implemented the same day; the affected fits were never publication candidates, and no published model is touched.

## 1. What happened

The VG22 rank family's first reporting-quality run failed its convergence gate on two of three ranks:

| fit                                | gate result                                                                                          | wall   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- | ------ |
| rank-2 (model-of-record candidate) | **FAIL** — 723 R-hat failures, max 1.708; min ESS 9                                                  | 45 min |
| rank-3 (sensitivity variant)       | **FAIL** — 1,030 R-hat failures, max 1.378; min ESS 11                                               | 52 min |
| rank-1 (sensitivity variant)       | still sampling at 90+ min when the fix landed; stopped and rerun from the fixing commit — see §3, §5 | —      |

The failure is not slow mixing. It is chains sampling different, mutually exclusive gauges of an under-identified loading matrix.

## 2. The evidence: mirror modes at rank 2, a rotation ridge at rank 3

Rank-2's chains split three against three into exact mirror images. Per-chain posterior means from the retained trace:

| parameter                                | chains 0/2/4 | chains 1/3/5 |
| ---------------------------------------- | -----------: | -----------: |
| `subject_factor_w_21`                    |        −1.17 |        +1.15 |
| `subject_factor_w_31`                    |        −0.63 |        +0.73 |
| `subject_factor_z[·, 2]` (example child) |        +0.29 |        −0.22 |

The products — what the likelihood sees through `b = z L'` — agree across all six chains; only the joint sign of (`w_21`, `w_31`, `z[:, 2]`) differs. Cross-chain R-hat then explodes on exactly that block: the two `w` entries at 1.708 and roughly seven hundred of the 737 × 2 `z` entries, which is the "723 parameters" in the gate report.

Rank-3 shows the continuous version. Chain 4 sits in a rotated frame — `w_31` ≈ +0.01 with `w_32` ≈ +1.42, the third effect's loading carried by column 3 instead of column 2 — while the other five chains cluster near `w_31` ≈ +1.2, `w_32` ≈ 0, with visible within-cluster drift. More parameters fail (the ridge smears three `z` columns, 1,030 failures) but the maximum R-hat is lower (1.378), which is the signature of a continuous ridge rather than two clean modes.

## 3. The mechanism: the constraint sat on the one effect with no variance

`build_child_factor` removes the rotational invariance `L → L Q` by making the leading `k × k` block of the raw loading matrix `W` lower-triangular with a HalfNormal (positive) diagonal — standard practice. The rows were taken in effect order `(b0u, b1u, b0q, b1q)`, so the anchors at rank 2 were `b0u` (comprehension level) and **`b1u` (comprehension rate)**.

`b1u` is the dead row. Every fit of this structure agrees its scale is negligible: Gate 1's ML estimate was 0.079 on repeat-measured children, VG22's dev fit collapsed it to 0.040, and the failed rank-2 fit puts `tau_subj_u_1` at 0.038 with an 89% interval of [0.003, 0.091]. With `tau[1] ≈ 0`, row 1 of `L = (tau / ‖W‖) ⊙ W` is ≈ 0 **whatever its direction**, so the likelihood is flat in (`w_10`, `w_11`) — and `w_11 > 0` is the only thing pinning the second factor's reflection. Negating column 2 everywhere it matters (`w_21`, `w_31`, `z[:, 2]`) is then a symmetry of the posterior up to the whisper of likelihood that 0.04 of scale lets through: two modes of almost equal height that NUTS cannot cross between. The residual asymmetry is visible in the failed fit — the mirror modes differ slightly in `tau_subj_u_1` (0.024 vs 0.052) and drag the correlated `kappa_u` block a little apart, which is why that block also shows elevated R-hat.

At rank 3 the same dead anchor frees a **continuous** gauge motion: rotations in the column-(2, 3) plane are constrained only through row 1 (dead) and row 2's `w_22 > 0` (a half-plane, not a point), so the loadings sit on a ridge, and chain 4's rotated frame is that ridge being explored incoherently.

Two things made this easy to miss. The dev fit "passed" because two short chains have an even chance of landing in the same mirror mode — which they did, and the smoke fit's covariance read sensibly. And the information that `b1u` was the wrong anchor was available at design time: the registration commit itself quotes the dev fit's `tau_u1` collapse to 0.040 without drawing the identification consequence.

Rank-1 is structurally exempt: with `k = 1` there is no off-leading column to reflect, and the single sign constraint sits on `b0u`, which is live. Its first-attempt fit was still sampling after 90 minutes — twice the other ranks' wall time, for reasons that may be its own (it is the Gate-1-rejected perfectly-correlated form) — when the fix was ready; since the rank-1 graph is identical under the fix and the same seed reproduces the same chains, the clean rerun carries all of its diagnostic value, and it was stopped rather than left to finish against a dirty tree.

## 4. The fix: anchor on rows that are known to be alive

The triangular constraint is a gauge choice — it decides which parameterisation of `Sigma = L L'` the sampler sees, not what `Sigma` is. The fix permutes the **anchor order** to `(b0u, b0q, b1q, b1u)`: the two levels first, then the production-ratio rate, with the comprehension rate last so it carries a diagonal at no registered rank. All three anchors are effects with demonstrated between-child variance (`tau_subj_u_0` ≈ 0.5, `tau_subj_q_0` ≈ 1.4, `tau_subj_q_1` ≈ 0.36 in the failed fit itself), and `b0q`'s residual direction after factor 1 is substantial (corr(`b0u`, `b0q`) ≈ +0.35 leaves ~94% of its direction free of the first factor), so each diagonal pins its column's sign against a quantity the likelihood actually constrains.

What the permutation preserves, by construction: `Sigma = L L'` and its positive semi-definiteness; `Sigma_ii = tau_i²` exactly, so every `tau_subj_*` keeps its meaning; the free covariance parameter counts 4 / 7 / 9 at ranks 1 / 2 / 3 (a permutation of the row widths); and **the rank-1 graph identically** — at `k = 1` old and new constructions emit the same variables with the same priors, so a passing rank-1 fit from this run remains valid under the fixed code. What changes: which `subject_factor_w_ij` entries exist and which are HalfNormal at ranks 2 and 3. Nothing downstream reads those names (checked across `tests/`, `recovery/`, `sensitivity/`, `scripts/`), and no fit of VG22 has ever been published, so the gauge change costs nothing.

A regression test now pins the anchor pattern per rank — the set of `w` entries, the HalfNormal placement, the free-parameter counts, and specifically that no diagonal sits on `b1u` for any registered rank — since no existing test constrained the structure at all.

## 5. What the refits did, and what they say about the rank

All three ranks were refitted from the fixing commit. The anchor fix works, and the family it produced settles the rank question rather differently from the way it was posed.

| rank | sampling                        | divergences | max R-hat | min ESS | min BFMI |      `rho_uq` | `tau_subj_u` | `tau_subj_q` |
| ---- | ------------------------------- | ----------: | --------: | ------: | -------: | ------------: | -----------: | -----------: |
| 1    | `rep`                           |  **failed** |     2.305 |       7 |        - |   pinned at 1 |            - |            - |
| 2    | `rep` hightune, 12k/8k, ta 0.99 |          18 |    1.0064 |   1,913 |    0.458 | 0.340 (0.055) |        0.751 |        1.172 |
| 3    | `rep`, 6k/6k, ta 0.95           |           8 |    1.0041 |     870 |    0.455 | 0.321 (0.061) |        0.740 |        1.179 |

**Rank 1 is not a member of the family.** With one factor every child effect is a multiple of a single latent, so the model asserts `rho_uq = 1` by construction — and that is what the trace shows, a point mass at 1 with zero spread. The rest follows: 838 parameters above the R-hat threshold, a maximum of 2.305, minimum ESS 7, and the loadings on the three non-anchored rows multimodal. It failed the hard gate after 4h27m and is retained under `output/failed/`. A hightune might yet drag it through that gate; it would not make the model right. Ranks 2 and 3 both put `rho_uq` near 0.33 with a standard error of 0.06, and rank 1 asserts 1 — the rank-1 form is not a harder version of the same model, it is a different and worse one.

**Rank 2 needed the soft-tier remedy; rank 3 did not.** The model of record at rank 2 came in at 178 divergences on the plain `rep` configuration and 18 after the hightune retry (tune 12,000, draws 8,000, target accept 0.99). What the retry bought was divergences and nothing else: R-hat and ESS were already passing on the plain configuration (1.0044 and 1,956) and came back marginally worse (1.0064 and 1,913) on a third again as many draws. That is the expected shape of a target-accept increase, and it leaves the published caveat mild rather than removing it. Rank 3 reached 8 divergences at plain `rep`. That ordering is worth stating plainly: **the larger factor model was the easier geometry**, which is the opposite of the usual expectation and consistent with rank 2 having to press four child effects into a two-dimensional space.

**Where the two agree, they agree closely; where they differ, it is the slopes.** The reported intercept-level quantities are stable — `rho_uq` 0.340 against 0.321, `tau_subj_u` 0.751 against 0.740, `tau_subj_q` 1.172 against 1.179, all within a standard error of each other. One child-**slope** scale is not. The spoken slope scale is 0.348 (sd 0.047) at rank 2 against 0.576 (sd 0.066) at rank 3, a gap of about 2.8 combined standard errors, and the two fits share their data, so it is a model difference rather than sampling noise. The understood slope scale moves the same way, 0.049 against 0.105, but both are small against standard errors of 0.031 and 0.047: that one is within noise and should not be quoted.

Why the spoken slope moves is an interpretation rather than a measurement. The rank does not constrain any single effect's variance — every row of the loading matrix is unit-normalised and scaled by its own `tau`, so all four scales are free at every rank — it constrains the **correlation** matrix to that rank. The reading that fits is that at rank 2 both dimensions are spent on the intercept structure and the slope effects are shrunk in the compromise. Either way the practical rule stands: a reading that leans on the intercept correlation is safe at rank 2; one that quantifies how differently children's _spoken_ trajectories steepen should use rank 3.

## 6. Consequences and the general lesson

Operationally: all three ranks refit from the fixing commit (~45–52 minutes each at `rep` for ranks 2 and 3; rank-1's first attempt suggests longer), so the whole family comes from one clean commit. The two failed fits are retained under `output/failed/` for this note's evidence and can be deleted once it is merged.

The lesson worth keeping: **an identification constraint is only as strong as the parameter it sits on.** Anchoring rotational gauge on a component is an implicit claim that the component has non-negligible scale, and that claim was already contradicted by three independent estimates of `tau_subj_u_1` before the first `rep` chain was drawn. When registering a model with a constrained-gauge block, the anchors should be chosen — and documented — against the evidence for which components are alive, and a multi-chain fit at more than dev scale is the earliest point the failure can show, because two chains agreeing is exactly what a mirror symmetry produces half the time.

## 7. Decision: the registered default moves to rank 3

Study owner, 2026-08-24, on §5. `SubjectFactorPriorParams.rank` on the canonical `VG22` is now 3; the sensitivity family becomes `rank-1` and `rank-2`. The family itself is unchanged — 1, 2 and 3 are all still fitted — and only which of them the registry treats as the default has moved.

Two things in §5 decided it, neither of which Gate 1 could see, because Gate 1 worked from the residual likelihood and both are properties of the fit:

1. **Rank 3 is the better-behaved fit.** It cleared the convergence gate on the plain `rep` configuration with 8 divergences. Rank 2 came in at 178 on the same configuration and needed a hightune (tune 12,000, draws 8,000, target accept 0.99) to reach 18, and that retry bought divergences and nothing else — R-hat and ESS came back marginally worse on a third again as many draws. Choosing the rank that needs a soft-tier remedy over the one that does not, when the two are 2.60 apart on 2 df in the gate that was supposed to separate them, is not a defensible default.
2. **The rank is not a free choice for the slope reading.** The spoken child-slope scale is 0.348 (sd 0.047) at rank 2 against 0.576 (sd 0.066) at rank 3, about 2.8 combined standard errors apart on shared data. A default that halves a reported scale relative to the alternative is making a substantive claim, and §5's reading — that at rank 2 both latent dimensions are spent on the intercept structure and the slope effects are shrunk in the compromise — says the rank-2 value is the compromised one. Rank 3 is also the rank at which the free 4x4's likelihood is reached exactly, so it is the conservative end of the pair rather than the extrapolated one.

What does **not** turn on this: the intercept-level quantities, which agree across both ranks well within a standard error (`rho_uq` 0.340 against 0.321, `tau_subj_u` 0.751 against 0.740, `tau_subj_q` 1.172 against 1.179). Anything resting on the intercept correlation reads the same at either rank. The understood slope scale moves in the same direction as the spoken one (0.049 against 0.105) but is within noise against standard errors of 0.031 and 0.047, and should not be quoted at either rank.

**This is not a promotion of VG22 to model of record for any reported estimand.** VG20 holds that role for the Down syndrome joint understood + spoken estimands and is untouched by this. What changed is which rank the VG22 registry entry means.

### What the change costs

Both stored fits are now stale, and neither can be renamed into its new role. `validate_fit_output` compares the whole recorded definition against the registered one, and the sensitivity machinery writes its suffix into both `config_name` and `banner`:

| stored fit                       | fitted rank | status after the change                          |
| -------------------------------- | ----------: | ------------------------------------------------ |
| `VG22-…-factor` (canonical)      |           2 | stale — the registered default is now rank 3     |
| `VG22-…-factor-rank-3` (variant) |           3 | orphaned — the `rank-3` variant no longer exists |

The rank-3 fit's _graph_ is what the new canonical asks for; it differs only in `config_name` and `banner`, both of which the validation compares. So **the canonical rank-3 fit and the rank-2 variant fit are both outstanding refits.** The precedent is the same one VG21 sits under: promoting a sensitivity variant to a registered identity means fitting it under that identity, not relabelling the variant's output. At the 45–52 minutes each that §6 records, this is cheap as refits go.

Riding along with the change: the canonical banner no longer hard-codes a rank number. It read "a rank-2 factor", and because the sensitivity machinery appends `[sensitivity: rank-N]` to it, the stored rank-3 fit is labelled "a rank-2 factor … [sensitivity: rank-3]". It now reads "a low-rank factor", which is true at every rank.
