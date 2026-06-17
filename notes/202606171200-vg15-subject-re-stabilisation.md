# VG15: subject random intercepts throughout + VG10 stabilisation (issue #59)

Status: 2026-06-17, build + dev/rep fits. Author: Ethan (with Claude Code).
Numbers from `output/models/VG15-age-joint-signspeech-ds/` and the matched
dev-config comparison runs described in §4. Builds on the merged VG15
(study-RE-only joint sign/speech model, PR #53).

## 1. What changed

Issue #59 asks for two things on the joint sign/speech engine
(`common_joint_modality.py`), both ported from the bivariate RE engine
(`common_bivariate_re.py`, VG07–VG10):

1. **Subject random intercepts throughout** — non-centred per-child offsets
   $\delta_{u,\text{subj}}, \delta_{q,\text{subj}}, \delta_{\text{sign},\text{subj}}$
   on the understood / speak-ratio / sign-ratio predictors, at observation level,
   so repeated within-child measurements are not treated as independent.
2. **VG10 stabilisation (Options A + D)** — (A) tighter $q$ age-anchor priors
   (`Beta(3,22)` / `Beta(20,4)`, the VG10 values) and (D) a per-draw GP anchor at
   the reference age (54 mo), applied to all three GPs, so the GP passes through
   zero there for every draw and no longer competes with the intercept/RE level.

All changes are gated by flags on `JointModelDefinition` (defaults off, so the
engine reduces exactly to the merged VG15); `VG15` turns them on. A `holdout`
column is honoured (K-fold LOSO ready); absent it, behaviour is unchanged.

## 2. Subject REs — data support

Signing is the sparsest modality, but the worry that subject REs on the sign
ratio would be unidentifiable did not materialise. Signed data carries
substantial repeated-subject structure across uk_01/02/04/05, and the sign
subject scale is **strongly data-identified**: dev posterior
`tau_subj_sign ≈ 1.40 ± 0.16`, far above the `HalfNormal(0.5)` prior mean
(≈0.40) and tight — large genuine between-child variation in signing.
`tau_subj_u ≈ 0.80`, `tau_subj_q ≈ 1.24` similarly sit above the prior. The
sign RE is **kept** (no fallback to study-RE-only needed); `use_subject_re_sign`
remains a one-line escape hatch.

Out-of-sample fit improves materially. Matched LOO/ELPD (dev, same seed, same
data; merged-VG15 baseline vs full #59):

| outcome    | baseline elpd | new elpd | Δ      |
| ---------- | ------------- | -------- | ------ |
| understood | −4005.5       | −3755.2  | +250.4 |
| spoken     | −4403.1       | −4084.5  | +318.6 |
| signed     | −1157.3       | −1101.3  | +56.0  |
| **total**  | **−9566.0**   | −8940.9  | +625.0 |

## 3. VG10 stabilisation — effect

A + D is the half of #59 aimed at clearing the VG09-style ridge (linear trend ↔
GP ↔ REs all carrying the global level) that worsens once subject REs add a
fourth level term. See `notes/202605131500-vg09-structural-options.md`.

## 4. The association psi — decomposition and the decoupling decision

The #59 brief expected `psi` (the headline within-understood sign/speech odds
ratio) to be **materially unchanged** — "a stabilisation + RE addition, not a
re-identification of `psi`". It is not, once subject REs enter the four-cell
likelihood. An additive ladder (matched dev, same seed) attributes the move:

| model (additive)                     | psi median | Δ from prev |
| ------------------------------------ | ---------- | ----------- |
| baseline (study-RE only)             | 1.78       | —           |
| + A/D stabilisation (no subject REs) | 1.76       | ≈ 0         |
| + u/q subject REs (sign-RE off)      | 2.00       | +0.24       |
| + sign subject RE (full)             | 2.81       | +0.81       |

Two findings:

- **A/D leaves `psi` untouched** (1.78 → 1.76) — the stabilisation does exactly
  what the brief wanted.
- The `psi` move is driven by the subject REs, and **dominated by the sign RE**
  (+0.81 of the +1.03 total). That is the term most thinly identified at uk*02:
  `psi` is identified from ~62 cross-tab rows (≈34 children, ~2 rows each), and
  the per-child sign offset is co-identified with `psi` from those \_same* rows.

**Decision (with Ethan): decouple `psi` from the subject REs.** The four-cell
Dirichlet-Multinomial is fed the **population+study** marginals only
(`r_obs_pop`, `q_obs_pop`); subject REs still enter all three marginal
likelihoods (so the +625 ELPD gain is retained). `psi` therefore stays a
**population-conditioned** within-understood odds ratio, comparable to the
study-RE-only VG15 (`psi ≈ 1.8`), and does not pivot on a thinly-identified
per-child sign offset. The coherent alternative (per-child marginals in the
four-cell, `psi ≈ 2.8` as a within-child odds ratio net of child signing level)
is defensible but was set aside because the headline would then lean on the
weakest-identified RE at uk_02; it remains a one-line change if wanted later.

## 5. Fitted result (rep)

`--config rep` (6 chains × 6000 draws = 36,000):

- **Diagnostics are clean and the ridge is gone.** 0 divergences / 36,000;
  max `r_hat` = 1.0045 (`slope_u`); **0 parameters with `r_hat` > 1.01**; min
  `ess_bulk` 1340, min `ess_tail` 2277. The merged study-RE-only VG15 still
  showed residual marginal `r_hat`/ESS shortfalls and 2 divergences; A + D clears
  them, which is the point of porting the VG10 stabilisation alongside the extra
  RE level.
- **Association preserved.** `psi` median 1.72, 90% HDI [1.17, 2.35],
  P(`psi` > 1) = 0.996 — comparable to merged VG15 (1.78 [1.19, 2.42]), as
  intended by the decoupling in §4.
- **Random-effect scales (logit SD).** Study: `tau_u` 0.51, `tau_q` 0.82,
  `tau_sign` 1.07. Subject: `tau_subj_u` 0.80 ± 0.04, `tau_subj_q` 1.12 ± 0.08,
  `tau_subj_sign` 1.46 ± 0.20 — all tight, `r_hat` ≈ 1.000, `ess_bulk` > 5,900.
  Between-child variation is large in every modality and largest in signing.
- **Signed level.** `intercept_sign` −1.29 ± 0.45 (data-informed; sd well below
  the 0.75 prior — the #54 freeing fix is preserved).
- **Total expressive `p_any`** sits just below the independence bound (modest
  positive `psi`): e.g. at 48 mo ≈ 167 vs 174 words (independence), at 90 mo
  ≈ 440 vs 443. See `posterior_summary_p_any.csv`.

## 6. Reconciliation with the merged VG15

This modifies VG15 in place (the issue's intent; VG15 is the engine's only
consumer). The prior study-RE-only VG15 is preserved in git history (PR #53) and
its rendered report. `psi`, the four-cell composition, and `p_any` are intended
to be substantively comparable to that model; the subject REs sharpen the
marginal trajectories and the dispersion, not the association.
