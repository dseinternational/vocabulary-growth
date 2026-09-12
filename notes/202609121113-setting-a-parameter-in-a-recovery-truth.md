# Setting a parameter in a recovery truth

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**Date:** 2026-09-12. **Question:** [#297](https://github.com/dseinternational/vocabulary-growth/issues/297) check 4 asks for VG25's `beta_sign_lag` recovered in three designed cells, and [#242](https://github.com/dseinternational/vocabulary-growth/issues/242) item 6 asks the same three of VG16; the registration note recorded that two of the three could not be run because the harness had no way to _set_ a parameter in the truth draw. **Evidence:** the free-variable structure of the correlated child block, measured against PyMC's own deterministics. **Change:** `--set-truth NAME=VALUE` on `scripts/fit_recovery.py`, and both missing cells demonstrated end to end on VG25. Nothing is fitted and no fit of record is touched.

## 1. What a truth draw could not express

Both truth sources take whatever the draw held. `--truth posterior` asks "does this model recover itself in the regime it reports in", which is the question the harness was built for; `--truth prior` asks the same over the whole prior mass. Neither can ask about a **designed** parameter setting, and that is exactly what the cross-lag gates want: the whole claim of VG25 is that the prospective lag can be told apart from the persistent correlation beside it, so the cells are `(beta = 0, rho ≠ 0)`, `(beta ≠ 0, rho = 0)` and both nonzero. The third is a draw. The other two are settings, and a coefficient that comes back looking right when its truth was zero is a different piece of evidence from one that comes back looking right at whatever the prior offered.

## 2. Free variables only, because the deterministics are recomputed

A setting names a variable the model **samples**. Every reported estimand — the trajectories, `rho_sign_q`, the child effects — is a deterministic, and the truth's deterministics are recomputed from the graph _after_ the settings are applied. That ordering is the whole design: it is what carries a setting into everything downstream of it, so the truth stays internally consistent rather than becoming a parameter vector the model could never have produced.

It also makes one mistake easy and worth refusing loudly. `rho_sign_q` is the name a reader of #297 has in mind, and setting it would be silently undone by that recomputation — the run would sample from an unmodified truth and score against a modified one, and nothing would say so. Naming a deterministic is refused, with the free variable that does carry it named in the message.

## 3. A correlation is not a number you can set

The joint and bivariate correlated blocks do not sample a correlation. `pm.LKJCholeskyCov` samples the packed lower triangle of the Cholesky factor of the child **covariance** — six numbers for VG25's 3×3 block, under the single name `subject_re` — and both the scales and the correlations are read off it: `tau_subj_u`, `tau_subj_q` and `tau_subj_sign` are its row norms, and each `rho_*` an entry of the implied correlation matrix. "No correlation" is therefore a statement about that factor's structure, not a value.

So the second kind of setting is a named transform. `subject_re=independent` replaces the drawn factor with `diag(row norms)`. Because the scales _are_ the row norms, this sets every correlation to exactly zero and leaves every scale untouched — which is what makes it the `rho = 0` cell of the same model rather than a differently scaled one. Verified against PyMC's own deterministics on a VG25 prior draw rather than asserted from the packing convention:

| quantity        |  drawn | after `subject_re=independent` |
| --------------- | -----: | -----------------------------: |
| `rho_uq`        | +0.399 |                          0.000 |
| `rho_u_sign`    | −0.620 |                          0.000 |
| `rho_sign_q`    | −0.182 |                          0.000 |
| `tau_subj_u`    | 0.3640 |                         0.3640 |
| `tau_subj_q`    | 0.4276 |                         0.4276 |
| `tau_subj_sign` | 1.2041 |                         1.2041 |

The three scales are bit-identical, not merely close. The packing itself is checked the same way: unpacking `subject_re` row-major and forming `LLᵀ` reproduces all three of PyMC's `tau_subj_*` and all three of its `rho_*` to every printed digit, so the transform rests on a measurement rather than on a reading of the upstream convention.

## 4. The setting is part of the run's identity

Two cells of one gate differ **only** in their truth. Without a marker in the name the second would simulate over the first's directory, refit into the first's fit directory and be scored into the first's matrix — one record where there are two. This is the failure `-under-<tag>` was added for on the cross-definition seam ([#226](https://github.com/dseinternational/vocabulary-growth/issues/226)), and it is handled the same way: the settings, sorted by variable name so the same cell always names the same directory, go into the recovery config name, the fit's banner and the scored label.

```
VG25-age-joint-signspeech-ds-corr-signlag-set-beta_sign_lag-0-recovery-r01
VG25-age-joint-signspeech-ds-corr-signlag-set-subject_re-independent-recovery-r01
```

`available_replicates` matches on the same stem, so a matrix cannot mix two settings of the same truth — the same reasoning as the sampling-tier filter ([#289](https://github.com/dseinternational/vocabulary-growth/issues/289) task 4.7). No settings adds no marker, so every existing recovery directory keeps the name it has; a test pins that in both directions.

One practical consequence on Windows: these names are longer, and the simulation's build directory nests a model output directory inside the recovery directory. That pushes the optional Graphviz model diagram past `MAX_PATH`, which `dot` obeys regardless of `LongPathsEnabled`. It warns and skips, and nothing else in the run is affected.

## 5. What this closes, and what it does not

Both missing cells were run end to end on VG25 at `dev` from the prior, 24 s each, sixteen coherence checks passing and the wave loop selected as usual:

- `--set-truth beta_sign_lag=0` — the truth records `beta_sign_lag` at exactly 0 with all three correlations left at their drawn values.
- `--set-truth subject_re=independent` — the truth records all three correlations at exactly 0 with `beta_sign_lag` at its drawn 0.088 and the scales unchanged.

So gate 4 is runnable, for VG25 and for VG16's first cell. **It has not been run**: that is sampling, at a tier `test` will probably not reach — two of three VG16 replicates missed the R-hat gate there, and this is the same shape of question on a smaller support.

Two things this deliberately does not do. It does not sanity-check a setting: the truth is refused if a setting leaves any reported quantity non-finite, which is where a boundary value shows up first, but whether a value is _sensible_ for a variable is the caller's judgement. And it does not supply VG16's `(beta = 0, rho ≠ 0)` cell, which needs a model carrying both a lag and a correlated child block — VG16 has no correlation to set and VG20 no lag, so that cell still wants the correlated-random-effects-plus-lag comparator #242 item 6 names. VG25 needs no such comparator: it carries both, so all three of its cells are settings of one model.
