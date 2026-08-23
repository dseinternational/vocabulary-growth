# One-off experiment harnesses

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

These are **not maintained tools**. Each was written to answer one question during
the 2026-08 reporting refit, and each produced a result that is now cited in a
note or a code comment. They are kept so those results can be reproduced or
challenged, which they could not be if the harnesses lived only in a session
scratchpad.

Treat anything here as a record of how a specific number was obtained, not as an
API. In particular several carry hard-coded paths — `/scratch/vg-geom-output` for
throwaway output roots, and the output root of the machine the run happened on —
and none is covered by the test suite.

## Why they are separate from `scripts/`

Everything in `scripts/` is a supported entry point that operates on registered
models through the normal pipeline. These instead **monkey-patch a model
definition before importing its module**, so they can fit a variant without
adding a field to a definition class — which matters, because adding a field to a
definition class invalidates every existing fit of that class (see
`fit_artifacts.validate_fit_output`). That trick is right for a throwaway arm and
wrong for anything shipped, so it is quarantined here.

Every arm writes to a **separate output root**, so none can overwrite a model of
record.

## What each one established

| Harness                             | Question                                                                                           | Result recorded in                                                                                                                                                                                             |
| ----------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `geom_arm.py`                       | Do centring the study block and the variance partition fix VG12's energy BFMI?                     | [`202608050900`](../../notes/202608050900-td-hierarchical-geometry.md) §9 — centring yes for ESS, partition **no** for BFMI                                                                                    |
| `vg12_eta_isolate.py`               | Which of three simultaneous changes caused VG12's divergences to go 2 → 29?                        | Commit `5d2714e` — the `eta` widening, isolated and reverted                                                                                                                                                   |
| `clamp_arm.py`                      | Does the mean clamp lower `eta` in the univariate models, as it did in the joint ones?             | [`202608060900`](../../notes/202608060900-three-prior-conflicts.md) §3 — partially; 14% lower, still pressing                                                                                                  |
| `vg13_ell_arm.py`                   | Is VG13's 10-month window hiding curvature its GP cannot express?                                  | [`202608060900`](../../notes/202608060900-three-prior-conflicts.md) §5                                                                                                                                         |
| `calibrate_partition.py`            | What budget and share priors reproduce the marginals the variance partition replaces?              | `definitions._TD_UNDERSTOOD_VARIANCE_PARTITION` comments                                                                                                                                                       |
| `compare_arms.py`                   | Scores a set of arms on BFMI, divergences, R-hat, ESS and the ridge correlation.                   | Used by `geom_arm.py` runs                                                                                                                                                                                     |
| `compare_marginal_arms.py`          | Scores the marginalisation arms on equivalence, geometry, and work per effective sample.           | Used by `marginal_arm.py` runs                                                                                                                                                                                 |
| `verify_geometry.py`                | Recomputes every geometry number quoted in the note directly from the traces.                      | [`202608050900`](../../notes/202608050900-td-hierarchical-geometry.md) §§2–5                                                                                                                                   |
| `early_signing_and_later_speech.py` | Does early signing predict later spoken vocabulary, beyond comprehension?                          | [`202608160930`](../../notes/202608160930-early-signing-and-later-speech.md) — the dose measure holds up, the binary signer contrast is not identifiable (signing is nearly a study-level property)            |
| `rank_stability.py`                 | Do children keep their relative vocabulary standing over time?                                     | [`202608141600`](../../notes/202608141600-rank-stability-tracking.md)                                                                                                                                          |
| `rhat_gate_calibration.py`          | Is the R-hat gate's maximum-over-parameters rule penalising large models?                          | [`202608150900`](../../notes/202608150900-rhat-gate-calibration.md) — no; the hypothesis is refuted                                                                                                            |
| `vg16_within_ridge_arm.py`          | Is the within-child cross-lag anomaly a joint-estimation ridge, or the data?                       | [`202608151500`](../../notes/202608151500-within-child-crosslag-feasibility.md) §7 — neither; it was `dev`-tier under-convergence, and the ridge biases **upward**                                             |
| `vg16_within_lag_bias.py`           | Does VG16's within-child cross-lag baseline carry the short-T bias the report attributes to it?    | [`202608151500`](../../notes/202608151500-within-child-crosslag-feasibility.md) §6 — **no**; the −0.60 does not reproduce at any truth, and an oracle-intercept variant rules out errors-in-variables          |
| `vg16_crosslag_quantification.py`   | What does VG16's cross-lag say in interpretable units, and does it belong in the models of record? | [`202608151120`](../../notes/202608151120-vg16-cross-lag-quantified.md), [`202608151140`](../../notes/202608151140-cross-lag-not-for-models-of-record.md) — reliably positive, between-child, not for adoption |
| `vg20_age_at_word_count.py`         | By what age has a given share of children reached N words? (VG20, exact for understood)            | [`202608231056`](../../notes/202608231056-age-at-word-count.md) §3 — a between-child quantity, not the population milestone `time_to_milestone.py` reports                                                     |
| `vg20_vg19_age_at_word_count.py`    | Does the child-slope structure change those ages?                                                  | [`202608231056`](../../notes/202608231056-age-at-word-count.md) §4 — the medians agree to within a month; the 75th/90th **spoken** percentiles do not, and VG20's are the optimistic ones                      |
| `marginal_arm.py`                   | Does integrating out the singleton child effects fix VG12's geometry, and at what cost?            | [`202608231745`](../../notes/202608231745-singleton-marginalisation.md) §7 — the bench is not yet run; the arms are `explicit`, `marginal`, and `marginal --nodes 40`                                          |

## Running one

Arms take an output root that is **not** the project's, so a failed or misleading
arm cannot become a model of record:

```bash
python scripts/experiments/clamp_arm.py clamped --output-dir /scratch/throwaway
```

`compare_arms.py` and `verify_geometry.py` read finished arms and print tables;
both need their path constants checked before use.
