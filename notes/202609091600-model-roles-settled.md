# Model roles settled: seven models in the default refit scope, fourteen out

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5.1).

**Date:** 2026-09-09. **Decides:** the "Model role assignments" item of [#320](https://github.com/dseinternational/vocabulary-growth/issues/320), on the study owner's instruction. **Evidence:** the 2026-09-07/08 full refit (twenty-one models at `rep`), the VG20/VG22 gate resolution ([`202609091200`](202609091200-vg20-vg22-gate-resolved.md)) and the VG22 recovery result ([`202609091400`](202609091400-is-vg22-the-better-description.md)), the reporting decision of 2026-08-22 ([`202608221200`](202608221200-reporting-source-by-quantity.md)), and the audit of the retirement evidence posted to #320 on 2026-09-08. **Record:** the roles table in [`docs/models/README.md`](../docs/models/README.md), which now names every classified model and which `tests/test_model_catalogue.py` pins against the catalogue in both directions.

## The decisions

Nine models had no declared role and so failed closed into the publication scope. All nine now have one.

| model                  | role                                 | basis                                                                                                                                                                                          |
| ---------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VG24                   | **model of record** for `rho_sign_q` | Registered for that deliverable; its refit is clean (R-hat 1.003, ESS 1,398, no divergences) and reads 0.390 [0.227, 0.539]. Nothing else estimates it. VG15 keeps the signing trajectories.   |
| VG23                   | **TD reference**                     | Sole source of the typically-developing side of the between-child correlation contrast: `rho_uq` 0.127 [0.095, 0.159] against VG20's 0.433 [0.355, 0.507].                                     |
| VG13                   | **superseded** by VG21               | Replaced as the TD joint comparator on 2026-09-02 (`TD_KEY` moved in both `compare_ds_td_*` scripts) because its support ends at about 221 understood words. Remains VG23's exact nested null. |
| VG19                   | **development step**                 | The 2026-08-22 decision: contributes a finding, not a number; a k-fold tie with VG20 at +0.93 SE; never a promotion candidate.                                                                 |
| VG22                   | **development step**                 | The gate resolved with VG20 keeping the role; its structure is supported and its magnitudes are not identified at this pool's follow-up, so it supplies no reportable number.                  |
| VG01, VG02, VG03, VG04 | **development steps**                | The single-level baselines each lineage was built on. VG20 carries VG01 and VG02's estimands; VG11 and VG12 are VG03 and VG04 rebuilt with a hierarchy.                                        |

Three models that were already classified in the catalogue but had no row in the roles table — VG15, VG11 and VG12 — and VG21, whose role dated from 2026-09-02, are now written into the table too, unchanged, so that the table is the complete record rather than a partial one that the catalogue's comments filled in.

## What the default refit now covers

| scope                | models                                                                                       | serial `rep` time on 2026-09-07/08 |
| -------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------: |
| `-Scope publication` | VG20, VG15, VG24 (models of record); VG11, VG12, VG21, VG23 (TD references)                  |                   about 12 h (39%) |
| excluded by role     | VG05, VG07, VG08, VG09, VG10, VG14, VG16 (before today); VG01–VG04, VG13, VG19, VG22 (today) |                    18 h 26 m (61%) |
| `-Scope all`         | every registered model                                                                       |                          30 h 27 m |

The seven newly excluded models cost 10 h 14 m of the cycle between them, VG03 (2 h 46 m) and VG19 (2 h 20 m) being half of it. Nothing is deregistered: every definition stays in `MODEL_REGISTRY`, so prior-simulated figures, structural exposition and lineage comparisons are unaffected, and `-Scope all` refits the whole registry whenever a lineage figure needs consistent fits.

## What changed, and where

- **`src/vocab_growth/models/catalogue.py`** declares the nine roles, each with a comment naming the record it rests on.
- **`docs/models/README.md`** gains rows for all thirteen models the roles table did not name (the nine decided today, plus VG15, VG11, VG12 and VG21). The VG13 row records that the replacement by VG21 is by decision, not a supersession under the report's two-grounds rule — no cross-window out-of-sample comparison exists — and why its 18-month ceiling was never arbitrary.
- **`tests/test_model_catalogue.py`** now pins every role in both directions: each model the table names must be declared with that role, and each declared role must have a row. Before this, only development steps were pinned, and only one way — a superseded model could have been declared with no row at all, which the 2026-09-08 audit flagged.
- **`src/vocab_growth/report_cells.py`** gains a `superseded` reading role, so VG13's page routes families and practitioners to VG21 in the same way a development step's page routes them to its model of record. The pages of VG03, VG04, VG13, VG19, VG22, VG23 and VG24 now state their roles.
- **`docs/report/methods-models.qmd`** §Models of record names VG21 as the matched-comprehension reference (it said VG13, pending a VG21 fit that has existed since 2026-09-02), VG23 for the correlation contrast, VG24 for `rho_sign_q`, and gives VG19 and VG22 their qualitative findings in place of "roles not yet assigned".
- **`scripts/compare_models.py`** retires the two single-level DS/TD overlays it drew from VG01–VG04, superseded by the random-effect overlays. The old PNGs on the comparisons root are now unclaimed by any manifest entry and will be reported as such at the next sync.
- **`docs/runbooks/full-refit.md`** says the default scope is seven models, and why.

## The cost, accepted

`catalogue.py` is under `src/vocab_growth`, so this edit changes the executable-code signature and **restales all twenty-one fits of 2026-09-07/08** for the `sync` and `publish` paths. They remain renderable and provisionally syncable. This was put to the study owner and accepted: it is consistent with the 2026-09-06 plan that code changes land before the next full refit on the VM, and the payoff is that the VM refit now runs the seven-model publication scope rather than twenty-one. The alternative — holding the catalogue edit until the refit window — was offered and declined.

## What this opens rather than closes

1. **VG15 → VG24.** All three of VG24's child correlations exclude zero (`rho_sign_q` 0.390, `rho_uq` 0.342, `rho_u_sign` 0.278), which contradicts VG15's independence assumption exactly as VG20's `rho_uq` contradicted VG10's. Whether VG24 should take the rest of VG15's role — the signing trajectories, `psi`, total expressive vocabulary — is a promotion question for a comparison note modelled on the VG10 → VG20 one: are the population trajectories unmoved, and does the correlated block recover? Not decided today.
2. **VG26** (VG21 plus `rho_uq`, [#240](https://github.com/dseinternational/vocabulary-growth/issues/240)) would retire VG21, and with it VG13 and VG23's separate existence as the correlation reference. The four checks it needs are on #240; nothing here pre-empts them.
3. **The comprehension reporting cap.** VG19 stays the instrument for the model-dependence check that returned the cap to 72 months, and that check has not been rerun on the enlarged pool. Rerunning it needs a VG19 fit consistent with VG20's, which a development step does not get by default — so the rerun is an explicit `-Models vg19,vg20` refit, not a by-product of a cycle.
4. **Two overlays and one taxonomy tension.** The retired `ds_td_*_by_age` overlays had no consumer, but if a single-level DS/TD comparison is ever wanted again it needs a `-Scope all` cycle. And the taxonomy's "superseded" is defined by a documented structural or data problem while the report's §supersedes rule requires a tabulated out-of-sample comparison as well; VG13 meets the first and cannot meet the second across windows. The row states which it is.
