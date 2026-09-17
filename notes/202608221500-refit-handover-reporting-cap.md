# August refit decisions

> [!NOTE]
> Original record by Claude Code/Opus 5. Consolidated by OpenAI Codex/GPT-6 on 2026-09-17.

This note retains decisions from the 22–26 August 2026 handover. Its work lists, resource estimates and fit-validity claims have been superseded. Use the [full-refit runbook](../docs/runbooks/full-refit.md) for a new run. The original handover remains in Git history.

## Decisions and evidence

- The comprehension reporting cap returned to 72 months. VG14 and VG15 had been fitted after the definition change but before the sign-ratio cap was implemented, so matching definitions did not guarantee correctly capped tables. Output checks exposed the discrepancy. See the [reporting decision](202608221200-reporting-source-by-quantity.md).
- The study owner kept `product_marginal` while fitting paired-only, fallback-dispersion and moment-matched alternatives before deciding on a replacement. The later [pre-refit decision](202609131600-pre-refit-sex-mask-gap-vg26-td-variants.md) retained that fallback.
- VG23 was registered as the correlated-child extension of VG13. Its initial development fit tested the pipeline but did not converge. The [role decisions](202609091600-model-roles-settled.md) record its later reporting use.
- Comprehension below recorded production was masked. The original mask used `produced`; the [September correction](202609131600-pre-refit-sex-mask-gap-vg26-td-variants.md) extended the lower bound to `max(produced, spoken)`.
- The proposed `uk_01` imputation branch was not adopted. Setting missing comprehension to `max(spoken, signed)` added no valid data under the then-used production-union check and would have forced many spoken shares to one. This was a rejected data change, not a preparation instruction.
- The VG16, VG21 and VG23 sensitivity variants were registered before the refit window. See the [variant record](202608251900-vg16-vg21-vg23-sensitivities.md) and the registry for the supported set.

## Operational lessons retained

Definition checks alone could not detect a loader change or a stale generated table. Prepared-frame hashes and output checks now cover those distinct problems. Explicit parallel model lists must cover the registry; `tests/test_runbook_model_lists.py` checks the runbook's lists.

Keep full traces when later work requires recovery scoring or plot regeneration. Prepare non-fit report figures after synchronising fitted output. Use the supported comparison-publishing script to avoid stale inputs and missing images.

The public output container must not receive traces. They contain observation-level data and identifiers. Leave `--include-traces` off for public publication; keep trace archives in the designated internal or local storage.
