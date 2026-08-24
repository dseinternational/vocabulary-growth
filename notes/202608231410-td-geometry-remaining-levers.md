# Remaining geometry levers for the TD hierarchical models, and the adopted order

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

> [!IMPORTANT]
> Decision note, 2026-08-23. Prompted by the study owner's question — is there anything further, statistically or structurally, that improves the typically-developing hierarchical models' geometry or reduces their complexity while preserving the reported results? — and by the previous day's finding that the reporting-tier minimums leave no sanctioned room to cut chains or draws below `rep-lite`. It builds directly on [202608050900](202608050900-td-hierarchical-geometry.md), which measured the cause and falsified the coordinate-change remedies, and it records the study owner's adoption decision (§6).

## 1. Where the previous note left the problem

[202608050900](202608050900-td-hierarchical-geometry.md) established, by measurement: the energy BFMI failure in VG12/VG13 is the `tau_subject`/`kappa` competition created by singleton children (§2–§4); centring and the variance partition move ESS and divergences but not BFMI, because the ridge rotates rather than dissolving (§9); removing the subject REs would fix the diagnostics while corrupting `kappa`, the reported DS/TD contrast, by ~6× in VIF terms (§7, the single-admin check); and subsampling the pool would remove the replication that is the binding constraint (§10). Its conclusion was that "the only remedy is more repeat measurement, which no reparameterisation, prior, or tuning budget can substitute for".

That conclusion is right about **identification** but overreaches on **sampling**. Those are different problems: the width of `share`'s posterior is fixed by the data, but how well the sampler traverses that (correctly wide) posterior is not. The levers below all share one admission criterion, which is what separates them from everything §7 rejected: **each preserves the reported posterior exactly**. №2 leaves the target untouched and changes only the sampler's coordinates; №3 is exact marginalisation, so the joint model and every retained quantity's posterior are unchanged; №4-adjacent №1 does not touch the model at all. Nothing here re-litigates the rejected levers (§5 below).

The numbering follows the conversation in which the decision was taken.

## 2. Lever №1 — stop storing the observation-sized deterministics (adopted, first)

This is [202608050900](202608050900-td-hierarchical-geometry.md) §11 open item 3, now scheduled. §10 measured the arithmetic: fit memory is dominated by observation-sized deterministics stored per draw, scaling as `n_obs × draws` — VG12 at `rep` implies ~65 GB, VG11 at the 48,000-draw hightune configuration was killed at 247 GB, and the 2026-08-22 batch measured VG13's `rep` fit plateauing near 178 GB. The compact-persistence work ([202608081445](202608081445-trace-persistence-tiers.md)) measured the same mass from the disk side: dropping these variables took VG10's stored trace from 9.8 GB to 3.2 GB with byte-identical reporting — roughly two thirds of stored bytes — but only at write time, after sampling, which is why the persistence tier is a disk lever and not a RAM lever.

The change: do not create the observation-sized quantities (`f_obs`, `p_obs`, `kappa_obs`, `z_obs` and their bivariate `*_u_obs`/`*_s_obs` counterparts, and the concatenated `*_all` grids) as `pm.Deterministic`s in the sampled graph at all. The grid-sized `*_plot` and `*_query` quantities stay — they are what the comparison suite and the experiment harnesses read, and they are small. Everything removed is a recomputable function of the free parameters; the persistence-tier work already catalogued exactly which, and proved reporting equivalence for the write-time version of the same cut.

Three properties make this the first move. It changes nothing statistical: the free-RV graph and the seed path are untouched, so the posterior draws are identical. It changes no definition, so existing fits stay valid (readers must tolerate both worlds: old traces carry the variables, new traces do not, and consumers recompute when a name is absent). And it removes the ceiling that has been driving configuration compromises ever since §10 — including the pressure, discussed 2026-08-22, to cut TD chains or draws below the tier minimums. VG21 at `rep` moves from VM-only towards workstation-plausible.

Obligations: enumerate the in-fit consumers (summary, plots, posterior predictive, LOO all run during the fit against the in-memory trace) and give each a recompute path; re-prove reporting equivalence end-to-end on a `dev` fit and one `rep`-class fit, since the existing byte-identical proof covers the write-time cut only; and note the tier interaction — `compact`'s obs-deterministic clause becomes vacuous for new fits, while `minimal`'s `log_likelihood`/`posterior_predictive` trade stays real, so the tier vocabulary survives.

## 3. Lever №3 — marginalise the singleton child effects (adopted, second)

A singleton child's `delta` enters exactly one likelihood term, so it can be integrated out exactly: the marginal likelihood of that row is a one-dimensional integral of the Beta-Binomial over the logit-normal child effect, evaluated by Gauss–Hermite quadrature (~20 nodes, summed in log space), vectorised over singleton rows. Explicit `delta`s remain only for repeat-measured children, whose rows are coupled and genuinely need a shared latent.

**Why this is not §7's rejected removal.** The marginal likelihood still contains `tau_subject` — the mixing is integrated, not deleted. Same joint model, same posterior for every retained quantity (up to quadrature error, which the acceptance test bounds), `kappa` keeps its meaning as within-child dispersion, and the DS/TD heterogeneity contrast is untouched. What changes is the sampled space:

| model | free dimensions now (approx.) | after marginalisation (approx.) |
| ----- | ----------------------------: | ------------------------------: |
| VG12  |         ~5.9k (5,819 `delta`) |              ~1.1k (1,000 kept) |
| VG11  |      ~14.6k (~14,585 `delta`) |              ~2.0k (1,947 kept) |
| VG13  |   ~10.0k (4,989 × 2 outcomes) |                 ~1.6k (770 × 2) |

**Why it should move BFMI.** The single-admin arm in [202608050900](202608050900-td-hierarchical-geometry.md) §7 took BFMI from 0.209 to 0.981 by removing the singleton competition — but did it by changing the model, which is why it was rejected. Marginalisation produces the same sampled-space geometry by algebra instead: the thousands of prior-dominated singleton dimensions whose conditional scale tracks `tau_subject` — the funnel mass — leave the sampled space, while the model stays the model. This is a prediction, not a guarantee: the `(tau_subject, kappa)` ridge survives in the marginal posterior and could still track energy. The VG12 `test`-config bench (§9's four-arm table) is the arbiter, as a fifth arm.

**Expected side benefit.** PSIS-LOO's Pareto-k pathology with per-child effects sits exactly on singleton rows, where the conditional predictive overfits its own observation; marginalised rows carry the marginal predictive instead, which is what leave-one-subject-out means for a singleton. The corresponding caveat: the pointwise `log_likelihood` semantics change (marginal for singletons, conditional for repeats), so elpd values are not comparable with pre-change fits and `loso_compare` must not mix the two.

Obligations: a `CustomDist` likelihood per outcome for the singleton block; an equivalence test against the explicit model at `test` config — exactness is the design property, so `tau_subject`, `kappa` and the trajectory posteriors must match within Monte Carlo error, and this test has teeth; quadrature-node sensitivity (double the nodes, require no drift); parameter recovery; and a definition flag on a subclass, per the serialisation trap in [202608050900](202608050900-td-hierarchical-geometry.md) §7 — adding any field to a shared definition class invalidates every existing fit of that class.

## 4. Lever №2 — nutpie's learned transform adaptation (deferred)

§9's falsification — "no change of coordinates can help" — covers the class that was tried: fixed, handwritten reparameterisations, each of which relocated the heavy energy direction into a new name (`share` inherited the whole correlation). A learned nonlinear transport map is the one sampler-side class outside that verdict: nutpie (0.16.11, already in the lock) ships experimental normalizing-flow adaptation (`transform_adapt`, jax backend, `flowjax` dependency) which learns the map during warmup; in the perfect-flow limit the sampled space is unit Gaussian and BFMI approaches 1 **with the posterior untouched**. Unlike hiding the caveat, this would resolve it honestly: `share` stays weakly identified and its interval stays wide, but the tails that §8 flagged as the least trustworthy part of the TD fits would actually be explored, so the wide interval would be correctly measured.

Deferred, by the study owner's decision, until №1 and №3 land: it is experimental; `flowjax` is not in the locked environment (it would be an opt-in overlay, like GPU); any new sampler setting must enter the manifest vocabulary before a `rep` fit can carry it, since `_sampling_parameter_errors` currently knows draws, tune, chains, `target_accept` and seed; and flow training on VG12's ~6k dimensions is expensive — №3 shrinks the space to ~1–2k first, which makes this cheaper if it is still needed at all.

## 5. What was checked, what it bought, and what stays settled

**Window-22 replication, measured.** Counted against the Wordbank export, approximating the bivariate pool filters (WG + Oxford CDI, English + Romance languages, monolingual, typically developing, dataset floor at 200 in-window rows; no defect-class masking, so the figures are approximate):

| bivariate TD pool |  rows | repeat-measured children | repeat rows |
| ----------------- | ----: | -----------------------: | ----------: |
| 8–18 (VG13)       | 5,789 |              770 (15.4%) |       1,570 |
| 8–22 (VG21)       | 6,217 |              885 (17.0%) |       1,902 |

192 of the 428 rows added above 18 months are new replication on already-seen children (mostly Floccia Oxford CDI). The direction is right — [202608050900](202608050900-td-hierarchical-geometry.md) §9 found the operative quantity is the absolute amount of within-child replication — but the magnitude is modest: VG11 clears BFMI with ~1,947 repeat-measured children, VG12 fails with ~1,000, and 885 is still on the failing side of that line. **Expectation, set before the first fit: VG21 at `rep` should carry a slightly milder BFMI caveat than VG13, not a clean pass.** A caveated first VG21 fit is the predicted outcome, not a regression.

**The admission floor hides nothing.** The `min_study_observations=200` floor drops 79 rows containing exactly one repeat-measured child, so no longitudinal data sits under it and lowering it is not a lever. The real data lever remains what the previous note said: new repeat administrations.

**Riding along.** Fixing `eta_q` in VG13/VG21 rather than estimating it ([202608050900](202608050900-td-hierarchical-geometry.md) §11 open item 5; contraction −0.003 means it is reported straight back from the prior) is an honest one-dimension reduction with the VG15 `ell_unit_sign` fix as in-code precedent; register it as a sensitivity variant when convenient.

**Settled negatives, unchanged by this note:** no subject-RE removal and no repeat-only REs (§7, decisive), no TD subsampling (§10), no chains or draws below the reporting-tier minimums — `rep-lite` is the sanctioned reduction and is classified as reporting quality.

## 6. Decision

Study owner, 2026-08-23: **implement №1 and №3 before considering №2.** Sequencing within that: №1 first — it is pure engineering, statistically inert, and removes the memory ceiling that currently constrains every experiment on this hardware — then №3 with its equivalence, quadrature and recovery obligations, benched as a fifth arm on the VG12 `test`-config table. №2 is revisited only if the BFMI caveat still stands after №3, at which point it will also be substantially cheaper to try.
