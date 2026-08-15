# Reporting refit, 13–14 August 2026: working record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **In progress.** The run of [#215](https://github.com/dseinternational/vocabulary-growth/issues/215) is not finished, and this is a working record rather than its completion record. It exists now because two infrastructure failures and one model decision happened inside it that should not have to be reconstructed from logs later. §6 is the honest state of what is and is not done.

## 1. What this run was for

A full reporting-quality (`rep`) refit of VG01–VG16, recorded and published. Two things changed under it that were not planned when it started: the Down syndrome joint family adopted `CLAMP_Q_ONLY`, and a project-wide reporting age policy landed. Both required refits of models already fitted, which is why the DS pool appears twice in the timeline.

## 2. Model decisions taken during the run

**The 84-month clamp was applied to the wrong things** ([202608141200](202608141200-clamp-q-only.md)). `clamp_mean_above_hi_anchor` levelled _both_ the understood mean and `q` above the high anchor; because spoken is `p_U * q`, the spoken trajectory inherited both and carried a visible corner at 84 months — the sharpest feature of its whole trajectory. Measurement settled it: extrapolating VG10's own fitted anchors past the clamp gives `q = 0.996` at 115 months with `P(mean > 0.99) = 0.999`, while understood reaches 0.962 and never crosses 0.99 in any draw. Only `q` ever needed clamping. `CLAMP_Q_ONLY` is now the setting on all eight DS joint models, and the old behaviour survives as the `clamp-both` variant so the decision keeps a check that can still fail. The variant proved **robust** — 395 of 395 reported quantities within the baseline 89% interval — which establishes the corner as a presentation defect, not an inferential one.

**Reporting ages are now a policy, not a per-figure decision.** `src/vocab_growth/reporting_ages.py` caps comprehension and anything expressed as a ratio of it at 84 months, and spoken at 90 (derived from `max(ages_query)` rather than a new definition field, which would have invalidated every fingerprint). An output-based test walks the fitted artefacts and fails on any that reports past its cap. It found two classes a code audit had missed: `pmf`/`cdf` carry age in _column names_ (`pmf_90m`), invisible to an age-column scan; and `regenerate_plots.py` re-runs the plot stage only, so summary-stage tables stay stale until a refit — recorded as `KNOWN_STALE`, with a guard that fails when an entry stops being needed. That guard fired within the hour of VG15's refit, which is what it is for.

**Proposal A1 is registered on VG10 and is not a candidate model of record** (§4).

## 3. Two infrastructure failures

### 3.1 Kernel OOM, 13 August 22:52

VG13 reached 232 GB on a 251 GB box with **zero swap**, and systemd tore down the whole shared scope — taking a seven-hour fit and three unrelated sensitivity fits with it. Precautions are in [`docs/runbooks/full-refit.md`](../docs/runbooks/full-refit.md) §"Surviving an OOM": provision swap first, give each phase its own capped scope, log per-process RSS. VG13 completed on the second attempt in 337 minutes, and **peaked at 243 GB** — above the figure that had killed it — surviving only because the swap was there.

That also disproved the sizing rule the runbook carried. "Budget ~2× the plateau" was refuted by the same fit plateauing at 120 GB on one run and 178 GB on another; the rule is now stated as headroom-and-reversibility instead.

### 3.2 Disk exhaustion, 14 August 16:12

The output volume reached 100% and five in-flight refits died on `[Errno 28] No space left on device` — VG16 and VG14 part-way through writing `trace.nc`, VG10, VG07 and VG05 before they started. Nothing warned first.

**The cause was the trace tier, not the disk size.** Every fit that run wrote at the default `full` when `compact` gives byte-identical reporting output. `scripts/compact_traces.py` (new) applies the tier to traces already written, reusing the fit pipeline's own policy code so what it drops is exactly what a compacted fit would never have written. Applied to 21 fits:

|            |    before |         after |
| ---------- | --------: | ------------: |
| 21 traces  |    229 GB |     **77 GB** |
| VG13 alone | 54.19 GiB | **15.06 GiB** |
| volume     | 100% full |       **63%** |

VG10, VG12 and VG15's models of record were left at `full`, because recovery scoring — and `regenerate_plots.py`, and `loso_compare.py` — refuses a compacted trace, and those three are `fit_recovery.py`'s headline set.

Two guards in that script earned their place immediately. The staging check refuses to rewrite a trace whose fit is mid-promotion; a blanket "staging is non-empty" test blocked everything, because the crashes left eight stale staging directories behind **and one belonged to the live VG11 fit** — it now reads the PID embedded in the staging name and checks `/proc`. And processing is smallest-first, because each rewrite needs room for its output beside the original: alphabetically, VG13's 54 GB would have been attempted with 11 GB free.

**Provisioning.** Check `lsblk` before assuming a box is at its limit — this VM had **two unmounted 440 GB NVMe devices** alongside the one in use. Longer term the output root belongs on a managed disk rather than the Azure temp disk: not for speed (the local NVMe measured 472 MB/s sequential, and each trace is written once, so storage is nowhere near the critical path) but because the temp disk is wiped on deallocation — which forces a 32-vCPU/251 GB VM to stay running through the idle stretches of a multi-day run — and because a managed disk grows online, making an event like this an expansion rather than five lost fits. Do not put `trace.nc` on blobfuse or Azure Files; netCDF here is HDF5, whose metadata I/O and POSIX assumptions make network filesystems a corruption risk.

## 4. What the run learned about the models

Three analyses ran alongside the fitting, in the order the questions arose.

**Children keep their standing, and the DS/TD contrast reverses under age matching** ([202608141600](202608141600-rank-stability-tracking.md)). Tracking ICC is 0.806 on DS spoken and 0.786 on comprehension; half of children stay in the same quartile between first and last observation and 86–87% within one. The apparent DS advantage over TD does **not** survive age matching, and young DS spoken tracking collapses because of a floor effect, not instability — the reliability bound there is 0.571 and the variance decomposition is outright incoherent (the measurement bound exceeds the entire within-child variance). The practical reading: **before about 30 months, an expressive-vocabulary count is a poor basis for judging a child's relative position; a comprehension measure is far better.**

**The between-child spread does widen with age**, and differently by modality: production spreads early (1.18 → 1.64 between 18–30 and 30–42 months) then plateaus; comprehension is flat to 42 months then rises to 2.06. The models of record can represent none of this, so `kappa` has been carrying it.

**Proposal A1, registered and fitted.** Given the chance to put age variation on the between-child scale, the data take it decisively:

|                       | understood               | production ratio `q`     |
| --------------------- | ------------------------ | ------------------------ |
| `tau_subj` 24 → 48 mo | 0.716 → 0.829            | 0.969 → 1.418            |
| ratio                 | **1.159** [1.084, 1.241] | **1.462** [1.347, 1.590] |
| `P(log ratio > 0)`    | 1.000                    | 1.000                    |

Against the model of record over the same anchors, where `kappa` carries all of it: `kappa_u` falls 2.64× and `kappa_s` 1.66×. A1's flat `kappa_u` of 38.0 sits at the geometric mean of the record's 65.4 and 24.7 (40.2) — **it replaces the decline with its average and re-expresses the age variation as widening.** Whether it fits _better_ is untested and needs LOO against a record fitted under the same clamp; and the widening it reports is an upper bound, because A1 cannot represent children crossing and drift has nowhere else to go.

**The relaxation was chosen by likelihood, not by argument** ([202608141900](202608141900-child-slope-implementation-plan.md) §2). Fitting three candidate within-child structures to the adjusted residuals rejects the AR(1) outright — persistence collapses to zero on both outcomes — and selects a random slope, which survives restriction to the 334 children with repeated production measures. As a by-product, A1's no-crossing assumption (the same model with the intercept–slope correlation pinned to 1) costs 6.28 on 1 df where there is power to test it. **That gate was run before the implementation plan was written and overturned a recommendation this project had recorded an hour earlier**; §10.6 of the tracking note states what was wrong with it.

## 5. Convergence

VG12 clears the hard tier (R-hat, ESS) but carries the documented soft-tier caveats — 4 divergent transitions and energy BFMI 0.215 — and VG13 the same on BFMI (0.242). That is the intrinsic geometry of the TD hierarchical models, established in [202608050900](202608050900-td-hierarchical-geometry.md) §9, and it needs `--allow-caveats` and `convergence_caveats.csv` carried into Appendix B, not a refit. A1's own fit is the cleanest of the day: 0 divergences, max R-hat 1.0028, minimum ESS 2,278. VG10's refit carries 1 divergence.

### 5a. VG11 failed the hard gate, and is published under a recorded exception

VG11 sampled for 4h 48m and then **failed the convergence gate**: one parameter of the whole model, `g_unit_hsgp_coeffs[4]`, at R-hat **1.0125** against the 1.01 threshold. No ESS failures. The gate stopped the pipeline at diagnostics, so no summaries, plots or report were produced — but the trace _was_ retained, complete, in `output/failed/`.

Inspecting it settles what kind of failure this is:

| quantity                         |  max R-hat | min ESS | grid points above 1.01 |
| -------------------------------- | ---------: | ------: | ---------------------- |
| `f_plot` / `p_plot` (trajectory) | **1.0032** |   3,407 | **0 of 500**           |
| `f_query` / `p_query`            | **1.0028** |   3,409 | **0 of 8**             |
| `kappa_plot` / `kappa_query`     | **1.0019** |   4,364 | **0 of 500**           |
| `g_unit_hsgp_coeffs[4]`          |     1.0125 |   1,139 | —                      |

**Every reported quantity converges with margin.** The failing parameter is one of sixteen HSGP basis coefficients, and it is the slowest-mixing one — lowest ESS of the sixteen, largest signal (mean 1.428). The coefficients near zero have ESS of 13,000–40,000 because they are prior-dominated. No chain is stuck and there is no multimodality: per-chain SDs are 0.76–0.84 and per-chain means span 0.19. Individual basis coefficients trade off against one another and are weakly identified; **the function they sum to is not**, which is exactly what the table shows. The basis coefficients are not reported — the GP reaches the report through `eta` (R-hat 1.006) and `ell` (1.003).

The sampler is otherwise healthier than two models already being published with caveats: 16 divergences in 48,000 draws (0.033%, spread 5/1/4/1/4/1 across chains) and BFMI 0.359–0.395 on every chain, against VG12's 0.215 and VG13's 0.242.

**Decision, 2026-08-15 (study owner): publish under a documented exception, provisionally, pending a longer refit.** The mechanism is deliberately narrow. `ConvergenceException` in [`fit_artifacts.py`](../src/vocab_growth/fit_artifacts.py) names the exact parameters and a ceiling R-hat; the gate closes again on an additional or different failing parameter, a worse R-hat, any ESS failure, or an incomplete scan, and it never applies to another model. It is recorded into `diagnostics_summary.json` rather than a marker file, because `convergence_caveats` recomputes from the payload on disk — so the exception reaches `check_fit`, the figure sync and Appendix B by the same route as a divergence, and **VG11 is publishable only through the `-with-caveats` purposes**. `tests/test_convergence_exception.py` pins all five ways it must refuse to widen.

The artefacts were produced by [`scripts/resume_from_trace.py`](../scripts/resume_from_trace.py), which re-runs the pipeline with the sampling stage replaced by a loader for the retained trace, rather than re-sampling a posterior already on disk. It verifies the definition, the raw-data fingerprint, the sampling configuration and the trace's variables and dimensions against the rebuilt model before proceeding, and records `artefacts.resumed_from` in the manifest so a reader can tell the trace predates the summaries. It cannot bypass the gate: the gate runs as normal and closes unless an exception covers the failure.

**What would retire the exception:** `g_unit_hsgp_coeffs[4]` needs roughly double its 1,139 ESS to land under 1.01, so a refit at about twice the draws. More _chains_ would buy the same ESS at the same wall-clock, but twelve will not fit in 251 GB — §3.1's peak was 246 GB with six.

### 5b. `vg11 / anchor-broad` fails the gate, and is accepted as an informative negative

The #147 Target 8 variant `vg11 / anchor-broad` widens both `q`-anchor priors — `p_slope_low` from `Beta(1, 30)` to `Beta(1, 15)` and `p_slope_hi` from `Beta(1.3, 1.3)` to `Beta(1.5, 1.1)`. At `rep` on 2026-08-15 it **failed the convergence gate**:

|                |         model of record |               `anchor-broad` |
| -------------- | ----------------------: | ---------------------------: |
| divergences    |                      16 |                       **60** |
| max R-hat      |                  1.0125 |                   **1.0136** |
| min ESS        |                   1,139 |                          853 |
| R-hat failures | `g_unit_hsgp_coeffs[4]` | **`ell_unit`, `eta`, `ell`** |

**The failure is interpretable, and it is the answer the variant was asking for.** The failing parameters are the GP length-scale and amplitude, and the divergence count nearly quadruples. That is the TD hierarchical geometry of [202608050900](202608050900-td-hierarchical-geometry.md) — the ridge between the GP and the explicit logit-linear trend — made worse. The anchor priors are what hold the trajectory against that trend, so loosening them loosens the constraint that keeps the ridge tractable. Note the contrast with the model of record's own failure, which is a single HSGP basis coefficient (a nuisance direction, §5a); here it is the GP's own scale parameters.

**Decision, 2026-08-15: accepted as an informative negative, not refitted at `rep-hightune`.** A robustness variant that cannot be sampled is not evidence that the model of record is fragile — it is evidence that this prior widening is not a viable specification for this model. Refitting at a higher tuning budget could well make it sample, but [202608020829](202608020829-kappa-and-eta-q-prior-recalibration.md) §16 already records that more tuning **masks** this geometry rather than removing it, so a `hightune` pass would buy a number whose meaning is worse, not better. The robustness matrix should therefore carry it as a **non-converged variant with its reason**, and never as a blank.

The retained trace was compacted from 43.6 to 4.5 GiB after the fit, to keep disk headroom for the two variants behind it (§3.2 is why that mattered). Every free parameter and all of `sample_stats` survive, so the post-mortem above is unaffected — but note that failed fits are pinned to `full` by design (`common.py`, "a post-mortem should not have to work around a storage policy"), and `scripts/compact_traces.py` scans only `output/models/`, so the project's own ENOSPC recovery path cannot currently reach the largest traces it keeps.

## 6. State at the time of writing

**Complete:** VG01–VG05, VG07 (13 Aug); VG03, VG04, VG13, VG12 (TD); VG08, VG09, VG15 under `CLAMP_Q_ONLY`; the `us01-implausible-reinstated`, `dse-native-only`, `tau-psi-*` and `psi-drop-*` sensitivity variants; VG10's `a1-tau-age-varying`.

**Running:** VG11 (the last TD model); VG16, VG14 refitting under `CLAMP_Q_ONLY`, with VG10, VG07, VG05 queued behind them and A1's recovery after that.

**Not started:** the three remaining #147 Target 8 anchor variants; the comparison suite; `sync_report_figures.py` and `prepare_report_figures.py`; both book renders; the upload. The item-difficulty note's fit-derived figures (§3.3 kernel-share, §4 dispersion) still need re-pinning against VG10's refit, and parameter recovery needs re-baselining.

**Known debt.** VG14 carries `KNOWN_STALE` reporting-age entries until its refit lands. VG10's fit on disk predates `CLAMP_Q_ONLY` and fails `check_fit --purpose publish` on definition mismatch — which is correct, and clears when its refit completes.
