# Reporting refit, 13–16 August 2026: run record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

> [!IMPORTANT]
> **The fitting and publication phases are complete; §6 records what is done, what is outstanding, and one class of defect the run itself introduced.** This began as a working record written mid-run — because two infrastructure failures and one model decision happened inside it that should not have to be reconstructed from logs later — and was closed out on 2026-08-16. §7 is the review that closed it, including seven sensitivity variants that had to be refitted because their baselines moved underneath them.

## 1. What this run was for

A full reporting-quality (`rep`) refit of VG01–VG16, recorded and published. Two things changed under it that were not planned when it started: the Down syndrome joint family adopted `CLAMP_Q_ONLY`, and a project-wide reporting age policy landed. Both required refits of models already fitted, which is why the DS pool appears twice in the timeline.

## 2. Model decisions taken during the run

**The 84-month clamp was applied to the wrong things** ([202608141200](202608141200-clamp-q-only.md)). `clamp_mean_above_hi_anchor` levelled _both_ the understood mean and `q` above the high anchor; because spoken is `p_U * q`, the spoken trajectory inherited both and carried a visible corner at 84 months — the sharpest feature of its whole trajectory. Measurement settled it: extrapolating VG10's own fitted anchors past the clamp gives `q = 0.996` at 115 months with `P(mean > 0.99) = 0.999`, while understood reaches 0.962 and never crosses 0.99 in any draw. Only `q` ever needed clamping. `CLAMP_Q_ONLY` is now the setting on all eight DS joint models, and the old behaviour survives as the `clamp-both` variant so the decision keeps a check that can still fail.

That check has now been run **in the direction that counts**. The robustness result quoted here originally came from the pre-adoption pairing — old baseline (clamping both) against a `clamp-q-only` variant — which stopped being the live comparison the moment `clamp-q-only` became the model of record, and left the registered inverse unfitted for two days (§7.1). `clamp-both` was fitted at `rep` on 2026-08-16 (0 divergences, max R-hat 1.0052, min ESS 1,491) and compared against the current baseline: **robust, 395 of 395 reported quantities within the baseline's 89% interval**, largest absolute difference 14.4 words. The corner is a presentation defect, not an inferential one, and that now rests on a comparison whose baseline is the model actually being published.

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

### 5c. `vg11 / eta-narrow` passes the gate the model of record fails, and reports the same trajectory

The second Target 8 variant narrows the GP amplitude prior alone — `eta_sigma` from 0.5 to 0.4 — and changes nothing else. At `rep` on 2026-08-15 it **passed the hard gate**:

|                            |    model of record |        `eta-narrow` |
| -------------------------- | -----------------: | ------------------: |
| divergences                |                 16 |               **3** |
| max R-hat (all parameters) | **1.0125 — fails** | **1.0075 — passes** |
| min ESS                    |              1,139 |                 863 |
| fitted `eta`               |              1.027 |               0.901 |

**And the estimates do not move.** Across all eight reporting ages the expected spoken count agrees to within **0.22%**, and every point lies inside the model of record's own 89% interval. The largest absolute difference is 0.44 words at 30 months, on a count of 430.

**This confirms §5a's diagnosis and qualifies [202608150900](202608150900-rhat-gate-calibration.md) §6.** That note concluded VG11 is "a correctly-specified model whose dominant GP direction mixes slowly", and that "the only honest remedy is more draws". The diagnosis is now corroborated from a second direction: narrowing precisely the GP amplitude prior is what clears the failure, which is what one would predict if that direction were the culprit. **The remedy claim is too strong as written.** It holds within a fixed specification; it is not true that no other remedy exists. A 20% prior narrowing resolves the failure at no cost to the reported estimates, and is cheaper than the double-draw refit §5a anticipates.

**This is a finding, not yet a decision.** Two things must be checked before `eta_sigma = 0.4` could be adopted on the model of record. It is a **model-definition change**, so it invalidates VG11's fingerprint and forces the TD comparison figures to be regenerated. And "the same trajectory" has been verified here only on the reported spoken grid — VG11 also supplies the typically-developing arm of the DS-versus-TD between-child spread contrast in [`comparison.py`](../src/vocab_growth/comparison.py), and that quantity depends on `tau_subject`, which is unchanged in the posterior (1.039 in both) but has not been checked through the contrast itself. The decision belongs with the exception in §5a and with [#190](https://github.com/dseinternational/vocabulary-growth/issues/190), not inside this run.

Taken with §5b, the two Target 8 anchor variants that have reported so far tell a consistent story about where VG11's difficulty lives: **widening the anchor priors makes the GP/linear ridge worse (60 divergences, the GP scale parameters failing), and narrowing the GP amplitude makes it better (3 divergences, a clean gate).** Both act on the same geometry from opposite directions.

## 6. State

**All fifteen models of record are fitted at `rep`** and published. Fourteen clear the hard convergence gate; VG11 is published under the §5a exception. VG12 and VG13 carry the documented soft-tier caveats, and VG10 one divergence — all four disclosures reach the report through `convergence_caveats.csv` and Appendix B.

**Downstream, complete:** the comparison suite; `sync_report_figures.py` (2,124 files, `--allow-caveats`); both book renders; the public upload of the fifteen model reports and the comparison book; the durable archive of the whole output root, traces included, to the private container. The item-difficulty note's fit-derived figures were re-pinned against VG10's refit, moving the §4 dispersion ratio from 0.52 to 0.805.

**Publication boundary, stated once because it is easy to get wrong.** `trace.nc` carries per-observation ages, subject codes, study codes and counts — individual-level data. `upload.py` excludes traces by default and that default is load-bearing: the public container holds reports and aggregate tables only, and the traces went to the **private** archive.

**Outstanding:** parameter recovery still needs re-baselining against the refitted headline set (VG10, VG12 and VG15 are at `full`, so nothing blocks it). The `eta_sigma = 0.4` adoption decision for VG11 (§5c) is deliberately held out of this run. Twenty of VG15's twenty-seven registered sensitivity variants, and ten of VG10's fourteen, have never been fitted at all — visible as `not-fitted` rows in the robustness matrices since 2026-08-16, rather than as absence.

**Known debt cleared during the run.** VG14's `KNOWN_STALE` reporting-age entries cleared when its refit landed, and the guard that fails when an entry stops being needed fired within the hour — which is what it is for. VG10's pre-`CLAMP_Q_ONLY` fit no longer fails `check_fit --purpose publish`.

## 7. The closing review, and the defect it found

A full review of the branch and this record on 2026-08-16 confirmed every data-derived and convergence figure quoted here against the artefacts, and found one substantive class of defect plus a set of stale documents.

### 7.1 Seven sensitivity variants were compared against a baseline that had moved

`CLAMP_Q_ONLY` and the 72 → 84 comprehension cap were adopted **during** the run, so the models of record were refitted mid-flight. Seven registered variants had been fitted before that and were never refitted:

| variant                              | fitted       | its baseline refitted |
| ------------------------------------ | ------------ | --------------------- |
| `vg10 / us01-implausible-reinstated` | 14 Aug 09:31 | 14 Aug 16:58          |
| `vg15 / dse-native-only`             | 13 Aug 22:14 | 14 Aug 13:54          |
| `vg15 / us01-implausible-reinstated` | 14 Aug 08:17 | 14 Aug 13:54          |
| `vg15 / psi-drop-es01`               | 14 Aug 08:17 | 14 Aug 13:54          |
| `vg15 / psi-drop-uk07`               | 14 Aug 08:54 | 14 Aug 13:54          |
| `vg15 / tau-psi-narrow`              | 14 Aug 08:54 | 14 Aug 13:54          |
| `vg15 / tau-psi-wide`                | 14 Aug 09:31 | 14 Aug 13:54          |

Each still differed from its baseline in `clamp_mean_above_hi_anchor`, so the reported delta mixed the variant's effect with the clamp change. **Nothing detected this**: `compare_sensitivity.py` read whatever summaries were on disk and produced a perfectly well-formed matrix. The verdicts it wrote were "robust" and "sensitive" — the two most confident things the matrix can say, and neither warranted.

The whole VG15 robustness matrix was therefore evidence about the superseded definition. All seven have been refitted at `rep`, and the tool now diffs the two recorded definitions and refuses to assess a pairing that differs outside the variant's own override keys (`status = stale-pairing`).

### 7.2 Three more ways that matrix could mislead, now closed

- **A failed variant vanished.** `vg11 / anchor-broad` failed the gate (§5b), so its output went to `output/failed/` and `compare_sensitivity.py` skipped it with a console note — leaving the matrix silent about it, which is exactly what §5b says must not happen. It now appears with its status, its R-hat and ESS, and a pointer to the retained fit.
- **A targeted rerun clobbered the matrix.** `--variant <name>` rewrote the whole file from that one row. That is how `robustness_matrix_vg11.csv` came to hold a single variant. Targeted runs now merge, and every row carries `computed_at_utc`.
- **Coverage collapsed silently.** The series are paired on exact `age_months`, which is safe for the query grid (integer months) and not for the plot-grid `gap` series: that is `linspace(min_age, max_age, n_plot)`, so two fits share its ages only if they share an age range. `vg10 / dse-native-only` restricts the pool, so its `gap` grid differs and the intersection collapsed to **3 of 355 points** — reported as an ordinary "sensitive: gap" verdict on 43 of 395 rows. Coverage is now measured with the same rule the comparison uses, and a pairing below 90% is reported as `partial-coverage` rather than assessed.

The common thread is worth stating plainly: **every one of these produced numbers.** None of them produced an error, a blank or a missing file. A comparison harness that reads two directories of CSVs will compare whatever it finds, and the failure mode is a confident verdict about the wrong thing.

### 7.3 Documentation found stale

`docs/models/README.md` and `docs/models/PRIORS.md` still described the **72-month** comprehension cap, the pre-`uk_07` row counts, and "signed keeps the full grid" — the policy this very run replaced, in the two documents `CLAUDE.md` names as the inventory and prior reference. The VG16 inventory row still said the within-child anomaly's cause was unestablished, one commit after §7.1 of [202608151500](202608151500-within-child-crosslag-feasibility.md) established it. Two notes carried "correction pending" statements for corrections already made, one of which had since been overtaken outright. All corrected 2026-08-16.

The pattern: **a change that lands in code and in a note does not thereby land in the reference documents**, and the reference documents are what a reader consults. Nothing tests them.
