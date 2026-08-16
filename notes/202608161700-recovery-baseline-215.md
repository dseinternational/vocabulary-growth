# Parameter recovery against the #215 refit: three models, three different answers

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Record of the recovery re-baseline run on 2026-08-16, the second half of #215's post-run verification. It covers the three models [#163](https://github.com/dseinternational/vocabulary-growth/issues/163) gates — VG10, VG12, VG15 — refitted against synthetic data generated from their own posteriors after the reporting refit. Two issues came out of it, [#225](https://github.com/dseinternational/vocabulary-growth/issues/225) and [#226](https://github.com/dseinternational/vocabulary-growth/issues/226); this note is the evidence they rest on.

## What was run

`scripts/fit_recovery.py <model> --config test`, three replicates each, truths from each model's own posterior, simulate → fit → score as the runbook sets out. `test` is the runbook's honest tier for a recovery claim.

One blocker had to be cleared first. VG15 aborted every replicate at the simulation stage with `Column 'subject_id' changed dtype on the Parquet round trip (object -> str)`. That is a **false positive**: under pandas 3 a column of Python strings written as `object` returns as the `str` dtype with values untouched, including the numeric-looking-id case the guard exists to catch. The guard now compares a dtype _class_ that unifies `object` and `str` and leaves everything else exact (`fae03f8`). The tests missed it because a DataFrame built from string literals is already `str` in pandas 3, so no fixture carried an `object` column at all — the two added tests build one explicitly.

## Headline: the three models behave very differently

| model | 89% coverage (nominal 0.89) | replicates clearing the gate | principal finding                                                |
| ----- | --------------------------- | ---------------------------- | ---------------------------------------------------------------- |
| VG10  | 0.72                        | 1 of 3                       | no single dominant miss; `tau_subj_u` weakly low                 |
| VG12  | 0.50–0.58                   | 2 of 3                       | `tau_subject` biased low 3 of 3 → #225                           |
| VG15  | 0.78–0.92                   | 1 of 3                       | best calibrated overall; `psi` and `psi_study` shrunk low → #226 |

Read the aggregate coverage cautiously: the recovery matrix labels it "indicative only — coverage over correlated quantities, not SBC", and it is dominated by grid quantities (`p_query`, `q_query`) whose points miss together when one curve is offset. The two findings below are different in kind — single scalars, one consistent direction, with a mechanism that reproduces through the parameters they derive from.

## VG12: the variance partition biases `tau_subject` low

`tau_subject` truths 0.676 / 0.687 / 0.694 return posterior means 0.631 / 0.651 / 0.655 — z = −3.66 / −3.02 / −3.37, truth outside the 89% interval **all three times**, a bias of −5.8% (relative misses of −6.6% / −5.3% / −5.6%; an earlier draft of this note and of #225 quoted the last of those three as though it were the average). Not a convergence artefact: r02 and r03 both cleared the hard gate and show it as strongly as the replicate that did not.

The bias enters through the split, not the budget:

| quantity                 | r01 z | r02 z | r03 z |
| ------------------------ | ----- | ----- | ----- |
| `v_total`                | +1.05 | +0.63 | −0.10 |
| `subject_variance_share` | −2.73 | −2.41 | −1.60 |
| `tau_subject`            | −3.66 | −3.02 | −3.37 |

`kappa_excess_young` moves the opposite way as the identity requires, confirming a mis-split of a correctly-estimated budget. `tau_subject = sqrt(share · v_total)` inherits the bias amplified, because the transformation concentrates the posterior (SD ≈ 0.012).

**The prior is not the cause in the obvious direction.** `share ~ Beta(3.9, 2.1)` has mean 0.650; the truths sit at 0.58–0.61 and the posteriors at 0.45–0.55, so the posterior is pulled _away_ from the prior mean. Shrinkage toward the prior would push `share` up, not down.

VG10, which has two free scales and no partition, does not behave this way: `tau_subj_u` gives z = −1.39 / −2.28 / −0.87 and `tau_subj_q` shows no consistent direction. The reparameterisation is the distinguishing feature, and it is carried only by VG11 and VG12.

Consequence: `tau_subject` is what the DS-vs-TD between-child contrast reads on the typically-developing side. If τ_TD is biased low by ~5.8%, the comprehension ratio TD/DS moves from 0.862 to about 0.92 — **the reported comprehension difference is overstated**. `build_variance_partition`'s docstring currently asserts that "the DS/TD heterogeneity contrast that `tau_subject` feeds is unaffected"; this is evidence against that.

## VG15: `psi` and especially `psi_study` are shrunk low

Population `psi` truths 2.549 / 2.969 / 2.658 return 1.917 / 2.013 / 2.409 — z = −2.65 / −3.77 / −0.86, posterior means clustering at 1.9–2.4 largely independently of the truth.

Per-study, for the four sources that identify `psi`:

| index | study   | r01         | r02             | r03         |
| ----- | ------- | ----------- | --------------- | ----------- |
| 0     | `es_01` | 0.91 → 0.91 | 1.14 → 1.18     | 1.44 → 1.40 |
| 4     | `nz_01` | 5.18 → 1.88 | 6.26 → **1.97** | 4.02 → 3.21 |
| 6     | `uk_02` | 2.67 → 2.81 | 2.16 → 2.14     | 1.71 → 2.01 |
| 11    | `uk_07` | 3.35 → 3.04 | 5.06 → 3.57     | 5.07 → 4.03 |

`psi_study[4]` at z = −6.20 is the worst single miss in the run. **The pattern tracks children, not administrations**: `es_01` (185 children) and `uk_02` (58) recover well; `nz_01` (33 children on 111 administrations) and `uk_07` (30) are over-shrunk. For a study-level parameter the administration count overstates a source's information.

Mechanism, and here the prior _is_ a live suspect unlike #225: `log_psi ~ Normal(0.3, 0.5)` puts `psi`'s prior median at 1.35 with 89% [0.61, 3.00], and the posteriors land between that median and the truth; `tau_psi ~ HalfNormal(1.0)` has prior median 0.67 against truths of 0.9–1.3, so the between-study scale is understated, which collapses the group estimates. The standard few-groups hierarchical pathology, with four groups.

The report already names the condition — `_caveats-signing.qmd` calls the population value "a shrunk centre, not a consensus" — so this quantifies an acknowledged limitation rather than contradicting one. Three consequences: the reported per-study spread (1.09 to 3.62) is **understated**, which strengthens the report's insistence that per-study values are the primary read; the population `psi` = 2.34 is **conservative** and `psi > 1` is unaffected; and there is an unresolved tension with the Type-M warning, which says selecting on an estimate that clears `psi = 1` inflates it while recovery says the estimator is biased low. Both can hold — selection up, shrinkage down — but the report currently states only the first.

## What this run cannot support

Three replicates per model at `test`, with 1 of 3 (VG10, VG15) and 2 of 3 (VG12) clearing the convergence gate. Truths drawn from each model's own posterior, so these are self-consistency checks: a bias means the pipeline does not recover its own generative parameter, which is the right question, but it says nothing about misspecification against reality. Magnitudes should not be quoted from this run — both issues ask for confirmation at `rep` with more replicates before acting.

The aggregate coverage figures in particular should not be read as calibrated, for the reason the matrix itself gives.

## What changed in the published material

Both findings are now visible to a reader of the report rather than only to whoever opens the issue tracker.

- **The comparison report** carries a callout in the between-child heterogeneity section, beside the existing one about the spoken side's independence assumption. The pairing is the point: that section previously offered comprehension as the clean contrast and spoken as the compromised one, and the measured bias is on comprehension.
- **The report's `How much children differ, in each population` section** carries the same caveat, and the sentence introducing it no longer calls the comprehension contrast "like-for-like" without qualification — it says like-for-like _in structure_, because structural symmetry was exactly what was being mistaken for an unbiased estimate.
- **`_caveats-signing.qmd` and the Discussion's limitations** now state the ψ shrinkage alongside the Type-M warning. Only the Type-M direction was stated before, which left a reader with one of two known biases and no way to know the other existed or that it ran the other way.
- **`build_variance_partition`'s docstring** no longer asserts the contrast is unaffected. It separates what the reparameterisation guarantees (the same quantities are reported) from what it does not (that their values are unbiased), and records the recovery result.

Every number in those caveats is computed from the scored replicate tables at render time, per the house style. That needed a fix to `sync_report_figures.py`: `_sync_dir` copies files and not sub-directories, so `output/comparisons/recovery/` had never reached `docs/report/figures/` and no chapter could have cited a recovery score at all. `recovery/` and `sensitivity/` are now synced explicitly.

Correcting the −5.6% figure was itself a consequence of computing rather than quoting: the wrong value only surfaced when the render produced −5.8% beside it.

## Housekeeping

Recovery fits are disposable and were reclaimed after scoring (the a1 variants, then VG10's and VG12's) to keep disk headroom for the Target 8 sensitivity fits. Their inputs — `truth.nc`, the synthetic frame and `simulation.json`, about 1.3 MB per replicate — are retained under `output/recovery/`, so any fit can be regenerated with `--fit-only` without re-simulating. Scored tables are in `output/comparisons/recovery/`.
