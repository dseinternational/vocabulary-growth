# The conversion rate for a freshly drawn child was computed and thrown away

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Date: 2026-09-07. Closes the "persist and report subject-marginal `q`" item on [#233](https://github.com/dseinternational/vocabulary-growth/issues/233), which the 2026-09-06 owner comment on [#289](https://github.com/dseinternational/vocabulary-growth/issues/289) named as one of the changes that has to land **before** the reporting-quality refit, because it adds a deterministic and so cannot be added to a fit that already exists.

## What the gap was

`common_bivariate.sample_posterior_predictive` builds one unseen child per posterior draw and reuses it across the plot and query grids, so the predictive trajectory is a coherent child rather than a different child at every age. It then stored two things from that child — `p_u_*_subject_marginal` and `p_s_*_subject_marginal` — and discarded the third, the conversion rate between them.

The consequence is narrow but real. Every report on this engine carries a population `q(a)` — the reference child in the average study, every random effect at zero — and no answer at all to "what share of their own comprehension does a child this age actually speak?", which is the reading a practitioner takes from a production ratio. The `p_s` column is the subject-marginal _count_, not the rate, and the rate is not recoverable from the stored tables afterwards: the summaries hold medians and quantiles, not draws, and the median of a ratio is not the ratio of medians. `p_u` and `p_s` are also correlated within a draw by construction, so no assumption about their joint behaviour rescues it either.

## What was done

Two deterministics, `q_query_subject_marginal` and `q_plot_subject_marginal`, alongside the `p_*` pair they were always implicit in, plus their `var_names` entries. `posterior_summary` gains `q_population_*` and `q_subject_marginal_*` blocks on `posterior_summary_q.csv`, mirroring the `p_population_*` / `p_subject_marginal_*` blocks the understood and spoken tables have carried since the estimands were separated.

The prose that reads the table is updated wherever there is prose to update: the shared `_bivariate_re_body.qmd` (VG10, VG19, VG20, VG22), and VG16, VG13 and VG21, which each carry their own. VG09 renders the table with no descriptive sentence at all and is left alone -- its report already says it exists to complete a structural argument rather than to supply an estimate.

Four decisions are worth recording because each could reasonably have gone the other way.

**The rate is stored, not recomputed.** Recomputing it in the plotting or reporting code would have needed no graph change and no refit. It was rejected because the unseen child differs by model — a correlated pair under `rho_uq`, a `(b0, b1)` pair for VG19, a factor block for VG22, one deviate scaled by `tau(age)` under the age-varying scale — and reproducing that construction outside the function that builds it is exactly how the correlation came to be silently dropped once before ([#224](https://github.com/dseinternational/vocabulary-growth/issues/224)). The test that pins this asserts identity rather than similarity: `p_s = p_u * q` draw by draw at zero tolerance on both grids, which can only hold if all three come from one child.

**No `Ey_` block.** `add_probability_estimand_columns` emits an expected-count block beside each probability, and the obvious move was to reuse it. `q` is the probability that a word a child _understands_ is one they also say, so `q * n_trials` is a word count only for a child who understands the whole 810-word reference inventory — the precise misreading the production-ratio figures already warn against. The new `add_rate_estimand_columns` is that function without the count half, and both now share one interval-block helper so the two cannot drift.

**The columns are gated on the model carrying a child effect on `q`.** VG08 puts a child effect on understood only, so its population and subject-marginal rates coincide; emitting both blocks there would tell a reader the model distinguishes two estimands it does not. Note that the _deterministic_ is still written for every model, as the `p_*` pair is — for VG08 it is the population rate, and that is a true statement about the graph rather than a mislabelling, since its subject-marginal `p_s` is already built from this same population `q`.

**The draws are optional on read.** No fit made before today stores them. Requiring them in `extract_model_samples` would have made every existing trace fail to load rather than merely lose one estimand, which would stop `regenerate_plots.py`, the sensitivity comparisons and `loso_compare.py` reading any fit of record. They read through `_optional_posterior_predictive`, and the reporting is gated on their presence as well as on the definition.

**One ordering consequence.** `--render-only` re-renders Quarto against the CSVs already on disk rather than rebuilding them, so rendering a fit made before today against the updated prose would describe columns that fit's `posterior_summary_q.csv` does not carry. That is harmless here only because #289 task 1.4 -- sync, render, upload -- is deferred until after the refit that regenerates them. It would not be harmless if the order changed.

## Scope, and two things deliberately not done

It covers the bivariate engines — VG05, VG07–VG10, VG13, VG16, VG19–VG23 — because `common_bivariate_re` imports this function rather than defining its own.

**The joint engine is untouched and is a separate gap.** `common_joint_modality.sample_posterior_predictive` (VG15, VG24) samples only the observed-row likelihoods: it builds no plot or query predictive grid and emits no `*_subject_marginal` node at all, although both models carry child effects on `q`. That is honestly declared rather than mis-stated — `vg15/index.qmd` says its curves are population-level and that a single child's range is much wider than any interval on the page — so it is an absent estimand, not a wrong one. Supplying it is a larger change than this one: a three-outcome unseen child with the sign block, on an engine whose predictive stage has no grid to hang it on.

**The sensitivity matrices do not compare it yet.** `sensitivity/compare.py`'s `_SERIES` treats a summary file that is present but lacks a required column as unreadable, and reports the quantity as unassessable. Adding a `q_subject_marginal` row today would therefore mark it missing on every existing baseline and variant. It belongs with the refit, once both sides of a pairing carry the column.

## An unrelated correction, found while here

`fit_consumers.EXEMPT_CONSUMERS`, added yesterday, gave this reason for `fit_recovery.py`:

> reads a posterior only as a truth generator … Staleness does not make the truth worse.

That is the opposite of what the code does. `recovery.simulate.truth_from_trace` puts the model of record through `validate_fit_output` against the registered definition, the raw-data fingerprint **and** the exact prepared-frame hash, refuses on any error, and says so in its own docstring — "A truth draw is only a truth draw for the data the fit saw". `load_simulation` separately compares the definition a simulation recorded against the one about to fit it, which is what the staged `--simulate-only` / `--fit-only` split needs.

The entry was not wrong to exist — `fit_recovery.py` does not use the shared helper, and the trace it opens directly is a recovery replicate's own fit in a variant directory, which is meant to differ from the registry. The _reason_ was wrong, and a reason that argues against a check the code makes is worse than no reason at all: it reads as a licence to remove it. Corrected, and `tests/test_fit_consumers.py` now checks both exemption claims against the code they describe rather than only checking that a reason is long enough to look considered.

That also settles the "harden recovery provenance checks" item on #233, which asks for exactly what `truth_from_trace` and `load_simulation` already do. It is unticked there rather than outstanding.
