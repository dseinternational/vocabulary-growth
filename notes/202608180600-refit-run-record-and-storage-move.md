# Refit run, 18 August 2026: working record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**In-progress.** Section 7 is the honest state of what is and is not done. Successor to [`202608142000`](202608142000-refit-run-record-and-disk-failure.md), whose run this one finishes.

## 1. What this run is for

Four refits, then VG20's gating checks, then [#229](https://github.com/dseinternational/vocabulary-growth/issues/229)'s diagnostics, then the re-publish. The refits exist for two different reasons, and conflating them wasted time before the survey was done:

| Model | Why                                                            | Config                       |
| ----- | -------------------------------------------------------------- | ---------------------------- |
| VG04  | stale by definition (#228 comprehension cap) **and** compacted | `rep`                        |
| VG14  | **not stale** — valid at publish tier, but compacted           | `rep`                        |
| VG12  | stale by definition **and** 4 divergences + BFMI 0.215         | hightune 12000/8000/**0.99** |
| VG11  | compacted **and** 16 divergences + an accepted R-hat exception | hightune 12000/8000/**0.99** |

All four at `full` persistence. VG11 and VG14 are being refitted _because_ their traces were compacted — which blocks `regenerate_plots.py`, `loso_compare.py` and recovery scoring — so refitting them compact would have recreated the reason for the refit.

`target_accept = 0.99` on both high-tuned refits was a bet on one piece of evidence: `vg11 anchor-broad` came back from those settings with zero divergences the night before, against VG11's 16 at plain `rep`.

## 2. The storage move, and the two defects it caused

`output/` moved to `/scratch2` at 20:52 on 17 August: 220,777,182,176 bytes and 5,655 files, byte-identical on both sides, verified before the original was deleted. `/scratch` went from 83% to 33%. The move itself was clean. Both defects were in things that had silently depended on `output/` living inside the checkout.

**`.gitignore` matched `output/` with a trailing slash**, which matches a directory and not a symlink. The link showed as untracked, so the checkout was dirty, and every fit started from it would have stamped `dirty: true` into its provenance block. Caught in the pre-flight before the queue started. Not hypothetical: `vg11 anchor-broad`'s manifest records `dirty: true` for the same class of reason, where every other Target 8 variant records `false`.

**`code-links: repo` became a hard render error.** Quarto resolves `repo` by walking up from the render directory for a git remote, and the per-model reports render _in the output directory_ — so it had been resolving by accident of layout. `/scratch2/vg-output/...` has no repository above it:

```
WARN: The 'repo' code link is not able to be created as the project isn't a GitHub project.
ERROR: Unknown code-link value 'repo'
```

It failed VG04's render after a 36-minute fit that had itself completed cleanly, and would have failed the other three and both books identically. Replaced with an explicit `href` in all 18 documents; it emits the byte-identical link and no longer depends on where the render happens.

## 3. A queue design fault of mine

The first driver ran `fit_model.py --render` as one hard-gated step, so a **render** failure killed the queue after a completed fit. The runbook's own rule is that "a rendering failure leaves the fit complete and available for a later `--render-only` retry" — the script was not honouring its own rulebook. Rebuilt with two step types: `run` for sampling (failure stops the queue) and `soft` for rendering (failure is logged, queue continues). Sampling is worth protecting; rendering is retryable in seconds.

## 4. The VG14 plotting crash, and what hid it

VG14 sampled for 40 minutes, printed its posterior summary, wrote its plots, and died:

```
ValueError: All arrays must be of the same length
```

`plot_modality_trajectories` built its CSV by pairing the full `X_plot` age column with median arrays trimmed at **three different caps** — understood 84, signed 84, spoken 90.

Two things hid it until a refit ran. The figure is written _before_ the CSV and draws each curve against its own trimmed x, so the plot was always correct and nothing looked wrong in published output. And `modality_trajectories` carries no outcome suffix, so it matched no stem in the reporting-age policy test's map — **the same blind spot that let the figure run to 115 months above a `p_any` table trimmed to 84**, which is what the per-outcome caps were added to fix. The fix for the first defect shipped the second one inside the same blind spot.

Worse, the single-quantity mapping (`modality_trajectories → SIGNED`) could not describe a file carrying four outcomes under three caps: it read the file as violating its cap whenever spoken legitimately outran signing. That mapping was **keeping VG14's stale-artefact exemption alive** — the exemption's own note said the first refit would clear it, and the refit landed while the test still reported the artefact as non-compliant.

Fixed by adopting `joint_trajectory`'s convention (shared age column to the widest cap, each series NaN past its own, extracted as `_multi_outcome_frame`), moving the stem into the multi-outcome set with its own test, and emptying `KNOWN_STALE`.

**VG14's plots were then regenerated rather than refitted** — 177 artefacts in under a minute. That only works because the refit wrote a `full` trace, and it is the first use of exactly the capability the full-tier decision was for.

## 5. What the fits found

| Model | Wall time | Divergences   | Max R-hat | Min ESS | Min BFMI              | Trace           |
| ----- | --------- | ------------- | --------- | ------- | --------------------- | --------------- |
| VG04  | 36 min    | —             | —         | —       | —                     | 4.9 G           |
| VG14  | 42 min    | —             | —         | —       | —                     | 12 G            |
| VG12  | 3h 26m    | **0** (was 4) | 1.0021    | 3,277   | **0.214** (was 0.215) | 29 G            |
| VG11  | _running_ |               |           |         |                       | ~95 G projected |

**The VG12 result is the one worth keeping.** Target-accept 0.99 removed every divergence, roughly tripled minimum ESS, and moved the energy BFMI by **0.001**. That converts `202608050900` §3 from an argument into a measurement: the divergences were a sampler-geometry problem and are now solved; the BFMI failure is structural and is immune to the strongest tuning the pipeline offers. VG12 now carries one convergence caveat instead of two. Recorded on #229.

## 6. Where the time and memory go

From VG12's 3h 26m:

| Stage                   | Time    | Share |
| ----------------------- | ------- | ----- |
| Posterior sampling      | 3h 09m  | 92%   |
| Prior predictive checks | 16m 33s | 8%    |
| Diagnostics             | 7m 45s  | 4%    |

The prior-predictive cost is **fixed** — `sample_prior_predictive(draws=1000, mode="FAST_COMPILE")` does not scale with the sampling tier, so it is a third of a short fit and 8% of a long one. `FAST_COMPILE` compiles quickly and executes slowly; the optimising default would trade seconds of compile for most of that, with the same graph, seed and draws. Not changed during this run.

VG11's memory climbs roughly linearly with sampling: 36 G at 2h, 62 G at 4h, 87 G at 6h, 112 G at 8h. `anchor-broad` peaked at 147 G at _compact_ tier, so this `full` run will peak higher, and the sharpest rise is at the end when the observation-sized deterministics are materialised. 127 G of swap now sits behind it — the earlier OOM had none.

## 7. State

Done: VG04, VG14, VG12 refitted, rendered and passing their gates. VG20 registered as a recovery target (#224 gate 2 could not have run without it). Gate 3's pass criteria committed _before_ any VG20 fit exists, so they cannot be tuned to the result. Target 8 written up on #147 — all seven variants robust — and the issue deliberately **left open**, because its scope is the full 50-variant matrix and only 15 are compared.

Outstanding: VG11 (running), then the two never-fitted `kappa` variants (armed behind the queue), then VG20's `rep` fit and its four gates, then #229's diagnostics, then the re-publish.

Not done and worth stating: this branch is behind `main` on `dse-research-utils` (0.9.2 against 0.10.0), and `main` already carries the corrected instruction files, so they were **not** edited here. Every fit in the set records 0.9.2 and the publish gate does not check package versions, so the set is internally consistent. Finishing the run and re-publishing on this branch, then merging, keeps it that way.
