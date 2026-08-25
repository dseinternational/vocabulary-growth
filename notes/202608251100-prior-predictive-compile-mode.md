# The prior-predictive compile mode: 26x for identical draws

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

Closes the lever left open in [#229](https://github.com/dseinternational/vocabulary-growth/issues/229) — "lever 1 is still untaken", 2026-08-24. All four engines drew their prior predictive under `mode="FAST_COMPILE"`, with no rationale recorded at any of the four call sites. The argument is now removed from all four.

## What it cost

`FAST_COMPILE` is a PyTensor mode that skips most graph rewrites: it compiles quickly and then executes slowly. That is the right trade for a graph you build once and evaluate a handful of times, and the wrong one for 500–1,000 draws of the full model graph, which is what this stage is.

Measured on VG01 (`scripts` not involved; the model graph built through the real `prepare` → `priors` → `build` stages, then the prior predictive drawn twice at the same seed):

| mode                 |     run 1 |     run 2 |
| -------------------- | --------: | --------: |
| `FAST_COMPILE`       |   101.1 s |   109.1 s |
| default (optimising) |     3.8 s |     2.8 s |
| **ratio**            | **26.6x** | **38.5x** |

Two runs, because the absolute times move by a few percent on a shared workstation. The ratio is 26.6x and 38.5x; treat it as "between twenty and forty" rather than as a two-figure constant.

The stage is a **fixed** cost: `draws` is hard-coded at 1,000 (500 in the joint engine) and does not scale with the sampling tier, so it is a constant tax on every fit at every configuration. It was measured at 16m33s of VG12's 3h26m hightune — 8% of a long fit, and about a third of a short one.

## Why this is safe

The draws are the same. Compile mode changes how the graph is evaluated, not what it represents, and the RNG is seeded identically:

- **Free variables agree to 7.2e-16** relative (worst: `kappa_excess_young`). Not bit-identical: several are drawn on a transformed scale, and the transform is itself subject to the rewrites.
- **Deterministics agree to 3.4e-15** relative (worst: `kappa_obs`), which is the same effect one layer further down the graph.

Both are float association — operations reordered or fused — an order of magnitude inside double-precision epsilon, not a different distribution. Verified across all 30 variables on VG01, 8 free and 22 deterministic. The argument is engine-independent, because it is a property of PyTensor's compile modes rather than of any particular graph, so the same removal applies unchanged to the bivariate, joint-modality and trivariate engines.

## Two qualifications

**The measurement was taken on a machine with no C compiler.** `pytensor.config.cxx` is empty on the Windows workstation, so both arms ran through the NumPy/Numba backend and the 20–40x is the effect of the **graph rewrites alone**. On a Linux VM with `g++` present the default mode additionally compiles to C, so the gap should widen rather than narrow — but the figure to quote from this note is the rewrite-only one, and anyone wanting the VM number should re-measure there.

**Three `FAST_COMPILE` sites remain in `src/vocab_growth/recovery/simulate.py`** (lines 214, 360, 770). They are not this stage — two are `pm.compute_deterministics` and one is a free-variable prior draw inside the recovery harness — and they have not been measured. #229's lever named only the four engine sites. Worth a look before a large recovery run, but not changed here on an untested hunch.

## What it does not do

Nothing about the posterior, the model graph, or any fit's provenance. `compile_kwargs` is not part of the model definition, so no fit is marked stale by this and no refit is required to pick it up — the next fit simply spends the time on sampling instead.
