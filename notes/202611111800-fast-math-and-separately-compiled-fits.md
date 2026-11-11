> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5.5).

# Fast-math and separately compiled fits

> [!IMPORTANT]
> Dated analysis, 2026-11-11, of an intermittent failure in `tests/test_observation_deterministics.py` seen in cold-cache runs of the parallel slow set during the dependency refresh in [#405](https://github.com/dseinternational/vocabulary-growth/pull/405). It qualifies the "identical draws" statement in [202608231530](202608231530-observation-deterministics-not-sampled.md) §3. Measured on Windows 11 AMD64 (i9-14900K) with Python 3.14.8, PyTensor 3.3.7, Numba 0.68.0, `llvmlite` 0.50.0, nutpie 0.16.11, pytest 9.2.0 and pytest-xdist 3.8.0. Every run used a scratch `PYTENSOR_FLAGS=base_compiledir=...`, so the shared cache was not touched.

## 1. The failure

The `two_fits` fixture samples VG07 twice on synthetic data with seed 11, 12 tuning and 12 retained draws per chain. The lean fit passes `var_names` without the observation-sized deterministics; the full fit stores everything. `test_not_storing_them_does_not_change_the_draws` asserts that the free-parameter draws are bit-identical. Two tests rebuild `f_u_obs` from the lean draws and compare it with the full fit's stored values at `rtol=1e-10`.

In cold runs of `pytest -n auto --dist loadgroup -m slow`, these assertions sometimes failed. The draws differed in the tenth significant figure, for example `p_slope_low_u` at 0.10410238594452771 against 0.10410238597319096. The module passed when run alone, cold or warm, and a warm full run passed.

## 2. What `var_names` changes

nutpie builds the log density and its gradient from `model.logp()` and the free variables alone (`nutpie.compile_pymc._make_functions`). `var_names` only filters the outputs of a separate expand function. The printed logp graphs of the two fits are identical, with 459 nodes each. Any difference in the draws must therefore come from compiling one graph to machine code that rounds differently.

## 3. One graph, different numbers

Each row below is a separate process with its own compiledir. The hashes cover the logp and gradient at 20 fixed points, and the draws of a fit with seed 11. Both fits in each process agreed with each other.

| Numba cache at process start                         | Fast-math on: logp | Fast-math on: draws | Fast-math off: logp | Fast-math off: draws |
| ---------------------------------------------------- | ------------------ | ------------------- | ------------------- | -------------------- |
| Empty                                                | `7090ce1f`         | `d75fbce3`          | `06a186b8`          | `f7dc9cad`           |
| Holding this graph's kernels (warm)                  | `6bc47442`         | `b1dadb73`          | `06a186b8`          | `f7dc9cad`           |
| Holding kernels from a VG07 graph on 30 observations | `5d7577b6`         | `90e6c68a`          | `06a186b8`          | `f7dc9cad`           |

With PyTensor's default fast-math flags, the same graph, seed and code give three different posteriors to the last bits, depending only on what the Numba cache held. With `numba__fastmath=False`, every state gives the same bits.

The hybrid processor is not the cause. LLVM reports the same CPU name and the same feature set on performance and efficiency cores.

## 4. Mechanism

PyTensor's Numba backend compiles every kernel with the fast-math flags `reassoc`, `contract`, `arcp`, `afn` and `nsz` (`pytensor.link.numba.dispatch.basic.numba_njit`, under the default `numba__fastmath=True`). These flags allow LLVM to reorder and fuse floating-point operations, so the same IR optimised in different contexts can round differently.

Numba builds a caller by linking the IR of the functions it calls into the caller's module as `linkonce_odr` functions and optimising the result. A cached function carries its object code and that linkable IR. Which definition of a shared called function runs, and in which optimisation context it was compiled, therefore depends on what the process compiled or loaded beforehand.

The measurements fit this account. The cached top-level logp kernel gives `7090ce1f` in the process that compiled it and `6bc47442` in a process that only loaded it. With strict arithmetic, LLVM has no licence to reorder, so every context gives the same result.

## 5. Why only cold parallel runs

pytest 9.2.0, in the lock since `7d24a80` on 2026-10-28, derives `Node.nodeid` from `Node._id`. pytest-xdist 3.8.0 implements `--dist loadgroup` by rewriting `item._nodeid`, which pytest 9.2 no longer reads. pytest 9.1.1 still did. Every `xdist_group` mark in the suite has therefore been ignored since that refresh, and several workers each set up `two_fits` in a slow run. The upstream report is [pytest-dev/pytest-xdist#1386](https://github.com/pytest-dev/pytest-xdist/issues/1386), fixed by [#1388](https://github.com/pytest-dev/pytest-xdist/pull/1388) but not yet released.

A cold run with a fresh compiledir and the original test reproduced the failure. Four workers set up the fixture, compiled the same lean logp at the same time and raced to save it. In one worker, the full fit loaded the kernel that worker had saved and matched its lean fit (`7090ce1f`). In the other three, the full fit loaded a kernel saved by another worker and gave `6bc47442`. Both rebuild tests ran in two of those three workers and failed. The draw test ran in the matching worker and passed: 390 passed, 2 failed.

Run alone, the module has one worker, whose full fit loads the kernel its own lean fit saved. Warm, both fits load the same kernel.

## 6. The change

The fixture runs both fits inside `pytensor.config.change_flags(numba__fastmath=False)`. Strict arithmetic makes the compiled logp a function of the graph alone, which is the claim under test, so the bit-identical assertion stays. Under strict arithmetic the stored grid deterministics also match exactly. Under fast-math they differ by up to 4.4e-16 absolute (`f_u_plot`). That is the difference [202608231530](202608231530-observation-deterministics-not-sampled.md) §3 attributed to loop fusion; it comes from fast-math. The test keeps its tolerance for them, because the two expand graphs are entitled to differ. The recomputation tests compile under the default flags, and their `rtol=1e-10` absorbs that.

With the change, a cold run with a fresh compiledir passed 392 of 392 tests. Five workers set up the fixture, and every lean and full fit gave `06a186b8`, whether its kernel was compiled or loaded. A warm rerun on the same compiledir and a second cold run on another fresh one also passed 392 of 392.

## 7. What this means for fits

- Production fits use fast-math. Two fits with the same definition, data, seed and code are bit-identical only when they compile in the same cache state. Otherwise their draws start to differ in the last bits, and nothing keeps that difference small as chains run. Both are valid samples from the same posterior. Reproduction checks across machines, cache states or library versions should compare estimates against Monte Carlo error, not bits.
- [202608231530](202608231530-observation-deterministics-not-sampled.md) §3 compared old and new code with one seed and found bit-identical free parameters. Both fits must have compiled in matching cache states. Its central point stands: `var_names` leaves the logp graph unchanged.
- The executable signature records code and numerical-library versions, not compiler cache state, so it cannot certify bit-identical reruns and was never meant to.
- Restoring `loadgroup` grouping under pytest 9.2 is separate work: a `pytest<9.2` pin, a conftest shim or a pytest-xdist release with #1388. Until then the slow set repeats every grouped fixture on several workers.
