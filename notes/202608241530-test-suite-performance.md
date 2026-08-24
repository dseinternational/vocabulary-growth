# Test suite performance: where the time went

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

**2026-08-24.** The suite took a little over seven minutes serially, on every push and pull request. This note records what the time was actually spent on, what was changed, and the two things a later reader should re-measure rather than trust.

## Headline

Measured on one Windows machine (32 logical cores, Python 3.14, the locked `uv` environment, DuckDB prepared beforehand as CI does). Serial baseline and the new CI configuration, both run in the same session:

| configuration                                       | wall    | result                               |
| --------------------------------------------------- | ------- | ------------------------------------ |
| before, serial (`pytest`)                           | 430.8 s | 1 failed, 957 passed, 9 skipped      |
| after, serial, everything (`-m "slow or not slow"`) | 282.8 s | 960 passed, 9 skipped                |
| after, four workers, everything (CI's shape)        | ~130 s  | 960 passed, 9 skipped                |
| after, `-n auto` on 32 cores, everything            | 104.3 s | 960 passed, 9 skipped                |
| after, local default (`pytest`, `slow` deselected)  | 69.8 s  | 910 passed, 9 skipped, 50 deselected |

Four workers is the CI-realistic row: GitHub-hosted runners have four vCPUs, and the extra cores on this machine buy little because the run is bounded by its slowest module rather than by worker count. The count goes 967 → 969 because two tests were added; the failure that disappears is the CRLF one described below; nothing was removed or silently skipped.

Run-to-run variance on this machine is 10–15% depending on background load, so configurations were only ever compared back to back within a single scripted job. Figures taken at different points in a session are not comparable and are not compared here.

## What the time was spent on

Not the modelling. Collection is 4 s, so imports were never the problem; the cost was two reporting side effects that the fit pipeline performs on every model build and that no test asserted on.

| measurement                                   | as-is   | neutralised |
| --------------------------------------------- | ------- | ----------- |
| `configure_bivariate_priors` (VG10)           | 4.30 s  | 0.00 s      |
| `configure_univariate_priors` (VG11)          | 4.13 s  | 0.00 s      |
| whole VG10 build (prepare + priors + graph)   | 4.96 s  | 0.57 s      |
| `describe_all` over the TD pool (18,815 rows) | 18.21 s | 0.00 s      |
| `load_data` for the same pool, for comparison | 0.06 s  | —           |

`_plot_and_print_dist` renders roughly ten priors per build to PNG _and_ SVG. `describe_all` runs Shapiro–Wilk and friends over the whole analysis frame; on VG11 that was 18 of the 21 seconds its data-preparation stage took. The load everyone suspects first — reading the pool out of DuckDB — is 60 milliseconds.

`test_trend_gp_consolidation.py` alone built 25 models (nineteen registered, six of them a second time for their dedicated graph test) and paid the prior-plot cost on every one. Silencing both side effects took that file from ~120 s to 12 s; caching the six duplicated builds took it to 8 s.

## What changed

1. **`tests/conftest.py`.** Sets the Agg backend once, and an autouse fixture silences `plot_distribution` and `describe_all`. Both are looked up as module attributes at call time, so one patch each reaches all six `common_*` engines. The five per-module stubs that did this by hand are gone.
2. **`tests/test_pipeline_reporting_artefacts.py`.** Opts back in via `@pytest.mark.emits_reporting_artefacts` and asserts a real build still writes its prior figures and still describes its frame. Silencing an untested side effect would otherwise leave it both untested and never run.
3. **`pytest-xdist`,** with `-n auto --dist loadfile` in CI. `loadfile` rather than the default `load`: several modules have module-scoped fixtures that are themselves fits, and per-test distribution rebuilds them on every worker that draws one of their tests (measured at 171 s against 206 s on the pre-change suite).
4. **A `slow` marker,** deselected by default through `addopts`, on the four modules that do real sampling or real optimisation. CI selects everything with `-m "slow or not slow"` — spelt out rather than an empty `-m ""`, which pwsh drops before pytest sees it, so the command is the same one a Windows contributor can paste.
5. **`test_subject_marginal.py` split.** Its sampler test was 68–124 s depending on load, by some way the longest in the suite, and under `--dist loadfile` it held the whole file — and therefore the whole run — on one worker. It now lives in `test_subject_marginal_sampling.py`; the shared fixture moved to the conftest and costs about two seconds to build twice.
6. **A session-scoped build cache** in `test_trend_gp_consolidation.py`, so the six models wanted by two tests each are built once.
7. **CI restructured.** The test job and the VG01 fit are now separate jobs: neither needs the other's result, and at 204 s the fit is the longer of the two once pytest is parallelised. A `lint` dependency group holds ruff alone — `uv run ruff` previously synced the whole project environment, 174 packages including PyMC and JAX, to run a self-contained Rust binary; it now installs one. `setup-node` gained its npm cache.
8. **The CRLF failure fixed.** `test_agent_instruction_copies_remain_identical` compared `read_bytes()`. `.gitattributes` marks `*.md` as `eol=lf`, but a tool that rewrites `CLAUDE.md` on Windows leaves CRLF in the working tree, and because git normalises on read `git status` still calls the tree clean. It now compares normalised text. The suite was red by default on Windows worktrees for this reason alone, which is its own argument.

## Two things to re-measure, not trust

**The PyTensor compile cache.** CI now redirects `base_compiledir` into the workspace and caches it. This is unverified: the machine these measurements were taken on has no C++ compiler (`pytensor.config.cxx` is empty), so PyTensor used its Python linker throughout and the cache could not do anything. On the Ubuntu runner `g++` is present and every graph is compiled through it. If the cache does not pay for its upload and download, delete the two steps — nothing else depends on them.

**The default deselection.** `addopts` carries `-m 'not slow'`, so a bare `pytest` no longer runs everything. The deselected count is printed on every run and CI is explicit, but it is a real change in what "the tests pass" means locally. If it causes a surprise, the honest fix is to drop it from `addopts` and put `-m "not slow"` in the documented fast command instead.

## What is left

After the changes, the suite is bounded by four modules doing real work: `test_subject_marginal_sampling.py` (nutpie compiling a quadrature graph, plus a posterior-predictive pass numba runs in object mode), `test_kappa_conditional_calibration.py` (twelve optimiser fits over 900 simulated children, across three seeds), `test_bivariate_re_holdout_mask.py` (predictive sampling through a VG07 graph) and `test_observation_deterministics.py` (two real fits of the same model, which is the only way to show the draws are unchanged). None of these should be optimised away. They are what the `slow` marker is for.

One upstream observation worth carrying: `plot_distribution` in `dse-research-utils` never closes the figure it creates, so a long run accumulates them. It did not matter once the calls were silenced here, but it is a real leak for anything that draws many priors in one process.
