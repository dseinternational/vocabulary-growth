> [!NOTE]
> Drafted by an LLM-based AI tool (Codex/GPT-6).

<!-- cspell:words reff invlogit unassessable nonfinite -->

# Shared library 0.13.0 migration

Issue [#309](https://github.com/dseinternational/vocabulary-growth/issues/309) upgrades `dse-research-utils` from 0.12.5 to 0.13.0. The lockfile selects commit `458cc41b1dc33f4c0204919253ac92251c61b2bb`. The existing extras and shared ownership of scientific dependencies remain in place. No other locked package changed. The review used the tagged [migration guide](https://github.com/dseinternational/research/blob/v0.13.0/docs/migrating-to-0.13.md) and [source review](https://github.com/dseinternational/research/blob/v0.13.0/docs/source-review-2026-09-06.md).

## Changed consumer behaviour

- Relative efficiency falls back to ArviZ's default only when sampled-parameter metadata is unavailable. Missing named variables, empty selections, and reader or calculation errors propagate. The local wrapper supplies its metadata reader to the shared helper. The local parameter-name helper raises the shared public `SampledParametersUnavailableError`.
- Model and comparison LOO tables retain the explicit non-finite and unusable counts. The unusable count includes each observation once, even when its positive infinite value is also above the threshold. The comparison warning uses that count. Model reports use it too. For older model tables, the complement of the finite good count gives the unusable count; adding the old bad bands would omit missing values. A zero threshold remains valid. Non-finite thresholds are rejected before writing a new table.
- Upload verification compares local filenames with the uploader's raw `relative_paths`. HTTP checks encode those paths once. Simulated Azure and HTTP clients cover spaces, plus signs, percent signs, non-ASCII names, skipped assets, excluded traces, and root versus nested reports. No test publishes files to Azure.
- The hard convergence gate reads `scan_completed` separately from whether each parameter could be assessed. A completed scan with unavailable values fails with an assessment explanation; a failed or empty scan fails with a completion explanation. Legacy summaries require finite extrema before their scan is treated as complete. The registered R-hat exception cannot excuse unavailable ESS or an unassessable parameter. Sensitivity comparisons and report verdicts now apply the same distinction. Report readers also accept null values in BFMI arrays and nested diagnostic amendments.

## Other release changes

This repository does not call the changed shared `build_hsgp_1d` or `build_tau_modifier` constructors. `models/gp_utils.py` builds `pm.gp.HSGP` directly with `GPGrid.M` and `GPGrid.L`, and uses the recorded grid centre where supplied. The shared constructor change therefore does not change this repository's basis or require a basis migration. The existing model-building and grid tests check the local path.

The interval wrapper delegates to the changed shared summaries. A regression test confirms that a median and its intervals use the same finite draws. Previously generated summaries containing infinite draws should be regenerated. No local code calls the shared numeric `invlogit`; symbolic model expressions use PyMC or PyTensor operations. No local code calls `save_plotcollection` directly. The shared pair-plot helper uses `close=False`, as does the local figure-saving wrapper, so the changed `close=True` behaviour requires no call-site edit.

## Validation and output follow-up

`uv sync --locked`, Ruff, mypy, Markdown formatting and spelling checks pass. The complete suite, selected with `uv run pytest -m "slow or not slow" -n 4 --dist loadfile`, passed with 2,243 tests passing and 11 skipped in 176 seconds. It included the slow sampling and numerical optimisation tests. After the final report edits, all 213 migration regression tests passed again. These cover the changed consumer contracts, including the decisions and explanations read back from written files. The full run emitted figure-count and Numba object-mode warnings; no test failed.

This checkout contains only a descriptive CSV under `output/models/TEST_VG15_NATIVE-test`, with no complete fitted model report or trace available for rendering. A Quarto HTML check therefore executes the actual report functions against labelled synthetic fixtures. It covers a clean legacy summary, a completed scan with null diagnostics, a failed scan, and LOO containing NaN and both infinities at a zero threshold. The resulting HTML contains the expected verdicts and reports four unusable observations out of six. This is a report-function check, not validation of fitted study results.

Before updating published results, regenerate diagnostic and LOO summaries and rerender reports from the corresponding fits. Rebuild comparison tables before aggregating them so they carry the new counts. The existing executable-code signature includes the shared library version and changed package code. It will reject older fits for publication and resume until they are refitted under the current implementation; this migration does not bypass or rewrite that provenance. Rendering an older fit for review remains permitted by the existing render policy. No reporting-quality refits or publication uploads were performed for this code migration.
