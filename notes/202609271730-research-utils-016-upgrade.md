> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5.5).

> [!IMPORTANT]
> Correction added on 28 September 2026 with assistance from Claude Code/Opus 5.5. Adding Noto Sans Math to `font.family` does work under matplotlib 3.11.2 for plain-text symbols, provided it is set before the text is created and the style is not applied again afterwards. `dse-research-utils` 0.16.1 now sets such a list, and `tests/test_figure_text_glyphs.py` has been removed. See the [0.16.1 upgrade](202609281111-research-utils-0161-upgrade.md). The original account below remains as a dated record.

<!-- cspell:words cachedir fontlist mathtext Noto pingouin -->

# Shared library 0.16.0 upgrade

Issue [#369](https://github.com/dseinternational/vocabulary-growth/issues/369) upgrades `dse-research-utils` from 0.15.2 to 0.16.0, following upstream [research#111](https://github.com/dseinternational/research/pull/111) and the tagged [0.16.0 upgrade notes](https://github.com/dseinternational/research/blob/v0.16.0/docs/migrating-to-0.16.md). The lockfile selects commit `3c5b4d994248ec1df2a9d2f9d677f911f10412af`, which is what the annotated `v0.16.0` tag resolves to. The release changes no Python API. It sets the shared plot style and model graphs in Noto Sans and Noto Sans Math, and raises three ArviZ minimums. Nothing was refitted and nothing was published.

## The lock moved four packages

```bash
uv lock --upgrade-package dse-research-utils
```

| Package              | 0.15.2 lock | 0.16.0 lock |
| -------------------- | ----------- | ----------- |
| `dse-research-utils` | 0.15.2      | 0.16.0      |
| `arviz-base`         | 1.3.0       | 1.3.1       |
| `arviz-plots`        | 1.3.1       | 1.3.2       |
| `arviz-stats`        | 1.3.2       | 1.3.3       |

The three ArviZ packages moved because the previous versions no longer met the new floors. A name-by-name diff of the lockfile adds and removes nothing else. The `arviz` meta-package stays at 1.3.0, and the numerical stack is unchanged: `numpy` 2.5.3, `numba` 0.67.0, `pytensor` 3.3.2 and `pymc` 6.3.2. The upstream Ruff and pingouin minimums belong to `research`'s own development groups and do not reach this lock.

## Fonts

Noto Sans, Noto Sans Mono and Noto Sans Math were already installed on this workstation. matplotlib's font cache (`fontlist-*.json` in `matplotlib.get_cachedir()`) predated them and was deleted; afterwards the shared style resolves both families to the installed files. [Environment locks](../docs/runbooks/environment-locks.md) now lists the fonts as a prerequisite for rendering figures as well as documents, with the cache step, and the agent guidance says the same.

The documents moved to the same fonts in the same change: Noto Sans for text, Noto Sans Mono for code and Noto Sans Math for equations, in the HTML, PDF and DOCX formats of the report, paper, summary and model pages. HTML now sets equations as MathML because MathJax cannot use Noto Sans Math, and a Quarto post-render script restores the DOCX template's math font, which Pandoc does not carry into its output.

CI installs no fonts, so figures drawn there fall back to DejaVu Sans. No test depends on the rendered font, so the fallback is harmless there.

## Noto Sans cannot draw arrows or relation symbols

The upgrade notes warn that Noto Sans is wider and that `\mathcal` loses its script letters. They do not mention a larger gap. In the Unicode blocks from U+2190 to U+27FF (arrows, mathematical operators, technical symbols, dingbats), Noto Sans draws only the minus sign, U+2212, and the dotted circle that displays combining marks. Source Sans 3 drew `≤`, `≥`, `→` and `≈`, so literal symbols in figure text used to render and now come out as missing-glyph boxes. matplotlib only logs a warning.

matplotlib does not fall back to another font for plain text under the style's generic `sans-serif` family: it takes the first installed font in `font.sans-serif` and nothing after it. Adding Noto Sans Math as a second family (`font.family = ["sans-serif", "Noto Sans Math"]`) did not help under matplotlib 3.11.2 either; the boxes remained. Mathtext does work, because the style takes mathtext symbols from Noto Sans Math.

Three labels in package code were affected and now use mathtext:

- the posterior predictive CDF axis label in `plotting.py`, drawn for every model, now reads `$P(Y \leq k)$`;
- the VG14 sign-to-speech title in `common_trivariate.py`;
- the within-understood composition title in `common_joint_modality.py` (VG15, VG24 and VG25).

The other literal symbols in `src/` and `scripts/` are in console output and docstrings. `tests/test_figure_text_glyphs.py` scans the figure code for string literals passed to matplotlib's text functions, or as a `label`, and fails on a character from those blocks outside `$...$`. It reads no font file, so it runs in CI. Checked against the pre-change files, it flags exactly these three labels. The shared library's upgrade notes should mention this for other consumers.

## Figures regenerated for review

`scripts/prepare_report_figures.py` regenerated the report's non-fit figures in the new style: the descriptive tables and figures, the introduction's illustrations, the methods chapter's prior figures and the placeholders. These write only to the Git-ignored figure caches, which were copied aside first.

Fit figures can be redrawn without sampling only by `scripts/regenerate_plots.py`, which re-runs a fit's plot stage from its saved trace. Of the 23 registered fits:

- ten were redrawn: VG03, VG04, VG13, VG15, VG20, VG21, VG23, VG24, VG25 and VG26;
- VG11 and VG12 have no redraw path for the univariate random-effects engine;
- eleven were refused, because their raw data and prepared analysis frames no longer match the fit: VG01, VG02, VG05, VG07, VG08, VG09, VG10, VG14, VG16, VG19 and VG22. These are the 7–8 September fits and need refitting whatever the fonts.

The redraw covers the plot stage, plus the prior-density figures and the model graph, which the rebuilt context writes on its way to the trace. Those two bypass the script's staging directory and go straight into the fit directory, as the aborted VG24 run below showed. Prior-predictive, diagnostic and summary figures, such as `energy_plot`, `trace_plot`, `pair_plot` and `expected_counts_by_month`, come from stages that need a refit. The model graph's SVG now declares `font-family="Noto Sans,sans-serif"` in place of `Helvetica,sans-Serif`.

The first run redrew all ten fits in one process. It aborted with exit code 3 and no Python traceback at VG24's model build and trace load, after seven fits and with the test suite running alongside. VG24, VG25 and VG26 were then redrawn in separate processes, and all completed. The abort did not recur at the same point, so it points to resource exhaustion in the long process rather than to the upgrade.

Before and after images were compared for every changed figure in VG03, VG15 and VG24 and for all the non-fit figures, and for the multi-panel and long-titled figures of VG13 and VG20. VG21, VG23, VG25 and VG26 share engines with fits in that set and were not reviewed separately. Apart from the missing glyphs above, the wider font caused no clipping. Titles, legends and tick labels that fitted still fit. Some overlaps predate the change: the legend over the histogram in `psi_posterior` (VG15 and VG24), the legend over the first bars of the four-cell predictive check, and the "12 mo" marker over the 100 tick in the understood-against-spoken figures. The wider legend in VG24's `psi_posterior` now covers slightly more of the histogram.

The fit directories were then restored from a copy made before the redraw, `pre-20260927-noto-regen-snapshot/` in the output root. Leaving them redrawn would have mixed Noto Sans figures with Source Sans 3 figures from the other stages in one directory. The CDF figures redrawn before the fix above would also have kept their missing-glyph boxes. The restored files match the copy byte for byte, with their original timestamps. No manifest, state file or trace was written. The next refit draws every figure in the new fonts.

Before the restore, the redrawn CSVs were compared with the originals. Nine fits reproduced theirs byte for byte. VG13's seventeen posterior-predictive CSVs did not. Its engine re-runs the posterior predictive from the sampling seed, and VG13 was fitted on the 0.14.0 stack (numpy 2.4.6, PyTensor 3.3.0), whose random draws the current stack does not reproduce. The differences are Monte Carlo noise, at most 4 words on predictive count quantiles and under 0.01 on probabilities, but a kept redraw would have changed numbers the report quotes. `regenerate_plots.py` says the re-run reproduces the stored draws. That holds only on the fit's own numerical stack.

## The signature was already broken, and stays broken

`models/implementation_identity.py` hashes the installed versions of `pymc`, `pytensor`, `numpy`, `scipy`, `pandas`, `arviz`, `nutpie` and `dse-research-utils`, alongside the package's own AST. Every fit on disk already failed it before this change. The fourteen fits made on 7 and 8 September predate the numerical-stack moves of 0.15.0 and later locks. The nine made on 16 and 17 September predate twelve module changes, all from [#367](https://github.com/dseinternational/vocabulary-growth/pull/367). This upgrade adds the library version, and the label fixes add three modules. As before, `render` and `provisional-sync` do not ask for the signature, and the publish path needs the reporting-quality refit it already needed.

## What was checked

On win-amd64 (native Windows, `PYTHONUTF8=1`):

- `uv lock --check` and `uv sync --locked` install the lockfile without re-resolving, and `dse-research-utils` reports 0.16.0. The workstation's environment had not been synced since the dependency refresh in [#368](https://github.com/dseinternational/vocabulary-growth/pull/368), so the sync also installed that refresh's packages, including `pandas` 3.0.6 and `jax` 0.11.2.
- The checks ran on a clean checkout of this change, with its own `uv sync --locked` environment and `prepare_data.py` output, as CI prepares them. `ruff check src/ scripts/ tests/` is clean, and `mypy` finds no issues in its four modules.
- The whole test suite passes there, run as CI runs it in two jobs: `-m "not slow"` with `--dist loadfile` gives 2,503 passed and 11 skipped, and `-m slow` with `--dist loadgroup` gives 391 passed.
- `npm run format:check`, `npm run spellcheck` and `tests/test_notes_index.py` pass.

No model was fitted and no fit was published.
