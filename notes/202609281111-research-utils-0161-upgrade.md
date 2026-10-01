> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5.5).

<!-- cspell:words cachedir findfont fontlist mathtext Neue Noto -->

# Shared library 0.16.1 upgrade

Issue [#371](https://github.com/dseinternational/vocabulary-growth/issues/371) upgrades `dse-research-utils` from 0.16.0 to 0.16.1, following upstream [research#113](https://github.com/dseinternational/research/pull/113) and the tagged [0.16.1 upgrade notes](https://github.com/dseinternational/research/blob/v0.16.1/docs/migrating-to-0.16.1.md). The release fixes the gap that the [0.16.0 upgrade](202609271730-research-utils-016-upgrade.md) found and reported as [research#112](https://github.com/dseinternational/research/issues/112): plain figure text containing a symbol that Noto Sans lacks drew a missing-glyph box. The lockfile selects commit `874a27358080579ff851abaa77f241036322b983`, which is what the annotated `v0.16.1` tag resolves to. No figure changes. Nothing was refitted and nothing was published.

## The lock moved one package

```bash
uv lock --upgrade-package dse-research-utils
```

Only `dse-research-utils` moved, from 0.16.0 (`3c5b4d99`) to 0.16.1 (`874a2735`). The release changes no dependency. Between the two tags, the only library code that changed is `plot/styles.py`, apart from the version string.

## What 0.16.1 changes

`set_matplotlib_default_style()`, and therefore `init_script()` and `init_workbook()`, now sets `font.family` to a list instead of the generic `sans-serif`: `["sans-serif", "Noto Sans Math", "DejaVu Sans"]` where Noto Sans Math is installed, and `["sans-serif", "DejaVu Sans"]` elsewhere. matplotlib falls back glyph by glyph across the families in that list. Ordinary text still comes from Noto Sans, and a symbol it lacks, such as →, ≈, ≤ or ✓, comes from Noto Sans Math and then DejaVu Sans.

Figures without such symbols render exactly as before. The posterior predictive CDF figure from `plotting.py`, with its mathtext axis label and a bold figure title added, gave identical pixels under the 0.16.0 and 0.16.1 settings, with and without Noto Sans Math. No figure text in `src/` or `scripts/` uses such a symbol outside mathtext.

matplotlib now logs lines such as `findfont: Failed to find font weight medium for DejaVu Sans, now using 400.`, a few per process. Neither fallback font has a medium weight, which the style uses for titles and axis labels. Only fallback symbols are affected, and they draw at regular weight. [Environment locks](../docs/runbooks/environment-locks.md) records that these lines are expected.

## Why the two-family attempt failed

The 0.16.0 note records that `font.family = ["sans-serif", "Noto Sans Math"]` did not help under matplotlib 3.11.2. That was not a matplotlib limitation. Reproduced here with a title containing ≤, → and ≈:

| When the list is set                                    | Agg, SVG, PDF and PS |
| ------------------------------------------------------- | -------------------- |
| After applying the style, before creating the text      | All symbols drawn    |
| After creating the text                                 | Three boxes          |
| Before the style is applied again, as 0.16.0 applied it | Three boxes          |

A `Text` object keeps the font list in effect when it was created, and applying the style again resets `font.family`. This repository applies the style repeatedly: `comparison.py` does so inside package code, `prepare_report_figures.py` does so after `init_script()`, and several scripts call it directly. The 0.16.0 note does not record where the override was set, so the re-applied style is the likely cause but not a certain one. Under 0.16.1, applying the style again sets the full list, so repeated calls are harmless.

## The glyph test was removed

`tests/test_figure_text_glyphs.py` failed on a literal symbol from U+2190 to U+27FF in figure text outside mathtext. Its premise, that matplotlib does not fall back for plain text under the style, is untrue under 0.16.1, and such symbols now draw. Kept as a style rule, it would require mathtext for symbols that plain text now draws from the same font, Noto Sans Math. It could not be replaced by a rendering check in CI either: CI installs no Noto fonts, so a check there would never lay out text in Noto Sans, the font that lacks these symbols. Upstream now tests the fallback in `test_plot_styles.py`.

The three labels rewritten as mathtext in the 0.16.0 upgrade are unchanged: the CDF axis label in `plotting.py`, the VG14 sign-to-speech title in `common_trivariate.py` and the within-understood composition title in `common_joint_modality.py`. They render as before, and rewriting them as plain text would change figures without need. Their comments, which said that the plot font cannot draw the literal symbols, were removed. Comments are not part of the executable-code signature, and `implementation_identity.executable_source` returns the same result for all three modules before and after.

## Fallback wording

The environment-locks runbook and the agent guidance said that, without the Noto fonts, matplotlib falls back to DejaVu Sans. For text it takes the next installed font in the style's `font.sans-serif` list, which is Noto Sans, Helvetica Neue LT Std, Helvetica, Arial and then DejaVu Sans. On a typical Windows machine that is Arial. Only mathtext, and now fallback symbols, use DejaVu Sans when Noto Sans Math is missing. Both documents now say so, and the runbook gives a command that shows whether matplotlib can find Noto Sans Math.

## Fonts on this workstation

This workstation has Noto Sans but not Noto Sans Math or Noto Sans Mono. Neither is in the Windows font directories, and matplotlib's font list does not include them. The 0.16.0 note records all three as installed on the workstation it used. Here, `default_font_families()` returns `["sans-serif", "DejaVu Sans"]`, and mathtext falls back to DejaVu Sans with a `findfont: Font family ['Noto Sans Math'] not found` line. The checks above were repeated with the upstream `NotoSansMath-Regular.ttf` loaded into matplotlib's font manager for that process only, without installing it. Install Noto Sans Math and Noto Sans Mono before rendering reporting figures or documents on this machine.

## The signature changes again

`models/implementation_identity.py` hashes the installed `dse-research-utils` version, so this upgrade changes the executable-code signature. The 0.16.0 note records that every fit on disk already failed it. This change adds no source difference to the signature, because the only package source edits remove comments.

## What was checked

On win-amd64 (native Windows, `PYTHONUTF8=1`):

- `uv lock --check` and `uv sync --locked` install the lockfile without re-resolving, and `dse-research-utils` reports 0.16.1.
- The checks ran in a new worktree of this change, with a new `uv sync --locked` environment and `prepare_data.py` output, as CI prepares them. `ruff check src/ scripts/ tests/` is clean, and `mypy` finds no issues in its four modules.
- The whole test suite passes there, run as CI runs it in two jobs: `-m "not slow"` with `--dist loadfile` gives 2,517 passed and 11 skipped, and `-m slow` with `--dist loadgroup` gives 391 passed. The 14 extra fast tests since the 0.16.0 note come from [#372](https://github.com/dseinternational/vocabulary-growth/pull/372), less the two removed glyph tests.
- The first slow run had one failure: `test_the_complete_student_example_executes` in `test_model_walkthrough.py` could not write a numba cache file. This workstation sets `PYTENSOR_FLAGS=base_compiledir=V:\packages\pytensor`. PyTensor parses that variable with `shlex` in POSIX mode, which drops the backslashes, so the cache directory becomes the drive-relative `V:packagespytensor`. The test changes directory, and under `xdist` the relative path then failed to resolve on one worker. The test passed alone. The slow suite passed in full when `PYTENSOR_FLAGS` gave an absolute path with forward slashes. The failure is in the local environment, not in this change.
- `npm run format:check`, `npm run spellcheck` and `tests/test_notes_index.py` pass.

No model was fitted and no fit was published.
