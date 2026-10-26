# Reproducing the software environment

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17, 2026-10-09, 2026-10-13, 2026-10-21 and 2026-10-26, and from Claude Code/Opus 5.5 on 2026-09-27, 2026-09-28 and 2026-10-10.

`pyproject.toml` is the readable environment specification: it states the extras this repository needs and explains why. The minimum versions for `numpy`, `scipy`, `pandas`, `pymc`, `pytensor`, `nutpie`, `arviz`, `preliz` and `xarray` come from `dse-research-utils`. This avoids maintaining duplicate dependency lists. `.python-version` fixes the interpreter series, and the generated `uv.lock` resolves every package to an exact version and hash for `linux-x86_64`, `linux-aarch64`, `macOS-arm64` and `win-amd64`, including the immutable Git commit of `dse-research-utils`. The project itself is installed editable from the checked-out Git revision.

The lock supports Linux x86-64 and ARM64, Apple Silicon macOS, and native Windows x86-64. Intel macOS is not supported by this environment. On Windows, set `PYTHONUTF8=1` for readable Unicode console output.

## Create an environment from the lock

```bash
uv sync --locked
```

`uv` provisions the interpreter itself from `.python-version`, creates `.venv/`, and installs exactly what `uv.lock` records. `--locked` fails rather than re-resolving if the lockfile is stale. Use it for CI and replication. Plain `uv sync` updates the lockfile when `pyproject.toml` has moved on, which is what you want while changing dependencies.

Run anything in that environment with `uv run`, which needs no activation:

```bash
uv run python scripts/prepare_data.py
uv run pytest
```

Five things are not Python packages and so are not in the lock:

- **Fonts.** Figures, model graphs and reports use Noto Sans for text and Noto Sans Math for equations, and reports use Noto Sans Mono for code. Install all three wherever figures or reports are rendered: from Google Fonts on Windows, `fonts-noto-core` on Debian and Ubuntu, or the `font-noto-sans`, `font-noto-sans-math` and `font-noto-sans-mono` casks on macOS. Use the static faces, because the PDF formats name each weight and style they use. matplotlib caches its font list, so after installing fonts delete `fontlist-*.json` from the directory that `uv run python -c "import matplotlib; print(matplotlib.get_cachedir())"` prints. `uv run python -c "from dse_research_utils.plot.styles import default_font_families; print(default_font_families())"` then lists Noto Sans Math if matplotlib can find it. Without Noto Sans, matplotlib silently sets text in the next installed font in the style's `font.sans-serif` list, such as Arial on Windows. Without Noto Sans Math, it sets mathematics in DejaVu Sans. Graphviz falls back to the system's sans-serif font. Symbols that Noto Sans lacks, such as → and ≤, come from Noto Sans Math and then DejaVu Sans, and matplotlib logs a few `findfont: Failed to find font weight` lines because neither has a medium weight. Those lines are expected.
- **Graphviz** (`brew install graphviz`, `apt install graphviz`, `winget install Graphviz.Graphviz`). The optional model-diagram figure uses `dot`; if it is absent, the fit skips that figure with a warning.
- **[Quarto](https://quarto.org/docs/get-started/)** renders reports. Quarto resolves its Jupyter kernel from `PATH`, independently of the interpreter that ran the fit; see [Full refit](full-refit.md) for what that means in practice.
- **LaTeX** (`quarto install tinytex`) is needed for the report book's `pdf` format. Use a XeLaTeX-capable distribution. The `html` format loads the three fonts from Google Fonts and does not need LaTeX. The `docx` format names them, so readers need them installed to see them. A Quarto post-render script, `scripts/restore_docx_math_settings.py`, restores the template's math font, which Pandoc drops.
- **Node.js** runs CSpell and Prettier. Use Node.js 26 and install the locked project tools with `pnpm install --frozen-lockfile`. See the pnpm setup below.

Quarto bundles Pandoc, Dart Sass, Deno and Typst. Run `quarto check` to inspect the installed versions and the resolved LaTeX, Python and Jupyter paths. Confirm that rendering uses this project's Python environment.

The lock intentionally covers CPU installations. GPU drivers and CUDA are host-specific and remain an opt-in overlay rather than part of the reporting baseline.

### PyTensor compilation on macOS 27

The lock now includes PyTensor 3.3.4, which fixes the obsolete `-ld64` linker flag on Apple Clang 21 and later ([pytensor#2432](https://github.com/pymc-devs/pytensor/pull/2432)). PyTensor 3.3.3 needed a compiler wrapper on macOS 27 because the Xcode tools read the flag as a request for a library named `d64`. The locked version compiles with `/usr/bin/clang++` directly.

If you set `PYTENSOR_FLAGS` only to select that wrapper, remove the setting from your shell configuration and unset it in the current shell. If the variable also contains other settings, remove only the wrapper's `cxx` entry. `PYTENSOR_FLAGS` is part of the numerical environment recorded in report and comparison checkpoints, so a changed value prevents reuse of outputs made under the old value.

## Install the documentation tools

Install [pnpm](https://pnpm.io/installation) alongside Node.js 26. `package.json` pins the pnpm version in `packageManager`; pnpm uses that version in the project, and CI reads the same pin. `pnpm-lock.yaml` records the exact documentation-tool versions and package hashes.

```bash
pnpm install --frozen-lockfile
pnpm run spellcheck
pnpm run format:check
```

`--frozen-lockfile` fails if the lock does not match `package.json`. After an intentional documentation-tool update, run `pnpm install` and commit `package.json` and `pnpm-lock.yaml` together. `pnpm run format` applies Markdown formatting.

## Refresh the lock after an intentional dependency change

```bash
uv lock
git diff -- uv.lock
```

To take upstream releases within the declared ranges, `uv lock --upgrade` (or `uv lock --upgrade-package <name>` for one). Commit the readable specification and the lockfile together. Do not refresh the lock as an unrelated formatting change: a lock diff is part of the scientific computing change and should be reviewed for unexpected solver upgrades.

Bumping the shared library is a change of the `tag` in `[tool.uv.sources]` followed by `uv lock --upgrade-package dse-research-utils`. Follow the release's upgrade notes in `research/docs/` as well: a raised ceiling can need its packages named explicitly, and a new release can need fonts or other system changes. To develop against a sibling checkout of `research` instead, comment that entry out and use the local-path override noted beside it.

### Dependency refresh on 26 October 2026

The shared library is pinned to `v0.19.2`, commit `1397003472170b5559df4f4652791c7db6a3957d`. Its [0.19 upgrade notes](https://github.com/dseinternational/research/blob/v0.19.2/docs/migrating-to-0.19.md) describe dependency and design-token changes since 0.18.0. The refreshed lock includes pandas 3.1.0, Polars 2.1.0, `python-dateutil` 3.0.0 and JAX/`jaxlib` 0.12.0.1. JAX 0.12 requires `AVX2` and `FMA` on x86-64 CPUs; the patch fixes Windows CPU intrinsics. DuckDB was already locked at 2.0.0, whose generated database files require DuckDB 2.x.

The categorical colour order is blue, orange, green, purple, teal and pink. The project's role-based indices stay the same: words understood, spoken and signed use indices 0, 1 and 2; Down syndrome, typically developing and their difference use 0, 2 and 1. In these role mappings, orange represents speech and differences, and green represents signing and typically developing references. Historical fit manifests retain their original environment and remain subject to the fit-compatibility checks.

## Relationship to fit manifests

The lock reconstructs a known software environment prospectively. Every completed model fit also writes the versions it actually used into `fit_manifest.json`. The two records answer different questions: the lock says what a clean environment should contain, while the manifest says what a particular fit did contain. A reporting run should agree with both.

## CI work selection and compilation reuse

CI always runs lint, type and documentation checks. `scripts/ci_changes.py` compares parsed dependency files before skipping model tests and the smoke fit for Ruff, mypy or changes limited to documentation-tool versions in `package.json`. It retains dependencies shared with the runtime or tests and treats unknown or unreadable changes as requiring the full checks. Changes to `pnpm-lock.yaml`, pnpm workspace settings, agent instructions and model documentation also run the full checks. The classifier uses only the Python standard library, so it does not parse pnpm's YAML files.

The compiled-cache key records the locked environment apart from dependencies used exclusively by Ruff and mypy, the Python interpreter, runner image, compiler and compilation settings. A source digest lets subsequent runs save newly compiled functions. Restore prefixes include the complete environment digest, so a numerical-library change cannot restore an older environment's cache. A changed environment digest starts a new compiled cache.
