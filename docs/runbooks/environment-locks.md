# Reproducing the software environment

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17.

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

Four things are not Python packages and so are not in the lock:

- **Graphviz** (`brew install graphviz`, `apt install graphviz`, `winget install Graphviz.Graphviz`). The optional model-diagram figure uses `dot`; if it is absent, the fit skips that figure with a warning.
- **[Quarto](https://quarto.org/docs/get-started/)** renders reports. Quarto resolves its Jupyter kernel from `PATH`, independently of the interpreter that ran the fit; see [Full refit](full-refit.md) for what that means in practice.
- **LaTeX** (`quarto install tinytex`) is needed for the report book's `pdf` format. Use a XeLaTeX-capable distribution and the Source Sans 3 and Monaspace Neon fonts. The `html` and `docx` formats need neither.
- **Node.js** runs CSpell and Prettier. Install these project tools with `npm install`.

Quarto bundles Pandoc, Dart Sass, Deno and Typst. Run `quarto check` to inspect the installed versions and the resolved LaTeX, Python and Jupyter paths. Confirm that rendering uses this project's Python environment.

The lock intentionally covers CPU installations. GPU drivers and CUDA are host-specific and remain an opt-in overlay rather than part of the reporting baseline.

## Refresh the lock after an intentional dependency change

```bash
uv lock
git diff -- uv.lock
```

To take upstream releases within the declared ranges, `uv lock --upgrade` (or `uv lock --upgrade-package <name>` for one). Commit the readable specification and the lockfile together. Do not refresh the lock as an unrelated formatting change: a lock diff is part of the scientific computing change and should be reviewed for unexpected solver upgrades.

Bumping the shared library is a change of the `tag` in `[tool.uv.sources]` followed by `uv lock`. To develop against a sibling checkout of `research` instead, comment that entry out and use the local-path override noted beside it.

## Relationship to fit manifests

The lock reconstructs a known software environment prospectively. Every completed model fit also writes the versions it actually used into `fit_manifest.json`. The two records answer different questions: the lock says what a clean environment should contain, while the manifest says what a particular fit did contain. A reporting run should agree with both.
