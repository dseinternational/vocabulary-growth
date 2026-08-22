# Reproducing the software environment

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

`pyproject.toml` is the readable environment specification: it states the extras this repository needs and explains why. It does not restate the scientific stack — the canonical floors for `numpy`, `scipy`, `pandas`, `pymc`, `pytensor`, `nutpie`, `arviz`, `preliz` and `xarray` live in `dse-research-utils`' own `pyproject.toml` and are inherited transitively, so they cannot drift between the repositories that share them. `.python-version` fixes the interpreter series, and the generated `uv.lock` resolves every package to an exact version and hash for `linux-x86_64`, `linux-aarch64`, `macOS-arm64` and `win-amd64`, including the immutable Git commit of `dse-research-utils`. The project itself is installed editable from the checked-out Git revision.

Intel macOS is not among the resolved platforms: `numba` publishes no macOS x86_64 wheels at all. That is upstream's decision, not a choice made here. Native Windows, on the other hand, is now supported and no longer needs WSL.

## Create an environment from the lock

```bash
uv sync --locked
```

`uv` provisions the interpreter itself from `.python-version`, creates `.venv/`, and installs exactly what `uv.lock` records. `--locked` fails rather than re-resolving if the lockfile is stale — which is what CI and any replication run wants. Plain `uv sync` updates the lockfile when `pyproject.toml` has moved on, which is what you want while changing dependencies.

Run anything in that environment with `uv run`, which needs no activation:

```bash
uv run python scripts/prepare_data.py
uv run pytest
```

Two things are not Python packages and so are not in the lock. The model-diagram figure shells out to the Graphviz `dot` binary (`brew install graphviz`, `apt install graphviz`, `winget install Graphviz.Graphviz`); a missing `dot` only skips that figure. Rendering reports needs [Quarto](https://quarto.org/docs/get-started/), and Quarto resolves its Jupyter kernel from `PATH` — see [Full refit](full-refit.md) for what that means in practice.

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
