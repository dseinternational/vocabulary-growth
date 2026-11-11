# Reproducing the software environment

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6 on 2026-09-17, 2026-10-09, 2026-10-13, 2026-10-21, 2026-10-26, 2026-10-28, 2026-11-03 and 2026-11-04, and from Claude Code/Opus 5.5 on 2026-09-27, 2026-09-28, 2026-10-10 and 2026-11-11.

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

The lock includes PyTensor 3.3.8. PyTensor 3.3.4 fixed the obsolete `-ld64` linker flag on Apple Clang 21 and later ([pytensor#2432](https://github.com/pymc-devs/pytensor/pull/2432)). PyTensor 3.3.3 needed a compiler wrapper on macOS 27 because the Xcode tools read the flag as a request for a library named `d64`. The locked version compiles with `/usr/bin/clang++` directly.

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

### Dependency refresh on 28 October 2026

The shared library is pinned to `v0.19.3`, commit `f881f5830fc68896a888651a08b78898111ba04a`. Its [0.19 upgrade notes](https://github.com/dseinternational/research/blob/v0.19.3/docs/migrating-to-0.19.md) describe dependency and design-token changes since 0.18.0. Version 0.19.3 raises the pandas and Polars minimums to versions already in this project's lock: pandas 3.1.0 and Polars 2.1.0. The lock retains `python-dateutil` 3.0.0 and JAX/`jaxlib` 0.12.0.1. JAX 0.12 requires `AVX2` and `FMA` on x86-64 CPUs; the patch fixes Windows CPU intrinsics. DuckDB remains at 2.0.0, whose generated database files require DuckDB 2.x.

The refresh advances pytest to 9.2.0, Ruff to 0.17.3 and the Hatchling build minimum to 1.33.0, together with compatible indirect Python dependencies. PyTensor 3.3.4 requires Numba at most 0.68.0, and PyMC 6.3.2 requires `cachetools` below 7, so newer releases of those two packages cannot enter this environment. The documentation packages, pnpm and GitHub Actions were already current when checked.

The categorical colour order is blue, orange, green, purple, teal and pink. The project's role-based indices stay the same: words understood, spoken and signed use indices 0, 1 and 2; Down syndrome, typically developing and their difference use 0, 2 and 1. In these role mappings, orange represents speech and differences, and green represents signing and typically developing references. Historical fit manifests retain their original environment and remain subject to the fit-compatibility checks.

### Dependency refresh on 3 November 2026

The shared library remains at `v0.19.3`. The lock advances PyMC from 6.3.2 to 6.4.0, PyTensor from 3.3.4 to 3.3.7 and PyArrow from 26.0.0 to 26.0.1. [PyMC 6.4.0](https://github.com/pymc-devs/pymc/releases/tag/v6.4.0) adds `better-optimize` for `find_MAP` and fixes LKJ forward sampling and log-density normalisation. This project does not call `find_MAP`, but it does use LKJ covariance priors. Existing fit-compatibility checks still apply; this refresh does not certify samples made with older numerical libraries. PyTensor still requires Numba at most 0.68.0, and PyMC still requires `cachetools` below 7, so those packages remain at 0.68.0 and 6.2.6.

The [LKJ correction](https://github.com/pymc-devs/pymc/pull/8497) changes the saved log probabilities for VG24 and VG25. Their three-dimensional LKJ prior has concentration 2. The normalising constant makes the prior density integrate to one; its correct logarithm is $-\log(3\pi^2/16)$, whereas the old implementation used the opposite sign. The resulting change is about $-1.2309666763$ at every parameter point. Both saved points for each model agree with this change within $4\times10^{-9}$. Only those log probabilities were regenerated; the saved parameters, graph structures and coordinates remain unchanged. This constant correction does not change the posterior shape with fixed concentration, although the separate forward-sampling correction changes prior draws.

The correlation-prior test evaluates the fitted density at 72 fixed combinations of correlation matrices, variable orderings and scales. It converts the density on Cholesky factors to a density on correlations and scales, then divides out the independent HalfNormal scale densities. The result must agree with the normalised LKJ(2) density $16\det(R)/(3\pi^2)$ for every case. This checks the density shape, normalisation and symmetry under all six variable orderings. The separate scale-prior sampling test remains. Model and fit sampling settings are unchanged.

The direct development minimums advance to IPython 9.18.1 and mypy 2.4.1. Compatible indirect packages also advance, including `python-dateutil` 4.0.0 and Soup Sieve 4.0. The [`python-dateutil` release](https://dateutil.readthedocs.io/en/stable/changelog.html#version-4-0-0-2026-11-01) drops Python versions older than 3.11 and rejects malformed numeric fields in ISO dates. [Soup Sieve 4.0](https://facelessuser.github.io/soupsieve/about/changelog/#40) changes CSS selection inside HTML templates and iframes. The project uses Python 3.14 and has no direct calls to either package.

The direct documentation packages remain unchanged. Their lock refresh updates `ansi-regex` to 6.5.0, `fast-equals` to 6.2.2 and `semver` to 7.8.6. pnpm advances from 12.15.0 to [12.16.1](https://github.com/pnpm/pnpm/releases/tag/v12.16.1), with matching package-manager entries in `pnpm-lock.yaml`. CI advances `pnpm/action-setup` to [6.1.1](https://github.com/pnpm/action-setup/releases/tag/v6.1.1) and `astral-sh/setup-uv` to [10.3.3](https://github.com/astral-sh/setup-uv/releases/tag/v10.3.3). The other GitHub Actions and the Hatchling build minimum were current when checked.

### Dependency refresh on 11 November 2026

The shared library is pinned to `v0.19.4`, commit `bd916566081579e19031cabb637475a26cbd62aa`. Its [0.19.4 upgrade notes](https://github.com/dseinternational/research/blob/v0.19.4/docs/migrating-to-0.19.md) raise eight minimums, including PyMC to 6.4.0, PyTensor to 3.3.7, PyArrow to 26.0.1, h5netcdf to 1.8.2 and pandas to 3.1.1. Only the pandas minimum is above the previous lock. The release changes no library code apart from its version string. Its design-token pin advances, but the research colour file and the figure colours are unchanged.

The lock advances PyTensor from 3.3.7 to [3.3.8](https://github.com/pymc-devs/pytensor/releases/tag/rel-3.3.8) and pandas from 3.1.0 to [3.1.1](https://pandas.pydata.org/docs/whatsnew/v3.1.1.html). PyTensor 3.3.8 speeds up Numba compilation and corrects Numba sampling from the unweighted `choice` operation. The model graphs do not use that operation; the project draws subsets with NumPy's generator outside the graph. Pandas 3.1.1 stops missing datetime values becoming Unix-epoch timestamps when converted to Arrow timestamp types, and corrects data sharing after Arrow-backed forward and backward fill. PyTensor 3.3.8 still requires Numba at most 0.68.0, and PyMC 6.4.0 still requires `cachetools` below 7, so those packages remain at 0.68.0 and 6.2.6 although Numba 0.68.1 and `cachetools` 8.0.0 are available.

PyTensor, pandas and `dse-research-utils` are among the numerical packages in the executable signature. Fits made before this refresh therefore fail the signature checks for resume, strict sync and publication under the new lock. This refresh does not certify their samples; the existing fit-compatibility rules apply.

The lint minimum and lock advance Ruff from 0.17.3 to 0.17.4. Compatible indirect packages also advance, including `filelock` 4.2.10, JupyterLab 4.6.6, Notebook 7.6.5, `markdown-it-py` 4.3.0, `mdit-py-plugins` 0.7.0, `platformdirs` 4.14.5, `pyzmq` 27.3.0, Soup Sieve 4.1, `threadpoolctl` 3.8.0 and Tornado 6.6.1.

CSpell advances from 10.4.0 to [10.4.1](https://github.com/streetsidesoftware/cspell/releases/tag/v10.4.1), with its bundled dictionaries, `ansi-regex` 6.6.0 and `commander` 15.1.0. pnpm advances from 12.16.1 to [12.18.0](https://github.com/pnpm/pnpm/releases/tag/v12.18.0), with matching package-manager entries in `pnpm-lock.yaml`; 12.19.0 is on pnpm's `next-12` channel. The GitHub Actions, Hatchling build minimum and other direct development minimums were current when checked.

Three releases published on 11 November, a few hours before this check, are held at their previous versions: h5netcdf 1.8.3, which writes the traces; `jsonpointer` 3.12; and Prettier 3.10.0. The next refresh can take them once they have been available for a few days. `uv lock --upgrade` and `pnpm update` do not retain this hold.

### pytest held below 9.2 on 11 November 2026

The development requirement is now `pytest>=9.1.1,<9.2`, and the lock returns pytest from 9.2.0 to 9.1.1. No other package changes. pytest 9.2.0 derives each test's node ID from a structured `Node._id`. pytest-xdist 3.8.0, still the latest release, marks `xdist_group` tests under `--dist loadgroup` by writing the older `item._nodeid` attribute, which pytest 9.2 no longer reads ([pytest-xdist#1386](https://github.com/pytest-dev/pytest-xdist/issues/1386)). With that pair, every group mark was ignored and no test failed. Each slow module that shares a module fixture under a group mark set the fixture up again on several workers. `test_graph_equivalence.py` rebuilt every registered graph on each of those workers. `test_observation_deterministics.py` ran its two VG07 fits on four or five workers, and their concurrent compilation exposed the cache-dependent result fixed in [#406](https://github.com/dseinternational/vocabulary-growth/pull/406).

A `conftest.py` hook that copied the group suffix into `Node._id` would keep pytest 9.2. It would depend on private attributes of both packages, however, and the upstream fix ([pytest-xdist#1388](https://github.com/pytest-dev/pytest-xdist/pull/1388), merged but unreleased) sends group names to the scheduler separately instead. The version cap uses only published package versions. Lift it when a pytest-xdist release includes that fix, and raise the `pytest-xdist` minimum to that release in the same change.

`tests/test_xdist_grouping.py` runs a small grouped suite on two workers under `--dist loadgroup` and fails if a group runs on more than one worker. It runs in the fast set. A lift without the pytest-xdist fix, or any later dependency change with the same effect, therefore fails CI.

## Relationship to fit manifests

The lock reconstructs a known software environment prospectively. Every completed model fit also writes the versions it actually used into `fit_manifest.json`. The two records answer different questions: the lock says what a clean environment should contain, while the manifest says what a particular fit did contain. A reporting run should agree with both.

## CI work selection and compilation reuse

CI always runs lint, type and documentation checks. `scripts/ci_changes.py` compares parsed dependency files before skipping model tests and the smoke fit for Ruff, mypy or changes limited to documentation-tool versions in `package.json`. It retains dependencies shared with the runtime or tests and treats unknown or unreadable changes as requiring the full checks. Changes to `pnpm-lock.yaml`, pnpm workspace settings, agent instructions and model documentation also run the full checks. The classifier uses only the Python standard library, so it does not parse pnpm's YAML files.

The compiled-cache key records the locked environment apart from dependencies used exclusively by Ruff and mypy, the Python interpreter, runner image, compiler and compilation settings. A source digest lets subsequent runs save newly compiled functions. Restore prefixes include the complete environment digest, so a numerical-library change cannot restore an older environment's cache. A changed environment digest starts a new compiled cache.
