# Agent instructions

> [!NOTE]
> Maintained with assistance from LLM-based AI tools, including OpenAI Codex/GPT-6.

Keep this file, `CLAUDE.md` and `.github/copilot-instructions.md` identical. Update all three together.

## Project and reading guide

This exploratory study describes vocabulary development in children with Down syndrome. It estimates words understood, spoken and signed, their relationships, and variation between children. The aim is to help families and practitioners interpret patterns of development. These observational models do not establish causes. Their predictive distributions describe a range of possible counts under the model, not a certain course for an individual child.

The Python package is `src/vocab_growth/`. Reports use Quarto (`.qmd`). Start with:

- [Model inventory](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/models/README.md) for model structures, reporting roles and registration requirements.
- [Prior specification](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/models/PRIORS.md) for the statistical assumptions.
- [Data guide](https://github.com/dseinternational/vocabulary-growth/blob/main/data/readme.md) and individual source notes for provenance and measurement limits.
- [Full-refit runbook](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/full-refit.md) for fitting, validation and publication.
- [Notes index](https://github.com/dseinternational/vocabulary-growth/blob/main/notes/README.md) for dated evidence and decisions. A note describes its own data and code revision, not necessarily the current analysis.

There are twenty-three registered models: `VG01`-`VG16` and `VG19`-`VG26`, excluding retired VG06. VG17 and VG18 are unregistered exploratory models. Use `MODEL_REGISTRY` and `models.catalogue.CATALOGUE` for the executable model set. The inventory records their reporting roles.

## Environment

Use the [locked uv environment](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/environment-locks.md):

```bash
uv sync --locked
uv run python scripts/prepare_data.py
npm ci
```

`uv` supplies Python from `.python-version` and installs the packages pinned in `uv.lock`. The supported platforms are Linux x86_64 and aarch64, Apple Silicon macOS, and native Windows AMD64. On Windows, set `PYTHONUTF8=1` so progress symbols display correctly.

`pyproject.toml` declares `dse-research-utils[columnar,graphs,io,jax,notebook,viz]`, installed from the public `dseinternational/research` tag. The shared library declares the scientific dependencies. Do not duplicate their version floors here. To use a sibling checkout, follow the commented path override in `[tool.uv.sources]`.

Refresh `uv.lock` only for an intentional dependency change. The lock covers CPU installations; GPU support is a separate, host-specific setup.

Quarto is needed to render reports. The report book's PDF format also needs a XeLaTeX distribution and the Source Sans 3 and Monaspace Neon fonts. Graphviz `dot` supplies model diagrams; a missing binary produces a warning during fitting but leaves the report without that figure. Node.js supplies the documentation tools. Run `quarto check` to inspect the rendering environment.

## Checks

```bash
uv run ruff check src/ scripts/ tests/
uv run mypy
uv run pytest
uv run pytest -m "slow or not slow"
npm run spellcheck
npm run format:check
python3 tests/test_notes_index.py
```

A bare `pytest` runs only tests marked `not slow`. Run the full set before pushing engine changes. For parallel runs, use `-n auto --dist loadfile -m "not slow"` for fast tests and `-n auto --dist loadgroup -m slow` for slow tests. Slow tests sharing an expensive fit or compiled fixture need an `xdist_group` mark so they stay on one worker.

`mypy` checks the four declaration modules listed in `pyproject.toml`, not the PyTensor graph code. CI splits fast and slow tests into separate jobs. Changes to agent instructions or `docs/models/` run the full CI checks even if they only edit prose.

The test fixtures use Matplotlib's Agg backend and suppress routine reporting figures. Mark a test `emits_reporting_artefacts` when it needs to check those outputs. Prepare the data before tests in a fresh checkout; the database and merged CSV are generated files.

Use `npm run format` to apply Markdown formatting. Prettier preserves prose line breaks. Write each paragraph on one line, with blank lines between paragraphs. Use British English; the spelling configuration is `.cspell.config.yaml`.

## Data preparation

```bash
uv run python scripts/prepare_data.py
uv run python scripts/build_us01_source.py --verify
```

The first command merges source CSVs into `data/vocab_data_merged.csv` and `data/vocabulary.duckdb`. The second rebuilds the Edgin Down syndrome source from item-level Wordbank contributor files and verifies in-window rows against the export. See [the source record](https://github.com/dseinternational/vocabulary-growth/blob/main/data/vocab_data_us_01.md). The Wordbank by-child export remains the source for the typically developing pool.

Both data loaders sort rows before masking. Preserve this deterministic order because `analysis_frame_hash` includes row order. `vocab_growth.analysis_frames` rebuilds each model's prepared frame without fitting it.

Before changing a data exclusion, read its governing constant's docstring in `data_utils.py` and the source note. In particular:

- Down syndrome administrations above a form's registered age window are admitted. An early-vocabulary form can be appropriate for an older child.
- Comprehension below production is checked against `max(produced, spoken)`, not `spoken + signed`. Speech and signing overlap. Only comprehension is masked, and equality is retained.
- `ie_02` uses DSE Checklists 1 and 2. It remains on the 810 reference scale with its own 476-word form ceiling. It is not a complete DSE form. The equivalent partial baseline wave in `ie_01` is masked.
- The typically developing language scope is part of each model definition. The hierarchical reference models include English, Italian and Spanish (European); VG03 and VG04 remain English-only.

## Fitting and reporting

```bash
uv run python scripts/fit_model.py vg20 --config dev
uv run python scripts/fit_model.py vg20 --config rep --render
uv run python scripts/fit_model.py vg20 --config rep --render-only
```

Use `--help` for the full interface and the [runbook](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/full-refit.md) for a reporting run. `all` takes its model set from the registry. `dev` is the default; `test` and `rep` provide longer sampling runs. A render failure leaves a completed fit available for `--render-only`.

Output goes to `<output-root>/models/<model-name>/`. The root is chosen from `--output-dir`, then `DSE_VOCAB_GROWTH_OUTPUT_DIR`, then the checkout's `output/`. Fit, sensitivity, sync and upload commands use this same rule. The report cache always stays at `docs/report/figures/`.

`--trace-persistence` accepts `full` (default), `compact` or `minimal`, overriding `DSE_VOCAB_GROWTH_TRACE_PERSISTENCE`. Observation-sized deterministic arrays are omitted at every tier and can be rebuilt with `posterior_recompute`. `compact` also omits scaled random effects; `minimal` additionally omits log likelihood and posterior predictive draws. These choices do not change the posterior, but recovery scoring, `loso_compare.py` and `regenerate_plots.py` require `full`. Check downstream needs before reducing storage. The manifest records omissions under `artefacts.trace`.

`--nutpie-backend` selects `numba` (default) or `jax`, overriding `DSE_VOCAB_GROWTH_NUTPIE_BACKEND`. JAX is the fallback for the documented VG15 `fallback-dispersion` compilation failure on Linux aarch64. The backend is recorded as runtime information.

For parameter recovery, see the [recovery runbook](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/parameter-recovery.md):

```bash
uv run python scripts/fit_recovery.py headline --config test --replicates 3
```

`headline` selects VG20, VG12 and VG15. `all` uses `recovery.spec.supported_models()`; excluded models have reasons in `UNSUPPORTED_REASONS`. Posterior truth requires an existing fit; `--truth prior` does not. `--set-truth NAME=VALUE` sets a free variable before derived quantities are recomputed. VG16 is supported. VG25 requires simulation in wave order because its lag reads a same-stage outcome. Score only converged replicates, and do not treat a few replicates as a coverage guarantee.

Prepare report assets in this order:

```bash
uv run python scripts/sync_report_figures.py --config rep
uv run python scripts/prepare_report_figures.py
```

Sync validates fits and comparison manifests before replacing cached plots and tables. It does not copy traces. `--allow-provisional` relaxes publication checks for local inspection only. The preparation command generates observed-data summaries, illustrations, prior figures and labelled placeholders for missing figures. A rendered placeholder is not a completed result.

## Fit compatibility and provenance

These checks answer different questions. Do not replace one with another:

- The serialised definition records graph, data, reporting and identity fields. Every difference fails validation unless a checked `fit_identity.BACKFILL_DEFAULTS` entry establishes that an absent field had exactly the recorded default behaviour.
- The prepared-frame hash checks the actual rows and values used by a model. A matching frame can excuse a changed raw-data fingerprint, because another population's input may have changed without affecting this model.
- The executable-code signature hashes the package's Python AST, excluding comments and docstrings, and records numerical library versions. Resume, sync and publication enforce it. For a reviewed code-only change, `resume_from_trace.py --allow-implementation-change REASON` records why the retained samples remain valid; numerical library versions must still match. The regenerated manifest records current reporting provenance and retains the original sampling manifest. Publication checks both checkouts for uncommitted changes. Rendering and provisional sync do not enforce the signature. `expected_implementation=None` means the signature is not checked.
- Lifecycle, sampling quality and clean provenance checks determine whether a fit is complete and suitable for publication.

Trace-reading scripts use `fit_consumers`, which checks definitions and data compatibility. Its explicit `--allow-stale-fit` override reports what it bypassed. New trace consumers and comparison writers must join the relevant coverage registries or record a justified exemption. Comparison manifests link outputs to their contributing fits or input-data fingerprint.

## Model code

Model wrappers select a definition and dispatch to a shared engine. Definitions belong in `models/definitions.py`; engine and reporting declarations belong in `models/catalogue.py`. Keep engine identity out of the serialised statistical definition. See the inventory's registration checklist before adding a model.

The engines expose `build_model_graph` separately from reporting. Shared helpers prepare observations, child and study effects, age functions and likelihoods. Most outcomes use a Beta-Binomial likelihood; signing cross-tabulations use a Dirichlet-Multinomial. Do not assume all engines use the same study reference or prediction target.

Use `reporting_ages` and `intervals` for reporting policy. The reporting models include sex as a covariate; the inventory explains its coding and interpretation. Read model-specific caveats before editing a report or interpreting a quantity.

## Contribution conventions

Every Python source file starts with:

```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

Use Ruff for Python style and import ordering. `E501` and `E741` are intentionally ignored. Notebooks use Jupytext percent-format `.py` files; `.ipynb` files are gitignored.

Use Conventional Commits, such as `docs(report): clarify predictive intervals`. Put issue-closing references in the commit body or pull-request description.

AI-assisted documents, pull requests, issues and comments must identify the actual tool and model near the top. Use a GitHub note for Markdown:

> [!NOTE]
> Drafted or revised with assistance from OpenAI Codex/GPT-6.

Use a Quarto callout in `.qmd` files:

```text
::: {.callout-note}
Drafted or revised with assistance from OpenAI Codex/GPT-6.
:::
```

Keep attribution concise and preserve earlier attribution when revising a document.
