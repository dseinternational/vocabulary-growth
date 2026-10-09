# Agent instructions

> [!NOTE]
> Maintained with assistance from OpenAI Codex/GPT-6 and Claude Code/Opus 5.5.

Keep this file, `CLAUDE.md` and `.github/copilot-instructions.md` identical. Update all three together.

## Project and reading guide

This exploratory study describes vocabulary development in children with Down syndrome. It estimates words understood, spoken and signed, their relationships and variation between children. The observational models do not establish causes. A predictive distribution describes possible counts under the model, not a certain course for an individual child.

The Python package is `src/vocab_growth/`. Reports use Quarto (`.qmd`). Start with:

- [Model inventory](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/models/README.md) for structures, reporting roles, interpretation and registration.
- [Prior guide](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/models/PRIORS.md) for assumptions and calibration.
- [Data guide](https://github.com/dseinternational/vocabulary-growth/blob/main/data/readme.md) and source records for provenance and measurement limits.
- [Full-refit runbook](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/full-refit.md) for fitting, validation and publication.
- [Notes index](https://github.com/dseinternational/vocabulary-growth/blob/main/notes/README.md) for dated evidence. A note describes its own revision; use successor notices and current guides before acting on old advice.

There are twenty-three registered models, `VG01`-`VG16` and `VG19`-`VG26`, excluding retired VG06. VG17 and VG18 are unregistered exploratory models. `MODEL_REGISTRY` and `models.catalogue.CATALOGUE` define the executable set. Registration alone does not establish a reporting role or certify a fit.

## Environment and checks

Use the [locked environment](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/environment-locks.md):

```bash
uv sync --locked
uv run python scripts/prepare_data.py
pnpm install --frozen-lockfile
```

`uv` supplies Python from `.python-version` and packages from `uv.lock`. Supported platforms are Linux x86_64 and aarch64, Apple Silicon macOS and native Windows AMD64. Set `PYTHONUTF8=1` on Windows. Refresh the lock only for an intentional dependency change. Scientific dependencies come from `dse-research-utils`; do not duplicate their version floors here. The commented `[tool.uv.sources]` override supports a sibling checkout. The lock covers CPU installations; GPU support is host-specific.

Figures, graphs and reports use Noto Sans, Noto Sans Math and, for report code, Noto Sans Mono. Install the fonts and delete `fontlist-*.json` from `matplotlib.get_cachedir()` before rendering. Missing fonts can cause silent substitutions. Quarto renders reports; PDF needs XeLaTeX. HTML equations use MathML. Graphviz `dot` supplies model diagrams. A missing `dot` warns and skips the figure. Node.js 24 supplies documentation tools. The environment runbook gives installation details; `quarto check` reports the rendering setup.

Run checks appropriate to the change:

```bash
uv run ruff check src/ scripts/ tests/
uv run mypy
uv run pytest
uv run pytest -m "slow or not slow"
pnpm run spellcheck
pnpm run format:check
python3 tests/test_notes_index.py
```

Bare `pytest` selects `not slow`. Run both sets before pushing engine changes. For parallel tests, use `-n auto --dist loadfile -m "not slow"` and `-n auto --dist loadgroup -m slow`. Slow tests sharing an expensive fixture need an `xdist_group` mark. `mypy` checks the four declaration modules in `pyproject.toml`, not the PyTensor graph code.

CI runs fast and slow jobs separately. Updates limited to Ruff, mypy or documentation-tool versions in `package.json` can skip model tests and the smoke fit. Changes to the pnpm lock, workspace settings, shared dependencies, agent instructions or `docs/models/` run full CI checks, as do unknown changes.

Prepare data before tests in a fresh checkout. The database and merged CSV are generated files. Test fixtures use Matplotlib's Agg backend and suppress routine reporting figures; mark a test `emits_reporting_artefacts` when it checks those outputs.

Use British English and one line per prose paragraph, with blank lines between paragraphs. `pnpm run format` applies Markdown formatting; `.cspell.config.yaml` defines spelling policy.

## Data rules

`prepare_data.py` builds `data/vocab_data_merged.csv` and `data/vocabulary.duckdb`. `uv run python scripts/build_us01_source.py --verify` rebuilds and verifies the Edgin Down syndrome source from item-level contributor files. The Wordbank by-child export supplies the typically developing pool.

Before changing an exclusion, read its constant's docstring in `data_utils.py` and the source record. Preserve deterministic row sorting before masking because `analysis_frame_hash` includes row order. `vocab_growth.analysis_frames` rebuilds prepared frames without fitting.

- Down syndrome administrations above a form's registered age window are admitted. An early-vocabulary form can be appropriate for an older child.
- The comprehension-below-production rule compares with `max(produced, spoken)`. Speech and signing overlap, so do not use their sum. Only comprehension is masked; equality is retained.
- `ie_02` uses Checklists 1 and 2 with a 476-word ceiling on the 810 reference scale. It is not a complete DSE form. The partial baseline wave in `ie_01` is masked.
- Language scope is part of the definition. Hierarchical TD references include English, Italian and Spanish (European); VG03 and VG04 remain English-only.

## Fitting and reporting

```bash
uv run python scripts/fit_model.py vg20 --config dev
uv run python scripts/fit_model.py vg20 --config rep --render
uv run python scripts/fit_model.py vg20 --config rep --render-only
```

`all` selects the registry. `dev` is the default; `test` and `rep` use longer sampling runs. A render failure leaves a completed fit available for `--render-only`. Checked reports and downstream outputs can be reused. `--force-render` rebuilds one report; the replication driver's `-FreshReports` rebuilds downstream outputs without forcing new samples. Read `--help` and the runbook for phase selection.

Fits go to `<output-root>/models/<model-name>/`. `--output-dir` overrides `DSE_VOCAB_GROWTH_OUTPUT_DIR`, then the checkout's `output/` is the fallback. Fit, sensitivity, sync and upload commands share this rule. The report cache stays at `docs/report/figures/`.

A fit checks at launch and promotion before replacing a complete, clean reporting fit made under another executable signature or at a higher tier. A deliberate refit cycle uses `--replace-model-of-record` or the driver's `-ReplaceModelOfRecord`. Run the cycle from its `fits/YYYY-MM-DD` tag and keep development fits in a separate output root. Full campaigns use a dedicated Linux VM; see the [VM runbook](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/vm-refit.md).

`--trace-persistence` selects `full` (default), `compact` or `minimal`, overriding `DSE_VOCAB_GROWTH_TRACE_PERSISTENCE`. All tiers omit observation-sized deterministic arrays, which `posterior_recompute` can rebuild. `compact` also omits scaled random effects; `minimal` also omits log likelihood and posterior predictive draws. Sampling is unchanged, but recovery scoring, `loso_compare.py` and `regenerate_plots.py` require `full`. Check downstream needs; the manifest records omissions under `artefacts.trace`.

`--nutpie-backend` selects `numba` (default) or `jax`, overriding `DSE_VOCAB_GROWTH_NUTPIE_BACKEND`. JAX is the documented fallback for VG15 `fallback-dispersion` compilation failure on Linux aarch64. The manifest records the runtime backend.

For [parameter recovery](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/runbooks/parameter-recovery.md):

```bash
uv run python scripts/fit_recovery.py headline --config test --replicates 3
```

`headline` selects VG20, VG12 and VG15. `all` uses `recovery.spec.supported_models()`; exclusions are in `UNSUPPORTED_REASONS`. Posterior truth needs a compatible stored fit; prior truth does not. The command resumes checked simulations and converged fits; `--fresh` reruns selected stages. `--set-truth NAME=VALUE` changes a free variable and recomputes derived quantities. VG16 is supported. VG25 needs simulation in wave order. Score only converged replicates; a few replicates cannot establish coverage.

Prepare report assets in this order:

```bash
uv run python scripts/sync_report_figures.py --config rep
uv run python scripts/prepare_report_figures.py
```

Sync validates fits and comparisons before replacing cached figures and tables; it does not copy traces. `--allow-provisional` is for local inspection. Figure preparation can create labelled placeholders, which are not completed results. Public model reports and the comparison book follow the runbook's publication rules. The technical report book is for local validation and review; do not upload it.

## Fit compatibility and provenance

Keep these checks distinct:

- The serialised definition records graph, data, reporting and identity fields. Differences fail unless a tested `fit_identity.BACKFILL_DEFAULTS` entry establishes the exact old default behaviour.
- The prepared-frame hash records rows, values and order. A matching frame can explain a changed raw-data fingerprint when another population's input changed.
- The executable signature hashes the package's Python AST, excluding comments and docstrings, and records numerical-library versions. Resume, strict sync and publication enforce it. Rendering and provisional sync do not; `expected_implementation=None` skips that check.
- Lifecycle, sampling quality and clean source checks establish whether a fit is complete and suitable for its intended use.

For a reviewed code-only change with unchanged numerical libraries, `resume_from_trace.py --allow-implementation-change REASON` records why samples remain valid. It retains original sampling provenance and records current reporting provenance. Publication checks both checkouts for uncommitted changes. Matching definitions and frames alone cannot exclude a changed likelihood.

Trace consumers use `fit_consumers` for definition and data compatibility. `--allow-stale-fit` explicitly reports bypassed checks. New consumers and comparison writers must join their coverage registries or record a justified exemption. Comparison manifests identify contributing fits or input data.

## Model code and writing conventions

Wrappers select definitions and dispatch to shared engines. Put statistical definitions in `models/definitions.py` and engine and reporting declarations in `models/catalogue.py`. Keep engine identity out of the serialised statistical definition. Follow the inventory's registration checklist.

Engines expose `build_model_graph` separately from reporting. Shared helpers build observations, effects, age functions and likelihoods. Most counts use Beta-Binomial likelihoods; signing cross-tabulations use Dirichlet-Multinomial likelihoods. Engines can use different study references and prediction targets. Use `reporting_ages` and `intervals` for reporting policy and [REPORT_STYLE.md](https://github.com/dseinternational/vocabulary-growth/blob/main/docs/models/REPORT_STYLE.md) for report wording.

Figure colours come from `dse_research_utils.plot.styles`, named by role. Words understood, spoken and signed take `CHART_COLOURS[0]`, `[1]` and `[2]`, as `"C0"`, `"C1"` and `"C2"` do. Down syndrome, typically developing and their difference take `[0]`, `[2]` and `[1]`. The design language allows six categorical colours, so a figure with more groups needs an explicit matplotlib palette. Annotation text and reference lines with meaning, such as zero or equality, use `MUTED_TEXT_COLOUR`; `LINE_COLOUR` is for hairlines.

Every Python source file starts with:

```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

Use Ruff for Python style and imports. `E501` and `E741` are ignored. Notebooks use Jupytext percent-format `.py`; `.ipynb` files are gitignored. Use Conventional Commits, such as `docs(report): clarify predictive intervals`. Put issue-closing references in the commit body or pull-request description.

Identify the actual AI tool and model near the top of assisted documents, pull requests, issues and comments. Preserve earlier attribution. For Markdown:

> [!NOTE]
> Drafted or revised with assistance from OpenAI Codex/GPT-6.

For Quarto:

```text
::: {.callout-note}
Drafted or revised with assistance from OpenAI Codex/GPT-6.
:::
```
