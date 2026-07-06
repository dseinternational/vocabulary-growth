# Agent Instructions

> **Keep in sync:** This file, `CLAUDE.md`, and `.github/copilot-instructions.md` share the same content. When updating one, update all three.

## Project overview

This project is an exploratory study of vocabulary development in children with Down syndrome that aims to characterise observed trajectories of word learning, spoken and gestured production, and relationships between words understood and produced. The primary goal of the study is to provide interpretable statistics that can accurately inform expectations, intervention and teaching practice. We evaluate and fit these models using Bayesian inference to estimate full probability distributions for parameters of interest, using an iterative workflow.

The Python package `vocab_growth` (in `src/vocab_growth/`) defines a series of PyMC models that are fitted to vocabulary assessment data aggregated from multiple international studies. Reports are authored in Quarto (`.qmd`).

This project depends on a sibling repository, `dseinternational/research`, which provides shared utilities via the `dse_research_utils` package. It is installed from the public git tag `v0.5.0` (see [Environment setup](#environment-setup)); a commented local-dev override in `environment.yml` lets you point at a sibling `../research/src/python` checkout instead.

## Environment setup

Hybrid two-layer environment (shared across DSE research repos):

- **Compiled core** — the scientific stack (`numpy`/`scipy`/`pandas`/`pymc`/`nutpie`/`jax`/`arviz`, …) comes from **conda-forge** and must match the canonical spec shipped in `dse-research-utils` (`data/environment-core.yml`) so it cannot drift across repos. Verify with `dse-check-env environment.yml`.
- **Pip layer** — the pure-Python tail and the shared library. `dse-research-utils` installs from the public git tag `v0.5.0` (`dse-research-utils[viz,notebook,io] @ git+https://github.com/dseinternational/research.git@v0.5.0#subdirectory=src/python`); the package itself installs editable (`-e ./`).

- **Python environment**: Conda/mamba (environment name `dse-vocab-growth`), Python 3.14, channel `conda-forge`. Create with `mamba env create -f environment.yml`; update with `conda env update -f environment.yml`.
- **Windows**: there is no conda-forge `jax`/`jaxlib` win-64 build, so the stack cannot solve natively — use **WSL** (Ubuntu, linux-64).
- **Local dev against research**: comment the `dse-research-utils[...] @ git+…` line in `environment.yml`'s pip block and uncomment the `-e ../research/src/python[...]` override.
- **GPU**: opt-in overlay (`jax[cuda]`); the base env is CPU-only and cross-platform.
- **Node dependencies** (spellcheck, formatting): `npm install`.

## Commands

### Lint

```bash
ruff check src/ scripts/
```

### Test

```bash
pytest              # run all tests
pytest tests/test_foo.py           # single file
pytest tests/test_foo.py::test_bar # single test
```

### Spellcheck (Markdown/Quarto docs)

```bash
npm run spellcheck
```

### Format Markdown

```bash
npm run format         # rewrite files in place
npm run format:check   # check only; fails if any file needs formatting (CI)
```

Uses Prettier. Configured in `.prettierrc.json`; ignore patterns in `.prettierignore`. `proseWrap: "preserve"` so existing line breaks are kept; tables are auto-aligned.

### Prepare data

```bash
python scripts/prepare_data.py
```

This merges CSV datasets from `data/` into `data/vocab_data_merged.csv` and a DuckDB database at `data/vocabulary.duckdb`.

### Fit a model

```bash
python scripts/fit_model.py <model_id> [--config <config>] [--render] [--upload] [--output-dir <dir>]
```

- `model_id`: one of `vg01`, `vg02`, `vg03`, `vg04`, `vg05`, `vg07`, `vg08`, `vg09`, `vg10`, `vg11`, `vg12`, `vg13`, `vg14`, `vg15`, `vg16`, or `all`.
- `--config`: sampling configuration — `dev` (fast, for development), `test`, or `rep` (full reporting quality). Defaults to `dev`.
- `--render`: render the Quarto model output after fitting.
- `--upload`: upload model output to Azure Blob Storage via AzCopy. Requires `DSERESEARCH_BLOB_CONTAINER_URL` environment variable set to the target container URL.
- `--output-dir`: root directory for model output. Overrides the `DSE_VOCAB_GROWTH_OUTPUT_DIR` environment variable; both fall back to the repository-local `output/`.

Output (traces, figures, summary tables) is written to `<output-root>/models/<model_name>/`. The output root is resolved (highest precedence first) from `--output-dir`, then the `DSE_VOCAB_GROWTH_OUTPUT_DIR` environment variable, then the repository-local `output/` default — so reporting-quality VM runs can redirect the multi-gigabyte traces to a scratch disk without changing the layout. `fit_model.py`, `fit_sensitivity.py`, `sync_report_figures.py`, and `upload.py` all honour the same resolution (`vocab_growth.environment.output_root`), and the disk preflight prints the resolved root. The report figure cache (`docs/report/figures/`, below) always stays in the checkout.

### Sync report figures

```bash
python scripts/sync_report_figures.py [--output-dir <dir>]
```

Copies the plots (`.svg`/`.png`) and summary tables (`.csv`) from the output root's `models/` and `comparisons/` (same resolution as above) into `docs/report/figures/` (gitignored), which is the only source the Quarto report reads. Traces (`.nc`) are excluded. Run after fitting models or regenerating comparisons, before rendering the report.

## Architecture

### Data pipeline

1. Raw study data lives in `data/` as CSVs (one per study, e.g. `vocab_data_uk_01.csv`).
2. `scripts/prepare_data.py` merges and harmonises them into a DuckDB database with a unified `vocab_combined` view.
3. Model code loads data via `vocab_growth.data_utils.load_combined_data()`.

### Model structure

Each model is a self-contained module in `src/vocab_growth/models/model_vgNN.py`. All follow the same pattern:

- A `ModelConfiguration` dataclass defines the prior distributions and model hyperparameters.
- A `ModelFitContext` dataclass carries state through the fitting pipeline.
- A top-level `fit(config)` function orchestrates the full pipeline: data prep → model build → prior predictive checks → MCMC sampling → diagnostics → posterior predictive → summary → plots → report.
- Models use **PyMC** with the **nutpie** sampler and **HSGP** (Hilbert-Space Gaussian Process) approximations for scalable nonparametric mean functions.
- The likelihood is **Beta-Binomial** with age-varying dispersion.

The full, canonical list of models -- each model's population, outcome, structure, and purpose -- is maintained in `docs/models/README.md`. Treat that inventory as the single source of truth: consult it for the current set of models, and update it whenever a model is added, removed, or changed.

There are currently sixteen models (`VG01`-`VG16`), spanning the Down syndrome and typically-developing populations across single-outcome, joint (understood + spoken), signing (understood + spoken + signed), and cross-lag structures.

### Shared utilities (`dse_research_utils`)

Plotting styles, sampling configurations, MCMC diagnostics, and reporting helpers come from the sibling `research` repository. Import paths start with `dse_research_utils.*`.

### Reports

Quarto documents in `docs/models/vgNN/index.qmd` render model-specific reports. These embed Python code cells that load fitted output.

## Conventions

### AI tool attribution

Content drafted or generated with the help of an LLM-based AI tool **must** carry a clearly visible label identifying the tool, placed at the top of the content. Substitute the actual tool and model that produced it — replace `Claude Code/Opus 4.8` with whichever assistant was used (for example, `GitHub Copilot`).

For GitHub-rendered content — Markdown files, pull request and issue descriptions, and comments on pull requests and issues — use a GitHub-flavoured Markdown alert:

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

For Quarto documents (`.qmd`), GitHub alert syntax does not render, so use a Quarto callout block instead:

```
::: {.callout-note}
Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).
:::
```

This requirement applies to:

- Document drafts (Markdown and Quarto `.qmd`)
- Pull request descriptions
- Issue descriptions
- Comments on pull requests
- Comments on issues

### File headers

Every Python source file starts with:

```python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
```

### Code style

- **Ruff** for linting and import sorting (config in `pyproject.toml`).
- `E501` (line length) and `E741` (ambiguous variable names) are intentionally ignored — mathematical/statistical variable names like `X`, `y`, `p`, `f`, `eta`, `kappa`, `ell` are standard and expected.
- Notebooks use **Jupytext** percent format (paired `.ipynb` + `.py`). The `.ipynb` files are gitignored; only the `.py` percent-format files are committed.

### Spelling

CSpell is configured in `.cspell.config.yaml` (British English). Custom allowed words are in `config/spellcheck/allow-en.txt`.

### Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): a `<type>(optional scope): <summary>` subject line in the imperative mood, with any detail and rationale in the body. Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`. Examples: `feat(vg16): add a within-child cross-lag`, `fix(data): tolerate a missing nz_01 source CSV`, `docs(report): add the words-understood-spoken chapter`. Reference the issue a commit or pull request closes (`Closes #123`) in the body or pull-request description.

### Writing Markdown

When generating Markdown — `notes/` entries and documents, and especially pull request and issue descriptions and comments — do not insert superfluous line breaks. Write each paragraph as one continuous line and let it reflow; do not hard-wrap prose at a fixed column, and avoid stray blank lines. Prettier is configured with `proseWrap: "preserve"`, so it will **not** rewrap prose for you, and pull-request / issue text is not run through Prettier at all — hard-wrapped paragraphs therefore render as awkward mid-sentence breaks on GitHub and stay that way.
