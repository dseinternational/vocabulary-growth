# Agent Instructions

> [!NOTE]
> Maintained with assistance from LLM-based AI tools, including OpenAI Codex/GPT-5.

> **Keep in sync:** This file, `CLAUDE.md`, and `.github/copilot-instructions.md` share the same content. When updating one, update all three.

## Project overview

This project is an exploratory study of vocabulary development in children with Down syndrome that aims to characterise observed trajectories of word learning, spoken and gestured production, and relationships between words understood and produced. The primary goal of the study is to provide interpretable statistics that can accurately inform expectations, intervention and teaching practice. We evaluate and fit these models using Bayesian inference to estimate full probability distributions for parameters of interest, using an iterative workflow.

The Python package `vocab_growth` (in `src/vocab_growth/`) defines a series of PyMC models that are fitted to vocabulary assessment data aggregated from multiple international studies. Reports are authored in Quarto (`.qmd`).

This project depends on a sibling repository, `dseinternational/research`, which provides shared utilities via the `dse_research_utils` package. It is installed from the public git tag `v0.7.1` (see [Environment setup](#environment-setup)); a commented local-dev override in `environment.yml` lets you point at a sibling `../research/src/python` checkout instead.

## Environment setup

Hybrid two-layer environment (shared across DSE research repos):

- **Compiled core** — the scientific stack (`numpy`/`scipy`/`pandas`/`pymc`/`nutpie`/`jax`/`arviz`, …) comes from **conda-forge** and must match the canonical spec shipped in `dse-research-utils` (`data/environment-core.yml`) so it cannot drift across repos. Verify with `dse-check-env environment.yml`.
- **Pip layer** — the pure-Python tail and the shared library. `dse-research-utils` installs from the public git tag `v0.7.1` (`dse-research-utils[viz,notebook,io] @ git+https://github.com/dseinternational/research.git@v0.7.1#subdirectory=src/python`); the package itself installs editable (`-e ./`).

- **Python environment**: Conda/mamba (environment name `dse-vocab-growth`), Python 3.14, channel `conda-forge`. Create with `mamba env create -f environment.yml`; update with `conda env update -f environment.yml`.
- **Exact replication**: `conda-lock.yml` pins the compiled environment for `linux-64` and `osx-arm64`; `requirements-pip.lock` pins the pip layer. See `docs/runbooks/environment-locks.md`. Refresh both with `scripts/lock_environment.py` only after an intentional dependency change.
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

One source is generated rather than committed by hand:

```bash
python scripts/build_us01_source.py --verify
```

This derives `data/vocab_data_us_01.csv` (the Edgin Down syndrome cohort, `us_01`) from the item-level contributor files in the public `langcog/wordbank` repository, with a provenance manifest. It is not read from `data/wordbank_administration_data.csv`, because Wordbank's by-child download page age-truncates every administration to its instrument's registered window (345 Down syndrome administrations reduced to 194) and cannot separate the four all-blank administrations it scores as zeros. `--verify` checks the in-window rows against the export as a multiset. See [`data/vocab_data_us_01.md`](data/vocab_data_us_01.md). The export is still the source for the typically-developing pool, for which the age filter is appropriate.

### Fit a model

```bash
python scripts/fit_model.py <model_id> [--config <config>] [--render | --render-only] [--upload] [--output-dir <dir>] [--trace-persistence <tier>]
```

- `model_id`: one of `vg01`, `vg02`, `vg03`, `vg04`, `vg05`, `vg07`, `vg08`, `vg09`, `vg10`, `vg11`, `vg12`, `vg13`, `vg14`, `vg15`, `vg16`, `vg20`, or `all`.
- `--config`: sampling configuration — `dev` (fast, for development), `test`, or `rep` (full reporting quality). Defaults to `dev`.
- `--render`: render the Quarto model output after the completed fit is atomically promoted. A rendering failure leaves the fit complete and available for a later `--render-only` retry.
- `--render-only`: validate and render an existing compatible fit without sampling again.
- `--upload`: upload model output to Azure Blob Storage via AzCopy. Requires `DSERESEARCH_BLOB_CONTAINER_URL` environment variable set to the target container URL.
- `--output-dir`: root directory for model output. Overrides the `DSE_VOCAB_GROWTH_OUTPUT_DIR` environment variable; both fall back to the repository-local `output/`.
- `--trace-persistence`: how much of the trace to keep in `trace.nc` — `full` (default), `compact`, or `minimal`. Overrides the `DSE_VOCAB_GROWTH_TRACE_PERSISTENCE` environment variable. It changes nothing about the posterior: `compact` drops the observation-sized deterministics (`f_obs`, `p_obs`, `kappa_obs`, the concatenated `*_all` grids) and the duplicated scaled random effects, all of which are recomputable from the free parameters — measured at 9.8 GB → 3.2 GB on VG10, with byte-identical reporting output. `minimal` additionally drops the stored `log_likelihood` and `posterior_predictive`, which is a real trade rather than a free saving: their consumers run during the fit, but recomputing LOO or a new predictive view later then needs a refit, and `kfold_loso.py` / `loso_compare.py`, `regenerate_plots.py` and parameter-recovery scoring all need a `full` fit and refuse a compacted one up front, before reading the trace. (`kfold_loso.py` is unaffected: it fits its own folds and reads the in-memory trace, never `trace.nc`.) The tier actually written is recorded in `fit_manifest.json` under `artefacts.trace`, so a missing variable can be told from a truncated file, and fits made before the setting existed are treated as `full`. See `notes/202608081445-trace-persistence-tiers.md`.

Output (traces, figures, summary tables) is written to `<output-root>/models/<model_name>/`. The output root is resolved (highest precedence first) from `--output-dir`, then the `DSE_VOCAB_GROWTH_OUTPUT_DIR` environment variable, then the repository-local `output/` default — so reporting-quality VM runs can redirect the multi-gigabyte traces to a scratch disk without changing the layout. `fit_model.py`, `fit_sensitivity.py`, `sync_report_figures.py`, and `upload.py` all honour the same resolution (`vocab_growth.environment.output_root`), and the disk preflight prints the resolved root. The report figure cache (`docs/report/figures/`, below) always stays in the checkout.

### Run parameter-recovery checks

```bash
python scripts/fit_recovery.py <model|headline|all> [--config <config>] [--replicates <n>] [--truth posterior|prior] [--simulate-only | --fit-only | --compare-only] [--output-dir <dir>]
```

Simulates a dataset from a model at a known parameter draw, refits the model to it with the engine's own pipeline, and scores the recovered posterior against the truth. `headline` is `vg20`, `vg12`, `vg15`; `all` is every supported model (`vg07`-`vg13`, `vg15`, `vg20` — VG16 is excluded because its cross-lag predictor is a function of the outcome). Truth defaults to the model of record's posterior (requires a fitted model of record); `--truth prior` needs no trace but tests parameter settings far from the reported regime. Recovery fits land in `<output-root>/models/<model_id>-<config>-recovery-rNN/` and never touch a model of record; tables land in `<output-root>/comparisons/recovery/`. A replicate is only assessed if its fit's convergence is confirmed. See `docs/runbooks/parameter-recovery.md`, including what a handful of replicates can and cannot establish.

### Sync report figures

```bash
python scripts/sync_report_figures.py [--config <config>] [--output-dir <dir>]
```

Validates the model definition, sampling configuration, raw-data fingerprint, complete lifecycle state, reporting quality, clean fit provenance and rendered model report before atomically replacing cached plots (`.svg`/`.png`) and summary tables (`.csv`) from the output root's `models/` and `comparisons/` in `docs/report/figures/` (gitignored), which is the only source the Quarto report reads. `--allow-provisional` keeps lifecycle/model/sampling checks for local dev/test work while relaxing publication provenance. Traces (`.nc`) are excluded. Run after fitting models or regenerating comparisons, before rendering the report.

## Architecture

### Data pipeline

1. Raw study data lives in `data/` as CSVs (one per study, e.g. `vocab_data_uk_01.csv`).
2. `scripts/prepare_data.py` merges and harmonises them into a DuckDB database with a unified `vocab_combined` view.
3. Model code loads data via `vocab_growth.data_utils.load_combined_data()`.

The Down syndrome pool masks or drops several documented defect classes by default, each with a reinstatement flag for sensitivity analysis: partial administrations, duplicated outcome columns, implausible production (near-ceiling and longitudinal-collapse signatures), administrations given below their form's lowest registered age, and children recorded only at their form's ceiling. Read the governing constant's docstring before reinstating any of them. Two are worth knowing about even if you never touch them: administrations _above_ a form's age window are deliberately **admitted**, because for a Down syndrome cohort an early-vocabulary form given to an older child is developmentally appropriate and those rows are `us_01`'s only comprehension observations between 19 and 27 months; and the ceiling-saturated preparation batch is identified by `CEILING_ONLY_CHILD_STUDIES` on the _provenance_ criterion that the affected children have no non-ceiling record, because age and count together cannot separate it from a legitimately able older child.

The typically-developing reference pool is drawn from Wordbank and scoped by language. It defaults to `ENGLISH_LANGUAGES`; the hierarchical models (VG11, VG12, VG13) use `ENGLISH_AND_ROMANCE_LANGUAGES`, adding Italian and Spanish (European) so the Down-syndrome-versus-typically-developing comparison spans several languages on both sides — the Down syndrome pool is already a quarter non-English. VG03/VG04 stay English-only: they carry no random effects to absorb between-language variation. The scope is a model-definition field (`td_languages`), so it is part of the model graph and changing it requires a refit. Admission criteria and the two measurement checks are on `ROMANCE_LANGUAGES` in `src/vocab_growth/models/definitions.py`.

### Model structure

Each model is a self-contained module in `src/vocab_growth/models/model_vgNN.py`. All follow the same pattern:

- A `ModelConfiguration` dataclass defines the prior distributions and model hyperparameters.
- A `ModelFitContext` dataclass carries state through the fitting pipeline.
- A top-level `fit(config)` function orchestrates the full pipeline: data prep → model build → prior predictive checks → MCMC sampling → diagnostics → posterior predictive → summary → plots → report.
- Models use **PyMC** with the **nutpie** sampler and **HSGP** (Hilbert-Space Gaussian Process) approximations for scalable nonparametric mean functions.
- The likelihood is **Beta-Binomial** with age-varying dispersion.

The full, canonical list of models -- each model's population, outcome, structure, and purpose -- is maintained in `docs/models/README.md`. Treat that inventory as the single source of truth: consult it for the current set of models, and update it whenever a model is added, removed, or changed.

There are currently sixteen registered models (`VG01`-`VG16` and `VG20`, with retired `VG06` omitted and `VG17`-`VG19` reserved), spanning the Down syndrome and typically-developing populations across single-outcome, joint (understood + spoken), signing (understood + spoken + signed), cross-lag and correlated-random-effect structures.

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
