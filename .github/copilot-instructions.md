# Copilot Instructions

> **Keep in sync:** This file, `CLAUDE.md`, and `AGENTS.md` share the same content. When updating one, update all three.

## Project overview

This project is an exploratory study of vocabulary development in children with Down syndrome that aims to characterise observed trajectories of word learning, spoken and gestured production, and relationships between words understood and produced. The primary goal of the study is to provide interpretable statistics that can accurately inform expectations, intervention and teaching practice. We evaluate and fit these models using Bayesian inference to estimate full probability distributions for parameters of interest, using an iterative workflow.

The Python package `vocab_growth` (in `src/vocab_growth/`) defines a series of PyMC models that are fitted to vocabulary assessment data aggregated from multiple international studies. Reports are authored in Quarto (`.qmd`).

This project depends on a sibling repository, `dseinternational/research`, which provides shared utilities via the `dse_research_utils` package. During local development it is installed as an editable package from `../research/src/python`.

## Environment setup

- **Python environment**: Conda (environment name `dse-vocab-growth`), Python 3.14, channels: `conda-forge`. Install/update with `conda env update -f environment.yml`.
- **Node dependencies** (spellcheck, formatting): `npm install`.
- The package itself is installed in editable mode (`-e ./`) via the Conda environment.

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
python scripts/fit_model.py <model_id> [--config <config>] [--render] [--upload]
```

- `model_id`: one of `vg01`, `vg02`, `vg03`, `vg04`, `vg05`, `vg06`, `vg07`, `vg08`, `vg09`, `vg10`, `vg11`, `vg12`, `vg13`, `vg14`, `vg15`, or `all`.
- `--config`: sampling configuration — `dev` (fast, for development), `test`, or `rep` (full reporting quality). Defaults to `dev`.
- `--render`: render the Quarto model output after fitting.
- `--upload`: upload model output to Azure Blob Storage via AzCopy. Requires `DSERESEARCH_BLOB_CONTAINER_URL` environment variable set to the target container URL.

Output (traces, figures, summary tables) is written to `output/models/<model_name>/`.

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

The models differ in which outcome, population, and structure they target:

| Model | Outcome                                               | Population           | Notes                                                                                                                                                                                                                                                                                                                          |
| ----- | ----------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| VG01  | Words spoken                                          | Down syndrome        |                                                                                                                                                                                                                                                                                                                                |
| VG02  | Words understood                                      | Down syndrome        |                                                                                                                                                                                                                                                                                                                                |
| VG03  | Words spoken                                          | Typically developing |                                                                                                                                                                                                                                                                                                                                |
| VG04  | Words understood                                      | Typically developing |                                                                                                                                                                                                                                                                                                                                |
| VG05  | Words understood + spoken (joint)                     | Down syndrome        |                                                                                                                                                                                                                                                                                                                                |
| VG06  | Words understood + spoken (joint)                     | Typically developing |                                                                                                                                                                                                                                                                                                                                |
| VG07  | Words understood + spoken (joint)                     | Down syndrome        | Study random intercepts                                                                                                                                                                                                                                                                                                        |
| VG08  | Words understood + spoken (joint)                     | Down syndrome        | Study random intercepts + subject random intercepts on understood                                                                                                                                                                                                                                                              |
| VG09  | Words understood + spoken (joint)                     | Down syndrome        | Study random intercepts + subject random intercepts on understood and on production ratio                                                                                                                                                                                                                                      |
| VG10  | Words understood + spoken (joint)                     | Down syndrome        | Study random intercepts + subject random intercepts on understood and on production ratio + tighter q-anchor priors + GP anchored at reference age                                                                                                                                                                             |
| VG11  | Words spoken                                          | Typically developing | Dataset-level study random intercepts + GP anchor at 19 months                                                                                                                                                                                                                                                                 |
| VG12  | Words understood                                      | Typically developing | Dataset-level study random intercepts + GP anchor at 19 months                                                                                                                                                                                                                                                                 |
| VG13  | Words understood + spoken (joint)                     | Typically developing | Ages 8–18 months only; dataset-level study random intercepts + GP anchor at 13 months                                                                                                                                                                                                                                          |
| VG14  | Words understood + spoken + signed (joint)            | Down syndrome        | Adds signing as a third production ratio `r(a)`; `p_Sign = p_U · r`; total expressive `p_any` derived assuming sign/speech independence given age                                                                                                                                                                              |
| VG15  | Words understood + spoken + signed (joint, four-cell) | Down syndrome        | Within-understood sign-speech association `psi` (Plackett copula, identified from the uk_02 four-cell cross-tab) + study **and subject** random intercepts on all three trajectories; VG10 stabilisation (tighter `q` anchors + per-draw GP anchor at 54 mo); data-identified `p_any` replaces VG14's independence upper bound |

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
