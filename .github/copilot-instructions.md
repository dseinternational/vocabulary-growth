# Agent Instructions

> [!NOTE]
> Maintained with assistance from LLM-based AI tools, including OpenAI Codex/GPT-5.

> **Keep in sync:** This file, `CLAUDE.md`, and `.github/copilot-instructions.md` share the same content. When updating one, update all three.

## Project overview

This project is an exploratory study of vocabulary development in children with Down syndrome that aims to characterise observed trajectories of word learning, spoken and gestured production, and relationships between words understood and produced. The primary goal of the study is to provide interpretable statistics that can accurately inform expectations, intervention and teaching practice. We evaluate and fit these models using Bayesian inference to estimate full probability distributions for parameters of interest, using an iterative workflow.

The Python package `vocab_growth` (in `src/vocab_growth/`) defines a series of PyMC models that are fitted to vocabulary assessment data aggregated from multiple international studies. Reports are authored in Quarto (`.qmd`).

This project depends on a sibling repository, `dseinternational/research`, which provides shared utilities via the `dse_research_utils` package. It is installed from the public git tag `v0.12.1` (see [Environment setup](#environment-setup)); a commented local-dev override in `pyproject.toml` lets you point at a sibling `../research/src/python` checkout instead.

## Environment setup

Single-layer [uv](https://docs.astral.sh/uv/) environment (shared across DSE research repos). Create or refresh it with `uv sync`; run anything in it with `uv run …`, which needs no activation.

- **Dependencies**: `pyproject.toml` declares only `dse-research-utils[columnar,graphs,io,jax,notebook,viz]` plus a `dev` dependency group. The scientific stack (`numpy`/`scipy`/`pandas`/`pymc`/`pytensor`/`nutpie`/`arviz`/`preliz`/`xarray`, …) is inherited transitively from the library's own `pyproject.toml`, which is the canonical set of floors — do not restate it here, or the two copies will drift. `nutpie` is deliberately not declared: PyMC auto-selects it as the default NUTS sampler when present.
- **Python**: provisioned by uv from `.python-version` (3.14). No separate Python installation is needed.
- **Exact replication**: `uv.lock` pins every package for `linux-x86_64`, `linux-aarch64`, `macOS-arm64` and `win-amd64`, including the immutable commit of `dse-research-utils`. `uv sync --locked` installs it and fails rather than re-resolving if it is stale. Refresh with `uv lock` only after an intentional dependency change. See `docs/runbooks/environment-locks.md`.
- **Platforms**: Linux, Apple Silicon macOS and **native Windows** (no WSL — `jaxlib` ships win-amd64 wheels on PyPI). Intel macOS is unsupported upstream: numba publishes no macOS x86_64 wheels. On Windows set `PYTHONUTF8=1`: the progress output uses `✓`/`·`, which cp1252 cannot encode. Since `dse-research-utils` v0.11.2 the shared console relaxes a non-UTF-8 `sys.stdout` to `errors="replace"`, so these degrade to `?` rather than killing a fit at its first completed stage — UTF-8 mode is what renders them properly. CI sets it for the fit job.
- **Local dev against research**: comment the `dse-research-utils` git entry in `[tool.uv.sources]` and uncomment the `path = "../research/src/python"` override beside it.
- **GPU**: opt-in overlay (`jax[cuda]`); the locked environment is CPU-only and cross-platform.
- **Not Python packages** (`uv sync` cannot supply these): the Graphviz `dot` binary (model-diagram figure only — skipped with a warning if absent, so it is the one optional tool); Quarto for report rendering, which bundles its own Pandoc, Dart Sass, Deno and Typst and so needs no separate Pandoc install; a XeLaTeX distribution (`quarto install tinytex`) plus the Source Sans 3 / Monaspace Neon fonts for the report book's `pdf` format only; and Node.js for spellcheck and Markdown formatting. `quarto check` reports what it resolved.
- **Node dependencies** (spellcheck, formatting): `npm install`.

## Commands

### Lint

```bash
uv run ruff check src/ scripts/ tests/
```

### Type check

```bash
uv run mypy
```

Deliberately narrow: the four modules that _declare_ things — `models/definitions.py`, `models/catalogue.py`, `models/subject_effects.py` and `analysis_frames.py` — listed in `[tool.mypy]`. That is where an annotation quietly disagreeing with the value does real damage, and turning it on found two: `tau_subj_u_sigma` was annotated `float` while three registered models put an object in it, and `TrivariateModelDefinition.kappa_u` said `KappaPriorParams` while VG14 and VG15 both passed the two-anchor form. The PyMC graph code is **not** covered and should not be until these are stable — PyTensor's tensor algebra is not usefully typed, and the noise would bury findings like those two. CI runs it in the lint job, from a `typecheck` dependency group holding mypy alone.

### Test

```bash
uv run pytest                                  # the fast set (see below)
uv run pytest -m "slow or not slow"            # everything (the union CI's two test jobs cover)
uv run pytest -n auto --dist loadfile          # the fast set, in parallel
uv run pytest tests/test_foo.py                # single file
uv run pytest tests/test_foo.py::test_bar      # single test
```

**A bare `pytest` does not run everything.** `addopts` carries `-m 'not slow'`, so the four modules that do real sampling or real numerical optimisation are deselected — the fast set is about 65 s against seven minutes for the whole suite. The deselected count is printed on every run. CI covers everything as two parallel jobs — `tests-fast` (`-m "not slow"`) and `tests-slow` (`-m slow`), whose union is the whole suite — and skips both, plus the VG01 smoke fit, when a change touches only documentation (notes, report chapters, Markdown), with two carve-outs that always run everything: the three agent-instruction copies, whose agreement a test compares, and `docs/models/**`, which the fit pipeline copies. Locally, `-m "slow or not slow"` selects everything in one run; use it before pushing anything that touches an engine. It is spelt out rather than an empty `-m ""` because pwsh drops the empty string before pytest sees it, so the same command works on Windows.

`pytest-xdist` is in the `dev` group, and CI runs `-n auto --dist loadfile`. Use `--dist loadfile` rather than the default: several modules have module-scoped fixtures that are themselves fits, and per-test distribution rebuilds them on every worker that draws one of their tests.

Two suite-wide behaviours live in `tests/conftest.py`: the matplotlib backend is fixed to Agg before anything imports pyplot, and an autouse fixture silences the fit pipeline's prior-distribution figures and its `describe_all` pass. Both were the bulk of the suite's run time and nothing asserted on either. Mark a test `@pytest.mark.emits_reporting_artefacts` to opt back in — `tests/test_pipeline_reporting_artefacts.py` does, and checks that a real build still produces them. See `notes/202608241530-test-suite-performance.md`.

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
uv run python scripts/prepare_data.py
```

This merges CSV datasets from `data/` into `data/vocab_data_merged.csv` and a DuckDB database at `data/vocabulary.duckdb`.

One source is generated rather than committed by hand:

```bash
uv run python scripts/build_us01_source.py --verify
```

This derives `data/vocab_data_us_01.csv` (the Edgin Down syndrome cohort, `us_01`) from the item-level contributor files in the public `langcog/wordbank` repository, with a provenance manifest. It is not read from `data/wordbank_administration_data.csv`, because Wordbank's by-child download page age-truncates every administration to its instrument's registered window (345 Down syndrome administrations reduced to 194) and cannot separate the four all-blank administrations it scores as zeros. `--verify` checks the in-window rows against the export as a multiset. See [`data/vocab_data_us_01.md`](data/vocab_data_us_01.md). The export is still the source for the typically-developing pool, for which the age filter is appropriate.

### Fit a model

```bash
uv run python scripts/fit_model.py <model_id> [--config <config>] [--render | --render-only] [--upload] [--output-dir <dir>] [--trace-persistence <tier>]
```

- `model_id`: one of `vg01`, `vg02`, `vg03`, `vg04`, `vg05`, `vg07`, `vg08`, `vg09`, `vg10`, `vg11`, `vg12`, `vg13`, `vg14`, `vg15`, `vg16`, `vg19`, `vg20`, `vg21`, `vg22`, `vg23`, or `all`. `all` is derived from `MODEL_REGISTRY` rather than from this list, so it always covers every registered model.
- `--config`: sampling configuration — `dev` (fast, for development), `test`, or `rep` (full reporting quality). Defaults to `dev`.
- `--render`: render the Quarto model output after the completed fit is atomically promoted. A rendering failure leaves the fit complete and available for a later `--render-only` retry.
- `--render-only`: validate and render an existing compatible fit without sampling again.
- `--upload`: upload model output to Azure Blob Storage via AzCopy. Requires `DSERESEARCH_BLOB_CONTAINER_URL` environment variable set to the target container URL.
- `--output-dir`: root directory for model output. Overrides the `DSE_VOCAB_GROWTH_OUTPUT_DIR` environment variable; both fall back to the repository-local `output/`.
- `--trace-persistence`: how much of the trace to keep in `trace.nc` — `full` (default), `compact`, or `minimal`. Overrides the `DSE_VOCAB_GROWTH_TRACE_PERSISTENCE` environment variable. It changes nothing about the posterior. Since 2026-08-23 the observation-sized deterministics (`f_obs`, `p_obs`, `kappa_obs`, their per-outcome counterparts and the concatenated `*_all` grids) are not sampled at any tier: the engines' `sample` stage gives `pm.sample` the `var_names` from `fit_artifacts.sampled_variable_names`, so nutpie never evaluates or stores them — the graph and the draws are unchanged, nothing in the fit pipeline read them, and storing them was what made fit memory scale as `n_obs × draws`. The trace records what was left out in its posterior attributes and the manifest under `artefacts.trace.not_sampled`; a reader that needs one rebuilds the model and recomputes it with `vocab_growth.posterior_recompute` (as `scripts/loso_compare.py` does), and `scripts/kfold_loso.py`, which reads them across every draw of its own fold fits, asks `sample()` to store them. `compact` therefore now drops only the duplicated scaled random effects, which are recomputable from the raw draws and the scales (the 9.8 GB → 3.2 GB measurement on VG10 dates from when the observation-sized variables were still stored). `minimal` additionally drops the stored `log_likelihood` and `posterior_predictive`, which is a real trade rather than a free saving: their consumers run during the fit, but recomputing LOO or a new predictive view later then needs a refit, and `loso_compare.py`, `regenerate_plots.py` and parameter-recovery scoring all need a `full` fit and refuse a compacted one up front, before reading the trace. The tier actually written is recorded in `fit_manifest.json` under `artefacts.trace`, so a missing variable can be told from a truncated file, and fits made before the setting existed are treated as `full`. See `notes/202608081445-trace-persistence-tiers.md`.

Output (traces, figures, summary tables) is written to `<output-root>/models/<model_name>/`. The output root is resolved (highest precedence first) from `--output-dir`, then the `DSE_VOCAB_GROWTH_OUTPUT_DIR` environment variable, then the repository-local `output/` default — so reporting-quality VM runs can redirect the multi-gigabyte traces to a scratch disk without changing the layout. `fit_model.py`, `fit_sensitivity.py`, `sync_report_figures.py`, and `upload.py` all honour the same resolution (`vocab_growth.environment.output_root`), and the disk preflight prints the resolved root. The report figure cache (`docs/report/figures/`, below) always stays in the checkout.

### Run parameter-recovery checks

```bash
uv run python scripts/fit_recovery.py <model|headline|all> [--config <config>] [--replicates <n>] [--truth posterior|prior] [--simulate-only | --fit-only | --compare-only] [--output-dir <dir>]
```

Simulates a dataset from a model at a known parameter draw, refits the model to it with the engine's own pipeline, and scores the recovered posterior against the truth. `headline` is `vg20`, `vg12`, `vg15`; `all` is every supported model (`vg07`-`vg13`, `vg15`, `vg19`, `vg20` — VG16 is excluded because its cross-lag predictor is a function of the outcome). Truth defaults to the model of record's posterior (requires a fitted model of record); `--truth prior` needs no trace but tests parameter settings far from the reported regime. Recovery fits land in `<output-root>/models/<model_id>-<config>-recovery-rNN/` and never touch a model of record; tables land in `<output-root>/comparisons/recovery/`. A replicate is only assessed if its fit's convergence is confirmed. See `docs/runbooks/parameter-recovery.md`, including what a handful of replicates can and cannot establish.

### Sync report figures

```bash
uv run python scripts/sync_report_figures.py [--config <config>] [--output-dir <dir>]
```

Validates the model definition, sampling configuration, raw-data fingerprint, exact prepared-frame hash, complete lifecycle state, reporting quality, clean fit provenance and rendered model report before atomically replacing cached plots (`.svg`/`.png`) and summary tables (`.csv`) from the output root's `models/` and `comparisons/` in `docs/report/figures/` (gitignored), which is the only source the Quarto report reads. Comparison outputs are validated too, against the `comparison_manifest.json` their generating script writes: a contributing fit that has been refitted since the comparison was generated fails the sync, and comparison files that no manifest entry claims are reported as warnings while the manifest is adopted script by script. `--allow-provisional` keeps lifecycle/model/sampling checks for local dev/test work while relaxing publication provenance, the prepared-frame check and comparison provenance. Traces (`.nc`) are excluded. Run after fitting models or regenerating comparisons, before rendering the report.

## Architecture

### Data pipeline

1. Raw study data lives in `data/` as CSVs (one per study, e.g. `vocab_data_uk_01.csv`).
2. `scripts/prepare_data.py` merges and harmonises them into a DuckDB database with a unified `vocab_combined` view.
3. Model code loads data via `vocab_growth.data_utils.load_combined_data()`.

Both loader paths return rows in a **deterministic order** (sorted on every column before the masking rules run). Nothing statistical depends on it — the likelihoods are order-invariant, and the sort changes the order only, not the content — but the fit manifest records `data.analysis_frame_hash`, an exact hash of the prepared frame including its row order, and that hash is what tells a stale posterior from a current one. The loader queries carry no `ORDER BY`, so without the sort the hash followed the DuckDB scan order and could not be recomputed for validation. Every registered model's prepared frame can be rebuilt outside a fit through `vocab_growth.analysis_frames`, which is how `fit_model.py`, `sync_report_figures.py` and `compare_models.py` now check that a fit's frame still matches the one the current loader rules produce. A change to a masking or exclusion rule therefore invalidates fitted output even though the raw CSVs are untouched, which is the point: the raw-data fingerprint alone cannot see rule changes, because the rules run in Python after the CSVs are read. The converse holds too: since 2026-08-31 a matching frame hash **excuses** a raw-data fingerprint mismatch, because the fingerprint hashes every CSV in `data/` while a model consumes the raw data only through its own prepared frame — so new data for one population (say, a new Down syndrome study CSV) no longer stales the other population's fits; the fingerprint stays a hard failure wherever the frame hash is unavailable or also differs.

The definition itself is compared **field by field through a classified payload** (`vocab_growth.models.fit_identity`), not by raw dictionary equality. Every field of every registered definition class is classified as graph-affecting, data-affecting, reporting or identity; the classification is complete (a test checks it against the registry) and fails closed (an unclassified field is treated as graph-affecting). **Every difference is still fatal**, reporting and identity ones included — what the classification adds is a failure message saying what kind of thing moved, and one documented excuse: `BACKFILL_DEFAULTS` names fields whose _absence_ from an older manifest is equivalent to a stated value, which is a claim that every fit made before that field existed behaved exactly as a fit with the field set to it. Without that mechanism, adding a field with a default invalidates every historical fit of its dataclass even when the default reproduces what those fits did — the constraint that pushed VG19's child slope and Proposal A1's age-varying scale into a scalar field holding an object, and VG20's correlation and VG22's factor onto sibling subclasses. The registry starts empty: nothing needs backfilling yet, and the first entry belongs with the change that adds the field it excuses. Whether a _reporting-only_ difference should stop a fit being published is a separate decision and has not been made — a changed `ages_query` leaves the stored query outputs describing ages the report no longer asks for.

The Down syndrome pool masks or drops several documented defect classes by default, each with a reinstatement flag for sensitivity analysis: partial administrations, duplicated outcome columns, implausible production (near-ceiling and longitudinal-collapse signatures), production counts contradicted by a same-day administration on another form, administrations given below their form's lowest registered age, children recorded only at their form's ceiling, and comprehension counts that fall below the child's own production count. Read the governing constant's docstring before reinstating any of them. Two are worth knowing about even if you never touch them: administrations _above_ a form's age window are deliberately **admitted**, because for a Down syndrome cohort an early-vocabulary form given to an older child is developmentally appropriate and those rows are `us_01`'s only comprehension observations between 19 and 27 months; and the ceiling-saturated preparation batch is identified by `CEILING_ONLY_CHILD_STUDIES` on the _provenance_ criterion that the affected children have no non-ceiling record, because age and count together cannot separate it from a legitimately able older child. A third is worth knowing because its denominator is easy to get wrong: the comprehension-below-production rule compares `understood` against the recorded `produced` **union**, not against `spoken + signed`. The two modalities overlap wherever a child both says and signs a word, so the sum overstates distinct production badly (`uk_07`: 77 of 82 rows) and would flag 87 administrations instead of 10. It masks the comprehension count only, keeps the row, and keeps equality — a child who produces everything they understand is legitimate.

The typically-developing reference pool is drawn from Wordbank and scoped by language. It defaults to `ENGLISH_LANGUAGES`; the hierarchical models (VG11, VG12, VG13) use `ENGLISH_AND_ROMANCE_LANGUAGES`, adding Italian and Spanish (European) so the Down-syndrome-versus-typically-developing comparison spans several languages on both sides — the Down syndrome pool is already a quarter non-English. VG03/VG04 stay English-only: they carry no random effects to absorb between-language variation. The scope is a model-definition field (`td_languages`), so it is part of the model graph and changing it requires a refit. Admission criteria and the two measurement checks are on `ROMANCE_LANGUAGES` in `src/vocab_growth/models/definitions.py`.

### Model structure

Each model is a self-contained module in `src/vocab_growth/models/model_vgNN.py`. All follow the same pattern:

- A `ModelConfiguration` dataclass defines the prior distributions and model hyperparameters.
- A `ModelFitContext` dataclass carries state through the fitting pipeline.
- A top-level `fit(config)` function orchestrates the full pipeline: data prep → model build → prior predictive checks → MCMC sampling → diagnostics → posterior predictive → summary → plots → report.
- Models use **PyMC** with the **nutpie** sampler and **HSGP** (Hilbert-Space Gaussian Process) approximations for scalable nonparametric mean functions.
- The likelihood is **Beta-Binomial** with age-varying dispersion.

The full, canonical list of models -- each model's population, outcome, structure, and purpose -- is maintained in `docs/models/README.md`. Treat that inventory as the single source of truth: consult it for the current set of models, and update it whenever a model is added, removed, or changed.

There are currently twenty registered models (`VG01`-`VG16` and `VG19`-`VG23`, with retired `VG06` omitted and `VG17`/`VG18` reserved for the exploratory sign-group modules), spanning the Down syndrome and typically-developing populations across single-outcome, joint (understood + spoken), signing (understood + spoken + signed), cross-lag, correlated-random-effect, child-slope and low-rank-factor structures.

### Registering a model

Registering a model takes two entries and nothing else: the statistical definition in `definitions.py` (added to `MODEL_REGISTRY`) and a `RegisteredModel` record in `src/vocab_growth/models/catalogue.py` naming the engine that fits it. Everything per-model that is not part of the statistical definition -- the analysis-frame builder, the prior-predictive hook and its calling convention, the plot hook, the pipeline stage factory, the wrapper module and the report template -- follows from that record. `FRAME_BUILDERS`, `regenerate_plots.py`'s engine tables, `prior_predictive_audit.py`'s dispatch, both sensitivity scripts' runner map and `recovery/spec.py`'s stage factory are all derived from it, so there is no second model-ID list to update.

The catalogue is deliberately **outside** the serialised statistical definition. A fit is validated by comparing the manifest's recorded definition field for field, so adding a field to a definition dataclass invalidates every existing fit of that class; engine identity and the reporting hooks must be free to change without a refit. Engine identity is **declared, not inferred from the definition class**: VG05 and VG07 share `BivariateModelDefinition` and run on different engines. `tests/test_model_catalogue.py` pins every declaration against the code it describes -- the wrapper's own import, each hook's signature, the report template's existence -- and `tests/test_report_cells.py` pins that every sampled parameter family is either rendered in the priors table or exempt with a recorded reason. Before the catalogue existed the same engine assignment was written in seven places and three of them were wrong at once (issue #273, `notes/202608311230-model-catalogue.md`).

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
