# Vocabulary growth in children with Down syndrome

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6.

> [!WARNING]
> This study is in progress. Models and findings are preliminary.

This study describes how vocabulary develops in children with Down syndrome. It brings together parent-reported checklist data from several countries to estimate words understood, spoken and signed at different ages, and the variation between children.

The aim is to give families, teachers and practitioners evidence to help interpret vocabulary development. The estimates describe groups of children. They do not set targets for an individual child or show which teaching methods cause better outcomes.

## Study approach

We fit Bayesian statistical models, which express uncertainty as probability distributions. The models allow vocabulary growth to vary with age and account for differences between studies and children. Joint models describe spoken and signed words as proportions of words understood.

The typically developing comparison data come from Wordbank. All models report on a common 810-word reference scale. Differences between checklists and limited follow-up remain important constraints on interpretation.

The study examines:

- Vocabulary counts and rates of growth at different ages.
- The spread of counts among children of the same age.
- How much of their understood vocabulary children also say or sign.
- How these relationships compare with those in typically developing children.

## Reading the project

| Start here                                                   | Purpose                                                             |
| ------------------------------------------------------------ | ------------------------------------------------------------------- |
| [Technical report](docs/report/index.qmd)                    | Methods and report chapters; some findings chapters are unfinished. |
| [Plain-language summary](docs/summary/README.md)             | Unfinished outline for families and practitioners.                  |
| [Model inventory](docs/models/README.md)                     | Model structures, reporting roles and links to individual reports.  |
| [Data guide](data/readme.md)                                 | Sources, variables and preparation.                                 |
| [Notes index](notes/README.md)                               | Dated analyses, decisions and run records.                          |
| [Code walkthrough](docs/tutorials/model-code-walkthrough.md) | A worked introduction to the model code.                            |

Read numerical results with their fit date, model definition and caveats. Older notes can explain a decision without describing the current data or fit.

## Getting started

Clone the repository and install the locked environment from its root:

```bash
git clone https://github.com/dseinternational/vocabulary-growth.git
cd vocabulary-growth
uv sync --locked
uv run python scripts/prepare_data.py
```

`uv` supplies Python and the project packages. The lock supports Linux, Apple Silicon macOS and native Windows. On Windows, set `PYTHONUTF8=1` to display progress symbols correctly. See [environment setup](docs/runbooks/environment-locks.md) for dependency changes and external tools.

Run a short development fit:

```bash
uv run python scripts/fit_model.py vg01 --config dev
```

Quarto renders reports. Graphviz supplies model diagrams. The report book's PDF format also needs LaTeX and its specified fonts. For a reporting-quality run, follow the [full-refit runbook](docs/runbooks/full-refit.md).

For code and documentation checks, see [AGENTS.md](AGENTS.md). Node dependencies are installed with `npm ci`.

## Contributing

We welcome contributions of data, statistical models, code and interpretation. Read [the contribution guide](CONTRIBUTING.md) before sharing participant data.

## Licences

- Source code uses GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See [LICENSE](LICENSE) and the source headers.
- Documentation, reports and papers use Creative Commons Attribution 4.0 International (CC BY 4.0). See [docs/LICENSE](docs/LICENSE).
- Data use CC BY 4.0. See [data/LICENSE](data/LICENSE) and the source records.
