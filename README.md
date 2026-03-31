# Vocabulary growth in children with Down syndrome

> [!WARNING]
> This is work in progress.

[TODO: description]

## Getting started

### Clone repositories

In the same directory (perhaps `dseinternational`):

```bash
git clone https://github.com/dseinternational/vocabulary-growth.git
```

For now, also:

```bash
git clone https://github.com/dseinternational/research.git
```

### Prerequisites

#### Fitting models

To fit models, a recent Python installation is required. Some of our dependencies are best installed from [conda-forge](https://conda-forge.org/), for which either [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) or [Miniforge](https://conda-forge.org/download/) is required.

Then, to install Python dependencies, from the repository root:

```bash
conda env update -f environment.yml
```

#### Creating reports

To update or create reports, [Quarto](https://quarto.org/docs/get-started/) is required. We also use CSpell for checking spelling, for which a recent installation of [Node.js](https://nodejs.org/en) is required.

To install Node dependencies, from the repository root:

```bash
npm install
```

### Preparing data

From the repository root:

Activate the Conda environment (if not yet activated):

```bash
conda activate dse-vocab-growth
```

Run the script:

```bash
python scripts/prepare_data.py
```

## License

All source code in this repository is licensed under the GNU Affero General Public License v3.0 **(AGPL-3.0-only)**. See `LICENSE`.

Some other artifacts are licensed under other licenses:

- **Code**: GNU Affero General Public License v3.0 (AGPL-3.0) — see `LICENSE`.
- **Documentation, reports and papers**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `docs/LICENSE`.
- **Data**: Creative Commons Attribution 4.0 International (CC BY 4.0) — see `data/LICENSE` for details.

AGPL-3.0 requires that if you modify and run this software to provide a network service, you must offer the corresponding source code to users of that service.
