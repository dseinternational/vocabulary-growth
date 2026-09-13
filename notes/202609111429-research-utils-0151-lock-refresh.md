> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:words fonttools tqdm wrapt -->

# Shared library 0.15.1 lock refresh

Issue [#333](https://github.com/dseinternational/vocabulary-growth/issues/333) upgrades `dse-research-utils` from 0.15.0 to 0.15.1, following upstream [research#105](https://github.com/dseinternational/research/pull/105) and the tagged [0.15.1 upgrade notes](https://github.com/dseinternational/research/blob/v0.15.1/docs/migrating-to-0.15.1.md). The lockfile selects commit `a16183955674720a011080b315ea1a97825f665a`, which is what the annotated `v0.15.1` tag resolves to. The release is a lock refresh, not an API or floor change: it moves six transitive packages and nothing else. No production code in `src/vocab_growth/` changed, nothing was refitted, and nothing was published.

## A downstream repository does not inherit the upstream lock

This is the whole reason the issue exists as a separate piece of work. `dse-research-utils` is installed from a git tag, and a Git consumer takes the library's `pyproject.toml` floors but not its `uv.lock`. Changing the tag alone would have moved `dse-research-utils` and left the six package updates the release exists to carry sitting inert, exactly as `uv lock`'s conservatism left the 0.15.0 ceiling lift inert until the three packages were named explicitly. So they were named:

```bash
uv lock --upgrade-package dse-research-utils --upgrade-package scikit-learn --upgrade-package fonttools \
        --upgrade-package pure-eval --upgrade-package tqdm --upgrade-package wrapt --upgrade-package ruff
```

Seven packages moved, and no others:

| Package              | 0.15.0 lock | 0.15.1 lock |
| -------------------- | ----------- | ----------- |
| `dse-research-utils` | 0.15.0      | 0.15.1      |
| `fonttools`          | 4.64.0      | 4.65.0      |
| `pure-eval`          | 0.2.3       | 0.2.4       |
| `ruff`               | 0.16.6      | 0.16.7      |
| `scikit-learn`       | 1.9.0       | 1.9.1       |
| `tqdm`               | 4.70.0      | 4.70.1      |
| `wrapt`              | 2.4.0       | 2.4.1       |

The resolved package set is otherwise identical — a name-by-name diff of the lockfile adds and removes nothing. The numerical stack is untouched: `numpy` 2.5.3, `numba` 0.67.0, `pytensor` 3.3.1 and `pymc` 6.3.2 are the versions 0.15.0 locked and the versions upstream still locks.

## The two upstream packages that deliberately did not come downstream

Upstream also moved `build` 1.6.0 → 1.6.1 and `uv` 0.12.11 → 0.12.13. Both are repository-only build tooling for `research` itself and have never been locked here; the upgrade notes say not to copy them across, and the lockfile confirms neither is present. Copying the upstream lock wholesale would have imported them along with a resolution made against a different set of extras — this repository takes `columnar`, `graphs`, `io`, `jax`, `notebook` and `viz`, which are preserved unchanged.

None of the six transitive moves is imported by anything in `src/`, `scripts/` or `tests/`. They arrive through `matplotlib` (`fonttools`), `dse-research-utils` (`scikit-learn`), `numpyro` (`tqdm`) and `formulaic` (`wrapt`); `pure-eval` is IPython traceback machinery and `ruff` is the linter. Nothing on the sampling path moved.

## The signature was already broken, and stays broken

`models/implementation_identity.py` hashes the installed versions of `pymc`, `pytensor`, `numpy`, `scipy`, `pandas`, `arviz`, `nutpie` and `dse-research-utils` alongside the package's own AST. Exactly one of the eight moved here — the library version itself — so every fit on disk continues to fail the executable-code signature under `resume`, `sync` and `publish`.

This changes nothing about the standing position. Those fits were already restaled by the 0.15.0 upgrade, which moved four of the eight; this adds a seventh version string to a signature that did not match anyway. `render` and `provisional-sync` do not ask for the signature, so existing fits stay renderable and locally reviewable, and the reporting-quality refit that the publish path needs is the same refit it needed before this change. Preserving the fitted results and their recorded environments is the explicit scope boundary of the issue.

## What was checked

On win-amd64 (native Windows, `PYTHONUTF8=1`), against the locked environment:

- `uv lock --check` and `uv sync --locked` install the lockfile without re-resolving.
- `dse-research-utils` reports 0.15.1, and all thirty-seven `dse_research_utils.*` modules this repository imports load, as does `vocab_growth` itself. The module list was taken from the source rather than guessed — the package has no top-level `plotting`, `sampling`, `diagnostics` or `reporting`, and the real paths are `plot.*`, `statistics.models.sampling`, `statistics.diagnostics` and `report.*`.
- `ruff check src/ scripts/ tests/` is clean under the new 0.16.7, and `mypy` finds no issues in its four modules.
- `npm run format:check` and `npm run spellcheck` pass (225 files, 0 issues).
- The whole test suite passes, run as CI runs it in two jobs: `-m "not slow"` with `--dist loadfile` gives 2,056 passed and 11 skipped in 45.9 s, and `-m slow` with `--dist loadgroup` gives 304 passed in 108.8 s. The union is 2,360 passed and 11 skipped, the same counts the 0.15.0 upgrade recorded.

`DSE_VOCAB_GROWTH_OUTPUT_DIR` is set machine-wide on this workstation and has to be unset for the test run, or `test_comparison.py::test_registry_resolution` fails on the redirected root. That is a property of the workstation, not of this upgrade.

No model was fitted and no fit was rendered as part of this change. A lock refresh does not refit a model; the fits of record remain the fits of record until they are re-run.
