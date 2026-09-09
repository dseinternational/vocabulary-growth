> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:words llvmlite Optuna renderable -->

# Shared library 0.15.0 upgrade

Issue [#326](https://github.com/dseinternational/vocabulary-growth/issues/326) upgrades `dse-research-utils` from 0.14.0 to 0.15.0. The lockfile selects commit `e818caf52c0a73aed4ba5286bdfdc58c7867ea69`. The release changes no library API and adds no packages: it raises the shared floors this repository inherits transitively and lifts the NumPy ceiling that has capped the stack since 0.11.0. The work followed the tagged [0.15.0 upgrade notes](https://github.com/dseinternational/research/blob/v0.15.0/docs/migrating-to-0.15.md). No production code in `src/vocab_growth/` changed, nothing was refitted, and nothing was published.

## The ceiling lift needed an explicit upgrade

Bumping the tag and running `uv lock` moved four packages and left the ceiling lift inert. `uv lock` is conservative: it preserves an existing resolution wherever it still satisfies the constraints, and numpy 2.4.6, numba 0.66.0 and pymc 6.3.1 all still satisfy the widened ranges. A ceiling that is merely raised buys nothing, so the three were upgraded explicitly:

```bash
uv lock --upgrade-package numpy --upgrade-package numba --upgrade-package pymc
```

That is the whole reason for preferring a targeted upgrade over `uv lock --upgrade`: it moves exactly what this release exists to unblock, and leaves the weekly Dependabot sweep to propose the rest on its own schedule.

Eight packages moved in total:

| Package                  | 0.14.0 lock | 0.15.0 lock |
| ------------------------ | ----------- | ----------- |
| `dse-research-utils`     | 0.14.0      | 0.15.0      |
| `numpy`                  | 2.4.6       | 2.5.3       |
| `numba`                  | 0.66.0      | 0.67.0      |
| `llvmlite`               | 0.48.0      | 0.49.0      |
| `pytensor`               | 3.3.0       | 3.3.1       |
| `pytensor-distributions` | 0.2.0       | 0.3.2       |
| `pymc`                   | 6.3.1       | 6.3.2       |
| `preliz`                 | 0.27.1      | 0.28.0      |

`llvmlite` is numba's own runtime and follows it; `pytensor-distributions` follows PyTensor. Everything else the release raised a floor for — `statsmodels` 0.15.0, `arviz-stats` 1.3.2, `arviz-plots` 1.3.1, `polars` 1.44.1, `pyreadstat` 1.3.6, `orjson` 3.12.0, `seaborn` 0.13.2, `networkx` 3.6.1 — was already at or above the new floor in this repository's lockfile, so the floor rise is a no-op here. Optuna does not appear: this repository takes neither the `tuning` nor the `boosting` extra, so Optuna 5.0's changed sampler defaults reach nothing here.

## The Dependabot rule moved with the pin

`.github/dependabot.yml` still ignored `numpy >=2.5.0`, a cap that no longer exists anywhere else. It becomes `>=2.6.0`, which is where the real ceiling now sits — PyTensor 3.3.1 admits `numba<=0.67.0`, and numba 0.67.0 pins `numpy<2.6`. The comment above the rule was rewritten rather than left describing the superseded chain, because the reason is the part that has to stay true: the cap is inherited from `dse-research-utils`, and widening it here only declares a range no resolver can use.

The two must move together. Leaving the rule at `>=2.5.0` would have had Dependabot keep proposing numpy 2.5 as though it were still blocked, and leaving it behind after the ceiling rose again would have it propose a widening that cannot resolve.

## The aarch64 numba workaround is now untested

`docs/runbooks/full-refit.md` records that `vg15 fallback-dispersion` could not compile on the linux-aarch64 refit VM under numba 0.66.0 / llvmlite 0.48.0: LLVM ran out of registers in `np_concatenate` over the 44 gradient arrays nutpie assembles in one call, and `--nutpie-backend jax` is the documented escape hatch. Both halves of that compiler moved here — numba to 0.67.0 and llvmlite to 0.49.0 — and nutpie still concatenates in one call at 0.16.11.

Nothing in this upgrade was run on aarch64, so the workaround is neither confirmed still necessary nor shown to be obsolete. The runbook is left as it stands: it describes a failure that was observed, and the right time to find out whether 0.67 changed it is the next refit cycle on that VM, where the default backend should be tried first as the runbook already says.

## Every existing fit now fails the signature check

`models/implementation_identity.py` hashes the installed versions of `pymc`, `pytensor`, `numpy`, `scipy`, `pandas`, `arviz`, `nutpie` and `dse-research-utils` alongside the package's own AST. Four of those eight moved. The executable-code signature therefore changes even though no line of `src/vocab_growth/` did, and every fit on disk fails it under `resume`, `sync` and `publish`.

That is the check working, not a problem to route around. `render` and `provisional-sync` do not ask for the signature, so existing fits stay renderable and locally reviewable; syndicating any of them into the report needs a reporting-quality refit on this stack. A saved fit and a new one must not be compared across this upgrade without re-establishing implementation identity first — numpy 2.5, numba 0.67 and PyTensor 3.3.1 are a genuinely different numerical stack from the one 0.14.0 locked, whatever the library's own tests say about its API.

## What was checked

On macOS-arm64, against the locked environment:

- `uv sync --locked` installs the lockfile without re-resolving.
- `ruff check src/ scripts/ tests/` and `mypy` are clean.
- The whole test suite — `pytest -m "slow or not slow"`, the union of CI's two test jobs — passes: 2,360 passed, 11 skipped, in 4 m 31 s. Several test modules import `preliz` directly, and each engine module imports `preliz.distributions.distributions.Continuous` to annotate its prior fields, so the 0.28.0 move is exercised rather than merely installed.
- `prepare_data.py` reproduces the prepared data on the new stack, and the VG01 `dev` fit — the end-to-end pipeline smoke CI runs — completes through all ten stages in 49 s.

**The convergence gate is not evidence this upgrade has produced.** The VG01 `dev` fit reports `passed: false` (max R-hat 1.017 against 1.01, min ESS 98 against 400, 0 divergences, BFMI 0.94/0.90). That is the expected reading at that tier and not a finding about the stack: `dev` is 2 chains × 500 draws, it is named in `fit_artifacts.NON_REPORTING_CONFIGS`, and the gate's thresholds are the reporting-quality ones. CI asserts that the fit runs, not that it converges. A gate result that means something needs a `rep`-tier fit, which belongs on the fitting VM and is not part of this change.

None of the above is evidence about the posterior of any model of record. A floor change does not refit a model; the fits of record were made on the 0.14.0 stack and remain the fits of record until they are re-run on this one.
