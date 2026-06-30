> [!NOTE]
> Drafted by an LLM-based AI tool (OpenAI Codex/GPT-5).

<!-- cspell:words Matplotlib MPLCONFIGDIR matplotlib Numba writable -->

# Full reporting replication run

Date: 2026-06-29

## Request

Run the full reporting-quality replication for all vocabulary-growth models, render
all model outputs, upload them to blob storage, and include the TD/DS comparison
outputs and analysis.

## Preflight

- Git branch/status at start: `main`, clean relative to `origin/main`.
- Python environment: Conda environment `dse-vocab-growth`, Python 3.14.6.
- Fallback Python environment: Conda environment `dse-research-vocab-growth`,
  Python 3.14.4, used with explicit `PYTHONPATH=src:../research/src/python/src`
  because the requested environment is missing a usable `jaxlib` installation.
- Actual fit interpreter: `dse-vocab-growth` Python 3.14.6 with
  `PYTHONPATH=src:../research/src/python/src:.cache/pythonpath-jax`. The final
  path contains symlinks to `jax==0.10.0` and `jaxlib==0.10.0` from the adjacent
  environment, leaving the rest of the declared project environment in use.
- Quarto: 1.9.36.
- Disk space before run: about 414 GiB free on the project volume.
- Project import check: `vocab_growth`, `pymc`, `nutpie`, and Azure Blob imports
  succeeded.
- Matplotlib cache: set `MPLCONFIGDIR=.cache/matplotlib` for run commands because
  the default home/cache locations are not writable in the current sandbox.
- Upload blocker at start: `DSERESEARCH_BLOB_CONTAINER_URL` is not set in this
  shell. The project uploader requires this variable and raises before upload if
  it is absent.

## Planned commands

```bash
conda run -n dse-vocab-growth python scripts/prepare_data.py
PYTHONPATH=src:/Users/frankbuckley/dev/dseinternational/research/src/python/src:.cache/pythonpath-jax MPLCONFIGDIR=.cache/matplotlib NUMBA_CACHE_DIR=.cache/numba XDG_CACHE_HOME=.cache/xdg PYTENSOR_FLAGS=base_compiledir=.cache/pytensor /Users/frankbuckley/miniconda3/envs/dse-vocab-growth/bin/python scripts/fit_model.py all --config rep --render
MPLCONFIGDIR=.cache/matplotlib conda run -n dse-vocab-growth python scripts/compare_ds_td_re.py --verify comprehension
MPLCONFIGDIR=.cache/matplotlib conda run -n dse-vocab-growth python scripts/compare_ds_td_re.py --verify
MPLCONFIGDIR=.cache/matplotlib conda run -n dse-vocab-growth python scripts/compare_ds_td_expressive.py --verify
MPLCONFIGDIR=.cache/matplotlib conda run -n dse-vocab-growth python scripts/compare_models.py
quarto render docs/comparison/index.qmd
```

If `DSERESEARCH_BLOB_CONTAINER_URL` becomes available, upload model outputs with:

```bash
MPLCONFIGDIR=.cache/matplotlib conda run -n dse-vocab-growth python scripts/upload.py all
```

## Run log

- 2026-06-29 12:00: preflight complete; upload target environment variable is
  missing.
- 2026-06-29 12:05: `scripts/prepare_data.py` completed successfully. It loaded
  10 source datasets, produced `data/vocab_data_merged.csv`, rebuilt
  `data/vocabulary.duckdb`, and reported 917 merged rows.
- 2026-06-29 12:06: initial `conda run ... scripts/fit_model.py all --config
  rep --render` attempt failed before model fitting because Numba could not cache
  a `preliz` function under the `conda run` wrapper. Retried with the Conda
  environment Python executable directly and workspace-local cache directories.
- 2026-06-29 12:08: direct `dse-vocab-growth` Python reached VG01 but failed
  before sampling because `jax` was installed without `jaxlib`. The declared
  project metadata includes `jaxlib>=0.9.2`, so this appears to be an incomplete
  local environment. The existing `dse-research-vocab-growth` environment has
  `jax`/`jaxlib`/`numpyro` and can import this repository with explicit
  `PYTHONPATH`.
- 2026-06-29 12:12: fallback environment could import the repository but failed
  to run `fit_model.py` because it lacks Azure packages imported by
  `vocab_growth.storage`. Created a workspace-local `.cache/pythonpath-jax`
  bridge to expose only the complete `jax==0.10.0`/`jaxlib==0.10.0` pair to the
  `dse-vocab-growth` interpreter. Import check then passed with PyMC 6.0.1 and
  nutpie 0.16.10.
- 2026-06-29 12:15: next VG01 attempt reached prior/hyperparameter processing
  but failed because PyTensor/Numba tried to write compiled cache files under
  `~/.pytensor`, which is not writable in this sandbox. Added
  `PYTENSOR_FLAGS=base_compiledir=.cache/pytensor`.
- 2026-06-29 12:23: VG01 completed in 5m 27.6s. Reported diagnostics passed
  (`r_hat <= 1.01`, ESS >= 400). LOO-CV reported one bad Pareto-k observation
  (1/1063, 0.1%) and no very-bad Pareto-k observations.
- 2026-06-29 12:27: VG02 completed in 3m 45.6s. Reported diagnostics passed.
  LOO-CV Pareto-k diagnostics were all good (818/818).
- 2026-06-29 13:19: VG03 completed in 51m 41.6s. Reported diagnostics passed.
  LOO-CV Pareto-k diagnostics were all good (4138/4138).
- 2026-06-29 13:32: VG04 completed in 12m 42.8s. Reported diagnostics passed.
  LOO-CV Pareto-k diagnostics were all good (1534/1534).
- 2026-06-29 13:43: VG05 completed in 10m 24.1s. Reported diagnostics passed.
  LOO-CV for words spoken reported two bad Pareto-k observations (2/1063,
  0.2%) and no very-bad observations; words understood was all good (818/818).
- 2026-06-29 14:05: the all-model process was killed with exit code 137 during
  VG06 posterior sampling. At this point VG01-VG05 had completed fitting and
  staged their model report source files, but the script had not reached its
  final Quarto render loop. `output/` was about 31 GiB with about 376 GiB free.
  Recovery plan: continue with single-model `fit_model.py <model> --config rep
  --render` runs from VG06 onward, then render VG01-VG05 separately.
- 2026-06-29 15:27: isolated VG06 fit completed in 1h 20m 24.0s. Reported
  diagnostics passed. LOO-CV for words spoken reported one bad Pareto-k
  observation (1/4138) and no very-bad observations; words understood was all
  good (1547/1547). The subsequent Quarto render step failed because Quarto
  tried to create `/Users/frankbuckley/Library/Caches/quarto`, which is not
  writable in this sandbox. Recovery plan: render with workspace-local
  `HOME`/cache settings and run remaining fits without relying on the script's
  final render step.
- 2026-06-29 15:31: VG06 `index.qmd` rendered successfully to `index.html`
  after setting workspace-local Quarto cache/home variables and allowing Quarto
  to start its local Jupyter kernel.
- 2026-06-29 15:42: VG07 completed in 10m 27.9s. Reported diagnostics passed.
  LOO-CV for words spoken reported two bad Pareto-k observations (2/1063,
  0.2%) and no very-bad observations; words understood was all good (818/818).
- 2026-06-29 15:50: VG08 completed in 8m 02.0s. Reported diagnostics passed.
  LOO-CV warnings increased after adding subject random intercepts on
  understood: spoken had 9 bad and 2 very-bad Pareto-k observations; understood
  had 74 bad and 4 very-bad Pareto-k observations.
- 2026-06-29 15:59: VG09 completed in 8m 15.3s. Sampling diagnostics did not
  fully pass: 3 parameters had `r_hat > 1.01` (max 1.021) and 3 had tail ESS
  below 400 (min 344). LOO-CV warnings were substantial: spoken had 121 bad and
  11 very-bad Pareto-k observations; understood had 108 bad and 4 very-bad
  Pareto-k observations.
- 2026-06-29 16:09: VG10 completed in 9m 51.4s. Sampling diagnostics improved
  but did not fully pass: 3 parameters had `r_hat > 1.01` (max 1.012) and 3 had
  tail ESS below 400 (min 357). LOO-CV warnings remained substantial: spoken had
  121 bad and 10 very-bad Pareto-k observations; understood had 106 bad and 5
  very-bad Pareto-k observations.
- 2026-06-29 20:58: VG11 completed in 4h 29m 07.9s. Posterior sampling took
  4h 22m 31.5s. Reported diagnostics passed (`r_hat <= 1.01`, ESS >= 400).
  LOO-CV Pareto-k diagnostics were all good (16,235/16,235). The fitted TD
  spoken trajectory had median expected proportions of about 0.004 at 9 months,
  0.013 at 12 months, 0.096 at 18 months, and 0.198 at 21 months. Disk space
  after VG11 was about 277 GiB free.
- 2026-06-29 22:29: VG12 completed in 1h 30m 51.8s. Posterior sampling took
  1h 28m 31.0s. Reported diagnostics passed. LOO-CV Pareto-k diagnostics were
  all good (5,997/5,997). The fitted TD understood trajectory had median
  expected proportions of about 0.058 at 9 months, 0.100 at 12 months, 0.273 at
  18 months, 0.440 at 24 months, and 0.497 at 30 months.
- 2026-06-29 23:52: VG13 completed in 1h 23m 35.9s. Posterior sampling took
  1h 12m 01.8s. The sampler progress table reported non-zero divergences across
  chains, but the model's reported `r_hat`/ESS diagnostics passed. LOO-CV
  Pareto-k diagnostics were all good for both outcomes (spoken 7,920/7,920;
  understood 5,406/5,406). In the young TD joint model, median understood
  expected proportions rose from 0.051 at 8 months to 0.287 at 18 months, median
  spoken expected proportions rose from 0.003 at 8 months to 0.099 at 18 months,
  and the median production ratio `q(a)` rose from 0.067 at 8 months to 0.347 at
  18 months. Disk space after VG13 was about 219 GiB free.
- 2026-06-30 00:05: VG14 completed in 12m 42.9s. Posterior sampling took
  9m 17.4s, with no divergences shown in the sampler progress table. Reported
  diagnostics passed. LOO-CV Pareto-k diagnostics were all good for understood
  (818/818) and signed (528/528); spoken had two bad Pareto-k observations
  (2/1,063, 0.2%) and no very-bad observations. Median `q(a)` for spoken
  production rose from 0.131 at 12 months to 0.842 at 90 months. Median signed
  rate `r(a)` peaked around 30 months (0.415) and then declined. Median total
  expressive vocabulary `p_any(a)` rose from 0.0097 at 12 months to 0.516 at
  90 months. The uk_02 validation check reported observed union mean 0.571,
  independence union mean 0.608, and VG14 `p_any / p_U` mean 0.569.
- 2026-06-30 00:15: VG15 completed in 9m 59.8s. Posterior sampling took
  9m 04.1s, with no divergences shown in the sampler progress table. Reported
  diagnostics passed. The model does not compute LOO in the same report path as
  VG14; it reports posterior predictive checks for the four-cell sign/speech
  composition. Median total expressive `p_any(a)` rose from 0.001 at 12 months
  to 0.553 at 90 months. The sign/speech association parameter had median
  `psi = 2.169`, 90% HDI [1.438, 2.948], with `P(psi > 1) = 1`, indicating
  positive within-understood association between spoken and signed production.
- 2026-06-30 00:18: rendered all 15 model Quarto reports under
  `output/models/*/index.html`.
- 2026-06-30 00:19: `scripts/compare_ds_td_re.py --verify spoken understood
  comprehension` completed and wrote RE-based TD/DS comparisons to
  `output/comparisons/`.
- 2026-06-30 00:22: `scripts/compare_ds_td_expressive.py --verify` completed
  and wrote expressive-delay, sign-inclusive, distributional, and peak-growth
  comparisons to `output/comparisons/`.
- 2026-06-30 00:22: `scripts/compare_models.py` completed and wrote headline
  overlay comparisons to `output/comparisons/`.
- 2026-06-30 00:23: copied regenerated comparison artifacts into
  `docs/comparison/` because `docs/comparison/index.qmd` reads its CSVs and
  figures as local files. The comparison report then rendered successfully to
  `docs/comparison/index.html`.
- 2026-06-30 00:24: final artifact check found 15 rendered model HTML reports,
  `docs/comparison/index.html`, and 81 files in `output/comparisons/`. Disk
  space was about 207 GiB free.
- 2026-06-30 00:24: upload initially not run because
  `DSERESEARCH_BLOB_CONTAINER_URL` was unset in this shell.
- 2026-06-30 08:31: after setting
  `DSERESEARCH_BLOB_CONTAINER_URL=https://dseresearch.blob.core.windows.net/public`,
  `scripts/upload.py all` uploaded all 15 model report artifact directories to
  Azure Blob Storage. Trace files were excluded, following the uploader default.
- 2026-06-30 08:31: uploaded the rendered TD/DS comparison report directory
  separately as `comparison-report`, skipping the local Quarto cache. Public URL:
  <https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f1770-f018-73b1-bf0e-47e2e53ae783/comparison-report/index.html>.

## Findings

- All 15 reporting-configuration model fits completed and rendered.
- Core convergence diagnostics passed for VG01-VG08 and VG11-VG15. VG09 and
  VG10 retained small convergence warnings (`r_hat` and tail ESS) under the
  reporting run, so their comparison use should be interpreted with that
  residual diagnostic caveat.
- LOO-CV Pareto-k diagnostics were clean for VG02, VG03, VG04, VG06 understood,
  VG07 understood, VG11, VG12, and VG13. Small numbers of bad Pareto-k values
  appeared in VG01, VG05 spoken, VG06 spoken, VG07 spoken, and VG14 spoken.
  VG08-VG10 had substantial Pareto-k warnings after adding subject random
  intercepts, especially on understood. These warnings concern approximate
  leave-one-out reliability for influential observations; they do not by
  themselves imply failed sampling.
- The RE-based TD/DS comparison shows a much larger production gap than
  comprehension gap. At 24 months, expected understood words were about 352
  for TD versus 87 for DS (about 4.0x), while expected spoken words were about
  257 for TD versus 4 for DS (about 61.9x). `P(TD > DS)` was 1.00 for spoken at
  the key overlap ages.
- Attainment delays increased with vocabulary level, supporting developmental
  stretch rather than a constant age shift. For spoken vocabulary, DS lagged TD
  by about 16.8 months at 10 words, 22.4 months at 50 words, 25.7 months at 100
  words, and 32.1 months at 200 words. For understood vocabulary, delays were
  about 9.8 months at 50 words, 12.1 months at 100 words, 21.3 months at 200
  words, and 32.2 months at 300 words.
- The comprehension-matched q(U=N) comparison shows a production-specific gap:
  at U=50 understood words, TD q was about 0.08 and DS q about 0.03, with
  Δq=+0.05 [0.03, 0.08] and `P(TD > DS)=1.00`. The gap remained positive
  through about U=150, with weaker evidence by U=200.
- The expressive-delay analysis estimated an extra production delay beyond
  comprehension delay of 12.3 months [9.6, 15.3] at N=50 words, with
  `P(Δ_exp > 0)=1.00`.
- Peak learning-rate ages were much later for DS: about 63 months for spoken
  and 71 months for understood, compared with about 23 and 18 months in TD.
  The DS estimates are partly boundary-censored, so the direction is robust but
  the upper timing is weakly identified.
- Sign-inclusive expressive estimates show that counting signs narrows the
  expressive gap but does not remove it. VG14 estimated total expressive
  `p_any(a)` rising from 0.0097 at 12 months to 0.516 at 90 months. VG15
  estimated a positive within-understood sign/speech association (`psi` median
  2.169, 90% HDI [1.438, 2.948], `P(psi > 1)=1`).
