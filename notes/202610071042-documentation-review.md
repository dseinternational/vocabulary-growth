# Documentation review

> [!NOTE]
> Drafted with assistance from OpenAI Codex/GPT-6.

This review examined repository documentation and instructions against code revision `3b424d6f0c9c7781c9787b68e4e455fc61b75c5f`, the current model declarations and dated follow-up records. The review covered the documentation inventory, navigation, maintained guides, model-report explanations and the status of older plans. It did not reproduce the fitted results or independently review every cited paper.

## Current guidance

The three agent instruction files remain identical. They now state the main data rules, fit checks and reporting workflow with less repeated explanation. The model inventory, prior guide, source records and runbooks remain the starting points for current practice.

Model pages and report methods now distinguish a zero-effect reference curve, a new child's expected trajectory and a future observed count. The review removed claims that changing child correlation must preserve a reference curve or widen an interval. It also clarified age-dependent production ratios, the limits of matching one marginal correlation prior, and the conditional nature of signing milestone ages.

The data guide explains the common 810-item scale and overlapping speech and signing. The Ireland source record now states the partial baseline rule. The UK 02 source record lists the fields in the prepared CSV rather than an obsolete research-file schema. VM resource figures are identified as dated estimates, and campaign storage planning uses the actual stage plan.

## Superseded material

Old model-retirement, implementation and report-template work lists were replaced with outcome or successor links. The September TD validation proposal now points to the implemented October harness. Earlier VG25 notes point to its revised spoken-only lag and existing recovery and forward-scoring tools. Historical measurements, alternatives, data decisions and preregistrations remain available with their dates and corrections.

The shared-library 0.17.0 upgrade memo moved from `docs/shared-utilities-0.17.md` to the [4 October dependency record](202610041345-research-utils-017-upgrade.md). The September file-coverage ledger was removed because it duplicated a completed review record and described an old file tree. The [statistical review corrections](202609231200-statistical-review-corrections.md) retain an archive link to that ledger.

The paper, technical report and practical summary remain unfinished. The review replaced empty report placeholders with explicit status text and drafted general guidance for the practical summary. It did not supply new numerical findings, implement interactive tools or approve publication.

## Verification

The locked environment and prepared data were rebuilt. Ruff and mypy passed. The full fast-and-slow test run reported 3,056 passes, 11 skips, nine model-test failures and six setup errors caused by the local PyTensor C++ linker, plus three report-link failures introduced during editing. Those three links were corrected. All 105 tests in the affected model and recovery files passed when rerun with `PYTENSOR_FLAGS=cxx=`. This confirms that those assertions pass without C++ compilation; it does not establish that the host's compiled path works.

Spelling, formatting and notes-index checks passed. All 923 local source links checked across 246 prose files resolve. Quarto inspection passed for the report, summary and paper projects. The final report-link, reference, notes-index, model-catalogue and campaign test selection passed all 341 tests. The three instruction copies are identical. Executable report cells and parsed package settings match the base revision. It does not render every model report, because no new fitting or figure-generation run forms part of this edit.
