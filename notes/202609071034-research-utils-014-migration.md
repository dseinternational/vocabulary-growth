> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5).

<!-- cspell:words nonfinite dtypes umask netCDF PSIS reff -->

# Shared library 0.14.0 migration

Issue [#313](https://github.com/dseinternational/vocabulary-growth/issues/313) upgrades `dse-research-utils` from 0.13.0 to 0.14.0 and routes this repository's remaining local copies through the helpers that release adds. The lockfile selects commit `50390f4f9cd19033e3d0dc9bd050b383bd413c36`. The existing extras and shared ownership of scientific dependencies are unchanged, and no other locked package moved. The work followed the tagged [0.14.0 upgrade notes](https://github.com/dseinternational/research/blob/v0.14.0/docs/migrating-to-0.14.md) and [combined migration guide](https://github.com/dseinternational/research/blob/v0.14.0/docs/consolidation-migration.md).

The guide's own summary of what the release does and does not establish applies here: it moved library code, and it did not migrate this repository's production code, refit anything or publish anything. What follows is what this repository decided, and what was checked.

## What now delegates, and what deliberately does not

### Files, directories and provenance

`write_json_atomic` keeps its serialisation — two-space indent, sorted keys, the existing `_json_default` encoder, trailing newline — and replaces its own temporary-file dance with `storage.files.atomic_write`. The stored bytes matter beyond readability: a `comparison_manifest.json` records the SHA-256 of each contributing fit's manifest, so a changed encoding would invalidate every comparison on disk.

The shared helper creates its temporary file with `mkstemp`, which is owner-only. Every file this repository has written came from a plain `open` and carries `0o666 & ~umask`, and those files are read back by other accounts on the fitting VM and by the uploader, so `write_atomic` restores that mode explicitly. It is now the single place that decision is made; `scripts/sync_report_figures.py` writes its two generated tables through it as well.

`promote_staged_fit` delegates its two renames to `storage.directories.promote_directory`, which validates the three paths against each other — no nesting, no aliasing through a mounted tree, no symlink in any component, one filesystem — before it touches the destination, and restores the previous directory itself when the second rename fails. Two decisions stay local because the helper leaves them to the caller. The lock is `fit_artifacts.PROMOTION_LOCK`, whose scope is threads of one process: a fit promotes the one `models/<label>/` directory it staged, and two concurrent fits of the same model into the same output root would already be racing for that directory long before promotion. The retained backup is deleted on success, as it always was — `.previous` is a rollback slot, not an archive, and a fit's output can be tens of gigabytes — and a cleanup failure after a successful promotion is reported rather than raised.

`promote_directory` refuses a symlinked ancestor, and on the fitting VM `<repo>/output` _is_ a symlink to a scratch volume, as is `/tmp` on macOS. `fit_artifacts.promotion_path` therefore resolves the parent chain deliberately, which is what the guide documents for that case, and leaves the final component alone so a destination that is itself a symlink is still refused. `scripts/sync_report_figures.py` promotes its figure cache through the same rule.

`git_metadata` keeps its four manifest keys and their meanings and obtains them from `metadata.provenance.git_snapshot`: one bounded `status --porcelain=v2 --branch` call instead of three commands, with inherited `GIT_*` overrides stripped and optional index locks and the filesystem-monitor hook disabled. `dirty` still counts staged, unstaged, unmerged and untracked changes and excludes ignored files, and is still `None` — never `False` — when the query could not establish it, because `validate_fit_output` turns `dirty is not False` into a refusal to resume or publish. The 10-second timeout this repository has always allowed is passed explicitly rather than inheriting the shared default of 5. `comparisons_provenance._file_sha256` is now `sha256_file` plus this repository's `sha256:` prefix.

Two provenance values are deliberately **not** delegated. `source_data_hash` is one digest over `name + NUL + bytes + NUL` for every CSV in `data/`, not a per-file digest; changing its composition would invalidate every recorded raw-data fingerprint. The manifest's `runtime.packages` enumerates every installed distribution and reads each one's `direct_url.json` for a commit, which `package_versions` — an explicit name list returning versions only — cannot reproduce.

Two write-then-replace sites elsewhere keep their own code, with reasons. `scripts/compact_traces.py` writes the rewritten trace while the source `DataTree` is still open and verifies it only after closing that handle; expressing that as one callback would mean two concurrent handles on the same netCDF file. `scripts/prepare_data.py` writes its DuckDB database through a connection opened at the top of the script, so the callback model would mean restructuring the whole script for no change in behaviour.

### Administration-level likelihood aggregation

`administration_loo.administration_log_likelihood` now hands its factors to `statistics.log_likelihood.aggregate_log_likelihood` with the output units stated explicitly: `unit_ids=np.flatnonzero(any_mask)`, the administration rows of the analysis frame, and `row_unit_ids=np.flatnonzero(mask)` per factor. `obs_joint` therefore _is_ the frame position, rather than a rank among the kept rows that happened to coincide with it.

Everything that decides which unit a row belongs to stays local: the factor/mask pairs each engine declares, the `None` return for a trace missing a factor or its mask, and the issue #266 finding-3 check that a mask marking recorded rather than likelihood rows fails loudly. One refusal is added here rather than taken from the library — a trace variable declared as two factors would be summed onto the same administrations twice, and the shared duplicate-array guard cannot see it because indexing a `Dataset` twice yields two distinct objects.

The shared aggregator adds two refusals of its own: `NaN` or `+inf` in a likelihood raises instead of propagating into a plausible-looking score, while `-inf` is retained because an impossible observation genuinely has that log likelihood. It also permits a zero-row factor, which the previous reshape did not.

`scripts/loo_compare.py` carried the original of this arithmetic and kept its own copy of it. It now delegates too, keeping its `y_joint` variable name — the comparison tables and every regenerated CSV use it — and its two refusals, since it is pointed at arbitrary traces on disk and must say why one cannot be scored.

### Predictive summaries

`models/calibration.predictive_calibration_table` takes its per-observation quantities from `statistics.predictive.predictive_observation_checks`: linear quantiles at `(1 - p) / 2` and its complement, closed-interval inclusion, the predictive mass inside those bounds, and the midpoint PIT with the same `(1 - sum p_k^3) / 12` reference. The age banding, the observed and predictive zero rates, the group means and the written table's columns and order stay here.

Interval widths are differenced in the bounds' own dtype and stored as float64 before the group means, which is the arithmetic the written table has always carried. Non-finite observations are refused here rather than by the shared checks, so the message names the table being written and how many rows are unusable; the shared helper's own empty-input refusal sits behind this repository's, which fires first.

`write_trace_calibration` extracts its draws through `statistics.samples.sample_matrix` and takes the observed values through `SampleMatrix.observed_values`. That is a real change: the observed group and the posterior-predictive group are separate arrays on the trace and were previously flattened and paired by position, so a re-labelled or reordered observed group would have been scored against the wrong replications. Every engine declares the observation dimension in the model's `coords`, so the labels are present in a real trace; the test fixtures now carry them too.

### Report reads and nearest-row access

`report_cells` reads its artefacts through `report.readers.read_csv` / `read_json`, which distinguish present, missing and invalid. The policy is this repository's: a file this engine never writes renders as a pending-fit placeholder, and a damaged one does not. `render_diagnostic_verdict`, `render_loo_section` and `render_variation_table` say which of the two it is; `_read` and `fitted_parameters` raise `ReportArtefactError`, because a reassuring "this fit has yet to produce it" is the wrong sentence about a corrupt file.

`empty_document` — a file with no columns at all — is deliberately treated as an empty table rather than damage. Several writers here build a table with `pd.DataFrame(rows)` and write it whatever `rows` contains; with no rows that produces a column-less frame and `to_csv` writes a single newline. `render_calibration_section` is where that mattered: it read the file with a bare `pd.read_csv` and a fit whose trace carried no recognised outcome aborted the report cell with `EmptyDataError`. It now prints that the table is empty.

`read_manifest` is deliberately left on the permissive parser. `write_json_atomic` serialises with Python's default `allow_nan`, so a definition field holding a non-finite float is written as a bare `NaN` token, and every manifest already on disk has to stay readable. The shared strict reader is used only for `diagnostics_summary.json`, which the shared diagnostics writer sanitises before writing.

`report.readers.nearest_row` replaces the two single-row lookups in `report_cells` — including the reference-child calibration's 0.6-month bound, which is now an explicit `max_distance` — and `comp_at` / `comp_row` in `docs/report/_report_data.qmd`. The two multi-row selections in that file keep their positional `argmin`: they build a set of row _positions_ to slice with, which `nearest_row` deliberately does not return.

### Report assets and upload checks

`publication_checks` is now an adapter over `report.assets`, which parses the HTML rather than matching quoted strings and reports each reference's state instead of dropping the ones it cannot use. `include_navigation=True` is deliberate: these pages offer their summary tables as ordinary download links, which the previous checks already treated as required. `follow_pages` stays off, because both publishers here upload a single rendered page and its assets.

Three behaviours changed, all of them in the direction the checks exist for.

- A required reference to a file that is not on disk is now a failure. The previous scanner dropped it, on the grounds that the render should have complained — so a page referencing a figure that was never written published silently, which is the 2026-09-03 comparison-book failure one step earlier. `scripts/publish_comparison.py` stops before collecting, and the model-report upload stops before requesting anything.
- `base href`, `srcset` and root-relative URLs are reported as unsupported rather than certified. Each of them decides which file a browser actually requests.
- An `href` is URL-decoded once. The existing test linked a file literally named `50%20.csv` as `href="50%20.csv"`, which asks the browser for `50 .csv`; the guide names that test as not a valid compatibility target. Both the mismatch and the correctly encoded `50%2520.csv` are now pinned.

`upload_to_blob_storage` gained an optional `fetch_status` transport, threaded through to `verify_published_assets`. It is how the publication path is now exercised end to end in the tests; before it, one of those tests reached a real Azure endpoint because the default transport is `build_opener(...).open`, not the `urlopen` the test patched.

### HSGP construction

`models/common.get_hsgp_hyperparams` returns the two lists `pm.gp.HSGP` takes, taken from a realised `HSGPDesign`; `hsgp_design_for` returns the record itself. The calibration is unchanged — `c_floor=None`, so no boundary floor is imposed and the basis count and boundary factor are exactly PyMC's `approx_hsgp_hyperparams` recommendation, with the boundary the recommendation's factor times the domain half-range and the centre its midpoint.

`gp_utils._gp_from_mean` builds the object with `create_hsgp`, which fixes the centre through a public `prior_linearized` call on the centre alone. That replaces an assignment to PyMC's private `_X_center`. The bounded length-scale prior, the variable names and their order, the dimensions, the observed-row projection and the anchoring are untouched. The unpinned branch remains for the exploratory modules and the experiment arms that deliberately test PyMC's default centring; `create_hsgp` requires an explicit centre, and those have none.

## What was checked

Numerical equivalence was established by comparing each migrated function against its previous implementation, inlined, rather than by reasoning about the code.

- **Administration likelihood.** 200 randomised cases — one to three factors, one of them a matrix-valued composition factor with a 2x2 cell dimension, over 3 to 40 administrations and 1 to 4 chains — are identical bit-for-bit, coordinates included. On a 60-administration, 4x500-draw case the pointwise PSIS-LOO `elpd_i` and Pareto-k arrays are identical too. Equal totals would not have established that; the pointwise arrays are what a comparison turns on.
- **Predictive calibration.** 364 complete tables compared with `assert_frame_equal(check_exact=True, check_dtype=True)` across 120 randomised shapes and three chunk sizes, plus one case each at float64, float32, int32 and int64 predictive draws. All identical, dtypes included. The float32 case is the one the guide flags, because the shared helper preserves the input reduction dtype in its bounds and medians.
- **HSGP.** 500 randomised calibrations reproduce the previous `m`, `L` and centre exactly. 200 randomised constructions give a bit-for-bit equal basis, spectral density and pinned centre under the factory and under the private assignment. The `trend_and_gp` latent draws identically under both for three centres and three seeds, with the same free RVs in the same order (`p_slope_low`, `p_slope_hi`, `ell_unit`, `eta`, `g_unit_hsgp_coeffs`) and the same deterministics. Saved geometry replays identically on a full grid, a subset of it, and a grid extended beyond the observed range — the issue #234 property, now a property of the record rather than of a pin.

`uv sync --locked`, Ruff, mypy, Prettier and CSpell pass. The complete suite selected with `-m "slow or not slow" -n auto --dist loadfile` passes: 2,283 tests, 11 skipped, in about three minutes, including the slow sampling and numerical-optimisation tests.

This checkout contains no complete fitted model output, so the report check is again a render of the actual report functions against labelled synthetic fixtures rather than of study results. Quarto rendered a page exercising a healthy fit, a fit whose diagnostics payload and LOO summary are corrupt, a fit whose calibration table is the column-less file described above, and an absent fit. The three states read as intended: the healthy fit renders its gate verdict and LOO table; the damaged one says each artefact "could not be read (parse_error)"; the absent one says the file is absent. The `_report_data.qmd` helpers were executed against a synthetic figure cache, which covers `val`, `words`, `comp_at`, `comp_row` and `comp_band`.

## Identity, refits and publication

Adopting these helpers changes files across `src/vocab_growth/`, and the executable-code signature (`models/implementation_identity.py`) hashes every module in the package together with the installed version of `dse-research-utils`. Both moved. Every existing fit therefore fails the signature check under `resume`, `sync` and `publish`, and needs a reporting-quality refit before its results are syndicated into the report or built on. That is the intended behaviour of that check and is not bypassed here; `render` and `provisional-sync` do not ask for the signature, so an existing fit can still be re-rendered and reviewed.

No manifest is rewritten. Historical manifests remain records of the code that actually produced them.

Model definitions are untouched, so `fit_manifest.json`'s `model.definition` payload is unchanged and no definition-level refit is implied beyond the signature. The verified equivalences above say what a refit should reproduce: the same administration likelihood arrays and pointwise LOO, the same calibration tables, and the same HSGP basis. A refit is required because the code that produces them moved, not because the numbers did.

Before updating published results, refit at reporting quality, regenerate the comparison tables so they carry manifests fingerprinting the new fits, and re-render. No reporting-quality refits and no publication uploads were performed for this migration.
