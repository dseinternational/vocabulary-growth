# CI critical path: where the time goes, what changed, and the floor that remains

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Investigation and implementation record, 28 August 2026. Follows the 2026-08-24 test-suite pass ([202608241530](202608241530-test-suite-performance.md)), which parallelised the suite and silenced the reporting side effects; this note measures what is left, applies the changes that pay, and records — with the mechanism — the one recommended change that did not survive scrutiny and was therefore not made.

## 1. Where the time goes

A representative successful run (push to main, 2026-08-28, wall 6m28s):

| job        | wall      | inside it                                               |
| ---------- | --------- | ------------------------------------------------------- |
| tests      | **6m22s** | checkout + uv sync + data prep ≈ 12 s; **pytest 6m08s** |
| model-fit  | 2m40s     | VG01 `dev` fit 2m21s, in parallel                       |
| spellcheck | 17 s      | npm cache warm                                          |
| lint       | 7 s       | ruff-only dependency group                              |

The infrastructure is already tight — the uv and npm caches are warm, `prepare_data` is 3 s, pull-request runs cancel on new pushes, and the runners are the free `ubuntu-26.04-arm` ones — so the wall is the pytest step, and nothing else is worth touching.

Since the 2026-08-24 pass the suite has grown from 960 to 1,146 tests (+20% in four days) and the CI pytest step reads 342 s.

## 2. The floor is compilation, not breadth

`--durations` over the full suite (32-core workstation; the ranking transfers to the runner even though the absolute numbers shrink there by core count) puts two items far above everything else, and both are single serial units that `-n auto` cannot split:

| unit                                                 | local         | what it is                                                                                                       |
| ---------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `test_observation_deterministics` `two_fits` fixture | 260 s (setup) | two real nutpie fits of the full VG07 graph — 24 rows, 12 draws + 12 tune, 2 chains                              |
| `test_subject_marginal_sampling` (one test)          | 248 s (call)  | one nutpie fit of the marginalised VG12 graph — 8 draws, 1 chain — plus an object-mode posterior-predictive pass |

The next tier is ≤38 s per test. At those draw counts the sampling itself is seconds, so both giants are **nutpie/numba compile time**, and that resolves two open questions from [202608241530](202608241530-test-suite-performance.md):

- **Why the PyTensor compile cache measured nothing** (§"did not pay"): nutpie compiles the log-density and expand functions through numba, not through PyTensor's C `compiledir`, so the cache was warming the wrong compiler. (The predictive passes do use PyTensor compilation, which is why the cache touched anything at all.) numba has no usable persistent cache here either — nutpie's functions are generated per model object.
- **Why the wall is "bounded by its slowest module"**: it is bounded by these two compile units specifically. Worker rebalancing, more cores, and `--dist loadgroup` all pack the remainder better and leave the wall where it is.

**The recommended fix that was not applied.** The 2026-08-28 review suggested a test-local VG07 definition with a smaller HSGP basis to cut the `two_fits` compile. Checked before implementing: numba's compile cost is structural — it scales with the number of ops in the fused graph, and the basis size `m` (like the row count and the grid lengths) enters as an array dimension, not as ops. A smaller basis changes what is multiplied, not what is compiled, so the change cannot pay and was not made. The fixture's draws are already trivial and the engine's `sample` stage is `pm.sample` plus trace attributes, so there is no local fat; a real reduction needs either upstream persistent compilation in nutpie or accepting the floor. Both tests stay exactly as they are — they validate nutpie's own behaviour (the `var_names` expand function; the quadrature graph end to end), so swapping the sampler or shrinking the graphs would change what is being tested.

## 3. What changed (this commit)

1. **Documentation-only changes skip the heavy jobs.** A `changes` gate job diffs against the push or pull-request base and classifies the changed paths in a plain `case` statement (deliberately not a third-party glob action: the semantics are readable and were tested against fourteen path fixtures). `tests-fast`, `tests-slow` and `model-fit` run only when something outside `notes/`, `docs/` and Markdown changed. Two carve-outs always count as code: the three agent-instruction copies (`tests/test_environment_locks.py` compares them byte for byte) and `docs/models/**` (the fit pipeline's report stage copies those files). Anything unrecognised counts as code, so the gate fails toward running the tests. 8 of the 19 main commits before this change were documentation-only and each paid the full pipeline; they now pay ~30 s (spellcheck + lint + the gate). Skipped jobs satisfy required status checks, so the gate cannot block a merge.
2. **The test job is split into `tests-fast` (`-m "not slow"`) and `tests-slow` (`-m slow`).** The union is exactly what `-m "slow or not slow"` selected. This buys feedback latency, not total wall: most failures now surface from the fast half in a couple of minutes instead of queueing behind the compile floor, and each half gets its own four vCPUs.
3. **`--durations=25` on both pytest commands**, so the next 20% of suite growth is visible in every run rather than discovered by the next investigation.
4. **The single-entry OS matrix is gone** from the test and fit jobs (it renamed the jobs to `… (ubuntu-26.04-arm)` for no information). None of the renamed jobs is a required status check, so nothing references the old names.
5. **The three agent-instruction copies** now describe the two-job split; the local everything-command (`-m "slow or not slow"`) is unchanged.

## 4. Deliberately not done

- **`--dist loadgroup` rebalancing.** `test_kappa_conditional_calibration`'s 21 independent optimiser fits are needlessly serialised by `--dist loadfile`, and grouping only the fixture-sharing modules would spread them. Worthless while the two compile units set the wall; worth revisiting only if the floor ever drops below ~90 s.
- **Bigger runners.** Paid even for public repositories, and the floor is single-threaded compilation — more cores move nothing.
- **Compile caches.** Measured dead on 2026-08-24 and now explained (§2); do not re-add.
- **Shrinking or splitting the two floor tests.** No local knob pays (§2), and the property each validates needs the graph it uses.
- **Required status checks.** `main` requires only `spellcheck` and `lint` — a red test job does not block a merge today. Making `tests-fast`/`tests-slow` required is an owner decision (repository settings, not a file in this tree); the gate above is compatible either way, because skipped runs satisfy required checks.

## 5. Expected outcomes

Documentation-only pushes and PRs: ~6.5 min → **~30 s**. Code changes: total wall unchanged (~6 min, the compile floor) until nutpie grows a persistent cache or the floor tests are redesigned, but the fast half reports in roughly 2–3 minutes and carries most of the failure surface; `model-fit` (2m40s) becomes the binding floor if the slow half ever shrinks. Re-measure after a week of runs rather than trusting these projections — the 2026-08-24 note's local projections lost a third of their value on contact with the runner.

## 6. Reproduction

Job and step timings: `gh run list --workflow CI` and `gh run view <id> --json jobs`. The durations table: `uv run pytest -n auto --dist loadfile -m "slow or not slow" --durations=40 -q`, run 2026-08-28 on the 32-core workstation; the 342 s / 1,146-passed figure in §1 is from the CI step's own log, not that run. The docs-only share: `git log --format='%H' -30 main` with `git diff-tree --no-commit-id --name-only -r` per commit, classifying against the same rules as the gate. The classification fixtures: fourteen path lists run through the gate's `case` statement verbatim.
