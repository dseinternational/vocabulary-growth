# A shared child-effect row primitive: deferred until a third consumer

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Fable 5).

Records an assessment made 2026-09-01 of whether `build_child_slope` (VG19) and `build_child_factor` (VG22), both in `src/vocab_growth/models/gp_utils.py`, should share their row-construction code — and the decision to defer, with the trigger that reopens it. No code changed.

## 1. The shared mathematics is real, and closer than the two docstrings suggest

VG19's per-outcome block is exactly the factor's two anchor-row types at rank 2. In `build_child_slope`, `b0 = tau0 * z[:, 0]` is the constant-`e_0` first-anchor row, and `b1 = tau1 * (rho * z0 + sqrt(1 - rho**2) * z1)` is the LKJ-form second-anchor row that `build_child_factor` special-cases — whose own comment says "written as VG20 writes it". Both builders are marginal SDs times unit-direction rows dotted with a standard-normal `z`, both emit the same `b0_{name}` / `b1_{name}` / name-preservation deterministics, and both close over `(age - ref) / 12`. The three-line LKJ idiom (`Beta(eta, eta)` → `2x - 1` → `rho * a + sqrt(1 - rho**2) * b`) appears at three sites: VG20's inline construction in `common_bivariate_re.py`, `build_child_slope`, and `build_child_factor`'s second-anchor branch.

A shared primitive is therefore well defined: a row-spec union — constant-`e_0`, LKJ(`eta`, name), normalised-normal(width, diagonal) — with `build_child_slope` becoming a two-effect call and `build_child_factor` a four-effect call. It would be a pure graph-code refactor: fit identity compares serialised definitions, not the builders, so no fit of record is invalidated, and `tests/test_graph_equivalence.py` pins free-RV names and order, dims, coords and a fixed-point log probability per model against a committed baseline — exactly the tool for verifying such a refactor landed op for op.

## 2. Why VG19 cannot simply route through `build_child_factor`

- **Representational.** Two independent 2x2 blocks are a block-diagonal 4x4, which the factor form cannot express below rank 4 — VG22's registration already documents that VG19 "is not nested here at any rank below 4". The factor builder has no notion of structural zeros beyond its triangular anchor pattern, so supporting VG19 means adding a block-structure argument, i.e. building the shared primitive anyway.
- **Names and dims are load-bearing.** VG19's trace carries `tau_subj_u_z` with dims `(subject_id, child_effect)` and `{name}_rho`; VG22 carries `subject_factor_z` with `(subject_id, factor)` and `rho_uq`. The summaries, the recovery scorer and the comparison suite read those names, and the graph baseline pins them.
- **The priors legitimately differ off-anchor.** The factor's general rows are normalised normals (deliberately, after the sphere-chart ESS measurement recorded in its comments); VG19's second row is explicit LKJ form.
- **A one-op wrinkle, which is the cost/benefit argument in miniature.** `build_child_factor`'s LKJ row guards its square root with `maximum(1 - rho**2, 1e-12)`; VG19's and VG20's do not. Even the smallest shared helper must either carry a per-site flag or move at least one model's graph fingerprint. Do **not** unify the guard as a drive-by inside a refactor sold as op-for-op.

## 3. Decision and trigger

Deferred. The duplicated arithmetic is roughly thirty lines across two call sites, each wrapped in site-specific reasoning (the anchor-order gauge argument, the name-preservation contracts) that would not move into a shared function, and the house style is written-out Cholesky with the rationale at the construction site.

**The trigger is a third consumer.** Candidates visible from here: a VG23-style child slope on the TD pool, a factor or slope structure on the trivariate (VG15-family) engine, or the Proposal A1 age-varying interaction. When one of those is registered, extract the row primitive first — parameterising the square-root guard per site so the two existing models' baselines do not move — and build the new model on it, rather than adding a third hand-written copy and refactoring afterwards.
