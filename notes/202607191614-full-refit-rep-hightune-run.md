# Full reporting-config refit of VG01–VG16 (rep + rep-hightune) — run record

> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 4.8).

Restart of the deliberately-paused 2026-07-17 run (`202607170935-full-refit-vm-run-147-163.md`) on the updated code (`main` @ `3e6f61d`, post-#168/#170/#171). Full reporting-quality refit of every registered model, render every report, and upload to `dseresearch/public`. This run carries the paused run's durable findings, most importantly: **the hierarchical TD models (vg11/vg12/vg13) are fitted at `rep-hightune`, not `rep-lite`** (their #164 child random effects re-introduced the trend/GP/study-intercept ridge that `rep-lite` under-converges).

## Environment and method

- **Machine:** Azure VM — 32 cores, 251 GB RAM, 410 GB free on `/scratch` (repo lives on `/scratch`). RAM is ample; the binding constraint is the 32-core CPU. No OOM risk on the full-data TD fits.
- **Env:** conda `dse-vocab-growth`; `dse-check-env environment.yml` clean (conda-forge core matches canonical spec). `dse-research-utils` ≥ 0.6.0 → unrounded R-hat/ESS gate (no false-pass rounding).
- **Code:** `main` @ `3e6f61d`, clean tree.
- **De-risk:** `dev` smoke fits of `vg05` (DS joint) and `vg13` (TD joint, hierarchical) built and initialised cleanly on the current code; `vg05` cleared the full pipeline (sample → diagnostics → **PSIS-LOO, no degenerate-n=0 crash** → posterior predictive → plots), confirming the #165/#170 changes are healthy end-to-end. Smoke output discarded.
- **Data:** re-prepared at run start — 1,026 raw rows → 1,221 analysis rows; DuckDB + `vocab_data_merged.csv` current.
- **Output root:** repo-local `output/` (gitignored; keeps quarto rendering in-repo so `code-links: [repo]` resolves).
- **Configs:** `rep` = 6 chains / 6000 tune / 6000 draws / ta 0.95. `rep-hightune` = 6 chains / 12000 tune / 8000 draws (ta 0.99 for vg13, 0.95 for vg11/vg12). A `rep-hightune` fit validates as `rep`-compatible because draws/tune/chains/target_accept are minimum requirements (`fit_artifacts._sampling_parameter_errors`).
- **Execution (Phase A):** `output/fitrun/fit_driver.sh` — 3 background `rep-hightune` fits (vg11/vg12/vg13, 18 cores) concurrent with a thread-pinned 2-wide pool of the 10 DS models + vg03/vg04 at `rep` (12 cores). ~30 of 32 cores. Resumable (each wrapper skips a model with a compatible complete fit). Detached via `setsid` so it survives disconnects. Logs `output/fitrun/logs/<model>.log`, status `output/fitrun/status.tsv`.

## Model split

| Config         | Models                                                      | n   |
| -------------- | ----------------------------------------------------------- | --- |
| `rep`          | vg01 vg02 vg03 vg04 vg05 vg07 vg08 vg09 vg10 vg14 vg15 vg16 | 12  |
| `rep-hightune` | vg11 vg12 (ta 0.95), vg13 (ta 0.99)                         | 3   |

## Plan (scope confirmed by user 2026-07-19)

- Phase A — fit all 15 (above).
- Phase B — verify convergence gate (unrounded); `run_replication.sh --config rep --no-fit --no-descriptives --no-upload` → validate, comparisons, render-only per model, sync figures, render `docs/report` + `docs/comparison`.
- Phase C — upload to `dseresearch/public` (needs interactive `az login`; `AZURE_TOKEN_CREDENTIALS=dev`).
- Not in scope this run: #147 Target-8 / #163 P0-P1 sensitivity study, parameter-recovery/SBC (deferred; can follow).

## Progress log

- **2026-07-19 ~16:14 UTC** — data re-prepared; env + dev smokes clean; Phase A driver launched (pid 9468). vg11/vg12/vg13 hightune + vg01/vg02 rep started.

## Incident: session death mid-run + dirty-manifest cleanup (2026-07-20 ~08:30 UTC)

- **~22:02 UTC (07-19):** DS/TD-baseline rep pool complete; vg12 hightune complete (20:48). Driver waiting on vg11/vg13 hightune.
- **Overnight:** the SSH/tmux session died; `systemd-logind` killed all user processes — including the `setsid`-detached driver (setsid does not protect against `KillUserProcesses`). vg11/vg13 interrupted mid-sampling; the monitor was killed. Machine itself stayed up (no reboot).
- **On disk after death:** 13/15 `state=complete` with traces (all except vg11/vg13, which left no dirs). BUT the fit manifests recorded **inconsistent `code.dirty` flags**: only vg01/vg02/vg12 were `dirty=False`; the other 10 were `dirty=True`. Cause: the run-notes file was written into the working tree (`notes/…md`) mid-run, so every model whose manifest was stamped after that captured an untracked-file dirty tree. `require_clean_fit` (publish gate) rejects `dirty=True`, so those 10 are not publishable.
- **Fix:** relocated the notes file out of the repo (tree clean at `3e6f61d`); deleted the 10 `dirty=True` dirs; kept vg01/vg02/vg12 (`dirty=False`). Refit set = vg03,04,05,07,08,09,10,14,15,16 (rep) + vg11 (ta 0.95), vg13 (ta 0.99) (hightune). **12 refits; vg12 hightune kept (saved a ~4.5h refit).**
- **Robustness fix:** relaunched the driver under `sudo systemd-run --unit=vgfitrun --uid=azureuser` (system-managed transient service) so it survives any future SSH/tmux/session death — the setsid+nohup approach did not. Restart at **2026-07-20 08:33 UTC**: vg11+vg13 hightune + 3-wide rep pool.
- **Lesson (carry to runbook):** never write into the working tree during a reporting run — it dirties in-flight manifests and blocks publication. Keep run notes outside the checkout until fitting completes. And launch long detached runs via `systemd-run`, not bare `setsid`/`nohup`, on hosts with logind `KillUserProcesses=yes`.

## Incident 2: OOM-kill of the concurrent hightune fits (2026-07-20 13:16 UTC)

- **13/15 rep fits refit clean** (`dirty=False`, all pass R-hat/ESS; soft-gate caveats: vg01 3 div, vg02 1 div, vg12 2 div + BFMI 0.287). vg12 hightune kept from run 1.
- **~13:16 UTC:** `vgfitrun.service` **OOM-killed** — memory peak **248.3 GB / 251 GB**. The rep refit pool finished 12:30; from then vg11 (16,235 obs) + vg13 hightune ran concurrently, and the vg11 posterior-predictive/LOO spike (its trace alone scales ~2.7× vg12's 25 GB) on top of vg13 holding its draws exceeded RAM. Both hightune fits + the driver died (unit killed → no FIT_DONE).
- **Why run 1 didn't OOM here:** run 1 was killed by logind (session death) before vg11 reached its PP peak; run 2 additionally stacked a 3-wide rep pool (incl. vg03 @ 16k obs) under the hightune fits, pushing the peak higher.
- **Mitigation (`fit_hightune_seq.sh`, `vghightune.service`):** run the two hightune fits **sequentially, alone** — vg13 first (draws 8000, ta 0.99; small data), then **vg11 at draws 6000** (ta 0.95). draws 6000 is still ≥ the rep minimum (rep-compatible) and tune stays 12000 (the ridge fix is tuning, not draws); ESS at 36k draws clears 400 with huge margin, and the ~25% lower draw count trims the PP/LOO peak. Armed a memory-headroom monitor as a backstop.
- **Lesson (carry to runbook):** the hierarchical TD hightune fits are memory-heavy; **do not run vg11 hightune concurrently with other large fits** on a 251 GB box. Fit vg11 alone, and consider draws 6000 to cap the PP/LOO peak.

## Incident 3: vg13 failed the R-hat gate at standard hightune (2026-07-20 17:14 UTC)

- **vg13** (young TD joint, 5,406 rows, 4 studies) **failed** at tune 12000 / draws 8000 / ta 0.99: **6 R-hat failures, max 1.0135**, 0 ESS failures, **0 divergences**, min BFMI 0.284. Failing params: `p_slope_low_u`, `p_slope_hi_u`, `intercept_u`, `delta_u_raw[0/2/3]` — the **understood trend/intercept + study random-intercept ridge**. ESS min 459 (barely passing) → slow-but-valid mixing of a weakly-identified block (few studies, narrow 8–18 mo age band). This is the first _post-#164_ (hierarchical) vg13 hightune attempt; July's ta-0.99 success was on the pre-#164 model.
- **Diagnosis:** 0 divergences ⇒ not a step-size problem; the lever is more tuning + more draws (ridge mixing), not higher ta.
- **Retry (queued):** tune 20000 / draws 12000 / ta 0.99 / chains 6, via `fit_vg13_strong.sh` (`vgvg13.service`), which **waits for vg11 to finish first** — vg13's 39 GB trace makes it as memory-heavy as vg11, so they cannot co-reside on the 251 GB box.
- **If the strong retry still fails (>1.01):** genuine decision point for the user — (a) accept-with-caveat and document vg13's understood-trend block as marginally non-converged, (b) exclude vg13 from the published set, or (c) reparameterise (sum-to-zero/non-centred study REs — a code change). Not doing any of these unilaterally.

## Incident 4: BOTH hierarchical TD models fail the R-hat gate at hightune (2026-07-20 22:17 UTC)

- **vg11** (TD spoken, full-data) failed at tune 12000 / draws 6000 / ta 0.95: **7 R-hat failures, max 1.0161**, ESS 500 ✓, 0 div, BFMI 0.383 ✓. Failing: `slope`, `intercept`, `p_slope_hi`, `delta_raw[3/4/5/6]` (dataset/study REs). ~8h56m wall. No OOM (the draws-6000 memory mitigation worked; vg11 ran to completion and failed only on the gate).
- This is the **same structural ridge as vg13**: the global **trend (slope/intercept) is weakly identified against the dataset/study random intercepts that #164 added**. The documented hightune remedy (tune 12000 / draws 8000) does **not** clear it post-#164 for either model (vg11 1.016, vg13 1.013). vg12 passed R-hat only marginally (1.006) with sub-threshold BFMI — same family.
- **Evidence test running:** vg13 strong retry (tune 20000 / draws 12000 / ta 0.99) via `vgvg13strong.service` — does much heavier sampling cross 1.01? Result will decide the strategy for vg11 too.
- **Status: 13/15 clean & gate-passing** (10 DS + vg03 + vg04 + vg12). **vg11, vg13 unconverged.** Phase B (comparisons/render) is blocked on the TD models.
- **Likely decision for the user** (pending vg13-strong result): (A) escalate sampling further on vg11+vg13 (costly, uncertain); (B) publish the clean set now and defer vg11/vg12/vg13 to a follow-up; (C) reparameterise the study REs (sum-to-zero/centred — a reviewed model change) — the proper structural fix; (D) accept-with-caveat (requires bypassing the hard R-hat gate; not recommended for a knowingly non-converged fit).

## Incident 5 + decision point: heavier sampling is memory-infeasible (2026-07-21 04:29 UTC)

- **vg13 strong retry (tune 20000 / draws 12000 / ta 0.99) was OOM-killed** at 04:28 (peak 248 GB) mid-sampling — never reached diagnostics. At draws 12000 × 6 chains the joint model (per-subject REs `delta_subj_*_raw` + PP on 5,406 rows) exceeds 251 GB. **draws 8000 (~39 GB trace) is the practical memory ceiling** for these models on this box, and at draws ≤8000 they fail the R-hat gate (vg13 1.013, vg11 1.016).
- **Conclusion:** the vg11/vg13 non-convergence is a **structural ridge** (global trend vs #164 dataset/study REs), not fixable by brute-force sampling here — and heavier sampling is memory-infeasible regardless. The proper fix is a **reparameterisation of the study/dataset REs (sum-to-zero / centred)**, a reviewed model change.
- **Authoritative state: 13/15 complete, all `dirty=False`, all pass the hard R-hat/ESS gate.** 11 fully clean; vg01 (3 div), vg02 (1 div), vg12 (2 div + BFMI 0.287) pass the hard gate with soft caveats. **vg11, vg13 unconverged / not complete.**
- **Compute stopped. Awaiting user decision:** (A) publish the 13 now, defer vg11/vg13; (B) reparameterise the REs and refit the TD trio before publishing; (C) one more memory-viable heavier attempt (low confidence).

## Structural fix: sum-to-zero study REs (user chose "reparameterise", 2026-07-21 ~10:30 UTC)

- **Change (commit `b563586`, branch `fix/study-re-sum-to-zero`):** study-level random intercepts now use `pm.ZeroSumNormal` for the unit offsets (`delta_raw = ZeroSumNormal(1)`, `delta = tau * delta_raw`), removing the intercept vs study-RE-mean ridge while keeping #65's funnel-avoiding non-centring and all public names. Applied to `common_univariate_re` (vg11/12), `common_bivariate_re` (vg07-10/13/16), `common_joint_modality` (vg15). Subject REs unchanged. Added a sum-to-zero guard to `test_study_re_parameterisation.py`. **Full suite: 232 passed.** ruff clean. vg11 dev fit built + sampled clean on the new path.
- **Refit scope:** only the **9 affected study-RE models** (vg07,08,09,10,11,12,13,15,16) at **plain rep** — the ridge fix should converge them without hightune. The 6 unaffected models (vg01,02,03,04,05,14; no study REs) are kept as-is at commit 3e6f61d (`dirty=False`) — provably identical under the change; avoids re-running the 16k-obs vg03. **Published set will span two commits (documented; scientifically coherent).**
- **Execution (`vgreparam.service`, `fit_reparam.sh`):** 3-wide rep pool {vg07,08,09,10,15,16,12} + background SEQUENTIAL heavy {vg13 then vg11}. At rep (6000 draws) peak stays well under 251 GB. Started 10:33 UTC.
- **Watching:** do vg11/vg13 (and vg12) now pass at plain rep? vg13 is the first TD test (~2h).

## Reparam validated; vg13 needs hightune ta 0.99 for a DIFFERENT block (2026-07-21 ~12:02 UTC)

- **The sum-to-zero reparam WORKED.** vg13's failing parameters changed completely: from the study-RE ridge (`p_slope_*_u`, `intercept_u`, `delta_u_raw[...]`) to the **dispersion block** (`kappa_min_u`, `a_kappa_u`, `b_kappa_mag_u`, `b_kappa_u`). The ridge is gone. All 6 DS RE models refit clean at plain rep (e.g. vg08 maxRhat 1.0026, BFMI 0.53, dirty=False @ b563586).
- **vg13 at plain rep FAILED** on the dispersion block: max R-hat 1.037, min ESS 107, **28 divergences**, BFMI 0.264. This is vg13's _separate, known_ difficulty (age-varying dispersion on a narrow young-TD age range) — the same block that needed **ta 0.99** in July. Plain rep (tune 6000, ta 0.95) can't clear it.
- **Plan:** refit vg13 at **hightune ta 0.99** (tune 12000 / draws 8000) on the reparam code — with the ridge gone, the heavy tuning + high ta should now resolve the dispersion (previously hightune was consumed fighting the ridge). Memory-OK at draws 8000 (~39 GB), run alone.
- **vg11, vg12 still fitting at plain rep** — awaiting their verdicts to see if they also need ta 0.99 or pass at rep.

## vg12 confirms: reparam fixed the study-RE ridge; TD trio needs hightune for trend/GP (2026-07-21 ~12:46 UTC)

- **vg12 at plain rep FAILED** on the **trend/GP block** (`p_slope_low`, `p_slope_hi`, `slope`, `intercept`, `g_unit_hsgp_coeffs[3]`), max R-hat 1.013, ESS 569, 4 div, BFMI 0.283. The study-RE `delta_raw` is **absent** from the failures — the reparam fixed that ridge here too.
- **Consistent finding across the TD trio:** the sum-to-zero reparam removed the study-RE ridge; what remains is the **standard trend-vs-GP redundancy** (vg11/vg12) and vg13's **dispersion + divergences**. These are the ordinary blocks that heavier tuning clears (runbook: DS understood-GP ridge cleared by hightune, e.g. vg16 1.024→1.009). Crucially, hightune's heavy tuning is no longer consumed fighting the study-RE ridge, so it should now land for the TD models.

## Current status + next steps (as of 2026-07-21 ~13:00 UTC)

**Complete & publishable (13/15), all `dirty=False`:**

- Baselines/unaffected @ 3e6f61d: vg01, vg02, vg03, vg04, vg05, vg14.
- Study-RE models refit @ b563586 (sum-to-zero): vg07, vg08, vg09, vg10, vg15, vg16.
- Soft caveats only (pass hard gate): vg01 (3 div), vg02 (1 div).

**Outstanding (TD trio) — need reparam + hightune:**

- vg12 (TD understood): fail at rep on trend/GP (1.013) → refit hightune tune 12000 / draws 8000 / ta 0.97.
- vg13 (TD joint young): fail at rep on dispersion + 28 div (1.037) → refit hightune tune 12000 / draws 8000 / **ta 0.99**.
- vg11 (TD spoken, 16k obs): fitting at plain rep now; if it fails on trend/GP → refit hightune tune 12000 / **draws 6000** (memory) / ta 0.97.
- Run the hightune refits **sequentially** (memory: vg11 16k + vg13 joint are heavy; never stack).

**Then:** Phase B (comparisons + render docs/report + docs/comparison) → Phase C (upload to dseresearch/public; needs `az login`).

**Provenance note:** published set will span commits 3e6f61d (baselines, code-identical) + b563586 (reparam) + the branch HEAD after this note. All docs/reparam commits are code-equivalent for each model's engine.

**Infra lessons captured:** (1) never write into the working tree during a reporting run (dirties in-flight manifests); (2) launch long runs under `systemd-run` (system scope), not bare `setsid`/`nohup` (logind `KillUserProcesses` kills the latter on logout); (3) the hierarchical TD hightune fits are memory-heavy — draws 12000 OOMs on 251 GB; keep draws ≤ 8000 and never run two heavy fits concurrently.

## GP orthogonalisation added, then TD trio landed at hightune (2026-07-21 late → 2026-07-22)

- **GP orthogonalisation (commit `5397561`).** To remove the trend-vs-GP redundancy structurally (rather than lean on hightune), the anchored HSGP was changed from a single-point anchor (`g_unit - g_unit[idx]`) to a **whole-grid** orthogonalisation: project the constant AND linear-in-age components out of `g_unit` over the full `X_all_z` grid. This improved the already-passing DS anchored models — **vg10** R-hat 1.0027 → 1.0013, min ESS 2135 → 5111; **vg15** R-hat 1.0035 → 1.0011, min ESS 2050 → 6428 (0 divergences both). (These before/after numbers are from the `5397561` whole-grid form; see the correction below — they are superseded and will be re-derived.)
- **TD trio** refit at hightune on the reparam + GP-orthogonalisation code: all three cleared the hard gate. Final soft caveats: **vg12** BFMI 0.29, **vg13** BFMI 0.28 (both pass R-hat/ESS); vg11 clean. All 15 models then passed the hard gate (11 fully clean, 4 soft-caveat: vg01 3 div, vg02 1 div, vg12/vg13 low BFMI).
- Phase B (comparisons + render) completed; PR #176 opened.

## Statistical review of #176 (PR reviewer via Codex/GPT-5, 2026-07-23) — four P1 + two P2

The review found the two core conditioning commits were **correct in intent but flawed in implementation**. All six points verified against the code and accepted:

- **P1-A — the whole-grid orthogonalisation leaked the reporting grid into inference.** `X_all_z` stacks observations with plot points, query ages and the anchor row; computing the projection statistics (`g.mean()`, the `z`-slope) over that whole stack makes the _observed-row_ latent — hence the likelihood — depend on `n_plot` / `ages_query`. A reporting-only choice must never move the posterior.
- **P1-B — the whole-grid form broke the `anchor_g*_at_ref` point-anchor contract** (`g` no longer passed through zero at the reference age, contradicting the docstrings, the `definitions` field and the build output), and for the `tent` mean it projected against only `[1, z]`, a smaller basis than the three-anchor tent spans.
- **P1-C — the sum-to-zero RE change was mis-described as prior-preserving.** `ZeroSumNormal(1)` has covariance `I − J/K` (marginal `1 − 1/K`, a 25 % shrink at K=4, plus a −1/K correlation); `tau` unchanged does not restore it.
- **P1-D — the sign study zero-sum spanned all studies, not the sign-informed subset.** Uninformed studies' `z_sign` could counterbalance a common shift among the informed ones, so the constraint need not remove the sign-intercept ridge.
- **P2-A** — the new guard tests skip in CI (`pytest` runs before `prepare_data.py`). **P2-B** — this note stopped before the GP-orthogonalisation commit and final validation.

## Corrected fixes on the branch (2026-07-23)

Implemented and unit-tested (full suite green; ruff clean). **No model definitions or priors semantics beyond the intentional, now-documented RE constraint.**

- **GP anchor (P1-A + P1-B), `gp_utils.py`.** Projection coefficients are now fitted on the **observed rows only** (`n_obs`, a fixed model design) and applied to every row, so plot/query grids cannot leak into inference; the residual is then **pinned to zero at the anchor row**, restoring the point-anchor contract. The nuisance basis is **mean-specific**: `[1, z]` for the logit-linear trend, `[1]` for the free-intercept mean (a linear GP direction there is genuine signal), and the **three tent hats** for the peak mean. New data-free unit tests (`test_gp_anchor_orthogonalisation.py`) assert anchor-zero, observed-row orthogonality, reporting-grid invariance, and the intercept/tent basis behaviour; they run in CI.
- **Sum-to-zero rescale (P1-C).** `ZeroSumNormal` sigma is rescaled by `sqrt(K/(K−1))` so each study effect's marginal prior variance stays `tau²`; only the group-mean DOF (the ridge) and a −1/K correlation remain. Relabelled in code/PR as an intentional identifiability constraint, not prior-preserving. Applied in `common_univariate_re`, `common_bivariate_re`, `common_joint_modality`.
- **Sign zero-sum over informed studies (P1-D), `common_joint_modality.py`.** `delta_sign` is now `ZeroSumNormal` over the **sign-informed studies only** (union of `idx_sign` / `idx_cells` / `idx_prod`), scattered into full `study_id` with **0 for uninformed studies**. Verified on VG15: of 12 studies, 5 are active and zero-sum, 7 are forced to exactly 0. U and q remain global (audited: informed by every retained study via their direct + nested terms).
- **P2-A** — added the synthetic CI tests above; updated the integration test's assertions to the new (observed-row, point-anchored) contract.

**Refit scope (pending).** The GP-anchor and sign-zero-sum changes alter the likelihood, and the RE rescale shifts the prior scale, so **every study-RE and every anchored model must be re-fit** before publication: the DS RE/anchored models (vg07–10, 15, 16) and the TD trio (vg11–13) at least. The whole-grid VG10/VG15 before/after numbers above are **superseded** and will be regenerated from the corrected fits. **Phase C (upload to `dseresearch/public`) is held** until the re-fit is clean. Baselines with no study REs and no anchor (vg01–05, 14) are unaffected by these changes.

## Clean re-fit + Phase B on the corrected code (2026-07-23/24)

**All 9 affected models converged at plain `rep`** — the corrected fixes removed the ridges structurally, so the TD trio no longer needs hightune (vg11/12/13 previously failed plain `rep` and required hightune; vg12 got _worse_ at hightune before the fixes). Diagnostics (commit `d2f2b96`, all `dirty=False`):

| model | hard gate | max R-hat | min ESS | div | min BFMI | soft caveat |
| ----- | --------- | --------- | ------- | --- | -------- | ----------- |
| vg07  | PASS      | 1.0005    | 8434    | 0   | 0.78     | —           |
| vg08  | PASS      | 1.0020    | 1899    | 0   | 0.54     | —           |
| vg09  | PASS      | 1.0033    | 1684    | 0   | 0.49     | —           |
| vg10  | PASS      | 1.0016    | 5020    | 0   | 0.50     | —           |
| vg15  | PASS      | 1.0012    | 4451    | 0   | 0.59     | —           |
| vg16  | PASS      | 1.0062    | 1630    | 0   | 0.50     | —           |
| vg12  | PASS      | 1.0083    | 451     | 0   | 0.28     | BFMI ~0.28  |
| vg13  | PASS      | 1.0041    | 1215    | 0   | 0.28     | BFMI ~0.28  |
| vg11  | PASS      | 1.0058    | 621     | 2   | 0.40     | 2 div       |

The DS six are pristine (0 div, BFMI 0.49–0.78; vg09 0.495 and vg16 0.496 sit just under 0.5). The TD trio's BFMI ≈ 0.28 is intrinsic to the age-varying dispersion posterior over the narrow young-TD window — the same soft caveat accepted last run, now reached at plain `rep`. vg10 (DS anchored) is well conditioned (min ESS 5020), confirming the corrected obs-only anchor matches the flawed whole-grid form it replaced. These numbers supersede the whole-grid VG10/VG15 metrics quoted earlier.

**Provenance gap found and fixed.** The first corrected re-fit recorded `dirty=True` on every manifest: `git_metadata` computes dirty from `git status --porcelain --untracked-files=normal`, and 68 untracked `docs/comparison/*.png|svg` figure copies (left over from an earlier render) were in the tree at launch — `.gitignore` covered `*.csv`/`index.html` but not the plots. Fixed in `d2f2b96` (ignore the plot copies + report LaTeX byproducts), cleaned the tree, and re-fit on a porcelain-clean tree so all manifests are `dirty=False`. Lesson: the "never fit on an unclean tree" rule includes untracked build artifacts, not just modified tracked files.

**Phase B complete.** All 15 model reports rendered; comparison suite (incl. the now-guarded `loo_compare`) ran clean; figures synced; report + comparison books rendered. `check_fit --purpose publish` passes for all 15 (dirty=False + reporting quality + rendered). **Phase C (blob upload) held** pending reviewer sign-off on PR #176.

## Phase C complete — published to `dseresearch/public` (2026-07-24)

All 15 model reports uploaded via `scripts/upload.py all --config rep` (traces excluded; `AZURE_TOKEN_CREDENTIALS=dev`, `DSERESEARCH_BLOB_CONTAINER_URL=https://dseresearch.blob.core.windows.net/public`). Each report is at a UUID-versioned path `.../public/projects/vocabulary-growth/output/<id>/<model-label>/index.html`:

| model | report URL                                                                                                                                                                           |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| VG01  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b0-86f2-7762-8560-518939d1ca12/VG01-age-spoken-ds/index.html                                |
| VG02  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b0-9b34-7209-b080-6b7a4aa8e287/VG02-age-understood-ds/index.html                            |
| VG03  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b0-afb6-7328-a803-b236b53bf7a6/VG03-age-spoken-td/index.html                                |
| VG04  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b0-c1ba-711c-a64a-6a6c1d450569/VG04-age-understood-td/index.html                            |
| VG05  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b0-d3b4-72ea-8e37-9a070c17e996/VG05-age-understood-spoken-ds/index.html                     |
| VG07  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b0-f96f-752c-a590-8fe04bd877df/VG07-age-understood-spoken-ds-re/index.html                  |
| VG08  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-1f08-7594-8d8a-dfce3caa8cac/VG08-age-understood-spoken-ds-re-subj/index.html             |
| VG09  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-4572-773c-a2e1-0733c00a453a/VG09-age-understood-spoken-ds-re-subj-uq/index.html          |
| VG10  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-6ca0-74b3-af97-f08b709e1656/VG10-age-understood-spoken-ds-re-subj-uq-anchored/index.html |
| VG11  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-9247-70dc-9e54-abc154e6f4fe/VG11-age-spoken-td-re/index.html                             |
| VG12  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-a49a-7734-8970-1837cc5cf49a/VG12-age-understood-td-re/index.html                         |
| VG13  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-b72b-74ea-b4c9-b1415a5f935f/VG13-age-understood-spoken-td-re-young/index.html            |
| VG14  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b1-d75b-7393-914a-0ec560fc5d8e/VG14-age-understood-spoken-signed-ds/index.html              |
| VG15  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b2-0a5d-73e9-9589-acf892c920e6/VG15-age-joint-signspeech-ds/index.html                      |
| VG16  | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93b2-2106-75b2-8ad6-5476b3bdac6a/VG16-age-understood-spoken-ds-re-subj-uq-crosslag/index.html |

Note: `upload.py` publishes the per-model reports; the consolidated report book (`docs/report`) and comparison book (`docs/comparison`) are rendered locally but not part of this per-model upload path.

## Phase C addendum — consolidated books uploaded (2026-07-24)

The consolidated report book (`output/report`, HTML + PDF + DOCX) and the comparison book (`docs/comparison`, rendered HTML; Quarto source excluded) were uploaded via `upload_directory_to_blob_storage` (shared run id `019f93ff-4065-706e-a592-d9e6839411bc`):

| artefact                 | index URL                                                                                                                                     |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Consolidated report book | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93ff-4065-706e-a592-d9e6839411bc/report/index.html     |
| Comparison analysis      | https://dseresearch.blob.core.windows.net/public/projects/vocabulary-growth/output/019f93ff-4065-706e-a592-d9e6839411bc/comparison/index.html |
