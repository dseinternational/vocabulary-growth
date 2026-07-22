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
