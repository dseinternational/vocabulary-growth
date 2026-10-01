# Refit campaigns on a dedicated VM

> [!NOTE]
> Drafted with assistance from Claude Code/Opus 5.5. It carries forward the lessons of the 2026-07 and 2026-08 VM runs, recorded in the full-refit runbook before 2026-09-15.

<!-- cspell:words Noto -->

The study owner decided on 2026-10-01 that full refits are scheduled on a separate VM, so the project workstation stays free for development. A campaign runs from a clone pinned at the cycle's `fits/YYYY-MM-DD` tag, which is the [pinned-worktree procedure](full-refit.md#5-running-a-cycle-from-a-pinned-worktree) with a whole machine in place of a worktree. Everything in the [full-refit runbook](full-refit.md) about convergence, escalation and publication still applies; this page covers the machine and the campaign scripts.

## What a campaign runs

`scripts/vm/campaign.sh` runs five stages. Each assumes the ones before it.

| Stage | What                                                                                                                                                                                                                               | Rough cost |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| A     | Refit the nine publication-scope models: VG15, VG20, VG24 and VG25 four at a time, then VG11, VG12, VG21, VG23 and VG26 alone                                                                                                      | 15 h       |
| B     | Refit the development steps: eleven DS models in a pool, then VG03, VG04 and VG13 alone                                                                                                                                            | 10 h       |
| C     | The runbook's data-handling arms, every arm of VG16, VG20 and VG25, VG14's fallback arms, VG16's and VG25's wave-forward scores, and recovery for VG20, VG25's and `vg16 corr`'s designed cells, VG11, VG12 and `vg12 free-scales` | 2.5–3 days |
| D     | Every remaining registered arm, recovery for the models never recovered (#289 task 3.3), the Gompertz comparison (#330) and the typically developing held-out k-fold (#240)                                                        | 9–10 days  |
| P     | Comparisons, robustness matrices, figure sync with `--allow-caveats` and the report-book render. **Uploads and publishes nothing**                                                                                                 | 2–3 h      |

The costs are scaled from the September cycles on the workstation and are rough: the k-fold alone is about 100 hours, and every typically developing fit runs alone. Together the stage lists cover every registered model and every registered sensitivity arm exactly once; `tests/test_vm_campaign.py` keeps them in step with the registries.

Publication is not part of any stage. After stage P, the study owner decides whether to upload the model reports and republish the comparison book, as in the full-refit runbook's section 3.

## 1. Provision the VM

- **Size.** At least 32 vCPUs and 128 GB of RAM; 256 GB is comfortable. The DS pool runs five fits of six chains each. The typically developing fits peak at 27–28 GB each and run alone. Prefer x86_64. On aarch64 (the data-science fleet's `Standard_E32pds_v6` tier), the `vg15 fallback-dispersion` arm has failed numba compilation; rerun it with `--nutpie-backend jax` if it does.
- **Disk.** Put the output root on an attached managed disk, mounted at `/data` on the fleet image, never on `/scratch` (wiped on deallocation) or a network filesystem (traces are HDF5). Budget about 1 TB at the `full` trace tier, and 2 TB to keep the previous cycle beside it: 23 models at about 6.6 GB, 118 arms at about 4 GB, about 50 recovery fits at about 5 GB, plus one VG11-sized promotion copy. Staging and promotion rename within the output root, so it must be one filesystem.
- **Swap.** The fleet image ships with none, and the 2026-08-13 run lost four fits to one overrun because of it. Add a backstop on the local NVMe before the first fit:

  ```bash
  sudo fallocate -l 128G /scratch/swapfile && sudo chmod 600 /scratch/swapfile
  sudo mkswap /scratch/swapfile && sudo swapon /scratch/swapfile
  sudo sysctl -w vm.swappiness=10
  ```

- **Lingering.** `sudo loginctl enable-linger $USER`, so the campaign's systemd scope survives the SSH session.
- **Tools.** The fleet image provides `uv`, Python, Node.js, Quarto with TinyTeX, Pandoc, Graphviz and PowerShell 7 as `pwsh`. Check rather than assume the report fonts: Noto Sans, Noto Sans Mono and Noto Sans Math. Install the Azure CLI if absent.
- **Credentials.** `az login --use-device-code` as the person publishing. The VM's managed identity has no write role on the container, and `DefaultAzureCredential` prefers it; `vm.env` sets `AZURE_TOKEN_CREDENTIALS=dev` so the upload path uses the `az login` session instead.

## 2. Tag the cycle

From the main checkout, once every change meant for the cycle has merged:

```bash
git tag fits/2026-10-02 origin/main
git push origin fits/2026-10-02
```

The tag marks the commit the cycle launches from. Anything that lands on `main` afterwards waits for the next cycle, unless it changes nothing hashed or validated (see the full-refit runbook's section 5).

## 3. Bootstrap

Copy `scripts/vm/bootstrap.sh` to the VM, or fetch it from the tag, and run it with the tag:

```bash
curl -fsSLO https://raw.githubusercontent.com/dseinternational/vocabulary-growth/fits/2026-10-02/scripts/vm/bootstrap.sh
bash bootstrap.sh fits/2026-10-02 --output-root /data/vocabulary-growth --with-tests
```

It checks the architecture, cores, memory, swap, lingering, the output root's filesystem and free space, the tools and the fonts. It then clones the repository detached at the tag, installs the locked environment, prepares the data, clears matplotlib's font cache and runs the fast tests. Finally it writes `~/.config/vocabulary-growth/vm.env`. That file lives outside the clone deliberately: an untracked file in the clone would mark every fit dirty and unpublishable. Fix whatever it reports and rerun it until it ends with `Ready`.

## 4. Plan and launch

```bash
cd ~/vocabulary-growth-2026-10-02
scripts/vm/campaign.sh plan A B C D P      # every step, and whether it is done
scripts/vm/campaign.sh launch A            # or: launch A B C D P
scripts/vm/campaign.sh status
tail -f /data/vocabulary-growth/campaign/fits-2026-10-02/campaign.log
```

`launch` detaches the run with `setsid` into its own systemd scope with `OOMPolicy=continue`, so an overrun elsewhere cannot tear it down and a closed session does not stop it. Each step's output is in `campaign/<tag>/logs/<step>.log`; the driver steps also keep their usual `replication-logs/`.

The campaign refuses to start a step from a checkout that is not exactly at a `fits/*` tag, or that is dirty. Steps run once: success leaves `<step>.ok`, and `run` or `launch` after an interruption resumes from the first step without one. A failed step leaves `<step>.failed` and its log, and the stage carries on with independent steps. Rerunning retries only the failed and unfinished steps. The refit steps (A, B) call `run_replication.ps1`, which itself skips any model with a complete, compatible fit, so a rerun after a partial failure refits only what is missing.

Pooled launches wait while available memory is below `VG_MIN_FREE_GB` (24 by default) and something is already running. `VG_POOL` (default 5) sets the DS pool width. Watch per-process memory with `pwsh scripts/memwatch.ps1 <logfile>`, and read `sudo dmesg -T | grep -i oom` before rerunning a failed step: an OOM kill, collateral scope teardown and a convergence failure all look alike in a status line.

## 5. Escalations

A step that fails the convergence gate needs a decision, not a blind retry. The known cases:

- **VG09** has needed `refit_hightune.py vg09 --tune 12000 --draws 8000 --target-accept 0.97 --chains 6`. Run it from the clone. The rerun of stage B then skips VG09 as complete and compatible.
- **`vg15 dse-native-only`** missed the gate in September; it needs `refit_hightune.py vg15 --variant dse-native-only` with a setting justified by its diagnostics. A rerun of its step would refit it at `rep` over the escalated fit, so after a successful escalation mark the step done: `touch campaign/<tag>/C-arm-vg15-dse-native-only.ok`.
- **`vg15 fallback-dispersion`** on aarch64: `fit_sensitivity.py vg15 fallback-dispersion --config rep --nutpie-backend jax`, then mark the step done the same way.
- **VG11** carries a recorded R-hat exception; read its record before interpreting a gate failure.

Record every escalation and every manually marked step in the cycle's run record.

## 6. Prepare, then ask

Stage P regenerates the comparisons, including the three quarantined on 2026-09-17, and the robustness matrices. It then syncs the figure cache with `--allow-caveats` and renders the report book, which is validation-only and never published. Review the caveats it records, then put publication to the study owner. Publishing can run on the VM (`upload.py <model> --config rep --allow-caveats`, `publish_comparison.py --allow-caveats`) or on the workstation from a worktree at the same tag once the output root is home. The code signature records source hashes and package versions, not the platform, so fits made on the VM validate there.

## 7. Bring the output root home, then tear down

Copy the whole output root, not only `models/`: `recovery/` holds every replicate's truth and synthetic frame, `comparisons/` the scored matrices, `failed/` the evidence for exclusions, `replication-logs/` and `campaign/` the run record. The workstation convention is `F:\projects\vocabulary-growth\<commit>\output` for the archive, and `D:\output\vocabulary-growth` for the working root that the next cycle's comparisons and the comparison book read. Verify the copy (file counts and sizes per directory) before deallocating the VM; its local disks are wiped, and only the attached disk survives.
