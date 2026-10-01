> [!NOTE]
> Drafted by an LLM-based AI tool (Claude Code/Opus 5.5).

# Calibration split by repeated and single-visit children

Issue [#236](https://github.com/dseinternational/vocabulary-growth/issues/236) asks for a repeated-child calibration check before `q`, `kappa` or the child scales are interpreted. The branch-stratified check (conditional against fallback spoken rows) landed in [#256](https://github.com/dseinternational/vocabulary-growth/pull/256); this note adds the repeated-child half. `write_trace_calibration` now also splits every outcome by whether its child is seen more than once among that outcome's own observed rows, when the prepared frame carries `subject_id`. A row with no recorded child counts as a single visit. The split is written only when both groups are present. Like the branch split, it is read from the prepared frame, so it applies to an existing trace without a refit.

## What the split shows on the current fits

Run on 2026-10-01 against the fits of record from `d409c3d` (2026-09-16/17), at the 89% level, pooled over ages. These fits are stale against the current code signature but not against their prepared frames, and the split reads only the stored predictive draws and the frame, so the numbers below describe those fits exactly.

| Model | Outcome    |  Rows | Group        | Coverage | Calibrated coverage | PIT variance | Calibrated PIT variance | PIT extreme rate |
| ----- | ---------- | ----: | ------------ | -------: | ------------------: | -----------: | ----------------------: | ---------------: |
| VG20  | understood |   859 | repeated     |    0.969 |               0.897 |        0.043 |                   0.083 |            0.037 |
| VG20  | understood |   441 | single-visit |    1.000 |               0.898 |        0.027 |                   0.083 |            0.000 |
| VG20  | spoken     |   990 | repeated     |    0.958 |               0.917 |        0.053 |                   0.078 |            0.047 |
| VG20  | spoken     |   430 | single-visit |    1.000 |               0.917 |        0.019 |                   0.078 |            0.002 |
| VG12  | understood | 2,228 | repeated     |    0.979 |               0.893 |        0.036 |                   0.083 |            0.020 |
| VG12  | understood | 4,821 | single-visit |    0.995 |               0.898 |        0.029 |                   0.083 |            0.009 |

## Reading it

Single-visit rows are covered almost without exception and almost never fall in a predictive tail. That is a property of the in-sample check, not evidence that the model is conservative for those children. With one observation, the child's own effect is free to absorb that observation's residual, so the replication conditions on a parameter fitted to the very count it replicates. A repeated child's effect has to serve every visit, so its rows are the less optimistic in-sample check.

So the pooled in-sample calibration in every report is propped up by the single-visit children, and the pooled table should not be read as the model's calibration for a child seen repeatedly. On the repeated rows VG20 still over-covers (0.969 and 0.958 against calibrated 0.897 and 0.917), with PIT variance about half its calibrated value. In this in-sample direction that is expected even of a well-specified model. It cannot establish that the reported intervals are conservative for a new child, which is the question the held-out checks answer.

What the split adds for #236 is the caveat the pooled table lacked: in-sample calibration pooled over children is dominated by the fraction of single-visit children, which is 34% of VG20's comprehension rows and 68% of VG12's. A difference in calibration between the two populations' pooled tables partly reflects that fraction, not only how well each model fits.

## Reproducing

Every fit from this change on writes the split into `posterior_predictive_calibration.csv`, beside the branch split, and each model page's calibration section renders it. To compute it for an existing trace, call `write_trace_calibration` on the trace's `posterior_predictive`, `observed_data` and `constant_data` groups with the frame from `vocab_growth.analysis_frames.build_analysis_frame`.
