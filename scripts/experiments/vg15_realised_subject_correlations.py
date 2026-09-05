# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Realised correlations between VG15's fitted subject intercepts.

Gate 1 of issue #296 (the VG24 proposal): before a correlated subject block is
built on VG15, measure how far the *fitted* child effects already correlate
under the independent priors, the way ``vg16_crosslag_quantification.py`` did
for VG16. VG15 carries three subject intercepts -- understood, the production
ratio ``q`` and the signed ratio -- with independent priors and no correlation
parameter, so any correlation in the realised effects is association the model
has nowhere else to put.

Two figures per pair, because VG15's subject shifts enter only the *marginal*
likelihoods (the cross-tab cells are fed population + study marginals, so the
subject block cannot pull ``psi``): the correlation across every child in the
frame, where children without a marginal on one of the two scales sit at their
prior, and the correlation across the children who carry a marginal on
**both** scales, which is the set that would identify the corresponding
correlation parameter. The subject sets are read from the trace's own
``constant_data`` masks, never from a rebuilt frame.

Reads three ``(chain, draw, subject)`` variables through ``h5netcdf`` rather
than opening the whole trace (the ``rep`` file is about 4.6 GB).

    uv run python scripts/experiments/vg15_realised_subject_correlations.py \
        [--fit-dir <dir>] [--output-dir <root>]

Recorded in ``notes/202609041722-sign-speech-modelling-proposals.md`` §8.
"""

from __future__ import annotations

import argparse
import json
import os

import h5netcdf
import numpy as np
from scipy.stats import halfnorm

from vocab_growth import environment as env

FIT_DIRNAME = "VG15-age-joint-signspeech-ds"

PAIRS: dict[str, tuple[str, str, str, str]] = {
    # label: (variable a, variable b, mask for a's marginal, mask for b's marginal)
    "sign ~ q": ("delta_subj_sign", "delta_subj_q", "obs_sign_mask", "obs_s_mask"),
    "u ~ q": ("delta_subj_u", "delta_subj_q", "obs_u_mask", "obs_s_mask"),
    "u ~ sign": ("delta_subj_u", "delta_subj_sign", "obs_u_mask", "obs_sign_mask"),
}
SCALES = ("tau_subj_u", "tau_subj_q", "tau_subj_sign")
ETI = (5.5, 94.5)


def _per_draw_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation across the subject axis, one value per draw."""
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))


def _summary(x: np.ndarray) -> str:
    lo, hi = np.percentile(x, ETI)
    return f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--fit-dir", help=f"fit directory (default: <output-root>/models/{FIT_DIRNAME})"
    )
    parser.add_argument(
        "--output-dir", help="output root, overriding DSE_VOCAB_GROWTH_OUTPUT_DIR"
    )
    args = parser.parse_args()
    if args.output_dir:
        env.set_output_root(args.output_dir)
    fit_dir = args.fit_dir or os.path.join(env.output_root(), "models", FIT_DIRNAME)

    with open(os.path.join(fit_dir, "fit_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    print(f"fit: {fit_dir}")
    print(
        f"created: {manifest.get('created_at_utc')}  frame hash: {manifest['data']['analysis_frame_hash']}"
    )

    with h5netcdf.File(os.path.join(fit_dir, "trace.nc"), "r") as f:
        const = f.groups["constant_data"]
        subject_obs = const.variables["subject_obs"][...].astype(int)
        masks = {
            name: const.variables[name][...].astype(bool)
            for name in (
                "obs_u_mask",
                "obs_s_mask",
                "obs_sign_mask",
                "obs_cells_mask",
                "obs_prod_mask",
            )
        }
        post = f.groups["posterior"]
        n_chains = post.dimensions["chain"].size
        n_subjects = post.dimensions["subject_id"].size

        def subjects_with(mask_name: str) -> np.ndarray:
            return np.unique(subject_obs[masks[mask_name]])

        any_sign = np.unique(
            np.concatenate(
                [
                    subjects_with(m)
                    for m in ("obs_sign_mask", "obs_cells_mask", "obs_prod_mask")
                ]
            )
        )
        print(
            f"children: {n_subjects} in the frame; {len(any_sign)} with any signing information; marginal rows on u/spoken/signed: {len(subjects_with('obs_u_mask'))}/{len(subjects_with('obs_s_mask'))}/{len(subjects_with('obs_sign_mask'))}"
        )

        subsets = {
            label: np.intersect1d(subjects_with(ma), subjects_with(mb))
            for label, (_, _, ma, mb) in PAIRS.items()
        }
        corr_all: dict[str, list[np.ndarray]] = {label: [] for label in PAIRS}
        corr_sub: dict[str, list[np.ndarray]] = {label: [] for label in PAIRS}
        scales: dict[str, list[np.ndarray]] = {name: [] for name in SCALES}
        for chain in range(n_chains):
            draws = {
                v: post.variables[v][chain, :, :].astype(np.float64)
                for v in ("delta_subj_u", "delta_subj_q", "delta_subj_sign")
            }
            for label, (va, vb, _, _) in PAIRS.items():
                corr_all[label].append(_per_draw_corr(draws[va], draws[vb]))
                sub = subsets[label]
                corr_sub[label].append(
                    _per_draw_corr(draws[va][:, sub], draws[vb][:, sub])
                )
            for name in SCALES:
                scales[name].append(post.variables[name][chain, :])

    print(
        "\nRealised correlation of the fitted subject intercepts (posterior mean, 89% ETI):"
    )
    print(f"{'pair':10s} {'all children':>32s} {'children with both marginals':>40s}")
    for label in PAIRS:
        print(
            f"{label:10s} {n_subjects:5d}  {_summary(np.concatenate(corr_all[label])):>24s} {len(subsets[label]):5d}  {_summary(np.concatenate(corr_sub[label])):>32s}"
        )

    print(
        "\nSubject scales (posterior mean, 89% ETI; prior CDF of the mean under HalfNormal(1.5)):"
    )
    for name in SCALES:
        x = np.concatenate(scales[name])
        lo, hi = np.percentile(x, ETI)
        print(
            f"{name:14s} {x.mean():.3f} [{lo:.3f}, {hi:.3f}]  prior CDF {halfnorm.cdf(x.mean(), scale=1.5):.2f}"
        )


if __name__ == "__main__":
    main()
