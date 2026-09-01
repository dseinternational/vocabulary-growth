# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Are VG20's fitted child effects heavy-tailed enough to justify a multivariate t?

The question, asked 2026-09-01: the child random-effect block is the one site in
the model family where a multivariate Student-t is even defensible, and the
robustness argument for one is that correlations are tail-sensitive and
``rho_uq`` is a headline number. Before considering structure, measure the tails
on the fit of record.

Method, per posterior draw of VG20's fit of record. Standardise the two
per-child effect vectors by that draw's own scales (``delta_subj_u / tau_subj_u``,
``delta_subj_q / tau_subj_q``), then across the 767 children compute (a) the
excess kurtosis of each margin, (b) the count of children beyond 3 SD, and
(c) the cross-child Pearson correlation of the standardised pair, full versus
with the top-1% Mahalanobis children (under that draw's ``rho_uq``) removed.
The normal reference band for (a) comes from simulating normal samples of the
same size. Draws are thinned 4x: these are cross-child summaries, not MCMC
estimands, so thinning costs nothing that matters.

Result, against the 2026-08-22 ``rep`` fit (commit ``d7ee170``): the tails are
mildly but genuinely heavy -- excess kurtosis +0.466 (u) / +0.336 (q) against a
normal 89% band of [-0.268, +0.282], roughly a t with 17-22 degrees of freedom
-- yet the trim moves the cross-child correlation by -0.003 with an 89% interval
spanning zero, so ``rho_uq`` is not driven by the extreme children and the
multivariate t was **not adopted**. See
``notes/202609011717-multivariate-t-not-adopted.md``.

The check is conservative in one direction only: the normal prior itself pulls
the per-draw fitted effects toward normality, so real heavy tails are
understated, never overstated. A positive here means more than the same number
would from raw data.

Reads the trace's posterior group lazily with xarray rather than going through
``comparison._load_reshaped_draws``: the per-child deterministics are
subject-dimensioned (36,000 draws x 767 children each), and the lazy open plus
thinning is what keeps the read at a few hundred megabytes instead of the full
10.5 GB file.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from vocab_growth import comparison as C

MODEL = "vg20"
THIN = 4
TRIM_SHARE = 0.01
RNG = np.random.default_rng(20260901)


def eti89(x: np.ndarray) -> np.ndarray:
    return np.percentile(x, [5.5, 94.5])


def excess_kurtosis(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.mean(axis=axis, keepdims=True)
    c = a - m
    m2 = (c**2).mean(axis=axis)
    m4 = (c**4).mean(axis=axis)
    return m4 / m2**2 - 3.0


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross-child Pearson correlation, one value per draw (row)."""
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    return (ac * bc).sum(axis=1) / np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))


def corr_rows_masked(a: np.ndarray, b: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """As ``corr_rows``, over the kept children only, per draw."""
    w = keep.astype(float)
    n = w.sum(axis=1)
    am = (a * w).sum(axis=1) / n
    bm = (b * w).sum(axis=1) / n
    ac = (a - am[:, None]) * w
    bc = (b - bm[:, None]) * w
    return (ac * bc).sum(axis=1) / np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))


def main() -> None:
    post = xr.open_dataset(C.trace_path(MODEL), group="posterior")
    sel = post[
        ["delta_subj_u", "delta_subj_q", "tau_subj_u", "tau_subj_q", "rho_uq"]
    ].isel(draw=slice(None, None, THIN))

    n_subj = sel.sizes["subject_id"]
    du = sel["delta_subj_u"].values.reshape(-1, n_subj)
    dq = sel["delta_subj_q"].values.reshape(-1, n_subj)
    tu = sel["tau_subj_u"].values.reshape(-1, 1)
    tq = sel["tau_subj_q"].values.reshape(-1, 1)
    rho = sel["rho_uq"].values.reshape(-1)
    n_draw = du.shape[0]
    print(f"draws used {n_draw} (thin {THIN}), children {n_subj}")

    eu = du / tu
    eq = dq / tq

    ku = excess_kurtosis(eu, axis=1)
    kq = excess_kurtosis(eq, axis=1)
    ref = excess_kurtosis(RNG.standard_normal((4000, n_subj)), axis=1)
    print(
        f"excess kurtosis u: mean {ku.mean():+.3f}, "
        f"89% ETI [{eti89(ku)[0]:+.3f}, {eti89(ku)[1]:+.3f}]"
    )
    print(
        f"excess kurtosis q: mean {kq.mean():+.3f}, "
        f"89% ETI [{eti89(kq)[0]:+.3f}, {eti89(kq)[1]:+.3f}]"
    )
    print(
        f"normal reference (n={n_subj}): mean {ref.mean():+.3f}, "
        f"89% band [{eti89(ref)[0]:+.3f}, {eti89(ref)[1]:+.3f}]"
    )

    # Normal expectation for |e| > 3 is n * 2 * (1 - Phi(3)).
    out_u = (np.abs(eu) > 3).sum(axis=1)
    out_q = (np.abs(eq) > 3).sum(axis=1)
    print(
        f"children |e|>3 SD: u mean {out_u.mean():.2f}, q mean {out_q.mean():.2f}; "
        f"normal expectation {n_subj * 2 * 0.001349898:.2f}"
    )

    r_full = corr_rows(eu, eq)
    det = 1.0 - rho[:, None] ** 2
    maha = (eu**2 - 2.0 * rho[:, None] * eu * eq + eq**2) / det
    k_trim = max(1, int(round(TRIM_SHARE * n_subj)))
    cut = np.partition(maha, n_subj - k_trim, axis=1)[:, n_subj - k_trim]
    r_trim = corr_rows_masked(eu, eq, maha < cut[:, None])
    d = r_trim - r_full
    print(
        f"sampled rho_uq:            mean {rho.mean():+.3f}, "
        f"89% ETI [{eti89(rho)[0]:+.3f}, {eti89(rho)[1]:+.3f}]"
    )
    print(
        f"cross-child corr, full:    mean {r_full.mean():+.3f}, "
        f"89% ETI [{eti89(r_full)[0]:+.3f}, {eti89(r_full)[1]:+.3f}]"
    )
    print(
        f"cross-child corr, trimmed: mean {r_trim.mean():+.3f} "
        f"(top {k_trim} Mahalanobis children removed per draw)"
    )
    print(
        f"trim effect (trim - full): mean {d.mean():+.4f}, "
        f"89% ETI [{eti89(d)[0]:+.4f}, {eti89(d)[1]:+.4f}]"
    )
    print(
        "caveat: the normal prior pulls per-draw effects toward normality, "
        "so this check understates real heavy tails"
    )


if __name__ == "__main__":
    main()
