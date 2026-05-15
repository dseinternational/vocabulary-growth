# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Verification harness for `scripts/compare_ds_td_latency.py`.

Runs four independent checks:

  (1) Synthetic check on shifted-linear curves: with U(a) = a and
      S(a) = a - shift, the algorithm must recover DA(N) = shift and
      extra(N) = shift for every N. Tests both first_crossing_age and
      evaluate_at_ages.

  (2) Per-draw round-trip on real traces: for a handful of randomly chosen
      posterior draws and N values, recompute a_U(N), a_S(N) and verify
      U(a_U(N)) ~= N and S(a_S(N)) ~= N (the crossing definition).

  (3) Cross-check that the per-draw extra(N) reported by the analysis
      pipeline matches a clean reimplementation using scipy/numpy.

  (4) Sanity-compare the per-draw medians against trajectory-from-CSV
      shortcut values. They are *not* expected to agree (median of a
      nonlinear function != nonlinear function of medians), but the
      magnitude of the gap should be small and in the expected direction
      (right-skew in extra(N) -> per-draw median >= median-trajectory
      shortcut).
"""

from __future__ import annotations

import os

import arviz as az
import numpy as np
import pandas as pd

from compare_ds_td_latency import (
    DS_DIR, TD_DIR, N_TRIALS_DS, N_TRIALS_TD, N_GRID,
    first_crossing_age, evaluate_at_ages, load_population_trajectory,
)


def check_synthetic() -> None:
    print("[1] Synthetic shifted-linear check …")
    ages = np.linspace(0.0, 100.0, 1001)
    shift = 5.0
    # 3 fake "draws"
    U = np.stack([ages + 0.0, ages + 0.5, ages - 0.3])
    S = U - shift
    for N in [1.0, 10.0, 50.0, 90.0]:
        aU = first_crossing_age(U, ages, N)
        aS = first_crossing_age(S, ages, N)
        DA = aS - aU
        U_at_aS = evaluate_at_ages(U, ages, aS)
        extra = U_at_aS - N
        # DA should == shift for every draw (within grid resolution)
        assert np.allclose(DA, shift, atol=1e-2), f"DA mismatch at N={N}: {DA}"
        # extra should == shift, similarly
        assert np.allclose(extra, shift, atol=1e-2), f"extra mismatch at N={N}: {extra}"
    print("    PASS — recovers shift exactly for every N and every draw.\n")


def check_per_draw_roundtrip(ages: np.ndarray, U: np.ndarray, S: np.ndarray,
                              label: str, n_draws_check: int = 20,
                              N_values: list[float] | None = None) -> None:
    print(f"[2] Per-draw round-trip on {label} …")
    if N_values is None:
        N_values = [10.0, 50.0, 200.0]
    rng = np.random.default_rng(123)
    draws = rng.choice(U.shape[0], size=n_draws_check, replace=False)
    max_dev_U = 0.0
    max_dev_S = 0.0
    for N in N_values:
        aU = first_crossing_age(U[draws], ages, N)
        aS = first_crossing_age(S[draws], ages, N)
        U_at_aU = evaluate_at_ages(U[draws], ages, aU)
        S_at_aS = evaluate_at_ages(S[draws], ages, aS)
        # Where the crossing exists, U(a_U(N)) and S(a_S(N)) should both == N
        valid_U = ~np.isnan(U_at_aU)
        valid_S = ~np.isnan(S_at_aS)
        if valid_U.any():
            dev_U = float(np.max(np.abs(U_at_aU[valid_U] - N)))
            max_dev_U = max(max_dev_U, dev_U)
        if valid_S.any():
            dev_S = float(np.max(np.abs(S_at_aS[valid_S] - N)))
            max_dev_S = max(max_dev_S, dev_S)
        print(f"    N={N:6.1f}  max |U(a_U)-N| = {dev_U:.2e}   "
              f"max |S(a_S)-N| = {dev_S:.2e}")
    if max_dev_U < 1e-6 and max_dev_S < 1e-6:
        print(f"    PASS — crossings recover N to machine precision.\n")
    else:
        print(f"    WARN — devs U={max_dev_U:.3e} S={max_dev_S:.3e}\n")


def check_clean_reimplementation(ages: np.ndarray, U: np.ndarray, S: np.ndarray,
                                  label: str, N_test: float = 50.0,
                                  n_check: int = 200) -> None:
    """Recompute DA and extra by a brute scan-the-array implementation
    on a subset of draws, and verify it matches the vectorised version."""
    print(f"[3] Clean reimplementation cross-check on {label}, N={N_test} …")
    rng = np.random.default_rng(7)
    draws = rng.choice(U.shape[0], size=n_check, replace=False)

    def slow_first_crossing(y: np.ndarray, a: np.ndarray, N: float) -> float:
        """Per-draw linear-interpolated first crossing; returns NaN if never."""
        idx = np.where(y >= N)[0]
        if idx.size == 0:
            return np.nan
        j = int(idx[0])
        if j == 0:
            return float(a[0])
        y0, y1 = float(y[j - 1]), float(y[j])
        if y1 == y0:
            return float(a[j])
        return float(a[j - 1] + (N - y0) * (a[j] - a[j - 1]) / (y1 - y0))

    def slow_eval(y: np.ndarray, a: np.ndarray, target: float) -> float:
        if np.isnan(target) or target < a[0] or target > a[-1]:
            return np.nan
        return float(np.interp(target, a, y))

    slow_DA = np.zeros(n_check)
    slow_extra = np.zeros(n_check)
    for k, d in enumerate(draws):
        aU = slow_first_crossing(U[d], ages, N_test)
        aS = slow_first_crossing(S[d], ages, N_test)
        slow_DA[k] = aS - aU
        slow_extra[k] = slow_eval(U[d], ages, aS) - N_test

    fast_aU = first_crossing_age(U[draws], ages, N_test)
    fast_aS = first_crossing_age(S[draws], ages, N_test)
    fast_DA = fast_aS - fast_aU
    fast_extra = evaluate_at_ages(U[draws], ages, fast_aS) - N_test

    valid = ~(np.isnan(slow_DA) | np.isnan(fast_DA))
    if valid.any():
        max_DA_diff = float(np.max(np.abs(slow_DA[valid] - fast_DA[valid])))
        max_extra_diff = float(np.max(np.abs(slow_extra[valid] - fast_extra[valid])))
        print(f"    max |slow_DA - fast_DA|       = {max_DA_diff:.3e}")
        print(f"    max |slow_extra - fast_extra| = {max_extra_diff:.3e}")
        if max_DA_diff < 1e-9 and max_extra_diff < 1e-9:
            print("    PASS — vectorised path matches per-draw reimplementation.\n")
        else:
            print("    WARN — implementations differ.\n")
    else:
        print("    No valid draws.\n")


def check_median_of_function_vs_function_of_median(
    ages: np.ndarray, U: np.ndarray, S: np.ndarray, label: str,
    N_test: float = 50.0,
) -> None:
    """Sanity-compare per-draw median extra(N) to the value implied by
    median-trajectories. They are NOT expected to agree exactly."""
    print(f"[4] Per-draw median vs median-trajectory shortcut, {label}, N={N_test} …")
    # Proper per-draw
    aU = first_crossing_age(U, ages, N_test)
    aS = first_crossing_age(S, ages, N_test)
    DA = aS - aU
    extra = evaluate_at_ages(U, ages, aS) - N_test
    DA_med = float(np.nanmedian(DA))
    extra_med = float(np.nanmedian(extra))

    # Shortcut
    U_med = np.median(U, axis=0)
    S_med = np.median(S, axis=0)
    aU_short = float(np.interp(N_test, U_med, ages))
    aS_short = float(np.interp(N_test, S_med, ages))
    DA_short = aS_short - aU_short
    extra_short = float(np.interp(aS_short, ages, U_med)) - N_test

    print(f"    per-draw median:        DA = {DA_med:6.3f} mo, extra = {extra_med:6.2f} words")
    print(f"    median-trajectory:      DA = {DA_short:6.3f} mo, extra = {extra_short:6.2f} words")
    print(f"    diff (per-draw - short): {DA_med - DA_short:+.3f} mo, "
          f"{extra_med - extra_short:+.2f} words   (right-skew typical)\n")


def main() -> None:
    print("=" * 70)
    print("Verification harness for compare_ds_td_latency.py")
    print("=" * 70 + "\n")

    check_synthetic()

    print("Loading traces …\n")
    ages_ds, U_ds, S_ds = load_population_trajectory(os.path.join(DS_DIR, "trace.nc"), N_TRIALS_DS)
    ages_td, U_td, S_td = load_population_trajectory(os.path.join(TD_DIR, "trace.nc"), N_TRIALS_TD)

    check_per_draw_roundtrip(ages_ds, U_ds, S_ds, "DS", n_draws_check=30,
                              N_values=[10.0, 50.0, 200.0, 400.0])
    check_per_draw_roundtrip(ages_td, U_td, S_td, "TD", n_draws_check=30,
                              N_values=[10.0, 50.0, 200.0, 400.0])

    check_clean_reimplementation(ages_ds, U_ds, S_ds, "DS", N_test=50.0, n_check=200)
    check_clean_reimplementation(ages_td, U_td, S_td, "TD", N_test=50.0, n_check=200)
    check_clean_reimplementation(ages_ds, U_ds, S_ds, "DS", N_test=200.0, n_check=200)

    check_median_of_function_vs_function_of_median(ages_ds, U_ds, S_ds, "DS", N_test=50.0)
    check_median_of_function_vs_function_of_median(ages_td, U_td, S_td, "TD", N_test=50.0)
    check_median_of_function_vs_function_of_median(ages_ds, U_ds, S_ds, "DS", N_test=200.0)
    check_median_of_function_vs_function_of_median(ages_td, U_td, S_td, "TD", N_test=200.0)


if __name__ == "__main__":
    main()
