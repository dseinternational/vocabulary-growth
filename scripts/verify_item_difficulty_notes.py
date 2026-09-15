# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recompute the checkable figures in the item-difficulty working note.

Verifies the numbers in
``notes/202607261540-item-difficulty-and-the-aggregate-likelihood.md``
(§§2, 3.3, 4, 5, 8, 9, 10, 11 — the consolidation of two earlier notes whose
figures this script originally pinned; their git history holds the trail) from
the raw CSVs and, where present, the fitted output of the Down syndrome model of
record (``FITTED_MODEL``). Each check prints CLAIM vs COMPUTED; the script exits
non-zero if any executed check fails. Sections that need fitted output are
skipped (not failed) when the output root has no directory for that model.

Run from anywhere::

    python scripts/verify_item_difficulty_notes.py

The output root is resolved from ``DSE_VOCAB_GROWTH_OUTPUT_DIR`` or the
repository-local ``output/``, mirroring ``vocab_growth.environment.output_root``
(re-implemented here so the script needs only numpy/pandas/scipy).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

REPO = Path(__file__).resolve().parents[1]
# The ie_01 checklists' printed word lists run to 120 / 340 / 350, and each also
# admits the child's own name, family and pet names, so the maximum achievable
# counts are 127 / 349 / 353 (settled 2026-08-25). The note's proportions are
# taken against the achievable sizes, on the study owner's decision of
# 2026-09-14 (#320): the recorded counts include the proper-noun slots, so only
# the achievable denominator matches its numerator.
PRINTED_STRATUM_SIZES = np.array([120, 340, 350])
STRATUM_SIZES = np.array([127, 349, 353])
N_ITEMS = int(STRATUM_SIZES.sum())
# Two different numbers that used to be written the same way. N_ITEMS is the
# checklists' inventory (829); MODEL_N_TRIALS is every model's `n_trials`, the
# 810-item reference scale, which the denominator decision leaves alone.
MODEL_N_TRIALS = 810
SIM_SEED = 20260726
# The fit §3.3 and §4 read. VG10 until 2026-09-14, when the study owner re-based
# both sections on VG20, the Down syndrome model of record (note §15 item 12).
# VG20's point values are reported, not pinned, until the next refit produces a fit
# current on its definition and frame: the 2026-09-07 fit predates the sex
# covariate and the uk_01 comprehension correction, so its digits will move.
# What is checked meanwhile is the argument each section makes -- a refit could
# overturn that, and a change of digits cannot.
FITTED_MODEL = "VG20"

_failures: list[str] = []


def check(label: str, computed, claim, tol: float = 0.0) -> None:
    """Print a CLAIM-vs-COMPUTED line and record a failure outside tolerance."""
    computed_arr = np.asarray(computed, dtype=float)
    claim_arr = np.asarray(claim, dtype=float)
    ok = bool(np.all(np.abs(computed_arr - claim_arr) <= tol + 1e-12))
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {label}: computed {np.round(computed_arr, 4).tolist()} vs claim {claim_arr.tolist()}")
    if not ok:
        _failures.append(label)


def report(label: str, computed) -> None:
    """Print a value the note quotes provisionally, without pinning it."""
    print(f"  [info] {label}: {np.round(np.asarray(computed, dtype=float), 4).tolist()} (reported, not pinned)")


def check_exchangeability(ie: pd.DataFrame) -> None:
    print("§2 — hypergeometric exchangeability test (ie_01 follow-up wave)")
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    complete = ~np.isnan(end_u).any(axis=1)
    totals = end_u.sum(axis=1)
    tested = complete & (totals > 0) & (totals < N_ITEMS)
    check("complete follow-up records", complete.sum(), 46)
    check("records with 0 < T < 829", tested.sum(), 44)

    counts = end_u[tested]
    T = totals[tested]
    mean_k = np.outer(T, STRATUM_SIZES / N_ITEMS)
    var_k = np.outer(T * (N_ITEMS - T) / (N_ITEMS - 1), (STRATUM_SIZES / N_ITEMS) * (1 - STRATUM_SIZES / N_ITEMS))
    z = (counts - mean_k) / np.sqrt(var_k)
    # Re-pinned 2026-09-14 against the achievable checklist sizes (#320). Under
    # the printed sizes these read RMS z 9.61, mean z 8.40 / 3.00 / -9.01, a
    # Checklist 3 deficit in 42 of 44 (p = 1.1e-10), pooled proportions 0.671 /
    # 0.452 / 0.253 of an overall 0.398, per-child means 0.701 / 0.473 / 0.264
    # and an outer-checklist spread of 1.8 logits. Every conclusion stands.
    check("RMS z, all strata", np.sqrt((z**2).mean()), 9.24, tol=0.005)
    check("mean z per checklist", z.mean(axis=0), [7.72, 2.86, -8.48], tol=0.005)

    n = len(T)
    pos1 = int((counts[:, 0] > mean_k[:, 0]).sum())
    neg3 = int((counts[:, 2] < mean_k[:, 2]).sum())
    check("Checklist 1 positive excess", pos1, 37)
    check("Checklist 3 negative deficit", neg3, 41)
    check("sign test p, Checklist 1 (1e-6)", stats.binomtest(pos1, n).pvalue / 1e-6, 5.3, tol=0.05)
    check("sign test p, Checklist 3 (1e-9)", stats.binomtest(neg3, n).pvalue / 1e-9, 1.6, tol=0.05)

    props = counts / STRATUM_SIZES
    monotone = (props[:, 0] >= props[:, 1]) & (props[:, 1] >= props[:, 2])
    check("monotone share", monotone.mean(), 0.795, tol=0.0005)

    rng = np.random.default_rng(SIM_SEED)
    rates = np.empty(2000)
    for r in range(2000):
        sim = np.array([rng.multivariate_hypergeometric(STRATUM_SIZES, int(t)) for t in T])
        sim_props = sim / STRATUM_SIZES
        rates[r] = ((sim_props[:, 0] >= sim_props[:, 1]) & (sim_props[:, 1] >= sim_props[:, 2])).mean()
    check("simulated null monotone rate, mean", rates.mean(), 0.192, tol=0.01)
    check("simulations reaching the observed rate", (rates >= monotone.mean()).sum(), 0)

    pooled_46 = end_u[complete]
    check("pooled proportions over the 46", pooled_46.sum(axis=0) / (complete.sum() * STRATUM_SIZES), [0.634, 0.441, 0.250], tol=0.0005)
    check("overall pooled proportion", pooled_46.sum() / (complete.sum() * N_ITEMS), 0.389, tol=0.0005)
    check("per-child mean proportions over the 44", props.mean(axis=0), [0.6625, 0.4607, 0.2618], tol=0.0005)
    pooled_props = pooled_46.sum(axis=0) / (complete.sum() * STRATUM_SIZES)
    spread = np.log(pooled_props[0] / (1 - pooled_props[0])) - np.log(pooled_props[2] / (1 - pooled_props[2]))
    check("outer-checklist spread (logits)", spread, 1.6, tol=0.05)


def check_production_gradient(ie: pd.DataFrame) -> None:
    print("§5 — production propensity gradient")
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    says = ie[["says_1_end", "says_2_end", "says_3_end"]].to_numpy(float)
    both = ~np.isnan(end_u).any(axis=1) & ~np.isnan(says).any(axis=1)
    coherent = both & (says <= end_u).all(axis=1) & (end_u.sum(axis=1) > 0)
    check("coherent records", coherent.sum(), 38)

    S, U = says[coherent], end_u[coherent]
    # The stratum table in docs/report/_caveats-ds.qmd: this coherence screen was
    # chosen for it by the study owner on 2026-09-14 (#320), and its shares are of
    # the achievable checklist sizes. The eight records it drops each report more
    # words said than understood on at least one checklist.
    check("caveats table: records dropped for said > understood", int((both & (says > end_u).any(axis=1)).sum()), 8)
    ages = ie["age_months_end"].to_numpy(float)[coherent]
    check("caveats table: age range (months)", [ages.min(), ages.max()], [27, 86])
    check("caveats table: understood share", U.sum(axis=0) / (len(U) * STRATUM_SIZES), [0.7536, 0.5002, 0.2814], tol=0.0001)
    check("caveats table: said share", S.sum(axis=0) / (len(S) * STRATUM_SIZES), [0.5025, 0.2633, 0.1466], tol=0.0001)
    with np.errstate(invalid="ignore", divide="ignore"):
        qk = np.where(U > 0, S / np.where(U > 0, U, 1), np.nan)
    check("q_k ratio of sums", S.sum(axis=0) / U.sum(axis=0), [0.667, 0.526, 0.521], tol=0.0005)
    check("q_k median per child", np.nanmedian(qk, axis=0), [0.683, 0.364, 0.255], tol=0.0005)
    check("q_k mean per child", np.nanmean(qk, axis=0), [0.612, 0.414, 0.318], tol=0.0005)

    paired = (U[:, 0] > 0) & (U[:, 2] > 0)
    gap = np.log((S[paired, 0] + 0.5) / (U[paired, 0] - S[paired, 0] + 0.5)) - np.log(
        (S[paired, 2] + 0.5) / (U[paired, 2] - S[paired, 2] + 0.5)
    )
    check("paired log-odds gap: n", paired.sum(), 30)
    check("paired log-odds gap: median", np.median(gap), 2.53, tol=0.005)
    check("paired log-odds gap: positive", (gap > 0).sum(), 27)
    check("paired log-odds gap: exact Wilcoxon p (1e-7)", stats.wilcoxon(gap, method="exact").pvalue / 1e-7, 4.7, tol=0.05)

    order = np.argsort(U.sum(axis=1))
    low, high = order[: len(order) // 2], order[len(order) // 2 :]
    w_lo = U[low].sum(axis=0) / U[low].sum()
    w_hi = U[high].sum(axis=0) / U[high].sum()
    q_lo, q_hi = S[low].sum() / U[low].sum(), S[high].sum() / U[high].sum()
    qk_lo, qk_hi = S[low].sum(axis=0) / U[low].sum(axis=0), S[high].sum(axis=0) / U[high].sum(axis=0)
    check("Checklist 1 weight, low -> high", [w_lo[0], w_hi[0]], [0.422, 0.210], tol=0.0005)
    check("Checklist 3 weight, low -> high", [w_lo[2], w_hi[2]], [0.124, 0.312], tol=0.0005)
    total = q_hi - q_lo
    within = (((w_lo + w_hi) / 2) * (qk_hi - qk_lo)).sum()
    composition = (((qk_lo + qk_hi) / 2) * (w_hi - w_lo)).sum()
    check("Kitagawa total / within / composition", [total, within, composition], [0.3399, 0.3950, -0.0550], tol=0.0005)


def check_data_defects(ie: pd.DataFrame, uk: pd.DataFrame) -> None:
    print("§8 — data defects")
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    complete = ~np.isnan(end_u).any(axis=1)
    start_1 = ie["understands_1_start"].to_numpy(float)
    check("pooled Checklist 1, baseline (46)", np.nansum(start_1[complete]) / (complete.sum() * STRATUM_SIZES[0]), 0.808, tol=0.0005)
    check("pooled Checklist 1, follow-up (46)", np.nansum(end_u[complete, 0]) / (complete.sum() * STRATUM_SIZES[0]), 0.634, tol=0.0005)
    # The printed-words-only proportion the aggregate counts cannot give directly:
    # bounded below by assuming every child filled every proper-noun slot, above
    # by assuming none did. The achievable-denominator figure lies between.
    slots = STRATUM_SIZES - PRINTED_STRATUM_SIZES
    pooled_counts = end_u[complete].sum(axis=0)
    n_complete = complete.sum()
    check(
        "printed-words-only pooled proportions, lower bound",
        (pooled_counts - n_complete * slots) / (n_complete * PRINTED_STRATUM_SIZES),
        [0.612, 0.426, 0.244],
        tol=0.0005,
    )
    check("printed-words-only pooled proportions, upper bound", pooled_counts / (n_complete * PRINTED_STRATUM_SIZES), [0.671, 0.452, 0.253], tol=0.0005)
    check(
        "mean understood total, baseline -> follow-up (46)",
        [ie["understands_total_start"].to_numpy(float)[complete].mean(), ie["understands_total_end"].to_numpy(float)[complete].mean()],
        [252, 323],
        tol=0.5,
    )
    delta_1 = end_u[:, 0] - start_1
    check("children whose Checklist 1 falls", (delta_1 < 0).sum(), 22)
    check("largest fall", np.nanmin(delta_1), -124)
    over = int((start_1 > PRINTED_STRATUM_SIZES[0]).sum() + (end_u[:, 0] > PRINTED_STRATUM_SIZES[0]).sum())
    check("records with Checklist 1 above the printed 120", over, 27)
    check("maximum Checklist 1 count (achievable 127)", np.nanmax(np.concatenate([start_1, end_u[:, 0]])), 124)

    # §8 item 3, settled 2026-09-14 (#320): uk_01's `c` columns count words
    # understood *only*, and the prepared `understood` now adds every word said
    # or signed. Before the correction these checks read "29 complete category
    # rows" and "spoken/understood > 1 for 2, maximum 1.95" -- the evidence that
    # led to it.
    category_cols = [c for c in uk.columns if c.endswith("c") and not c.startswith("t")]
    check("uk_01 understood-only category columns", len(category_cols), 19)
    wg = uk[uk["survey"] == "WG"]
    check("uk_01 Words and Gestures rows / with comprehension", [len(wg), int(wg["understood"].notna().sum())], [70, 70])
    check("understood_only reconciles with the c categories", np.abs(wg[category_cols].sum(axis=1) - wg["understood_only"]).max(), 0)
    check("understood == understood_only + produced", np.abs(wg["understood"] - wg["understood_only"] - wg["produced"]).max(), 0)
    check("rows with spoken above understood", int((wg["spoken"] > wg["understood"]).sum()), 0)
    ws = uk[uk["survey"] == "WS"]
    check("Words and Sentences rows marking any word understood-only", int((ws["understood_only"] > 0).sum()), 0)


def check_fitted_dispersion(output_root: Path) -> None:
    print(f"§4 — {FITTED_MODEL} fitted dispersion (skipped if no fitted output)")
    fit_dirs = sorted((output_root / "models").glob(f"{FITTED_MODEL}-*")) if (output_root / "models").is_dir() else []
    if not fit_dirs:
        print(f"  [skip] no {FITTED_MODEL} output found under", output_root / "models")
        return
    fit = fit_dirs[0]
    ages = np.array([12, 18, 24, 30, 48, 66])

    def at(table: pd.DataFrame, column: str) -> np.ndarray:
        return np.array([table.loc[(table["age_months"] - a).abs().idxmin(), column] for a in ages])

    p_fit = at(pd.read_csv(fit / "posterior_summary_u.csv"), "p_median")
    kappa = at(pd.read_csv(fit / "posterior_kappa_u.csv"), "kappa_median")
    tau_subj = float(pd.read_csv(fit / "diagnostics.csv", index_col=0).loc["tau_subj_u", "mean"])
    residual_sd = 1 / np.sqrt(p_fit * (1 - p_fit) * (kappa + 1))
    total_sd = np.sqrt(tau_subj**2 + residual_sd**2)
    report("ages", ages)
    report("fitted p (posterior_summary_u medians)", p_fit)
    report("fitted kappa (posterior_kappa_u medians)", kappa)
    report("tau_subj_u posterior mean", tau_subj)
    report("implied residual latent SD", residual_sd)
    report("total latent SD with tau_subj_u", total_sd)

    # Until 2026-09-14 these were pinned to VG10's August fit (ratio 0.805 over
    # 12-66 months) and were the note's *argument* rather than measurements it
    # reports -- which is why the checks below test the argument, not the digits.
    #
    # The ratio is the fitted decline in log(kappa + 1) divided by the decline a
    # constant latent spread would produce. Read it that way round: below 1,
    # kappa falls *less* than the level effect alone requires, so the residual
    # latent spread narrows with age; above 1 it would widen. The 2026-08-16
    # comment here read 0.805 as "constant spread accounts for four-fifths of the
    # observed decline", which inverts it. §4's claim that kappa's decline is not
    # children fanning out needs the ratio below 1 -- not near 1 -- and it depends
    # strongly on where the range starts (12 months sits below the 18-month
    # kappa anchor), so it is tested from each start age.
    end = int(np.where(ages == 66)[0][0])
    ratios = []
    for start_age in (12, 18, 24):
        start = int(np.where(ages == start_age)[0][0])
        decline = np.log((kappa[start] + 1) / (kappa[end] + 1))
        predicted = np.log((p_fit[end] * (1 - p_fit[end])) / (p_fit[start] * (1 - p_fit[start])))
        report(f"{start_age}-66 months: log decline in kappa+1 / constant-spread prediction / ratio", [decline, predicted, decline / predicted])
        ratios.append(decline / predicted)
    check("kappa falls less than a constant latent spread requires, from every start age (all ratios < 1)", float(all(r < 1 for r in ratios)), 1.0)
    early = residual_sd[ages <= 18].min()
    check("residual latent SD at 48 and 66 months is below its value at 12 and 18 (it does not grow)", float(residual_sd[ages >= 48].max() < early), 1.0)
    late = total_sd[ages >= 30]
    check("total latent SD is nearly flat from 30 months (max / min < 1.05)", float(late.max() / late.min() < 1.05), 1.0)

    check_kernel_share(fit)


def check_kernel_share(fit: Path) -> None:
    """Note §3.3: how much total variance the item-exchangeability kernel carries.

    Rasch sufficiency means heterogeneous item difficulty can only reach the model
    through the distribution of the total, so this share bounds the whole concern.
    It is `1 / VIF` where `VIF = (N + kappa) / (kappa + 1)` is the Beta-Binomial's
    inflation over its Binomial kernel. An earlier draft of the note quoted VG07's
    figures as though they were the model of record's, understating the exposure
    threefold -- hence reading the model of record's own output here. The shares
    are reported; what is checked is §3.3's claim that the worst case stays under
    3% of the total standard deviation.
    """
    print(f"§3.3 — kernel share of total variance ({FITTED_MODEL})")
    for outcome in ("u", "s"):
        table_path = fit / f"posterior_kappa_{outcome}.csv"
        if not table_path.exists():
            print(f"  [skip] {table_path.name} not present")
            continue
        table = pd.read_csv(table_path)
        kappa = table["kappa_median"].to_numpy(float)
        share = 100.0 * (kappa + 1.0) / (MODEL_N_TRIALS + kappa)
        report(f"kappa range ({outcome}), over ages {table['age_months'].min():.0f}-{table['age_months'].max():.0f}", [kappa.min(), kappa.max()])
        report(f"kernel share % ({outcome}, min/max)", [share.min(), share.max()])
        # A 40% error in a component carrying `share` of the variance (the
        # underdispersion at a 2-logit difficulty spread) moves the total SD by:
        worst = 100.0 * (1.0 - np.sqrt(1.0 - 0.40 * share.max() / 100.0))
        report(f"worst-case total SD shift % ({outcome})", worst)
        check(f"worst-case total SD shift under 3% ({outcome})", float(worst < 3.0), 1.0)


def _pooled_profile(ie: pd.DataFrame) -> np.ndarray:
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    complete = ~np.isnan(end_u).any(axis=1)
    return end_u[complete].sum(axis=0) / (complete.sum() * STRATUM_SIZES)


def check_link_tables(ie: pd.DataFrame) -> None:
    print("§4 and §9 — implied-kappa and mixed-link tables (sigma = 1, exact)")
    weights = STRATUM_SIZES / N_ITEMS
    profile = _pooled_profile(ie)
    d_k = -np.log(profile / (1 - profile))
    nodes, gh_weights = np.polynomial.hermite_e.hermegauss(199)
    gh_weights = gh_weights / gh_weights.sum()

    def plain_mean(mu: float) -> float:
        return float((gh_weights / (1 + np.exp(-(mu + nodes)))).sum())

    def mixed_mean(mu: float) -> float:
        return float(sum(w * (gh_weights / (1 + np.exp(-(mu + nodes - d)))).sum() for w, d in zip(weights, d_k, strict=True)))

    def implied_kappa(mean_fn, mixed: bool, target: float) -> float:
        mu = optimize.brentq(lambda m: mean_fn(m) - target, -40, 40)
        if mixed:
            values = sum(w / (1 + np.exp(-(mu + nodes - d))) for w, d in zip(weights, d_k, strict=True))
        else:
            values = 1 / (1 + np.exp(-(mu + nodes)))
        mean = float((gh_weights * values).sum())
        var = float((gh_weights * (values - mean) ** 2).sum())
        return mean * (1 - mean) / var - 1

    plain = [implied_kappa(plain_mean, False, p) for p in (0.05, 0.20, 0.50)]
    mixed = [implied_kappa(mixed_mean, True, p) for p in (0.05, 0.20, 0.50)]
    check("plain-link implied kappa", plain, [16.28, 6.47, 4.76], tol=0.01)
    # Re-pinned 2026-09-14 against the achievable checklist sizes (#320); under
    # the printed sizes: 17.94 / 7.44 / 5.50, and a mixed link reaching p = 0.5
    # at f = 0.45 and p = 0.9 at 2.79 (difference 2.34), peak slope 0.228.
    check("difficulty-mixed implied kappa", mixed, [17.65, 7.29, 5.41], tol=0.01)

    def mixed_p(f: float) -> float:
        return float(sum(w / (1 + np.exp(-(f - d))) for w, d in zip(weights, d_k, strict=True)))

    f_50 = optimize.brentq(lambda f: mixed_p(f) - 0.5, -30, 30)
    f_90 = optimize.brentq(lambda f: mixed_p(f) - 0.9, -30, 30)
    grid = np.linspace(-6, 9, 6001)
    peak = float(np.gradient([mixed_p(f) for f in grid], grid).max())
    check("mixed link: f for p = 0.5", f_50, 0.49, tol=0.005)
    check("mixed link: f for p = 0.9", f_90, 2.81, tol=0.005)
    check("mixed link: f(0.9) - f(0.5)", f_90 - f_50, 2.32, tol=0.005)
    check("mixed link: peak dp/df", peak, 0.230, tol=0.001)
    check("plain link: f for p = 0.9", np.log(9), 2.20, tol=0.005)


def check_frame_counts(merged: pd.DataFrame) -> None:
    print("§9 and §10 — frame counts and the Edgin anchor")
    # Re-pinned 2026-09-14 (note §15 item 12): the pool grew with the us_03
    # ingestion and moved with the masking and withholding rules added since
    # August (1,636 / 845 / 413 / 432 then).
    with_age = merged.dropna(subset=["age"])
    raw_pairs = with_age.groupby(["study", "subject_id"]).size()
    check("raw view: observations / children / singletons / repeated", [len(with_age), len(raw_pairs), (raw_pairs == 1).sum(), (raw_pairs > 1).sum()], [1918, 1024, 487, 537])

    # The note's frame and understood-pool figures (§9, §8 item 3, §12 item 6)
    # are counted through the package itself. Two hand-derived versions of the
    # understood counts went stale when masking rules changed under them, and so
    # did a hand-built "fitted frame" here -- the merged CSV with only ie_01's
    # baseline wave masked -- which by 2026-09-14 counted 1,011 children against
    # the 943 the model of record actually fits (832 / 460 / 372 in August).
    try:
        from vocab_growth.analysis_frames import build_analysis_frame
        from vocab_growth.data_utils import load_combined_data
        from vocab_growth.models.definitions import MODEL_REGISTRY

        pool = load_combined_data()
        key = FITTED_MODEL.lower()
        frame, _meta = build_analysis_frame(key, MODEL_REGISTRY[key])
    except Exception as error:  # pragma: no cover - environment-dependent
        print(f"  [skip] package-derived counts unavailable ({error})")
    else:
        pairs = frame.groupby(["study", "subject_id"]).size()
        check(
            f"{FITTED_MODEL} analysis frame: rows / children / singletons / repeated",
            [len(frame), len(pairs), (pairs == 1).sum(), (pairs > 1).sum()],
            [1708, 943, 504, 439],
        )
        understood = pool[pool["understood"].notna()]
        native = understood[understood["survey_vocab_max"] == 810]
        check("understood observations after all masking", len(understood), 1301)
        check(
            "dse-native understood observations / children / sources",
            [len(native), native["subject_id"].nunique(), native["study"].nunique()],
            [250, 170, 4],
        )

    us_01 = merged[merged["study"] == "us_01"]
    check("us_01 rows / children / rows with comprehension", [len(us_01), us_01["subject_id"].nunique(), int(us_01["understood"].notna().sum())], [345, 122, 174])


def check_imitation_decomposition(ie: pd.DataFrame) -> None:
    print("§11 — pooled imitation decomposition (follow-up wave)")
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    imitates = ie[["imitates_1_end", "imitates_2_end", "imitates_3_end"]].to_numpy(float)
    says = ie[["says_1_end", "says_2_end", "says_3_end"]].to_numpy(float)
    ok = ~np.isnan(end_u).any(axis=1) & ~np.isnan(imitates).any(axis=1) & ~np.isnan(says).any(axis=1)
    coherent = ok & (says <= imitates).all(axis=1) & (imitates <= end_u).all(axis=1)
    check("coherent says <= imitates <= understands records", coherent.sum(), 24)
    check("pooled P(imitate | understand) by stratum", imitates[coherent].sum(axis=0) / end_u[coherent].sum(axis=0), [0.68, 0.56, 0.63], tol=0.005)
    check("pooled P(say | imitate) by stratum", says[coherent].sum(axis=0) / imitates[coherent].sum(axis=0), [0.85, 0.78, 0.63], tol=0.005)


def main() -> int:
    ie = pd.read_csv(REPO / "data" / "vocab_data_ie_01.csv")
    uk = pd.read_csv(REPO / "data" / "vocab_data_uk_01.csv")
    merged_path = REPO / "data" / "vocab_data_merged.csv"
    output_root = Path(os.environ.get("DSE_VOCAB_GROWTH_OUTPUT_DIR") or (REPO / "output"))

    check_exchangeability(ie)
    check_production_gradient(ie)
    check_data_defects(ie, uk)
    check_fitted_dispersion(output_root)
    check_link_tables(ie)
    if merged_path.exists():
        check_frame_counts(pd.read_csv(merged_path))
    else:
        print("§9/§10 — [skip] data/vocab_data_merged.csv not present (run scripts/prepare_data.py)")
    check_imitation_decomposition(ie)

    if _failures:
        print(f"\n{len(_failures)} check(s) FAILED: " + "; ".join(_failures))
        return 1
    print("\nAll executed checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
