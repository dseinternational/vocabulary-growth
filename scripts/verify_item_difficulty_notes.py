# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recompute the checkable figures in the item-difficulty working note.

Verifies the numbers in
``notes/202607261540-item-difficulty-and-the-aggregate-likelihood.md``
(§§2, 3.3, 4, 5, 8, 9, 10, 11 — the consolidation of two earlier notes whose
figures this script originally pinned; their git history holds the trail) from
the raw CSVs and, where present, the fitted VG10 output. Each check prints
CLAIM vs COMPUTED; the script exits non-zero if any executed check fails.
Sections that need fitted output are skipped (not failed) when the output root
has no VG10 directory.

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
STRATUM_SIZES = np.array([120, 340, 350])
N_ITEMS = 810
SIM_SEED = 20260726

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


def check_exchangeability(ie: pd.DataFrame) -> None:
    print("§2 — hypergeometric exchangeability test (ie_01 follow-up wave)")
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    complete = ~np.isnan(end_u).any(axis=1)
    totals = end_u.sum(axis=1)
    tested = complete & (totals > 0) & (totals < N_ITEMS)
    check("complete follow-up records", complete.sum(), 46)
    check("records with 0 < T < 810", tested.sum(), 44)

    counts = end_u[tested]
    T = totals[tested]
    mean_k = np.outer(T, STRATUM_SIZES / N_ITEMS)
    var_k = np.outer(T * (N_ITEMS - T) / (N_ITEMS - 1), (STRATUM_SIZES / N_ITEMS) * (1 - STRATUM_SIZES / N_ITEMS))
    z = (counts - mean_k) / np.sqrt(var_k)
    check("RMS z, all strata", np.sqrt((z**2).mean()), 9.61, tol=0.005)
    check("mean z per checklist", z.mean(axis=0), [8.40, 3.00, -9.01], tol=0.005)

    n = len(T)
    pos1 = int((counts[:, 0] > mean_k[:, 0]).sum())
    neg3 = int((counts[:, 2] < mean_k[:, 2]).sum())
    check("Checklist 1 positive excess", pos1, 37)
    check("Checklist 3 negative deficit", neg3, 42)
    check("sign test p, Checklist 1 (1e-6)", stats.binomtest(pos1, n).pvalue / 1e-6, 5.3, tol=0.05)
    check("sign test p, Checklist 3 (1e-10)", stats.binomtest(neg3, n).pvalue / 1e-10, 1.1, tol=0.05)

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
    check("pooled proportions over the 46", pooled_46.sum(axis=0) / (complete.sum() * STRATUM_SIZES), [0.671, 0.452, 0.253], tol=0.0005)
    check("overall pooled proportion", pooled_46.sum() / (complete.sum() * N_ITEMS), 0.398, tol=0.0005)
    check("per-child mean proportions over the 44", props.mean(axis=0), [0.701, 0.473, 0.264], tol=0.0005)
    pooled_props = pooled_46.sum(axis=0) / (complete.sum() * STRATUM_SIZES)
    spread = np.log(pooled_props[0] / (1 - pooled_props[0])) - np.log(pooled_props[2] / (1 - pooled_props[2]))
    check("outer-checklist spread (logits)", spread, 1.8, tol=0.05)


def check_production_gradient(ie: pd.DataFrame) -> None:
    print("§5 — production propensity gradient")
    end_u = ie[["understands_1_end", "understands_2_end", "understands_3_end"]].to_numpy(float)
    says = ie[["says_1_end", "says_2_end", "says_3_end"]].to_numpy(float)
    both = ~np.isnan(end_u).any(axis=1) & ~np.isnan(says).any(axis=1)
    coherent = both & (says <= end_u).all(axis=1) & (end_u.sum(axis=1) > 0)
    check("coherent records", coherent.sum(), 38)

    S, U = says[coherent], end_u[coherent]
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
    check("pooled Checklist 1, baseline (46)", np.nansum(start_1[complete]) / (complete.sum() * 120), 0.855, tol=0.0005)
    check("pooled Checklist 1, follow-up (46)", np.nansum(end_u[complete, 0]) / (complete.sum() * 120), 0.671, tol=0.0005)
    check(
        "mean understood total, baseline -> follow-up (46)",
        [ie["understands_total_start"].to_numpy(float)[complete].mean(), ie["understands_total_end"].to_numpy(float)[complete].mean()],
        [252, 323],
        tol=0.5,
    )
    delta_1 = end_u[:, 0] - start_1
    check("children whose Checklist 1 falls", (delta_1 < 0).sum(), 22)
    check("largest fall", np.nanmin(delta_1), -124)
    over = int((start_1 > 120).sum() + (end_u[:, 0] > 120).sum())
    check("records with Checklist 1 > 120", over, 27)
    check("maximum Checklist 1 count", np.nanmax(np.concatenate([start_1, end_u[:, 0]])), 124)

    category_cols = [c for c in uk.columns if c.endswith("c") and not c.startswith("t")]
    check("uk_01 comprehension category columns", len(category_cols), 19)
    categories = uk[category_cols].to_numpy(float)
    complete_uk = ~np.isnan(categories).any(axis=1)
    check("uk_01 complete category rows", complete_uk.sum(), 29)
    understood = uk.loc[complete_uk, "understood"].to_numpy(float)
    check("category sums reconcile exactly", np.abs(categories[complete_uk].sum(axis=1) - understood).max(), 0)
    ratio = uk.loc[complete_uk, "spoken"].to_numpy(float) / understood
    check("children with spoken/understood > 1", (ratio > 1).sum(), 2)
    check("maximum spoken/understood", np.nanmax(ratio), 1.95, tol=0.005)


def check_fitted_dispersion(output_root: Path) -> None:
    print("§4 — VG10 fitted dispersion (skipped if no fitted output)")
    vg10_dirs = sorted((output_root / "models").glob("VG10-*")) if (output_root / "models").is_dir() else []
    if not vg10_dirs:
        print("  [skip] no VG10 output found under", output_root / "models")
        return
    vg10 = vg10_dirs[0]
    ages = np.array([12, 24, 30, 48, 66])

    summary = pd.read_csv(vg10 / "posterior_summary_u.csv")
    p_fit = np.array([summary.loc[(summary["age_months"] - a).abs().idxmin(), "p_median"] for a in ages])
    check("fitted p at 12/24/30/48/66", p_fit, [0.028, 0.146, 0.227, 0.379, 0.526], tol=0.0005)

    kappa_tab = pd.read_csv(vg10 / "posterior_kappa_u.csv")
    kappa = np.array([kappa_tab.loc[(kappa_tab["age_months"] - a).abs().idxmin(), "kappa_median"] for a in ages])
    check("fitted kappa (posterior_kappa_u medians)", kappa, [39.77, 29.64, 25.66, 16.97, 11.68], tol=0.02)

    diagnostics = pd.read_csv(vg10 / "diagnostics.csv", index_col=0)
    tau_subj = float(diagnostics.loc["tau_subj_u", "mean"])
    check("tau_subj_u posterior mean", tau_subj, 0.754, tol=0.0005)

    residual_sd = 1 / np.sqrt(p_fit * (1 - p_fit) * (kappa + 1))
    check("implied residual latent SD", residual_sd, [0.956, 0.512, 0.462, 0.486, 0.562], tol=0.002)
    check("total latent SD with tau_subj_u", np.sqrt(tau_subj**2 + residual_sd**2), [1.218, 0.911, 0.884, 0.897, 0.941], tol=0.002)

    decline_kp1 = np.log((kappa[0] + 1) / (kappa[-1] + 1))
    predicted = np.log((p_fit[-1] * (1 - p_fit[-1])) / (p_fit[0] * (1 - p_fit[0])))
    check("log decline in kappa+1", decline_kp1, 1.168, tol=0.005)
    check("constant-spread prediction", predicted, 2.230, tol=0.005)
    check("ratio", decline_kp1 / predicted, 0.52, tol=0.005)
    check("log decline in kappa", np.log(kappa[0] / kappa[-1]), 1.225, tol=0.005)

    check_kernel_share(vg10)


def check_kernel_share(vg10: Path) -> None:
    """Note §3.3: how much total variance the item-exchangeability kernel carries.

    Rasch sufficiency means heterogeneous item difficulty can only reach the model
    through the distribution of the total, so this share bounds the whole concern.
    It is `1 / VIF` where `VIF = (N + kappa) / (kappa + 1)` is the Beta-Binomial's
    inflation over its Binomial kernel. An earlier draft of the note quoted VG07's
    figures as though they were the model of record's, understating the exposure
    threefold — hence checking against VG10's own output here.
    """
    print("§3.3 — kernel share of total variance (VG10)")
    for outcome, share_claim, sd_claim in (("u", [0.77, 5.27], 1.06), ("s", [0.83, 0.86], 0.17)):
        table = vg10 / f"posterior_kappa_{outcome}.csv"
        if not table.exists():
            print(f"  [skip] {table.name} not present")
            continue
        kappa = pd.read_csv(table)["kappa_median"].to_numpy(float)
        share = 100.0 * (kappa + 1.0) / (N_ITEMS + kappa)
        check(f"kernel share % ({outcome}, min/max)", [share.min(), share.max()], share_claim, tol=0.01)
        # A 40% error in a component carrying `share` of the variance (the
        # underdispersion at a 2-logit difficulty spread) moves the total SD by:
        worst = 100.0 * (1.0 - np.sqrt(1.0 - 0.40 * share.max() / 100.0))
        check(f"worst-case total SD shift % ({outcome})", worst, sd_claim, tol=0.01)


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
    check("difficulty-mixed implied kappa", mixed, [17.94, 7.44, 5.50], tol=0.01)

    def mixed_p(f: float) -> float:
        return float(sum(w / (1 + np.exp(-(f - d))) for w, d in zip(weights, d_k, strict=True)))

    f_50 = optimize.brentq(lambda f: mixed_p(f) - 0.5, -30, 30)
    f_90 = optimize.brentq(lambda f: mixed_p(f) - 0.9, -30, 30)
    grid = np.linspace(-6, 9, 6001)
    peak = float(np.gradient([mixed_p(f) for f in grid], grid).max())
    check("mixed link: f for p = 0.5", f_50, 0.45, tol=0.005)
    check("mixed link: f for p = 0.9", f_90, 2.79, tol=0.005)
    check("mixed link: f(0.9) - f(0.5)", f_90 - f_50, 2.34, tol=0.005)
    check("mixed link: peak dp/df", peak, 0.228, tol=0.001)
    check("plain link: f for p = 0.9", np.log(9), 2.20, tol=0.005)


def check_frame_counts(merged: pd.DataFrame) -> None:
    print("§9 and §10 — frame counts and the Edgin anchor")
    with_age = merged.dropna(subset=["age"])
    raw_pairs = with_age.groupby(["study", "subject_id"]).size()
    check("raw view: observations / children / singletons / repeated", [len(with_age), len(raw_pairs), (raw_pairs == 1).sum(), (raw_pairs > 1).sum()], [1219, 626, 235, 391])

    fitted = with_age.copy()
    outcome_cols = ["understood", "spoken", "signed", "produced"]
    masked = (fitted["study"] == "ie_01") & (fitted["survey_vocab_max"] == 460)
    fitted.loc[masked, outcome_cols] = np.nan
    fitted = fitted[~fitted[outcome_cols].isna().all(axis=1)]
    fit_pairs = fitted.groupby(["study", "subject_id"]).size()
    check("fitted frame: children / singletons / repeated", [len(fit_pairs), (fit_pairs == 1).sum(), (fit_pairs > 1).sum()], [613, 282, 331])

    understood = merged.dropna(subset=["understood"])
    check("understood observations, raw / after masking", [len(understood), len(understood) - int((understood["study"] == "ie_01").sum() - 46)], [739, 680])

    us_01 = merged[merged["study"] == "us_01"]
    check("us_01 rows / children / rows with comprehension", [len(us_01), us_01["subject_id"].nunique(), int(us_01["understood"].notna().sum())], [196, 119, 87])


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
