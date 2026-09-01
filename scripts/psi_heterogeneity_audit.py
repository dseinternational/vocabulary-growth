# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Audit the between-study heterogeneity in VG15's sign-speech association.

Recomputes every number in ``notes/202608121030-psi-heterogeneity-and-age-invariance.md``
from the committed source CSVs, so the evidence for two model decisions can be
re-run after any later data or loader change:

1.  ``psi`` carries a **study-level term** (``delta_psi``), because the four
    cross-tab sources disagree about the association by an order of magnitude.
2.  ``psi`` does **not** vary with age, because the apparent age gradient is
    entirely between-study confounding — uk_07 is both the oldest sample and the
    highest-association one.

Everything here is descriptive: Mantel-Haenszel odds ratios and weighted least
squares on per-child log odds ratios, computed directly from the four cells. None
of it is the fitted ``psi``, which is a population-conditioned quantity defined
against the model's own r and q. These statistics are diagnostic of heterogeneity
and of confounding, not estimates of the parameter — but they are computed
identically for every source, which is what makes the comparison meaningful.

Run from the repository root:

    python scripts/psi_heterogeneity_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vocab_growth import cross_tab_sources
from vocab_growth.reporting import console, dataframe_table, heading, key_value_table

# Haldane-Anscombe correction: added to every cell before taking a per-child log
# odds ratio, so children with an empty cell still contribute. It biases each
# ratio toward 1, most strongly where counts are small — which is why the very
# youngest / smallest-vocabulary bands below should not be read as point estimates.
HALDANE = 0.5


def _cells() -> pd.DataFrame:
    """Every within-understood four-cell row, from all three sources that have one.

    nz_01 is deliberately absent: it records no comprehension total, so its cells
    are a within-*produced* three-cell partition and its "neither" would span all
    unproduced items rather than understood-but-unproduced. It is reported
    separately in :func:`report_association_by_source` with that caveat attached,
    because the reference set changes the odds ratio (see the uk_07 both-ways row).
    """
    four_02, _ = cross_tab_sources.load_uk02_four_cell()
    four_07, _ = cross_tab_sources.load_uk07_four_cell()
    four_es, _ = cross_tab_sources.load_es01_four_cell()

    def frame(study, age, neither, sign_only, speak_only, both, subject):
        return pd.DataFrame({
            "study": study,
            "subject_id": np.asarray(subject),
            "age": np.asarray(age, dtype=float),
            "neither": np.asarray(neither, dtype=float),
            "sign_only": np.asarray(sign_only, dtype=float),
            "speak_only": np.asarray(speak_only, dtype=float),
            "both": np.asarray(both, dtype=float),
        })

    frames = [
        frame("uk_02", four_02["age"], four_02["understood_only"],
              four_02["signed_only"], four_02["spoken_only"],
              four_02["signed_spoken"], four_02["subject_id"]),
        # uk_07's expressive columns are modality-exclusive cells, so they map
        # straight onto the partition (see data/vocab_data_uk_07.md).
        frame("uk_07", four_07["age"], four_07["understood_only"],
              four_07["signed"], four_07["spoken"],
              four_07["spoken_signed"], four_07["subject_id"]),
        frame("es_01", four_es["age"], four_es["understood_only"],
              four_es["signed_only"], four_es["spoken_only"],
              four_es["signed_spoken"], four_es["subject_id"]),
    ]
    out = pd.concat(frames, ignore_index=True)
    out["produced"] = out["sign_only"] + out["speak_only"] + out["both"]
    out["understood"] = out["neither"] + out["produced"]
    return out


def mantel_haenszel(df: pd.DataFrame) -> float:
    """Mantel-Haenszel odds ratio with each child as its own stratum.

    Stratifying by child removes between-child mixing, which a pooled ratio over
    summed cells does not: a source whose children differ in vocabulary level can
    show a pooled ratio on the wrong side of every individual one.
    """
    total = df["neither"] + df["sign_only"] + df["speak_only"] + df["both"]
    keep = total > 0
    if not keep.any():
        return float("nan")
    numerator = (df["both"][keep] * df["neither"][keep] / total[keep]).sum()
    denominator = (df["sign_only"][keep] * df["speak_only"][keep] / total[keep]).sum()
    return float(numerator / denominator) if denominator else float("nan")


def _log_or_and_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Per-child log odds ratio and its inverse-variance weight."""
    cells = df[["neither", "sign_only", "speak_only", "both"]] + HALDANE
    out = df.copy()
    out["log_or"] = np.log(
        cells["both"] * cells["neither"] / (cells["sign_only"] * cells["speak_only"])
    )
    out["weight"] = 1.0 / (1.0 / cells).sum(axis=1)
    return out[np.isfinite(out["log_or"]) & np.isfinite(out["weight"])]


def _wls(y: np.ndarray, X: np.ndarray, w: np.ndarray):
    """Weighted least squares. Returns (coefficients, standard errors, weighted SSR)."""
    xtw = X.T @ np.diag(w)
    beta = np.linalg.solve(xtw @ X, xtw @ y)
    resid = y - X @ beta
    ssr = float((w * resid**2).sum())
    dof = len(y) - X.shape[1]
    cov = (ssr / dof) * np.linalg.inv(xtw @ X)
    return beta, np.sqrt(np.diag(cov)), ssr


def report_association_by_source(cells: pd.DataFrame) -> None:
    """The heterogeneity that motivated ``delta_psi``."""
    heading("Association by source")
    rows = []
    for study, group in cells.groupby("study"):
        scored = _log_or_and_weight(group)
        per_child = np.exp(scored["log_or"])
        non_vocal = group["sign_only"].sum() + group["both"].sum()
        rows.append({
            "source": study,
            "rows": len(group),
            "MH_odds_ratio": round(mantel_haenszel(group), 2),
            "reference_set": "within understood",
            "per_child_OR_below_1": f"{(per_child < 1).mean():.0%}",
            "non_vocal_also_spoken": f"{group['both'].sum() / non_vocal:.1%}",
        })

    # nz_01: within-PRODUCED cells, so its "neither" spans all unproduced items.
    nz = cross_tab_sources.load_nz01_produced_cells()
    nz_cells = pd.DataFrame({
        "neither": nz["prod_signed_only"] * 0 + (675 - nz["prod_total"]),
        "sign_only": nz["prod_signed_only"],
        "speak_only": nz["prod_spoken_only"],
        "both": nz["prod_signed_spoken"],
    })
    nz_scored = _log_or_and_weight(nz_cells.assign(study="nz_01", age=nz["age"]))
    rows.append({
        "source": "nz_01",
        "rows": len(nz),
        "MH_odds_ratio": round(mantel_haenszel(nz_cells), 2),
        "reference_set": "all 675 items",
        "per_child_OR_below_1": f"{(np.exp(nz_scored['log_or']) < 1).mean():.0%}",
        "non_vocal_also_spoken":
            f"{nz_cells['both'].sum() / (nz_cells['sign_only'] + nz_cells['both']).sum():.1%}",
    })
    dataframe_table(pd.DataFrame(rows), title="Sign-speech association by source",
                    show_index=False)

    # The reference set is not a detail: the same uk_07 data both ways.
    uk07 = cells[cells["study"] == "uk_07"]
    widened = uk07.assign(neither=674 - uk07["produced"])
    key_value_table("Why the reference set matters (uk_07, same rows)", [
        ("within understood", round(mantel_haenszel(uk07), 2)),
        ("over all 674 items", round(mantel_haenszel(widened), 2)),
    ])
    console.print(
        "[yellow]Magnitudes compare only within a reference set. The per-child sign "
        "and the share-also-spoken column need no 'neither' cell and compare "
        "throughout.[/yellow]"
    )


def report_age_confounding(cells: pd.DataFrame) -> None:
    """Why psi carries no age term: the age gradient is between-study confounding."""
    heading("Age: gradient or confounding?")

    scored = _log_or_and_weight(cells)
    studies = sorted(scored["study"].unique())
    dummies = np.column_stack([(scored["study"] == s).astype(float) for s in studies])
    age_years = scored["age"].to_numpy() / 12.0
    y, w = scored["log_or"].to_numpy(), scored["weight"].to_numpy()

    key_value_table("Age distribution of psi-informing rows", [
        (s, f"median {g['age'].median():.0f} mo (IQR {g['age'].quantile(.25):.0f}"
            f"-{g['age'].quantile(.75):.0f}), n={len(g)}")
        for s, g in cells.groupby("study")
    ])

    # Age WITHOUT a study term is free to claim the between-study variation.
    beta, se, ssr_age = _wls(y, np.column_stack([np.ones(len(y)), age_years]), w)
    _, _, ssr_study = _wls(y, dummies, w)
    _, _, ssr_both = _wls(y, np.column_stack([dummies, age_years]), w)
    beta_adj, se_adj, _ = _wls(y, np.column_stack([dummies, age_years]), w)

    dataframe_table(pd.DataFrame([
        {"model": "age only (no study term)", "age_slope_per_year": round(beta[1], 3),
         "SE": round(se[1], 3), "z": round(beta[1] / se[1], 2),
         "weighted_SSR": round(ssr_age)},
        {"model": "study only", "age_slope_per_year": None, "SE": None, "z": None,
         "weighted_SSR": round(ssr_study)},
        {"model": "study + age", "age_slope_per_year": round(beta_adj[-1], 3),
         "SE": round(se_adj[-1], 3), "z": round(beta_adj[-1] / se_adj[-1], 2),
         "weighted_SSR": round(ssr_both)},
    ]), title="Age versus study as an explanation", show_index=False)
    console.print(
        "[yellow]Age alone looks strong, but fits worse than study alone and adds "
        "nothing on top of it. Study fixed effects absorb between-study age "
        "differences by construction, so the adjusted slope answers only the "
        "WITHIN-study question.[/yellow]"
    )

    # The direct test: hold age fixed and compare sources.
    for low, high in [(34, 56), (30, 60)]:
        window = cells[(cells["age"] >= low) & (cells["age"] < high)]
        dataframe_table(pd.DataFrame([
            {"source": s, "MH_psi": round(mantel_haenszel(g), 2), "rows": len(g),
             "median_age": round(g["age"].median())}
            for s, g in window.groupby("study")
        ]), title=f"Age-matched contrast, {low}-{high} months", show_index=False)


def report_level_gradient(cells: pd.DataFrame) -> None:
    """The level gradient, and the circularity that makes half of it unreadable."""
    heading("Developmental level")

    scored = _log_or_and_weight(cells)
    log_produced = np.log(scored["produced"].to_numpy() + 1.0)
    y, w = scored["log_or"].to_numpy(), scored["weight"].to_numpy()
    studies = sorted(scored["study"].unique())
    dummies = np.column_stack([(scored["study"] == s).astype(float) for s in studies])
    beta, se, _ = _wls(y, np.column_stack([dummies, log_produced]), w)

    rows = [{"scope": "pooled (study-adjusted)", "slope": round(beta[-1], 3),
             "SE": round(se[-1], 3), "z": round(beta[-1] / se[-1], 2), "n": len(scored)}]
    for study in studies:
        g = scored[scored["study"] == study]
        b, s, _ = _wls(g["log_or"].to_numpy(),
                       np.column_stack([np.ones(len(g)), np.log(g["produced"] + 1.0)]),
                       g["weight"].to_numpy())
        rows.append({"scope": study, "slope": round(b[1], 3), "SE": round(s[1], 3),
                     "z": round(b[1] / s[1], 2), "n": len(g)})
    dataframe_table(pd.DataFrame(rows),
                    title="Slope of log psi on log produced vocabulary",
                    show_index=False)
    console.print(
        "[yellow]CIRCULAR COVARIATE: produced = sign_only + speak_only + both, three "
        "of the four cells that define psi. Raising produced at fixed comprehension "
        "shrinks `neither` and mechanically LOWERS the ratio, so the induced bias is "
        "negative. Positive slopes run against it and are conservative; negative ones "
        "are not established.[/yellow]"
    )

    # es_01 is the only source carrying a developmental measure external to the
    # cells, so it is the only place the clean version of this test can be run.
    es = pd.read_csv("./data/vocab_data_es_01.csv")
    es = es.assign(
        neither=es["understood"] - es["spoken_or_gestured"],
        speak_only=es["spoken_or_gestured"] - es["gestured"],
        sign_only=es["spoken_or_gestured"] - es["spoken"],
        both=es["spoken"] + es["gestured"] - es["spoken_or_gestured"],
    )
    es = es[(es[["neither", "speak_only", "sign_only", "both"]] >= 0).all(axis=1)]
    es_scored = _log_or_and_weight(es.assign(study=es["group"], age=es["age"]))
    rows = []
    for group in ("DS", "TD"):
        g = es_scored[es_scored["study"] == group]
        for label, covariate in [("mental age (per 6 mo) [external]",
                                  g["mental_age"].to_numpy() / 6.0),
                                 ("chronological age (per year)",
                                  g["age"].to_numpy() / 12.0)]:
            b, s, _ = _wls(g["log_or"].to_numpy(),
                           np.column_stack([np.ones(len(g)), covariate]),
                           g["weight"].to_numpy())
            rows.append({"group": group, "covariate": label, "slope": round(b[1], 3),
                         "SE": round(s[1], 3), "z": round(b[1] / s[1], 2), "n": len(g)})
    dataframe_table(pd.DataFrame(rows),
                    title="es_01 against a covariate external to the cells",
                    show_index=False)
    key_value_table("es_01, identical instrument, both groups", [
        (group, round(mantel_haenszel(es_scored[es_scored["study"] == group]), 2))
        for group in ("DS", "TD")
    ])


def report_uk07_trial_arm(cells: pd.DataFrame) -> None:
    """Whether uk_07's intervention explains its association. It does not."""
    heading("uk_07 trial arm")
    source = pd.read_csv("./data/vocab_data_uk_07.csv")
    source = source[source["understood"] >= source["produced"]]
    source = source.assign(
        neither=source["understood"] - source["produced"],
        sign_only=source["signed"], speak_only=source["spoken"],
        both=source["spoken_signed"],
    )
    dataframe_table(pd.DataFrame([
        {"arm": arm, "timepoint": tp, "MH_psi": round(mantel_haenszel(g), 2), "n": len(g)}
        for (arm, tp), g in source.groupby(["group", "timepoint"])
    ]), title="Association by arm and timepoint", show_index=False)
    dataframe_table(pd.DataFrame([
        {"arm": arm, "MH_psi": round(mantel_haenszel(g), 2), "n": len(g)}
        for arm, g in source.groupby("group")
    ]), title="Association by arm (pooled over timepoints)", show_index=False)
    console.print(
        "[yellow]The CONTROL arm is higher, and the gap is present at t1 before any "
        "intervention — a baseline imbalance at 15 per arm, not an effect. uk_07's "
        "association is a property of its context, not of its trial.[/yellow]"
    )


def main() -> None:
    cells = _cells()
    report_association_by_source(cells)
    report_age_confounding(cells)
    report_level_gradient(cells)
    report_uk07_trial_arm(cells)
    heading("Conclusion")
    console.print(
        "psi carries a STUDY-level term (delta_psi) and NO age term. The sources "
        "disagree by an order of magnitude at matched ages; age alone fits worse "
        "than study alone and adds nothing on top of it. See "
        "notes/202608121030-psi-heterogeneity-and-age-invariance.md."
    )


if __name__ == "__main__":
    main()
