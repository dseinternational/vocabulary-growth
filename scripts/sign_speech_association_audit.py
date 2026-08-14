# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Is signing additive or substitutive? Between-child association and cross-lag.

Two questions the fitted models do **not** answer, measured directly from the
four-cell cross-tabulation sources. ``psi`` (VG15) is a *within*-child,
item-level overlap parameter with a study-level term; it says how much of a
child's signing lands on words that child also speaks. Neither it nor any other
fitted parameter says whether children who sign more produce more overall, or
whether signing at one wave precedes speech at the next.

Part A -- between-child (cross-sectional).
    At matched comprehension and age, is a child's spoken vocabulary larger or
    smaller when their signed vocabulary is larger? Two specifications bracket
    the mechanical bias, which is why both are reported:

    * ``spoken ~ sign_only`` -- the cells are disjoint, but the four cells sum
      to ``understood``, so at fixed comprehension they compete for one budget
      and the compositional constraint biases the slope **negative**. A positive
      slope here is therefore conservative; a negative one is not established.
    * ``spoken ~ signed`` -- both totals contain the ``both`` cell, so sharing
      biases the slope **positive**. This is the upper bracket.

    The truth sits between them. Reporting one alone would be a choice of answer.

Part B -- cross-lag (longitudinal).
    Among children measured more than once, does the sign-only vocabulary at one
    wave predict the *gain* in spoken vocabulary by the next, over and above the
    spoken vocabulary already there? The reverse direction is fitted too: a
    result that appears in both directions is general growth, not a lead.

Reference sets are not interchangeable. uk_02, uk_07 and es_01 partition the
words a child *understands*; nz_01 records no comprehension total, so its cells
partition only what the child *produces*. nz_01 is therefore reported in its own
section against a produced-vocabulary denominator, never pooled with the rest.

The cell mapping matches ``scripts/psi_heterogeneity_audit.py``: uk_07's
expressive columns are modality-exclusive cells, where uk_02's and es_01's
``signed``/``spoken`` are totals and the cells carry ``_only`` names.

Run from the repository root:

    python scripts/sign_speech_association_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from vocab_growth.models import common_joint_modality as cjm
from vocab_growth.reporting import console, dataframe_table, heading, key_value_table

WITHIN_UNDERSTOOD = ("uk_02", "uk_07", "es_01")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def within_understood_cells() -> pd.DataFrame:
    """Four-cell rows from the three sources that partition understood words."""
    four_02, _ = cjm._load_uk02_four_cell()
    four_07, _ = cjm._load_uk07_four_cell()
    four_es, _ = cjm._load_es01_four_cell()

    def frame(study, src, age, neither, sign_only, speak_only, both, subject, **extra):
        out = pd.DataFrame({
            "study": study,
            "subject_id": np.asarray(src[subject]),
            "age": np.asarray(src[age], dtype=float),
            "neither": np.asarray(src[neither], dtype=float),
            "sign_only": np.asarray(src[sign_only], dtype=float),
            "speak_only": np.asarray(src[speak_only], dtype=float),
            "both": np.asarray(src[both], dtype=float),
        })
        for key, col in extra.items():
            out[key] = np.asarray(src[col], dtype=float)
        return out

    frames = [
        frame("uk_02", four_02, "age", "understood_only", "signed_only",
              "spoken_only", "signed_spoken", "subject_id"),
        # uk_07's expressive columns are the exclusive cells themselves.
        frame("uk_07", four_07, "age", "understood_only", "signed",
              "spoken", "spoken_signed", "subject_id"),
        frame("es_01", four_es, "age", "understood_only", "signed_only",
              "spoken_only", "signed_spoken", "subject_id",
              mental_age="mental_age"),
    ]
    out = pd.concat(frames, ignore_index=True)

    # Derive the totals from the cells rather than trusting a separate source
    # column: four uk_02 rows carry a `comprehension` value 1-3 words above its
    # own cells, and the partition is what every estimand here is defined on.
    out["produced"] = out["sign_only"] + out["speak_only"] + out["both"]
    out["understood"] = out["neither"] + out["produced"]
    out["spoken"] = out["speak_only"] + out["both"]
    out["signed"] = out["sign_only"] + out["both"]
    return out


def nz01_cells() -> pd.DataFrame:
    """nz_01's produced-only three-cell partition (no comprehension total)."""
    raw = cjm._load_nz01_produced_cells()
    out = pd.DataFrame({
        "study": "nz_01",
        "subject_id": raw["subject_id"].to_numpy(),
        "age": raw["age"].to_numpy(dtype=float),
        "speak_only": raw["prod_spoken_only"].to_numpy(dtype=float),
        "sign_only": raw["prod_signed_only"].to_numpy(dtype=float),
        "both": raw["prod_signed_spoken"].to_numpy(dtype=float),
    })
    out["produced"] = out["speak_only"] + out["sign_only"] + out["both"]
    out["spoken"] = out["speak_only"] + out["both"]
    out["signed"] = out["sign_only"] + out["both"]
    return out


# ----------------------------------------------------------------------------
# Regression helper
# ----------------------------------------------------------------------------
def ols_cluster(df: pd.DataFrame, y: str, x: str, controls: list[str],
                cluster: str = "subject_id", *, standardise: bool = True) -> dict:
    """OLS slope of ``y`` on ``x`` given ``controls``, clustered on ``cluster``.

    Standard errors are clustered so repeated waves of the same child do not
    count as independent. With ``standardise`` the slope is in SD-per-SD units,
    comparable across sources whose vocabulary scales differ by an order of
    magnitude; without it the slope is in words per word, which is what the
    additive/substitutive decomposition needs.
    """
    cols = [y, x, *controls, cluster]
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < len(controls) + 4 or d[x].std() == 0 or d[y].std() == 0:
        return {"n": len(d), "n_children": 0, "beta": np.nan, "se": np.nan,
                "z": np.nan, "lo": np.nan, "hi": np.nan}

    def z(col):
        return (d[col] - d[col].mean()) / d[col].std() if standardise and d[col].std() else d[col]

    yv = z(y)
    design = pd.DataFrame({x: z(x)})
    for c in controls:
        design[c] = z(c)
    design = sm.add_constant(design, has_constant="add")

    fit = sm.OLS(yv.to_numpy(), design.to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster].to_numpy()}
    )
    beta, se = float(fit.params[1]), float(fit.bse[1])
    return {
        "n": int(len(d)),
        "n_children": int(d[cluster].nunique()),
        "beta": beta,
        "se": se,
        "z": float(fit.tvalues[1]),
        "lo": beta - 1.96 * se,
        "hi": beta + 1.96 * se,
    }


def _fmt(res: dict) -> str:
    if not np.isfinite(res["beta"]):
        return "—"
    return f"{res['beta']:+.3f} ({res['se']:.3f})"


# ----------------------------------------------------------------------------
# Part A — between-child
# ----------------------------------------------------------------------------
def report_additive_share(cells: pd.DataFrame, nz: pd.DataFrame) -> None:
    heading("A1. How much does signing add to a count of speech?")
    console.print(
        "Share of a child's produced vocabulary that is [bold]signed but not "
        "spoken[/bold] — what a spoken-only count misses. Pooled is the ratio of "
        "sums; per-child is the median over children, one row per child (first "
        "wave), which weights a small vocabulary equally with a large one.\n"
    )
    rows = []
    for study, d in pd.concat([cells, nz]).groupby("study"):
        first = d.sort_values("age").groupby("subject_id").first()
        share_first = first["sign_only"] / first["produced"].replace(0, np.nan)
        share_rows = d["sign_only"] / d["produced"].replace(0, np.nan)
        rows.append({
            "source": study,
            "children": int(d["subject_id"].nunique()),
            "rows": int(len(d)),
            "pooled (word-weighted)": f"{d['sign_only'].sum() / d['produced'].sum():.1%}",
            "median child (first wave)": f"{share_first.median():.1%}",
            "median observation": f"{share_rows.median():.1%}",
            "obs IQR": f"{share_rows.quantile(.25):.0%}-{share_rows.quantile(.75):.0%}",
        })
    dataframe_table(pd.DataFrame(rows), title="Sign-only share of produced vocabulary",
                    show_index=False)
    console.print(
        "[yellow]Three denominators, three different numbers, and none of them is "
        "'the' share. Pooled weights children by vocabulary size, so the oldest "
        "sample dominates it. The observation median counts a child once per wave, "
        "so repeat-measured children pull it toward their later, lower-signing "
        "waves — for nz_01 and uk_07 that is a 3- to 6-fold difference from the "
        "per-child figure. Quoting any single one as 'signing adds X%' would be "
        "misleading; the gradient below is the result.[/yellow]"
    )

    heading("A1b. Signs fully absorbed into speech")
    console.print(
        "Share of observations where the child signs nothing they do not also say "
        "— signing adding no vocabulary at all. No modelling, no denominator "
        "choice, and the clearest single view of the developmental pattern.\n"
    )
    rows = []
    for study, d in pd.concat([cells, nz]).groupby("study"):
        signing = d[d["signed"] > 0]
        rows.append({
            "source": study,
            "median age": round(float(d["age"].median())),
            "median spoken": round(float(d["spoken"].median())),
            "obs fully duplicated": f"{(d['sign_only'] == 0).mean():.1%}",
            "among obs with any signs": f"{(signing['sign_only'] == 0).mean():.1%}",
        })
    dataframe_table(pd.DataFrame(rows).sort_values("median spoken"),
                    title="Every sign also spoken", show_index=False)
    console.print(
        "[yellow]Ordered by how much speech the sample has, this rises from ~2% to "
        "~28%. But it is a contrast between four studies that differ in age, "
        "country, instrument and recruitment, so it confounds development with "
        "everything else. Part C tests the same thing within children.[/yellow]"
    )


def report_between_child(cells: pd.DataFrame) -> None:
    heading("A2. At matched comprehension, do signers speak more or less?")
    console.print(
        "Standardised slope (cluster-robust SE) of spoken vocabulary on the two "
        "signing measures, controlling for understood vocabulary and age.\n"
        "  [bold]sign_only[/bold] — disjoint cells; compositional constraint "
        "biases this [bold]negative[/bold], so a positive slope is conservative.\n"
        "  [bold]signed[/bold]    — shares the 'both' cell with spoken, which "
        "biases it [bold]positive[/bold]. Upper bracket.\n"
    )
    rows = []
    for study in WITHIN_UNDERSTOOD:
        d = cells[cells["study"] == study]
        lo = ols_cluster(d, "spoken", "sign_only", ["understood", "age"])
        hi = ols_cluster(d, "spoken", "signed", ["understood", "age"])
        rows.append({
            "source": study, "rows": lo["n"], "children": lo["n_children"],
            "beta_sign_only (lower)": _fmt(lo), "z": round(lo["z"], 2),
            "beta_signed (upper)": _fmt(hi), "z ": round(hi["z"], 2),
        })

    pooled = cells.copy()
    for study in WITHIN_UNDERSTOOD[1:]:
        pooled[f"study_{study}"] = (pooled["study"] == study).astype(float)
    study_dummies = [f"study_{s}" for s in WITHIN_UNDERSTOOD[1:]]
    lo = ols_cluster(pooled, "spoken", "sign_only", ["understood", "age", *study_dummies])
    hi = ols_cluster(pooled, "spoken", "signed", ["understood", "age", *study_dummies])
    rows.append({
        "source": "pooled (study FE)", "rows": lo["n"], "children": lo["n_children"],
        "beta_sign_only (lower)": _fmt(lo), "z": round(lo["z"], 2),
        "beta_signed (upper)": _fmt(hi), "z ": round(hi["z"], 2),
    })
    dataframe_table(pd.DataFrame(rows), title="Spoken vocabulary on signing, given comprehension and age",
                    show_index=False)

    # es_01 is the only source carrying a developmental measure external to the
    # cells, so it is the only place the comprehension control can be replaced
    # by something the cells do not define.
    es = cells[cells["study"] == "es_01"]
    ext_lo = ols_cluster(es, "spoken", "sign_only", ["mental_age", "age"])
    ext_hi = ols_cluster(es, "spoken", "signed", ["mental_age", "age"])
    key_value_table("es_01 only — controlling on mental age instead of comprehension", [
        ("sign_only slope (lower bracket)", f"{_fmt(ext_lo)}  z={ext_lo['z']:.2f}"),
        ("signed slope (upper bracket)", f"{_fmt(ext_hi)}  z={ext_hi['z']:.2f}"),
        ("n", ext_lo["n"]),
    ])


def report_total_decomposition(cells: pd.DataFrame) -> None:
    """Does the TOTAL rise? Raw words-per-word, where the identity is legible."""
    heading("A3. Per extra signed-not-spoken word, what happens to the total?")
    console.print(
        "Total production is an identity: [bold]produced = spoken + "
        "sign_only[/bold]. So the slope of produced on sign_only is exactly "
        "1 + (slope of spoken on sign_only), and the whole question is where it "
        "sits between 0 and 1:\n"
        "  [bold]1.0[/bold] — signing is purely additive; every sign is a word "
        "speech would have missed and nothing is displaced.\n"
        "  [bold]0.0[/bold] — signing fully displaces speech; the total does not "
        "move.\n"
        "Raw words per word, at matched comprehension and age.\n"
        "[bold red]Do not read this table alone.[/bold red] A single slope is the "
        "wrong summary here — A4 shows it averages a sign change across the "
        "comprehension range, and the pooled row is dominated by the top of it.\n"
    )
    rows = []
    for study in [*WITHIN_UNDERSTOOD, "pooled"]:
        if study == "pooled":
            d = cells.copy()
            for s in WITHIN_UNDERSTOOD[1:]:
                d[f"study_{s}"] = (d["study"] == s).astype(float)
            ctrl = ["understood", "age", *[f"study_{s}" for s in WITHIN_UNDERSTOOD[1:]]]
        else:
            d, ctrl = cells[cells["study"] == study], ["understood", "age"]
        spo = ols_cluster(d, "spoken", "sign_only", ctrl, standardise=False)
        tot = ols_cluster(d, "produced", "sign_only", ctrl, standardise=False)
        rows.append({
            "source": study,
            "children": spo["n_children"],
            "spoken per sign-only word": f"{spo['beta']:+.2f}",
            "total per sign-only word": f"{tot['beta']:.2f}",
            "95% CI": f"[{tot['lo']:.2f}, {tot['hi']:.2f}]",
        })
    dataframe_table(pd.DataFrame(rows),
                    title="Additive (1.0) vs fully substitutive (0.0)", show_index=False)
    console.print(
        "[yellow]Association, not effect. A child who signs more may be a child "
        "whose speech is harder, which produces exactly this pattern with no "
        "causal role for signing at all. No source in the pool records whether "
        "its children were taught to sign.[/yellow]"
    )


def report_stratified_check(cells: pd.DataFrame) -> None:
    """Model-free version of A3: the same contrast, with no regression at all.

    A3's slopes are large enough to be worth verifying without a functional form.
    One row per child, comprehension quartiles, then a within-quartile split on
    sign-only vocabulary. If the regression is describing the data rather than an
    artefact of the linear control, spoken should fall across the split and the
    total should be flat-to-falling.
    """
    heading("A4. The same contrast without a model")
    first = cells.sort_values("age").groupby(["study", "subject_id"], as_index=False).first()
    first["U_band"] = pd.qcut(first["understood"], 4,
                              labels=["U q1", "U q2", "U q3", "U q4"])
    rows = []
    for band, d in first.groupby("U_band", observed=True):
        if len(d) < 6:
            continue
        d = d.assign(grp=pd.qcut(d["sign_only"].rank(method="first"), 3,
                                 labels=["low sign", "mid", "high sign"]))
        for grp, g in d.groupby("grp", observed=True):
            rows.append({
                "comprehension band": str(band),
                "understood (mean)": round(float(g["understood"].mean())),
                "sign-only group": str(grp),
                "n": len(g),
                "sign_only": round(float(g["sign_only"].mean())),
                "spoken": round(float(g["spoken"].mean())),
                "TOTAL produced": round(float(g["produced"].mean())),
            })
    dataframe_table(pd.DataFrame(rows),
                    title="One row per child; within comprehension quartile, by sign-only tertile",
                    show_index=False)
    console.print(
        "[bold red]The pooled slope in A3 does not describe this.[/bold red] "
        "Spoken falls with signing in the upper bands, but the TOTAL is flat or "
        "higher for high signers in the middle bands and only collapses in the "
        "top one. A single linear slope averages a sign change.\n"
    )

    # Quantify the heterogeneity the pooled row hides.
    rows = []
    for band, d in first.groupby("U_band", observed=True):
        tot = ols_cluster(d, "produced", "sign_only", ["understood", "age"],
                          standardise=False)
        spo = ols_cluster(d, "spoken", "sign_only", ["understood", "age"],
                          standardise=False)
        rows.append({
            "comprehension band": str(band),
            "children": tot["n"],
            "understood range": f"{d['understood'].min():.0f}-{d['understood'].max():.0f}",
            "spoken per sign-only word": f"{spo['beta']:+.2f}",
            "total per sign-only word": f"{tot['beta']:+.2f}",
            "95% CI": f"[{tot['lo']:.2f}, {tot['hi']:.2f}]",
            "largest source": d["study"].value_counts().idxmax(),
        })
    dataframe_table(pd.DataFrame(rows),
                    title="A3's slope refitted within comprehension quartile",
                    show_index=False)

    # es_01 is 185 of the 243 children and is the one source whose within-child
    # association is ~1 where the others are 6-15, so the gradient above could be
    # es_01's alone. Split each source at its own comprehension median.
    rows = []
    for study, d in first.groupby("study"):
        med = d["understood"].median()
        for half, g in (("lower half", d[d["understood"] <= med]),
                        ("upper half", d[d["understood"] > med])):
            tot = ols_cluster(g, "produced", "sign_only", ["understood", "age"],
                              standardise=False)
            rows.append({
                "source": study, "half": half, "children": tot["n"],
                "understood range": f"{g['understood'].min():.0f}-{g['understood'].max():.0f}",
                "total per sign-only word": f"{tot['beta']:+.2f}",
                "95% CI": f"[{tot['lo']:.2f}, {tot['hi']:.2f}]",
            })
    dataframe_table(pd.DataFrame(rows),
                    title="Is the gradient es_01's alone? Each source split at its own median",
                    show_index=False)


# ----------------------------------------------------------------------------
# Part B — cross-lag
# ----------------------------------------------------------------------------
def make_lag_pairs(df: pd.DataFrame, denom: str) -> pd.DataFrame:
    """Consecutive within-child wave pairs, as rates over ``denom``."""
    d = df.sort_values(["subject_id", "age"]).copy()
    for col in ("spoken", "signed", "sign_only", "produced"):
        d[f"rate_{col}"] = d[col] / d[denom].replace(0, np.nan)
    grouped = d.groupby("subject_id")
    nxt = grouped.shift(-1)
    pairs = pd.DataFrame({
        "study": d["study"],
        "subject_id": d["subject_id"],
        "age_t": d["age"],
        "gap": nxt["age"] - d["age"],
        "denom_t": d[denom],
        "spoken_t": d["rate_spoken"],
        "signed_t": d["rate_signed"],
        "sign_only_t": d["rate_sign_only"],
        "d_spoken": nxt["rate_spoken"] - d["rate_spoken"],
        "d_signed": nxt["rate_signed"] - d["rate_signed"],
    })
    return pairs.dropna(subset=["d_spoken", "d_signed", "gap"])


def report_cross_lag(cells: pd.DataFrame, nz: pd.DataFrame) -> None:
    heading("B. Cross-lag: does signing at one wave precede speech at the next?")

    longitudinal = cells[cells["study"].isin(["uk_02", "uk_07"])]
    pairs = make_lag_pairs(longitudinal, "understood")
    key_value_table("Available wave pairs (within-understood sources)", [
        ("uk_02 pairs", int((pairs["study"] == "uk_02").sum())),
        ("uk_07 pairs", int((pairs["study"] == "uk_07").sum())),
        ("children contributing", int(pairs["subject_id"].nunique())),
        ("median months between waves", round(float(pairs["gap"].median()), 1)),
    ])
    console.print(
        "Both directions are fitted. Forward asks whether signs the child does "
        "not yet say predict a later gain in speech; reverse asks the mirror. A "
        "result that appears in both is shared growth, not a lead.\n"
    )

    pairs = pairs.assign(study_uk07=(pairs["study"] == "uk_07").astype(float))
    controls = ["spoken_t", "signed_t", "denom_t", "age_t", "gap", "study_uk07"]
    fwd = ols_cluster(pairs, "d_spoken", "sign_only_t",
                      [c for c in controls if c != "signed_t"])
    rev = ols_cluster(pairs, "d_signed", "spoken_t",
                      ["signed_t", "denom_t", "age_t", "gap", "study_uk07"])
    rows = [
        {"direction": "sign-only(t) -> gain in spoken", "rows": fwd["n"],
         "children": fwd["n_children"], "beta (SE)": _fmt(fwd), "z": round(fwd["z"], 2)},
        {"direction": "spoken(t) -> gain in signed", "rows": rev["n"],
         "children": rev["n_children"], "beta (SE)": _fmt(rev), "z": round(rev["z"], 2)},
    ]
    dataframe_table(pd.DataFrame(rows),
                    title="Cross-lag, rates over understood (uk_02 + uk_07)",
                    show_index=False)

    # nz_01: same question, produced-vocabulary denominator. Never pooled above.
    nz_pairs = make_lag_pairs(nz, "produced")
    nz_fwd = ols_cluster(nz_pairs, "d_spoken", "sign_only_t",
                         ["spoken_t", "denom_t", "age_t", "gap"])
    nz_rev = ols_cluster(nz_pairs, "d_signed", "spoken_t",
                         ["signed_t", "denom_t", "age_t", "gap"])
    dataframe_table(pd.DataFrame([
        {"direction": "sign-only(t) -> gain in spoken", "rows": nz_fwd["n"],
         "children": nz_fwd["n_children"], "beta (SE)": _fmt(nz_fwd),
         "z": round(nz_fwd["z"], 2)},
        {"direction": "spoken(t) -> gain in signed", "rows": nz_rev["n"],
         "children": nz_rev["n_children"], "beta (SE)": _fmt(nz_rev),
         "z": round(nz_rev["z"], 2)},
    ]), title="Cross-lag, rates over produced vocabulary (nz_01, different reference set)",
        show_index=False)

    console.print(
        "[yellow]Rates over produced vocabulary are compositional: spoken and "
        "sign-only shares sum with 'both' to 1, so a negative forward slope here "
        "is partly arithmetic. The uk_02/uk_07 panel, whose denominator is "
        "comprehension rather than production, carries the interpretable "
        "estimate.[/yellow]"
    )


def report_within_child(cells: pd.DataFrame, nz: pd.DataFrame) -> None:
    """Does an individual child's signing get absorbed into speech as speech grows?

    The cross-study gradient in A1b confounds development with four studies'
    worth of other differences. Every test here is within child, so the child is
    their own control and nothing that is fixed about them — severity, family,
    study, instrument, whether they were taught to sign — can drive it.
    """
    heading("C. Within child: is signing absorbed as speech grows?")
    lon = pd.concat([cells[cells["study"].isin(["uk_02", "uk_07"])], nz])
    lon = lon.assign(share=lon["sign_only"] / lon["produced"].replace(0, np.nan))
    lon = lon.sort_values("age")
    # Children seen once contribute nothing to a first-vs-last or fixed-effects
    # comparison (their delta is identically zero and they demean to zero), so
    # they are dropped rather than counted: nz_01 has 5 and uk_07 3.
    seen = lon.groupby(["study", "subject_id"])["age"].transform("size")
    lon = lon[seen > 1]

    rows, deltas = [], []
    for study, d in lon.groupby("study"):
        first, last = d.groupby("subject_id").first(), d.groupby("subject_id").last()
        delta = (last["share"] - first["share"]).dropna()
        deltas.append(delta)
        fell, rose = int((delta < 0).sum()), int((delta > 0).sum())
        p = stats.binomtest(fell, fell + rose, 0.5).pvalue if fell + rose else np.nan
        rows.append({
            "source": study, "children": len(delta),
            "share, first wave": f"{first['share'].median():.1%}",
            "share, last wave": f"{last['share'].median():.1%}",
            "spoken, first": round(float(first["spoken"].median())),
            "spoken, last": round(float(last["spoken"].median())),
            "fell": fell, "rose": rose, "tied": len(delta) - fell - rose,
            "sign test p": f"{p:.4f}",
        })
    dataframe_table(pd.DataFrame(rows),
                    title="C1. Sign-only share of production, first wave vs last",
                    show_index=False)
    alld = pd.concat(deltas)
    fell, rose = int((alld < 0).sum()), int((alld > 0).sum())
    key_value_table("C1 pooled", [
        ("children measured more than once", len(alld)),
        ("share fell / rose", f"{fell} / {rose}"),
        ("sign test p", f"{stats.binomtest(fell, fell + rose, 0.5).pvalue:.2e}"),
    ])

    # C2 -- the same question as a slope, in words. Child-demeaning removes every
    # time-invariant confound; SEs stay clustered because waves are not independent.
    rows = []
    for study, d in lon.groupby("study"):
        d = d.dropna(subset=["sign_only", "spoken", "age"])
        dm = d.copy()
        for c in ("sign_only", "spoken", "age"):
            dm[c] = dm[c] - dm.groupby("subject_id")[c].transform("mean")
        fit = sm.OLS(dm["sign_only"].to_numpy(),
                     sm.add_constant(dm[["spoken", "age"]].to_numpy())).fit(
            cov_type="cluster", cov_kwds={"groups": d["subject_id"].to_numpy()})
        beta = float(fit.params[1])
        rows.append({
            "source": study, "children": int(d["subject_id"].nunique()),
            "d(sign-only)/d(spoken)": f"{beta:+.3f}",
            "SE": f"{fit.bse[1]:.3f}", "z": f"{fit.tvalues[1]:+.2f}",
            "implied d(total)/d(spoken)": f"{1 + beta:.2f}",
        })
    dataframe_table(pd.DataFrame(rows),
                    title="C2. Child fixed effects: sign-only vocabulary against the child's own speech",
                    show_index=False)
    console.print(
        "The last column is the identity again: as a child gains a spoken word, "
        "total production gains [bold]0.6-0.8[/bold] words, not one, because some "
        "signs stop being sign-only. Signs are absorbed, and the total still "
        "rises.\n"
    )

    rows = []
    for study, d in lon.groupby("study"):
        j = (d.groupby("subject_id")["sign_only"].first().rename("first")
             .to_frame().join(d.groupby("subject_id")["sign_only"].last().rename("last")))
        rows.append({
            "source": study, "children": len(j),
            "moved INTO fully duplicated": int(((j["first"] > 0) & (j["last"] == 0)).sum()),
            "moved out": int(((j["first"] == 0) & (j["last"] > 0)).sum()),
        })
    dataframe_table(pd.DataFrame(rows),
                    title="C3. Transitions into 'every sign also spoken'", show_index=False)
    console.print(
        "[yellow]Direction, not cause. Signing receding as speech arrives is what "
        "a transitional channel looks like — and also what dropping an "
        "intervention looks like once it is judged no longer needed. Nothing here "
        "separates those.[/yellow]"
    )


def report_power(cells: pd.DataFrame) -> None:
    heading("What this design can and cannot detect")
    longitudinal = cells[cells["study"].isin(["uk_02", "uk_07"])]
    pairs = make_lag_pairs(longitudinal, "understood")
    n_children = pairs["subject_id"].nunique()
    # Cluster-robust SE on a standardised slope is ~ 1/sqrt(n_clusters) before
    # controls; the detectable effect at 80% power, alpha 0.05, is ~2.8 SE.
    se_floor = 1.0 / np.sqrt(max(n_children, 1))
    key_value_table("Cross-lag detectable effect (order of magnitude)", [
        ("children contributing pairs", int(n_children)),
        ("approx SE floor on standardised beta", round(float(se_floor), 3)),
        ("smallest reliably detectable beta (~2.8 SE)", round(float(2.8 * se_floor), 2)),
        ("interpretation", "only a large effect could clear this"),
    ])


def main() -> None:
    cells = within_understood_cells()
    nz = nz01_cells()

    heading("Sign-speech association: between-child and cross-lag")
    key_value_table("Sources", [
        ("within-understood rows", len(cells)),
        ("within-understood children", int(cells["subject_id"].nunique())),
        ("nz_01 rows / children", f"{len(nz)} / {nz['subject_id'].nunique()}"),
        ("longitudinal children (uk_02, uk_07, nz_01)",
         int(pd.concat([cells[cells['study'] != 'es_01'], nz])
             .groupby(['study', 'subject_id']).size().gt(1).sum())),
    ])

    report_additive_share(cells, nz)
    report_between_child(cells)
    report_total_decomposition(cells)
    report_stratified_check(cells)
    report_cross_lag(cells, nz)
    report_within_child(cells, nz)
    report_power(cells)


if __name__ == "__main__":
    main()
