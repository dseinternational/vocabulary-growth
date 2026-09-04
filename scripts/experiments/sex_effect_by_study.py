#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Female-minus-male difference in CDI vocabulary counts, on the logit scale.

Descriptive, not a model fit. For each source with a recorded sex, an
empirical-logit OLS of the count against a cubic in age and a female indicator
(plus study dummies for the pooled Down syndrome rows), with HC0 robust standard
errors. Three outcomes: words understood, words spoken, and the production ratio
(spoken against the child's own understood count). Intervals are 89% (z = 1.598)
to match the report's interval convention.

Sources: the merged Down syndrome analysis view (``data/vocab_data_merged.csv``,
before the loader's masking rules but restricted to the models' age domain, which
ends at 115 months), the Wordbank English (American) typically developing export
as a reference, and the ``es_01`` typically developing matches, which sit on the
same 651-item CDI-Down as their Down syndrome pairs.

Writes ``<output-root>/comparisons/sex-effect/sex_effect_by_study.csv``, a
forest plot beside it, and ``sex_effect_age_interaction.csv`` -- the
age-by-female slope that asks whether the effect changes with age on the logit
scale. Cited by
``notes/202609041206-sex-differences-in-vocabulary.md``.

Run: ``python scripts/experiments/sex_effect_by_study.py [--output-dir DIR]``
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from vocab_growth import data_utils as du
from vocab_growth import environment as env

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Z89 = 1.598
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MIN_ROWS = 20
MAX_AGE_MONTHS = 115  # the Down syndrome GP domain's upper bound; VG20's own frame runs to 115
STUDIES = ["es_01", "uk_01", "uk_02", "uk_05", "uk_07", "us_01", "ie_02"]
OUTCOMES = [
    ("understood", "Words understood"),
    ("spoken", "Words spoken"),
    ("q", "Production ratio (spoken given understood)"),
]
POOLED = "DS pooled, study-adjusted"
WORDBANK = "TD Wordbank, English (US)"
ES_TD = "TD es_01 matches, same form"


def emp_logit(y, n):
    return np.log((y + 0.5) / (n - y + 0.5))


def ols_hc0(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    XtXi = np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(XtXi @ (X.T * (r**2)) @ X @ XtXi))


def female_effect(d, y, group_col=None):
    """Female coefficient on the logit scale after a cubic in age (and groups)."""
    a = (d["age"] - 36) / 12
    cols = [np.ones(len(d)), a, a**2, a**3, (d["sex"] == "F").astype(float).values]
    if group_col is not None:
        for lev in sorted(d[group_col].unique())[1:]:
            cols.append((d[group_col] == lev).astype(float).values)
    beta, se = ols_hc0(np.column_stack(cols), y)
    return dict(
        est=beta[4],
        se=se[4],
        n=len(d),
        n_f=int((d["sex"] == "F").sum()),
        n_m=int((d["sex"] == "M").sum()),
    )


def age_by_female(d, y, group_col=None, centre=36.0):
    """Female effect at ``centre`` months and its slope per year of age."""
    a = (d["age"] - centre) / 12
    f = (d["sex"] == "F").astype(float).values
    cols = [np.ones(len(d)), a, a**2, a**3, f, a * f]
    if group_col is not None:
        for lev in sorted(d[group_col].unique())[1:]:
            cols.append((d[group_col] == lev).astype(float).values)
    beta, se = ols_hc0(np.column_stack(cols), y)
    return dict(
        centre_months=centre,
        female_at_centre=beta[4],
        se_female=se[4],
        age_by_female_per_year=beta[5],
        se_slope=se[5],
        n=len(d),
        age_min=float(d["age"].min()),
        age_max=float(d["age"].max()),
    )


def ds_frame(ds, outcome):
    if outcome == "q":
        d = ds[ds["understood"].notna() & ds["spoken"].notna() & (ds["understood"] > 0)]
        d = d[d["spoken"] <= d["understood"]]
        return d, emp_logit(d["spoken"].values, d["understood"].values)
    d = ds.dropna(subset=[outcome, "survey_vocab_max"])
    d = d[d[outcome] <= d["survey_vocab_max"]]
    return d, emp_logit(d[outcome].values, d["survey_vocab_max"].values)


def wordbank_frame(w, outcome):
    if outcome == "understood":
        d = w[(w["form"] == "WG") & w["comprehension"].notna()]
        return d, emp_logit(d["comprehension"].values, 396), None
    if outcome == "spoken":
        d = w[w["production"].notna()]
        return d, emp_logit(d["production"].values, d["nmax"].values), "form"
    d = w[
        (w["form"] == "WG")
        & w["comprehension"].notna()
        & w["production"].notna()
        & (w["comprehension"] > 0)
    ]
    d = d[d["production"] <= d["comprehension"]]
    return d, emp_logit(d["production"].values, d["comprehension"].values), None


def es_td_frame(es, outcome):
    if outcome == "q":
        d = es[(es["understood"] > 0) & (es["spoken"] <= es["understood"])]
        return d, emp_logit(d["spoken"].values, d["understood"].values)
    return es, emp_logit(es[outcome].values, 651)


def load():
    data_dir = du.local_env.DATA_DIR
    m = pd.read_csv(os.path.join(data_dir, "vocab_data_merged.csv"), low_memory=False)
    ds = m[m["sex"].isin(["M", "F"]) & m["age"].notna() & (m["age"] <= MAX_AGE_MONTHS)].copy()

    w = pd.read_csv(os.path.join(data_dir, "wordbank_administration_data.csv"), low_memory=False)
    td = w["typically_developing"].astype(str).str.lower().isin(["true", "1"])
    w = w[(w["language"] == "English (American)") & td].copy()
    w["sex"] = w["sex"].astype(str).str.upper().str[0]
    w = w[w["sex"].isin(["M", "F"]) & w["form"].isin(["WG", "WS"])]
    w["nmax"] = np.where(w["form"] == "WG", 396, 680)

    es = pd.read_csv(os.path.join(data_dir, "vocab_data_es_01.csv"))
    es["sex"] = es["sex"].map({1: "M", 2: "F"})
    return ds, w, es[es["group"] == "TD"].copy()


def estimate(ds, w, es_td):
    rows = []
    for outcome, _ in OUTCOMES:
        d_all, y_all = ds_frame(ds, outcome)
        for study in STUDIES:
            mask = (d_all["study"] == study).values
            g = d_all[mask]
            if len(g) >= MIN_ROWS and g["sex"].nunique() == 2:
                rows.append(dict(row=study, kind="ds", outcome=outcome, **female_effect(g, y_all[mask])))
        rows.append(dict(row=POOLED, kind="ds_pooled", outcome=outcome, **female_effect(d_all, y_all, "study")))
        d, y, group = wordbank_frame(w, outcome)
        rows.append(dict(row=WORDBANK, kind="td", outcome=outcome, **female_effect(d, y, group)))
        d, y = es_td_frame(es_td, outcome)
        rows.append(dict(row=ES_TD, kind="td", outcome=outcome, **female_effect(d, y)))
    out = pd.DataFrame(rows)
    out["lo89"] = out["est"] - Z89 * out["se"]
    out["hi89"] = out["est"] + Z89 * out["se"]
    return out


def interaction(ds, w):
    """Does the female effect change with age on the logit scale?"""
    rows = []
    for outcome, _ in OUTCOMES:
        d, y = ds_frame(ds, outcome)
        rows.append(dict(row=POOLED, outcome=outcome, **age_by_female(d, y, "study")))
    for form, col, nmax in (("WG", "comprehension", 396), ("WG", "production", 396), ("WS", "production", 680)):
        d = w[(w["form"] == form) & w[col].notna()]
        y = emp_logit(d[col].values, nmax)
        rows.append(dict(row=f"{WORDBANK} {form}", outcome=col, **age_by_female(d, y, centre=float(d["age"].mean()))))
    return pd.DataFrame(rows)


def plot(table, path):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": INK2,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK,
        }
    )
    order = STUDIES + [POOLED, "", WORDBANK, ES_TD]
    ypos = {lab: i for i, lab in enumerate(reversed(order))}
    xmax = 2.45
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.4), sharey=True, gridspec_kw={"wspace": 0.10})
    for ax, (outcome, title) in zip(axes, OUTCOMES, strict=True):
        sub = table[table["outcome"] == outcome].set_index("row")
        pooled = sub.loc[POOLED]
        ax.axvspan(pooled["lo89"], pooled["hi89"], color=BLUE, alpha=0.08, lw=0, zorder=0)
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        for lab in order:
            if lab not in sub.index:
                continue
            r = sub.loc[lab]
            y = ypos[lab]
            colour = BLUE if r["kind"].startswith("ds") else ORANGE
            big = r["kind"] in ("ds_pooled", "td")
            ax.plot([r["lo89"], r["hi89"]], [y, y], color=colour, lw=2.2 if big else 1.6, solid_capstyle="butt", zorder=2)
            ax.plot(r["est"], y, marker="D" if big else "o", ms=9 if big else 7, color=colour, mec="white", mew=1.2, zorder=3)
            if big:
                ax.annotate(f"{r['est']:+.2f}", (r["est"], y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
            ax.text(xmax - 0.05, y, f"{r['n_f']:,} / {r['n_m']:,}", ha="right", va="center", fontsize=7.5, color=INK2, zorder=3)
        ax.text(xmax - 0.05, len(order) - 0.45, "F / M rows", ha="right", va="center", fontsize=7.5, color=INK2, style="italic")
        ax.set_title(title, fontsize=11, loc="left", pad=8)
        ax.set_xlim(-1.6, xmax)
        ax.set_xticks([-1, 0, 1, 2])
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[lab] for lab in order])
    axes[0].set_yticklabels(order, fontsize=9.5)
    for tick, lab in zip(axes[0].get_yticklabels(), order, strict=True):
        if lab in (POOLED, WORDBANK, ES_TD):
            tick.set_fontweight("bold")
    axes[0].set_ylim(-0.7, len(order) - 0.2)
    fig.text(0.5, 0.035, "Female − male difference on the logit scale, with 89% interval", ha="center", fontsize=10, color=INK2)
    fig.suptitle(
        "Sex difference in CDI vocabulary counts: girls relative to boys at the same age",
        x=0.01, y=0.985, ha="left", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.01, 0.925,
        "Blue: Down syndrome studies and their pooled estimate; the shaded band is the pooled 89% interval, repeated in every panel for reference.",
        fontsize=8.5, color=INK2,
    )
    fig.text(
        0.01, 0.895,
        "Orange: typically developing reference pools. Empirical-logit regression on a cubic in age with robust SEs; descriptive, not a model fit. Studies with under 20 sexed rows omitted.",
        fontsize=8.5, color=INK2,
    )
    fig.subplots_adjust(left=0.20, right=0.99, top=0.83, bottom=0.10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=None, help="Output root (defaults to the repository's resolved output root).")
    args = parser.parse_args()
    if args.output_dir:
        env.set_output_root(args.output_dir)
    out_dir = os.path.join(env.output_root(), "comparisons", "sex-effect")
    os.makedirs(out_dir, exist_ok=True)

    ds, w, es_td = load()
    table = estimate(ds, w, es_td)
    csv_path = os.path.join(out_dir, "sex_effect_by_study.csv")
    png_path = os.path.join(out_dir, "sex_effect_by_study.png")
    table.to_csv(csv_path, index=False)
    plot(table, png_path)
    slopes = interaction(ds, w)
    slopes_path = os.path.join(out_dir, "sex_effect_age_interaction.csv")
    slopes.to_csv(slopes_path, index=False)

    header = f"{'row':30s} {'outcome':10s} {'female effect (89%)':>24s} {'n':>6s} {'F':>6s} {'M':>6s}"
    print(header)
    for _, r in table.iterrows():
        interval = f"{r['est']:+.2f} [{r['lo89']:+.2f}, {r['hi89']:+.2f}]"
        print(f"{r['row']:30s} {r['outcome']:10s} {interval:>24s} {r['n']:6d} {r['n_f']:6d} {r['n_m']:6d}")
    print()
    print("age-by-female slope on the logit scale (does the effect change with age?)")
    for _, r in slopes.iterrows():
        print(
            f"{r['row']:30s} {r['outcome']:13s} ages {r['age_min']:3.0f}-{r['age_max']:3.0f}m"
            f"  female at {r['centre_months']:4.1f}m {r['female_at_centre']:+.3f} ({r['se_female']:.3f})"
            f"  age x female {r['age_by_female_per_year']:+.3f}/yr ({r['se_slope']:.3f})  n={r['n']}"
        )
    print()
    for written in (csv_path, png_path, slopes_path):
        print(f"wrote {written}")


if __name__ == "__main__":
    main()
