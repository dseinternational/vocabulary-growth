# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the matching variable decides: mental age, comprehension or chronological age.

The Down syndrome language literature reports that children with Down syndrome
matched to typically developing children on **mental age** produce similar or
more vocabulary. This repository reports that matched on **comprehension** they
say a markedly smaller share of the words they understand. Both can hold, and
``es_01`` is the source that settles it: Galeote et al. (2011) supplied 186
children with Down syndrome each matched pairwise on mental age and sex to a
typically developing child, all on the same 651-item CDI-Down, with the
Brunet-Lezine developmental age recorded for every child. One instrument, one
sample, both groups, and the matching variable available rather than inferred.

Four analyses, written to the comparisons output directory:

``matched_design_mental_age.csv``
    The paired comparison at matched mental age -- comprehension, spoken,
    gestured, the union, and the conversion ratio -- with Wilcoxon signed-rank
    tests over the pairs.

``matched_design_decomposition.csv``
    The identity ``log(spoken) = log(understood) + log(ratio)`` differenced
    within pairs. It is exact, so the group difference in production splits
    additively into a comprehension term and a conversion term with no
    residual, and the two can be read against each other.

``matched_design_bands.csv``
    The same children grouped three ways -- by mental-age band (the design as
    supplied), by comprehension band, and by chronological-age band -- so the
    three matching choices are compared on one sample.

``matched_design_regression.csv``
    A binomial regression of the conversion ratio on group with comprehension,
    mental age and chronological age entered in turn, reported with
    **heteroskedasticity-robust (HC0) standard errors**. A binomial likelihood
    on 651 words treats items within a child as independent; the count-scale
    Pearson dispersion here is 46 to 106, so the model-based standard errors
    are roughly ten times too small. That is the same overdispersion the fitted
    models in this repository carry a Beta-Binomial likelihood for. Both
    standard errors are written out so the size of the correction is visible.

    Note for anyone tempted by the shorter route: ``GLM.fit(scale="X2")`` does
    **not** apply this correction to a two-column endog. It returns a scale of
    1.54 where the count-scale statistic is 92.3, inflating the standard error
    by 1.24x instead of 9.6x. The first version of this script used it and
    reported a dispersion column of 0.3-1.5, which is why the sandwich
    estimator is used here instead.

Usage::

    uv run python scripts/compare_matched_designs.py [--source <csv>] [--out <dir>]

``es_01``'s Down syndrome children are all in the Down syndrome analysis pool,
so this is not independent evidence of the conversion shortfall. Its typically
developing children are **not** in the Wordbank-scoped reference pool, which is
what makes the matched comparison itself new. Nothing here reads a fitted
posterior; it is a descriptive analysis of one source's own counts.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from vocab_growth import environment as env
from vocab_growth.comparisons_provenance import write_comparison_manifest

DEFAULT_SOURCE = os.path.join("data", "vocab_data_es_01.csv")
N_ITEMS = 651

#: Comprehension bands for the comprehension-matched view. The lowest band is
#: reported but never interpreted: typically developing children there have a
#: median spoken count of zero, so the ratio comparison is a floor artefact.
COMPREHENSION_BANDS = [0, 50, 100, 200, 300, 450, N_ITEMS + 1]
CHRONOLOGICAL_BANDS = [11, 18, 24, 34]
MIN_BAND_ROWS = 5


def load_pairs(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The matched pairs as two frames aligned on ``pair_id``."""
    d = pd.read_csv(path)
    ds = d[d["group"] == "DS"].set_index("pair_id").sort_index()
    td = d[d["group"] == "TD"].set_index("pair_id").sort_index()
    shared = ds.index.intersection(td.index)
    return ds.loc[shared], td.loc[shared]


def _ratio(frame: pd.DataFrame) -> np.ndarray:
    return (frame["spoken"] / frame["understood"].clip(lower=1)).to_numpy(float)


def mental_age_table(ds: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    """The paired comparison at matched mental age."""
    rows = []
    measures: list[tuple[str, np.ndarray, np.ndarray]] = [
        (name, ds[name].to_numpy(float), td[name].to_numpy(float))
        for name in ("understood", "spoken", "gestured", "spoken_or_gestured")
    ]
    measures.append(("ratio_spoken_over_understood", _ratio(ds), _ratio(td)))
    for name, a, b in measures:
        test = stats.wilcoxon(a, b)
        rows.append(
            {
                "measure": name,
                "n_pairs": int(len(a)),
                "ds_median": round(float(np.median(a)), 3),
                "td_median": round(float(np.median(b)), 3),
                "ds_over_td_medians": round(float(np.median(a) / max(np.median(b), 1e-9)), 3),
                "paired_median_difference": round(float(np.median(a - b)), 3),
                "n_pairs_ds_at_least_td": int((a >= b).sum()),
                "wilcoxon_statistic": round(float(test.statistic), 1),
                "wilcoxon_p": float(f"{test.pvalue:.3g}"),
            }
        )
    return pd.DataFrame(rows)


def decomposition_table(ds: pd.DataFrame, td: pd.DataFrame) -> pd.DataFrame:
    """``log(spoken) = log(understood) + log(ratio)``, differenced within pairs.

    Restricted to pairs where both children have a non-zero count on both
    measures, because the identity is on logs. The dropped pairs are recorded
    on every row so the restriction is visible beside the estimates.
    """
    usable = (
        (ds["spoken"] > 0).to_numpy()
        & (td["spoken"] > 0).to_numpy()
        & (ds["understood"] > 0).to_numpy()
        & (td["understood"] > 0).to_numpy()
    )
    u_ds, u_td = ds["understood"].to_numpy(float)[usable], td["understood"].to_numpy(float)[usable]
    s_ds, s_td = ds["spoken"].to_numpy(float)[usable], td["spoken"].to_numpy(float)[usable]
    terms = {
        "comprehension_advantage": np.log(u_ds) - np.log(u_td),
        "conversion_shortfall": np.log(s_ds / u_ds) - np.log(s_td / u_td),
        "net_production": np.log(s_ds) - np.log(s_td),
    }
    residual = float(
        np.abs(
            terms["comprehension_advantage"] + terms["conversion_shortfall"] - terms["net_production"]
        ).max()
    )
    rows = []
    for name, values in terms.items():
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(
            {
                "term": name,
                "n_pairs": int(usable.sum()),
                "n_pairs_dropped_zero_count": int((~usable).sum()),
                "paired_mean_log": round(float(values.mean()), 4),
                "se": round(se, 4),
                "t": round(float(values.mean() / se), 2),
                "multiplier": round(float(np.exp(values.mean())), 3),
                "identity_max_abs_residual": residual,
            }
        )
    return pd.DataFrame(rows)


def _band_rows(frame: pd.DataFrame, design: str, band_column: str) -> list[dict]:
    rows = []
    for band, group in frame.groupby(band_column, observed=True):
        ds_side = group[group["group"] == "DS"]
        td_side = group[group["group"] == "TD"]
        row = {
            "design": design,
            "band": str(band),
            "n_ds": int(len(ds_side)),
            "n_td": int(len(td_side)),
            "assessed": bool(len(ds_side) >= MIN_BAND_ROWS and len(td_side) >= MIN_BAND_ROWS),
        }
        if row["assessed"]:
            r_ds = float(np.median(_ratio(ds_side)))
            r_td = float(np.median(_ratio(td_side)))
            row.update(
                {
                    "ds_ratio_median": round(r_ds, 3),
                    "td_ratio_median": round(r_td, 3),
                    # Withheld where the typically developing side sits on the
                    # floor: a ratio of zero makes the quotient meaningless
                    # rather than large.
                    "ds_over_td": round(r_ds / r_td, 3) if r_td > 0 else np.nan,
                    "ds_understood_median": round(float(ds_side["understood"].median()), 1),
                    "td_understood_median": round(float(td_side["understood"].median()), 1),
                    "ds_spoken_median": round(float(ds_side["spoken"].median()), 1),
                    "td_spoken_median": round(float(td_side["spoken"].median()), 1),
                    "ds_age_median": round(float(ds_side["age"].median()), 1),
                    "td_age_median": round(float(td_side["age"].median()), 1),
                    "ds_mental_age_median": round(float(ds_side["mental_age"].median()), 1),
                    "td_mental_age_median": round(float(td_side["mental_age"].median()), 1),
                }
            )
        rows.append(row)
    return rows


def band_table(source: pd.DataFrame) -> pd.DataFrame:
    """The same children grouped by each of the three candidate matching variables."""
    frame = source.copy()
    frame["comprehension_band"] = pd.cut(frame["understood"], COMPREHENSION_BANDS, right=False)
    frame["chronological_band"] = pd.cut(frame["age"], CHRONOLOGICAL_BANDS, right=False)
    rows: list[dict] = []
    rows += _band_rows(frame, "mental_age_level", "mental_age_level")
    rows += _band_rows(frame, "comprehension", "comprehension_band")
    rows += _band_rows(frame, "chronological_age", "chronological_band")
    return pd.DataFrame(rows)


def _count_scale_dispersion(fitted, trials: np.ndarray, successes: np.ndarray, n_params: int) -> float:
    """Pearson dispersion on the count scale, which is what overdispersion means here.

    Computed rather than read off the fit because ``GLM.fit(scale="X2")``
    returns a different statistic for a two-column endog -- see this module's
    docstring.
    """
    p = np.clip(np.asarray(fitted.fittedvalues, dtype=float), 1e-12, 1 - 1e-12)
    residual = (successes - trials * p) ** 2 / (trials * p * (1 - p))
    return float(residual.sum() / (len(trials) - n_params))


def regression_table(source: pd.DataFrame) -> pd.DataFrame:
    """Conversion-ratio models with robust standard errors, one control at a time.

    The response is words spoken out of words understood. Standard errors are
    HC0 sandwich estimates; the model-based ones and the count-scale dispersion
    are reported beside them because the correction is a factor of about ten
    and a reader should be able to see that rather than take it on trust.
    """
    frame = source[source["understood"] > 0].copy()
    frame["ds"] = (frame["group"] == "DS").astype(float)
    frame["log_comprehension"] = np.log(frame["understood"] / N_ITEMS)
    frame["mental_age_months"] = frame["mental_age"]
    frame["chronological_age_months"] = frame["age"].astype(float)
    response = np.c_[
        frame["spoken"], (frame["understood"] - frame["spoken"]).clip(lower=0)
    ]
    specifications = [
        ("group only", ["ds"]),
        ("group + comprehension", ["ds", "log_comprehension"]),
        ("group + mental age", ["ds", "mental_age_months"]),
        ("group + comprehension + mental age", ["ds", "log_comprehension", "mental_age_months"]),
        (
            "group + comprehension + mental age + chronological age",
            ["ds", "log_comprehension", "mental_age_months", "chronological_age_months"],
        ),
    ]
    trials = frame["understood"].to_numpy(float)
    successes = frame["spoken"].to_numpy(float)
    rows = []
    for label, columns in specifications:
        design = sm.add_constant(frame[columns].astype(float))
        model = sm.GLM(response, design, family=sm.families.Binomial())
        fitted = model.fit()
        robust = model.fit(cov_type="HC0")
        estimate = float(fitted.params["ds"])
        robust_se = float(robust.bse["ds"])
        row = {
            "specification": label,
            "n_children": int(len(frame)),
            "ds_effect_logit": round(estimate, 4),
            "se_hc0": round(robust_se, 4),
            "z_hc0": round(estimate / robust_se, 2),
            "ds_odds_multiplier": round(float(np.exp(estimate)), 3),
            "se_model_based": round(float(fitted.bse["ds"]), 4),
            "count_scale_dispersion": round(
                _count_scale_dispersion(fitted, trials, successes, len(columns) + 1), 1
            ),
        }
        for column in columns:
            if column != "ds":
                row[f"coef_{column}"] = round(float(fitted.params[column]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="es_01 raw CSV")
    parser.add_argument("--out", default=None, help="output root (see fit_model.py)")
    args = parser.parse_args()

    env.set_output_root(args.out)
    out_dir = env.comparisons_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    source = pd.read_csv(args.source)
    ds, td = load_pairs(args.source)
    print(
        f"[source] {args.source}: {len(source)} rows, {len(ds)} matched pairs; "
        f"mental age within {abs(ds['mental_age'] - td['mental_age']).max():.2f} months on every pair, "
        f"identical band on {int((ds['mental_age_level'] == td['mental_age_level']).sum())} of {len(ds)}"
    )
    print(
        f"[source] chronological age: DS median {ds['age'].median():.0f} months "
        f"({ds['age'].min():.0f}-{ds['age'].max():.0f}), TD median {td['age'].median():.0f} "
        f"({td['age'].min():.0f}-{td['age'].max():.0f})"
    )

    tables = {
        "matched_design_mental_age.csv": mental_age_table(ds, td),
        "matched_design_decomposition.csv": decomposition_table(ds, td),
        "matched_design_bands.csv": band_table(source),
        "matched_design_regression.csv": regression_table(source),
    }
    for name, table in tables.items():
        table.to_csv(os.path.join(out_dir, name), index=False)
        print(f"\n=== {name}")
        print(table.to_string(index=False))
    # No fit contributes -- this is a descriptive analysis of one source's own
    # counts -- so the provenance recorded is the source file itself (#289 task
    # 4.9): a change to es_01 stales these tables the way a refit stales the
    # fit-derived comparisons.
    write_comparison_manifest(
        out_dir,
        script="compare_matched_designs.py",
        contributing={},
        outputs=list(tables),
        source_files={"es_01": args.source},
    )
    print(f"\n[written] {out_dir}")


if __name__ == "__main__":
    main()
