# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Systematic data audit of the Edgin (``us_01``) Down syndrome subset.

Successive patches to this subset — the removed ``production <= 100`` inclusion
rule, the Words & Sentences comprehension proxy, the duplicated outcome columns —
each fixed one defect found by hand. This script instead enumerates every defect
class the aggregate data can expose, so the subset is characterised once rather
than patched repeatedly, and so any later change can be re-checked by re-running
it.

Each check prints its finding and the affected administrations. Nothing is
mutated: the script reads ``data/wordbank_administration_data.csv`` and reports.
Exit status is 0 when the audit completes, whatever it finds — this is a
descriptive tool, not a gate.

Run from anywhere::

    python scripts/audit_edgin_subset.py [--all-checks] [--csv PATH]

``--csv`` writes the per-administration verdict table. The reference benchmarks
are the same-age typically-developing Wordbank children on the same form, and the
independent Berglund et al. (2001) Down syndrome cohort quoted in
``docs/models/PRIORS.md``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WORDBANK = REPO / "data" / "wordbank_administration_data.csv"

FORM_CEILINGS = {"WG": 396, "WS": 680}
FORM_AGE_RANGES = {"WG": (8, 18), "WS": (16, 30)}

# Independent Down syndrome spoken-vocabulary medians by chronological age, from
# Berglund et al. (2001) (330 children, 710-item Swedish CDI) as tabulated in
# docs/models/PRIORS.md. Used only as an order-of-magnitude implausibility
# benchmark, never as a filter threshold.
BERGLUND_SPOKEN_MEDIAN = {12: 0, 24: 10, 36: 30, 48: 50, 60: 65}
BERGLUND_MAX_CHILD = 668  # the single most able child, at 48 months

# Kept in step with vocab_growth.data_utils, so the audit's verdict and the
# implemented masking agree rather than differing by an unscoped check.
COLLAPSE_FACTOR = 5.0
COLLAPSE_MIN_VALUE = 50
COLLAPSE_MAX_AGE_MONTHS = 30


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (Edgin Down syndrome rows, the whole Edgin cohort, reference rows).

    ``us_01`` is not a study file supplied by the source team: it is the
    ``dataset_name == 'Edgin'`` Down syndrome subset of the public Wordbank
    by-child export. The cohort also contains pre-term, autism and
    unlabelled-condition children, who are *not* in ``us_01`` but who share the
    dataset's preparation. Batch-structure checks therefore run over the whole
    cohort: a defect introduced during preparation does not respect the
    condition label we happen to filter on, and looking only at the Down
    syndrome rows truncates any such run at the subset boundary.

    The reference pool excludes Edgin. Two of its rows satisfy the
    typically-developing filter (``typically_developing`` true, no health
    condition), and one of the two is itself at the Words & Sentences ceiling
    inside the flagged run — so leaving them in would benchmark the defect
    partly against itself.
    """
    frame = pd.read_csv(WORDBANK, low_memory=False)
    english = frame[frame["language"] == "English (American)"]
    cohort = english[english["dataset_name"] == "Edgin"].copy()
    conditions = cohort["health_conditions"].astype(str).str.lower()
    edgin = cohort[conditions.str.contains("down", na=False)].copy()
    reference = english[
        english["typically_developing"].eq(True)
        & english["health_conditions"].isna()
        & english["dataset_name"].ne("Edgin")
    ].copy()
    return edgin, cohort, reference


def _td_percentile(reference: pd.DataFrame, form: str, outcome: str, age, value) -> float:
    """Percentile of ``value`` among same-form reference children within a month."""
    if pd.isna(value) or pd.isna(age):
        return np.nan
    band = reference[
        (reference["form"] == form) & reference["age"].between(age - 1, age + 1)
    ][outcome].dropna()
    if len(band) < 50:
        return np.nan
    return float(100 * (band < value).mean())


def _berglund_reference(age) -> float:
    """Interpolated Berglund median spoken count at an age, for the benchmark."""
    if pd.isna(age):
        return np.nan
    ages = np.array(sorted(BERGLUND_SPOKEN_MEDIAN))
    return float(np.interp(age, ages, [BERGLUND_SPOKEN_MEDIAN[a] for a in ages]))


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def describe(edgin: pd.DataFrame, cohort: pd.DataFrame, reference: pd.DataFrame) -> None:
    section("0. Composition")
    print(f"administrations: {len(edgin)}   children: {edgin['child_id'].nunique()}")
    per_child = edgin.groupby("child_id").size().value_counts().sort_index()
    print("administrations per child: "
          + ", ".join(f"{k}x={v}" for k, v in per_child.items()))
    for form, group in edgin.groupby("form"):
        print(f"  {form}: {len(group):>3} administrations, "
              f"{group['child_id'].nunique():>3} children, "
              f"ages {group['age'].min():.0f}-{group['age'].max():.0f}, "
              f"comprehension present {group['comprehension'].notna().sum():>3}, "
              f"production present {group['production'].notna().sum():>3}")
    conditions = (
        cohort["health_conditions"].fillna("(unlabelled)").replace("", "(unlabelled)")
    )
    print(f"\nwhole Edgin cohort: {len(cohort)} administrations, "
          f"{cohort['child_id'].nunique()} children")
    print("  condition groups (children): "
          + ", ".join(f"{k}={v}" for k, v in
                      cohort.groupby(conditions)["child_id"].nunique().items()))
    print("  us_01 is the Down syndrome group only; the rest share the dataset's")
    print("  preparation and are used below as internal controls.")
    print(f"\nreference (typically developing, American English, Edgin excluded): "
          f"{len(reference)} rows")


def check_impossible_values(edgin: pd.DataFrame) -> pd.Series:
    section("1. Values impossible for their form")
    flags = pd.Series(False, index=edgin.index)
    for outcome in ("comprehension", "production"):
        for form, ceiling in FORM_CEILINGS.items():
            bad = (
                (edgin["form"] == form)
                & edgin[outcome].notna()
                & ((edgin[outcome] > ceiling) | (edgin[outcome] < 0))
            )
            flags |= bad
            print(f"  {form} {outcome} outside [0, {ceiling}]: {int(bad.sum())}")
    non_integer = pd.Series(False, index=edgin.index)
    for outcome in ("comprehension", "production"):
        present = edgin[outcome].notna()
        non_integer |= present & (edgin[outcome] % 1 != 0)
    print(f"  non-integer counts: {int(non_integer.sum())}")
    return flags | non_integer


def check_duplicate_administrations(edgin: pd.DataFrame) -> pd.Series:
    section("2. Duplicate administrations (same child, age and form)")
    key = ["child_id", "age", "form"]
    dup = edgin.duplicated(subset=key, keep=False)
    print(f"  duplicated (child, age, form) rows: {int(dup.sum())}")
    if dup.any():
        print(edgin.loc[dup, key + ["comprehension", "production"]].to_string(index=False))
    return dup


def check_form_age_mismatch(edgin: pd.DataFrame) -> pd.Series:
    section("3. Form administered outside its intended age range")
    flags = pd.Series(False, index=edgin.index)
    for form, (lo, hi) in FORM_AGE_RANGES.items():
        bad = (edgin["form"] == form) & ~edgin["age"].between(lo, hi)
        flags |= bad
        print(f"  {form} outside {lo}-{hi} months: {int(bad.sum())}")
    return flags


def check_outcome_duplication(edgin: pd.DataFrame) -> pd.Series:
    section("4. Outcome-column duplication (production tracking comprehension)")
    print("  Only assessable on WG, which has an independent comprehension column;")
    print("  on WS, comprehension == production is Wordbank's documented convention.")
    wg = edgin["form"] == "WG"
    ratio = edgin["production"] / edgin["comprehension"]
    assessable = wg & edgin["comprehension"].ge(100) & ratio.notna()
    ordered = ratio[assessable].sort_values(ascending=False)
    print(f"\n  WG administrations with comprehension >= 100: {int(assessable.sum())}")
    print("  ratio, descending: " + ", ".join(f"{v:.2f}" for v in ordered))
    if len(ordered) > 2:
        gaps = ordered.values[:-1] - ordered.values[1:]
        cut = int(np.argmax(gaps))
        print(f"  largest gap: {ordered.values[cut]:.3f} -> {ordered.values[cut + 1]:.3f} "
              f"(width {gaps[cut]:.3f}), so a cut in that interval separates the cluster")
    flags = assessable & ratio.ge(0.75)
    print(f"  flagged at ratio >= 0.75: {int(flags.sum())}")
    return flags


def check_form_saturation(edgin: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    section("5. Saturation at the form maximum")
    flags = pd.Series(False, index=edgin.index)
    for form, ceiling in FORM_CEILINGS.items():
        group = edgin[edgin["form"] == form]
        at_ceiling = group["production"].eq(ceiling)
        print(f"  {form} production at {ceiling}: {int(at_ceiling.sum())} of {len(group)}")
        if at_ceiling.any():
            band = group.loc[at_ceiling, "age"]
            print(f"    ages {band.min():.0f}-{band.max():.0f}")
            for lo, hi in ((16, 19), (20, 24), (25, 30)):
                ref = reference[
                    (reference["form"] == form) & reference["age"].between(lo, hi)
                ]["production"].dropna()
                if len(ref) < 50:
                    continue
                rate = 100 * ref.eq(ceiling).mean()
                print(f"    reference {lo}-{hi} mo at ceiling: {rate:.2f}% "
                      f"(max {ref.max():.0f}, n={len(ref)})")
        flags |= edgin.index.isin(group.index[at_ceiling])
    return flags


def _run_probability(n: int, m: int, k: int) -> float:
    """Exact P(some run of >= k flags) when m flags are placed uniformly in n slots.

    The m flags are exchangeable, so every one of ``C(n, m)`` arrangements is
    equally likely. Laying out the ``n - m`` unflagged slots creates
    ``n - m + 1`` gaps, and an arrangement avoids a k-run exactly when every gap
    holds at most ``k - 1`` flags — so the count is the coefficient of ``x^m`` in
    ``(1 + x + ... + x^(k-1)) ** (n - m + 1)``. The complement is taken over the
    exact integer counts before dividing: the avoiding arrangements outnumber the
    rest so heavily here that subtracting in floating point underflows to zero.
    """
    if m == 0 or k > m:
        return 0.0
    gaps = n - m + 1
    poly = [1]
    for _ in range(gaps):
        nxt = [0] * (len(poly) + k - 1)
        for i, c in enumerate(poly):
            if c:
                for j in range(k):
                    nxt[i + j] += c
        poly = nxt[: m + 1]
    # The truncation above can leave the polynomial shorter than m + 1, which
    # means no arrangement avoids a k-run at all (k == 1 is the degenerate case).
    avoid = poly[m] if m < len(poly) else 0
    total = math.comb(n, m)
    return (total - avoid) / total


def check_batch_structure(cohort: pd.DataFrame, edgin: pd.DataFrame) -> pd.Series:
    section("6. Preparation-batch structure (runs at the ceiling, in child_id order)")
    print("  Run over the WHOLE Edgin cohort, not just its Down syndrome subset:")
    print("  a preparation defect does not respect the condition label, and the")
    print("  us_01 filter truncates any run at the subset boundary.")
    flags = pd.Series(False, index=edgin.index)
    for form, ceiling in FORM_CEILINGS.items():
        group = cohort[cohort["form"] == form].sort_values("child_id")
        if group.empty:
            continue
        at_ceiling = group["production"].eq(ceiling)
        print(f"\n  {form}: {int(at_ceiling.sum())} of {len(group)} records at exactly {ceiling}")
        if not at_ceiling.any():
            continue
        runs = (at_ceiling != at_ceiling.shift()).cumsum()
        longest = 0
        for _, block in group.groupby(runs):
            if not at_ceiling.loc[block.index].iloc[0] or len(block) < 3:
                continue
            longest = max(longest, len(block))
            ids = block["child_id"].tolist()
            span = ids[-1] - ids[0] + 1
            groups = sorted(
                block["health_conditions"].fillna("(unlabelled)").replace("", "(unlabelled)").unique()
            )
            others = cohort[
                cohort["child_id"].isin(ids) & (cohort["form"] != form)
            ]
            print(f"    run of {len(block)}: ids {ids[0]}-{ids[-1]}, "
                  f"ages {block['age'].min():.0f}-{block['age'].max():.0f}")
            print(f"      consecutive in id ORDER (adjacent present ids), not consecutive "
                  f"integers: {len(ids)} ids across a span of {span}")
            print(f"      condition groups spanned: {groups}")
            print(f"      other administrations for these children: {len(others)}")
            flags |= edgin.index.isin(block.index)
        if longest >= 3:
            n, m = len(group), int(at_ceiling.sum())
            p = _run_probability(n, m, longest)
            print(f"    P(a run this long | {m} ceiling records placed at random "
                  f"in {n}) = {p:.2e}")
        else:
            print("    no run of three or more consecutive records at the ceiling")
    return flags


def check_longitudinal_collapse(
    edgin: pd.DataFrame,
    factor: float = COLLAPSE_FACTOR,
    floor: float = COLLAPSE_MIN_VALUE,
    max_age: float = COLLAPSE_MAX_AGE_MONTHS,
) -> pd.Series:
    section(f"7. Longitudinal collapse (a value exceeding a LATER record by >= {factor:g}x)")
    print("  Vocabulary does not shrink with age, so a hit is unambiguous. Two")
    print(f"  calibrations are applied, matching the implemented rule: a floor of {floor:g}")
    print("  (without it, 5 words falling to 1 would fire) and an age scope of")
    print(f"  {max_age:g} months (at older ages a decline can follow a form change).")
    flags = pd.Series(False, index=edgin.index)
    rows = []
    for child, group in edgin.sort_values("age").groupby("child_id"):
        if len(group) < 2:
            continue
        for outcome in ("comprehension", "production"):
            values = group[outcome]
            for i, (idx, value) in enumerate(values.items()):
                if pd.isna(value) or value <= 0:
                    continue
                later = values.iloc[i + 1:].dropna()
                if later.empty or later.min() * factor > value:
                    continue
                scoped = value >= floor and group.loc[idx, "age"] <= max_age
                flags[idx] = flags[idx] or scoped
                rows.append(dict(child_id=child, outcome=outcome,
                                 age=group.loc[idx, "age"], value=value,
                                 later_min=later.min(),
                                 within_rule=scoped))
    table = pd.DataFrame(rows)
    print(f"\n  raw collapses detected: {len(table)}")
    print(f"  within the rule's floor and age scope: {int(flags.sum())}")
    if len(table):
        print(table.sort_values(["outcome", "age"]).to_string(index=False))
    return flags


def check_external_implausibility(edgin: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    section("8. Implausibility against external benchmarks")
    work = edgin.copy()
    work["berglund_median"] = work["age"].map(_berglund_reference)
    work["x_berglund"] = (work["production"] / work["berglund_median"].replace(0, np.nan)).round(1)
    work["td_pct_production"] = [
        _td_percentile(reference, r.form, "production", r.age, r.production)
        for r in work.itertuples()
    ]
    extreme = work[work["production"].gt(BERGLUND_MAX_CHILD * 0.9)]
    print("  administrations claiming more spoken words than 90% of Berglund's most")
    print(f"  able child ever reached ({BERGLUND_MAX_CHILD} words, at 48 months): {len(extreme)}")
    if len(extreme):
        print(f"    their ages: {sorted(extreme['age'].unique().astype(int))}")
    above99 = work[work["td_pct_production"].ge(99)]
    print("  administrations above the 99th typically-developing percentile for")
    print(f"  production at their own age: {len(above99)}")
    return work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None,
                        help="write the per-administration verdict table here")
    args = parser.parse_args(argv)

    if not WORDBANK.exists():
        print(f"[error] {WORDBANK} not present.")
        return 1
    edgin, cohort, reference = _load()

    describe(edgin, cohort, reference)
    verdicts = {
        "impossible_value": check_impossible_values(edgin),
        "duplicate_administration": check_duplicate_administrations(edgin),
        "form_age_mismatch": check_form_age_mismatch(edgin),
        "outcome_duplication": check_outcome_duplication(edgin),
        "form_saturation": check_form_saturation(edgin, reference),
        "id_batch": check_batch_structure(cohort, edgin),
        "longitudinal_collapse": check_longitudinal_collapse(edgin),
    }
    table = check_external_implausibility(edgin, reference)
    for name, flags in verdicts.items():
        table[name] = flags.reindex(table.index).fillna(False)
    table["n_flags"] = table[list(verdicts)].sum(axis=1)

    section("9. Summary")
    for name, flags in verdicts.items():
        print(f"  {name:26} {int(flags.sum()):>4}")
    flagged = table[table["n_flags"] > 0]
    print(f"\n  administrations with at least one flag: {len(flagged)} of {len(table)} "
          f"({100 * len(flagged) / len(table):.0f}%)")
    print(f"  children affected: {flagged['child_id'].nunique()} of {edgin['child_id'].nunique()}")
    print("\n  flagged administrations:")
    columns = ["child_id", "age", "form", "comprehension", "production",
               "x_berglund", "td_pct_production", "n_flags"]
    print(flagged[columns + list(verdicts)].sort_values(["form", "age"]).to_string(index=False))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"\n  verdict table written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
