# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does the 200-row study threshold cost, and how thin is the age support?

The question, asked 2026-09-06 for [#240](https://github.com/dseinternational/vocabulary-growth/issues/240)'s
study-threshold item. The VG11-VG13 review found that ``min_study_observations
= 200`` removes a third of the study units while removing under 2% of the rows,
and that age, study, language and instrument overlap unevenly -- naming VG12's
19-24 months as carried by two studies and 25 by one. Both halves were stated
from a spot check; neither had been counted across all three models on the
current loader rules.

Method. For each of VG11, VG12 and VG13, rebuild the frame the model sees by
calling ``load_data`` with that definition's own population, language scope, age
bound and ceiling-exclusion settings -- the same arguments its engine's
``prepare_*`` uses -- then apply ``filter_studies_by_min_obs`` at the
definition's own threshold. Report what the threshold removed, each retained
study's age span, and the number of *distinct studies* contributing at each
rounded integer age.

The retained row counts are the check that the frames are the models' own: they
must equal the 18,500 / 7,049 / 6,356 that
``tests/test_kappa_conditional_calibration.py::test_frames_use_the_registered_language_scope``
pins, and the script asserts it rather than leaving it to the reader.

Result: recorded in ``notes/202609062350-td-common-age-support.md``. VG11 never
falls below four studies at any reported age and VG13 never below three, so the
uneven-overlap concern does not bite on either; VG12 falls to two from 19 months
and to one (Floccia) at 25, which is the top of its reported range.

Reads the prepared DuckDB only -- no trace, no fit, nothing written.
"""

from __future__ import annotations

import pandas as pd

import vocab_growth.data_utils as du
from vocab_growth.models.definitions import VG11, VG12, VG13

# The frame each model's engine builds, pinned by the calibration test.
EXPECTED_ROWS = {"VG11": 18_500, "VG12": 7_049, "VG13": 6_356}


def _frame(definition):
    """The eligible frame and the frame after the study threshold."""
    if definition is VG13:
        columns = ["age", "understood", "spoken", "study", "subject_id"]
        load_columns = list(columns)
        if definition.exclude_us01_spoken_ceiling:
            load_columns.append("survey_vocab_max")
        df = du.load_data(
            population=definition.population,
            columns=load_columns,
            sample_fraction=definition.sample_fraction,
            random_seed=definition.random_seed,
            languages=definition.td_languages,
            max_age_months=definition.max_age_months,
            include_implausible_production=definition.include_implausible_production,
        )
        if definition.exclude_us01_spoken_ceiling:
            df, _ = du.exclude_us01_spoken_ceiling_rows(df)
        df = df[columns].copy().dropna(subset=["age"])
        df = df[df["understood"].notna() | df["spoken"].notna()].reset_index(drop=True)
    else:
        y = definition.outcome.value
        columns = ["age", y, "study", "subject_id"]
        df = du.load_data(
            population=definition.population,
            columns=columns,
            sample_fraction=definition.sample_fraction,
            random_seed=definition.random_seed,
            languages=definition.td_languages,
        )
        df = df[columns].dropna(subset=["age", y]).reset_index(drop=True)
    kept, _ = du.filter_studies_by_min_obs(df, definition.min_study_observations)
    return df, kept


def _studies_per_age(kept):
    ages = kept["age"].round().astype(int)
    return (
        pd.DataFrame({"age": ages, "study": kept["study"].to_numpy()})
        .groupby("age")
        .agg(rows=("study", "size"), studies=("study", "nunique"))
    )


def main() -> None:
    for name, definition in (("VG11", VG11), ("VG12", VG12), ("VG13", VG13)):
        eligible, kept = _frame(definition)
        assert len(kept) == EXPECTED_ROWS[name], (
            f"{name}: rebuilt {len(kept):,} rows, expected {EXPECTED_ROWS[name]:,} -- "
            "the frame is not the model's own"
        )
        dropped = sorted(set(eligible["study"]) - set(kept["study"]))
        sizes = eligible[eligible["study"].isin(dropped)]["study"].value_counts()

        print("=" * 78)
        print(
            f"{name}  threshold={definition.min_study_observations}  "
            f"eligible {eligible['study'].nunique()} studies / {len(eligible):,} rows"
            f"  ->  retained {kept['study'].nunique()} / {len(kept):,}"
        )
        if dropped:
            print(
                f"  dropped {len(dropped)} studies, {int(sizes.sum()):,} rows "
                f"({100 * sizes.sum() / len(eligible):.1f}%): "
                + ", ".join(f"{s} {int(sizes[s])}" for s in dropped)
            )

        span = kept.groupby("study")["age"].agg(n="size", lo="min", hi="max")
        print("  retained study age coverage:")
        for study, row in span.sort_values("lo").iterrows():
            print(
                f"    {study:28s} n={int(row['n']):6,}  "
                f"ages {row['lo']:.0f}-{row['hi']:.0f}"
            )

        per_age = _studies_per_age(kept)
        print(
            f"  studies per age: min {per_age['studies'].min()}, "
            f"median {per_age['studies'].median():.0f}, over "
            f"{len(per_age)} ages"
        )
        print(
            "    "
            + " ".join(f"{age}:{n}" for age, n in per_age["studies"].items())
        )
        thin = per_age[per_age["studies"] <= 2]
        for age, row in thin.iterrows():
            names = sorted(set(kept.loc[kept["age"].round().astype(int) == age, "study"]))
            print(
                f"    thin: {age:3d} mo  {int(row['rows']):5,} rows  "
                f"{row['studies']} study/studies: {', '.join(names)}"
            )


if __name__ == "__main__":
    main()
