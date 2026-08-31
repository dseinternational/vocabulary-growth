# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the descriptive-data report artefacts (issue #111).

Writes a per-study summary table, study-coloured scatter plots and
repeated-measures trajectory plots (individual children linked within form,
against the pooled median/IQR) for the pooled Down syndrome dataset — plus the
equivalent trajectory plots for the typically-developing Wordbank reference
pool — into ``docs/descriptive/figures/``, the standalone descriptive report
(``docs/descriptive/index.qmd``, a sibling of the model and comparison
reports). The same artefacts are mirrored into the main report's figure cache
(``docs/report/figures/descriptives/``) so its "Data and measures" chapter
keeps rendering. Run after ``prepare_data.py``:

    python scripts/generate_descriptive_report.py
"""

import os
import shutil

import dse_research_utils.environment.setup as setup
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
from dse_research_utils.plot.styles import categorical_palette
from matplotlib.colors import to_rgb

import vocab_growth.environment as local_env
from vocab_growth.data_utils import load_combined_data, load_data
from vocab_growth.descriptive import (
    plot_monthly_violins,
    plot_observations_by_group,
    plot_repeat_measures_by_group,
    scatter_by_group,
    summarise_by_group,
    summary_table_by_group,
)
from vocab_growth.models.definitions import Population

SCATTERS = [
    # (x, y, xlabel, ylabel, filename)
    ("age", "understood", "Age (months)", "Words understood", "scatter_age_understood_ds"),
    ("age", "spoken", "Age (months)", "Words spoken", "scatter_age_spoken_ds"),
    ("understood", "spoken", "Words understood", "Words spoken", "scatter_understood_spoken_ds"),
    ("age", "signed", "Age (months)", "Words signed", "scatter_age_signed_ds"),
]

REPEAT_OUTCOMES = [
    # (outcome, ylabel)
    ("understood", "Words understood (raw count)"),
    ("spoken", "Words spoken (raw count)"),
]

# The Down syndrome trajectory figures stop at 72 months — the reporting
# window; the sparse older visits are excluded rather than clipped. The
# typically-developing pool is bounded at 30 months by its own admission
# window, and its density supports finer bins.
DS_REPEAT_AGE_RANGE = (8, 72)
TD_REPEAT_AGE_RANGE = (8, 30)
TD_REPEAT_BIN_WIDTH = 3


def _shared_group_colours(groups):
    """One group -> colour mapping shared by every figure of a population.

    Built over ALL groups the population's frames contain, so a study keeps
    one colour across the observation scatters and the repeated-measures
    trajectories alike (each figure's legend lists only the groups it draws).
    Reddish palette entries are skipped: the pooled-summary overlay every
    figure draws is pure red, and a red-toned study would read as part of it.
    """
    groups = sorted(set(groups))
    colours: list = []
    extra = 0
    while len(colours) < len(groups):
        candidates = categorical_palette(len(groups) + extra)
        colours = [c for c in candidates if not _is_reddish(c)]
        extra += 2
    mapping = dict(zip(groups, colours[: len(groups)], strict=True))
    # teal, used by no palette entry, was chosen for uk_01 when its default
    # colour collided with the overlay; keep that choice stable.
    if "uk_01" in mapping:
        mapping["uk_01"] = "teal"
    return mapping


def _is_reddish(colour) -> bool:
    r, g, b = to_rgb(colour)
    return r > 0.55 and g < 0.45 and b < 0.45


def main():
    setup.init_script()
    # The 12pt base font the figures render at comes from the shared style
    # (dse-research-utils FONT_SIZE_DEFAULT, 10 -> 12 in v0.12.1), applied here
    # explicitly so the generator does not depend on init_script's defaults.
    plot_styles.set_matplotlib_default_style()
    # Primary home: the standalone descriptive report (docs/descriptive/).
    out_dir = os.path.join(local_env.DOCS_DIR, "descriptive", "figures")
    # Mirror: the main report's figure cache, so methods-data.qmd keeps rendering.
    report_dir = os.path.join(local_env.REPORT_FIGS_DIR, "descriptives")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Pooled Down syndrome data (one row per observation), with per-study labels.
    df = load_combined_data()

    summary = summarise_by_group(df, group="study")
    summary.to_csv(os.path.join(out_dir, "dataset_summary_ds.csv"), index=False)
    print(f"Wrote dataset summary for {len(summary)} studies to {out_dir}")

    # Report summary table: per-study range/median/mean/SD of age and both
    # outcomes, plus a pooled All row.
    summary_table_by_group(df, {"understood": df, "spoken": df}).to_csv(
        os.path.join(out_dir, "summary_table_ds.csv"), index=False
    )
    print("Wrote summary_table_ds")

    for x, y, xlabel, ylabel, filename in SCATTERS:
        if y not in df.columns or not df[y].notna().any():
            print(f"Skipping {filename}: no '{y}' observations")
            continue
        fig = scatter_by_group(
            df, x, y, group="study",
            xlabel=xlabel, ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

    # Age-trajectory views of the Down syndrome pool against the pooled
    # median/IQR: every observation as a study-coloured point over the FULL
    # age range (so where the data thins is visible), and the repeat-measures
    # children linked within form over the reporting window.
    ds = load_combined_data(max_age_months=DS_REPEAT_AGE_RANGE[1])
    ds_colours = _shared_group_colours(df["study"].unique())
    for outcome, ylabel in REPEAT_OUTCOMES:
        filename = f"observations_age_{outcome}_ds"
        fig = plot_observations_by_group(
            df, outcome,
            age_range=(DS_REPEAT_AGE_RANGE[0], None),
            group_colors=ds_colours,
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

        filename = f"repeat_measures_age_{outcome}_ds"
        fig = plot_repeat_measures_by_group(
            ds, outcome,
            age_range=DS_REPEAT_AGE_RANGE,
            group_colors=ds_colours,
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

    # The same view of the typically-developing Wordbank reference pool. Each
    # outcome is loaded on its own: requesting understood restricts the TD
    # loader to the bivariate forms, so a joint frame would silently drop the
    # Words & Sentences spoken observations. The ~1,000 repeat-measures
    # children need fainter lines.
    td_frames = {
        outcome: load_data(
            Population.TYPICALLY_DEVELOPING,
            ["study", "subject_id", "form", "age", outcome],
        )
        for outcome, _ in REPEAT_OUTCOMES
    }
    td_colours = _shared_group_colours(
        set().union(*[set(frame["study"].unique()) for frame in td_frames.values()])
    )

    # The TD summary table: the spoken frame covers every admitted form, so it
    # supplies the age statistics and counts; understood comes from its own
    # bivariate-form frame.
    summary_table_by_group(
        td_frames["spoken"],
        {"understood": td_frames["understood"], "spoken": td_frames["spoken"]},
    ).to_csv(os.path.join(out_dir, "summary_table_td.csv"), index=False)
    print("Wrote summary_table_td")
    for outcome, ylabel in REPEAT_OUTCOMES:
        # The TD pool is too dense for a scatter (hundreds of administrations
        # at each integer age), so the by-age view is monthly violins. The
        # axis ends at the measure's own data — comprehension stops where the
        # bivariate forms do, before the pool's 30-month bound.
        filename = f"violins_age_{outcome}_td"
        fig = plot_monthly_violins(
            td_frames[outcome], outcome,
            age_range=(TD_REPEAT_AGE_RANGE[0], None),
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

        filename = f"repeat_measures_age_{outcome}_td"
        fig = plot_repeat_measures_by_group(
            td_frames[outcome], outcome,
            form_col="form",
            age_range=TD_REPEAT_AGE_RANGE,
            bin_width=TD_REPEAT_BIN_WIDTH,
            line_alpha=0.15,
            group_colors=td_colours,
            ylabel=ylabel,
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")

    # Mirror every artefact into the report figure cache.
    artefacts = sorted(os.listdir(out_dir))
    for name in artefacts:
        shutil.copy2(os.path.join(out_dir, name), os.path.join(report_dir, name))
    print(f"Mirrored {len(artefacts)} artefacts into {report_dir}")


if __name__ == "__main__":
    main()



