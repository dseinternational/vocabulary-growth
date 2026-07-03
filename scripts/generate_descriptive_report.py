# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generate the descriptive-data report artefacts (issue #111).

Writes a per-study summary table and study-coloured scatter plots for the pooled
Down syndrome dataset into ``docs/report/figures/descriptives/``, which the
report's "Data and measures" chapter (``methods-data.qmd``) reads. Run after
``prepare_data.py`` and before rendering the report:

    python scripts/generate_descriptive_report.py
"""

import os

import dse_research_utils.environment.setup as setup
import matplotlib.pyplot as plt

import vocab_growth.environment as local_env
from vocab_growth.data_utils import load_combined_data
from vocab_growth.descriptive import scatter_by_group, summarise_by_group

SCATTERS = [
    # (x, y, xlabel, ylabel, filename)
    ("age", "understood", "Age (months)", "Words understood", "scatter_age_understood_ds"),
    ("age", "spoken", "Age (months)", "Words spoken", "scatter_age_spoken_ds"),
    ("understood", "spoken", "Words understood", "Words spoken", "scatter_understood_spoken_ds"),
    ("age", "signed", "Age (months)", "Words signed", "scatter_age_signed_ds"),
]


def main():
    setup.init_script()
    out_dir = os.path.join(local_env.REPORT_FIGS_DIR, "descriptives")
    os.makedirs(out_dir, exist_ok=True)

    # Pooled Down syndrome data (one row per observation), with per-study labels.
    df = load_combined_data()

    summary = summarise_by_group(df, group="study")
    summary.to_csv(os.path.join(out_dir, "dataset_summary_ds.csv"), index=False)
    print(f"Wrote dataset summary for {len(summary)} studies to {out_dir}")

    for x, y, xlabel, ylabel, filename in SCATTERS:
        if y not in df.columns or not df[y].notna().any():
            print(f"Skipping {filename}: no '{y}' observations")
            continue
        fig = scatter_by_group(
            df, x, y, group="study",
            xlabel=xlabel, ylabel=ylabel,
            title=f"{ylabel} vs {xlabel.lower()} (Down syndrome), by study",
            output_dir=out_dir, filename=filename,
        )
        plt.close(fig)
        print(f"Wrote {filename}")


if __name__ == "__main__":
    main()
