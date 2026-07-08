# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Compute time-to-milestone tables for every fitted model.

For each target word count we report the posterior **age at which the population
trajectory first reaches it**, summarised as the median and 90% HDI of the
*per-draw* crossing age (median-of-crossings). This is the statistically correct
inversion: it draws each posterior trajectory, finds where that draw crosses the
target, and summarises those ages — as opposed to inverting the median/HDI count
curves (crossing-of-median), which differs for a nonlinear trajectory and cannot
be relabelled as "percentile children". The reported interval is posterior
uncertainty on the population milestone age, not a spread across individual
children (that would need new-child posterior-predictive draws; see the
predictive-interval caveat in the report).

Reads each model's `trace.nc` directly (the per-draw ``p_plot`` /
``p_u_plot``/``p_s_plot`` trajectories), so it requires fitted traces.

Targets: 25, 50, 100, 200, 400 words.

Output per model (paths are under the configured output root — default `output/`;
see `vocab_growth.environment.output_root`):

- `<output-root>/models/<MODEL>/time_to_milestone[_<u|s>].csv`
- `<output-root>/models/<MODEL>/time_to_milestone[_<u|s>].png/.svg`

The same CSVs are also concatenated into
`<output-root>/comparisons/time_to_milestone_all.csv` for cross-population
contrasts.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt
import pandas as pd

from vocab_growth import comparison as C
from vocab_growth import environment as env
from vocab_growth.comparison import (
    load_population_trajectory,
    load_univariate_trajectory,
    milestone_table,
)
from vocab_growth.models.definitions import MODEL_REGISTRY, ModelType

MODELS_DIR = env.models_output_dir()
COMPARE_DIR = env.comparisons_output_dir()

TARGETS = [25, 50, 100, 200, 400]

# Trivariate (VG14) and joint (VG15) models are intentionally excluded: they
# carry per-modality (understood/spoken/signed) trajectories rather than the
# {single, u, s} shape handled here, and would need their own loader.
UNIVARIATE = {
    key: d for key, d in MODEL_REGISTRY.items() if d.model_type == ModelType.UNIVARIATE
}
BIVARIATE = {
    key: d for key, d in MODEL_REGISTRY.items() if d.model_type == ModelType.BIVARIATE
}


def plot_milestone(table: pd.DataFrame, title: str, out_base: str) -> None:
    """Plot the median milestone age (with 90% HDI) against each target count."""
    t = table.dropna(subset=["age_median"])
    fig, ax = plt.subplots(figsize=plot_styles.FIGSIZE_XL)
    if not t.empty:
        yerr = [
            (t["age_median"] - t["age_hdi_lo"]).to_numpy(),
            (t["age_hdi_hi"] - t["age_median"]).to_numpy(),
        ]
        ax.errorbar(
            t["target_words"], t["age_median"], yerr=yerr,
            fmt="o-", color=plot_styles.COLOUR_BLUE, lw=2, capsize=4,
            label="Median age (90% HDI)",
        )
        for _, r in t.iterrows():
            if r["prop_reaching"] < 0.999:
                ax.annotate(
                    f"{r['prop_reaching']:.0%} of draws",
                    (r["target_words"], r["age_hdi_hi"]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=plot_styles.LINE_COLOUR,
                )
    ax.set_xlabel("Target words")
    ax.set_ylabel("Age reached (months)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right", frameon=True)
    fig.savefig(out_base + ".png")
    fig.savefig(out_base + ".svg")
    plt.close(fig)


def _emit(table: pd.DataFrame, model_id: str, pop: str, outcome: str,
          out_csv: str, title: str, out_base: str, merged: list[dict]) -> None:
    table = table.copy()
    table.insert(0, "outcome", outcome)
    table.insert(0, "population", pop)
    table.insert(0, "model", model_id)
    table.to_csv(out_csv, index=False)
    plot_milestone(table, title, out_base)
    merged.extend(table.to_dict("records"))


def process_univariate(key: str, merged: list[dict]) -> None:
    d = MODEL_REGISTRY[key]
    trace = C.trace_path(key)
    if not os.path.exists(trace):
        print(f"  skip {d.model_id}: not fitted yet ({trace} absent)")
        return
    ages, W = load_univariate_trajectory(trace, d.n_trials)
    pop, outcome = d.population.value.upper(), d.outcome.value
    model_dir = C.model_dir(key)
    _emit(
        milestone_table(W, ages, TARGETS), d.model_id, pop, outcome,
        os.path.join(model_dir, "time_to_milestone.csv"),
        f"Time-to-milestone — {d.model_id} ({pop}, {outcome})",
        os.path.join(model_dir, "time_to_milestone"), merged,
    )


def process_bivariate(key: str, merged: list[dict]) -> None:
    d = MODEL_REGISTRY[key]
    trace = C.trace_path(key)
    if not os.path.exists(trace):
        print(f"  skip {d.model_id}: not fitted yet ({trace} absent)")
        return
    ages, U, S = load_population_trajectory(trace, d.n_trials)
    pop = d.population.value.upper()
    model_dir = C.model_dir(key)
    for outcome, W, suffix in (("understood", U, "u"), ("spoken", S, "s")):
        _emit(
            milestone_table(W, ages, TARGETS), d.model_id, pop, outcome,
            os.path.join(model_dir, f"time_to_milestone_{suffix}.csv"),
            f"Time-to-milestone — {d.model_id} ({pop}, {outcome})",
            os.path.join(model_dir, f"time_to_milestone_{suffix}"), merged,
        )


def main() -> None:
    env.preflight_disk(2.0, env.output_root(), label="milestone outputs")
    plot_styles.set_matplotlib_default_style()
    os.makedirs(COMPARE_DIR, exist_ok=True)

    merged: list[dict] = []
    for key in UNIVARIATE:
        process_univariate(key, merged)
    for key in BIVARIATE:
        process_bivariate(key, merged)

    merged_df = pd.DataFrame(merged)
    merged_df.to_csv(os.path.join(COMPARE_DIR, "time_to_milestone_all.csv"),
                     index=False)
    n_models = len(UNIVARIATE) + len(BIVARIATE)
    print(f"Wrote per-model CSV+plot for up to {n_models} models.")
    print(f"Combined: {os.path.join(COMPARE_DIR, 'time_to_milestone_all.csv')}")


if __name__ == "__main__":
    main()
