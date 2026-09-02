# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prepare everything the Quarto report needs that is not fitted model output.

Four stages, run in this order by default or named individually on the command
line (the order given does not matter; ``pending`` always runs last):

``descriptives``
    Per-study summary tables and observed-data figures, written to
    ``docs/descriptive/figures/`` for the standalone descriptive report and
    mirrored into ``docs/report/figures/descriptives/`` for the "Data" chapter.
    Reads the prepared dataset, so ``prepare_data.py`` must have run.
``illustrations``
    The introduction's Bayesian-updating and Binomial-versus-Beta-Binomial
    figures, simulated, into ``docs/report/figures/``.
``priors``
    The methods chapter's prior-trajectory and GP-anchoring figures, simulated
    from the registered definitions' own priors, into
    ``docs/report/figures/methods/``. The anchoring figure projects onto the
    observed ages, so this stage also needs the prepared dataset.
``pending``
    A labelled placeholder for every figure a report chapter references that is
    still absent, so ``quarto render`` never fails on a missing file.

    python scripts/prepare_report_figures.py                      # all four
    python scripts/prepare_report_figures.py descriptives         # after prepare_data.py
    python scripts/prepare_report_figures.py illustrations priors pending

None of this is validated by ``sync_report_figures.py``, which covers fit
artefacts only; the sync replaces the per-model and comparison directories and
leaves these alone, but nothing else regenerates them either. The figure code
lives in :mod:`vocab_growth.descriptive` and
:mod:`vocab_growth.report_illustrations`; this script only routes it to the
figure cache.
"""

import argparse
import os
import re
import shutil
from pathlib import Path

import dse_research_utils.environment.setup as setup
import dse_research_utils.plot.styles as plot_styles
import matplotlib.pyplot as plt

import vocab_growth.environment as local_env
from vocab_growth.descriptive import write_descriptive_artefacts
from vocab_growth.report_illustrations import (
    RANDOM_SEED,
    write_intro_illustrations,
    write_prior_illustrations,
)

FIGURE_REF_RE = re.compile(r"!\[[^\n]*?\]\((figures/[^)\s]+)\)")


def run_descriptives() -> None:
    # Primary home: the standalone descriptive report (docs/descriptive/).
    out_dir = os.path.join(local_env.DOCS_DIR, "descriptive", "figures")
    # Mirror: the main report's figure cache, so methods-data.qmd keeps rendering.
    mirror = os.path.join(local_env.REPORT_FIGS_DIR, "descriptives")
    os.makedirs(mirror, exist_ok=True)
    write_descriptive_artefacts(out_dir)
    artefacts = sorted(os.listdir(out_dir))
    for name in artefacts:
        shutil.copy2(os.path.join(out_dir, name), os.path.join(mirror, name))
    print(f"Mirrored {len(artefacts)} artefacts into {mirror}")


def run_illustrations(seed: int) -> None:
    write_intro_illustrations(local_env.REPORT_FIGS_DIR, seed=seed)


def run_priors(n_draws: int, seed: int) -> None:
    write_prior_illustrations(
        os.path.join(local_env.REPORT_FIGS_DIR, "methods"), n_draws=n_draws, seed=seed
    )


def _figure_path_for_ref(ref: str) -> Path:
    """Filesystem path for a Markdown figure reference in the report."""
    path = Path(local_env.REPORT_DIR) / ref
    return path if path.suffix else path.with_suffix(".png")


def report_figure_refs() -> list[Path]:
    """Figure paths referenced by report QMD files.

    Quarto's report config sets ``default-image-extension: png``, so extensionless
    references resolve to ``.png`` during PDF rendering.
    """
    paths = set()
    for qmd in Path(local_env.REPORT_DIR).glob("*.qmd"):
        text = qmd.read_text(encoding="utf-8")
        paths.update(
            _figure_path_for_ref(match.group(1))
            for match in FIGURE_REF_RE.finditer(text)
        )
    return sorted(paths)


def write_pending_figure(path: Path) -> None:
    """Create an obvious placeholder for a figure that is not synced yet."""
    path.parent.mkdir(parents=True, exist_ok=True)

    rel = path.relative_to(local_env.REPORT_DIR).as_posix()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle(
            (0.02, 0.02),
            0.96,
            0.96,
            fill=False,
            linewidth=1.5,
            edgecolor="#777777",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.5,
        0.58,
        "Pending figure",
        ha="center",
        va="center",
        fontsize=20,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.42,
        rel,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        wrap=True,
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_pending() -> None:
    """Create placeholders for report figures that are not available locally."""
    written = 0
    for path in report_figure_refs():
        if not path.exists():
            write_pending_figure(path)
            written += 1
    print(f"Pending figure placeholders written: {written}")


STAGES = ("descriptives", "illustrations", "priors", "pending")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stages", nargs="*", choices=STAGES, metavar="stage",
                    help=f"Any of {', '.join(STAGES)}; default all, in that order.")
    ap.add_argument("--draws", type=int, default=250,
                    help="Prior draws per model-structure figure.")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help="Seed for every simulated figure.")
    args = ap.parse_args()
    selected = [s for s in STAGES if not args.stages or s in args.stages]

    setup.init_script()
    # The 12pt base font the figures render at comes from the shared style
    # (dse-research-utils FONT_SIZE_DEFAULT), applied explicitly so the output
    # does not depend on init_script's defaults.
    plot_styles.set_matplotlib_default_style()
    os.makedirs(local_env.REPORT_FIGS_DIR, exist_ok=True)
    print(f"Report figure cache: {local_env.REPORT_FIGS_DIR}")

    for stage in selected:
        print(f"\n== {stage} ==")
        if stage == "descriptives":
            run_descriptives()
        elif stage == "illustrations":
            run_illustrations(args.seed)
        elif stage == "priors":
            run_priors(args.draws, args.seed)
        elif stage == "pending":
            run_pending()


if __name__ == "__main__":
    main()
