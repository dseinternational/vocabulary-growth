# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A figure past its outcome's reporting cap must not reach the report.

``tests/test_reporting_age_policy.py`` checks the written tables, so it cannot
see a figure that has no table row behind it. That is exactly the gap VG02 fell
through: an earlier run wrote a 90-month comprehension figure, the cap was
tightened to 84, the next run wrote a capped table and a capped set of figures
but left the older 90-month file in place, and the gallery -- which globs the
directory -- kept publishing it. It reached ``docs/report/figures/`` too.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from vocab_growth.plotting import (
    plot_posterior_predictive_count_distributions_by_query_age as plot_by_age,
)
from vocab_growth.plotting import ppc_count_distribution_gallery

PREFIX = "posterior_predictive_count_distributions"


def _draws(ages, n_trials=810, n_draws=64):
    rng = np.random.default_rng(3)
    return rng.binomial(n_trials, 0.3, size=(len(ages), n_draws))


def test_gallery_skips_figures_absent_from_the_capped_table(tmp_path, capsys):
    """The guard that protects fits produced before the writer cleared stale files."""
    pd.DataFrame({"age_months": [12.0, 24.0], "median": [1, 2]}).to_csv(
        tmp_path / f"{PREFIX}.csv", index=False
    )
    for age in (12, 24, 90):  # 90 is the orphan from a looser earlier cap
        (tmp_path / f"{PREFIX}_{age}m.png").write_bytes(b"")

    ppc_count_distribution_gallery(PREFIX, directory=str(tmp_path))
    out = capsys.readouterr().out
    assert "12 months" in out
    assert "24 months" in out
    assert "90 months" not in out


def test_writer_removes_the_previous_run_s_figures(tmp_path):
    """A tightened cap must leave no figure behind it."""
    ages = np.array([12.0, 24.0, 90.0])
    plot_by_age(ages, _draws(ages), n_trials=810, output_dir=str(tmp_path), filename=PREFIX)
    assert (tmp_path / f"{PREFIX}_90m.png").exists()

    # Refit at a tighter cap: the 90-month figure must not survive.
    plot_by_age(
        ages,
        _draws(ages),
        n_trials=810,
        output_dir=str(tmp_path),
        filename=PREFIX,
        max_age_months=84.0,
    )
    assert (tmp_path / f"{PREFIX}_24m.png").exists()
    assert not (tmp_path / f"{PREFIX}_90m.png").exists()
    assert not (tmp_path / f"{PREFIX}_90m.svg").exists()


def test_writer_does_not_remove_unrelated_figures(tmp_path):
    """The cleanup is scoped to this prefix's own per-age files."""
    keep = tmp_path / "posterior_kappa.png"
    keep.write_bytes(b"")
    other = tmp_path / f"{PREFIX}_s_12m.png"
    other.write_bytes(b"")

    ages = np.array([12.0])
    plot_by_age(ages, _draws(ages), n_trials=810, output_dir=str(tmp_path), filename=PREFIX)

    assert keep.exists()
    assert other.exists()
