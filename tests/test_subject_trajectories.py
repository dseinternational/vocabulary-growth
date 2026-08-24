# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the observed per-child trajectories on the median-trend plot.

The overlay exists to contrast individual growth with the population median, and
its one real hazard is the recording form: the Down syndrome pool spans item
counts from 396 to 810, roughly half the children with three or more
administrations are recorded on more than one form, and a child near the ceiling
of a short form can record *fewer* words on a longer form a month later. Joining
those points with a plain line draws a developmental reversal that did not
happen, so the segment across a form change is drawn dashed and the compressed
observations are marked.

These pin that behaviour, the administration threshold, and -- the defect most
likely to be introduced later and least likely to be noticed -- that the
``max_age_months`` cut is applied to the trajectory inputs as well as to the
observations, since a mask applied to one and not the other joins one child's
points to another's.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vocab_growth import plotting

_COLOUR = plotting._TRAJECTORY_COLOUR
_PP_COLOUR = plotting._PREDICTIVE_TRAJECTORY_COLOUR


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _trajectory_lines():
    """The trajectory segments drawn on the current axes."""
    return [ln for ln in plt.gca().get_lines() if ln.get_color() == _COLOUR]


def _draw(subjects, ages, counts, forms=None, minimum=3):
    plt.figure()
    return plotting._draw_subject_trajectories(
        np.asarray(ages, dtype=float),
        np.asarray(counts, dtype=float),
        np.asarray(subjects),
        form_max=None if forms is None else np.asarray(forms, dtype=float),
        min_administrations=minimum,
    )


# ---------------------------------------------------------------------------
# Which children are drawn
# ---------------------------------------------------------------------------
def test_a_child_needs_the_minimum_administrations():
    """Two points are a segment, not a trajectory, and must not be drawn."""
    summary = _draw(
        subjects=["a", "a", "a", "b", "b", "c"],
        ages=[12, 18, 24, 12, 18, 12],
        counts=[10, 40, 90, 5, 20, 3],
    )
    assert summary["children"] == 1  # only "a" has three
    assert len(_trajectory_lines()) == 2  # three points -> two segments


def test_the_threshold_is_configurable():
    summary = _draw(
        subjects=["a", "a", "b", "b"],
        ages=[12, 18, 12, 18],
        counts=[10, 40, 5, 20],
        minimum=2,
    )
    assert summary["children"] == 2


def test_no_eligible_child_draws_nothing():
    summary = _draw(subjects=["a", "b"], ages=[12, 18], counts=[10, 40])
    assert summary == {"children": 0, "form_changes": 0, "near_ceiling": 0}
    assert _trajectory_lines() == []


def test_observations_are_ordered_by_age_not_by_input_order():
    """A frame that arrives unsorted must still draw a monotonic path."""
    summary = _draw(
        subjects=["a", "a", "a"],
        ages=[24, 12, 18],
        counts=[90, 10, 40],
    )
    assert summary["children"] == 1
    drawn = sorted(tuple(ln.get_xdata()) for ln in _trajectory_lines())
    assert drawn == [(12.0, 18.0), (18.0, 24.0)]


# ---------------------------------------------------------------------------
# The recording form
# ---------------------------------------------------------------------------
def test_a_form_change_is_dashed_and_counted():
    """The real hazard: 393 words on a 416-item form, then 347 on an 810."""
    summary = _draw(
        subjects=["a"] * 3,
        ages=[24, 47, 48],
        counts=[82, 393, 347],
        forms=[810, 416, 810],
    )
    assert summary["form_changes"] == 2  # 810->416 and 416->810
    styles = {ln.get_linestyle() for ln in _trajectory_lines()}
    assert "-" not in styles  # both segments cross a form change


def test_segments_within_one_form_stay_solid():
    summary = _draw(
        subjects=["a"] * 3,
        ages=[12, 18, 24],
        counts=[10, 40, 90],
        forms=[810, 810, 810],
    )
    assert summary["form_changes"] == 0
    assert all(ln.get_linestyle() == "-" for ln in _trajectory_lines())


def test_a_child_on_two_forms_keeps_its_within_form_segments_solid():
    """Only the crossing segment is untrustworthy; the rest still carries shape."""
    summary = _draw(
        subjects=["a"] * 4,
        ages=[12, 18, 24, 30],
        counts=[10, 40, 90, 120],
        forms=[396, 396, 810, 810],
    )
    assert summary["form_changes"] == 1
    solid = [ln for ln in _trajectory_lines() if ln.get_linestyle() == "-"]
    assert len(solid) == 2


def test_without_form_information_every_segment_is_solid():
    """Callers whose frame lacks the column degrade, rather than dashing all."""
    summary = _draw(subjects=["a"] * 3, ages=[12, 18, 24], counts=[10, 40, 90])
    assert summary["form_changes"] == 0
    assert summary["near_ceiling"] == 0
    assert all(ln.get_linestyle() == "-" for ln in _trajectory_lines())


def test_near_ceiling_observations_are_counted():
    """393/416 is 94% and must be marked; 90/810 is 11% and must not."""
    summary = _draw(
        subjects=["a"] * 3,
        ages=[12, 18, 24],
        counts=[90, 393, 100],
        forms=[810, 416, 810],
    )
    assert summary["near_ceiling"] == 1


def test_near_ceiling_only_considers_the_child_s_own_form():
    """400 words is near the ceiling of a 416 form and nowhere near an 810."""
    summary = _draw(
        subjects=["a"] * 3 + ["b"] * 3,
        ages=[12, 18, 24] * 2,
        counts=[400, 401, 402, 400, 401, 402],
        forms=[416, 416, 416, 810, 810, 810],
    )
    assert summary["near_ceiling"] == 3  # child "a" only


# ---------------------------------------------------------------------------
# Integration with the median-trend plot
# ---------------------------------------------------------------------------
def _median_trend(**kwargs):
    rng = np.random.default_rng(0)
    ages = np.linspace(10, 40, 25)
    y_plot = np.linspace(5, 300, 25)[:, None] * rng.lognormal(0, 0.2, size=(25, 40))
    return plotting.plot_posterior_predictive_median_trend(ages, y_plot, **kwargs)


def test_without_subject_ids_the_plot_is_unchanged():
    """Every caller that predates trajectories must get the plain scatter."""
    _median_trend(
        x_obs=pd.Series([12.0, 18.0, 24.0]),
        y_obs=pd.Series([10.0, 40.0, 90.0]),
    )
    assert _trajectory_lines() == []
    labels = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    assert not any("Same child" in label for label in labels)


def test_the_legend_states_the_figure_s_own_composition():
    frame = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b", "b", "b"],
            "age": [12.0, 18.0, 24.0, 12.0, 18.0, 24.0],
            "count": [10.0, 40.0, 90.0, 380.0, 30.0, 60.0],
            "form": [810.0, 810.0, 810.0, 396.0, 810.0, 810.0],
        }
    )
    _median_trend(
        x_obs=frame["age"],
        y_obs=frame["count"],
        subject_ids=frame["subject"],
        form_max=frame["form"],
    )
    labels = " | ".join(t.get_text() for t in plt.gca().get_legend().get_texts())
    assert "Same child (2 with 3+)" in labels
    assert "Form changed - not comparable (1)" in labels
    assert "Near that form's ceiling (1)" in labels


def test_the_age_cap_is_applied_to_the_trajectory_inputs_too():
    """A mask applied to the ages but not the subjects joins the wrong children.

    Child "a" is entirely under the cap; "b" is entirely over it. If the cut is
    applied to ``x_obs``/``y_obs`` only, the subject column still carries b's
    labels, so a's three points get read as two children and the trajectory
    silently disappears -- or worse, joins across children.
    """
    frame = pd.DataFrame(
        {
            "subject": ["a", "a", "a", "b", "b", "b"],
            "age": [12.0, 18.0, 24.0, 60.0, 66.0, 72.0],
            "count": [10.0, 40.0, 90.0, 300.0, 350.0, 400.0],
            "form": [810.0] * 6,
        }
    )
    _median_trend(
        x_obs=frame["age"],
        y_obs=frame["count"],
        subject_ids=frame["subject"],
        form_max=frame["form"],
        max_age_months=40.0,
    )
    lines = _trajectory_lines()
    assert len(lines) == 2, "child a's two segments, and nothing from b"
    drawn_ages = sorted(x for ln in lines for x in ln.get_xdata())
    assert max(drawn_ages) <= 40.0
    labels = " | ".join(t.get_text() for t in plt.gca().get_legend().get_texts())
    assert "Same child (1 with 3+)" in labels


def test_rows_with_a_missing_outcome_are_not_joined_across():
    """A child measured on only one outcome must not gain a phantom segment."""
    summary = _draw(
        subjects=["a"] * 4,
        ages=[12, 18, 24, 30],
        counts=[10, np.nan, 90, 120],
        forms=[810] * 4,
    )
    assert summary["children"] == 1
    drawn = sorted(tuple(ln.get_xdata()) for ln in _trajectory_lines())
    assert drawn == [(12.0, 24.0), (24.0, 30.0)]


# ---------------------------------------------------------------------------
# Posterior predictive unseen-child trajectories
# ---------------------------------------------------------------------------
def _pp_lines():
    return [ln for ln in plt.gca().get_lines() if ln.get_color() == _PP_COLOUR]


def _samples(n_grid=25, n_samples=200, seed=0):
    """Coherent curves: one child offset per column, applied across the grid."""
    rng = np.random.default_rng(seed)
    base = np.linspace(5, 300, n_grid)[:, None]
    offset = rng.normal(1.0, 0.35, size=(1, n_samples))
    return base * offset


def test_predictive_trajectories_draw_the_requested_number():
    plt.figure()
    drawn = plotting._draw_predictive_trajectories(
        np.linspace(10, 40, 25), _samples(), n_trajectories=12, seed=0
    )
    assert drawn == 12
    assert len(_pp_lines()) == 12


def test_more_requested_than_available_draws_what_there_is():
    plt.figure()
    drawn = plotting._draw_predictive_trajectories(
        np.linspace(10, 40, 25), _samples(n_samples=7), n_trajectories=60, seed=0
    )
    assert drawn == 7


def test_no_samples_draws_nothing():
    plt.figure()
    drawn = plotting._draw_predictive_trajectories(
        np.linspace(10, 40, 25), np.zeros((25, 0)), n_trajectories=10, seed=0
    )
    assert drawn == 0
    assert _pp_lines() == []


def test_the_same_seed_redraws_the_same_figure():
    """A figure regenerated from one trace must not change which draws it shows."""
    samples = _samples()
    ages = np.linspace(10, 40, 25)
    plt.figure()
    plotting._draw_predictive_trajectories(ages, samples, 10, seed=3)
    first = [tuple(ln.get_ydata()) for ln in _pp_lines()]
    plt.figure()
    plotting._draw_predictive_trajectories(ages, samples, 10, seed=3)
    second = [tuple(ln.get_ydata()) for ln in _pp_lines()]
    assert first == second
    plt.figure()
    plotting._draw_predictive_trajectories(ages, samples, 10, seed=4)
    assert [tuple(ln.get_ydata()) for ln in _pp_lines()] != first


def test_a_grid_mismatch_is_refused():
    plt.figure()
    with pytest.raises(ValueError, match="must match"):
        plotting._draw_predictive_trajectories(
            np.linspace(10, 40, 25), _samples(n_grid=30), 5, seed=0
        )


def test_a_one_dimensional_sample_array_is_refused():
    plt.figure()
    with pytest.raises(ValueError, match="n_grid, n_samples"):
        plotting._draw_predictive_trajectories(
            np.linspace(10, 40, 25), np.zeros(25), 5, seed=0
        )


def test_the_plot_reports_the_predictive_trajectories_in_its_legend():
    _median_trend(
        x_obs=pd.Series([12.0, 18.0, 24.0]),
        y_obs=pd.Series([10.0, 40.0, 90.0]),
        trajectory_samples=_samples(),
        n_trajectories=15,
    )
    labels = " | ".join(t.get_text() for t in plt.gca().get_legend().get_texts())
    assert "Predicted child (15 draws)" in labels
    assert len(_pp_lines()) == 15


def test_the_age_cap_trims_the_predictive_trajectories_with_the_grid():
    """Trimmed with y_plot, or the curves run past the reported range."""
    _median_trend(
        x_obs=pd.Series([12.0, 18.0]),
        y_obs=pd.Series([10.0, 40.0]),
        trajectory_samples=_samples(),
        n_trajectories=5,
        max_age_months=25.0,
    )
    for line in _pp_lines():
        assert max(line.get_xdata()) <= 25.0


def test_both_overlays_coexist():
    """The observed and predicted trajectories must stay visually separable."""
    frame = pd.DataFrame(
        {
            "subject": ["a"] * 3 + ["b"] * 3,
            "age": [12.0, 18.0, 24.0] * 2,
            "count": [10.0, 40.0, 90.0, 20.0, 60.0, 130.0],
            "form": [810.0] * 6,
        }
    )
    _median_trend(
        x_obs=frame["age"],
        y_obs=frame["count"],
        subject_ids=frame["subject"],
        form_max=frame["form"],
        trajectory_samples=_samples(),
        n_trajectories=8,
    )
    assert len(_pp_lines()) == 8
    assert len(_trajectory_lines()) == 4  # two children, two segments each
    assert _PP_COLOUR != _COLOUR
    labels = " | ".join(t.get_text() for t in plt.gca().get_legend().get_texts())
    assert "Predicted child (8 draws)" in labels
    assert "Same child (2 with 3+)" in labels
