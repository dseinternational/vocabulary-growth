# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the parameter-recovery simulation invariants (issue #163).

Data-free and sampling-free, so these run in CI. They cover the three places the
simulator can be quietly wrong:

1. **Nested-likelihood coherence.** The synthetic data must be fitted under the
   same decomposition that generated it. The simulator draws a child count
   against the simulated parent, so every observed child row whose parent is a
   valid total must come back classified as nested — including the rows that were
   *marginal* in the real data because the recorded child count exceeded its
   parent. If that stopped holding, the refit would fit a different likelihood
   from the one that produced the data.
2. **The frame round trip.** The synthetic frame is handed to the refit through a
   file. A lossy write would surface as an unexplained difference in the refit
   rather than as an error here.
3. **Write-back placement.** Simulated values must land on exactly the rows the
   likelihood covers, leaving the real missingness pattern untouched.
"""

import numpy as np
import pandas as pd
import pytest

from vocab_growth.models.likelihood_utils import nested_outcome_spec
from vocab_growth.recovery.simulate import (
    _apply_parent_totals,
    _neutralise_child_columns,
    _neutralise_composition_cells,
    _read_frame,
    _verify_coherence,
    _write_column,
    _write_frame,
)
from vocab_growth.recovery.spec import BIVARIATE_RE_SPEC, JOINT_SPEC

N_TRIALS = 810


def _bivariate_frame():
    """A frame in the bivariate engine's shape, including the awkward rows.

    Row 2 is the case that matters: the recorded spoken count exceeds the recorded
    comprehension count, which the real-data pipeline routes to the *marginal*
    likelihood as a source-data violation.
    """
    return pd.DataFrame(
        {
            "age": [12.0, 24.0, 36.0, 48.0, 60.0],
            "understood": [100.0, np.nan, 40.0, 30.0, 800.0],
            "spoken": [25.0, 10.0, 50.0, np.nan, 700.0],
            "study": ["a", "a", "b", "b", "b"],
            "study_code": [0, 0, 1, 1, 1],
        }
    )


def test_neutralising_children_makes_nesting_depend_only_on_the_parent():
    frame = _bivariate_frame()
    real = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=N_TRIALS
    )
    # Row 2 (spoken 50 > understood 40) starts out as a marginal violation.
    assert real.n_parent_violations == 1
    np.testing.assert_array_equal(real.is_conditional, [True, False, False, True])

    _neutralise_child_columns(frame, BIVARIATE_RE_SPEC, {"spoken"})

    neutral = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=N_TRIALS
    )
    # Same rows are in the likelihood, but classification now follows the parent
    # alone: every row with a valid comprehension total is nested.
    np.testing.assert_array_equal(neutral.indices, real.indices)
    np.testing.assert_array_equal(neutral.is_conditional, [True, False, True, True])
    np.testing.assert_array_equal(neutral.trials, [100, N_TRIALS, 40, 800])
    assert neutral.n_parent_violations == 0
    # Missingness is untouched — only observed child counts were zeroed.
    assert frame["spoken"].isna().tolist() == [False, False, False, True, False]


def test_simulated_child_reproduces_the_classification_used_to_generate_it():
    """The invariant the simulator relies on, stated as a test.

    Drawing the child against the simulated parent guarantees child <= parent, so
    recomputing the classification from the finished synthetic frame returns
    exactly what the simulation used.
    """
    frame = _bivariate_frame()
    _neutralise_child_columns(frame, BIVARIATE_RE_SPEC, {"spoken"})
    used = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=N_TRIALS
    )

    # Stand in for the draw: a conditional row gets a count within its parent, a
    # marginal row a count within the inventory.
    rng = np.random.default_rng(0)
    drawn = np.array(
        [rng.integers(0, trials + 1) for trials in used.trials], dtype=float
    )
    frame.iloc[used.indices, frame.columns.get_loc("spoken")] = drawn

    final = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=N_TRIALS
    )
    np.testing.assert_array_equal(final.indices, used.indices)
    np.testing.assert_array_equal(final.is_conditional, used.is_conditional)
    np.testing.assert_array_equal(final.trials, used.trials)
    assert final.n_parent_violations == 0


def test_coherence_check_rejects_a_mismatched_denominator():
    """A denominator disagreement must abort rather than be fitted.

    Simulates the failure mode the guard exists for: the model rebuilt from the
    synthetic frame derives a different set of nested rows from the one the
    simulation drew against.
    """
    frame = _bivariate_frame()
    _neutralise_child_columns(frame, BIVARIATE_RE_SPEC, {"spoken"})
    used = nested_outcome_spec(
        frame, parent_col="understood", outcome_col="spoken", n_trials=N_TRIALS
    )

    class _FakeData:
        def __init__(self, value):
            self._value = value

        def get_value(self):
            return self._value

    class _FakeModel:
        def __init__(self, trials, is_conditional):
            self.named_vars = {
                "s_likelihood_n": _FakeData(trials),
                "s_is_conditional": _FakeData(is_conditional),
            }

        def __getitem__(self, name):
            return self.named_vars[name]

    recorded = {
        "s_likelihood_n": used.trials.copy(),
        "s_is_conditional": used.is_conditional.astype(int).copy(),
    }

    matching = _FakeModel(used.trials, used.is_conditional.astype(int))
    report = _verify_coherence(matching, BIVARIATE_RE_SPEC, frame, recorded)
    assert report["s_likelihood_n"] == "matches"

    tampered_trials = used.trials.copy()
    tampered_trials[0] += 1
    with pytest.raises(RuntimeError, match="different s_likelihood_n"):
        _verify_coherence(
            _FakeModel(tampered_trials, used.is_conditional.astype(int)),
            BIVARIATE_RE_SPEC,
            frame,
            recorded,
        )


def test_cross_tab_cells_are_made_consistent_with_a_repointed_total():
    """The engine refuses to build unless cells sum to their total.

    Once the four-cell total is repointed at the simulated comprehension count,
    the real cells no longer sum to it, so the pending cells must be reset to a
    valid partition before the rebuild.
    """
    frame = pd.DataFrame(
        {
            "age": [24.0, 30.0],
            # Simulated comprehension, which the four-cell total must follow.
            "understood": [120.0, 200.0],
            "spoken": [np.nan, np.nan],
            "signed": [np.nan, np.nan],
            "understood_only": [50.0, 60.0],
            "signed_only": [10.0, 20.0],
            "spoken_only": [20.0, 30.0],
            "signed_spoken": [5.0, 10.0],
            "cell_total": [85.0, 120.0],
        }
    )

    _apply_parent_totals(frame, JOINT_SPEC)
    np.testing.assert_array_equal(frame["cell_total"], [120.0, 200.0])
    # Stale cells no longer partition the new total.
    cells = ["understood_only", "signed_only", "spoken_only", "signed_spoken"]
    assert not np.allclose(frame[cells].sum(axis=1), frame["cell_total"])

    _neutralise_composition_cells(frame, JOINT_SPEC, {"cells_obs"})

    np.testing.assert_allclose(frame[cells].sum(axis=1), frame["cell_total"])
    # The cross-tab rows must stay recognisable as cross-tab rows: the engine keys
    # off signed_spoken being present, so it must be zero, never missing.
    assert frame["signed_spoken"].notna().all()


def test_conditioned_total_is_not_repointed_at_the_parent():
    # nz_01's produced total is conditioned on, not generated, so simulating
    # comprehension must not move it.
    frame = pd.DataFrame(
        {
            "age": [30.0],
            "understood": [np.nan],
            "prod_signed_only": [5.0],
            "prod_spoken_only": [10.0],
            "prod_signed_spoken": [3.0],
            "prod_total": [18.0],
        }
    )
    _apply_parent_totals(frame, JOINT_SPEC)
    assert frame["prod_total"].tolist() == [18.0]
    assert "prod_total" in JOINT_SPEC.conditioned_totals


def test_write_column_places_values_on_the_likelihood_rows_only():
    frame = _bivariate_frame()
    rows = np.array([0, 2, 4])
    _write_column(frame, "understood", rows, np.array([1.0, 2.0, 3.0]))
    assert frame["understood"].tolist()[0] == 1.0
    assert frame["understood"].tolist()[2] == 2.0
    assert frame["understood"].tolist()[4] == 3.0
    # The unobserved row keeps its missingness.
    assert np.isnan(frame["understood"].tolist()[1])


def test_write_column_rejects_a_length_mismatch():
    frame = _bivariate_frame()
    with pytest.raises(ValueError, match="simulated value"):
        _write_column(frame, "spoken", np.array([0, 1]), np.array([5.0]))


def test_write_column_rejects_an_unknown_column():
    frame = _bivariate_frame()
    with pytest.raises(KeyError):
        _write_column(frame, "not_a_column", np.array([0]), np.array([1.0]))


def test_frame_round_trip_preserves_values_dtypes_and_missingness(tmp_path):
    """The frame reaches the refit through a file, so it must survive exactly.

    Parquet via DuckDB keeps dtypes as well as values, so this asserts dtype
    identity — the property a text round trip could not offer, and the reason a
    numeric-looking ``subject_id`` cannot silently become an integer.
    """
    frame = _bivariate_frame()
    frame["subject_key"] = ["a::1", "a::2", "b::3", "b::4", "b::5"]
    # All-numeric-looking string ids: the case a CSV round trip would coerce.
    frame["subject_id"] = ["0012", "34", "56", "78", "90"]
    path = tmp_path / "synthetic.parquet"

    schema = _write_frame(frame, str(path))
    reloaded = _read_frame(str(path))

    assert list(reloaded.columns) == list(frame.columns)
    pd.testing.assert_frame_equal(reloaded, frame)
    assert reloaded["understood"].isna().tolist() == frame["understood"].isna().tolist()
    assert reloaded["subject_id"].tolist() == ["0012", "34", "56", "78", "90"]
    assert schema["study"] == str(frame["study"].dtype)


def test_frame_round_trip_reports_a_lossy_write(tmp_path, monkeypatch):
    """A lossy write must fail at simulation time, not in the refit."""
    frame = _bivariate_frame()
    path = tmp_path / "synthetic.parquet"

    def _corrupting_read(read_path):
        loaded = _read_frame(read_path)
        loaded.loc[0, "understood"] = 999.0
        return loaded

    monkeypatch.setattr(
        "vocab_growth.recovery.simulate._read_frame", _corrupting_read
    )
    with pytest.raises(RuntimeError, match="changed on the Parquet round trip"):
        _write_frame(frame, str(path))


def test_frame_round_trip_catches_a_dtype_change(tmp_path, monkeypatch):
    """Values surviving is not enough — a dtype change must also abort.

    An integer study code arriving back as a float would still compare equal
    numerically, but the engines index with it.
    """
    frame = _bivariate_frame()
    path = tmp_path / "synthetic.parquet"

    def _dtype_shifting_read(read_path):
        loaded = _read_frame(read_path)
        loaded["study_code"] = loaded["study_code"].astype(float)
        return loaded

    monkeypatch.setattr(
        "vocab_growth.recovery.simulate._read_frame", _dtype_shifting_read
    )
    with pytest.raises(RuntimeError, match="changed dtype on the Parquet round trip"):
        _write_frame(frame, str(path))


def test_reading_a_missing_frame_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="No synthetic frame"):
        _read_frame(str(tmp_path / "absent.parquet"))


def test_truth_draws_are_spread_and_distinct_within_a_chain():
    """Replicates must land on well-separated draws, not adjacent ones.

    Adjacent draws in a Markov chain are autocorrelated, so they would be
    near-duplicate truths. The positions must also be stable: adding a fourth
    replicate must not move the draws the first three already used.
    """
    from vocab_growth.recovery.simulate import _spread_index

    n_draws = 2000
    positions = [_spread_index(r, n_draws) for r in range(1, 9)]

    assert len(set(positions)) == len(positions), positions
    assert all(0 <= p < n_draws for p in positions)
    # No two selected draws within 1% of the chain length of each other.
    ordered = sorted(positions)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    assert min(gaps) > n_draws * 0.01, positions
    # Adding replicates does not move earlier ones (positions are stateless).
    assert [_spread_index(r, n_draws) for r in range(1, 4)] == positions[:3]
    # The sequence covers the chain rather than drifting to one end.
    assert min(positions) < n_draws * 0.2
    assert max(positions) > n_draws * 0.8


def test_spread_index_handles_degenerate_and_invalid_sizes():
    from vocab_growth.recovery.simulate import _spread_index

    assert _spread_index(1, 1) == 0
    with pytest.raises(ValueError, match="No draws available"):
        _spread_index(1, 0)


def test_coherence_check_rejects_a_total_that_drifts_from_its_parent():
    """A cross-tab total must equal the comprehension count it partitions.

    Otherwise the comprehension likelihood and the cross-tab likelihood would
    condition on two different totals for the same child.
    """
    frame = pd.DataFrame(
        {
            "age": [24.0],
            "understood": [120.0],
            "spoken": [np.nan],
            "signed": [np.nan],
            "understood_only": [60.0],
            "signed_only": [20.0],
            "spoken_only": [20.0],
            "signed_spoken": [20.0],
            # Cells sum to this, but it is not the simulated comprehension count.
            "cell_total": [120.0],
        }
    )

    class _EmptyModel:
        named_vars: dict = {}

        def __getitem__(self, name):  # pragma: no cover - never reached
            raise KeyError(name)

    # Consistent to start with.
    report = _verify_coherence(_EmptyModel(), JOINT_SPEC, frame, {})
    assert report["cell_total_equals_understood"] == "ok"

    frame.loc[0, "understood"] = 119.0
    with pytest.raises(RuntimeError, match="cell_total disagrees"):
        _verify_coherence(_EmptyModel(), JOINT_SPEC, frame, {})
