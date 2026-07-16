# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for coherent nested vocabulary outcome likelihoods."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NestedOutcomeSpec:
    """Observed child outcome rows and their row-specific denominators.

    A child outcome such as words spoken is modelled conditionally on the
    observed parent count (words understood) when both counts are available and
    logically nested. Rows without a usable parent count retain a marginal
    likelihood over the full inventory.
    """

    indices: np.ndarray
    observed: np.ndarray
    trials: np.ndarray
    is_conditional: np.ndarray
    n_parent_violations: int

    @property
    def n_observed(self) -> int:
        """Return the number of observed child outcomes."""
        return int(self.observed.size)

    @property
    def n_conditional(self) -> int:
        """Return the number of rows using the nested likelihood."""
        return int(self.is_conditional.sum())

    @property
    def n_marginal(self) -> int:
        """Return the number of rows using the marginal fallback."""
        return self.n_observed - self.n_conditional


def nested_outcome_spec(
    df: pd.DataFrame,
    *,
    parent_col: str,
    outcome_col: str,
    n_trials: int,
    eligible_mask: np.ndarray | pd.Series | None = None,
) -> NestedOutcomeSpec:
    """Classify observed outcomes into conditional and marginal likelihood rows.

    The nested likelihood is used only when the parent count is observed,
    integer-valued, within the inventory bounds, and at least as large as the
    child count. A child count greater than its observed parent is retained via
    the marginal likelihood and reported as a source-data violation rather than
    silently discarded.
    """
    missing = {parent_col, outcome_col}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    if eligible_mask is None:
        eligible = np.ones(len(df), dtype=bool)
    else:
        eligible = np.asarray(eligible_mask, dtype=bool)
        if eligible.shape != (len(df),):
            raise ValueError("eligible_mask must have one value per dataframe row.")

    outcome_numeric = pd.to_numeric(df[outcome_col], errors="coerce")
    observed_mask = eligible & outcome_numeric.notna().to_numpy()
    indices = np.flatnonzero(observed_mask)
    observed_values = outcome_numeric.iloc[indices].to_numpy(dtype=float)

    if not np.all(np.isfinite(observed_values)):
        raise ValueError(f"{outcome_col} contains non-finite observed counts.")
    if not np.all(observed_values == np.floor(observed_values)):
        raise ValueError(f"{outcome_col} contains non-integer observed counts.")
    if not np.all((observed_values >= 0) & (observed_values <= n_trials)):
        raise ValueError(f"{outcome_col} must lie between 0 and n_trials.")

    parent_numeric = pd.to_numeric(df[parent_col], errors="coerce")
    parent_values = parent_numeric.iloc[indices].to_numpy(dtype=float)
    parent_valid = (
        np.isfinite(parent_values)
        & (parent_values == np.floor(parent_values))
        & (parent_values >= 0)
        & (parent_values <= n_trials)
    )
    is_conditional = parent_valid & (observed_values <= parent_values)
    parent_violations = parent_valid & (observed_values > parent_values)

    trials = np.full(indices.size, n_trials, dtype=int)
    trials[is_conditional] = parent_values[is_conditional].astype(int)

    return NestedOutcomeSpec(
        indices=indices,
        observed=observed_values.astype(int),
        trials=trials,
        is_conditional=is_conditional,
        n_parent_violations=int(parent_violations.sum()),
    )
