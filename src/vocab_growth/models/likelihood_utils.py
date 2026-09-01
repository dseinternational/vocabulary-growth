# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Helpers for coherent nested vocabulary outcome likelihoods."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# PyMC is imported inside the two graph builders at the foot of this module
# rather than here: `definitions.py` reads the treatment constants above, and it
# is the one module in the package that anything -- a script listing models, a
# manifest reader -- may want without paying for the PyMC import.

# --------------------------------------------------------------------------
# How a nested outcome treats the rows it cannot condition on
# --------------------------------------------------------------------------
#
# The paired model is U ~ BB(N, p_U, kappa_U) then S | U ~ BB(U, q, kappa_S).
# Rows with no usable parent count cannot take the second line, and the engines
# have always substituted S ~ BB(N, p_U*q, kappa_S). That substitution has the
# right *mean* but is not the paired model's marginal, and it affects 455 of the
# 1,428 spoken observations in the current frame (issues #233 and #236). These
# four treatments are the sensitivity family over that choice; every one of them
# leaves the conditional branch alone.
#
# What the exact marginal is, and why `moment_matched` is not a guess:
# binomial thinning gives S | theta_U, theta_S ~ Bin(N, theta_U*theta_S)
# exactly, so the marginal of S is a Binomial mixed over the *product* of the
# two independent Betas. That product is not Beta, but its first two moments are
# closed-form, so a Beta-Binomial can match them exactly -- see
# `product_marginal_concentration`.

SPOKEN_FALLBACK_PRODUCT = "product_marginal"
"""S ~ BB(N, p_U*q, kappa_S) on fallback rows. The historical treatment.

Mean-correct, and wrong in the variance -- but not in a fixed direction, which
is why the branch has to be measured rather than bounded. Writing
``a = Var(theta_U)``, ``b = Var(theta_S)`` and working through
``Var(theta_U theta_S) = ab + a q^2 + b p_U^2`` against this treatment's
``p_U q (1 - p_U q) / (1 + kappa_S)``, everything cancels to

    the true marginal is MORE dispersed than this one  <=>  q kappa_S > kappa_U

with ``p_U`` dropping out entirely. Whichever of the two Beta processes is less
concentrated dominates; at ``q kappa_S = kappa_U`` the two variances are exactly
equal. Which side the fitted models sit on is a question about their posterior
concentrations, not one that can be settled from the structure."""

SPOKEN_FALLBACK_PAIRED_ONLY = "paired_only"
"""Drop the fallback rows from the outcome's likelihood entirely.

The cleanest sensitivity and the most expensive one: it answers "what would the
model say if the approximation had never been made" at the cost of a third of
the spoken observations, which are older and concentrated by study, so the loss
is not a random subsample. Read it as a bound, not as a better model."""

SPOKEN_FALLBACK_SEPARATE_DISPERSION = "separate_dispersion"
"""As `product_marginal`, but the fallback branch carries its own concentration.

One scalar, `log_kappa_<outcome>_fallback`, multiplying the shared age-varying
kappa on the fallback rows only. Nests the default exactly at zero, so its
posterior is a direct readout of how much dispersion the branch wants and in
which direction -- which a duplicated two-anchor block, weakly identified on 455
rows, would not give. Unlike `moment_matched` it is agnostic: it lets the data
say what the branch needs instead of imposing what the paired model implies,
which makes the two together a check on each other."""

SPOKEN_FALLBACK_MOMENT_MATCHED = "moment_matched"
"""Match the exact first two moments of the paired model's true marginal.

Same cost as the default, and correct where the default is not: see
`product_marginal_concentration`."""

# --------------------------------------------------------------------------
# How the cross-lag predictor handles a zero-count source (issue #242)
# --------------------------------------------------------------------------
#
# The lag predictor is the logit of the source wave's understood proportion.
# A source of zero has no logit, so the proportion is bounded away from the
# boundary before the transform -- and *how* it is bounded is a modelling
# choice that was never registered as one.

LAG_ZERO_CLIP = "clip"
"""Clip the proportion into ``[1e-4, 1 - 1e-4]``, the historical treatment.

On an 810-item reference a zero source becomes ``logit(1e-4) = -9.21``. That
value is set by the clip, not by the data: it would be the same on a source of
zero out of 396. Seven of the 477 rows carrying a lag source have a zero source
on the current frame, and they sit at the extreme of the predictor's range,
where a regression coefficient takes its leverage from."""

LAG_ZERO_CONTINUITY = "continuity"
"""Apply a ``+0.5 / +1`` continuity correction instead of clipping.

``(u + 0.5) / (n + 1)`` is the standard Bayes/Jeffreys-style adjustment for a
boundary count. It puts a zero source at ``logit(6.17e-4) = -7.39`` rather than
at -9.21 -- nearly two logit units in, and derived from the inventory size
rather than from an arbitrary floor. Non-boundary sources move by less than
0.002 logits, so this is a boundary treatment rather than a rescaling."""

LAG_ZERO_TREATMENTS = (LAG_ZERO_CLIP, LAG_ZERO_CONTINUITY)

LAG_BASELINES = ("population", "within")
"""What the lag predictor is measured *from*, for ``lag_baseline``.

Both are defined relative to the child's understood subject intercept: ``"within"``
subtracts it, so the predictor is the child's own deviation from their own level;
``"population"`` adds it back, so the predictor is the level itself. They coincide
when there is no comprehension child effect, which is why both
``definitions.validate_model_definition`` and ``cross_lag.validate_cross_lag``
refuse that combination.

Here beside :data:`LAG_ZERO_TREATMENTS` so the definition-level check and the
engine-level one read one tuple. They were two literals with different messages,
in ``definitions.py`` and in the engine."""

SPOKEN_FALLBACK_TREATMENTS = (
    SPOKEN_FALLBACK_PRODUCT,
    SPOKEN_FALLBACK_PAIRED_ONLY,
    SPOKEN_FALLBACK_SEPARATE_DISPERSION,
    SPOKEN_FALLBACK_MOMENT_MATCHED,
)

# Floors for the moment-matched concentration. Both quantities are positive by
# construction -- E[X^2] > (E X)^2 for a non-degenerate X, and Var <= m(1-m) for
# any variable on [0, 1] -- so these only guard the arithmetic at the clipped
# extremes, never the intended regime.
_VARIANCE_FLOOR = 1e-12
_CONCENTRATION_FLOOR = 1e-6


def resolve_fallback_treatment(definition, *, field: str = "spoken_fallback") -> str:
    """Read and validate a definition's nested-outcome fallback treatment.

    Definitions written before the field existed, and engines that do not carry
    it, resolve to the historical behaviour.
    """
    treatment = getattr(definition, field, SPOKEN_FALLBACK_PRODUCT)
    if treatment not in SPOKEN_FALLBACK_TREATMENTS:
        raise ValueError(
            f"Unknown {field} {treatment!r}; expected one of "
            f"{list(SPOKEN_FALLBACK_TREATMENTS)}."
        )
    return treatment


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

    def conditional_only(self) -> NestedOutcomeSpec:
        """Return the same spec with the marginal-fallback rows removed.

        For `SPOKEN_FALLBACK_PAIRED_ONLY`. `n_parent_violations` is carried
        through unchanged rather than zeroed: the violations are a property of
        the source data, and a build report that stopped mentioning them because
        the variant had dropped them would hide the thing worth knowing.
        """
        keep = np.asarray(self.is_conditional, dtype=bool)
        return replace(
            self,
            indices=self.indices[keep],
            observed=self.observed[keep],
            trials=self.trials[keep],
            is_conditional=self.is_conditional[keep],
        )


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

    outcome_raw = df[outcome_col]
    outcome_numeric = pd.to_numeric(outcome_raw, errors="coerce")
    unparseable = outcome_raw.notna() & outcome_numeric.isna()
    if unparseable.any():
        raise ValueError(
            f"{outcome_col} contains {int(unparseable.sum())} non-numeric "
            "observed count(s)."
        )
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


# ==========================================================================
# The likelihood's alpha/beta, and the fallback treatment that shapes them
# ==========================================================================


def product_marginal_concentration(parent_p, parent_kappa, child_p, child_kappa, *, epsilon):
    """Concentration of the Beta-Binomial matching the true marginal's variance.

    The paired model draws ``theta_U ~ Beta(p_U kappa_U)`` and
    ``theta_S ~ Beta(q kappa_S)`` independently, then ``U ~ Bin(N, theta_U)``
    and ``S | U ~ Bin(U, theta_S)``. Binomial thinning collapses those two
    lines to ``S | theta_U, theta_S ~ Bin(N, theta_U theta_S)`` exactly, so the
    marginal of ``S`` is a Binomial mixed over the product of two independent
    Betas.

    That product has no Beta form, but both moments are elementary. With
    ``E[theta^2] = p (p kappa + 1) / (kappa + 1)`` for each factor and
    independence,

        m   = p_U q
        var = E[theta_U^2] E[theta_S^2] - m^2
        kappa_eff = m (1 - m) / var - 1

    is the concentration of the Beta with that mean and that variance. The
    resulting Beta-Binomial is exact in both moments where the historical
    fallback is exact only in the first, and it reduces to the historical one in
    the limit the historical one assumes: at ``kappa_U -> inf`` and ``p_U = 1``,
    ``kappa_eff = kappa_S``.

    ``kappa_eff`` is not uniformly above or below ``kappa_S``: it is below --
    the true marginal more dispersed -- exactly when ``q kappa_S > kappa_U``.
    See `SPOKEN_FALLBACK_PRODUCT`.

    ``var`` is strictly positive for any finite pair of concentrations, and
    ``var <= m(1-m)`` because the product lives on [0, 1], so ``kappa_eff >= 0``
    always. The floors below guard only the clipped extremes.
    """
    import pymc as pm

    pu = pm.math.clip(parent_p, epsilon, 1 - epsilon)
    pc = pm.math.clip(child_p, epsilon, 1 - epsilon)
    m = pu * pc
    e2_parent = pu * (pu * parent_kappa + 1.0) / (parent_kappa + 1.0)
    e2_child = pc * (pc * child_kappa + 1.0) / (child_kappa + 1.0)
    variance = pm.math.maximum(e2_parent * e2_child - m * m, _VARIANCE_FLOOR)
    return pm.math.maximum(m * (1.0 - m) / variance - 1.0, _CONCENTRATION_FLOOR)


def nested_outcome_alpha_beta(
    *,
    treatment: str,
    is_conditional,
    conditional_p,
    marginal_p,
    parent_p,
    parent_kappa,
    kappa,
    epsilon: float,
    outcome: str,
    fallback_kappa_sigma: float,
):
    """Return ``(alpha, beta)`` for a nested outcome's Beta-Binomial likelihood.

    Shared by both bivariate engines so the two graphs cannot drift: before this
    existed they carried the same eight lines twice. Under
    `SPOKEN_FALLBACK_PRODUCT` and `SPOKEN_FALLBACK_PAIRED_ONLY` the ops emitted
    are exactly the ones the engines emitted before -- op for op, so those fits
    reproduce bit-for-bit -- because the paired-only treatment is applied at data
    preparation, by dropping the rows, and never reaches the graph.

    ``is_conditional`` selects the branch per row; every argument sized by
    observation is already restricted to the outcome's own rows.
    """
    import pymc as pm

    p = pm.math.switch(is_conditional, conditional_p, marginal_p)
    p = pm.math.clip(p, epsilon, 1 - epsilon)

    if treatment == SPOKEN_FALLBACK_MOMENT_MATCHED:
        kappa_row = pm.math.switch(
            is_conditional,
            kappa,
            product_marginal_concentration(
                parent_p, parent_kappa, conditional_p, kappa, epsilon=epsilon
            ),
        )
    elif treatment == SPOKEN_FALLBACK_SEPARATE_DISPERSION:
        log_offset = pm.Normal(
            f"log_kappa_{outcome}_fallback", mu=0.0, sigma=fallback_kappa_sigma
        )
        kappa_row = kappa * pm.math.exp(
            pm.math.switch(is_conditional, 0.0, log_offset)
        )
    else:
        kappa_row = kappa

    return p * kappa_row, (1 - p) * kappa_row
