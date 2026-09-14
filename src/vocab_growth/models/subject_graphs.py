# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Construct child effects on the logit scale inside an active PyMC model.

A child has one persistent set of effects, indexed by ``subject_id``. The
returned shifts have one entry per observation row, so repeated observations
of a child use the same effects at their respective ages. Inactive effects
return scalar zero, which broadcasts over observations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from vocab_growth.models.build_utils import AgeGrids, standardize_anchor_ages
from vocab_growth.models.definitions import (
    BivariateModelDefinition,
    JointModelDefinition,
)
from vocab_growth.models.gp_utils import (
    build_child_factor,
    build_child_slope,
    build_subject_scale_of_z,
)
from vocab_growth.models.subject_effects import SubjectEffectPlan

# Matrix order also defines the suffixes of the stored child effects.
SUBJECT_RE_OUTCOMES = ("u", "q", "sign")
SUBJECT_RE_CORRELATIONS = (
    ("rho_uq", 0, 1),
    ("rho_u_sign", 0, 2),
    ("rho_sign_q", 1, 2),
)


@dataclass(frozen=True)
class BivariateChildEffects:
    """Observation shifts and optional scale functions of standardised age."""

    understood: TensorVariable | float
    spoken_ratio: TensorVariable | float
    understood_scale_of_z: Callable | None = None
    spoken_scale_of_z: Callable | None = None


@dataclass(frozen=True)
class JointChildEffects:
    """Observation shifts for comprehension and the two production ratios."""

    understood: TensorVariable | float
    spoken_ratio: TensorVariable | float
    signed_ratio: TensorVariable | float


def build_bivariate_child_effects(
    definition: BivariateModelDefinition,
    plan: SubjectEffectPlan,
    *,
    age_obs_months: np.ndarray,
    subject_obs: TensorVariable | None,
    X_all_z_data: TensorVariable,
    grids: AgeGrids,
    X_obs_mean: float,
    X_obs_std: float,
) -> BivariateChildEffects:
    """Build the resolved intercept, slope, age-scale or factor structure.

    ``age_obs_months`` and ``subject_obs`` each have one entry per observation.
    Age-scale functions accept standardised ages; child slopes use years from
    the plan's reference age. Stored variable names remain shared with prediction.
    """
    use_subject_re_u = plan["u"].is_active
    use_subject_re_q = plan["q"].is_active
    i_obs0, i_obs1 = grids.i_obs
    spec_u = plan["u"].age_varying
    spec_q = plan["q"].age_varying
    slope_u = plan["u"].slope
    slope_q = plan["q"].slope
    slope_ref_age = plan.slope_ref_age_months
    corr_eta = plan.correlation_eta
    if plan.factor is not None:
        factor_shift_u, factor_shift_q, _, _ = build_child_factor(
            plan.factor,
            tau0_u_sigma=definition.tau_subj_u_sigma,
            tau0_q_sigma=definition.tau_subj_q_sigma,
            age_obs_months=age_obs_months,
            subject_obs=subject_obs,
        )
    else:
        factor_shift_u = factor_shift_q = None

    tau_u_of_z = tau_q_of_z = None
    z_obs_raw = (
        X_all_z_data[i_obs0:i_obs1, 0]
        if (spec_u is not None or spec_q is not None)
        else None
    )

    if use_subject_re_u:
        if factor_shift_u is not None:
            subject_shift_u = factor_shift_u
        elif slope_u is not None:
            subject_shift_u, _ = build_child_slope(
                slope_u,
                age_obs_months=age_obs_months,
                subject_obs=subject_obs,
                ref_age_months=slope_ref_age,
                name="tau_subj_u",
            )
        elif spec_u is None:
            tau_subj_u = pm.HalfNormal("tau_subj_u", sigma=definition.tau_subj_u_sigma)
            delta_subj_u_raw = pm.Normal(
                "delta_subj_u_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            delta_subj_u = pm.Deterministic(
                "delta_subj_u", tau_subj_u * delta_subj_u_raw, dims="subject_id"
            )
            subject_shift_u = delta_subj_u[subject_obs]
        else:
            tau_u_of_z, tau_subj_u_young = build_subject_scale_of_z(
                spec_u,
                anchor_z=standardize_anchor_ages(
                    spec_u.anchor_ages,
                    X_obs_mean=X_obs_mean,
                    X_obs_std=X_obs_std,
                ),
                name="tau_subj_u",
            )
            delta_subj_u_raw = pm.Normal(
                "delta_subj_u_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            _ = pm.Deterministic(
                "delta_subj_u",
                tau_subj_u_young * delta_subj_u_raw,
                dims="subject_id",
            )
            subject_shift_u = tau_u_of_z(z_obs_raw) * delta_subj_u_raw[subject_obs]
    else:
        subject_shift_u = 0.0

    if use_subject_re_q:
        if factor_shift_q is not None:
            subject_shift_q = factor_shift_q
        elif slope_q is not None:
            subject_shift_q, _ = build_child_slope(
                slope_q,
                age_obs_months=age_obs_months,
                subject_obs=subject_obs,
                ref_age_months=slope_ref_age,
                name="tau_subj_q",
            )
        elif spec_q is None:
            tau_subj_q = pm.HalfNormal("tau_subj_q", sigma=definition.tau_subj_q_sigma)
            delta_subj_q_raw = pm.Normal(
                "delta_subj_q_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            if corr_eta is None:
                delta_subj_q_value = tau_subj_q * delta_subj_q_raw
            else:
                # Share the first normal coordinate; rho = 0 gives independent offsets.
                rho_raw = pm.Beta("rho_uq_raw", alpha=corr_eta, beta=corr_eta)
                rho_uq = pm.Deterministic("rho_uq", 2.0 * rho_raw - 1.0)
                delta_subj_q_value = tau_subj_q * (
                    rho_uq * delta_subj_u_raw
                    + pm.math.sqrt(1.0 - rho_uq**2) * delta_subj_q_raw
                )
            delta_subj_q = pm.Deterministic(
                "delta_subj_q", delta_subj_q_value, dims="subject_id"
            )
            subject_shift_q = delta_subj_q[subject_obs]
        else:
            tau_q_of_z, tau_subj_q_young = build_subject_scale_of_z(
                spec_q,
                anchor_z=standardize_anchor_ages(
                    spec_q.anchor_ages,
                    X_obs_mean=X_obs_mean,
                    X_obs_std=X_obs_std,
                ),
                name="tau_subj_q",
            )
            delta_subj_q_raw = pm.Normal(
                "delta_subj_q_raw", mu=0.0, sigma=1.0, dims="subject_id"
            )
            _ = pm.Deterministic(
                "delta_subj_q",
                tau_subj_q_young * delta_subj_q_raw,
                dims="subject_id",
            )
            subject_shift_q = tau_q_of_z(z_obs_raw) * delta_subj_q_raw[subject_obs]
    else:
        subject_shift_q = 0.0

    return BivariateChildEffects(
        understood=subject_shift_u,
        spoken_ratio=subject_shift_q,
        understood_scale_of_z=tau_u_of_z,
        spoken_scale_of_z=tau_q_of_z,
    )


def build_joint_child_effects(
    definition: JointModelDefinition,
    subject_obs: TensorVariable | None,
) -> JointChildEffects:
    """Build independent or correlated child intercepts for three trajectories."""
    use_subject_re_u = definition.use_subject_re_u
    use_subject_re_q = definition.use_subject_re_q
    use_subject_re_sign = definition.use_subject_re_sign

    def subject_shift(flag, tau_sigma, suffix):
        if not flag:
            return 0.0
        tau = pm.HalfNormal(f"tau_subj_{suffix}", sigma=tau_sigma)
        z = pm.Normal(f"z_subj_{suffix}", 0.0, 1.0, dims="subject_id")
        delta = pm.Deterministic(f"delta_subj_{suffix}", tau * z, dims="subject_id")
        return delta[subject_obs]

    subject_re_correlation_eta = getattr(definition, "subject_re_correlation_eta", None)
    if subject_re_correlation_eta is None:
        subject_shift_u = subject_shift(
            use_subject_re_u, definition.tau_subj_u_sigma, "u"
        )
        subject_shift_q = subject_shift(
            use_subject_re_q, definition.tau_subj_q_sigma, "q"
        )
        subject_shift_sign = subject_shift(
            use_subject_re_sign, definition.tau_subj_sign_sigma, "sign"
        )
    else:
        # LKJCholeskyCov preserves the three scale priors. The density and
        # correlation marginals are checked in test_joint_correlated_subject_re.py.
        sd_dist = pm.HalfNormal.dist(
            sigma=[
                definition.tau_subj_u_sigma,
                definition.tau_subj_q_sigma,
                definition.tau_subj_sign_sigma,
            ],
            shape=len(SUBJECT_RE_OUTCOMES),
        )
        subject_re_chol, subject_re_corr, subject_re_stds = pm.LKJCholeskyCov(
            "subject_re",
            eta=subject_re_correlation_eta,
            n=len(SUBJECT_RE_OUTCOMES),
            sd_dist=sd_dist,
            compute_corr=True,
            store_in_trace=False,
        )

        for position, suffix in enumerate(SUBJECT_RE_OUTCOMES):
            _ = pm.Deterministic(f"tau_subj_{suffix}", subject_re_stds[position])

        z_subj = [
            pm.Normal(f"z_subj_{suffix}", 0.0, 1.0, dims="subject_id")
            for suffix in SUBJECT_RE_OUTCOMES
        ]
        # The covariance factor already contains the scales. Do not apply them twice.
        subject_deviations = pt.stack(z_subj, axis=1) @ subject_re_chol.T

        subject_shifts = []
        for position, suffix in enumerate(SUBJECT_RE_OUTCOMES):
            delta = pm.Deterministic(
                f"delta_subj_{suffix}",
                subject_deviations[:, position],
                dims="subject_id",
            )
            subject_shifts.append(delta[subject_obs])
        subject_shift_u, subject_shift_q, subject_shift_sign = subject_shifts

        for name, row, column in SUBJECT_RE_CORRELATIONS:
            _ = pm.Deterministic(name, subject_re_corr[row, column])

    return JointChildEffects(
        understood=subject_shift_u,
        spoken_ratio=subject_shift_q,
        signed_ratio=subject_shift_sign,
    )
