# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Sign/speech overlap within understood words and conditional cell likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import dse_research_utils.math.constants as math_constants
import pymc as pm
from pytensor.tensor.variable import TensorVariable

from vocab_growth.models.observation_arrays import JointObservations
from vocab_growth.models.study_effects import informed_studies, zero_sum_study_offsets

EPSILON = math_constants.EPSILON


def plackett_pi_both(r, q, psi):
    """P(both | understood) under a Plackett copula with odds ratio psi.

    Closed-form root, continuous at independence (psi == 1), then clipped to
    the Frechet bounds [max(0, r+q-1), min(r, q)].
    """
    # Rationalising (S - disc) / (2 * (psi - 1)) avoids cancellation near
    # independence. This equivalent form gives r*q at psi = 1 without 0/0.
    S = 1.0 + (r + q) * (psi - 1.0)
    disc = pm.math.sqrt(pm.math.maximum(S * S - 4.0 * psi * (psi - 1.0) * r * q, 0.0))
    pi_both = 2.0 * psi * r * q / pm.math.maximum(S + disc, 1e-12)
    lo = pm.math.maximum(0.0, r + q - 1.0)
    hi = pm.math.minimum(r, q)
    return pm.math.clip(pi_both, lo, hi)


def composition_probabilities(r, q, psi):
    """Four Plackett cell probabilities in neither/sign/speech/both order.

    Floor and normalise once so every marginal and conditional composition
    uses the same Dirichlet parameters, including near the probability limits.
    """
    both = plackett_pi_both(r, q, psi)
    cells = pm.math.stack([1 - r - q + both, r - both, q - both, both], axis=1)
    cells = pm.math.maximum(cells, EPSILON)
    return cells / cells.sum(axis=1, keepdims=True)


@dataclass(frozen=True)
class CompositionParameters:
    """Association odds ratio, total concentration and per-row log association."""

    association: TensorVariable
    concentration: TensorVariable
    log_association_obs: TensorVariable


def build_composition_parameters(
    config, observations: JointObservations
) -> CompositionParameters:
    """Build association and concentration priors for the joint model.

    The population log association is centred over studies contributing observed
    cell counts. Studies without such counts have zero association offsets.
    """
    log_psi = config.log_psi_dist.to_pymc("log_psi")
    psi = pm.Deterministic("psi", pm.math.exp(log_psi))
    informed = informed_studies(
        observations.study_codes,
        observations.idx_cells,
        observations.idx_prod,
    )
    tau_psi = (
        pm.HalfNormal("tau_psi", sigma=config.tau_psi_sigma)
        if len(informed) > 1
        else 0.0
    )
    delta_psi = zero_sum_study_offsets(
        "delta_psi",
        scale=tau_psi,
        n_studies=observations.n_studies,
        raw_name="z_psi",
        informed=informed,
    )
    pm.Deterministic("psi_study", pm.math.exp(log_psi + delta_psi), dims="study_id")
    log_psi_obs = log_psi + delta_psi[observations.study_codes]
    log_conc = config.log_conc_dist.to_pymc("log_conc")
    conc = pm.Deterministic("conc", pm.math.exp(log_conc))
    return CompositionParameters(psi, conc, log_psi_obs)


def build_composition_likelihood(
    observations: JointObservations,
    parameters: CompositionParameters,
    *,
    signed_ratio: TensorVariable,
    spoken_ratio: TensorVariable,
) -> None:
    """Observe four cells within understood, or three conditional on produced.

    Ratios include population and study terms, plus the configured lag term for
    speech. They exclude direct child offsets, as the joint model specifies.
    """
    r_c = pm.math.clip(signed_ratio[observations.idx_cells], EPSILON, 1 - EPSILON)
    q_c = pm.math.clip(spoken_ratio[observations.idx_cells], EPSILON, 1 - EPSILON)
    psi_c = pm.math.exp(parameters.log_association_obs[observations.idx_cells])
    pi_stack = composition_probabilities(r_c, q_c, psi_c)
    pm.DirichletMultinomial(
        "cells_obs",
        n=observations.cell_total,
        a=parameters.concentration * pi_stack,
        observed=observations.cell_counts,
        dims=("obs_cells_id", "cell_id"),
    )

    # Conditioning on produced drops the neither cell, retaining the other
    # Dirichlet parameters. Their sum is conc * P(produced | understood).
    # Restoring the original concentration would change the probability model.
    if observations.idx_prod.size:
        r_p = pm.math.clip(signed_ratio[observations.idx_prod], EPSILON, 1 - EPSILON)
        q_p = pm.math.clip(spoken_ratio[observations.idx_prod], EPSILON, 1 - EPSILON)
        psi_p = pm.math.exp(parameters.log_association_obs[observations.idx_prod])
        pi_prod = composition_probabilities(r_p, q_p, psi_p)[:, 1:]
        pm.DirichletMultinomial(
            "nz_prod_cells_obs",
            n=observations.prod_total,
            a=parameters.concentration * pi_prod,
            observed=observations.prod_counts,
            dims=("obs_prod_id", "prod_cell_id"),
        )
