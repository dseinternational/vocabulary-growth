# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Study offsets with an explicit choice of population reference."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def informed_studies(study_codes: np.ndarray, *index_arrays: np.ndarray) -> np.ndarray:
    """Study indices with at least one likelihood row for an outcome."""
    present = [study_codes[index] for index in index_arrays if len(index)]
    return (
        np.unique(np.concatenate(present).astype(int))
        if present
        else np.zeros(0, dtype=int)
    )


def zero_sum_study_offsets(
    name: str,
    *,
    scale,
    n_studies: int,
    raw_name: str | None = None,
    informed: np.ndarray | None = None,
    centred: bool = False,
):
    """Build a length-``n_studies`` vector of logit offsets.

    ``informed=None`` constrains all retained studies. Supplying indices instead
    constrains only those studies and sets the remaining offsets to zero. These
    choices define different priors; the caller must choose the intended
    population reference from the likelihood rows it actually uses.

    For K >= 2 constrained studies, each offset has conditional prior variance
    ``scale**2`` and each pair has correlation ``-1/(K-1)``. With one study the
    constraint forces its offset to zero, and no study contrast is estimable.
    The scale then retains its prior. Existing trace names are caller supplied.
    """
    if n_studies < 1:
        raise ValueError("Study offsets need at least one retained study.")
    k = n_studies if informed is None else len(informed)
    if k < 2:
        return pm.Deterministic(name, pt.zeros(n_studies), dims="study_id")
    sigma = float(np.sqrt(k / (k - 1)))
    if centred:
        if informed is not None:
            raise ValueError("Centred offsets currently require all retained studies.")
        return pm.ZeroSumNormal(name, sigma=scale * sigma, dims="study_id")
    raw_name = raw_name or f"{name}_raw"
    if informed is None:
        raw = pm.ZeroSumNormal(raw_name, sigma=sigma, dims="study_id")
        offsets = scale * raw
    else:
        raw = pm.ZeroSumNormal(raw_name, sigma=sigma, shape=k)
        offsets = pt.set_subtensor(pt.zeros(n_studies)[informed], scale * raw)
    return pm.Deterministic(name, offsets, dims="study_id")
