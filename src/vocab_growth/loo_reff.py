# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PSIS-LOO's relative efficiency, pinned to the sampled parameters.

ArviZ's ``loo`` derives its relative efficiency ``reff`` — the factor that
scales the PSIS tail length and so shapes every Pareto-k estimate — from the
mean effective sample size over **every variable in the posterior group**
(``arviz_stats.loo.loo_helper._get_r_eff``). In these models the posterior
group has carried thousands of elements of observation-level deterministics,
so that average was dominated by derived quantities rather than by the
parameters the sampler moved: VG10 at ``dev`` gave 0.561 over everything
stored, 0.554 once the observation-sized variables stopped being stored, and
0.905 over the sampled parameters alone. A convention that moves when the
storage policy moves is not measuring the sampler.

Decision (study owner, 2026-08-23): ``reff`` is the mean ESS over the model's
free random variables — the sampled parameters — divided by the number of
draws. Every ``az.loo``/``az.compare`` call in this project goes through
:func:`sampled_parameter_reff` (or passes precomputed results to ``compare``),
and the ``sample`` stage records the parameter names in the trace
(``fit_artifacts.SAMPLED_PARAMETERS_ATTR``) so stored fits can be read the same
way without the model. A trace written before the attribute existed can still be
pinned when the caller can name the parameters (a rebuilt model); otherwise
:func:`reff_or_default` falls back to ArviZ's convention and says so.

See ``notes/202608231530-observation-deterministics-not-sampled.md`` §3 and §7.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr

from vocab_growth.fit_artifacts import read_sampled_parameters_attr


def _posterior(trace: Any) -> xr.Dataset:
    node = trace["posterior"] if not isinstance(trace, xr.Dataset) else trace
    return node.to_dataset() if hasattr(node, "to_dataset") else node


def sampled_parameter_names(trace: Any, *, names: Sequence[str] | None = None) -> list[str]:
    """The sampled parameters of ``trace``: ``names`` if given, else its attribute.

    Raises ``LookupError`` when neither is available — a trace from before the
    attribute existed, read without a model to name the parameters.
    """
    if names is not None:
        return list(names)
    recorded = read_sampled_parameters_attr(trace)
    if recorded is None:
        raise LookupError(
            "The trace does not record its sampled parameters (written before "
            "2026-08-23) and none were supplied; pass names=[rv.name for rv in "
            "model.free_RVs] from a rebuilt model, or accept ArviZ's default."
        )
    return recorded


def sampled_parameter_reff(trace: Any, *, names: Sequence[str] | None = None) -> float:
    """Mean ESS over the sampled parameters divided by the number of draws.

    The same quantity ``arviz_stats`` computes by default, restricted to the
    model's free random variables: 1.0 for a single chain (as ArviZ), else
    ``mean(ess_mean over every element of every named variable) / n_samples``.
    Every named variable must be in the posterior; a compacted trace keeps the
    free random variables, so this holds for every tier.
    """
    import arviz_stats  # noqa: F401  (registers the ``azstats`` accessor)

    posterior = _posterior(trace)
    wanted = sampled_parameter_names(trace, names=names)
    missing = [name for name in wanted if name not in posterior.data_vars]
    if missing:
        raise KeyError(f"Sampled parameters absent from the posterior: {missing}")
    if posterior.sizes.get("chain", 1) == 1:
        return 1.0
    n_samples = int(posterior.sizes["chain"] * posterior.sizes["draw"])
    ess = posterior[wanted].azstats.ess(method="mean")
    values = np.hstack([ess[name].values.ravel() for name in ess.data_vars])
    return float(values.mean() / n_samples)


def reff_or_default(
    trace: Any, *, names: Sequence[str] | None = None, label: str = "", warn=print
) -> float | None:
    """``sampled_parameter_reff`` where it can be pinned, else ``None`` and a notice.

    ``None`` hands ``az.loo`` its default — the posterior-wide average — which is
    the only option for a trace that neither records its sampled parameters nor
    comes with a model to name them. The notice is printed rather than swallowed
    because a comparison that mixes the two conventions should say so.
    """
    try:
        return sampled_parameter_reff(trace, names=names)
    except LookupError:
        warn(
            f"  {label + ': ' if label else ''}reff left at ArviZ's posterior-wide "
            "default — the trace predates the sampled-parameters record and no "
            "model was supplied to pin it."
        )
        return None
