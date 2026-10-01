#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a Gompertz mean predict as well as the trend-plus-HSGP mean? (#330)

``notes/202609101230-gompertz-comparison.md`` measured how much of our fitted
**shape** a Gompertz curve can express. It could not say whether a Gompertz
*mean function* inside our likelihood would predict the data as well as the
logit-linear trend plus HSGP does. This harness is the arm that answers that: it
swaps the mean function and holds everything else fixed.

The Gompertz arm replaces ``expit(trend + GP)`` by

    p_ij = expit(logit(A * exp(-exp(-k_g * (age - T_i)))) + study_j + child_i)

with the asymptote share ``A`` free (as a share of ``n_trials``). The
composition on the logit scale is the registered one -- the population latent is
the Gompertz logit, and the study effects, child effects, sex covariate and
per-study terms the definition declares are added to it exactly as they are to
``trend + GP``. The Beta-Binomial likelihood with its age-varying dispersion, the
two-anchor ``kappa``, the random-effect scales, the analysis frame, the sampling
configuration and the seed are untouched, because the graph is the registered
engine's own: the harness replaces the engine module's ``trend_and_gp`` for the
duration of the build and restores it afterwards. Nothing in
``src/vocab_growth/`` changes, so no fit of record is restaled.

Curves and arms
---------------

``vg12`` (TD understood), ``vg11`` (TD spoken) and ``vg20`` (DS understood and
spoken, one joint fit), each with a ``baseline`` arm (the registered graph) and a
``gompertz`` arm. Both arms are fitted here, at one configuration, into the
experiment root; the models of record are never read for scoring because their
sampling tier differs.

VG20's spoken mean is ``p_u * q``. ``--ratio-mean`` decides what happens to the
production-ratio mean ``q``:

* ``gompertz`` (default) -- a second free-asymptote Gompertz replaces the ratio's
  trend plus HSGP as well, so the arm is parametric on both outcomes. Without
  this the spoken comparison would still carry a 16-coefficient HSGP and would
  test only how comprehension's mean propagates into speech. The ratio rises
  from about 0.02 at 12 months to 0.76 at 72 in the VG20 fit of record, a
  saturating curve the family can express, and the forward projection
  (``notes/202609091900``) already fitted a Gompertz to it.
* ``flexible`` -- the Gompertz replaces the comprehension mean only. A cheap
  decomposition arm: if the default arm loses on speech, this one says whether
  the comprehension mean or the ratio mean is responsible.

Priors
------

``k_g ~ LogNormal``, ``T_i ~ Normal`` (months) and ``A ~ Beta``, per curve, in
:data:`PRIORS`. They were set by the ``prior`` subcommand's prior-predictive
check, recorded in ``notes/202610011200-gompertz-mean-harness.md``. The
registered trend anchors (``p_slope_low``/``p_slope_hi``) have no counterpart
and are not used by the Gompertz arm. The Gompertz is bounded above by ``A``, so
the registered mean clamp -- which stops a *linear* trend extrapolating -- and
the GP anchor have nothing to act on there and are ignored.

Subcommands
-----------

``prior``   prior-predictive summaries for one curve, both arms.
``fit``     fit one curve's arm.
``compare`` paired PSIS-LOO per outcome on the same rows, with Pareto-k counts,
            the Gompertz parameters and the population curves of both arms.
``loso``    leave-one-subject-out K-fold for VG20, both arms, through the fold
            machinery of ``scripts/kfold_loso.py``; per-subject ELPDs for both
            outcomes together and for each one alone.

Output root
-----------

Everything goes under ``<output-root>/experiments/gompertz-mean/``, where
``<output-root>`` is ``--output-root`` or, by default, the project's resolved
output root. The harness refuses to run if that destination resolves inside a
canonical ``models/`` directory, so no model of record can be overwritten.

Usage::

    uv run python scripts/experiments/gompertz_mean_arm.py prior vg20
    uv run python scripts/experiments/gompertz_mean_arm.py fit vg20 baseline --config test
    uv run python scripts/experiments/gompertz_mean_arm.py fit vg20 gompertz --config test
    uv run python scripts/experiments/gompertz_mean_arm.py compare --config test
    uv run python scripts/experiments/gompertz_mean_arm.py loso --config test --folds 5

The one-off harness conventions of ``scripts/experiments/README.md`` apply.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

#: Where the experiment lives under an output root.
EXPERIMENT_SUBDIR = ("experiments", "gompertz-mean")

#: Curve key -> registered model id.
CURVES = {"vg12": "VG12", "vg11": "VG11", "vg20": "VG20"}
ARMS = ("baseline", "gompertz")
RATIO_MEANS = ("gompertz", "flexible")
DEFAULT_RATIO_MEAN = "gompertz"

#: The registered mean function's free parameters, as name prefixes before the
#: outcome suffix: the trend's two anchors, the HSGP's length scale and
#: amplitude, and the HSGP basis coefficients (``g_unit<suffix>_hsgp_coeffs``).
MEAN_FUNCTION_PREFIXES = ("p_slope_low", "p_slope_hi", "ell_unit", "eta", "g_unit")
#: The Gompertz mean's free parameters, before the outcome suffix.
GOMPERTZ_PARAMETERS = ("k_g", "T_i", "A_share")

#: Likelihood variable(s) and their outcome labels, per curve.
OUTCOMES = {
    "vg12": (("understood", "y_obs"),),
    "vg11": (("spoken", "y_obs"),),
    "vg20": (("understood", "y_u_obs"), ("spoken", "y_s_obs")),
}
#: Population curve deterministics on the query grid, per curve.
QUERY_CURVES = {
    "vg12": (("understood", "p_query"),),
    "vg11": (("spoken", "p_query"),),
    "vg20": (("understood", "p_u_query"), ("spoken", "p_s_query"), ("ratio", "q_query")),
}
#: Age bands for the row-level prior-predictive check.
AGE_BANDS = {
    "td": ((8, 12), (12, 16), (16, 20), (20, 25), (25, 31)),
    "ds": ((8, 24), (24, 36), (36, 48), (48, 60), (60, 72), (72, 116)),
}
QUANTILES = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class GompertzPrior:
    """``k_g ~ LogNormal(log k_median, k_log_sd)``, ``T_i ~ Normal(t_mean, t_sd)``,
    ``A ~ Beta(a_alpha, a_beta)``. ``k_g`` is per month and ``T_i`` in months."""

    k_median: float
    k_log_sd: float
    t_mean: float
    t_sd: float
    a_alpha: float
    a_beta: float

    def describe(self) -> str:
        return (
            f"k_g ~ LogNormal(log {self.k_median:g}, {self.k_log_sd:g}); "
            f"T_i ~ Normal({self.t_mean:g}, {self.t_sd:g}); "
            f"A ~ Beta({self.a_alpha:g}, {self.a_beta:g})"
        )


#: Priors per mean-function slot, set by the prior-predictive check recorded in
#: notes/202610011200-gompertz-mean-harness.md (§3). The centres follow the
#: issue: Day et al. (2025) for typically-developing production; our own pooled
#: least-squares and free-asymptote fits (notes/202609101230 §§4-5) for
#: typically-developing comprehension, which the paper does not model, and for
#: the Down syndrome curves. The spreads are what the check settled on.
PRIORS: dict[str, GompertzPrior] = {
    "vg12": GompertzPrior(k_median=0.14, k_log_sd=0.4, t_mean=16.0, t_sd=3.0, a_alpha=6.0, a_beta=4.0),
    "vg11": GompertzPrior(k_median=0.15, k_log_sd=0.35, t_mean=23.3, t_sd=3.0, a_alpha=16.0, a_beta=4.0),
    "vg20_u": GompertzPrior(k_median=0.04, k_log_sd=0.5, t_mean=38.0, t_sd=10.0, a_alpha=7.0, a_beta=3.0),
    "vg20_q": GompertzPrior(k_median=0.07, k_log_sd=0.5, t_mean=42.0, t_sd=10.0, a_alpha=9.0, a_beta=2.0),
}


# ---------------------------------------------------------------------------
# The Gompertz mean
# ---------------------------------------------------------------------------


def gompertz_share(age, a_share, k_g, t_i):
    """``A * exp(-exp(-k_g (age - T_i)))``: the expected share of the inventory."""
    age = np.asarray(age, dtype=float)
    return a_share * np.exp(-np.exp(-k_g * (age - t_i)))


def _log1mexp(x):
    """``log(1 - exp(x))`` for ``x < 0``, stable at both ends (Maechler 2012)."""
    x = np.asarray(x, dtype=float)
    small = x > -np.log(2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            small, np.log(-np.expm1(np.where(small, x, -1.0))), np.log1p(-np.exp(x))
        )


def gompertz_logit(age, a_share, k_g, t_i):
    """``logit(gompertz_share(...))``, computed on the log scale.

    ``log p = log A - exp(-k_g (age - T_i))``, and ``logit p = log p - log(1 - p)``
    with ``log(1 - p)`` from :func:`_log1mexp`, so a share of 1e-30 at a young
    age is a finite logit rather than ``-inf``.
    """
    age = np.asarray(age, dtype=float)
    log_p = np.log(a_share) - np.exp(-k_g * (age - t_i))
    return log_p - _log1mexp(log_p)


def gompertz_logit_pt(age, a_share, k_g, t_i):
    """The PyTensor form of :func:`gompertz_logit`, used in the graph."""
    import pytensor.tensor as pt

    log_p = pt.log(a_share) - pt.exp(-k_g * (age - t_i))
    return log_p - pt.log1mexp(log_p)


def age_scale_from_anchors(slope_anchors, grid) -> tuple[float, float]:
    """Recover the engine's age standardisation from the slope anchors.

    The engine standardises ages by the observed mean and standard deviation and
    hands ``trend_and_gp`` only the standardised grid, plus the standardised
    slope anchors. The anchors are known in months, so the two values are exact:
    ``std = (b - a) / (sb_z - sa_z)`` and ``mean = a - sa_z * std``.
    """
    a, b = (float(v) for v in slope_anchors)
    std = (b - a) / (float(grid.sb_z) - float(grid.sa_z))
    return a - float(grid.sa_z) * std, std


def gompertz_mean(
    *,
    prior: GompertzPrior,
    suffix: str,
    X_all_z_data,
    x_mean: float,
    x_std: float,
    latent_name: str | None,
    store_deterministic: bool,
):
    """The Gompertz logit on the engine's full age grid, in place of trend + GP."""
    import pymc as pm

    k_g = pm.LogNormal(f"k_g{suffix}", mu=float(np.log(prior.k_median)), sigma=prior.k_log_sd)
    t_i = pm.Normal(f"T_i{suffix}", mu=prior.t_mean, sigma=prior.t_sd)
    a_share = pm.Beta(f"A_share{suffix}", alpha=prior.a_alpha, beta=prior.a_beta)
    age_months = X_all_z_data[:, 0] * x_std + x_mean
    latent = gompertz_logit_pt(age_months, a_share, k_g, t_i)
    if store_deterministic:
        return pm.Deterministic(latent_name, latent, dims=("all_id",))
    return latent


def mean_slots(curve: str, ratio_mean: str = DEFAULT_RATIO_MEAN) -> dict[str, str | None]:
    """``trend_and_gp`` suffix -> :data:`PRIORS` slot, or ``None`` to keep the registered mean."""
    if ratio_mean not in RATIO_MEANS:
        raise ValueError(f"ratio_mean must be one of {RATIO_MEANS}, not {ratio_mean!r}")
    if curve in ("vg12", "vg11"):
        return {"": curve}
    if curve == "vg20":
        return {"_u": "vg20_u", "_q": "vg20_q" if ratio_mean == "gompertz" else None}
    raise ValueError(f"unknown curve {curve!r}; expected one of {sorted(CURVES)}")


@contextlib.contextmanager
def gompertz_mean_installed(
    definition,
    curve: str,
    ratio_mean: str = DEFAULT_RATIO_MEAN,
    priors: dict[str, GompertzPrior] | None = None,
) -> Iterator[dict[str, int]]:
    """Replace the engine's ``trend_and_gp`` with the Gompertz mean while open.

    The patch is on the engine *module's* global, which ``build_model_graph``
    looks up at call time, so every other line of the registered build runs
    unchanged. Yields a call counter keyed by suffix so a caller can check the
    build reached the mean function as many times as expected. A suffix the
    curve does not declare raises rather than falling through to the HSGP.
    """
    from vocab_growth.models.catalogue import engine_for_definition

    priors = PRIORS if priors is None else priors
    slots = mean_slots(curve, ratio_mean)
    module = importlib.import_module(engine_for_definition(definition).module)
    original = module.trend_and_gp
    calls: dict[str, int] = {}

    def patched(**kwargs):
        suffix = kwargs["suffix"]
        if suffix not in slots:
            raise RuntimeError(
                f"{definition.model_id}: the engine built a mean with suffix {suffix!r}, "
                f"which the {curve} Gompertz arm does not declare ({sorted(slots)})."
            )
        calls[suffix] = calls.get(suffix, 0) + 1
        slot = slots[suffix]
        if slot is None:
            return original(**kwargs)
        x_mean, x_std = age_scale_from_anchors(definition.slope_anchors, kwargs["grid"])
        return gompertz_mean(
            prior=priors[slot],
            suffix=suffix,
            X_all_z_data=kwargs["X_all_z_data"],
            x_mean=x_mean,
            x_std=x_std,
            latent_name=kwargs.get("latent_name"),
            store_deterministic=bool(kwargs["store_deterministic"]),
        )

    module.trend_and_gp = patched
    try:
        yield calls
    finally:
        module.trend_and_gp = original


def arm_tag(curve: str, arm: str, ratio_mean: str = DEFAULT_RATIO_MEAN) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, not {arm!r}")
    if arm == "gompertz" and curve == "vg20":
        return f"gompertz-ratio-{ratio_mean}"
    return arm


def arm_definition(curve: str, arm: str, ratio_mean: str = DEFAULT_RATIO_MEAN):
    """The registered definition under a short experiment-only ``config_name``.

    Only the identity changes; every statistical field is the registered one.
    The name is kept short because the fit directory sits several levels deep
    and Graphviz still fails past Windows' MAX_PATH.
    """
    from vocab_growth.models import definitions as D

    base = D.MODEL_REGISTRY[curve]
    if base.model_id != CURVES[curve]:
        raise RuntimeError(f"registry key {curve!r} holds {base.model_id}, not {CURVES[curve]}")
    tag = arm_tag(curve, arm, ratio_mean)
    return dataclasses.replace(
        base,
        config_name=f"gompertz-mean-{tag}",
        banner=f"Gompertz-mean experiment (#330): {base.model_id} arm '{tag}' -- not a model of record",
    )


def arm_dir(experiment_root: str, definition) -> str:
    return os.path.join(experiment_root, "models", f"{definition.model_id}-{definition.config_name}")


def arm_context(definition, curve: str, arm: str, ratio_mean: str):
    if arm == "gompertz":
        return gompertz_mean_installed(definition, curve, ratio_mean)
    return contextlib.nullcontext({})


# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------


def _normalised(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(path))))


def _is_within(path: str, parent: str) -> bool:
    path, parent = _normalised(path), _normalised(parent)
    try:
        return os.path.commonpath([path, parent]) == parent
    except ValueError:  # different drives on Windows
        return False


def canonical_output_roots() -> list[str]:
    """Every root whose ``models/`` holds, or could hold, models of record.

    The checkout's ``output/`` default and ``$DSE_VOCAB_GROWTH_OUTPUT_DIR`` --
    on the reporting workstation the latter is the shared root.
    """
    from vocab_growth import environment as env

    roots = [os.path.join(env.ROOT_DIR, "output")]
    configured = os.environ.get(env.OUTPUT_DIR_ENV_VAR)
    if configured:
        roots.append(configured)
    return roots


def resolve_experiment_root(output_root: str, canonical_roots: Iterable[str]) -> str:
    """``<output_root>/experiments/gompertz-mean``, refused inside any canonical ``models/``."""
    root = os.path.join(output_root, *EXPERIMENT_SUBDIR)
    for canonical in canonical_roots:
        canonical_models = os.path.join(canonical, "models")
        if _is_within(root, canonical_models):
            raise ValueError(
                f"Refusing to write the Gompertz-mean experiment to {root}: it resolves "
                f"inside the canonical models directory {canonical_models}, where a fit "
                "of record could be replaced. Pass an --output-root outside it."
            )
    return root


def install_output_root(output_root: str | None) -> str:
    """Resolve the experiment root, refuse a canonical destination, and set it."""
    from vocab_growth import environment as env

    canonical = canonical_output_roots()
    base = output_root if output_root is not None else env.output_root()
    try:
        root = resolve_experiment_root(base, canonical)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    env.set_output_root(root)
    # Defence in depth: what the pipeline will actually resolve.
    for candidate in canonical:
        if _is_within(env.models_output_dir(), os.path.join(candidate, "models")):
            raise SystemExit(f"Resolved models directory {env.models_output_dir()} is canonical.")
    print(f"[gompertz_mean_arm] experiment root: {root}")
    return root


# ---------------------------------------------------------------------------
# Prior-predictive check
# ---------------------------------------------------------------------------


def _band_key(definition) -> str:
    from vocab_growth.data_utils import Population

    return "ds" if definition.population is Population.DOWN_SYNDROME else "td"


def build_context(definition, curve: str, arm: str, ratio_mean: str, output_root_dir: str):
    """prepare -> priors -> build through the definition's own engine; no sampling."""
    import dse_research_utils.statistics.models.reporting as reporting
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models.catalogue import engine_for_definition
    from vocab_growth.models.common import ModelFitContext

    engine = engine_for_definition(definition)
    context = ModelFitContext(
        report_build=False,
        reporting=reporting.ReportingConfiguration(
            model_name=definition.model_id,
            config_name=definition.config_name,
            output_root_dir=output_root_dir,
            ci_prob=0.89,
            interval_kind="eti",
        ),
        sampling=sampling.get_sampling_configuration("dev"),
    )
    os.makedirs(context.reporting.output_dir, exist_ok=True)
    with arm_context(definition, curve, arm, ratio_mean):
        engine.resolve("prepare")(context, definition)
        engine.resolve("priors")(context, definition)
        engine.resolve("build")(context, definition)
    return context


def _quantile_row(values: np.ndarray) -> dict:
    q = np.quantile(np.asarray(values, dtype=float), QUANTILES)
    return {"q05": float(q[0]), "q50": float(q[1]), "q95": float(q[2])}


def prior_predictive_tables(idata, definition, curve: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Population-curve and row-level summaries of one prior-predictive sample."""
    n_trials = float(definition.n_trials)
    prior = idata["prior"]
    ages_query = np.asarray(idata["constant_data"]["X_query"].values, dtype=float)
    curve_rows = []
    for outcome, var in QUERY_CURVES[curve]:
        p = np.asarray(prior[var].values, dtype=float).reshape(-1, len(ages_query))
        scale = 1.0 if outcome == "ratio" else n_trials
        for j, age in enumerate(ages_query):
            values = scale * p[:, j]
            row = {"arm": label, "outcome": outcome, "unit": "share" if outcome == "ratio" else "words", "age_months": float(age)}
            row.update(_quantile_row(values))
            if outcome != "ratio":
                row["share_below_1_word"] = float((values < 1.0).mean())
                row["share_above_95pct_of_n"] = float((values > 0.95 * n_trials).mean())
            curve_rows.append(row)

    observed = idata["observed_data"]
    predictive = idata["prior_predictive"]
    constant = idata["constant_data"]
    x_obs = np.asarray(constant["X_obs"].values, dtype=float)
    row_rows = []
    for outcome, var in OUTCOMES[curve]:
        y = np.asarray(observed[var].values, dtype=float)
        if curve == "vg20":
            mask = "obs_u_mask" if var == "y_u_obs" else "obs_s_mask"
            ages = x_obs[np.flatnonzero(np.asarray(constant[mask].values))]
        else:
            ages = x_obs
        rep = np.asarray(predictive[var].values, dtype=float).reshape(-1, len(y))
        lo, hi = np.quantile(rep, [0.05, 0.95], axis=0)
        inside = (y >= lo) & (y <= hi)
        for band in AGE_BANDS[_band_key(definition)]:
            in_band = (ages >= band[0]) & (ages < band[1])
            n = int(in_band.sum())
            if n == 0:
                continue
            obs_q = np.quantile(y[in_band], QUANTILES)
            rep_median = np.median(rep[:, in_band], axis=1)
            row = {
                "arm": label,
                "outcome": outcome,
                "age_band": f"{band[0]}-{band[1]}",
                "n_rows": n,
                "observed_q05": float(obs_q[0]),
                "observed_median": float(obs_q[1]),
                "observed_q95": float(obs_q[2]),
            }
            row.update({f"band_median_{k}": v for k, v in _quantile_row(rep_median).items()})
            row["row_coverage_90"] = float(inside[in_band].mean())
            row_rows.append(row)
    return pd.DataFrame(curve_rows), pd.DataFrame(row_rows)


def prior_command(args) -> int:
    import pymc as pm

    root = install_output_root(args.output_root)
    out_dir = os.path.join(root, "prior-predictive")
    os.makedirs(out_dir, exist_ok=True)
    curves, rows = [], []
    for arm in args.arms:
        definition = arm_definition(args.curve, arm, args.ratio_mean)
        label = arm_tag(args.curve, arm, args.ratio_mean)
        context = build_context(definition, args.curve, arm, args.ratio_mean, os.path.join(out_dir, "scratch"))
        var_names = [v for _, v in QUERY_CURVES[args.curve]] + [v for _, v in OUTCOMES[args.curve]]
        with context.model:
            idata = pm.sample_prior_predictive(draws=args.draws, var_names=var_names, random_seed=args.seed)
        c, r = prior_predictive_tables(idata, definition, args.curve, label)
        curves.append(c)
        rows.append(r)
    curve_df = pd.concat(curves, ignore_index=True).assign(curve=args.curve)
    row_df = pd.concat(rows, ignore_index=True).assign(curve=args.curve)
    suffix = f"{args.curve}" + (f"-ratio-{args.ratio_mean}" if args.curve == "vg20" else "")
    curve_df.to_csv(os.path.join(out_dir, f"prior_predictive_curves_{suffix}.csv"), index=False)
    row_df.to_csv(os.path.join(out_dir, f"prior_predictive_rows_{suffix}.csv"), index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    used = {slot for slot in mean_slots(args.curve, args.ratio_mean).values() if slot}
    for slot in sorted(used):
        print(f"[prior] {slot}: {PRIORS[slot].describe()}")
    print("\n=== Population curve (words, or share for the ratio), prior 5/50/95%")
    print(curve_df.round(3).to_string(index=False))
    print("\n=== Rows by age band: observed against the prior predictive")
    print(row_df.round(2).to_string(index=False))
    print(f"\nWritten to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

ARM_RECORD_FILENAME = "gompertz_mean_arm.json"


def arm_record(curve: str, arm: str, ratio_mean: str, config: str) -> dict:
    slots = mean_slots(curve, ratio_mean) if arm == "gompertz" else {}
    return {
        "issue": "https://github.com/dseinternational/vocabulary-growth/issues/330",
        "curve": curve,
        "arm": arm,
        "ratio_mean": ratio_mean if curve == "vg20" and arm == "gompertz" else None,
        "sampling_config": config,
        "priors": {
            suffix or "(univariate)": dataclasses.asdict(PRIORS[slot])
            for suffix, slot in slots.items()
            if slot is not None
        },
    }


def fit_command(args) -> int:
    import dse_research_utils.environment.setup as setup

    from vocab_growth.models.catalogue import engine_for_definition
    from vocab_growth.models.common import run_fit_pipeline
    from vocab_growth.models.exploratory import write_exploratory_marker

    root = install_output_root(args.output_root)
    definition = arm_definition(args.curve, args.arm, args.ratio_mean)
    engine = engine_for_definition(definition)
    setup.init_script()
    target = arm_dir(root, definition)
    print(f"[gompertz_mean_arm] {args.curve} arm={arm_tag(args.curve, args.arm, args.ratio_mean)} config={args.config} -> {target}")
    started = time.perf_counter()
    with arm_context(definition, args.curve, args.arm, args.ratio_mean) as calls:
        run_fit_pipeline(args.config, definition, stages=engine.resolve("stages")(definition))
    if args.arm == "gompertz":
        print(f"[gompertz_mean_arm] mean-function builds by suffix: {calls}")
    write_exploratory_marker(
        target,
        model_label=f"{definition.model_id} Gompertz-mean experiment arm '{arm_tag(args.curve, args.arm, args.ratio_mean)}'",
        note=(
            "Exploratory mean-function comparison (issue #330): the registered graph, or "
            "the registered graph with its trend + HSGP mean replaced by a free-asymptote "
            "Gompertz through an experiment-local patch. The manifest's definition and code "
            "signature do not describe the patch. Not a model of record."
        ),
    )
    record = arm_record(args.curve, args.arm, args.ratio_mean, args.config)
    record["wall_seconds"] = time.perf_counter() - started
    with open(os.path.join(target, ARM_RECORD_FILENAME), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    return 0


# ---------------------------------------------------------------------------
# Compare (PSIS-LOO)
# ---------------------------------------------------------------------------


class FittedArm:
    def __init__(self, root: str, curve: str, arm: str, ratio_mean: str, config: str):
        import arviz as az

        self.curve, self.arm = curve, arm
        self.tag = arm_tag(curve, arm, ratio_mean)
        self.definition = arm_definition(curve, arm, ratio_mean)
        self.dir = arm_dir(root, self.definition)
        if not os.path.isfile(os.path.join(self.dir, "trace.nc")):
            raise FileNotFoundError(f"no finished fit for {curve} arm {self.tag!r} at {self.dir}")
        with open(os.path.join(self.dir, "fit_manifest.json"), encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        record_path = os.path.join(self.dir, ARM_RECORD_FILENAME)
        with open(record_path, encoding="utf-8") as fh:
            self.record = json.load(fh)
        expected = arm_record(curve, arm, ratio_mean, config)
        for key in ("priors", "sampling_config", "ratio_mean"):
            if self.record.get(key) != expected[key]:
                raise RuntimeError(
                    f"{self.dir} was fitted with {key}={self.record.get(key)!r}, but this "
                    f"comparison asks for {expected[key]!r}. Refit, or compare at its settings."
                )
        self.idata = az.from_netcdf(os.path.join(self.dir, "trace.nc"))

    def convergence(self) -> dict:
        from vocab_growth.sensitivity.compare import diagnostics_gate

        gate = diagnostics_gate(self.dir)
        return {
            "divergences": int(self.idata["sample_stats"]["diverging"].values.sum()),
            "max_r_hat": gate.max_rhat,
            "min_ess_bulk": gate.min_ess,
            "converged": gate.converged,
            "caveats": "; ".join(gate.caveats),
        }


def paired_loo(baseline: FittedArm, gompertz: FittedArm, var: str) -> dict:
    """PSIS-LOO of both arms on the same rows; difference is Gompertz minus baseline.

    A spoken row with zero observed comprehension has a constant log-likelihood
    and is dropped from both arms by one shared mask, so the pointwise scores
    stay paired (as ``vg20_sex_arm.loo_table`` does).
    """
    import arviz as az
    import dse_research_utils.statistics.loo as shared_loo

    from vocab_growth.loo_reff import sampled_parameter_reff

    arms = {"baseline": baseline, "gompertz": gompertz}
    das = {name: arm.idata["log_likelihood"][var] for name, arm in arms.items()}
    sample_dims = [d for d in ("chain", "draw") if d in das["baseline"].dims]
    obs_dim = [d for d in das["baseline"].dims if d not in sample_dims][0]
    keep = np.ones(das["baseline"].sizes[obs_dim], dtype=bool)
    for da in das.values():
        if da.sizes[obs_dim] != keep.size:
            raise RuntimeError(f"{var}: the arms have different numbers of rows.")
        keep &= (da.var(dim=sample_dims) > 1e-12).values
    loos, reffs = {}, {}
    for name, arm in arms.items():
        sampled = ((arm.manifest.get("artefacts") or {}).get("trace") or {}).get("sampled_parameters")
        reffs[name] = sampled_parameter_reff(arm.idata, names=sampled)
        source = arm.idata.copy(deep=False)
        source["log_likelihood"] = source["log_likelihood"].isel({obs_dim: keep})
        loos[name] = az.loo(source, var_name=var, pointwise=True, reff=reffs[name])
    li = {name: np.asarray(loo.elpd_i.values, dtype=float).ravel() for name, loo in loos.items()}
    ks = {name: np.asarray(loo.pareto_k.values, dtype=float).ravel() for name, loo in loos.items()}
    diff = li["gompertz"] - li["baseline"]
    n = len(diff)
    row = {
        "n_rows": n,
        "n_dropped_degenerate": int(keep.size - keep.sum()),
        "elpd_baseline": float(li["baseline"].sum()),
        "elpd_gompertz": float(li["gompertz"].sum()),
        "elpd_diff_gompertz_minus_baseline": float(diff.sum()),
        "se_diff": float(np.sqrt(n * diff.var(ddof=1))),
    }
    for name in arms:
        k = ks[name]
        # The shared summary row, so p_loo and the sample-size-dependent good-k
        # threshold are read as every fit's own loo_summary.csv reads them.
        shared = shared_loo.loo_summary_row(loos[name], label=name, reff=reffs[name])
        row[f"p_loo_{name}"] = shared["p_loo"]
        row[f"reff_{name}"] = shared["reff"]
        row[f"k_threshold_{name}"] = shared["k_threshold"]
        row[f"pareto_k_unusable_{name}"] = shared["pareto_k_unusable"]
        row[f"pareto_k_unusable_share_{name}"] = shared["pareto_k_unusable"] / k.size
        row[f"pareto_k_gt_0.7_{name}"] = int(((k > 0.7) | ~np.isfinite(k)).sum())
        row[f"pareto_k_max_{name}"] = float(np.nanmax(k))
    row["psis_reliable"] = bool(
        row["pareto_k_unusable_baseline"] == 0 and row["pareto_k_unusable_gompertz"] == 0
    )
    return row


def gompertz_parameter_rows(arm: FittedArm, n_trials: int) -> list[dict]:
    rows = []
    post = arm.idata["posterior"]
    for suffix in ("", "_u", "_q"):
        if f"k_g{suffix}" not in post:
            continue
        k = np.asarray(post[f"k_g{suffix}"].values, dtype=float).ravel()
        t = np.asarray(post[f"T_i{suffix}"].values, dtype=float).ravel()
        a = np.asarray(post[f"A_share{suffix}"].values, dtype=float).ravel()
        derived = {}
        if suffix != "_q":
            # Not for the ratio, which is a share of comprehension, not of n_trials.
            # The maximum rate of the population (median-child) curve is k_g A / e.
            derived["A_words"] = a * n_trials
            derived["peak_rate_words_per_month"] = k * a * n_trials / np.e
        for name, values in {"k_g": k, "T_i": t, "A_share": a, **derived}.items():
            q = np.quantile(values, (0.055, 0.5, 0.945))
            rows.append({"curve": arm.curve, "arm": arm.tag, "slot": suffix or "(univariate)", "parameter": name, "median": q[1], "lo89": q[0], "hi89": q[2]})
    return rows


def curve_rows(arm: FittedArm, n_trials: int) -> list[dict]:
    post = arm.idata["posterior"]
    ages = np.asarray(arm.idata["constant_data"]["X_query"].values, dtype=float)
    rows = []
    for outcome, var in QUERY_CURVES[arm.curve]:
        p = np.asarray(post[var].values, dtype=float).reshape(-1, len(ages))
        scale = 1.0 if outcome == "ratio" else float(n_trials)
        for j, age in enumerate(ages):
            q = np.quantile(scale * p[:, j], (0.055, 0.5, 0.945))
            rows.append({"curve": arm.curve, "arm": arm.tag, "outcome": outcome, "age_months": float(age), "median": q[1], "lo89": q[0], "hi89": q[2]})
    return rows


def compare_command(args) -> int:
    root = install_output_root(args.output_root)
    out_dir = os.path.join(root, "comparisons", "gompertz-mean")
    os.makedirs(out_dir, exist_ok=True)
    loo_rows, param_rows, curve_tab, gate_rows = [], [], [], []
    for curve in args.curves:
        try:
            base = FittedArm(root, curve, "baseline", args.ratio_mean, args.config)
            gomp = FittedArm(root, curve, "gompertz", args.ratio_mean, args.config)
        except FileNotFoundError as exc:
            print(f"[compare] skipping {curve}: {exc}")
            continue
        n_trials = int(base.definition.n_trials)
        for arm in (base, gomp):
            gate_rows.append({"curve": curve, "arm": arm.tag, "config": arm.record["sampling_config"], **arm.convergence()})
            curve_tab.extend(curve_rows(arm, n_trials))
        param_rows.extend(gompertz_parameter_rows(gomp, n_trials))
        for outcome, var in OUTCOMES[curve]:
            loo_rows.append({"curve": curve, "outcome": outcome, "gompertz_arm": gomp.tag, **paired_loo(base, gomp, var)})
    # A non-default VG20 ratio arm writes its own files rather than replacing
    # the default arm's, and the sampling tier is in every name.
    stem = f"gompertz_mean_{args.config}" + (
        "" if args.ratio_mean == DEFAULT_RATIO_MEAN else f"_ratio-{args.ratio_mean}"
    )
    tables = {
        f"{stem}_loo.csv": pd.DataFrame(loo_rows),
        f"{stem}_convergence.csv": pd.DataFrame(gate_rows),
        f"{stem}_parameters.csv": pd.DataFrame(param_rows),
        f"{stem}_curves.csv": pd.DataFrame(curve_tab),
    }
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    for filename, table in tables.items():
        if table.empty:
            continue
        table.to_csv(os.path.join(out_dir, filename), index=False)
        print(f"\n=== {filename}")
        print(table.round(3).to_string(index=False))
    print(f"\nWritten to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# Leave-one-subject-out (VG20)
# ---------------------------------------------------------------------------

LOSO_OUTCOMES = {
    "both": ("understood", "spoken"),
    "understood": ("understood",),
    "spoken": ("spoken",),
}


def load_kfold_loso():
    """``scripts/kfold_loso.py``, loaded by path; call after the output root is set."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kfold_loso.py")
    spec = importlib.util.spec_from_file_location("kfold_loso", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def loso_command(args) -> int:
    import dse_research_utils.environment.setup as setup
    import dse_research_utils.statistics.models.sampling as sampling

    from vocab_growth.models.common_bivariate_re import (
        build_bivariate_re_analysis_frame,
    )

    root = install_output_root(args.output_root)
    setup.init_script()
    kl = load_kfold_loso()
    out_dir = os.path.join(root, "comparisons", "gompertz-mean-loso")
    kl.OUT_DIR = out_dir
    kl.KFOLD_TMP_DIR = os.path.join(root, "kfold_tmp")
    os.makedirs(out_dir, exist_ok=True)

    arms = tuple(args.arms)
    definitions = {arm: arm_definition("vg20", arm, args.ratio_mean) for arm in arms}
    # VG20's own prepared frame, not kfold_loso's loader: the comparison holds
    # the analysis frame fixed, and this is the frame the fitted arms used.
    frame, _ = build_bivariate_re_analysis_frame(definitions[arms[0]])
    waves = kl.wave_index(frame)
    folds, _ = kl.stratified_subject_folds(frame, K=args.folds)
    sampling_cfg = sampling.get_sampling_configuration(args.config)
    print(f"[loso] VG20 frame: {len(frame)} rows, {frame['subject_code'].nunique()} children; K={args.folds}; config={args.config}; arms={arms}")

    elpds = {key: {arm: {} for arm in arms} for key in LOSO_OUTCOMES}
    records = []
    for k, fold in enumerate(folds):
        df = kl.build_fold_frame(frame, fold, waves, "subject", "all-outcomes")
        holdout = df["holdout"].to_numpy()
        for arm in arms:
            definition = definitions[arm]
            started = time.perf_counter()
            with arm_context(definition, "vg20", arm, args.ratio_mean):
                trace, _, gate = kl.fit_fold(definition, df, sampling_cfg, f"gm-{arm}-f{k}")
            for key, outcomes in LOSO_OUTCOMES.items():
                elpds[key][arm].update(kl.holdout_subject_elpds(df, trace, fold, outcomes=outcomes))
            record = kl.FoldFitRecord(
                model_short=arm_tag("vg20", arm, args.ratio_mean),
                fold=k,
                n_holdout_subjects=len(fold),
                n_holdout_obs_u=int((df["understood"].notna() & holdout).sum()),
                n_holdout_obs_s=int((df["spoken"].notna() & holdout).sum()),
                wall_seconds=time.perf_counter() - started,
                **kl.fold_gate_fields(gate),
            )
            records.append(record)
            print(f"[loso] fold {k} {record.model_short}: {record.wall_seconds:.0f}s, gate {'PASS' if record.passed else 'FAIL'}")
            del trace

    flags = {arm: all(r.passed for r in records if r.model_short == arm_tag("vg20", arm, args.ratio_mean)) for arm in arms}
    pd.set_option("display.width", 220)
    for key in LOSO_OUTCOMES:
        rows = [
            {"model": arm, "subject_code": code, "elpd": value}
            for arm in arms
            for code, value in elpds[key][arm].items()
        ]
        table = pd.DataFrame(rows).pivot_table(index="subject_code", columns="model", values="elpd")
        table.to_csv(os.path.join(out_dir, f"loso_subject_elpds_{key}.csv"))
        summary = kl.summarise_models(table, arms, flags)
        summary.to_csv(os.path.join(out_dir, f"loso_summary_{key}.csv"), index=False)
        print(f"\n=== LOSO, {key}")
        print(summary.to_string(index=False))
        if len(arms) == 2:
            pairs = kl.pairwise_compare(table, arms, flags)
            pairs.to_csv(os.path.join(out_dir, f"loso_compare_{key}.csv"), index=False)
            print(pairs.to_string(index=False))
    fits = pd.DataFrame([r.__dict__ for r in records])
    fits.to_csv(os.path.join(out_dir, "loso_fits.csv"), index=False)
    print("\n=== Fold fits")
    print(fits.to_string(index=False))
    print(f"\nWritten to {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--output-root",
        default=None,
        help="Base output root (default: the project's resolved root). The experiment "
        "writes under <root>/experiments/gompertz-mean/ and refuses a canonical models/ destination.",
    )
    ap.add_argument(
        "--ratio-mean",
        choices=RATIO_MEANS,
        default=DEFAULT_RATIO_MEAN,
        help="VG20 only: whether the Gompertz arm also replaces the production-ratio mean "
        "(gompertz, the default) or keeps its trend + HSGP (flexible).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_prior = sub.add_parser("prior", help="Prior-predictive summaries for one curve.")
    p_prior.add_argument("curve", choices=sorted(CURVES))
    p_prior.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p_prior.add_argument("--draws", type=int, default=500)
    p_prior.add_argument("--seed", type=int, default=20261001)
    p_prior.set_defaults(func=prior_command)

    p_fit = sub.add_parser("fit", help="Fit one curve's arm.")
    p_fit.add_argument("curve", choices=sorted(CURVES))
    p_fit.add_argument("arm", choices=ARMS)
    p_fit.add_argument("--config", default="test")
    p_fit.set_defaults(func=fit_command)

    p_cmp = sub.add_parser("compare", help="Paired PSIS-LOO of the finished arms.")
    p_cmp.add_argument("--curves", nargs="+", choices=sorted(CURVES), default=["vg12", "vg11", "vg20"])
    p_cmp.add_argument("--config", default="test", help="The sampling configuration the arms must have been fitted at.")
    p_cmp.set_defaults(func=compare_command)

    p_loso = sub.add_parser("loso", help="Leave-one-subject-out K-fold for VG20.")
    p_loso.add_argument("--folds", type=int, default=5)
    p_loso.add_argument("--config", default="test")
    p_loso.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p_loso.set_defaults(func=loso_command)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
