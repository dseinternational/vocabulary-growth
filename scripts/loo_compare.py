# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Compute per-model LOO (leave-one-out cross-validation) and pairwise
comparisons via `az.compare()`.

For every saved trace under `output/models/<MODEL>/trace.nc`:

- `output/comparisons/loo_<MODEL>.csv` — single-model LOO summary
  (elpd_loo, p_loo, looic, plus high-Pareto-k counts).

For comparable model pairs (same observed outcomes on the same data):

- `output/comparisons/loo_compare_<TAG>.csv` — `az.compare()` ranking.

Pairs compared (joint models contribute their joint likelihood):

- DS bivariate: VG05 vs VG07 (effect of adding study random intercepts).

Where the same observed-y vector underlies two models with different
likelihood shapes (e.g. VG02 univariate-u vs VG05 joint), LOO would not
be directly comparable, so those are skipped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from vocab_growth import environment as env
from vocab_growth.loo_reff import reff_or_default
from vocab_growth.models.definitions import MODEL_REGISTRY, ModelType

MODELS_DIR = env.models_output_dir()
OUT_DIR = env.comparisons_output_dir()

# Registry-derived so a newly added univariate/bivariate model is picked up
# automatically. Trivariate (VG14) and joint (VG15) models have a different
# likelihood shape (three+ outcomes, a Dirichlet-Multinomial component) that
# this script's LOO logic does not handle, so they are excluded here — not a
# staleness gap, a scope boundary.
UNIVARIATE = {
    d.model_id: f"{d.model_id}-{d.config_name}"
    for d in MODEL_REGISTRY.values()
    if d.model_type == ModelType.UNIVARIATE
}

BIVARIATE = {
    d.model_id: f"{d.model_id}-{d.config_name}"
    for d in MODEL_REGISTRY.values()
    if d.model_type == ModelType.BIVARIATE
}

MODEL_LABELS = {**UNIVARIATE, **BIVARIATE}


@dataclass
class LooEntry:
    name: str
    elpd_loo: float
    se: float
    p_loo: float
    looic: float
    n_high_pareto: int


def _loo_summary_row(label: str, loo, reff=None) -> dict:
    if hasattr(loo, "pareto_k"):
        k = loo.pareto_k.values
    else:
        k = loo.diagnostics.values
    return {
        "label": label,
        "elpd_loo": float(loo.elpd),
        "se": float(loo.se),
        "p_loo": float(loo.p),
        # The relative efficiency this LOO used: pinned to the sampled
        # parameters where the trace records them (loo_reff), else ArviZ's
        # posterior-wide default, so a mixed-convention table shows it.
        "reff": None if reff is None else float(reff),
        "looic": float(-2.0 * loo.elpd),
        "looic_se": float(2.0 * loo.se),
        "pareto_k_gt_0.7": int((k > 0.7).sum()),
        "n_observations": int(k.size),
    }


def _attach_joint_log_likelihood(idata: xr.DataTree) -> None:
    """For bivariate traces with `y_u_obs` and `y_s_obs`, attach a
    combined `y_joint` log-likelihood whose per-observation entry is the
    concatenation across the two outcomes. Required so az.loo can treat
    the joint likelihood as a single coherent quantity."""
    ll = idata.log_likelihood
    if "y_joint" in ll.data_vars:
        return
    u = ll["y_u_obs"]
    s = ll["y_s_obs"]
    u_dim = [d for d in u.dims if d not in ("chain", "draw")][0]
    s_dim = [d for d in s.dims if d not in ("chain", "draw")][0]
    # Build a stacked array along a new `obs_joint` dimension.
    u_renamed = u.rename({u_dim: "obs_joint"}).assign_coords(
        obs_joint=np.arange(u.sizes[u_dim])
    )
    s_renamed = s.rename({s_dim: "obs_joint"}).assign_coords(
        obs_joint=np.arange(u.sizes[u_dim], u.sizes[u_dim] + s.sizes[s_dim])
    )
    joint = xr.concat([u_renamed, s_renamed], dim="obs_joint")
    idata.log_likelihood = ll.assign({"y_joint": joint})


def per_model_loo() -> dict[str, list[dict]]:
    """Compute LOO for every fitted model and write per-model CSVs."""
    out: dict[str, list[dict]] = {}
    for short, label in MODEL_LABELS.items():
        trace_path = os.path.join(MODELS_DIR, label, "trace.nc")
        if not os.path.exists(trace_path):
            print(f"  {short}: trace not found at {trace_path} — skipped")
            continue
        print(f"  {short}: loading trace …", flush=True)
        idata = az.from_netcdf(trace_path)
        if "log_likelihood" not in [g.rsplit("/", 1)[-1] for g in idata.groups]:
            print(f"  {short}: no log_likelihood group — skipped")
            continue

        rows = []
        # PSIS-LOO can fail numerically for the heavily subject-random-effect
        # models (leaving out one observation swings that child's intercept, so
        # the importance-weight tail degenerates — "All tail values are the
        # same"). Guard each az.loo call so one such model is skipped with a
        # warning rather than aborting the whole comparison, matching the
        # missing-trace / missing-log_likelihood skips above. The high Pareto-k
        # counts already flag the models whose PSIS-LOO is unreliable.
        reff = reff_or_default(idata, label=short)
        if short in UNIVARIATE:
            try:
                loo = az.loo(idata, pointwise=True, reff=reff)
            except Exception as exc:  # noqa: BLE001 - any LOO failure -> skip
                print(f"  {short}: LOO failed ({type(exc).__name__}: {exc}) — skipped")
                continue
            row = _loo_summary_row("y_obs", loo, reff)
            rows.append(row)
            print(
                f"  {short}: elpd_loo = {row['elpd_loo']:.1f} "
                f"± {row['se']:.1f}, p_loo = {row['p_loo']:.1f}, "
                f"high-k = {row['pareto_k_gt_0.7']}/{row['n_observations']}"
            )
        else:
            _attach_joint_log_likelihood(idata)
            for var in ("y_u_obs", "y_s_obs", "y_joint"):
                try:
                    loo = az.loo(idata, pointwise=True, var_name=var, reff=reff)
                except Exception as exc:  # noqa: BLE001 - any LOO failure -> skip
                    print(f"  {short} [{var}]: LOO failed ({type(exc).__name__}: {exc}) — skipped")
                    continue
                row = _loo_summary_row(var, loo, reff)
                rows.append(row)
                print(
                    f"  {short} [{var}]: elpd_loo = {row['elpd_loo']:.1f} "
                    f"± {row['se']:.1f}, p_loo = {row['p_loo']:.1f}, "
                    f"high-k = {row['pareto_k_gt_0.7']}/{row['n_observations']}"
                )

        if not rows:
            print(f"  {short}: no usable LOO — skipped")
            continue
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"loo_{short}.csv"),
                                  index=False)
        out[short] = rows
    return out


def compare_pair(
    tag: str,
    members: list[tuple[str, xr.DataTree]],
    *,
    var_name: str | None = None,
) -> None:
    """Run az.compare on the given InferenceData objects and write CSV.

    Each member's LOO is computed here first, with its relative efficiency
    pinned to the sampled parameters (loo_reff), and the results are handed to
    ``az.compare`` as ``ELPDData`` -- passing the traces would have it recompute
    LOO with ArviZ's posterior-wide default.
    """
    compare_dict = {}
    for name, idata in members:
        reff = reff_or_default(idata, label=name)
        try:
            compare_dict[name] = az.loo(idata, pointwise=True, var_name=var_name, reff=reff)
        except Exception as exc:  # noqa: BLE001 - degenerate PSIS-LOO -> skip this comparison
            print(f"  compare {tag}: LOO for {name} failed ({type(exc).__name__}: {exc}) — skipped")
            return
    kwargs = {"method": "stacking"}
    try:
        df = az.compare(compare_dict, **kwargs)
    except Exception as exc:  # noqa: BLE001 - degenerate PSIS-LOO -> skip this comparison
        print(f"  compare {tag}: failed ({type(exc).__name__}: {exc}) — skipped")
        return
    df.to_csv(os.path.join(OUT_DIR, f"loo_compare_{tag}.csv"))
    print(f"\n=== az.compare(): {tag} ===")
    print(df.to_string())


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Computing per-model LOO …\n")
    per_model_loo()

    # Compare VG05 vs VG07 — same DS data, same outcomes, RE on vs off.
    print("\nLoading VG05 + VG07 for joint comparison …", flush=True)
    vg05 = az.from_netcdf(os.path.join(MODELS_DIR, MODEL_LABELS["VG05"], "trace.nc"))
    vg07 = az.from_netcdf(os.path.join(MODELS_DIR, MODEL_LABELS["VG07"], "trace.nc"))
    _attach_joint_log_likelihood(vg05)
    _attach_joint_log_likelihood(vg07)
    compare_pair("ds_bivariate_re_vs_no_re_joint",
                 [("VG05", vg05), ("VG07", vg07)], var_name="y_joint")
    compare_pair("ds_bivariate_re_vs_no_re_understood",
                 [("VG05", vg05), ("VG07", vg07)], var_name="y_u_obs")
    compare_pair("ds_bivariate_re_vs_no_re_spoken",
                 [("VG05", vg05), ("VG07", vg07)], var_name="y_s_obs")


if __name__ == "__main__":
    main()
