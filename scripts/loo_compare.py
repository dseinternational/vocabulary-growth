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

Which outputs are leave-one-administration-out, and which are not (#266):

- The `y_joint` rows and the `..._joint` comparison ARE. `_attach_joint_log_likelihood`
  sums a paired row's two factors into one pointwise entry, so the held-out
  case is log p(U_i) + log p(S_i | U_i) — one PSIS weight per administration
  (#236). Repeated administrations of the same child are separate cases, so it
  scores prediction of another administration like those in the frame, not
  generalisation to a new child.
- The univariate models' single row IS: those traces carry one likelihood over
  administration rows, so the term and the administration are the same thing.
- The per-outcome `y_u_obs` / `y_s_obs` rows written by `per_model_loo`, and the
  `..._understood` / `..._spoken` comparisons run by `main`, are NOT. Each holds
  out one likelihood **term**, not an administration. The spoken likelihood's
  trial count is the same row's observed understood count, so a held-out spoken
  term is scored conditional on that observed comprehension, and a held-out
  understood term leaves its own observed value in the spoken term's
  denominator. They are comparable across models — every model is scored on the
  same conditional units — but they are not whole-administration predictive
  accuracy, and that is exactly why `y_joint` exists alongside them.

For new-child questions use grouped leave-one-subject-out
(`scripts/kfold_loso.py`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import arviz as az
import dse_research_utils.statistics.loo as shared_loo
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


#: Share of observations above Pareto-k 0.7 beyond which the PSIS estimate is
#: reported as unusable rather than merely caveated. PSIS-LOO degenerates for the
#: subject-random-effect models -- leaving one observation out swings the child
#: intercept it is nearly the only evidence for -- and past this fraction the
#: elpd is not comparable with another model's. Held well below the 50% VG11
#: reaches so the notice fires before a table looks authoritative.
HIGH_PARETO_K_UNUSABLE_SHARE = 0.20

#: Pareto-k above which an observation counts as "high k" in the summary tables.
#: Fixed at 0.7 rather than the fit's own ``good_k`` because the published column
#: name (``pareto_k_gt_0.7``) states the threshold and ``aggregate_summary.py``
#: reads it.
HIGH_PARETO_K_THRESHOLD = 0.7


def _warn_if_unusable(label: str, row: dict) -> None:
    """Print a notice when a row's PSIS-LOO has degenerated past use."""
    n = row["n_observations"]
    if not n:
        return
    share = row["pareto_k_gt_0.7"] / n
    if share < HIGH_PARETO_K_UNUSABLE_SHARE:
        return
    print(
        f"      [unusable] {label}: {share:.0%} of observations above Pareto-k 0.7 "
        f"(p_loo = {row['p_loo']:.0f} on {n} observations).\n"
        "      PSIS-LOO has degenerated here; do not compare this elpd with another "
        "model's.\n"
        "      Leaving out one observation removes most of what identifies that "
        "child's effect,\n"
        "      which is what leave-one-subject-out (loso_compare.py) exists to "
        "measure instead."
    )


def _loo_summary_row(label: str, loo, reff=None) -> dict:
    """One row of the LOO comparison table.

    Built from the shared :func:`dse_research_utils.statistics.loo.loo_summary_row`.
    ``reff`` is the relative efficiency this LOO used: pinned to the sampled
    parameters where the trace records them (``loo_reff``), else ArviZ's
    posterior-wide default, so a mixed-convention table shows it. The
    ``pareto_k_gt_0.7`` column keeps its fixed 0.7 threshold (rather than the
    shared default of the fit's own ``good_k``) because that threshold is part
    of the published column name and ``aggregate_summary.py`` reads it.
    """
    shared = shared_loo.loo_summary_row(
        loo,
        label=label,
        reff=reff,
        k_threshold=HIGH_PARETO_K_THRESHOLD,
        include_looic=True,
    )
    # Re-emitted in the published column order; the shared builder names the
    # count column generically and also carries the threshold it used.
    return {
        "label": shared["label"],
        "elpd_loo": shared["elpd_loo"],
        "se": shared["se"],
        "p_loo": shared["p_loo"],
        "reff": shared["reff"],
        "looic": shared["looic"],
        "looic_se": shared["looic_se"],
        "pareto_k_gt_0.7": shared["pareto_k_above"],
        "n_observations": shared["n_observations"],
    }


def _attach_joint_log_likelihood(idata: xr.DataTree) -> None:
    """For bivariate traces with `y_u_obs` and `y_s_obs`, attach a combined
    `y_joint` log-likelihood with one entry per administration.

    The coherent pointwise unit of the joint likelihood is the
    administration: log p(U_i) + log p(S_i | U_i) where both factors exist,
    and the single observed factor for an understood-only or spoken-only
    row. The stored `obs_u_mask` / `obs_s_mask` constant data map each
    outcome's likelihood rows back to administration rows, so paired
    factors are **summed**, not concatenated. Concatenating them (the
    pre-#236 behaviour) made every paired administration two held-out cases
    with two PSIS weights, and when its understood factor was held out the
    spoken factor still conditioned on the observed understood count —
    leaking information relative to leave-one-administration-out."""
    ll = idata.log_likelihood
    if "y_joint" in ll.data_vars:
        return
    if "constant_data" not in [g.rsplit("/", 1)[-1] for g in idata.groups]:
        raise ValueError(
            "trace has no constant_data group, so its likelihood rows cannot "
            "be mapped back to administrations for joint LOO."
        )
    const = idata.constant_data
    u_mask = np.asarray(const["obs_u_mask"].values, dtype=bool)
    s_mask = np.asarray(const["obs_s_mask"].values, dtype=bool)
    u = ll["y_u_obs"]
    s = ll["y_s_obs"]
    u_dim = [d for d in u.dims if d not in ("chain", "draw")][0]
    s_dim = [d for d in s.dims if d not in ("chain", "draw")][0]
    if int(u_mask.sum()) != u.sizes[u_dim] or int(s_mask.sum()) != s.sizes[s_dim]:
        raise ValueError(
            f"stored masks ({int(u_mask.sum())} understood, "
            f"{int(s_mask.sum())} spoken) do not match the likelihood rows "
            f"({u.sizes[u_dim]}, {s.sizes[s_dim]}); factors cannot be mapped "
            "to administrations."
        )
    any_mask = u_mask | s_mask
    # Position of each administration row among the rows kept in y_joint.
    joint_pos = np.cumsum(any_mask) - 1
    u_vals = u.transpose("chain", "draw", u_dim).values
    s_vals = s.transpose("chain", "draw", s_dim).values
    joint = np.zeros(u_vals.shape[:2] + (int(any_mask.sum()),), dtype=float)
    joint[..., joint_pos[u_mask]] += u_vals
    joint[..., joint_pos[s_mask]] += s_vals
    joint_da = xr.DataArray(
        joint,
        dims=("chain", "draw", "obs_joint"),
        coords={
            "chain": u["chain"].values,
            "draw": u["draw"].values,
            "obs_joint": np.flatnonzero(any_mask),
        },
    )
    idata.log_likelihood = ll.assign({"y_joint": joint_da})


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
        # missing-trace / missing-log_likelihood skips above. A model whose
        # PSIS-LOO survives numerically but has degenerated is caught after the
        # fact by _warn_if_unusable, which judges the high Pareto-k share rather
        # than leaving the reader to.
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
            _warn_if_unusable(short, row)
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
                _warn_if_unusable(f"{short} [{var}]", row)

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
