# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Aggregate diagnostics + timing into a single per-model summary CSV.

Reads:

- `output/models/<MODEL>/diagnostics.csv` (per-parameter ESS / R-hat)
- the per-model trace (for divergence count, sample_stats)
- `output/logs/fit_all_rep_*.log` (the most recent — for per-model
  wall-time)
- `output/comparisons/loo_<MODEL>.csv` if present (LOO summary rows)

Writes:

- `output/comparisons/model_summary.csv` — one row per model with the
  headline diagnostic ranges, divergence counts, sampling time and (if
  LOO has been computed) elpd_loo and Pareto-k tail counts.
- `output/logs/run_summary.json` — structured timing log derived from
  the most recent rep log.
"""

from __future__ import annotations

import glob
import json
import os
import re
from datetime import UTC, datetime
from typing import Any

import arviz as az
import pandas as pd

from vocab_growth import environment as env
from vocab_growth.models.definitions import MODEL_REGISTRY

# (short model id, output-folder label), derived from the registry so a newly
# added model is picked up automatically rather than requiring a second,
# hand-maintained list here.
MODELS = [
    (d.model_id, f"{d.model_id}-{d.config_name}") for d in MODEL_REGISTRY.values()
]

LOG_DIR = os.path.join(env.output_root(), "logs")
MODELS_DIR = env.models_output_dir()
COMPARE_DIR = env.comparisons_output_dir()


def parse_log_timings(log_path: str) -> dict[str, Any]:
    """Extract per-model wall times + total run time from a fit log."""
    if not os.path.exists(log_path):
        return {"log": None}
    with open(log_path, encoding="utf-8") as f:
        text = f.read()

    # The log has lines like "Total run wall time: 3h 09m 20.7s"
    total_match = re.search(r"Total run wall time:\s*([\dhms\. ]+)", text)
    total = total_match.group(1).strip() if total_match else None

    # Per-model timings are reported in a "Run summary" table near the
    # end. Rich draws this with Unicode box-drawing chars `│` (so the
    # plain ASCII `|` does not appear). Accept either form.
    per_model = {}
    rows = re.findall(
        r"[│|]\s*(vg\d{2}[a-z]?)\s*[│|]\s*([\dhms\. ]+?)\s*[│|]\s*([\d\.]+%)\s*[│|]",
        text,
    )
    for model_id, wall, pct in rows:
        per_model[model_id.upper()] = {"wall": wall.strip(), "pct": pct.strip()}

    sampling_match = re.search(
        r"Sampling configuration[\s\S]*?draws[\s\S]*?(\d[\d,]*)[\s\S]*?"
        r"tune[\s\S]*?(\d[\d,]*)[\s\S]*?"
        r"chains[\s\S]*?(\d)[\s\S]*?"
        r"target_accept[\s\S]*?([\d\.]+)",
        text,
    )
    if sampling_match:
        draws, tune, chains, target = sampling_match.groups()
        sampling = {
            "draws": int(draws.replace(",", "")),
            "tune": int(tune.replace(",", "")),
            "chains": int(chains),
            "target_accept": float(target),
        }
    else:
        sampling = None

    return {
        "log": os.path.basename(log_path),
        "total_wall": total,
        "per_model": per_model,
        "sampling": sampling,
    }


def trace_divergences(trace_path: str) -> int | None:
    if not os.path.exists(trace_path):
        return None
    idata = az.from_netcdf(trace_path)
    if "sample_stats" not in idata.groups():
        return None
    diverging = idata.sample_stats.get("diverging")
    if diverging is None:
        return 0
    return int(diverging.values.sum())


def per_model_summary(short: str, label: str,
                      log_timings: dict[str, Any]) -> dict[str, Any]:
    model_dir = os.path.join(MODELS_DIR, label)
    diag_path = os.path.join(model_dir, "diagnostics.csv")
    if not os.path.exists(diag_path):
        return {"model": short, "label": label, "note": "missing diagnostics"}

    diag = pd.read_csv(diag_path, index_col=0)
    ess_min = float(diag["ess_bulk"].min())
    ess_max = float(diag["ess_bulk"].max())
    ess_tail_min = float(diag["ess_tail"].min())
    rhat_max = float(diag["r_hat"].max())

    divergences = trace_divergences(os.path.join(model_dir, "trace.nc"))

    timing = log_timings.get("per_model", {}).get(short, {})

    summary = {
        "model": short,
        "label": label,
        "n_parameters_reported": int(len(diag)),
        "ess_bulk_min": ess_min,
        "ess_bulk_max": ess_max,
        "ess_tail_min": ess_tail_min,
        "rhat_max": rhat_max,
        "divergences": divergences,
        "wall_time": timing.get("wall"),
        "wall_pct_of_run": timing.get("pct"),
    }

    loo_path = os.path.join(COMPARE_DIR, f"loo_{short}.csv")
    if os.path.exists(loo_path):
        loo = pd.read_csv(loo_path)
        # Univariate single-row, or bivariate multi-row → take y_joint or y_obs
        target = (
            loo[loo["label"] == "y_joint"]
            if "y_joint" in loo["label"].values
            else loo[loo["label"] == "y_obs"]
        )
        if not target.empty:
            row = target.iloc[0]
            summary["elpd_loo"] = float(row["elpd_loo"])
            summary["elpd_loo_se"] = float(row["se"])
            summary["pareto_k_gt_0.7"] = int(row["pareto_k_gt_0.7"])
            summary["n_loo_obs"] = int(row["n_observations"])

    return summary


def main() -> None:
    os.makedirs(COMPARE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    logs = sorted(glob.glob(os.path.join(LOG_DIR, "fit_all_rep_*.log")))
    log_path = logs[-1] if logs else ""
    timings = parse_log_timings(log_path)

    # Persist parsed timings as JSON
    run_summary = {
        "source_log": timings.get("log"),
        "captured_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "total_wall": timings.get("total_wall"),
        "sampling_config": timings.get("sampling"),
        "per_model": timings.get("per_model"),
    }
    with open(os.path.join(LOG_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    rows = [per_model_summary(short, label, timings) for short, label in MODELS]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(COMPARE_DIR, "model_summary.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\nWrote: {os.path.join(COMPARE_DIR, 'model_summary.csv')}")
    print(f"       {os.path.join(LOG_DIR, 'run_summary.json')}")


if __name__ == "__main__":
    main()
