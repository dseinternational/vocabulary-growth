# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Generate a fit's publication artefacts from a trace that already exists.

The convergence gate fails closed, and it stops the pipeline at *diagnostics* --
before posterior-predictive sampling, the summary tables, the plots and the
report. So a fit that fails the gate leaves a complete trace behind in
``output/failed/`` with none of the artefacts that would let it be read.

That is the right default. This is the escape hatch for the case where the
failure has since been accepted as a registered
:class:`~vocab_growth.fit_artifacts.ConvergenceException`: rather than spend
another five hours re-sampling a posterior that is already on disk, re-run the
pipeline with the *sampling* stage replaced by a loader for the retained trace.

    python scripts/resume_from_trace.py vg11 output/failed/VG11-...-20260814T210639Z

Everything else runs unchanged -- the same build, the same gate, the same
summary, plot and report code -- so the artefacts are the ones the pipeline
would have produced had the gate let it through. The gate still runs, and still
closes unless an exception covers the failure; this script cannot bypass it.

**What it does not do.** It does not re-sample, so it cannot fix a convergence
problem, and it is not a way to avoid a refit that is actually needed. The
resumed fit records its provenance in ``fit_manifest.json`` under
``artefacts.resumed_from``, so a reader can always tell that its trace predates
its summaries.

Guards, all fail-closed, because a trace and a model that disagree would produce
silently wrong summaries rather than an error:

* the retained manifest's model definition must equal the current registered one;
* its raw-data fingerprint must equal the current one;
* its sampling configuration name must equal the one being resumed under;
* the trace's free variables and their dimensions must match the rebuilt model's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import freeze_support

import dse_research_utils.environment.setup as setup
import xarray as xr

from vocab_growth import environment as env
from vocab_growth.fit_artifacts import (
    FIT_MANIFEST_FILENAME,
    normalise_for_json,
    source_data_hash,
)
from vocab_growth.models.common import run_fit_pipeline
from vocab_growth.models.definitions import MODEL_REGISTRY
from vocab_growth.reporting import console, key_value_table

SAMPLING_STAGE_NAME = "Posterior sampling"
TRACE_FILENAME = "trace.nc"


def _stages_for(model_key: str, definition):
    """The engine stage list for this model, resolved as the recovery harness does."""
    from vocab_growth.recovery.spec import recovery_target

    try:
        return recovery_target(model_key).resolve_stages(definition)
    except KeyError:
        # Not every model is recovery-registered; fall back to the engine's own
        # stage factory by importing the module the definition names.
        import importlib

        for module, factory in (
            ("common_univariate_re", "univariate_re_stages"),
            ("common_bivariate_re", "bivariate_re_stages"),
            ("common_joint_modality", "joint_stages"),
        ):
            mod = importlib.import_module(f"vocab_growth.models.{module}")
            fn = getattr(mod, factory, None)
            if fn is None:
                continue
            try:
                return fn(definition)
            except Exception:  # noqa: BLE001 - wrong engine for this definition
                continue
        raise


def _verify(retained_dir: str, definition, config: str) -> dict:
    """Fail closed unless the retained fit is the one we are about to resume."""
    manifest_path = os.path.join(retained_dir, FIT_MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No {FIT_MANIFEST_FILENAME} in {retained_dir}.")
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    recorded = manifest.get("model", {}).get("definition")
    if recorded != normalise_for_json(definition):
        raise ValueError(
            "The retained fit's model definition differs from the current "
            "registered definition; resuming would summarise one model's trace "
            "under another model's specification."
        )
    recorded_config = manifest.get("sampling", {}).get("configuration_name")
    if recorded_config != config:
        raise ValueError(
            f"The retained fit used sampling configuration {recorded_config!r}, "
            f"not {config!r}."
        )
    recorded_hash = manifest.get("data", {}).get("source_data_hash")
    current_hash = source_data_hash(env.DATA_DIR)
    if recorded_hash != current_hash:
        raise ValueError(
            "The raw data has changed since the retained fit; its trace no "
            "longer corresponds to the data the summaries would describe."
        )
    return manifest


def _loader_stage(trace_path: str):
    """The stage that replaces sampling: load the retained trace and check it fits."""

    def load_retained_trace(context) -> None:
        console.print(f"[dim]Loading retained trace: {trace_path}[/dim]")
        trace = xr.open_datatree(trace_path)
        posterior = trace["posterior"].to_dataset()

        # The trace and the freshly-built model must agree on every sampled
        # variable and every dimension. A mismatch means the design moved
        # underneath the samples -- different rows, children or studies -- and
        # every summary computed from it would be quietly wrong.
        expected = {rv.name for rv in context.model.free_RVs}
        missing = expected - set(posterior.data_vars)
        if missing:
            raise ValueError(
                f"The retained trace is missing sampled variable(s) "
                f"{sorted(missing)[:5]}; it was not produced by this model."
            )
        for dim, size in context.model.coords.items():
            if dim in posterior.sizes and size is not None:
                if posterior.sizes[dim] != len(size):
                    raise ValueError(
                        f"Dimension {dim!r} is {posterior.sizes[dim]} in the "
                        f"retained trace and {len(size)} in the rebuilt model."
                    )
        context.set_trace(trace)

    return load_retained_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Registered model key, e.g. vg11.")
    parser.add_argument("retained_dir", help="Directory holding the retained trace.")
    parser.add_argument("--config", default="rep")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    env.set_output_root(args.output_dir)
    setup.init_script()

    if args.model not in MODEL_REGISTRY:
        console.print(f"[bold red]Unknown model {args.model!r}.[/bold red]")
        return 1
    definition = MODEL_REGISTRY[args.model]
    trace_path = os.path.join(args.retained_dir, TRACE_FILENAME)
    if not os.path.isfile(trace_path):
        console.print(f"[bold red]No trace at {trace_path}.[/bold red]")
        return 1

    try:
        manifest = _verify(args.retained_dir, definition, args.config)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 1

    key_value_table(
        "Resume from retained trace",
        [
            ("Model", f"{definition.model_id} ({args.model})"),
            ("Retained fit", args.retained_dir),
            ("Fitted at", manifest.get("created_at_utc", "unknown")),
            ("Sampling config", args.config),
            ("Trace size", f"{os.path.getsize(trace_path) / 1024**3:.1f} GiB"),
            ("Output root", env.output_root()),
        ],
    )

    stages = _stages_for(args.model, definition)
    names = [name for name, _ in stages]
    if SAMPLING_STAGE_NAME not in names:
        console.print(
            f"[bold red]No {SAMPLING_STAGE_NAME!r} stage to replace; found "
            f"{names}.[/bold red]"
        )
        return 1
    index = names.index(SAMPLING_STAGE_NAME)
    stages[index] = (
        f"{SAMPLING_STAGE_NAME} (loaded from retained trace)",
        _loader_stage(trace_path),
    )
    # Prior predictive checks re-draw from the prior and cost real time without
    # informing anything the posterior artefacts need; the retained fit already
    # has them.
    stages = [s for s in stages if s[0] != "Prior predictive checks"]

    context = run_fit_pipeline(args.config, definition, stages=stages)

    # Record the provenance so a reader can tell that this fit's trace predates
    # its summaries. Written after promotion, into the promoted directory.
    manifest_path = os.path.join(context.reporting.output_dir, FIT_MANIFEST_FILENAME)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            promoted = json.load(handle)
        promoted.setdefault("artefacts", {})["resumed_from"] = {
            "retained_dir": os.path.abspath(args.retained_dir),
            "retained_fit_created_at_utc": manifest.get("created_at_utc"),
            "note": (
                "Posterior samples were taken from the retained trace; every "
                "downstream artefact was regenerated by the standard pipeline."
            ),
        }
        from vocab_growth.fit_artifacts import write_json_atomic

        write_json_atomic(manifest_path, promoted)
    return 0


if __name__ == "__main__":
    freeze_support()
    sys.exit(main())
