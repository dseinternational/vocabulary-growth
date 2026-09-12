#!/usr/bin/env python
# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Write the leave-one-out summary for fits that computed it and threw it away.

Every fit runs PSIS-LOO, prints it to the console and discards it, while the
predictive-calibration section of every model report points the reader at
leave-one-out as the out-of-sample counterpart to its in-sample checks. The
number they were sent to find has never existed in the output directory.

``vocab_growth.models.common.emit_loo_summary`` closes that for future fits.
This closes it for the fits already on disk, without a refit, because LOO is a
**deterministic function of the stored ``log_likelihood`` group** -- the fit's
own recorded output. That is the same basis on which ``regenerate_plots.py``
rebuilds plot-stage artefacts from a stored trace, and it is the line this
script stays on the right side of: it recomputes from what the fit wrote, and
never manufactures anything the fit did not record. A trace saved under the
``minimal`` persistence tier has no ``log_likelihood`` group and is refused
rather than approximated.

Usage::

    python scripts/emit_loo_summaries.py <model_id> [<model_id> ...]
    python scripts/emit_loo_summaries.py all
"""

import argparse
import os
import re
import sys

import arviz as az

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vocab_growth import environment as env  # noqa: E402
from vocab_growth.fit_consumers import (  # noqa: E402
    add_allow_stale_argument,
    require_current_fit,
)
from vocab_growth.models.common import (  # noqa: E402
    LOO_SUMMARY_FILENAME,
    emit_loo_summary,
    loo_dropping_degenerate,
)
from vocab_growth.models.definitions import MODEL_REGISTRY  # noqa: E402
from vocab_growth.reporting import heading  # noqa: E402

# The labels the engines pass as ``loo_var_names``, keyed by the log-likelihood
# variable. Kept identical to the engines so a backfilled table and a refitted
# one carry the same rows under the same names.
OUTCOME_LABELS = {
    "y_u_obs": "words understood",
    "y_s_obs": "words spoken",
    "y_sign_obs": "words signed",
}

# Single-outcome models name their likelihood `y_obs`; the engine reports one
# unlabelled block for them, which `emit_loo_summary` records as "all".
SINGLE_OUTCOME = "y_obs"

# VG15 additionally stores four-cell and produced-cell likelihoods. The engine
# does not include them in its LOO reporting -- a Dirichlet-Multinomial over
# cells is a different estimand on a different set of rows, not the
# per-observation Beta-Binomial the marginal terms are -- so neither do we.
#
# State the consequence rather than leaving it to be inferred (#266): these two
# terms are the ones that identify the sign-speech association psi, so psi is
# not scored by leave-one-out at all, in a refitted table or a backfilled one.
# A cross-tabulation row still contributes its `y_u_obs` term, so the "words
# understood" row holds out one of that row's two factors while its composition
# factor stays in the conditioning set.
EXCLUDED = {"cells_obs", "nz_prod_cells_obs"}

# A cross-lag model's engine refuses to compute the scores its predictor leaks
# across, and a backfilled table must refuse the same ones -- otherwise the
# table this script writes contradicts the table a refit would write, for the
# same fit, and the leaking number is the one that ends up published.
#
# Which terms leak is a property of what the predictor reads: VG16's lag reads
# an earlier wave's `understood`, and VG25's reads that wave's `signed` as well.
# Keyed by the definition field so a model without the lag is untouched.
LEAKING_TERMS = {
    "use_cross_lag": {"y_u_obs"},
    "use_sign_cross_lag": {"y_u_obs", "y_sign_obs"},
}


def suppressed_outcomes(model_id: str) -> set[str]:
    """Log-likelihood terms this model's own engine will not score.

    Empty for every model without a cross-lag, which is every model but two.
    An unregistered ``model_id`` suppresses nothing rather than guessing: the
    caller has already refused to write a table for a fit it cannot identify.
    """
    definition = next(
        (d for d in MODEL_REGISTRY.values() if d.model_id == model_id), None
    )
    if definition is None:
        return set()
    return {
        term
        for field, terms in LEAKING_TERMS.items()
        if getattr(definition, field, False)
        for term in terms
    }


def model_directories(output_root: str) -> dict[str, str]:
    """Models of record, keyed by lower-case model id."""
    found: dict[str, str] = {}
    models_dir = os.path.join(output_root, "models")
    for name in sorted(os.listdir(models_dir)):
        match = re.match(r"^(VG\d\d)-[a-z]", name)
        if not match:
            continue
        model_id = match.group(1).lower()
        # The shortest name for an id is the model of record; the longer ones
        # are registered sensitivity variants and recovery replicates.
        if model_id not in found or len(name) < len(found[model_id]):
            found[model_id] = name
    return {k: os.path.join(models_dir, v) for k, v in found.items()}


def emit_for(model_id: str, directory: str, *, allow_stale: bool = False) -> bool:
    trace_path = os.path.join(directory, "trace.nc")
    if not os.path.isfile(trace_path):
        print(f"  {model_id}: no trace.nc — skipped")
        return False

    # ``loo_<MODEL>.csv`` is a published comparison table, so the fit behind it
    # is checked against the registered definition and the frame today's loader
    # rules produce (issue #266 finding 1).
    require_current_fit(
        model_id,
        directory,
        consumer="emit_loo_summaries.py",
        allow_stale=allow_stale,
    )
    idata = az.from_netcdf(trace_path)
    # ArviZ 1.x returns an xarray DataTree whose ``groups`` is a tuple of paths
    # ("/log_likelihood"), not a method returning bare names as in 0.x.
    if "log_likelihood" not in idata:
        print(
            f"  {model_id}: trace has no log_likelihood group (persistence tier "
            "'minimal') — refit to produce a LOO summary"
        )
        return False

    suppressed = suppressed_outcomes(model_id)
    available = [
        str(name)
        for name in idata.log_likelihood.data_vars
        if str(name) not in EXCLUDED and str(name) not in suppressed
    ]
    if suppressed:
        print(
            f"  {model_id}: not scoring {', '.join(sorted(suppressed))} — the "
            "cross-lag predictor embeds those counts in later rows, so a "
            "pointwise hold-out still conditions on the held-out outcome "
            "(issue #242). Its engine suppresses them too."
        )
    loo_by_label: dict = {}
    dropped_by_label: dict[str, int] = {}

    if available == [SINGLE_OUTCOME]:
        loo, dropped = loo_dropping_degenerate(idata)
        loo_by_label["all"] = loo
        dropped_by_label["all"] = dropped
    else:
        for var_name in available:
            label = OUTCOME_LABELS.get(var_name, var_name)
            loo, dropped = loo_dropping_degenerate(idata, var_name=var_name)
            loo_by_label[label] = loo
            dropped_by_label[label] = dropped

    if not loo_by_label:
        print(f"  {model_id}: no usable log-likelihood variables — skipped")
        return False

    frame = emit_loo_summary(loo_by_label, dropped_by_label, directory)
    rows = ", ".join(
        f"{r.outcome} elpd {r.elpd_loo:,.1f} (SE {r.se:.1f})"
        for r in frame.itertuples()
    )
    print(f"  {model_id}: wrote {LOO_SUMMARY_FILENAME} — {rows}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="model ids, or 'all'")
    parser.add_argument("--output-dir", default=None)
    add_allow_stale_argument(parser)
    args = parser.parse_args()

    output_root = args.output_dir or env.output_root()
    directories = model_directories(output_root)

    requested = (
        sorted(directories) if args.models == ["all"] else [m.lower() for m in args.models]
    )
    heading("LOO summaries", style="bold cyan")
    written = 0
    for model_id in requested:
        if model_id not in directories:
            print(f"  {model_id}: no model-of-record output found — skipped")
            continue
        if emit_for(
            model_id, directories[model_id], allow_stale=args.allow_stale_fit
        ):
            written += 1
    print(f"\n{written}/{len(requested)} model(s) updated.")


if __name__ == "__main__":
    main()
