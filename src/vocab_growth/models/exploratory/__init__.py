# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exploratory models. **Their output is not validatable and must not be published.**

Issue #273 finding 4 asked for a decision about VG17 and VG18: either give them
frozen definitions and a supported lifecycle, or say plainly that they are
exploratory and mark their output accordingly. This package is the second, taken
on 2026-08-31.

It is what the modules already claimed about themselves -- both docstrings said
"Exploratory", "not in ``MODEL_REGISTRY``", and that folding the covariate into
``common_univariate_re`` "would be the productionisation step" -- made
structural rather than left as prose. Three things follow from living here:

**They are unreachable from the registered path by construction.** A registered
model is a ``model_vgNN.py`` module beside this package, and ``fit_model.py``
resolves exactly that name from the catalogue. Nothing in here matches, so no
amount of forgetting can route a fit through an exploratory module.

**Their output declares itself.** :func:`write_exploratory_marker` writes
``exploratory_output.json`` into every output directory these modules produce,
saying what the fit does *not* carry. ``sync_report_figures.py`` already skips
them as unregistered output; the marker is for the person who finds the
directory, who otherwise sees something shaped exactly like a publishable fit.

**What they do not carry, and why that is disqualifying rather than untidy.**
The custom fit path writes a trace and a contrast table and nothing else: no
``fit_manifest.json``, so there is no record of the definition, the data
fingerprint or the prepared-frame hash to validate against; no ``fit_state.json``
and no staged promotion, so an interrupted run leaves a half-written directory
that looks complete; no prior or posterior predictive checks; no calibration; no
LOO; and no convergence gate -- the maximum R-hat over the three contrasts is
printed, not enforced, and the other parameters are not screened at all. It also
borrows VG01's dispersion prior, which was calibrated marginally, while adding
study and child random effects that change what that prior means.

Productionising either model is a **statistical** decision, not a packaging one,
and is deliberately not taken here. ``common_univariate_re`` constrains its study
effects to sum to zero while VG17 uses unconstrained offsets, so routing VG17
through it would change the model rather than move it. That work belongs with
`#266 <https://github.com/dseinternational/vocabulary-growth/issues/266>`_.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

#: Written into every exploratory output directory. Named so it sorts beside
#: ``fit_manifest.json`` -- the file a reader looks for and will not find.
EXPLORATORY_MARKER_FILENAME = "exploratory_output.json"

#: What a registered fit carries and an exploratory one does not. Recorded in
#: the marker so the directory states its own gaps rather than requiring the
#: reader to know them.
MISSING_ARTEFACTS: tuple[str, ...] = (
    "fit_manifest.json (definition, sampling configuration, raw-data "
    "fingerprint and prepared-frame hash)",
    "fit_state.json and atomic staged promotion",
    "prior predictive checks",
    "posterior predictive checks",
    "predictive calibration",
    "leave-one-out cross-validation",
    "the all-parameter R-hat and ESS convergence gate",
)


def write_exploratory_marker(
    output_dir: str, *, model_label: str, note: str | None = None
) -> str:
    """Declare an output directory exploratory, and say what it lacks.

    Written by the exploratory ``fit()`` paths before anything else lands, so an
    interrupted run still leaves the directory labelled. Returns the path.
    """
    os.makedirs(output_dir, exist_ok=True)
    payload: dict[str, Any] = {
        "exploratory": True,
        "validatable": False,
        "publishable": False,
        "model": model_label,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "summary": (
            "Exploratory output. This directory was produced by a module in "
            "vocab_growth.models.exploratory, which does not run the shared fit "
            "pipeline. It must not be published, synced into the report's figure "
            "cache, or cited as a fitted result."
        ),
        "missing_artefacts": list(MISSING_ARTEFACTS),
        "issue": "https://github.com/dseinternational/vocabulary-growth/issues/273",
    }
    if note:
        payload["note"] = note
    path = os.path.join(output_dir, EXPLORATORY_MARKER_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def is_exploratory_output(output_dir: str) -> bool:
    """Whether ``output_dir`` was produced by an exploratory module."""
    return os.path.isfile(os.path.join(output_dir, EXPLORATORY_MARKER_FILENAME))
