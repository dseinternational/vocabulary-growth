# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unregistered exploratory models whose outputs must not be published.

VG17 and VG18 use a custom fitting path outside the catalogue and shared fit
pipeline. Their directories carry an exploratory_output.json marker but lack
registered definitions, validated provenance, staged promotion, predictive
checks and an all-parameter convergence gate.

Routing these models through a registered engine would require a statistical
review. VG17 uses unconstrained study offsets, while common_univariate_re
constrains study offsets to sum to zero. Moving it between those engines would
change the model.
See issue #266 for that work and #273 for the exploratory status decision.
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
