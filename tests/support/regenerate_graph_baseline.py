# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rewrite ``graph_baseline.json`` from the current code.

Run this **only** when a deliberate statistical change has moved a graph, and
review the resulting diff as part of that change: it is the change's own
statement of what moved. Running it to make a failing refactor pass discards
the guard.

    uv run python tests/support/regenerate_graph_baseline.py [model ...]
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from support.synthetic_graphs import (  # noqa: E402
    build_registered_model,
    fixed_point,
    graph_fingerprint,
)
from vocab_growth.models.catalogue import CATALOGUE  # noqa: E402

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "graph_baseline.json")
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "graph_reference_points.json")


class _Patcher:
    """The one hook ``build_registered_model`` wants, outside pytest."""

    def setattr(self, obj, name, value, raising=True):
        setattr(obj, name, value)


def main(argv: list[str]) -> int:
    selected = [key.lower() for key in argv[1:]] or list(CATALOGUE)
    unknown = [key for key in selected if key not in CATALOGUE]
    if unknown:
        print(f"Unknown model(s): {unknown}", file=sys.stderr)
        return 1

    baseline = {}
    if os.path.isfile(BASELINE_PATH):
        with open(BASELINE_PATH, encoding="utf-8") as handle:
            baseline = json.load(handle)
    references = {}
    if os.path.isfile(REFERENCE_PATH):
        with open(REFERENCE_PATH, encoding="utf-8") as handle:
            references = json.load(handle)

    with tempfile.TemporaryDirectory() as directory:
        for key in selected:
            context = build_registered_model(
                key, output_dir=directory, monkeypatch=_Patcher()
            )
            model = context.model
            initial = model.initial_point()
            saved = references.get(key, {}).get("points", [])
            compatible = saved and all(
                set(point) == set(initial)
                and all(
                    np.shape(point[name]) == np.shape(value)
                    for name, value in initial.items()
                )
                for point in saved
            )
            if compatible:
                points = [
                    {name: np.asarray(value) for name, value in point.items()}
                    for point in saved
                ]
            else:
                first = fixed_point(model)
                points = [first, {name: value + 0.137 for name, value in first.items()}]
            evaluate = model.compile_logp()
            logps = [float(evaluate(point)) for point in points]
            entry = graph_fingerprint(model)
            entry["logp_at_fixed_point"] = logps[0]
            baseline[key] = entry
            references[key] = {
                "points": [
                    {name: np.asarray(value).tolist() for name, value in point.items()}
                    for point in points
                ],
                "logps": logps,
                "coordinates": {
                    name: None if values is None else np.asarray(values).tolist()
                    for name, values in model.coords.items()
                },
            }
            print(
                f"{key}: {len(entry['free_RVs'])} free, "
                f"{len(entry['deterministics'])} deterministic, "
                f"logp={entry['logp_at_fixed_point']:.6f}"
            )

    # Only registered models, so a retired one cannot linger in the baseline.
    baseline = {key: baseline[key] for key in CATALOGUE if key in baseline}
    with open(BASELINE_PATH, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=1, sort_keys=True)
        handle.write("\n")
    references = {key: references[key] for key in CATALOGUE if key in references}
    with open(REFERENCE_PATH, "w", encoding="utf-8") as handle:
        json.dump(references, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"\nWrote {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
