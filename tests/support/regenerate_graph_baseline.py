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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from support.synthetic_graphs import (  # noqa: E402
    build_registered_model,
    fixed_point_logp,
    graph_fingerprint,
)
from vocab_growth.models.catalogue import CATALOGUE  # noqa: E402

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "graph_baseline.json")


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

    with tempfile.TemporaryDirectory() as directory:
        for key in selected:
            context = build_registered_model(
                key, output_dir=directory, monkeypatch=_Patcher()
            )
            entry = graph_fingerprint(context.model)
            entry["logp_at_fixed_point"] = fixed_point_logp(context.model)
            baseline[key] = entry
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
    print(f"\nWrote {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
