# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Edgin subset audit's exact run probability.

The audit's strongest finding is that 21 consecutive Words & Sentences records
sit at exactly the 680-word form ceiling, which
``notes/202607261245-edgin-duplicated-outcome-records.md`` §13 quotes at a
probability of 1.3e-22 against chance. That number carries real weight — the
source author no longer holds the original files, so no confirmation is coming
and the published claim rests on this arithmetic — so it is pinned here against
cases countable by hand rather than trusted from a single ad-hoc run.
"""

import importlib.util
import math
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "audit_edgin_subset.py"
_SPEC = importlib.util.spec_from_file_location("audit_edgin_subset_script", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_run_probability = _MODULE._run_probability


def test_run_probability_matches_hand_enumerated_cases():
    # 1 flag in 5 slots is trivially a run of 1.
    assert _run_probability(5, 1, 1) == 1.0
    # 2 flags in 3 slots: of the 3 arrangements, 2 are adjacent.
    assert _run_probability(3, 2, 2) == 2 / 3
    # 2 flags in 4 slots: 3 adjacent pairs of the 6 arrangements.
    assert _run_probability(4, 2, 2) == 0.5
    # 3 flags in 5 slots: 3 of the 10 arrangements hold all three together.
    assert _run_probability(5, 3, 3) == 3 / 10
    # A run longer than the number of flags is impossible.
    assert _run_probability(5, 2, 3) == 0.0


def test_run_probability_is_exhaustively_correct_on_small_cases():
    """Brute-force every arrangement for small (n, m, k) and compare."""
    from itertools import combinations

    for n in range(1, 11):
        for m in range(0, n + 1):
            for k in range(1, m + 2):
                hits = 0
                total = 0
                for positions in combinations(range(n), m):
                    total += 1
                    flags = set(positions)
                    longest = current = 0
                    for i in range(n):
                        current = current + 1 if i in flags else 0
                        longest = max(longest, current)
                    hits += longest >= k
                expected = hits / total
                assert math.isclose(
                    _run_probability(n, m, k), expected, rel_tol=1e-12, abs_tol=1e-15
                ), f"n={n} m={m} k={k}"


def test_no_flags_can_never_produce_a_run():
    assert _run_probability(100, 0, 1) == 0.0


def test_the_reported_edgin_probabilities_are_reproduced():
    """The two figures §13 of the note publishes."""
    # 21 consecutive ceiling records among 27, across the whole Edgin cohort's
    # 235 Words & Sentences records.
    assert _run_probability(235, 27, 21) < 1e-21
    assert _run_probability(235, 27, 21) > 1e-23
    # The Down-syndrome-only view the first pass saw truncates the run to 13.
    assert _run_probability(109, 18, 13) < 1e-10
