# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
DEPRECATED — superseded by ``scripts/compare_ds_td_re.py``.

The learn-to-say latency ``a_S(N) - a_U(N)`` is now produced — as an
individual, linear-axis figure ``ds_td_comprehension_latency.{png,svg}`` (with
the q-at-matched-comprehension and gap panels) — by
``compare_ds_td_re.run_comprehension_matched``. This shim delegates there so the
canonical outputs stay in sync; please call ``compare_ds_td_re.py comprehension``
directly.
"""

from __future__ import annotations

from compare_ds_td_re import OUT_DIR, run_comprehension_matched

from vocab_growth import environment as env


def _main() -> None:
    print(
        "[deprecated] compare_ds_td_latency.py is superseded by "
        "compare_ds_td_re.py; delegating to run_comprehension_matched().",
    )
    env.preflight_disk(2.0, OUT_DIR, label="DS/TD comprehension-matched outputs")
    run_comprehension_matched()


if __name__ == "__main__":
    _main()
