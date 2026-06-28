# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
DEPRECATED — superseded by ``scripts/compare_ds_td_re.py``.

The q = S/U overlap at matched comprehension (q(U=N)) and at matched age (q(a))
is now produced — as individual, linear-axis figures
``ds_td_comprehension_q_at_U.{png,svg}`` and ``ds_td_comprehension_q_at_age.{png,svg}``
— by ``compare_ds_td_re.run_comprehension_matched``. (The secondary p_U(N)
overlay this script also drew is dropped as redundant with the expected-words
panels of ``compare_ds_td_re.run_outcome``.) This shim delegates to the
canonical run; please call ``compare_ds_td_re.py comprehension`` directly.
"""

from __future__ import annotations

from compare_ds_td_re import OUT_DIR, run_comprehension_matched

from vocab_growth import environment as env


def _main() -> None:
    print(
        "[deprecated] compare_ds_td_q_overlap.py is superseded by "
        "compare_ds_td_re.py; delegating to run_comprehension_matched().",
    )
    env.preflight_disk(2.0, OUT_DIR, label="DS/TD comprehension-matched outputs")
    run_comprehension_matched()


if __name__ == "__main__":
    _main()
