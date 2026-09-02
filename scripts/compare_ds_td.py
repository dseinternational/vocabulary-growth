# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
DEPRECATED — superseded by ``scripts/compare_ds_td_re.py comprehension``.

This one-off produced the early VG07-vs-VG06 ``ds_td_q_vs_understood`` overlay
and ``ds_td_q_crossings.csv``, and for a time delegated to a successor in
``compare_models``. Both drew the population production ratio at the age each
population's median reaches U words and labelled it "matched-comprehension" —
the conditional reading issue #233 rules out, and one the observed children
contradict. They were retired on 2026-09-02. The figure of record for the
contrast is the comparison book's ``ds_td_comprehension_q_at_U`` (DS VG20 vs
TD VG13, with the observed children set beside the curve), which
``compare_ds_td_re.run_comprehension_matched`` produces. This shim delegates
there; please call ``compare_ds_td_re.py comprehension`` directly.
"""

from __future__ import annotations

from compare_ds_td_re import OUT_DIR, run_comprehension_matched

from vocab_growth import environment as env


def _main() -> None:
    print(
        "[deprecated] compare_ds_td.py is superseded by compare_ds_td_re.py; "
        "delegating to run_comprehension_matched().",
    )
    env.preflight_disk(2.0, OUT_DIR, label="DS/TD comprehension-matched outputs")
    run_comprehension_matched()


if __name__ == "__main__":
    _main()
