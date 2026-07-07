# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
DEPRECATED — superseded by ``scripts/compare_models.py``.

This one-off produced the early VG07-vs-VG06 ``ds_td_q_vs_understood`` overlay
and ``ds_td_q_crossings.csv``. The canonical figure (DS VG09 vs TD VG13, with
VG07 as a dashed reference) is now produced by
``compare_models.ds_td_q_vs_understood``. This shim delegates there so the
output files stay in sync; please call ``compare_models.py`` directly.
"""

from __future__ import annotations

import os

import dse_research_utils.plot.styles as plot_styles
from compare_models import OUT_DIR, ds_td_q_vs_understood

from vocab_growth import environment as env


def _main() -> None:
    print(
        "[deprecated] compare_ds_td.py is superseded by compare_models.py; "
        "delegating to compare_models.ds_td_q_vs_understood().",
    )
    env.preflight_disk(2.0, OUT_DIR, label="DS/TD q-vs-understood outputs")
    plot_styles.set_matplotlib_default_style()
    os.makedirs(OUT_DIR, exist_ok=True)
    ds_td_q_vs_understood()


if __name__ == "__main__":
    _main()
