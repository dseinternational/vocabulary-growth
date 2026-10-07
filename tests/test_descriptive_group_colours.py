# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Study colours in the descriptive figures.

Since dse-research-utils 0.18.0, ``categorical_palette`` gives the six chart
colours and refuses more. The descriptive figures colour by study, and the
shared mappings span fifteen Down syndrome studies and twelve typically
developing ones. These tests pin that the mappings still build at those counts,
give each study its own colour and never give a study the red of the pooled
summary drawn over it.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from dse_research_utils.plot.styles import CHART_COLOURS
from matplotlib.colors import to_hex

from vocab_growth import descriptive

DS_STUDIES = [
    "es_01",
    "ie_01",
    "ie_02",
    "it_01",
    "nz_01",
    "uk_01",
    "uk_02",
    "uk_03",
    "uk_04",
    "uk_05",
    "uk_06",
    "uk_07",
    "us_01",
    "us_02",
    "us_03",
]
TD_STUDIES = [
    "Armon-Lotem",
    "Byers",
    "ByersHeinlein",
    "Floccia",
    "Frank",
    "Hoff",
    "Kalashnikova",
    "Marchman",
    "OToole",
    "Poulin-Dubois",
    "Smith",
    "Thal",
]


@pytest.mark.parametrize("studies", [DS_STUDIES, TD_STUDIES], ids=["ds", "td"])
def test_shared_mapping_gives_every_study_its_own_colour(studies):
    mapping = descriptive._shared_group_colours(studies)

    assert sorted(mapping) == sorted(studies)
    assert len({to_hex(colour) for colour in mapping.values()}) == len(studies)
    assert not any(descriptive._is_reddish(colour) for colour in mapping.values())


def test_up_to_six_groups_take_the_chart_colours_in_order():
    groups = ["a", "b", "c", "d", "e", "f"]

    # Chart-3 orange is not the overlay's red, so all six are used.
    assert list(descriptive._shared_group_colours(groups).values()) == list(
        CHART_COLOURS
    )
    assert descriptive._group_palette(6) == list(CHART_COLOURS)


def test_scatter_by_group_draws_every_down_syndrome_study():
    frame = pd.DataFrame(
        {
            "study": DS_STUDIES * 2,
            "age": range(2 * len(DS_STUDIES)),
            "understood": range(2 * len(DS_STUDIES)),
        }
    )

    fig = descriptive.scatter_by_group(frame, "age", "understood")
    try:
        assert len(fig.axes[0].collections) == len(DS_STUDIES)
    finally:
        plt.close(fig)
