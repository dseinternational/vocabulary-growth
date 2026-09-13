# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Values a model builder returns for the pipeline to report."""

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class BuildTable:
    title: str
    rows: list[tuple[str, object]]
    key_header: str = "Parameter"
    value_header: str = "Value"


@dataclass
class BuildReport:
    """Build settings and lag audits, with no printing or file writes."""

    tables: list[BuildTable] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    understood_lag_audit: pd.DataFrame | None = None
    signed_lag_audit: pd.DataFrame | None = None
    sign_lag_in_cells: bool = True

    def add_table(
        self,
        title: str,
        rows: list[tuple[str, object]],
        *,
        key_header: str = "Parameter",
        value_header: str = "Value",
    ) -> None:
        self.tables.append(BuildTable(title, rows, key_header, value_header))
