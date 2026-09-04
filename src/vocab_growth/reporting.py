# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Console reporting shim for the ``vocab_growth`` pipelines.

Historically this module implemented a bespoke set of rich-based primitives
(banner, heading, key/value table, dataframe table, pipeline summary,
timed section, ...). Those primitives now live in
:mod:`dse_research_utils.console` and are shared across DSE research
projects. This module preserves the legacy public API — signatures, default
column headers, early-return behaviour for empty inputs — so no caller has
to change, and routes every call through the shared implementation.
"""

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import is_dataclass
from typing import Any

import pandas as pd
from dse_research_utils.console.console import get_console, print_panel, print_table
from dse_research_utils.console.format import format_duration as _format_duration
from dse_research_utils.console.format import format_value as _format_value
from dse_research_utils.console.sections import banner as _banner
from dse_research_utils.console.sections import section_header as _section_header
from dse_research_utils.console.sections import timed_section as _timed_section
from dse_research_utils.console.summary import (
    pipeline_summary as _pipeline_summary_table,
)
from dse_research_utils.console.tables import dataframe_table as _dataframe_table
from dse_research_utils.console.tables import key_value_table as _key_value_table
from rich.console import Console

console: Console = get_console()


def run_banner(title: str, subtitle: str | None = None) -> None:
    """Top-of-run banner. Use once per model fit."""
    _banner(title, subtitle)


def heading(title: str, *, style: str = "bold green") -> None:
    """Section heading without timing."""
    _section_header(title, style=style)


@contextmanager
def section(
    title: str,
    *,
    timings: dict[str, float] | None = None,
    key: str | None = None,
    style: str = "bold green",
):
    """Emit a section heading, run the block, and record its wall time."""
    with _timed_section(title, timings=timings, key=key, style=style):
        yield


def key_value_table(
    title: str,
    pairs: Iterable[tuple[str, Any]] | Mapping[str, Any],
    *,
    key_header: str = "Parameter",
    value_header: str = "Value",
) -> None:
    """Render a two-column table of (name, value) pairs."""
    table = _key_value_table(
        pairs,
        title=title,
        key_header=key_header,
        value_header=value_header,
    )
    print_table(table)


def dataframe_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    max_rows: int | None = None,
    float_format: str = ".4g",
    show_index: bool = True,
) -> None:
    """Render a pandas DataFrame as a rich Table (prints; no return value)."""
    precision = _precision_from_float_format(float_format)
    if df.empty:
        console.print(_format_value(title) if title else "(empty)")
        return
    table = _dataframe_table(
        df,
        title=title,
        max_rows=max_rows,
        truncation="head_tail",
        show_index=show_index,
        precision=precision,
    )
    print_table(table)


def config_table(title: str, config: Any) -> None:
    """Render a dataclass (or mapping) as a key/value table."""
    if isinstance(config, Mapping):
        pairs = list(config.items())
    elif is_dataclass(config):
        pairs = [
            (f.name, getattr(config, f.name))
            for f in config.__dataclass_fields__.values()
        ]
    else:
        console.print(config)
        return
    key_value_table(title, pairs)


def pipeline_summary(
    title: str,
    timings: Mapping[str, float],
    *,
    total_label: str = "Total",
) -> None:
    """Render a summary table of per-stage timings.

    Mirrors the historical behaviour of emitting a blank line before the
    table and returning silently when ``timings`` is empty.
    """
    if not timings:
        return
    table = _pipeline_summary_table(timings, title=title, total_label=total_label)
    console.print()
    print_table(table)


def format_duration(seconds: float) -> str:
    """Public alias for the shared duration formatter."""
    return _format_duration(seconds)


def _precision_from_float_format(float_format: str) -> int:
    """Extract an integer precision from a legacy format spec like ``".4g"``."""
    digits = "".join(ch for ch in float_format if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            pass
    return 4


# Backwards-compatible re-exports of the private formatter names used by a
# few callers and tests that reached into the module.
__all__ = [
    "_format_duration",
    "_format_value",
    "config_table",
    "console",
    "dataframe_table",
    "format_duration",
    "heading",
    "key_value_table",
    "pipeline_summary",
    "print_panel",
    "print_table",
    "run_banner",
    "section",
]


def stage_report_sources(model_key: str, output_dir: str, *, docs_dir: str | None = None) -> list[str]:
    """Copy a model's report template and the shared includes into ``output_dir``.

    The report stage copies ``docs/models/<model>/index.qmd`` into the fitted
    output directory and renders it there, so a Quarto ``{{< include >}}`` in the
    template resolves relative to that directory, not to the repository. The
    shared prediction body the bivariate random-effects family transcludes
    (``docs/models/_bivariate_re_body.qmd``) therefore has to travel with the
    template, or every page that uses it renders with a missing-include error.
    Every ``_*.qmd`` under ``docs/models/`` is copied, so a second include needs
    no change here. Returns the destination paths, the template first.
    """
    import glob
    import os
    import shutil

    from vocab_growth import environment as local_env

    root = docs_dir or local_env.DOCS_DIR
    template = os.path.join(root, "models", model_key.lower(), "index.qmd")
    if not os.path.isfile(template):
        raise FileNotFoundError(f"Report template is missing: {template}")
    destinations = [os.path.join(output_dir, "index.qmd")]
    shutil.copy(template, destinations[0])
    for include in sorted(glob.glob(os.path.join(root, "models", "_*.qmd"))):
        destination = os.path.join(output_dir, os.path.basename(include))
        shutil.copy(include, destination)
        destinations.append(destination)
    return destinations
