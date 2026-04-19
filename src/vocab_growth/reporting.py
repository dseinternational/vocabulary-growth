# Copyright (c) 2026 Down Syndrome Education International and contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Console reporting helpers for the vocab_growth pipelines.

Provides a small set of rich-based primitives used by the fit pipelines to emit
consistently formatted, easy-to-scan diagnostics while a model fit is running.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import is_dataclass
from typing import Any

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value) >= 1e4 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.3e}"
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_value(v) for v in value)
    return str(value)


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m {secs:04.1f}s"


def run_banner(title: str, subtitle: str | None = None) -> None:
    """Top-of-run banner. Use once per model fit."""
    body = Text(title, style="bold green", justify="center")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style="dim")
    console.print()
    console.print(Panel(body, border_style="green", expand=True))


def heading(title: str, *, style: str = "bold green") -> None:
    """Section heading without timing. Use for non-stage sub-sections."""
    console.print()
    console.print(Rule(Text(title, style=style), style=style))


@contextmanager
def section(
    title: str,
    *,
    timings: dict[str, float] | None = None,
    key: str | None = None,
    style: str = "bold green",
):
    """
    Emit a section heading, run the block, and record its wall time.

    If ``timings`` is provided, records elapsed seconds under ``key`` (defaulting
    to ``title``). On exit, prints the stage duration so it is visible inline.
    """
    heading(title, style=style)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if timings is not None:
            timings[key or title] = elapsed
        console.print(
            Text(f"  ✓ {title} — {_format_duration(elapsed)}", style="dim green")
        )


def key_value_table(
    title: str,
    pairs: Iterable[tuple[str, Any]] | Mapping[str, Any],
    *,
    key_header: str = "Parameter",
    value_header: str = "Value",
) -> None:
    """Render a two-column table of (name, value) pairs."""
    items = pairs.items() if isinstance(pairs, Mapping) else list(pairs)

    table = Table(
        title=title,
        title_style="bold",
        title_justify="left",
        show_header=True,
        header_style="bold cyan",
        expand=False,
        pad_edge=False,
    )
    table.add_column(key_header, style="cyan", no_wrap=True)
    table.add_column(value_header, style="white")

    for k, v in items:
        table.add_row(str(k), _format_value(v))

    console.print(table)


def dataframe_table(
    df: pd.DataFrame,
    *,
    title: str | None = None,
    max_rows: int | None = None,
    float_format: str = ".4g",
    show_index: bool = True,
) -> None:
    """
    Render a pandas DataFrame as a rich Table.

    Long frames are truncated to ``max_rows`` (head/tail split) with a note.
    """
    if df.empty:
        console.print(Text(title or "(empty)", style="dim"))
        return

    truncated = False
    display_df = df
    if max_rows is not None and len(df) > max_rows:
        half = max(1, max_rows // 2)
        display_df = pd.concat([df.head(half), df.tail(max_rows - half)])
        truncated = True

    table = Table(
        title=title,
        title_style="bold",
        title_justify="left",
        show_header=True,
        header_style="bold cyan",
        expand=False,
        pad_edge=False,
    )

    if show_index:
        index_name = df.index.name or ""
        table.add_column(str(index_name), style="cyan", no_wrap=True)

    for col in display_df.columns:
        table.add_column(str(col), justify="right")

    def _fmt(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return format(value, float_format)
        return str(value)

    for idx, row in display_df.iterrows():
        cells = [_fmt(row[c]) for c in display_df.columns]
        if show_index:
            cells = [str(idx)] + cells
        table.add_row(*cells)

    console.print(table)
    if truncated:
        console.print(
            Text(
                f"  … showing {max_rows} of {len(df)} rows (head/tail) …",
                style="dim",
            )
        )


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
    """Render a summary table of per-stage timings."""
    if not timings:
        return

    total = sum(timings.values())

    table = Table(
        title=title,
        title_style="bold green",
        title_justify="left",
        show_header=True,
        header_style="bold cyan",
        expand=False,
        pad_edge=False,
    )
    table.add_column("Stage", style="cyan")
    table.add_column("Duration", justify="right")
    table.add_column("% total", justify="right", style="dim")

    for stage, seconds in timings.items():
        pct = (seconds / total * 100) if total > 0 else 0.0
        table.add_row(stage, _format_duration(seconds), f"{pct:5.1f}%")

    table.add_section()
    table.add_row(
        Text(total_label, style="bold"),
        Text(_format_duration(total), style="bold"),
        "",
    )

    console.print()
    console.print(table)


def format_duration(seconds: float) -> str:
    """Public alias for the internal duration formatter."""
    return _format_duration(seconds)
