"""Shared console, option types and small output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()

RootOption = Annotated[
    Path | None,
    typer.Option(
        "--root",
        help="Library root. Defaults to VEX_ROOT, then ~/Screenshots.",
    ),
]


def summary_table(title: str, rows: list[tuple[str, str]]) -> Table:
    """A simple two-column summary table."""
    table = Table(title=title, show_header=False, title_justify="left")
    table.add_column("Item", style="bold")
    table.add_column("Value", justify="right")
    for label, value in rows:
        table.add_row(label, value)
    return table


def counts_table(title: str, header: str, rows: list[tuple[str, int]]) -> Table:
    """A labelled count table."""
    table = Table(title=title, title_justify="left")
    table.add_column(header)
    table.add_column("Count", justify="right")
    for label, value in rows:
        table.add_row(str(label), str(value))
    return table
