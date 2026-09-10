"""Delete library originals whose value is already held by the note and thumbnail."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from vex import capture, config, db, purge
from vex.commands.stats import human_size
from vex.console import console, counts_table

#: How often a real run reports where it is up to.
PROGRESS_EVERY = 200


def candidates_table(candidates: list[purge.Candidate]) -> Table:
    """What would be reclaimed, by category, with a total row."""
    table = Table(title="Purgeable originals", title_justify="left")
    table.add_column("Category")
    table.add_column("Files", justify="right")
    table.add_column("Reclaimed", justify="right")
    for name, files, size in purge.by_category(candidates):
        table.add_row(name, str(files), human_size(size))
    total_bytes = sum(c.bytes for c in candidates)
    table.add_section()
    table.add_row("Total", str(len(candidates)), human_size(total_bytes), style="bold")
    return table


def delete_originals(
    conn: sqlite3.Connection, root: Path, candidates: list[purge.Candidate]
) -> tuple[int, int, list[str]]:
    """Delete each original and mark its row purged. Returns (deleted, bytes, problems)."""
    deleted = 0
    reclaimed = 0
    problems: list[str] = []
    purged_at = capture.to_iso(datetime.now())  # noqa: DTZ005
    for index, candidate in enumerate(candidates, start=1):
        target = config.absolute_path(root, candidate.path)
        try:
            target.unlink()
        except OSError as exc:
            problems.append(f"{candidate.path}: {exc}")
            continue
        # The path stays as the historical location; purged_at is what says it is gone.
        db.update_screenshot(conn, candidate.hash, {"purged_at": purged_at})
        deleted += 1
        reclaimed += candidate.bytes
        if index % PROGRESS_EVERY == 0:
            console.print(f"Purged {index} of {len(candidates)}...")
    return deleted, reclaimed, problems


def command(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would go, without deleting. The default.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Actually delete the originals.")] = False,
    category: Annotated[
        list[str] | None,
        typer.Option("--category", help="Only purge these categories. Repeatable."),
    ] = None,
    min_text: Annotated[
        int,
        typer.Option("--min-text", help="Characters of recognised text a row needs to qualify."),
    ] = purge.DEFAULT_MIN_TEXT,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Delete originals that the note and thumbnail already capture."""
    # The root has to be resolved first: its taxonomy is what says what is purgeable.
    library_root = config.get_root(root)
    try:
        selection = purge.resolve_selection(category)
    except purge.CategorySelectionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    conn = db.connect(library_root)
    candidates, reasons = purge.plan(conn, library_root, min_text=min_text, categories=selection)

    if not candidates:
        console.print("Nothing is eligible to purge.")
        if reasons:
            console.print(counts_table("Why not", "Reason", purge.ordered_reasons(reasons)))
        conn.close()
        return

    console.print(candidates_table(candidates))
    if reasons:
        console.print(counts_table("Kept back", "Reason", purge.ordered_reasons(reasons)))

    if dry_run or not yes:
        total = human_size(sum(c.bytes for c in candidates))
        console.print(
            f"Dry run. {len(candidates)} originals would go, freeing {total}. "
            "Re-run with --yes to delete them."
        )
        conn.close()
        return

    deleted, reclaimed, problems = delete_originals(conn, library_root, candidates)
    noun = "original" if deleted == 1 else "originals"
    console.print(f"Purged {deleted} {noun}, reclaiming {human_size(reclaimed)}.")
    console.print("Thumbnails, notes and catalogue rows are untouched.")
    for problem in problems:
        console.print(f"[yellow]{problem}[/yellow]")
    conn.close()
