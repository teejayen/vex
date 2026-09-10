"""Pin rows so purge leaves their originals alone, and unpin them again."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vex import config, db
from vex.console import console

HASHES_ARGUMENT = typer.Argument(help="Full hashes, or unique prefixes of at least 8 characters.")
MIN_PREFIX = 8


def set_keep(hashes: list[str], root: Path | None, *, keep: bool) -> None:
    """Set or clear the keep pin on each named row."""
    library_root = config.get_root(root)
    conn = db.connect(library_root)
    changed = 0
    failed = 0
    for raw in hashes:
        if len(raw.strip()) < MIN_PREFIX:
            console.print(
                f"[red]'{raw}' is too short. Give at least {MIN_PREFIX} characters.[/red]"
            )
            failed += 1
            continue
        try:
            file_hash = db.resolve_hash(conn, raw)
        except db.HashPrefixError as exc:
            console.print(f"[red]{exc}[/red]")
            failed += 1
            continue
        db.update_screenshot(conn, file_hash, {"keep": 1 if keep else 0})
        changed += 1
        console.print(f"{file_hash[:8]} {'pinned' if keep else 'unpinned'}.")
    conn.close()
    console.print(f"{changed} changed, {failed} failed.")
    if failed:
        raise typer.Exit(code=1)


def keep_command(
    hashes: Annotated[list[str], HASHES_ARGUMENT],
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Pin rows so purge never deletes their originals."""
    set_keep(hashes, root, keep=True)


def unkeep_command(
    hashes: Annotated[list[str], HASHES_ARGUMENT],
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Remove the keep pin from rows."""
    set_keep(hashes, root, keep=False)
