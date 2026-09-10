"""Generate missing thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vex import config, db, thumbs
from vex.console import console, summary_table


def command(
    force: Annotated[
        bool, typer.Option("--force", help="Regenerate thumbnails that already exist.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Generate thumbnails for catalogued screenshots."""
    library_root = config.get_root(root)
    conn = db.connect(library_root)
    rows = conn.execute("SELECT hash, path FROM screenshots").fetchall()

    created = 0
    skipped = 0
    failed = 0
    with console.status("Generating thumbnails...") as status:
        for index, row in enumerate(rows, start=1):
            status.update(f"[{index}/{len(rows)}] {row['hash'][:8]}")
            destination = thumbs.thumb_path(library_root, row["hash"])
            if destination.exists() and not force:
                skipped += 1
                continue
            source = config.absolute_path(library_root, row["path"])
            if thumbs.generate(source, destination):
                created += 1
            else:
                failed += 1

    console.print(
        summary_table(
            "Thumbnails",
            [
                ("Created", str(created)),
                ("Already present", str(skipped)),
                ("Failed", str(failed)),
            ],
        )
    )
    conn.close()
