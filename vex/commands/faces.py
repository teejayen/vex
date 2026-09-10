"""Count the faces in each screenshot, so the purge command can spare them."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from vex import capture, config, db, faces
from vex.console import console, summary_table


@dataclass
class Counts:
    """Running totals for one face-detection pass."""

    checked: int = 0
    with_faces: int = 0
    no_thumbnail: int = 0
    failed: int = 0


def pending_rows(conn: sqlite3.Connection, *, force: bool, limit: int | None) -> list[sqlite3.Row]:
    """Rows still needing a face count, oldest capture first."""
    sql = "SELECT hash FROM screenshots"
    if not force:
        sql += " WHERE faces IS NULL"
    sql += " ORDER BY captured_at"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql).fetchall()


def detect_into_catalogue(
    conn: sqlite3.Connection,
    root: Path,
    rows: list[sqlite3.Row],
    detect: Callable[[Path], int],
    *,
    progress: Callable[[int, str], None] | None = None,
) -> Counts:
    """Run a detector over each row's thumbnail, committing one row at a time.

    The detector is passed in so the pass can be exercised without Vision.
    """
    counts = Counts()
    for index, row in enumerate(rows, start=1):
        file_hash = row["hash"]
        if progress is not None:
            progress(index, file_hash)
        thumb = config.thumb_path(root, file_hash)
        if not thumb.is_file():
            counts.no_thumbnail += 1
            continue
        try:
            found = int(detect(thumb))
        except (OSError, ValueError) as exc:
            counts.failed += 1
            console.print(f"[yellow]{file_hash[:8]}: {exc}[/yellow]")
            continue
        db.update_screenshot(
            conn,
            file_hash,
            {"faces": found, "faces_at": capture.to_iso(datetime.now())},  # noqa: DTZ005
        )
        counts.checked += 1
        if found:
            counts.with_faces += 1
    return counts


def command(
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only process this many rows.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-check rows that already have a face count.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Count faces in each screenshot using Apple Vision."""
    try:
        faces.require_vision()
    except faces.FaceDetectionUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    library_root = config.get_root(root)
    conn = db.connect(library_root)
    rows = pending_rows(conn, force=force, limit=limit)

    if not rows:
        console.print("Every row already has a face count.")
        conn.close()
        return

    with console.status("Detecting faces...") as status:

        def progress(index: int, file_hash: str) -> None:
            status.update(f"[{index}/{len(rows)}] {file_hash[:8]}")

        counts = detect_into_catalogue(
            conn, library_root, rows, faces.detect_faces, progress=progress
        )

    console.print(
        summary_table(
            "Faces",
            [
                ("Checked", str(counts.checked)),
                ("With a face", str(counts.with_faces)),
                ("No thumbnail", str(counts.no_thumbnail)),
                ("Failed", str(counts.failed)),
            ],
        )
    )
    conn.close()
