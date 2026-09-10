"""Run text recognition over catalogued screenshots."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from vex import capture, config, db, ocr
from vex.console import console, summary_table

ENGINES = ("auto", "vision", "tesseract")


def command(
    engine: Annotated[str, typer.Option("--engine", help=f"One of {', '.join(ENGINES)}.")] = "auto",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only process this many rows.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run OCR on rows that already have text.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Extract text from vex."""
    if engine not in ENGINES:
        console.print(f"[red]Unknown engine '{engine}'. Choose one of {', '.join(ENGINES)}.[/red]")
        raise typer.Exit(code=2)

    try:
        resolved = ocr.resolve_engine(engine)
    except ocr.OcrUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    library_root = config.get_root(root)
    conn = db.connect(library_root)

    sql = "SELECT hash, path FROM screenshots"
    if not force:
        sql += " WHERE ocr_text IS NULL"
    sql += " ORDER BY captured_at"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()

    if not rows:
        console.print("Nothing to OCR.")
        conn.close()
        return

    processed = 0
    empty = 0
    failed = 0
    with console.status(f"Running {resolved} OCR...") as status:
        for index, row in enumerate(rows, start=1):
            status.update(f"[{index}/{len(rows)}] {row['hash'][:8]}")
            image_path = config.absolute_path(library_root, row["path"])
            try:
                text, engine_used = ocr.recognise(image_path, resolved)
            except (OSError, ValueError) as exc:
                failed += 1
                console.print(f"[yellow]{row['path']}: {exc}[/yellow]")
                continue
            text = text.strip()
            if not text:
                empty += 1
            db.update_screenshot(
                conn,
                row["hash"],
                {
                    "ocr_text": text,
                    "ocr_engine": engine_used,
                    "ocr_at": capture.to_iso(datetime.now()),  # noqa: DTZ005
                },
            )
            processed += 1

    console.print(
        summary_table(
            f"OCR ({resolved})",
            [
                ("Processed", str(processed)),
                ("Empty results", str(empty)),
                ("Failed", str(failed)),
            ],
        )
    )
    conn.close()
