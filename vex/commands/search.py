"""Full-text search over the catalogue."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from vex import config, db
from vex.console import console

DEFAULT_LIMIT = 20
SNIPPET_TOKENS = 20


def command(
    query: Annotated[str, typer.Argument(help="FTS5 query.")],
    category: Annotated[
        str | None, typer.Option("--category", help="Restrict to one category.")
    ] = None,
    faces: Annotated[
        bool | None,
        typer.Option(
            "--faces/--no-faces",
            help="Only rows with a face in them, or only rows with none.",
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Maximum results.")] = DEFAULT_LIMIT,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print results as JSON instead of a table.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Search screenshots by their text, caption, tags and original name."""
    library_root = config.get_root(root)
    conn = db.connect(library_root)

    sql = """
        SELECT s.hash, s.captured_at, s.category, s.caption, s.path, s.faces, s.purged_at,
               snippet(screenshots_fts, 0, '', '', ' ... ', ?) AS snippet,
               bm25(screenshots_fts) AS rank
        FROM screenshots_fts
        JOIN screenshots s ON s.rowid = screenshots_fts.rowid
        WHERE screenshots_fts MATCH ?
    """
    params: list[object] = [SNIPPET_TOKENS, query]
    if category:
        sql += " AND s.category = ?"
        params.append(category)
    if faces is not None:
        # Null means never checked, which is neither with nor without a face.
        sql += " AND s.faces > 0" if faces else " AND s.faces = 0"
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        console.print(f"[red]Search failed: {exc}[/red]")
        conn.close()
        raise typer.Exit(code=2) from exc

    if as_json:
        payload = [
            {
                "hash": row["hash"],
                "captured_at": row["captured_at"],
                "category": row["category"],
                "caption": row["caption"],
                "path": row["path"],
                "faces": row["faces"],
                "purged": bool(row["purged_at"]),
                "snippet": (row["snippet"] or "").strip(),
            }
            for row in rows
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        conn.close()
        return

    if not rows:
        console.print("No matches.")
        conn.close()
        return

    table = Table(title=f"Results for {query}", title_justify="left")
    table.add_column("Captured")
    table.add_column("Category")
    table.add_column("Caption")
    table.add_column("Snippet")
    table.add_column("Path")
    for row in rows:
        # A purged row is still a result; there is just no original behind the path.
        path = f"{row['path']} (purged)" if row["purged_at"] else row["path"]
        table.add_row(
            (row["captured_at"] or "").replace("T", " "),
            row["category"] or "",
            row["caption"] or "",
            " ".join((row["snippet"] or "").split()),
            path,
        )
    console.print(table)
    conn.close()
