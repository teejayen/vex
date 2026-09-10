"""Rebuild the category folder views as copies of library files."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from vex import config, db, naming
from vex.console import console, summary_table

UNCLASSIFIED_DIR = "_unclassified"
NEAR_DUPLICATES_DIR = "_near-duplicates"


def clear_directory(path: Path) -> None:
    """Remove a generated views subdirectory if it exists."""
    if path.exists():
        shutil.rmtree(path)


def copy_into(root: Path, relative_path: str, destination_dir: Path, prefix: str = "") -> bool:
    """Copy one library file into a views directory. Returns False if the source is missing."""
    source = config.absolute_path(root, relative_path)
    if not source.is_file():
        return False
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = naming.safe_filename(f"{prefix}{source.name}")
    shutil.copy2(source, destination_dir / name)
    return True


def build_near_duplicate_pairs(conn: sqlite3.Connection, root: Path, views_root: Path) -> int:
    """Copy each near-duplicate pair into its own folder for review."""
    pairs_dir = views_root / NEAR_DUPLICATES_DIR
    rows = conn.execute(
        "SELECT hash, path, near_duplicate_of FROM screenshots "
        "WHERE near_duplicate_of IS NOT NULL AND is_junk = 0 AND purged_at IS NULL"
    ).fetchall()
    pairs = 0
    for row in rows:
        other = db.get_screenshot(conn, row["near_duplicate_of"])
        if other is None:
            continue
        pair_dir = pairs_dir / f"{row['hash'][:8]}_{other['hash'][:8]}"
        copy_into(root, row["path"], pair_dir, prefix="b_")
        copy_into(root, other["path"], pair_dir, prefix="a_")
        pairs += 1
    return pairs


def command(
    category: Annotated[
        str | None,
        typer.Option("--category", help="Only rebuild this category."),
    ] = None,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Rebuild views/ as category folders of copied files."""
    library_root = config.get_root(root)
    conn = db.connect(library_root)
    views_root = library_root / "views"

    # Purged rows have no original left to copy, so they are simply not in the views.
    sql = (
        "SELECT hash, path, category, near_duplicate_of FROM screenshots "
        "WHERE is_junk = 0 AND purged_at IS NULL"
    )
    params: list[str] = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    rows = conn.execute(sql, params).fetchall()

    if category:
        clear_directory(views_root / naming.safe_filename(category))
    else:
        for existing in sorted(views_root.glob("*")):
            if existing.is_dir():
                clear_directory(existing)

    purged = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE purged_at IS NOT NULL"
    ).fetchone()[0]
    copied = 0
    missing = 0
    unclassified = 0
    near_duplicate_pairs = 0

    for row in rows:
        row_category = row["category"] or UNCLASSIFIED_DIR
        if row_category == UNCLASSIFIED_DIR:
            unclassified += 1
        target = views_root / naming.safe_filename(row_category)
        if copy_into(library_root, row["path"], target):
            copied += 1
        else:
            missing += 1

    if not category:
        near_duplicate_pairs = build_near_duplicate_pairs(conn, library_root, views_root)

    console.print(
        summary_table(
            "Views",
            [
                ("Copied", str(copied)),
                ("Unclassified", str(unclassified)),
                ("Near-duplicate pairs", str(near_duplicate_pairs)),
                ("Purged (no original)", str(purged)),
                ("Missing library files", str(missing)),
            ],
        )
    )
    conn.close()
