"""Summary statistics for the catalogue."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vex import config, db
from vex.console import console, counts_table, summary_table

BYTES_PER_MIB = 1024 * 1024
MIB_PER_GIB = 1024


def human_size(total_bytes: int) -> str:
    """Render a byte count in MiB or GiB."""
    mib = total_bytes / BYTES_PER_MIB
    if mib >= MIB_PER_GIB:
        return f"{mib / MIB_PER_GIB:.2f} GiB"
    return f"{mib:.1f} MiB"


def command(
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Show what is in the catalogue."""
    library_root = config.get_root(root)
    conn = db.connect(library_root)

    total = conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0]
    if not total:
        console.print("The catalogue is empty. Run 'vex ingest' first.")
        conn.close()
        return

    with_ocr = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE ocr_text IS NOT NULL"
    ).fetchone()[0]
    classified = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE category IS NOT NULL"
    ).fetchone()[0]
    junk = conn.execute("SELECT COUNT(*) FROM screenshots WHERE is_junk = 1").fetchone()[0]
    face_checked = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE faces IS NOT NULL"
    ).fetchone()[0]
    with_faces = conn.execute("SELECT COUNT(*) FROM screenshots WHERE faces > 0").fetchone()[0]
    pinned = conn.execute("SELECT COUNT(*) FROM screenshots WHERE keep = 1").fetchone()[0]
    from_photos = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE photos_uuid IS NOT NULL"
    ).fetchone()[0]
    deleted_from_photos = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE photos_deleted_at IS NOT NULL"
    ).fetchone()[0]
    purged, reclaimed = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM screenshots WHERE purged_at IS NOT NULL"
    ).fetchone()
    near_pairs = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE near_duplicate_of IS NOT NULL"
    ).fetchone()[0]
    total_bytes = conn.execute(
        "SELECT COALESCE(SUM(bytes), 0) FROM screenshots WHERE purged_at IS NULL"
    ).fetchone()[0]

    console.print(
        summary_table(
            "Catalogue",
            [
                ("Screenshots", str(total)),
                ("OCR coverage", f"{with_ocr} ({with_ocr * 100 // total}%)"),
                ("Classified", f"{classified} ({classified * 100 // total}%)"),
                ("Faces checked", f"{face_checked} ({face_checked * 100 // total}%)"),
                ("With a face", str(with_faces)),
                ("Marked junk", str(junk)),
                ("Pinned with keep", str(pinned)),
                ("Near-duplicate rows", str(near_pairs)),
                ("From the Photos library", str(from_photos)),
                ("Deleted from Photos", str(deleted_from_photos)),
                ("Purged originals", str(purged)),
                ("Reclaimed by purge", human_size(reclaimed)),
                ("Library size on disk", human_size(total_bytes)),
            ],
        )
    )

    by_source = conn.execute(
        "SELECT source, COUNT(*) FROM screenshots GROUP BY source ORDER BY COUNT(*) DESC"
    ).fetchall()
    console.print(counts_table("By source", "Source", [(r[0], r[1]) for r in by_source]))

    by_category = conn.execute(
        "SELECT COALESCE(category, 'unclassified'), COUNT(*) FROM screenshots "
        "GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    console.print(counts_table("By category", "Category", [(r[0], r[1]) for r in by_category]))

    by_month = conn.execute(
        "SELECT substr(captured_at, 1, 7), COUNT(*) FROM screenshots "
        "WHERE captured_at IS NOT NULL GROUP BY 1 ORDER BY 1 DESC LIMIT 24"
    ).fetchall()
    console.print(counts_table("By month", "Month", [(r[0], r[1]) for r in by_month]))
    conn.close()
