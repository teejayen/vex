"""Generate one Markdown note per catalogued screenshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from vex import config, db, naming, taxonomy
from vex.console import console, summary_table

UNCLASSIFIED_DIR = "_unclassified"


def note_path(root: Path, row: sqlite3.Row) -> Path:
    """Location of the note for a catalogue row."""
    category = naming.safe_filename(row["category"] or UNCLASSIFIED_DIR)
    stamp = naming.timestamp_slug(row["captured_at"] or "")
    return root / "notes" / category / f"{stamp}_{row['hash'][:8]}.md"


def is_note(path: Path) -> bool:
    """A real note, not an AppleDouble the drive left beside one.

    exFAT cannot hold macOS metadata, so the Finder writes it into a companion
    ``._name`` file. Those are not ours to read or to delete.
    """
    return path.suffix == ".md" and not path.name.startswith(".")


def sweep_stale(root: Path, expected: set[Path]) -> int:
    """Delete notes no row claims any more. Returns how many went.

    Re-classifying a row writes its note under the new category and orphans the
    old one, so the folder quietly accumulates copies under categories the row has
    not been in for weeks. Anything outside the expected set has nothing behind it:
    a category that changed, a row that became junk, or a row that is gone.
    """
    notes_root = Path(root) / "notes"
    if not notes_root.is_dir():
        return 0
    removed = 0
    for path in sorted(notes_root.rglob("*.md")):
        if not is_note(path) or path in expected:
            continue
        path.unlink()
        removed += 1
    return removed


def yaml_value(value: object) -> str:
    """Render a frontmatter value, quoting strings that need it."""
    if value is None:
        return "null"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def frontmatter(row: sqlite3.Row) -> str:
    """YAML frontmatter for one note."""
    tags = db.load_json(row["tags"], []) or []
    tag_list = ", ".join(yaml_value(t) for t in tags)
    lines = [
        "---",
        f"hash: {yaml_value(row['hash'])}",
        f"captured: {yaml_value(row['captured_at'])}",
        f"source: {yaml_value(row['source'])}",
        f"category: {yaml_value(row['category'])}",
        f"tags: [{tag_list}]",
        f"caption: {yaml_value(row['caption'])}",
        f"confidence: {yaml_value(row['classify_confidence'])}",
        f"path: {yaml_value(row['path'])}",
    ]
    if row["purged_at"]:
        # The original is gone; the thumbnail below is what is left of it.
        lines.append(f"purged: {yaml_value(row['purged_at'])}")
    lines.append("---")
    return "\n".join(lines)


def recipe_body(structured: dict) -> list[str]:
    """Body sections for a recipe note."""
    lines: list[str] = []
    title = structured.get("title")
    if title:
        lines += [f"# {title}", ""]
    serves = structured.get("serves")
    if serves:
        lines += [f"Serves: {serves}", ""]
    ingredients = structured.get("ingredients") or []
    if ingredients:
        lines += ["## Ingredients", ""]
        lines += [f"- {item}" for item in ingredients]
        lines += [""]
    method = structured.get("method") or []
    if method:
        lines += ["## Method", ""]
        lines += [f"{i}. {step}" for i, step in enumerate(method, start=1)]
        lines += [""]
    source = structured.get("source")
    if source:
        lines += [f"Source: {source}", ""]
    return lines


def render(row: sqlite3.Row) -> str:
    """Render the full Markdown note for a row."""
    category = row["category"] or ""
    shape = taxonomy.active().shape(category)
    is_recipe = shape == taxonomy.SHAPE_RECIPE
    structured = db.load_json(row["structured"], None)
    lines = [frontmatter(row), ""]

    if is_recipe and isinstance(structured, dict):
        lines += recipe_body(structured)

    lines += [f"![thumb](../../thumbs/{row['hash']}.jpg)", ""]

    if row["caption"]:
        lines += [row["caption"], ""]

    use_note = structured.get("use") if isinstance(structured, dict) else None
    if shape == taxonomy.SHAPE_USE_NOTE and use_note:
        lines += ["## Use", "", use_note, ""]

    heading = "## Source text" if is_recipe else "## Text"
    lines += [heading, "", "```", row["ocr_text"] or "", "```", ""]
    return "\n".join(lines)


def command(
    category: Annotated[
        str | None, typer.Option("--category", help="Only write notes for this category.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Rewrite notes that already exist.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Write Markdown notes for catalogued screenshots."""
    library_root = config.get_root(root)
    conn = db.connect(library_root)

    sql = "SELECT * FROM screenshots WHERE is_junk = 0"
    params: list[str] = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    rows = conn.execute(sql, params).fetchall()

    written = 0
    skipped = 0
    for row in rows:
        destination = note_path(library_root, row)
        if destination.exists() and not force:
            skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render(row), encoding="utf-8")
        written += 1

    if category:
        # Scoped to one category, so only that category's junk is ours to tidy.
        removed = 0
        junk_rows = conn.execute(
            "SELECT * FROM screenshots WHERE is_junk = 1 AND category = ?", [category]
        ).fetchall()
        for row in junk_rows:
            destination = note_path(library_root, row)
            if destination.exists():
                destination.unlink()
                removed += 1
    else:
        # A full run knows every note that should exist, so anything else is stale.
        removed = sweep_stale(library_root, {note_path(library_root, row) for row in rows})

    console.print(
        summary_table(
            "Notes",
            [
                ("Written", str(written)),
                ("Already present", str(skipped)),
                ("Removed as stale", str(removed)),
            ],
        )
    )
    conn.close()
