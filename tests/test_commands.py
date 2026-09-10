"""Thumbnails, notes, views, search and stats."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import make_image
from vex import db
from vex.cli import app
from vex.commands import notes as notes_command

runner = CliRunner()


def run(*args: str, root: Path) -> str:
    result = runner.invoke(app, [*args, "--root", str(root)])
    assert result.exit_code == 0, result.output
    return result.output


def ingest_one(root: Path, inbox: Path, name: str = "Screenshot 2026-07-20 at 14.03.50.png") -> str:
    make_image(inbox / name, "invoice total")
    run("ingest", root=root)
    conn = db.connect(root)
    file_hash = conn.execute("SELECT hash FROM screenshots").fetchone()[0]
    conn.close()
    return file_hash


def classify_row(root: Path, file_hash: str, **values: object) -> None:
    conn = db.connect(root)
    db.update_screenshot(conn, file_hash, values)
    conn.close()


def test_thumbs_regenerates_when_forced(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    thumb = root / "thumbs" / f"{file_hash}.jpg"
    assert thumb.is_file()

    thumb.unlink()
    run("thumbs", root=root)
    assert thumb.is_file()

    output = run("thumbs", root=root)
    assert "Already present" in output


def test_search_finds_ocr_text(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, ocr_text="quarterly invoice for the plumber")

    output = run("search", "invoice", root=root)
    assert "library/2026/07" in output.replace("\n", "")

    empty = run("search", "unrelatedterm", root=root)
    assert "No matches" in empty


def test_search_json_output(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, ocr_text="quarterly invoice", category="receipt")

    output = run("search", "invoice", "--json", root=root)
    payload = json.loads(output)
    assert payload[0]["hash"] == file_hash
    assert payload[0]["category"] == "receipt"


def test_search_filters_by_category(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, ocr_text="quarterly invoice", category="receipt")

    assert json.loads(run("search", "invoice", "--json", "--category", "receipt", root=root))
    assert not json.loads(run("search", "invoice", "--json", "--category", "recipe", root=root))


def test_notes_render_frontmatter_and_text(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(
        root,
        file_hash,
        ocr_text="invoice total 249.00",
        category="receipt",
        caption="A plumbing invoice",
        tags=db.dump_json(["invoice", "trades"]),
        classify_confidence=0.91,
    )
    run("notes", root=root)

    note = root / "notes" / "receipt" / f"20260720-140350_{file_hash[:8]}.md"
    assert note.is_file()
    body = note.read_text(encoding="utf-8")
    assert 'category: "receipt"' in body
    assert 'tags: ["invoice", "trades"]' in body
    assert "confidence: 0.91" in body
    assert f"![thumb](../../thumbs/{file_hash}.jpg)" in body
    assert "invoice total 249.00" in body


def test_notes_render_a_recipe(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(
        root,
        file_hash,
        ocr_text="raw text",
        category="recipe",
        structured=db.dump_json(
            {
                "title": "Butter chicken",
                "ingredients": ["chicken", "butter"],
                "method": ["Marinate", "Cook"],
                "serves": "4",
                "source": "taste.com.au",
            }
        ),
    )
    run("notes", root=root)

    note = root / "notes" / "recipe" / f"20260720-140350_{file_hash[:8]}.md"
    body = note.read_text(encoding="utf-8")
    assert "# Butter chicken" in body
    assert "## Ingredients" in body
    assert "- chicken" in body
    assert "1. Marinate" in body
    assert "## Source text" in body


def test_notes_render_the_use_line(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(
        root,
        file_hash,
        category="article",
        structured=db.dump_json({"use": "A post angle about determinism."}),
    )
    run("notes", root=root)

    note = root / "notes" / "article" / f"20260720-140350_{file_hash[:8]}.md"
    body = note.read_text(encoding="utf-8")
    assert "## Use" in body
    assert "A post angle about determinism." in body


def test_notes_skip_existing_unless_forced(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="article")
    run("notes", root=root)

    note = root / "notes" / "article" / f"20260720-140350_{file_hash[:8]}.md"
    note.write_text("hand edited", encoding="utf-8")
    run("notes", root=root)
    assert note.read_text(encoding="utf-8") == "hand edited"

    run("notes", "--force", root=root)
    assert note.read_text(encoding="utf-8") != "hand edited"


def test_notes_are_removed_when_a_row_becomes_junk(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="article")
    run("notes", root=root)
    note = root / "notes" / "article" / f"20260720-140350_{file_hash[:8]}.md"
    assert note.is_file()

    classify_row(root, file_hash, is_junk=1)
    run("notes", root=root)
    assert not note.exists()


def test_note_path_uses_the_unclassified_folder(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    conn = db.connect(root)
    row = db.get_screenshot(conn, file_hash)
    conn.close()
    assert row is not None
    assert notes_command.note_path(root, row).parent.name == "_unclassified"


def test_views_rebuild_as_copies(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="receipt")
    run("views", root=root)

    copies = list((root / "views" / "receipt").glob("*.png"))
    assert len(copies) == 1
    assert not copies[0].is_symlink()

    # Rebuilding after a category change leaves no stale copy behind.
    classify_row(root, file_hash, category="article")
    run("views", root=root)
    assert not list((root / "views" / "receipt").glob("*.png"))
    assert len(list((root / "views" / "article").glob("*.png"))) == 1


def test_views_exclude_junk(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="receipt", is_junk=1)
    run("views", root=root)
    assert not (root / "views" / "receipt").exists()


def test_views_build_near_duplicate_pairs(root: Path, inbox: Path) -> None:
    ingest_one(root, inbox)
    make_image(
        inbox / "Screenshot 2026-07-20 at 14.03.51.png", "invoice total", box=(50, 60, 300, 201)
    )
    run("ingest", root=root)
    run("views", root=root)

    pairs = list((root / "views" / "_near-duplicates").glob("*"))
    assert len(pairs) == 1
    assert len(list(pairs[0].glob("*.png"))) == 2


def test_stats_reports_counts(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, ocr_text="text", category="receipt")

    output = run("stats", root=root)
    assert "Screenshots" in output
    assert "receipt" in output
    assert "2026-07" in output


def test_stats_on_an_empty_catalogue(root: Path) -> None:
    assert "empty" in run("stats", root=root)


def test_classify_on_an_empty_catalogue(root: Path) -> None:
    # The classify tiers themselves are covered in tests/test_classify.py.
    assert "Nothing to classify" in run("classify", root=root)


# ---------------------------------------------------------------------------
# Notes left behind when a row moves category
# ---------------------------------------------------------------------------


def note_files(root: Path) -> set[str]:
    """Every real note, relative to notes/, ignoring AppleDouble companions."""
    return {
        str(p.relative_to(root / "notes"))
        for p in (root / "notes").rglob("*.md")
        if not p.name.startswith(".")
    }


def test_reclassifying_a_row_leaves_no_note_behind(root: Path, inbox: Path) -> None:
    """The old category's copy is orphaned the moment the row moves."""
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="social-post", ocr_text="a post")
    run("notes", root=root)
    assert len(note_files(root)) == 1
    assert next(iter(note_files(root))).startswith("social-post/")

    classify_row(root, file_hash, category="article")
    output = run("notes", root=root)

    remaining = note_files(root)
    assert len(remaining) == 1
    assert next(iter(remaining)).startswith("article/")
    assert "Removed as stale" in output


def test_a_note_for_a_row_that_is_gone_is_removed(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="article", ocr_text="an idea")
    run("notes", root=root)

    conn = db.connect(root)
    conn.execute("DELETE FROM screenshots WHERE hash = ?", (file_hash,))
    conn.commit()
    conn.close()

    run("notes", root=root)
    assert note_files(root) == set()


def test_a_row_that_became_junk_loses_its_note(root: Path, inbox: Path) -> None:
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="article", ocr_text="an idea")
    run("notes", root=root)
    assert len(note_files(root)) == 1

    classify_row(root, file_hash, is_junk=1, category="junk")
    run("notes", root=root)
    assert note_files(root) == set()


def test_appledouble_files_are_left_alone(root: Path, inbox: Path) -> None:
    """exFAT metadata is not ours to read or delete."""
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="article", ocr_text="an idea")
    run("notes", root=root)

    stray = root / "notes" / "article" / "._20260720-140350_abcdef12.md"
    stray.write_bytes(b"\x00\x05\x16\x07finder metadata")
    orphan = root / "notes" / "social-post" / "20260101-000000_deadbeef.md"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("stale", encoding="utf-8")

    output = run("notes", root=root)

    assert stray.is_file()  # untouched, and not counted as a removal
    assert not orphan.exists()  # swept
    assert "Removed as stale" in output
    assert note_files(root) == {f"article/20260720-140350_{file_hash[:8]}.md"}


def test_a_category_run_does_not_sweep_other_categories(root: Path, inbox: Path) -> None:
    """Asking for one category means that category, not a tidy-up of everything."""
    file_hash = ingest_one(root, inbox)
    classify_row(root, file_hash, category="article", ocr_text="an idea")
    run("notes", root=root)

    orphan = root / "notes" / "social-post" / "20260101-000000_deadbeef.md"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("stale, but not in the category asked for", encoding="utf-8")

    run("notes", "--category", "article", root=root)
    assert orphan.is_file()

    run("notes", root=root)
    assert not orphan.exists()


def test_the_sweep_counts_what_it_removed(root: Path) -> None:
    notes_root = root / "notes" / "social-post"
    notes_root.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (notes_root / f"2026010{i}-000000_abcdef1{i}.md").write_text("stale", encoding="utf-8")
    (notes_root / "._hidden.md").write_bytes(b"metadata")

    removed = notes_command.sweep_stale(root, expected=set())
    assert removed == 3
    assert (notes_root / "._hidden.md").is_file()
