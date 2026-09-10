"""Purge eligibility, the dry run, the real delete, and what survives it."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_image
from vex import config, db, purge
from vex.cli import app

runner = CliRunner()

LONG_TEXT = "a linkedin post worth keeping the words of " * 4


def add_row(
    conn: sqlite3.Connection,
    root: Path,
    file_hash: str,
    *,
    category: str = "social-post",
    ocr_text: str = LONG_TEXT,
    faces: int | None = 0,
    is_junk: int = 0,
    keep: int = 0,
    thumb: bool = True,
    original: bool = True,
    size: int = 4096,
) -> str:
    """Insert a row with real files behind it, so the purge checks have something to see."""
    relative = f"library/2026/07/{file_hash[:8]}.png"
    if original:
        make_image(root / relative, "content")
    if thumb:
        make_image(config.thumb_path(root, file_hash), "thumb")
    db.insert_screenshot(
        conn,
        {
            "hash": file_hash,
            "path": relative,
            "original_name": f"{file_hash[:8]}.png",
            "source": "mac",
            "captured_at": "2026-07-20T14:03:50",
            "ingested_at": "2026-07-21T09:00:00",
            "bytes": size,
            "ocr_text": ocr_text,
            "category": category,
            "faces": faces,
            "is_junk": is_junk,
            "keep": keep,
        },
    )
    return relative


def verdict(
    conn: sqlite3.Connection, root: Path, file_hash: str, **kwargs: object
) -> purge.Verdict:
    row = db.get_screenshot(conn, file_hash)
    return purge.evaluate(
        row,
        thumb_exists=config.thumb_path(root, file_hash).is_file(),
        file_exists=config.absolute_path(root, row["path"]).is_file(),
        **kwargs,
    )


def run(*args: str, root: Path) -> str:
    result = runner.invoke(app, [*args, "--root", str(root)])
    assert result.exit_code == 0, result.output
    return result.output


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_a_wordy_social_post_with_no_faces_is_eligible(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "a" * 64)
    assert verdict(conn, root, "a" * 64).eligible
    conn.close()


@pytest.mark.parametrize("category", ["recipe", "receipt", "message", "document", "other"])
def test_protected_categories_are_never_eligible(root: Path, category: str) -> None:
    conn = db.connect(root)
    add_row(conn, root, "b" * 64, category=category)
    result = verdict(conn, root, "b" * 64)
    assert not result.eligible
    assert result.reason == purge.REASON_CATEGORY
    conn.close()


def test_a_face_blocks_the_purge(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "c" * 64, faces=1)
    assert verdict(conn, root, "c" * 64).reason == purge.REASON_FACES
    conn.close()


def test_an_unchecked_row_is_not_treated_as_faceless(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "d" * 64, faces=None)
    assert verdict(conn, root, "d" * 64).reason == purge.REASON_UNCHECKED
    conn.close()


def test_thin_text_keeps_the_original(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "e" * 64, ocr_text="too short")
    assert verdict(conn, root, "e" * 64).reason == purge.REASON_THIN_TEXT
    # A lower bar lets the same row through.
    assert verdict(conn, root, "e" * 64, min_text=5).eligible
    conn.close()


def test_a_missing_thumbnail_keeps_the_original(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "f" * 64, thumb=False)
    assert verdict(conn, root, "f" * 64).reason == purge.REASON_NO_THUMBNAIL
    conn.close()


def test_junk_is_purgeable_with_no_text_at_all(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "1" * 64, category="junk", ocr_text="", is_junk=1)
    assert verdict(conn, root, "1" * 64).eligible
    # ... but not when there is a face in it.
    add_row(conn, root, "2" * 64, category="junk", ocr_text="", is_junk=1, faces=2)
    assert verdict(conn, root, "2" * 64).reason == purge.REASON_FACES
    conn.close()


def test_a_keep_pin_blocks_the_purge(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "3" * 64, keep=1)
    assert verdict(conn, root, "3" * 64).reason == purge.REASON_KEPT
    conn.close()


def test_category_selection_narrows_and_never_widens(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "4" * 64, category="social-post")
    add_row(conn, root, "5" * 64, category="article")
    chosen = purge.resolve_selection(["article"])
    assert not verdict(conn, root, "4" * 64, categories=chosen).eligible
    assert verdict(conn, root, "5" * 64, categories=chosen).eligible

    with pytest.raises(purge.CategorySelectionError, match="recipe"):
        purge.resolve_selection(["recipe"])
    conn.close()


def test_junk_selection_is_explicit(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "6" * 64, category="junk", ocr_text="", is_junk=1)
    assert verdict(conn, root, "6" * 64, categories=purge.resolve_selection(["junk"])).eligible
    assert not verdict(
        conn, root, "6" * 64, categories=purge.resolve_selection(["social-post"])
    ).eligible
    conn.close()


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def test_dry_run_reports_and_deletes_nothing(root: Path) -> None:
    conn = db.connect(root)
    relative = add_row(conn, root, "7" * 64)
    add_row(conn, root, "8" * 64, category="recipe")
    conn.close()

    output = run("purge", root=root)
    assert "Dry run" in output
    assert "social-post" in output
    assert "category not purgeable" in output
    assert (root / relative).is_file()

    conn = db.connect(root)
    assert db.get_screenshot(conn, "7" * 64)["purged_at"] is None
    conn.close()


def test_purge_deletes_only_the_original(root: Path) -> None:
    conn = db.connect(root)
    purgeable = add_row(conn, root, "9" * 64)
    protected = add_row(conn, root, "0" * 64, category="recipe")
    conn.close()

    output = run("purge", "--yes", root=root)
    assert "Purged 1 original," in output

    assert not (root / purgeable).exists()
    assert (root / protected).is_file()
    assert config.thumb_path(root, "9" * 64).is_file()

    conn = db.connect(root)
    row = db.get_screenshot(conn, "9" * 64)
    assert row["purged_at"] is not None
    # The path stays as the historical location, so the row is still findable.
    assert row["path"] == purgeable
    conn.close()

    # Running again finds nothing left to do.
    assert "Nothing is eligible" in run("purge", "--yes", root=root)


def test_purge_respects_a_keep_pin_set_from_the_cli(root: Path) -> None:
    conn = db.connect(root)
    relative = add_row(conn, root, "ab" + "c" * 62)
    conn.close()

    run("keep", "abcccccc", root=root)
    output = run("purge", "--yes", root=root)
    assert "Nothing is eligible" in output
    assert "pinned with keep" in output
    assert (root / relative).is_file()

    run("unkeep", "abcccccc", root=root)
    run("purge", "--yes", root=root)
    assert not (root / relative).exists()


def test_keep_rejects_an_unknown_or_short_prefix(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "d" * 64)
    conn.close()

    short = runner.invoke(app, ["keep", "dd", "--root", str(root)])
    assert short.exit_code == 1
    assert "too short" in short.output

    unknown = runner.invoke(app, ["keep", "eeeeeeee", "--root", str(root)])
    assert unknown.exit_code == 1
    assert "No catalogue row" in unknown.output


def test_purge_rejects_a_protected_category(root: Path) -> None:
    result = runner.invoke(app, ["purge", "--category", "receipt", "--root", str(root)])
    assert result.exit_code == 2
    assert "Not purgeable" in result.output


def test_junk_purge_over_a_synthetic_library(root: Path) -> None:
    conn = db.connect(root)
    junk = add_row(conn, root, "e" * 64, category="junk", ocr_text="", is_junk=1)
    post = add_row(conn, root, "f" * 64, category="social-post")
    conn.close()

    run("purge", "--category", "junk", "--yes", root=root)
    assert not (root / junk).exists()
    assert (root / post).is_file()
    assert config.thumb_path(root, "e" * 64).is_file()


# ---------------------------------------------------------------------------
# What the rest of the CLI does with a purged row
# ---------------------------------------------------------------------------


def test_notes_still_render_for_a_purged_row(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "1" * 64, category="article")
    db.update_screenshot(conn, "1" * 64, {"caption": "an idea worth keeping"})
    conn.close()

    run("purge", "--yes", root=root)
    run("notes", "--force", root=root)

    written = list((root / "notes" / "article").glob("*.md"))
    assert len(written) == 1
    body = written[0].read_text(encoding="utf-8")
    assert f"![thumb](../../thumbs/{'1' * 64}.jpg)" in body
    assert "purged:" in body
    assert "an idea worth keeping" in body


def test_views_skip_purged_rows_and_stats_reports_them(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "2" * 64, category="social-post", size=4096)
    add_row(conn, root, "3" * 64, category="recipe", size=1024)
    conn.close()

    run("purge", "--yes", root=root)
    views = run("views", root=root)
    assert "Purged (no original)" in views
    assert not (root / "views" / "social-post").exists()
    assert (root / "views" / "recipe").is_dir()

    stats = run("stats", root=root)
    assert "Purged originals" in stats
    assert "Reclaimed by purge" in stats


def test_search_marks_a_purged_row(root: Path) -> None:
    conn = db.connect(root)
    add_row(conn, root, "4" * 64, ocr_text=LONG_TEXT + " invoice")
    conn.close()

    run("purge", "--yes", root=root)
    import json  # noqa: PLC0415

    payload = json.loads(run("search", "invoice", "--json", root=root))
    assert payload[0]["purged"] is True
    assert "(purged)" in run("search", "invoice", root=root).replace("\n", "")


def test_reingesting_a_purged_file_is_a_duplicate(root: Path, inbox: Path, tmp_path: Path) -> None:
    name = "Screenshot 2026-07-20 at 14.03.50.png"
    make_image(inbox / name, "a long linkedin post")
    run("ingest", root=root)

    conn = db.connect(root)
    row = conn.execute("SELECT hash, path FROM screenshots").fetchone()
    file_hash, relative = row["hash"], row["path"]
    db.update_screenshot(
        conn, file_hash, {"category": "social-post", "ocr_text": LONG_TEXT, "faces": 0}
    )
    conn.close()

    # Keep the exact bytes so the same file can be offered up again afterwards.
    kept = tmp_path / "again.png"
    shutil.copy2(root / relative, kept)

    run("purge", "--yes", root=root)
    assert not (root / relative).exists()

    shutil.copy2(kept, inbox / name)
    output = run("ingest", root=root)
    assert "Duplicates" in output

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    after = db.get_screenshot(conn, file_hash)
    assert after["purged_at"] is not None
    assert not (root / relative).exists()
    actions = conn.execute(
        "SELECT action, detail FROM ingest_log WHERE hash = ? ORDER BY id", (file_hash,)
    ).fetchall()
    assert [a["action"] for a in actions] == ["added", "duplicate"]
    assert "purged" in actions[-1]["detail"]
    conn.close()

    # The re-offered file is parked in duplicates/, never deleted.
    assert list((root / "duplicates").glob("*.png"))
