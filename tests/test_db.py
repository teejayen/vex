"""Schema, migrations and the FTS index."""

from __future__ import annotations

from pathlib import Path

import pytest

from vex import db


def base_row(file_hash: str = "a" * 64, path: str = "library/2026/07/a.png") -> dict[str, object]:
    return {
        "hash": file_hash,
        "phash": "0" * 16,
        "path": path,
        "original_name": "Screenshot 2026-07-20 at 14.03.50.png",
        "source": "mac",
        "captured_at": "2026-07-20T14:03:50",
        "ingested_at": "2026-07-21T09:00:00",
    }


def test_connect_creates_schema_and_sets_version(root: Path) -> None:
    conn = db.connect(root)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"screenshots", "ingest_log", "classify_cache", "screenshots_fts"} <= tables
    conn.close()


def test_connect_is_idempotent(root: Path) -> None:
    db.connect(root).close()
    conn = db.connect(root)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    conn.close()


def test_migrate_refuses_a_newer_schema(root: Path) -> None:
    conn = db.connect(root)
    conn.execute(f"PRAGMA user_version={db.SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer than this build"):
        db.migrate(conn)
    conn.close()


def test_insert_and_fetch(root: Path) -> None:
    conn = db.connect(root)
    db.insert_screenshot(conn, base_row())
    assert db.hash_exists(conn, "a" * 64)
    row = db.get_screenshot(conn, "a" * 64)
    assert row is not None
    assert row["source"] == "mac"
    assert row["is_junk"] == 0
    conn.close()


def test_insert_rejects_unknown_columns(root: Path) -> None:
    conn = db.connect(root)
    with pytest.raises(ValueError, match="Unknown screenshot columns"):
        db.insert_screenshot(conn, {**base_row(), "nonsense": 1})
    conn.close()


def test_fts_follows_inserts_and_updates(root: Path) -> None:
    conn = db.connect(root)
    db.insert_screenshot(conn, base_row())
    db.update_screenshot(conn, "a" * 64, {"ocr_text": "quarterly invoice total"})

    matches = conn.execute(
        "SELECT s.hash FROM screenshots_fts JOIN screenshots s ON s.rowid = screenshots_fts.rowid "
        "WHERE screenshots_fts MATCH ?",
        ("invoice",),
    ).fetchall()
    assert [m[0] for m in matches] == ["a" * 64]

    db.update_screenshot(conn, "a" * 64, {"ocr_text": "something else entirely"})
    matches = conn.execute(
        "SELECT rowid FROM screenshots_fts WHERE screenshots_fts MATCH ?", ("invoice",)
    ).fetchall()
    assert matches == []
    conn.close()


def test_fts_follows_deletes(root: Path) -> None:
    conn = db.connect(root)
    db.insert_screenshot(conn, {**base_row(), "ocr_text": "unique marker word"})
    conn.execute("DELETE FROM screenshots WHERE hash = ?", ("a" * 64,))
    conn.commit()
    matches = conn.execute(
        "SELECT rowid FROM screenshots_fts WHERE screenshots_fts MATCH ?", ("marker",)
    ).fetchall()
    assert matches == []
    conn.close()


def test_classify_cache_upserts(root: Path) -> None:
    conn = db.connect(root)
    db.cache_classification(
        conn, "a" * 64, "some/model", "text", '{"category": "article"}', 10, 5, 0.1
    )
    db.cache_classification(
        conn, "a" * 64, "some/model", "text", '{"category": "recipe"}', 11, 6, 0.2
    )
    cached = db.get_cached_classification(conn, "a" * 64, "some/model", "text")
    assert cached is not None
    assert cached["response_json"] == '{"category": "recipe"}'
    assert cached["input_tokens"] == 11
    conn.close()


def test_ingest_log(root: Path) -> None:
    conn = db.connect(root)
    db.log_ingest(conn, "2026-07-21T09:00:00", "/somewhere/x.png", None, "skipped", "not an image")
    rows = conn.execute("SELECT action, detail FROM ingest_log").fetchall()
    assert rows[0]["action"] == "skipped"
    conn.close()


def test_json_helpers() -> None:
    assert db.load_json(None, []) == []
    assert db.load_json("not json", "fallback") == "fallback"
    assert db.load_json('["a"]') == ["a"]
    assert db.dump_json(None) is None
    assert db.dump_json(["a"]) == '["a"]'


def make_version_two_catalogue(root: Path) -> None:
    """Take a current catalogue back to version 2, populated, for a migration test."""
    conn = db.connect(root)
    db.insert_screenshot(
        conn, {**base_row(), "ocr_text": "quarterly invoice", "category": "receipt"}
    )
    db.insert_screenshot(conn, {**base_row("b" * 64, "library/2026/07/b.png"), "is_junk": 1})
    for column in ("faces", "faces_at", "keep", "purged_at", "photos_deleted_at"):
        conn.execute(f"ALTER TABLE screenshots DROP COLUMN {column}")
    conn.execute("DROP TABLE skipped_uuids")
    conn.execute("PRAGMA user_version=2")
    conn.commit()
    conn.close()


def test_migration_from_version_two_on_a_populated_catalogue(root: Path) -> None:
    make_version_two_catalogue(root)

    conn = db.connect(root)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION

    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert "skipped_uuids" in tables

    columns = {r[1] for r in conn.execute("PRAGMA table_info(screenshots)").fetchall()}
    assert {"faces", "faces_at", "keep", "purged_at", "photos_deleted_at"} <= columns

    rows = conn.execute("SELECT * FROM screenshots ORDER BY hash").fetchall()
    assert len(rows) == 2
    assert rows[0]["category"] == "receipt"
    # Existing rows arrive unchecked and unpinned, so purge leaves them alone.
    assert rows[0]["faces"] is None
    assert rows[0]["faces_at"] is None
    assert rows[0]["keep"] == 0
    assert rows[0]["purged_at"] is None

    # The FTS index survived the rewrite.
    matches = conn.execute(
        "SELECT rowid FROM screenshots_fts WHERE screenshots_fts MATCH ?", ("invoice",)
    ).fetchall()
    assert len(matches) == 1
    conn.close()


def test_resolve_hash_by_prefix(root: Path) -> None:
    conn = db.connect(root)
    db.insert_screenshot(conn, base_row())
    db.insert_screenshot(conn, base_row("ab" + "c" * 62, "library/2026/07/b.png"))
    db.insert_screenshot(conn, base_row("ab" + "d" * 62, "library/2026/07/c.png"))

    assert db.resolve_hash(conn, "a" * 64) == "a" * 64
    assert db.resolve_hash(conn, "abccc") == "ab" + "c" * 62

    with pytest.raises(db.HashPrefixError, match="more than one"):
        db.resolve_hash(conn, "ab")
    with pytest.raises(db.HashPrefixError, match="No catalogue row"):
        db.resolve_hash(conn, "zzzz")
    conn.close()


def test_skipped_uuids_round_trip(root: Path) -> None:
    conn = db.connect(root)
    assert db.skipped_uuids(conn) == set()

    db.record_skipped_uuid(conn, "UUID-1", "IMG_7376.DNG", "not an image", "2026-09-09T15:00:00")
    db.record_skipped_uuid(conn, "UUID-2", "IMG_1.MOV", "not an image", "2026-09-09T15:00:00")
    assert db.skipped_uuids(conn) == {"UUID-1", "UUID-2"}

    # Setting the same one aside twice updates rather than raising.
    db.record_skipped_uuid(conn, "UUID-1", "IMG_7376.DNG", "corrupt", "2026-09-09T16:00:00")
    row = conn.execute("SELECT * FROM skipped_uuids WHERE photos_uuid = 'UUID-1'").fetchone()
    assert row["reason"] == "corrupt"
    assert db.skipped_uuids(conn) == {"UUID-1", "UUID-2"}
    conn.close()
