"""Catalogue schema, migrations and connection handling."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 5

DB_FILENAME = "catalog.sqlite"

SCREENSHOT_COLUMNS = (
    "hash",
    "phash",
    "path",
    "original_name",
    "original_path",
    "source",
    "captured_at",
    "captured_at_source",
    "device",
    "width",
    "height",
    "bytes",
    "format",
    "ingested_at",
    "near_duplicate_of",
    "ocr_text",
    "ocr_engine",
    "ocr_at",
    "category",
    "tags",
    "caption",
    "classify_confidence",
    "classify_method",
    "classify_model",
    "classified_at",
    "structured",
    "is_junk",
    "reviewed",
    "notes",
    "photos_uuid",
    "faces",
    "faces_at",
    "keep",
    "purged_at",
    "photos_deleted_at",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS screenshots (
  hash TEXT PRIMARY KEY,
  phash TEXT,
  path TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL,
  original_path TEXT,
  source TEXT NOT NULL,
  captured_at TEXT,
  captured_at_source TEXT,
  device TEXT,
  width INTEGER, height INTEGER, bytes INTEGER, format TEXT,
  ingested_at TEXT NOT NULL,
  near_duplicate_of TEXT,
  ocr_text TEXT, ocr_engine TEXT, ocr_at TEXT,
  category TEXT,
  tags TEXT,
  caption TEXT,
  classify_confidence REAL,
  classify_method TEXT,
  classify_model TEXT,
  classified_at TEXT,
  structured TEXT,
  is_junk INTEGER NOT NULL DEFAULT 0,
  reviewed INTEGER NOT NULL DEFAULT 0,
  notes TEXT,
  photos_uuid TEXT,
  faces INTEGER,
  faces_at TEXT,
  keep INTEGER NOT NULL DEFAULT 0,
  purged_at TEXT,
  photos_deleted_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_screenshots_category ON screenshots(category);
CREATE INDEX IF NOT EXISTS idx_screenshots_captured_at ON screenshots(captured_at);
CREATE INDEX IF NOT EXISTS idx_screenshots_phash ON screenshots(phash);
CREATE INDEX IF NOT EXISTS idx_screenshots_photos_uuid ON screenshots(photos_uuid);

CREATE TABLE IF NOT EXISTS ingest_log (
  id INTEGER PRIMARY KEY, run_at TEXT, source_path TEXT, hash TEXT,
  action TEXT,
  detail TEXT
);

CREATE TABLE IF NOT EXISTS skipped_uuids (
  photos_uuid TEXT PRIMARY KEY,
  original_name TEXT,
  reason TEXT,
  skipped_at TEXT
);

CREATE TABLE IF NOT EXISTS classify_cache (
  hash TEXT, model TEXT, mode TEXT,
  response_json TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  created_at TEXT,
  PRIMARY KEY (hash, model, mode)
);

CREATE VIRTUAL TABLE IF NOT EXISTS screenshots_fts USING fts5(
  ocr_text, caption, tags, original_name,
  content='screenshots', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS screenshots_ai AFTER INSERT ON screenshots BEGIN
  INSERT INTO screenshots_fts(rowid, ocr_text, caption, tags, original_name)
  VALUES (new.rowid, new.ocr_text, new.caption, new.tags, new.original_name);
END;

CREATE TRIGGER IF NOT EXISTS screenshots_ad AFTER DELETE ON screenshots BEGIN
  INSERT INTO screenshots_fts(screenshots_fts, rowid, ocr_text, caption, tags, original_name)
  VALUES ('delete', old.rowid, old.ocr_text, old.caption, old.tags, old.original_name);
END;

CREATE TRIGGER IF NOT EXISTS screenshots_au AFTER UPDATE ON screenshots BEGIN
  INSERT INTO screenshots_fts(screenshots_fts, rowid, ocr_text, caption, tags, original_name)
  VALUES ('delete', old.rowid, old.ocr_text, old.caption, old.tags, old.original_name);
  INSERT INTO screenshots_fts(rowid, ocr_text, caption, tags, original_name)
  VALUES (new.rowid, new.ocr_text, new.caption, new.tags, new.original_name);
END;
"""


def db_path(root: Path) -> Path:
    """Location of the catalogue file under root."""
    return Path(root) / DB_FILENAME


def connect(root: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) the catalogue under root, migrated to the current schema."""
    path = db_path(Path(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    migrate(conn)
    return conn


# One entry per schema version above 1, holding the statements that upgrade to it.
# A fresh catalogue is created from SCHEMA_SQL and stamped at SCHEMA_VERSION, so
# these run only for a catalogue built by an earlier release.
MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: (
        "ALTER TABLE screenshots ADD COLUMN photos_uuid TEXT",
        "CREATE INDEX IF NOT EXISTS idx_screenshots_photos_uuid ON screenshots(photos_uuid)",
    ),
    3: (
        "ALTER TABLE screenshots ADD COLUMN faces INTEGER",
        "ALTER TABLE screenshots ADD COLUMN faces_at TEXT",
        "ALTER TABLE screenshots ADD COLUMN keep INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE screenshots ADD COLUMN purged_at TEXT",
    ),
    4: (
        (
            "CREATE TABLE IF NOT EXISTS skipped_uuids ("
            "  photos_uuid TEXT PRIMARY KEY,"
            "  original_name TEXT,"
            "  reason TEXT,"
            "  skipped_at TEXT"
            ")"
        ),
    ),
    5: ("ALTER TABLE screenshots ADD COLUMN photos_deleted_at TEXT",),
}


def migrate(conn: sqlite3.Connection) -> int:
    """Bring the schema up to SCHEMA_VERSION. Returns the version now in force."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA_SQL)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()
        return SCHEMA_VERSION

    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Catalogue schema version {version} is newer than this build supports "
            f"({SCHEMA_VERSION}). Upgrade the vex CLI."
        )

    while version < SCHEMA_VERSION:
        version += 1
        for statement in MIGRATIONS.get(version, ()):
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version={version}")
        conn.commit()
    return version


def rebuild_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the external-content FTS index from the screenshots table."""
    conn.execute("INSERT INTO screenshots_fts(screenshots_fts) VALUES ('rebuild')")
    conn.commit()


def insert_screenshot(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Insert one catalogue row. Keys must be a subset of SCREENSHOT_COLUMNS."""
    unknown = set(row) - set(SCREENSHOT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown screenshot columns: {sorted(unknown)}")
    columns = list(row)
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO screenshots ({', '.join(columns)}) VALUES ({placeholders})"  # noqa: S608
    conn.execute(sql, [row[c] for c in columns])
    conn.commit()


def update_screenshot(conn: sqlite3.Connection, file_hash: str, values: dict[str, Any]) -> None:
    """Update columns on one catalogue row by hash."""
    unknown = set(values) - set(SCREENSHOT_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown screenshot columns: {sorted(unknown)}")
    if not values:
        return
    assignments = ", ".join(f"{c} = ?" for c in values)
    conn.execute(
        f"UPDATE screenshots SET {assignments} WHERE hash = ?",  # noqa: S608
        [*values.values(), file_hash],
    )
    conn.commit()


def get_screenshot(conn: sqlite3.Connection, file_hash: str) -> sqlite3.Row | None:
    """Fetch one catalogue row by hash."""
    return conn.execute("SELECT * FROM screenshots WHERE hash = ?", (file_hash,)).fetchone()


def hash_exists(conn: sqlite3.Connection, file_hash: str) -> bool:
    """True if this exact file hash is already in the catalogue."""
    return (
        conn.execute("SELECT 1 FROM screenshots WHERE hash = ?", (file_hash,)).fetchone()
        is not None
    )


def all_phashes(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Every (hash, phash) pair with a perceptual hash recorded."""
    rows = conn.execute("SELECT hash, phash FROM screenshots WHERE phash IS NOT NULL").fetchall()
    return [(r["hash"], r["phash"]) for r in rows]


def log_ingest(
    conn: sqlite3.Connection,
    run_at: str,
    source_path: str,
    file_hash: str | None,
    action: str,
    detail: str | None = None,
) -> None:
    """Append a row to the ingest log."""
    conn.execute(
        "INSERT INTO ingest_log (run_at, source_path, hash, action, detail) VALUES (?, ?, ?, ?, ?)",
        (run_at, source_path, file_hash, action, detail),
    )
    conn.commit()


def cache_classification(
    conn: sqlite3.Connection,
    file_hash: str,
    model: str,
    mode: str,
    response_json: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    created_at: str | None = None,
) -> None:
    """Store a raw model response against (hash, model, mode)."""
    conn.execute(
        """
        INSERT INTO classify_cache
          (hash, model, mode, response_json, input_tokens, output_tokens, cost_usd, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hash, model, mode) DO UPDATE SET
          response_json=excluded.response_json,
          input_tokens=excluded.input_tokens,
          output_tokens=excluded.output_tokens,
          cost_usd=excluded.cost_usd,
          created_at=excluded.created_at
        """,
        (
            file_hash,
            model,
            mode,
            response_json,
            input_tokens,
            output_tokens,
            cost_usd,
            created_at,
        ),
    )
    conn.commit()


def get_cached_classification(
    conn: sqlite3.Connection, file_hash: str, model: str, mode: str
) -> sqlite3.Row | None:
    """Fetch a cached model response, if one exists."""
    return conn.execute(
        "SELECT * FROM classify_cache WHERE hash = ? AND model = ? AND mode = ?",
        (file_hash, model, mode),
    ).fetchone()


def load_json(value: str | None, default: Any = None) -> Any:
    """Parse a JSON column, returning default on null or malformed content."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def dump_json(value: Any) -> str | None:
    """Serialise a value for a JSON column. None stays None."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


class HashPrefixError(LookupError):
    """Raised when a hash prefix matches no row, or more than one."""


def resolve_hash(conn: sqlite3.Connection, prefix: str) -> str:
    """Turn a hash or unique hash prefix into the full hash of one catalogue row."""
    cleaned = prefix.strip().lower()
    if not cleaned:
        raise HashPrefixError("Empty hash.")
    matches = conn.execute(
        "SELECT hash FROM screenshots WHERE hash LIKE ? ORDER BY hash LIMIT 2",
        (cleaned + "%",),
    ).fetchall()
    if not matches:
        raise HashPrefixError(f"No catalogue row starts with '{prefix}'.")
    if len(matches) > 1:
        raise HashPrefixError(f"'{prefix}' matches more than one row. Give more characters.")
    return str(matches[0][0])


def record_skipped_uuid(
    conn: sqlite3.Connection,
    photos_uuid: str,
    original_name: str,
    reason: str,
    skipped_at: str,
) -> None:
    """Remember that this Photos asset was set aside, so it is never fetched again.

    Moving a file out of the inbox frees its UUID, because the sidecar that marked
    it exported goes with it. Without this the next export fetches the same
    unusable asset, ingest sets it aside again, and the two of them do that
    forever. One DNG went round twenty times before anyone noticed.
    """
    conn.execute(
        "INSERT INTO skipped_uuids (photos_uuid, original_name, reason, skipped_at) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(photos_uuid) DO UPDATE SET "
        "original_name=excluded.original_name, reason=excluded.reason, "
        "skipped_at=excluded.skipped_at",
        (photos_uuid, original_name, reason, skipped_at),
    )
    conn.commit()


def skipped_uuids(conn: sqlite3.Connection) -> set[str]:
    """Photos assets already set aside, which no export should offer again."""
    return {
        str(row[0])
        for row in conn.execute("SELECT photos_uuid FROM skipped_uuids").fetchall()
        if row[0]
    }
