"""Photos export and delete, the schema migration, and the sidecar path."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_image
from vex import capture, db, photos
from vex.cli import app
from vex.commands import photos as photos_command

runner = CliRunner()


def write_sidecar(image: Path, **values: object) -> Path:
    """Write a sidecar of the shape photos export produces."""
    payload = {
        "photos_uuid": "2CF6584B-E1C2-4913-9B12-F2D712608243",
        "original_filename": image.name,
        "date": "2022-05-28T17:16:54+10:00",
        "device": None,
        "source": "photos",
        "exported_at": "2026-09-07T22:10:59",
    }
    payload.update(values)
    sidecar = image.with_suffix(image.suffix + ".json")
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


# --- schema migration -------------------------------------------------------


V1_SCHEMA = """
CREATE TABLE screenshots (
  hash TEXT PRIMARY KEY, phash TEXT, path TEXT NOT NULL UNIQUE,
  original_name TEXT NOT NULL, original_path TEXT, source TEXT NOT NULL,
  captured_at TEXT, captured_at_source TEXT, device TEXT,
  width INTEGER, height INTEGER, bytes INTEGER, format TEXT,
  ingested_at TEXT NOT NULL, near_duplicate_of TEXT,
  ocr_text TEXT, ocr_engine TEXT, ocr_at TEXT,
  category TEXT, tags TEXT, caption TEXT, classify_confidence REAL,
  classify_method TEXT, classify_model TEXT, classified_at TEXT, structured TEXT,
  is_junk INTEGER NOT NULL DEFAULT 0, reviewed INTEGER NOT NULL DEFAULT 0, notes TEXT
);
"""


def test_migration_adds_photos_uuid_to_an_existing_catalogue(root: Path) -> None:
    path = db.db_path(root)
    conn = sqlite3.connect(path)
    conn.executescript(V1_SCHEMA)
    conn.execute(
        "INSERT INTO screenshots (hash, path, original_name, source, ingested_at) "
        "VALUES ('a', 'library/2026/07/a.png', 'a.png', 'mac', '2026-07-20T00:00:00')"
    )
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()

    migrated = db.connect(root)
    assert migrated.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    columns = {r[1] for r in migrated.execute("PRAGMA table_info(screenshots)").fetchall()}
    assert "photos_uuid" in columns
    # The existing row survives, with the new column empty.
    row = migrated.execute("SELECT photos_uuid FROM screenshots WHERE hash = 'a'").fetchone()
    assert row["photos_uuid"] is None
    migrated.close()


def test_a_fresh_catalogue_is_stamped_at_the_current_version(root: Path) -> None:
    conn = db.connect(root)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    columns = {r[1] for r in conn.execute("PRAGMA table_info(screenshots)").fetchall()}
    assert "photos_uuid" in columns
    conn.close()


# --- sidecars ---------------------------------------------------------------


def test_read_sidecar(tmp_path: Path) -> None:
    image = make_image(tmp_path / "IMG_0013.PNG")
    write_sidecar(image)
    data = capture.read_sidecar(image)
    assert data is not None
    assert data["photos_uuid"] == "2CF6584B-E1C2-4913-9B12-F2D712608243"


def test_read_sidecar_without_one(tmp_path: Path) -> None:
    assert capture.read_sidecar(make_image(tmp_path / "a.png")) is None


def test_read_sidecar_ignores_malformed_json(tmp_path: Path) -> None:
    image = make_image(tmp_path / "a.png")
    image.with_suffix(".png.json").write_text("{not json", encoding="utf-8")
    assert capture.read_sidecar(image) is None


def test_sidecar_date_is_used_when_the_image_carries_none(tmp_path: Path) -> None:
    image = make_image(tmp_path / "IMG_0013.PNG")
    write_sidecar(image)
    info = capture.extract(image)
    assert info.captured_at == "2022-05-28T17:16:54"
    assert info.source == "sidecar"


# --- ingest carrying the UUID through ---------------------------------------


def test_ingest_records_the_photos_uuid_and_clears_the_sidecar(root: Path) -> None:
    inbox = root / "inbox" / "iphone"
    image = make_image(inbox / "IMG_0013.PNG")
    sidecar = write_sidecar(image)

    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    row = conn.execute("SELECT photos_uuid, source, captured_at FROM screenshots").fetchone()
    conn.close()
    assert row["photos_uuid"] == "2CF6584B-E1C2-4913-9B12-F2D712608243"
    assert row["source"] == "iphone"
    assert row["captured_at"] == "2022-05-28T17:16:54"
    assert not sidecar.exists()


def test_sidecars_are_never_counted_as_skipped(root: Path) -> None:
    inbox = root / "inbox" / "iphone"
    image = make_image(inbox / "IMG_0013.PNG")
    write_sidecar(image)

    runner.invoke(app, ["ingest", "--root", str(root)])
    conn = db.connect(root)
    actions = dict(
        conn.execute("SELECT action, COUNT(*) FROM ingest_log GROUP BY action").fetchall()
    )
    conn.close()
    assert actions.get("skipped") is None
    assert actions["added"] == 1


def test_a_sidecar_outside_the_inbox_is_left_alone(root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "desktop"
    image = make_image(elsewhere / "IMG_0013.PNG")
    sidecar = write_sidecar(image)

    runner.invoke(app, ["ingest", str(elsewhere), "--root", str(root)])
    assert sidecar.exists()


def test_device_falls_back_to_the_sidecar(root: Path) -> None:
    inbox = root / "inbox" / "iphone"
    image = make_image(inbox / "IMG_0013.PNG")
    write_sidecar(image, device="iPhone 13 Pro")

    runner.invoke(app, ["ingest", "--root", str(root)])
    conn = db.connect(root)
    row = conn.execute("SELECT device FROM screenshots").fetchone()
    conn.close()
    assert row["device"] == "iPhone 13 Pro"


# --- module helpers ---------------------------------------------------------


def test_chunked() -> None:
    assert photos.chunked(list(range(5)), 2) == [[0, 1], [2, 3], [4]]
    assert photos.chunked([], 2) == []


def test_exported_uuids_reads_the_inbox(tmp_path: Path) -> None:
    inbox = tmp_path / "iphone"
    inbox.mkdir()
    image = make_image(inbox / "IMG_0013.PNG")
    write_sidecar(image)
    (inbox / "junk.json").write_text("not json", encoding="utf-8")

    assert photos.exported_uuids(inbox) == {"2CF6584B-E1C2-4913-9B12-F2D712608243"}


def test_exported_uuids_on_a_missing_directory(tmp_path: Path) -> None:
    assert photos.exported_uuids(tmp_path / "nope") == set()


def test_sidecar_content_shape() -> None:
    class FakePhoto:
        uuid = "UUID-1"
        original_filename = "IMG_0013.PNG"
        date = None
        exif_info = None

    content = photos.sidecar_content(FakePhoto(), "2026-09-07T22:10:59")
    assert set(content) == {
        "photos_uuid",
        "original_filename",
        "date",
        "device",
        "source",
        "exported_at",
    }
    assert content["source"] == "photos"
    assert content["date"] is None


# --- the delete command -----------------------------------------------------


def catalogue_a_photos_row(root: Path, *, ocr: bool = True, on_disk: bool = True) -> str:
    """Put one row in the catalogue that came from Photos."""
    inbox = root / "inbox" / "iphone"
    image = make_image(inbox / "IMG_0013.PNG")
    write_sidecar(image)
    runner.invoke(app, ["ingest", "--root", str(root)])

    conn = db.connect(root)
    row = conn.execute("SELECT hash, path FROM screenshots").fetchone()
    if ocr:
        db.update_screenshot(conn, row["hash"], {"ocr_text": "some text"})
    conn.close()
    if not on_disk:
        (root / row["path"]).unlink()
    return row["hash"]


def test_delete_lists_but_does_nothing_without_yes(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalogue_a_photos_row(root)
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    called: list[list[str]] = []
    monkeypatch.setattr(photos, "delete_assets", lambda uuids: called.append(uuids) or len(uuids))

    result = runner.invoke(app, ["photos", "delete", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "Eligible for deletion" in result.output
    assert "Nothing was deleted" in result.output
    assert called == []


def test_delete_acts_only_with_yes(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalogue_a_photos_row(root)
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    called: list[list[str]] = []

    def fake_delete(uuids: list[str]) -> int:
        called.append(uuids)
        return len(uuids)

    monkeypatch.setattr(photos, "delete_assets", fake_delete)

    result = runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert called == [["2CF6584B-E1C2-4913-9B12-F2D712608243"]]
    assert "Recently Deleted" in result.output


def test_delete_skips_rows_without_text(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalogue_a_photos_row(root, ocr=False)
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    called: list[list[str]] = []
    monkeypatch.setattr(photos, "delete_assets", lambda uuids: called.append(uuids) or 0)

    result = runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])
    assert "Nothing to delete" in result.output
    assert called == []


def test_delete_skips_rows_whose_library_file_is_gone(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalogue_a_photos_row(root, on_disk=False)
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    called: list[list[str]] = []
    monkeypatch.setattr(photos, "delete_assets", lambda uuids: called.append(uuids) or 0)

    result = runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])
    assert "Nothing to delete" in result.output
    assert called == []


def test_delete_never_touches_a_row_without_a_uuid(
    root: Path, inbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    runner.invoke(app, ["ingest", "--root", str(root)])
    conn = db.connect(root)
    row = conn.execute("SELECT hash FROM screenshots").fetchone()
    db.update_screenshot(conn, row["hash"], {"ocr_text": "text"})
    conn.close()

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    called: list[list[str]] = []
    monkeypatch.setattr(photos, "delete_assets", lambda uuids: called.append(uuids) or 0)

    runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])
    assert called == []


def test_photos_commands_report_when_unavailable(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable() -> None:
        raise photos.PhotosUnavailableError("The Photos commands only run on macOS.")

    monkeypatch.setattr(photos_command.photos, "require_photos", unavailable)
    for args in (["photos", "export"], ["photos", "delete"]):
        result = runner.invoke(app, [*args, "--root", str(root)])
        assert result.exit_code == 1
        assert "only run on macOS" in result.output


# ---------------------------------------------------------------------------
# The machine-readable line the backlog loop reads
# ---------------------------------------------------------------------------


def roomy() -> photos.DiskCheck:
    """Plenty of space on both volumes."""
    return photos.DiskCheck(root_gb=100.0, photos_gb=100.0, same_volume=True)


def cramped() -> photos.DiskCheck:
    """A full internal disk, which is what stops Photos serving originals."""
    return photos.DiskCheck(root_gb=4.1, photos_gb=4.1, same_volume=True)


class FakePhoto:
    """Just enough of an osxphotos photo for the export command."""

    def __init__(self, uuid: str, uti: str = "public.png", *, ismovie: bool = False) -> None:
        self.uuid = uuid
        self.original_filename = f"{uuid}.PNG"
        self.uti = uti
        self.ismovie = ismovie


def test_export_reports_what_is_left_even_when_it_writes_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The count that matters is what is still waiting, not what got written.

    An export whose downloads all fail writes nothing, exactly like a year that
    is finished. Only this line tells the two apart.
    """
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: roomy())
    monkeypatch.setattr(
        photos,
        "load_screenshots",
        lambda _year: ([FakePhoto(f"U{i}") for i in range(5)], "test selector"),
    )
    # Every export fails, the way an iCloud download does under pressure.
    monkeypatch.setattr(photos, "export_one", lambda *_args: None)

    result = runner.invoke(
        app, ["photos", "export", "--year", "2023", "--limit", "2", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output

    reported = photos.parse_result(result.output)
    assert reported is not None
    assert reported.detected == 5
    assert reported.new == 5
    assert reported.pending == 2
    assert reported.exported == 0
    assert reported.failed == 2
    # Nothing was written, but the year is plainly not finished.
    assert not reported.drained


def test_export_reports_a_drained_year(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: roomy())
    monkeypatch.setattr(photos, "load_screenshots", lambda _year: ([], "test selector"))

    result = runner.invoke(app, ["photos", "export", "--year", "2023", "--root", str(root)])
    assert result.exit_code == 0, result.output

    reported = photos.parse_result(result.output)
    assert reported is not None
    assert reported.drained
    assert "Nothing new to export" in result.output


def test_a_dry_run_still_reports_what_is_waiting(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(
        photos,
        "load_screenshots",
        lambda _year: ([FakePhoto("U1"), FakePhoto("U2")], "test selector"),
    )
    called: list[object] = []
    monkeypatch.setattr(photos, "export_one", lambda *args: called.append(args))

    result = runner.invoke(
        app, ["photos", "export", "--year", "2023", "--dry-run", "--root", str(root)]
    )
    reported = photos.parse_result(result.output)
    assert reported is not None
    assert reported.new == 2
    assert reported.exported == 0
    assert called == []


def test_export_refuses_below_the_disk_floor(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A manual run gets the same explanation the overnight loop does."""
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: cramped())
    loaded: list[object] = []
    monkeypatch.setattr(
        photos, "load_screenshots", lambda year: (loaded.append(year) or [], "test selector")
    )

    result = runner.invoke(app, ["photos", "export", "--year", "2023", "--root", str(root)])

    assert result.exit_code == 1
    # Rich wraps the message, so compare against the flattened text.
    flat = " ".join(result.output.split())
    assert "Disk space is very low" in flat
    assert f"CloudPhotoLibrary error {photos.CLOUD_LOW_DISK_CODE}" in flat
    # It refuses before paying to load the Photos database.
    assert loaded == []


def test_a_dry_run_still_works_below_the_floor(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Planning downloads nothing, so a full disk is no reason to refuse it."""
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: cramped())
    monkeypatch.setattr(
        photos, "load_screenshots", lambda _year: ([FakePhoto("U1")], "test selector")
    )

    result = runner.invoke(
        app, ["photos", "export", "--year", "2023", "--dry-run", "--root", str(root)]
    )

    assert result.exit_code == 0
    reported = photos.parse_result(result.output)
    assert reported is not None
    assert reported.new == 1


# ---------------------------------------------------------------------------
# Two volumes, once the root moves to a drive
# ---------------------------------------------------------------------------


def check(root_gb: float, photos_gb: float, *, same: bool = False) -> photos.DiskCheck:
    return photos.DiskCheck(root_gb=root_gb, photos_gb=photos_gb, same_volume=same)


def test_a_roomy_drive_does_not_excuse_a_full_internal_disk() -> None:
    """The drive holds the library; the Mac holds the pictures. Only one downloads."""
    blocked = check(root_gb=1000.0, photos_gb=5.0)
    assert blocked.blocked == "photos"
    assert blocked.near_floor

    # 8 GB clears the 6 GB floor but is still inside the 2 GB headroom, so a wall
    # of failures there is read as the disk rather than the network. That is the
    # state the Mac is expected to be in once the root moves to the drive.
    borderline = check(root_gb=1000.0, photos_gb=8.0)
    assert borderline.blocked is None
    assert borderline.near_floor

    fine = check(root_gb=1000.0, photos_gb=30.0)
    assert fine.blocked is None
    assert not fine.near_floor


def test_a_roomy_separate_root_volume_skips_the_root_floor() -> None:
    """A terabyte of drive is not a constraint worth enforcing at 10 GB."""
    on_a_drive = check(root_gb=1000.0, photos_gb=20.0)
    assert not on_a_drive.root_floor_applies
    assert on_a_drive.blocked is None

    # A small separate volume is still checked.
    small_drive = check(root_gb=4.0, photos_gb=20.0)
    assert small_drive.root_floor_applies
    assert small_drive.blocked == "root"


def test_one_disk_for_both_keeps_the_root_floor() -> None:
    together = check(root_gb=8.0, photos_gb=8.0, same=True)
    assert together.root_floor_applies
    assert together.blocked == "root"  # 8 clears the 6 GB Photos floor but not the 10 GB one


def test_the_lower_volume_is_the_one_that_decides() -> None:
    assert check(root_gb=4.0, photos_gb=100.0).blocked == "root"
    assert check(root_gb=100.0, photos_gb=4.0).blocked == "photos"
    assert check(root_gb=100.0, photos_gb=100.0).blocked is None


def test_build_check_measures_both_paths(tmp_path: Path) -> None:
    library = tmp_path / "Pictures" / "Photos Library.photoslibrary"
    library.mkdir(parents=True)
    seen: list[Path] = []

    def fake_free(path: Path) -> int:
        seen.append(path)
        return int((1000 if "Pictures" not in str(path) else 7) * photos.BYTES_PER_GB)

    measured = photos.build_check(tmp_path / "root", photos_library=library, free_bytes=fake_free)
    assert len(seen) == 2
    assert measured.root_gb == 1000.0
    assert measured.photos_gb == 7.0


def test_build_check_walks_up_to_a_library_that_is_not_there(tmp_path: Path) -> None:
    """A missing library still has a volume; measure the nearest parent that exists."""
    measured = photos.build_check(
        tmp_path, photos_library=tmp_path / "gone" / "Photos Library.photoslibrary"
    )
    assert measured.photos_gb > 0


# ---------------------------------------------------------------------------
# Photos' screenshot flag is broader than the word suggests
# ---------------------------------------------------------------------------


def test_still_images_we_can_read_are_kept() -> None:
    keep, set_aside = photos.split_by_type(
        [
            FakePhoto("A", "public.png"),
            FakePhoto("B", "public.jpeg"),
            FakePhoto("C", "public.heic"),
        ]
    )
    assert [p.uuid for p in keep] == ["A", "B", "C"]
    assert set_aside == {}


def test_raw_images_and_movies_are_set_aside_and_counted() -> None:
    """Exporting one only parks a file in the inbox that nothing can read."""
    keep, set_aside = photos.split_by_type(
        [
            FakePhoto("A", "public.png"),
            FakePhoto("B", "com.adobe.raw-image"),
            FakePhoto("C", "com.adobe.raw-image"),
            FakePhoto("D", "com.apple.quicktime-movie", ismovie=True),
        ]
    )
    assert [p.uuid for p in keep] == ["A"]
    assert set_aside == {"com.adobe.raw-image": 2, "movie": 1}


def test_an_unknown_type_is_set_aside_rather_than_assumed_readable() -> None:
    """An allow-list: a type we have not met is counted, not silently exported."""
    keep, set_aside = photos.split_by_type([FakePhoto("A", "com.example.something-new")])
    assert keep == []
    assert set_aside == {"com.example.something-new": 1}


def test_export_never_offers_a_movie(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: roomy())
    monkeypatch.setattr(
        photos,
        "load_screenshots",
        lambda _year: (
            [
                FakePhoto("KEEP", "public.heic"),
                FakePhoto("SKIP", "com.apple.quicktime-movie", ismovie=True),
            ],
            "test selector",
        ),
    )
    offered: list[str] = []
    monkeypatch.setattr(
        photos, "export_one", lambda photo, *_args: offered.append(photo.uuid) or None
    )

    result = runner.invoke(app, ["photos", "export", "--year", "2023", "--root", str(root)])
    assert result.exit_code == 0, result.output

    assert offered == ["KEEP"]
    reported = photos.parse_result(result.output)
    assert reported is not None
    assert reported.detected == 1  # the movie is not a screenshot we count
    assert "set aside: 1 movie" in " ".join(result.output.split())


# ---------------------------------------------------------------------------
# A purged row is not a missing one
# ---------------------------------------------------------------------------


def mark_purged(root: Path, file_hash: str, *, keep_thumbnail: bool = True) -> None:
    """Delete the original the way purge does, leaving the note and thumbnail."""
    conn = db.connect(root)
    row = db.get_screenshot(conn, file_hash)
    (root / row["path"]).unlink(missing_ok=True)
    db.update_screenshot(conn, file_hash, {"purged_at": "2026-09-09T10:00:00"})
    conn.close()
    if not keep_thumbnail:
        (root / "thumbs" / f"{file_hash}.jpg").unlink(missing_ok=True)


def delete_rows(root: Path) -> list:
    conn = db.connect(root)
    rows = conn.execute(
        "SELECT hash, photos_uuid, path, captured_at, purged_at FROM screenshots "
        "WHERE photos_uuid IS NOT NULL AND ocr_text IS NOT NULL"
    ).fetchall()
    conn.close()
    return rows


def test_a_purged_row_is_still_eligible_for_deletion(root: Path) -> None:
    """Its note and thumbnail hold it, so Photos is not the last copy any more."""
    file_hash = catalogue_a_photos_row(root)
    mark_purged(root, file_hash)

    with_original, purged, refused = photos_command.sort_for_deletion(root, delete_rows(root))

    assert with_original == []
    assert [r["hash"] for r in purged] == [file_hash]
    assert refused == []


def test_a_row_with_neither_original_nor_thumbnail_is_refused(root: Path) -> None:
    file_hash = catalogue_a_photos_row(root)
    mark_purged(root, file_hash, keep_thumbnail=False)

    with_original, purged, refused = photos_command.sort_for_deletion(root, delete_rows(root))

    assert (with_original, purged) == ([], [])
    assert [r["hash"] for r in refused] == [file_hash]


def test_a_missing_original_that_was_never_purged_is_refused(root: Path) -> None:
    """The file went somewhere unplanned. That is not a reason to delete the source."""
    file_hash = catalogue_a_photos_row(root, on_disk=False)

    _, purged, refused = photos_command.sort_for_deletion(root, delete_rows(root))

    assert purged == []
    assert [r["hash"] for r in refused] == [file_hash]


def test_a_row_with_its_original_is_unaffected(root: Path) -> None:
    file_hash = catalogue_a_photos_row(root)
    with_original, purged, refused = photos_command.sort_for_deletion(root, delete_rows(root))
    assert [r["hash"] for r in with_original] == [file_hash]
    assert (purged, refused) == ([], [])


def test_the_dry_run_counts_purged_and_unpurged_separately(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    kept = catalogue_a_photos_row(root)

    # A second row from Photos, purged.
    conn = db.connect(root)
    db.insert_screenshot(
        conn,
        {
            "hash": "b" * 64,
            "path": "library/2026/07/gone.png",
            "original_name": "IMG_0099.PNG",
            "source": "iphone",
            "captured_at": "2026-07-20T14:03:50",
            "ingested_at": "2026-07-21T09:00:00",
            "ocr_text": "plenty of words",
            "photos_uuid": "UUID-PURGED",
            "purged_at": "2026-09-09T10:00:00",
        },
    )
    conn.close()
    make_image(root / "thumbs" / f"{'b' * 64}.jpg", "thumb")

    called: list[list[str]] = []
    monkeypatch.setattr(photos, "delete_assets", lambda uuids: called.append(uuids) or 0)

    result = runner.invoke(app, ["photos", "delete", "--root", str(root)])
    assert result.exit_code == 0, result.output

    flat = " ".join(result.output.split())
    # Rich elides the long labels, so check the listing rather than the table text.
    assert "UUID-PURGED" in flat  # the purged row is offered for deletion
    assert "Nothing was deleted" in flat
    assert called == []

    # And the split itself is what the table is counting.
    with_original, purged, _ = photos_command.sort_for_deletion(root, delete_rows(root))
    assert [r["hash"] for r in with_original] == [kept]
    assert [r["photos_uuid"] for r in purged] == ["UUID-PURGED"]


# ---------------------------------------------------------------------------
# Selecting on Photos' own judgement rather than osxphotos' narrower one
# ---------------------------------------------------------------------------


def build_photos_db(path: Path, rows: list[tuple[str, int, int, str]]) -> Path:
    """A stand-in Photos.sqlite with just the columns the selector reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE ZASSET (ZUUID TEXT, ZISDETECTEDSCREENSHOT INTEGER, "
        "ZTRASHEDSTATE INTEGER, ZUNIFORMTYPEIDENTIFIER TEXT)"
    )
    conn.executemany("INSERT INTO ZASSET VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def test_the_flag_is_read_straight_out_of_the_library(tmp_path: Path) -> None:
    library = tmp_path / "Photos Library.photoslibrary"
    build_photos_db(
        library / "database" / "Photos.sqlite",
        [
            ("KEEP-1", 1, 0, "public.png"),
            ("KEEP-2", 1, 0, "public.heic"),
            ("NOT-A-SHOT", 0, 0, "public.png"),
            ("BINNED", 1, 1, "public.png"),
            ("A-MOVIE", 1, 0, "com.apple.quicktime-movie"),
        ],
    )

    found = photos.detected_screenshot_uuids(library)

    # Unflagged, trashed and unreadable types are all out.
    assert found == {"KEEP-1", "KEEP-2"}


def test_a_library_that_cannot_be_read_returns_none(tmp_path: Path) -> None:
    """Every failure sends the caller back to osxphotos rather than stopping."""
    assert photos.detected_screenshot_uuids(tmp_path / "not there") is None

    library = tmp_path / "broken.photoslibrary"
    database = library / "database" / "Photos.sqlite"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"this is not a database")
    assert photos.detected_screenshot_uuids(library) is None


def test_a_library_without_the_column_returns_none(tmp_path: Path) -> None:
    """An older schema is a reason to fall back, not to crash."""
    library = tmp_path / "old.photoslibrary"
    path = library / "database" / "Photos.sqlite"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ZASSET (ZUUID TEXT)")
    conn.commit()
    conn.close()
    assert photos.detected_screenshot_uuids(library) is None


def test_the_flag_selector_finds_what_osxphotos_calls_an_ordinary_photo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point: 301 real screenshots osxphotos declines to call screenshots."""
    agreed = FakePhoto("AGREED")
    agreed.screenshot = True
    agreed.date = None
    disputed = FakePhoto("DISPUTED", "public.heic")
    disputed.screenshot = False  # osxphotos says no, Photos says yes
    disputed.date = None
    ordinary = FakePhoto("ORDINARY")
    ordinary.screenshot = False
    ordinary.date = None

    class FakeDB:
        def photos(self) -> list:
            return [agreed, disputed, ordinary]

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "detected_screenshot_uuids", lambda: {"AGREED", "DISPUTED"})
    monkeypatch.setitem(sys.modules, "osxphotos", type("M", (), {"PhotosDB": FakeDB})())

    found, selector = photos.load_screenshots()

    assert {p.uuid for p in found} == {"AGREED", "DISPUTED"}
    assert selector == photos.SELECTOR_DETECTED


def test_it_falls_back_to_osxphotos_when_the_flag_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agreed = FakePhoto("AGREED")
    agreed.screenshot = True
    agreed.date = None
    disputed = FakePhoto("DISPUTED")
    disputed.screenshot = False
    disputed.date = None

    class FakeDB:
        def photos(self) -> list:
            return [agreed, disputed]

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "detected_screenshot_uuids", lambda: None)
    monkeypatch.setitem(sys.modules, "osxphotos", type("M", (), {"PhotosDB": FakeDB})())

    found, selector = photos.load_screenshots()

    assert {p.uuid for p in found} == {"AGREED"}
    assert selector == photos.SELECTOR_OSXPHOTOS


def test_the_export_says_which_selector_ran(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: roomy())
    monkeypatch.setattr(photos, "load_screenshots", lambda _year: ([], photos.SELECTOR_DETECTED))

    result = runner.invoke(app, ["photos", "export", "--year", "2023", "--root", str(root)])

    assert photos.SELECTOR_DETECTED in " ".join(result.output.split())


class RawPhoto:
    """Photos records this asset as a JPEG. The original it hands over is a DNG."""

    uuid = "RAW-1"
    original_filename = "IMG_7376.DNG"
    uti = "public.jpeg"
    uti_original = "com.adobe.raw-image"
    ismovie = False
    screenshot = True
    date = None


def test_a_raw_original_behind_a_jpeg_record_is_set_aside() -> None:
    """The exact asset that went round twenty times: public.jpeg, DNG original."""
    keep, set_aside = photos.split_by_type([RawPhoto(), FakePhoto("FINE")])

    assert [p.uuid for p in keep] == ["FINE"]
    assert set_aside == {"public.jpeg": 1}


def test_the_original_filename_alone_is_enough_to_set_one_aside() -> None:
    """Even when both recorded types look fine, a .DNG original is not an image."""

    class Lying:
        uuid = "LIAR"
        original_filename = "IMG_9999.DNG"
        uti = "public.jpeg"
        uti_original = "public.jpeg"
        ismovie = False

    keep, set_aside = photos.split_by_type([Lying()])
    assert keep == []
    assert set_aside == {"public.jpeg": 1}


def test_an_asset_with_no_type_at_all_is_set_aside() -> None:
    class Blank:
        uuid = "BLANK"
        original_filename = "IMG_1.PNG"
        uti = None
        ismovie = False

    keep, _ = photos.split_by_type([Blank()])
    assert keep == []


# ---------------------------------------------------------------------------
# Never offering the same unusable asset twice
# ---------------------------------------------------------------------------


def test_ingest_remembers_the_uuid_it_set_aside(root: Path) -> None:
    """Moving the file frees the UUID, so the fact has to live in the catalogue."""
    iphone = root / "inbox" / "iphone"
    iphone.mkdir(parents=True, exist_ok=True)
    (iphone / "IMG_7376.DNG").write_bytes(b"raw bytes we cannot read")
    write_sidecar(iphone / "IMG_7376.DNG", photos_uuid="UUID-DNG")

    runner.invoke(app, ["ingest", "--root", str(root)])

    conn = db.connect(root)
    assert db.skipped_uuids(conn) == {"UUID-DNG"}
    row = conn.execute("SELECT * FROM skipped_uuids").fetchone()
    conn.close()
    assert row["original_name"] == "IMG_7376.DNG"
    assert "not an image" in row["reason"]
    assert (root / "skipped" / "IMG_7376.DNG").is_file()


def test_export_never_offers_an_asset_already_set_aside(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db.connect(root)
    db.record_skipped_uuid(conn, "SET-ASIDE", "IMG_7376.DNG", "not an image", "2026-09-09T15:00:00")
    conn.close()

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "build_check", lambda _root: roomy())
    monkeypatch.setattr(
        photos,
        "load_screenshots",
        lambda _year: ([FakePhoto("SET-ASIDE"), FakePhoto("FRESH")], "test selector"),
    )
    offered: list[str] = []
    monkeypatch.setattr(
        photos, "export_one", lambda photo, *_args: offered.append(photo.uuid) or None
    )

    result = runner.invoke(app, ["photos", "export", "--year", "2024", "--root", str(root)])
    assert result.exit_code == 0, result.output

    assert offered == ["FRESH"]
    reported = photos.parse_result(result.output)
    assert reported is not None
    assert reported.new == 1  # the set-aside one is not outstanding work


def test_setting_the_same_asset_aside_twice_does_not_break_ingest(root: Path) -> None:
    iphone = root / "inbox" / "iphone"
    iphone.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        (iphone / "IMG_7376.DNG").write_bytes(b"raw bytes")
        write_sidecar(iphone / "IMG_7376.DNG", photos_uuid="UUID-DNG")
        result = runner.invoke(app, ["ingest", "--root", str(root)])
        assert result.exit_code == 0, result.output

    conn = db.connect(root)
    assert db.skipped_uuids(conn) == {"UUID-DNG"}
    conn.close()
    assert len(list((root / "skipped").glob("*.DNG"))) == 2


# ---------------------------------------------------------------------------
# One dialogue, not seventy-four
# ---------------------------------------------------------------------------


def test_chunking_defaults_to_a_single_request() -> None:
    """Each request is one confirmation dialogue, so the default is one of both."""
    everything = [f"UUID-{i}" for i in range(14749)]
    assert photos.chunked(everything) == [everything]
    assert photos.chunked(everything, 0) == [everything]
    assert photos.chunked(everything, -1) == [everything]
    assert photos.chunked([]) == []


def test_an_explicit_chunk_size_still_splits() -> None:
    items = [f"UUID-{i}" for i in range(5)]
    assert photos.chunked(items, 2) == [items[:2], items[2:4], items[4:]]


def deletable_row(conn: object, uuid: str, file_hash: str, root: Path) -> None:
    db.insert_screenshot(
        conn,
        {
            "hash": file_hash,
            "path": f"library/2026/07/{file_hash}.png",
            "original_name": f"{uuid}.PNG",
            "source": "iphone",
            "captured_at": "2026-07-20T14:03:50",
            "ingested_at": "2026-07-21T09:00:00",
            "ocr_text": "some text",
            "photos_uuid": uuid,
            "purged_at": "2026-09-09T10:00:00",
        },
    )
    make_image(root / "thumbs" / f"{file_hash}.jpg", "thumb")


def test_a_whole_library_goes_in_one_request(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = db.connect(root)
    for i in range(300):
        deletable_row(conn, f"UUID-{i}", f"{i:064d}", root)
    conn.close()

    requests: list[list[str]] = []
    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "delete_assets", lambda uuids: requests.append(uuids) or len(uuids))

    result = runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])
    assert result.exit_code == 0, result.output

    assert len(requests) == 1  # one dialogue for the lot
    assert len(requests[0]) == 300
    assert "Moved 300 screenshots" in " ".join(result.output.split())


def test_chunking_reports_each_request(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    conn = db.connect(root)
    for i in range(5):
        deletable_row(conn, f"UUID-{i}", f"{i:064d}", root)
    conn.close()

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "delete_assets", len)

    result = runner.invoke(app, ["photos", "delete", "--yes", "--chunk", "2", "--root", str(root)])
    flat = " ".join(result.output.split())

    assert "3 requests" in flat
    assert "request 1: deleted 2 of 2" in flat
    assert "request 3: deleted 1 of 1" in flat


def test_deleted_rows_are_marked_and_not_offered_again(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db.connect(root)
    for i in range(3):
        deletable_row(conn, f"UUID-{i}", f"{i:064d}", root)
    conn.close()

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "delete_assets", len)
    runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])

    conn = db.connect(root)
    marked = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE photos_deleted_at IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert marked == 3

    # A rerun has nothing left to consider.
    second = runner.invoke(app, ["photos", "delete", "--root", str(root)])
    assert "Nothing to delete" in second.output


def test_declining_the_dialogue_is_not_an_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = db.connect(root)
    deletable_row(conn, "UUID-0", "0" * 64, root)
    conn.close()

    def declined(_uuids: list[str]) -> int:
        raise photos.DeletionDeclinedError("The confirmation dialogue was dismissed.")

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "delete_assets", declined)

    result = runner.invoke(app, ["photos", "delete", "--yes", "--root", str(root)])

    assert result.exit_code == 0  # a decision, not a fault
    assert "Declined, nothing deleted" in " ".join(result.output.split())

    conn = db.connect(root)
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM screenshots WHERE photos_deleted_at IS NOT NULL"
        ).fetchone()[0]
        == 0
    )
    conn.close()


def test_declining_partway_keeps_what_already_went(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted run must not lose track of the requests that succeeded."""
    conn = db.connect(root)
    for i in range(6):
        deletable_row(conn, f"UUID-{i}", f"{i:064d}", root)
    conn.close()

    seen: list[int] = []

    def declines_on_the_second(uuids: list[str]) -> int:
        seen.append(len(uuids))
        if len(seen) > 1:
            raise photos.DeletionDeclinedError("dismissed")
        return len(uuids)

    monkeypatch.setattr(photos, "require_photos", lambda: None)
    monkeypatch.setattr(photos, "delete_assets", declines_on_the_second)

    result = runner.invoke(app, ["photos", "delete", "--yes", "--chunk", "2", "--root", str(root)])
    assert result.exit_code == 0

    conn = db.connect(root)
    marked = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE photos_deleted_at IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    assert marked == 2  # the first request stands; the rest are still there to retry
