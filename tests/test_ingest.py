"""End-to-end ingest behaviour."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.conftest import make_heic, make_image
from vex import db, settings
from vex.cli import app
from vex.commands import ingest as ingest_command

runner = CliRunner()


def run(*args: str, root: Path) -> str:
    result = runner.invoke(app, [*args, "--root", str(root)])
    assert result.exit_code == 0, result.output
    return result.output


def test_ingest_moves_files_and_writes_rows(root: Path, inbox: Path) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png", "invoice total")
    run("ingest", root=root)

    conn = db.connect(root)
    rows = conn.execute("SELECT * FROM screenshots").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "mac"
    assert row["captured_at"] == "2026-07-20T14:03:50"
    assert row["captured_at_source"] == "filename"
    assert row["path"] == f"library/2026/07/20260720-140350_mac_{row['hash'][:8]}.png"
    assert row["original_name"] == "Screenshot 2026-07-20 at 14.03.50.png"
    assert (root / row["path"]).is_file()
    assert (root / "thumbs" / f"{row['hash']}.jpg").is_file()
    assert not (inbox / "Screenshot 2026-07-20 at 14.03.50.png").exists()
    conn.close()


def test_ingest_skips_files_that_are_not_screenshots(root: Path, tmp_path: Path) -> None:
    """The filename guess is for files out in the world, not our own inbox."""
    elsewhere = tmp_path / "pictures"
    make_image(elsewhere / "holiday.png")
    output = run("ingest", str(elsewhere), root=root)
    assert "Skipped" in output

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 0
    actions = [r[0] for r in conn.execute("SELECT action FROM ingest_log").fetchall()]
    assert actions == ["skipped", "left-behind"]
    conn.close()
    assert (elsewhere / "holiday.png").exists()


def test_anything_in_our_own_inbox_is_taken_whatever_it_is_called(root: Path, inbox: Path) -> None:
    """We put things in the inbox on purpose, so the name proves nothing."""
    make_image(inbox / "holiday.png")
    run("ingest", root=root)

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()
    assert not (inbox / "holiday.png").exists()


def test_explicit_source_overrides_the_pattern_check(root: Path, inbox: Path) -> None:
    make_image(inbox / "holiday.png")
    run("ingest", "--source", "other", root=root)

    conn = db.connect(root)
    row = conn.execute("SELECT source FROM screenshots").fetchone()
    assert row["source"] == "other"
    conn.close()


def test_exact_duplicates_move_to_the_duplicates_folder(root: Path, inbox: Path) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png", "invoice")
    run("ingest", root=root)
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png", "invoice")
    output = run("ingest", root=root)

    assert "Duplicates" in output
    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()
    duplicates = list((root / "duplicates").glob("*.png"))
    assert len(duplicates) == 1


def test_near_duplicates_are_linked(root: Path, inbox: Path) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png", "one")
    run("ingest", root=root)
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.51.png", "one", box=(50, 60, 300, 201))
    run("ingest", root=root)

    conn = db.connect(root)
    linked = conn.execute(
        "SELECT COUNT(*) FROM screenshots WHERE near_duplicate_of IS NOT NULL"
    ).fetchone()[0]
    assert linked == 1
    conn.close()


def test_iphone_names_are_detected_even_inside_the_mac_inbox(root: Path, inbox: Path) -> None:
    make_image(inbox / "IMG_8425.jpg.png")
    run("ingest", root=root)

    conn = db.connect(root)
    row = conn.execute("SELECT source, device, path FROM screenshots").fetchone()
    assert row["source"] == "iphone"
    assert row["device"] == "iphone"
    assert row["path"].endswith(".png")
    conn.close()


def test_copy_leaves_the_original_in_place(root: Path, inbox: Path) -> None:
    source = make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    run("ingest", "--copy", root=root)
    assert source.exists()

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()


def test_dry_run_changes_nothing(root: Path, inbox: Path) -> None:
    source = make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    output = run("ingest", "--dry-run", root=root)

    assert "dry run" in output.lower()
    assert source.exists()
    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 0
    conn.close()


def test_ingest_from_an_explicit_directory(root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "desktop"
    make_image(elsewhere / "Screenshot 2026-07-20 at 14.03.50.png")
    run("ingest", str(elsewhere), root=root)

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()


def test_directories_are_not_recursed_without_the_flag(root: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "desktop"
    make_image(elsewhere / "nested" / "Screenshot 2026-07-20 at 14.03.50.png")
    run("ingest", str(elsewhere), root=root)

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 0
    conn.close()

    run("ingest", str(elsewhere), "--recursive", root=root)
    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()


def test_ingest_is_idempotent(root: Path, inbox: Path) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    run("ingest", root=root)
    run("ingest", root=root)

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()


def test_unknown_source_is_rejected(root: Path) -> None:
    result = runner.invoke(app, ["ingest", "--source", "nope", "--root", str(root)])
    assert result.exit_code == 2


# ---------------------------------------------------------------------------
# Photos re-exports of rows already held
# ---------------------------------------------------------------------------


def write_sidecar(image: Path, uuid: str) -> Path:
    """An osxphotos-style sidecar of the shape the export command writes."""
    sidecar = image.with_suffix(image.suffix + ".json")
    sidecar.write_text(
        json.dumps({"photos_uuid": uuid, "source": "photos", "date": "2026-07-20T14:03:50"}),
        encoding="utf-8",
    )
    return sidecar


def test_a_reexported_duplicate_is_discarded_and_gives_up_its_uuid(root: Path) -> None:
    iphone = root / "inbox" / "iphone"
    name = "IMG_4242.png"
    make_image(iphone / name, "a screenshot")
    first = write_sidecar(iphone / name, "UUID-1")

    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert not first.exists()

    conn = db.connect(root)
    row = conn.execute("SELECT hash, path, photos_uuid FROM screenshots").fetchone()
    file_hash, relative = row["hash"], row["path"]
    # The first pass took the UUID off the sidecar as usual.
    assert row["photos_uuid"] == "UUID-1"
    # Clear it, so the second pass has something to record.
    db.update_screenshot(conn, file_hash, {"photos_uuid": None})
    conn.close()

    # Photos hands the very same bytes back on a later export.
    shutil.copy2(root / relative, iphone / name)
    again = write_sidecar(iphone / name, "UUID-1")

    output = runner.invoke(app, ["ingest", "--root", str(root)]).output
    assert "Re-exports discarded" in output

    # The copy and its sidecar are gone, not parked.
    assert not (iphone / name).exists()
    assert not again.exists()
    assert list((root / "duplicates").glob("*")) == []

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    # The UUID was recorded, so the next export skips the asset entirely.
    assert db.get_screenshot(conn, file_hash)["photos_uuid"] == "UUID-1"
    logged = conn.execute(
        "SELECT action, detail FROM ingest_log WHERE hash = ? ORDER BY id", (file_hash,)
    ).fetchall()
    assert [entry["action"] for entry in logged] == ["added", "duplicate"]
    assert ingest_command.DISCARDED_DETAIL in logged[-1]["detail"]
    assert "photos_uuid recorded" in logged[-1]["detail"]
    conn.close()


def test_a_reexport_of_a_purged_row_is_discarded_without_resurrecting_it(root: Path) -> None:
    iphone = root / "inbox" / "iphone"
    name = "IMG_4243.png"
    make_image(iphone / name, "a long social post worth keeping")
    write_sidecar(iphone / name, "UUID-2")
    runner.invoke(app, ["ingest", "--root", str(root)])

    conn = db.connect(root)
    row = conn.execute("SELECT hash, path FROM screenshots").fetchone()
    file_hash, relative = row["hash"], row["path"]
    kept = root / "duplicates" / "kept-bytes.png"
    kept.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / relative, kept)
    db.update_screenshot(
        conn,
        file_hash,
        {"category": "social-post", "ocr_text": "x" * 200, "faces": 0, "photos_uuid": None},
    )
    conn.close()

    runner.invoke(app, ["purge", "--yes", "--root", str(root)])
    assert not (root / relative).exists()

    shutil.copy2(kept, iphone / name)
    write_sidecar(iphone / name, "UUID-2")
    kept.unlink()
    runner.invoke(app, ["ingest", "--root", str(root)])

    assert not (iphone / name).exists()
    conn = db.connect(root)
    after = db.get_screenshot(conn, file_hash)
    # Still purged, still one row, and now carrying its UUID.
    assert after["purged_at"] is not None
    assert after["photos_uuid"] == "UUID-2"
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    assert not (root / relative).exists()
    detail = conn.execute("SELECT detail FROM ingest_log WHERE action = 'duplicate'").fetchone()[
        "detail"
    ]
    assert "original purged" in detail
    assert ingest_command.DISCARDED_DETAIL in detail
    conn.close()


def test_a_duplicate_without_a_photos_sidecar_is_still_parked(root: Path, inbox: Path) -> None:
    name = "Screenshot 2026-07-20 at 14.03.50.png"
    make_image(inbox / name, "no sidecar here")
    runner.invoke(app, ["ingest", "--root", str(root)])

    conn = db.connect(root)
    relative = conn.execute("SELECT path FROM screenshots").fetchone()[0]
    conn.close()

    shutil.copy2(root / relative, inbox / name)
    output = runner.invoke(app, ["ingest", "--root", str(root)]).output

    # Nothing said where this came from, so it is kept rather than deleted.
    assert "Duplicates" in output
    assert not (inbox / name).exists()
    assert len(list((root / "duplicates").glob("*.png"))) == 1


def test_a_duplicate_outside_the_inbox_is_never_deleted(root: Path, tmp_path: Path) -> None:
    iphone = root / "inbox" / "iphone"
    name = "IMG_4244.png"
    make_image(iphone / name, "first time through")
    write_sidecar(iphone / name, "UUID-3")
    runner.invoke(app, ["ingest", "--root", str(root)])

    # The same bytes and sidecar, but sitting somewhere outside the library.
    elsewhere = tmp_path / "desktop"
    elsewhere.mkdir()
    conn = db.connect(root)
    relative = conn.execute("SELECT path FROM screenshots").fetchone()[0]
    conn.close()
    shutil.copy2(root / relative, elsewhere / name)
    sidecar = write_sidecar(elsewhere / name, "UUID-3")

    runner.invoke(app, ["ingest", str(elsewhere), "--root", str(root)])

    # Moved into duplicates/, as ever, and the sidecar left alone.
    assert not (elsewhere / name).exists()
    assert sidecar.is_file()
    assert len(list((root / "duplicates").glob("*.png"))) == 1


# ---------------------------------------------------------------------------
# The macOS drop zone outside the root
# ---------------------------------------------------------------------------


def test_ingest_sweeps_the_mac_drop_zone(root: Path, isolated_mac_inbox: Path) -> None:
    """Screenshots land in the home directory while the drive is unplugged."""
    make_image(
        isolated_mac_inbox / "Screenshot 2026-07-20 at 14.03.50.png", "from the internal disk"
    )
    make_image(root / "inbox" / "mac" / "Screenshot 2026-07-21 at 09.15.00.png", "from the drive")

    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    rows = conn.execute(
        "SELECT source, original_name FROM screenshots ORDER BY captured_at"
    ).fetchall()
    conn.close()

    assert len(rows) == 2
    assert {r["source"] for r in rows} == {"mac"}
    # The file was moved out of the drop zone into the library, wherever the root is.
    assert not (isolated_mac_inbox / "Screenshot 2026-07-20 at 14.03.50.png").exists()
    assert len(list((root / "library").rglob("*.png"))) == 2


def test_the_drop_zone_is_created_so_macos_has_somewhere_to_save(
    root: Path, isolated_mac_inbox: Path
) -> None:
    assert not isolated_mac_inbox.exists()
    runner.invoke(app, ["ingest", "--root", str(root)])
    assert isolated_mac_inbox.is_dir()


def test_a_drop_zone_inside_the_root_is_not_swept_twice(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointed at the root's own inbox, it must not be counted a second time."""
    monkeypatch.setenv(settings.ENV_MAC_INBOX, str(root / "inbox" / "mac"))
    make_image(root / "inbox" / "mac" / "Screenshot 2026-07-20 at 14.03.50.png", "once only")

    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()


def test_named_paths_do_not_pull_in_the_drop_zone(root: Path, isolated_mac_inbox: Path) -> None:
    """Asking for one directory means that directory, not the drop zone as well."""
    isolated_mac_inbox.mkdir(parents=True, exist_ok=True)
    make_image(isolated_mac_inbox / "Screenshot 2026-07-20 at 14.03.50.png", "not asked for")
    elsewhere = root.parent / "somewhere"
    make_image(elsewhere / "Screenshot 2026-07-21 at 09.15.00.png", "asked for")

    runner.invoke(app, ["ingest", str(elsewhere), "--root", str(root)])

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()
    assert (isolated_mac_inbox / "Screenshot 2026-07-20 at 14.03.50.png").is_file()


# ---------------------------------------------------------------------------
# What Photos actually hands over
# ---------------------------------------------------------------------------


def test_a_heic_screenshot_is_ingested(root: Path) -> None:
    """Photos flags edited and shared screenshots as HEIC, and Pillow needs help."""
    iphone = root / "inbox" / "iphone"
    make_heic(iphone / "IMG_6797.heic", "an edited screenshot")
    write_sidecar(iphone / "IMG_6797.heic", "UUID-HEIC")

    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    row = conn.execute("SELECT * FROM screenshots").fetchone()
    conn.close()

    assert row is not None
    assert row["format"] == "HEIF"
    assert row["width"] == 400
    assert row["photos_uuid"] == "UUID-HEIC"
    # The library keeps the original extension and bytes; the thumbnail is JPEG.
    assert row["path"].endswith(".heic")
    assert (root / row["path"]).is_file()
    assert (root / "thumbs" / f"{row['hash']}.jpg").is_file()
    # Nothing is left in the inbox to be missed next time.
    assert list(iphone.iterdir()) == []


def test_a_heic_named_png_is_still_read(root: Path) -> None:
    """Photos names an export from the asset, which need not match its contents."""
    iphone = root / "inbox" / "iphone"
    make_heic(iphone / "IMG_8880.PNG", "heic wearing a png name")
    write_sidecar(iphone / "IMG_8880.PNG", "UUID-MISNAMED")

    runner.invoke(app, ["ingest", "--root", str(root)])

    conn = db.connect(root)
    row = conn.execute("SELECT format, photos_uuid FROM screenshots").fetchone()
    conn.close()
    assert row is not None
    assert row["format"] == "HEIF"


def test_an_incremented_photos_export_is_not_rejected_on_its_name(root: Path) -> None:
    """osxphotos writes IMG_7390 (1).PNG on a name clash, which the pattern rejects."""
    iphone = root / "inbox" / "iphone"
    make_image(iphone / "IMG_7390 (1).PNG", "a real screenshot")
    write_sidecar(iphone / "IMG_7390 (1).PNG", "UUID-INCREMENTED")

    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    row = conn.execute("SELECT source, photos_uuid FROM screenshots").fetchone()
    actions = [r[0] for r in conn.execute("SELECT action FROM ingest_log").fetchall()]
    conn.close()

    assert row is not None
    # Photos said it was a screenshot from the phone, so the filename does not vote.
    assert row["source"] == "iphone"
    assert row["photos_uuid"] == "UUID-INCREMENTED"
    assert actions == ["added"]
    assert list(iphone.iterdir()) == []


def test_a_photos_sidecar_beats_the_pattern_even_outside_the_inbox(
    root: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "somewhere"
    make_image(elsewhere / "IMG_7390 (2).PNG", "still a screenshot")
    write_sidecar(elsewhere / "IMG_7390 (2).PNG", "UUID-OUTSIDE")

    runner.invoke(app, ["ingest", str(elsewhere), "--root", str(root)])

    conn = db.connect(root)
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 1
    conn.close()


def test_what_is_left_behind_is_named_in_the_log(root: Path, tmp_path: Path) -> None:
    """A file that stays put must say so, or it is invisible twice over."""
    elsewhere = tmp_path / "pictures"
    make_image(elsewhere / "holiday.png")
    make_image(elsewhere / "sunset.png")

    output = run("ingest", str(elsewhere), root=root)
    assert "2 left in place" in output
    assert "holiday.png (filename does not match a screenshot pattern)" in output.replace("\n", "")

    conn = db.connect(root)
    detail = conn.execute("SELECT detail FROM ingest_log WHERE action = 'left-behind'").fetchone()[
        "detail"
    ]
    conn.close()
    assert detail.startswith("2 left in place")
    assert "holiday.png" in detail
    assert "sunset.png" in detail


def test_nothing_left_behind_means_no_summary(root: Path, inbox: Path) -> None:
    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png")
    output = run("ingest", root=root)
    assert "left in place" not in output

    conn = db.connect(root)
    assert (
        conn.execute("SELECT COUNT(*) FROM ingest_log WHERE action = 'left-behind'").fetchone()[0]
        == 0
    )
    conn.close()


def test_the_summary_stops_naming_files_after_ten() -> None:
    left = [(f"file{i}.png", "could not read the image") for i in range(14)]
    line = ingest_command.left_behind_summary(left)
    assert line.startswith("14 left in place")
    assert "file9.png" in line
    assert "file10.png" not in line
    assert "and 4 more" in line


# ---------------------------------------------------------------------------
# Types that are flagged as screenshots but are not ones
# ---------------------------------------------------------------------------


def test_a_movie_in_the_inbox_is_moved_out_of_the_way(root: Path) -> None:
    """Left in the inbox it blocks its own UUID for good."""
    iphone = root / "inbox" / "iphone"
    iphone.mkdir(parents=True, exist_ok=True)
    (iphone / "IMG_9001.MOV").write_bytes(b"not really a movie, but not an image either")
    sidecar = write_sidecar(iphone / "IMG_9001.MOV", "UUID-MOVIE")

    output = run("ingest", root=root)
    assert "Moved to skipped/" in output

    # Gone from the inbox, sidecar with it, so the asset is free to come again.
    assert not (iphone / "IMG_9001.MOV").exists()
    assert not sidecar.exists()
    assert (root / "skipped" / "IMG_9001.MOV").is_file()
    assert (root / "skipped" / "IMG_9001.MOV.json").is_file()

    conn = db.connect(root)
    detail = conn.execute("SELECT detail FROM ingest_log WHERE action = 'skipped'").fetchone()[0]
    conn.close()
    assert "not an image" in detail
    assert "moved to skipped/" in detail


def test_an_unreadable_file_is_moved_out_of_the_way_too(root: Path) -> None:
    """The corrupt export on the drive had exactly this problem."""
    iphone = root / "inbox" / "iphone"
    iphone.mkdir(parents=True, exist_ok=True)
    (iphone / "IMG_8880.PNG").write_bytes(b"\xff" * 512)
    write_sidecar(iphone / "IMG_8880.PNG", "UUID-CORRUPT")

    run("ingest", root=root)

    assert not (iphone / "IMG_8880.PNG").exists()
    assert (root / "skipped" / "IMG_8880.PNG").is_file()
    assert (root / "skipped" / "IMG_8880.PNG.json").is_file()

    conn = db.connect(root)
    detail = conn.execute("SELECT detail FROM ingest_log WHERE action = 'error'").fetchone()[0]
    assert conn.execute("SELECT COUNT(*) FROM screenshots").fetchone()[0] == 0
    conn.close()
    assert "could not read the image" in detail
    assert "moved to skipped/" in detail


def test_a_file_outside_the_inbox_is_never_moved(root: Path, tmp_path: Path) -> None:
    """Somewhere outside the library is not ours to tidy."""
    elsewhere = tmp_path / "desktop"
    elsewhere.mkdir()
    (elsewhere / "holiday.mov").write_bytes(b"a real movie of theirs")

    output = run("ingest", str(elsewhere), root=root)

    assert (elsewhere / "holiday.mov").is_file()
    assert not (root / "skipped" / "holiday.mov").exists()
    assert "1 left in place" in output


def test_parking_does_not_collide_on_a_repeated_name(root: Path) -> None:
    iphone = root / "inbox" / "iphone"
    iphone.mkdir(parents=True, exist_ok=True)
    (root / "skipped").mkdir(parents=True, exist_ok=True)
    (root / "skipped" / "IMG_9001.MOV").write_bytes(b"an earlier one")
    (iphone / "IMG_9001.MOV").write_bytes(b"a later one")

    run("ingest", root=root)

    parked = sorted(p.name for p in (root / "skipped").iterdir())
    assert len(parked) == 2  # the earlier one is not overwritten
