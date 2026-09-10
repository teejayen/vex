"""Face counting: the pass over thumbnails, and the hook on the ingest path."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tests.conftest import make_image
from vex import config, db, faces
from vex.cli import app
from vex.commands import faces as faces_command

runner = CliRunner()


def catalogue_row(root: Path, conn: object, file_hash: str, *, thumb: bool = True) -> None:
    """Insert one row, optionally with a thumbnail on disk."""
    relative = f"library/2026/07/{file_hash[:8]}.png"
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
            "bytes": 1000,
        },
    )


def test_detection_stores_counts_and_timestamps(root: Path) -> None:
    conn = db.connect(root)
    catalogue_row(root, conn, "a" * 64)
    catalogue_row(root, conn, "b" * 64)

    rows = faces_command.pending_rows(conn, force=False, limit=None)
    assert len(rows) == 2

    counts = {"a" * 64: 3, "b" * 64: 0}
    seen: list[Path] = []

    def fake_detect(path: Path) -> int:
        seen.append(path)
        return counts[path.stem]

    result = faces_command.detect_into_catalogue(conn, root, rows, fake_detect)
    assert result.checked == 2
    assert result.with_faces == 1

    # Detection reads the thumbnail, never the original.
    assert {p.parent.name for p in seen} == {"thumbs"}

    stored = db.get_screenshot(conn, "a" * 64)
    assert stored["faces"] == 3
    assert stored["faces_at"] is not None
    assert db.get_screenshot(conn, "b" * 64)["faces"] == 0

    # A second pass has nothing left to do, because null is what marks a row pending.
    assert faces_command.pending_rows(conn, force=False, limit=None) == []
    assert len(faces_command.pending_rows(conn, force=True, limit=None)) == 2
    conn.close()


def test_rows_without_a_thumbnail_stay_unchecked(root: Path) -> None:
    conn = db.connect(root)
    catalogue_row(root, conn, "c" * 64, thumb=False)

    rows = faces_command.pending_rows(conn, force=False, limit=None)
    result = faces_command.detect_into_catalogue(conn, root, rows, lambda _: 1)

    assert result.no_thumbnail == 1
    assert result.checked == 0
    assert db.get_screenshot(conn, "c" * 64)["faces"] is None
    conn.close()


def test_a_failing_detector_leaves_the_row_unchecked(root: Path) -> None:
    conn = db.connect(root)
    catalogue_row(root, conn, "d" * 64)

    def broken(_: Path) -> int:
        raise OSError("Vision request failed")

    rows = faces_command.pending_rows(conn, force=False, limit=None)
    result = faces_command.detect_into_catalogue(conn, root, rows, broken)

    assert result.failed == 1
    assert db.get_screenshot(conn, "d" * 64)["faces"] is None
    conn.close()


def test_best_effort_returns_none_without_vision(root: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(faces, "vision_available", lambda: False)
    assert faces.detect_best_effort(root, "e" * 64) is None


def test_best_effort_returns_none_when_detection_raises(root: Path, monkeypatch) -> None:  # noqa: ANN001
    make_image(config.thumb_path(root, "f" * 64), "thumb")
    monkeypatch.setattr(faces, "vision_available", lambda: True)

    def broken(_: Path) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(faces, "detect_faces", broken)
    assert faces.detect_best_effort(root, "f" * 64) is None


def test_ingest_records_a_face_count(root: Path, inbox: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(faces, "vision_available", lambda: True)
    monkeypatch.setattr(faces, "detect_faces", lambda _: 2)

    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png", "a face")
    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "Face-checked" in result.output

    conn = db.connect(root)
    row = conn.execute("SELECT faces, faces_at FROM screenshots").fetchone()
    assert row["faces"] == 2
    assert row["faces_at"] is not None
    conn.close()


def test_ingest_survives_a_broken_detector(root: Path, inbox: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(faces, "vision_available", lambda: True)

    def broken(_: Path) -> int:
        raise RuntimeError("Vision fell over")

    monkeypatch.setattr(faces, "detect_faces", broken)

    make_image(inbox / "Screenshot 2026-07-20 at 14.03.50.png", "hello")
    result = runner.invoke(app, ["ingest", "--root", str(root)])
    assert result.exit_code == 0, result.output

    conn = db.connect(root)
    row = conn.execute("SELECT faces FROM screenshots").fetchone()
    assert row["faces"] is None
    conn.close()


def test_search_filters_on_faces(root: Path) -> None:
    conn = db.connect(root)
    catalogue_row(root, conn, "1" * 64)
    catalogue_row(root, conn, "2" * 64)
    db.update_screenshot(conn, "1" * 64, {"ocr_text": "quarterly invoice", "faces": 1})
    db.update_screenshot(conn, "2" * 64, {"ocr_text": "quarterly invoice", "faces": 0})
    conn.close()

    import json  # noqa: PLC0415

    def search(*flags: str) -> list[dict]:
        result = runner.invoke(app, ["search", "invoice", "--json", *flags, "--root", str(root)])
        assert result.exit_code == 0, result.output
        return json.loads(result.output)

    assert [r["hash"] for r in search("--faces")] == ["1" * 64]
    assert [r["hash"] for r in search("--no-faces")] == ["2" * 64]
    assert len(search()) == 2
