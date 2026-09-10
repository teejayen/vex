"""Hashing, perceptual hashing and iCloud dataless detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from vex import files

if TYPE_CHECKING:
    import pytest

from tests.conftest import make_image


def test_sha256_is_stable_and_distinct(tmp_path: Path) -> None:
    a = make_image(tmp_path / "a.png", "alpha")
    b = make_image(tmp_path / "b.png", "alpha")
    c = make_image(tmp_path / "c.png", "beta", colour=(10, 20, 30))
    assert files.sha256(a) == files.sha256(b)
    assert files.sha256(a) != files.sha256(c)


def test_image_properties(tmp_path: Path) -> None:
    path = make_image(tmp_path / "a.png", "alpha", size=(320, 240))
    properties = files.image_properties(path)
    assert (properties.width, properties.height) == (320, 240)
    assert properties.format == "PNG"
    assert properties.phash


def test_image_properties_on_a_non_image(tmp_path: Path) -> None:
    path = tmp_path / "notes.png"
    path.write_text("not an image", encoding="utf-8")
    assert files.image_properties(path).width is None


def test_hamming_distance() -> None:
    assert files.hamming_distance("ff", "ff") == 0
    assert files.hamming_distance("f0", "f1") == 1
    assert files.hamming_distance("", "ff") == files.PHASH_BITS


def test_find_near_duplicate_within_threshold() -> None:
    known = [("older", "0000000000000000"), ("far", "ffffffffffffffff")]
    assert files.find_near_duplicate("0000000000000001", known) == "older"
    assert files.find_near_duplicate("0f0f0f0f0f0f0f0f", known) is None
    assert files.find_near_duplicate(None, known) is None


def test_place_file_moves_and_avoids_collisions(tmp_path: Path) -> None:
    source = make_image(tmp_path / "src" / "a.png")
    destination = tmp_path / "dest" / "a.png"

    first = files.place_file(source, destination, copy=False)
    assert first == destination
    assert not source.exists()

    other = make_image(tmp_path / "src" / "a.png", "different", colour=(1, 2, 3))
    second = files.place_file(other, destination, copy=True)
    assert second.name == "a-1.png"
    assert other.exists()


def test_dataless_detection_on_a_normal_file(tmp_path: Path) -> None:
    path = make_image(tmp_path / "a.png")
    assert not files.is_dataless(path)
    result = files.ensure_local(path)
    assert result.available
    assert not result.needed_download


def test_ensure_local_reports_a_missing_file(tmp_path: Path) -> None:
    result = files.ensure_local(tmp_path / "nope.png")
    assert not result.available
    assert result.detail == "file does not exist"


def test_ensure_local_times_out_on_a_stubbornly_evicted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_image(tmp_path / "a.png")
    monkeypatch.setattr(files, "is_dataless", lambda _: True)
    monkeypatch.setattr(files, "is_macos", lambda: True)

    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        calls.append(command)

        class Completed:
            stdout = ""
            stderr = "not signed in to iCloud"

        return Completed()

    monkeypatch.setattr(files.subprocess, "run", fake_run)
    result = files.ensure_local(path, timeout=0.05, poll=0.01)

    assert calls
    assert calls[0][:2] == ["/usr/bin/brctl", "download"]
    assert result.needed_download
    assert not result.available
    assert "still evicted" in (result.detail or "")


def test_ensure_local_succeeds_once_blocks_appear(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_image(tmp_path / "a.png")
    states = iter([True, True, False])
    monkeypatch.setattr(files, "is_dataless", lambda _: next(states, False))
    monkeypatch.setattr(files, "is_macos", lambda: True)
    monkeypatch.setattr(
        files.subprocess, "run", lambda *_, **__: type("C", (), {"stdout": "", "stderr": ""})()
    )

    result = files.ensure_local(path, timeout=5, poll=0.01)
    assert result.needed_download
    assert result.available


def test_is_dataless_uses_block_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_image(tmp_path / "a.png")
    real_stat = path.stat()

    class FakeStat:
        st_size = real_stat.st_size
        st_blocks = 0

    monkeypatch.setattr(Path, "stat", lambda _self, **_kw: FakeStat())
    assert files.is_dataless(path)
