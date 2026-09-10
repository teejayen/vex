"""Capture-date extraction and its precedence order."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from vex import capture

XMP_TEMPLATE = (
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="rdf">'
    "<rdf:Description><xmp:CreateDate>{value}</xmp:CreateDate>"
    "</rdf:Description></rdf:RDF></x:xmpmeta>"
)


def small_image() -> Image.Image:
    return Image.new("RGB", (60, 40), (200, 200, 200))


def test_filename_parsing_macos() -> None:
    parsed = capture.from_filename("Screenshot 2026-07-20 at 14.03.50.png")
    assert parsed is not None
    assert parsed.isoformat() == "2026-07-20T14:03:50"


def test_filename_parsing_handles_duplicate_marker() -> None:
    parsed = capture.from_filename("Screenshot 2026-07-20 at 14.03.50 (2).png")
    assert parsed is not None
    assert parsed.isoformat() == "2026-07-20T14:03:50"


def test_filename_parsing_twelve_hour_pm() -> None:
    parsed = capture.from_filename("Screen Shot 2020-01-02 at 3.04.05 pm.png")
    assert parsed is not None
    assert parsed.isoformat() == "2020-01-02T15:04:05"


def test_filename_parsing_underscore_form() -> None:
    parsed = capture.from_filename("Screenshot_20-7-2026_140350_host.jpg")
    assert parsed is not None
    assert parsed.isoformat() == "2026-07-20T14:03:50"


def test_filename_parsing_rejects_unknown() -> None:
    assert capture.from_filename("holiday.png") is None


def test_exif_wins_over_filename(tmp_path: Path) -> None:
    path = tmp_path / "Screenshot 2026-07-20 at 14.03.50.jpg"
    exif = Image.Exif()
    exif[0x0110] = "iPhone 15 Pro"
    exif[0x8769] = {0x9003: "2026:03:04 05:06:07"}
    small_image().save(path, exif=exif)

    info = capture.extract(path)
    assert info.captured_at == "2026-03-04T05:06:07"
    assert info.source == "exif"
    assert info.device == "iPhone 15 Pro"


def test_xmp_used_when_no_exif_date(tmp_path: Path) -> None:
    path = tmp_path / "unnamed.png"
    png_info = PngInfo()
    png_info.add_text("XML:com.adobe.xmp", XMP_TEMPLATE.format(value="2025-11-12T13:14:15"))
    small_image().save(path, pnginfo=png_info)

    info = capture.extract(path)
    assert info.captured_at == "2025-11-12T13:14:15"
    assert info.source == "xmp"


def test_png_text_chunk_date(tmp_path: Path) -> None:
    path = tmp_path / "unnamed.png"
    png_info = PngInfo()
    png_info.add_text("date:create", "2024-01-02T03:04:05+10:00")
    small_image().save(path, pnginfo=png_info)

    info = capture.extract(path)
    assert info.captured_at == "2024-01-02T03:04:05"


def test_filename_used_when_no_metadata(tmp_path: Path) -> None:
    path = tmp_path / "Screenshot 2026-07-20 at 14.03.50.png"
    small_image().save(path)

    info = capture.extract(path)
    assert info.captured_at == "2026-07-20T14:03:50"
    assert info.source == "filename"


def test_sidecar_used_when_filename_has_no_date(tmp_path: Path) -> None:
    path = tmp_path / "IMG_8425.png"
    small_image().save(path)
    sidecar = tmp_path / "IMG_8425.png.json"
    sidecar.write_text(json.dumps({"date": "2023-05-06T07:08:09"}), encoding="utf-8")

    info = capture.extract(path)
    assert info.captured_at == "2023-05-06T07:08:09"
    assert info.source == "sidecar"


def test_falls_back_to_file_times(tmp_path: Path) -> None:
    path = tmp_path / "IMG_8425.png"
    small_image().save(path)
    when = time.mktime((2022, 4, 5, 6, 7, 8, 0, 0, -1))
    os.utime(path, (when, when))

    info = capture.extract(path)
    assert info.source in {"birthtime", "mtime"}
    assert info.captured_at is not None
    assert info.captured_at.startswith("20")
