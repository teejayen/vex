"""Filename patterns, source detection and exFAT-safe naming."""

from __future__ import annotations

import pytest

from vex import naming


@pytest.mark.parametrize(
    "name",
    [
        "Screenshot 2026-07-20 at 14.03.50.png",
        "Screenshot 2026-07-20 at 14.03.50 (2).png",
        "Screen Shot 2020-01-02 at 3.04.05 pm.png",
        "IMG_8425.jpg.png",
        "IMG_8426.PNG.png",
        "IMG_2968.png",
        "Screenshot_20-7-2026_140350_host.jpg",
        "CleanShot 2026-07-20 at 14.03.50@2x.png",
    ],
)
def test_recognises_screenshot_names(name: str) -> None:
    assert naming.looks_like_screenshot(name)


@pytest.mark.parametrize("name", ["holiday.jpg", "diagram.png", "IMG.png", "notes.txt"])
def test_rejects_other_names(name: str) -> None:
    assert not naming.looks_like_screenshot(name)


def test_detect_source() -> None:
    assert naming.detect_source("Screenshot 2026-07-20 at 14.03.50.png") == "mac"
    assert naming.detect_source("IMG_8425.jpg.png") == "iphone"
    assert naming.detect_source("Screenshot_20-7-2026_140350.jpg") == "other"


def test_strip_doubled_extensions() -> None:
    assert naming.strip_extensions("IMG_8425.jpg.png") == "IMG_8425"
    assert naming.final_extension("IMG_8425.jpg.png") == ".png"


@pytest.mark.parametrize(
    ("raw", "expected_absent"),
    [
        ('a:b*c?d"e<f>g|h.png', ':*?"<>|'),
        ("trailing dots...", "."),
    ],
)
def test_safe_filename_strips_illegal_characters(raw: str, expected_absent: str) -> None:
    safe = naming.safe_filename(raw)
    for char in expected_absent:
        if char == ".":
            assert not safe.endswith(".")
        else:
            assert char not in safe


def test_safe_filename_avoids_reserved_windows_names() -> None:
    assert naming.safe_filename("CON.png") != "CON.png"
    assert naming.safe_filename("") == "untitled"


def test_library_filename_and_dir() -> None:
    name = naming.library_filename("2026-07-20T14:03:50", "mac", "abcdef1234567890", ".PNG")
    assert name == "20260720-140350_mac_abcdef12.png"
    assert naming.library_relative_dir("2026-07-20T14:03:50") == "library/2026/07"
