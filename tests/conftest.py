"""Shared fixtures: synthetic screenshots and a temporary library root."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from vex import config, settings, taxonomy


def make_image(
    path: Path,
    text: str = "hello",
    size: tuple[int, int] = (400, 300),
    colour: tuple[int, int, int] = (250, 250, 250),
    box: tuple[int, int, int, int] | None = (50, 60, 300, 200),
) -> Path:
    """Write a small synthetic PNG with some drawn content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill=(0, 0, 0))
    if box:
        draw.rectangle(box, outline=(0, 0, 180), width=3)
    image.save(path)
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A library root with the standard layout created."""
    return config.ensure_layout(tmp_path / "library-root")


@pytest.fixture
def inbox(root: Path) -> Path:
    """The macOS inbox drop zone."""
    return root / "inbox" / "mac"


@pytest.fixture(autouse=True)
def isolated_mac_inbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the macOS drop zone somewhere safe for every test.

    Its default is a real path in the home directory, and ingest with no arguments
    sweeps it. Without this, a test run would hoover up the real screenshots.
    """
    zone = tmp_path / "mac-drop-zone"
    monkeypatch.delenv(settings.LEGACY_ENV_MAC_INBOX, raising=False)
    monkeypatch.setenv(settings.ENV_MAC_INBOX, str(zone))
    return zone


@pytest.fixture(autouse=True)
def _built_in_taxonomy() -> Iterator[None]:
    """Resolving a root loads that root's taxonomy, so put it back afterwards."""
    taxonomy.reset()
    yield
    taxonomy.reset()


def make_heic(path: Path, text: str = "heic", size: tuple[int, int] = (400, 300)) -> Path:
    """A real HEIC file, which Pillow can only open once pillow-heif is registered."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (240, 240, 250))
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, fill=(0, 0, 0))
    draw.rectangle((40, 50, 260, 180), outline=(180, 0, 0), width=3)
    image.save(path, format="HEIF")
    return path
