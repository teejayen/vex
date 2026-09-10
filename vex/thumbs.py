"""Thumbnail generation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_SIDE = 512
QUALITY = 80


def thumb_path(root: Path, file_hash: str) -> Path:
    """Where the thumbnail for a hash lives."""
    return Path(root) / "thumbs" / f"{file_hash}.jpg"


def generate(source: Path, destination: Path, max_side: int = MAX_SIDE) -> bool:
    """Write an RGB JPEG thumbnail. Returns False if the source could not be read."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened) or opened
            thumb = image.convert("RGB")
            thumb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            thumb.save(destination, "JPEG", quality=QUALITY, optimize=True)
    except (OSError, UnidentifiedImageError, ValueError):
        return False
    return True


def ensure(root: Path, file_hash: str, image_path: Path, *, force: bool = False) -> bool:
    """Create the thumbnail for a catalogue row if it is missing, or when forced."""
    destination = thumb_path(root, file_hash)
    if destination.exists() and not force:
        return True
    return generate(image_path, destination)
