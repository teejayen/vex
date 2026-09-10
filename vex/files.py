"""File hashing, image properties, and iCloud dataless-file handling."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

import imagehash
from PIL import Image, UnidentifiedImageError

CHUNK_SIZE = 1024 * 1024
ICLOUD_TIMEOUT_SECONDS = 120.0
ICLOUD_POLL_SECONDS = 0.5
NEAR_DUPLICATE_DISTANCE = 6
PHASH_BITS = 64


class ImageProperties(NamedTuple):
    """Dimensions, format and perceptual hash of an image."""

    width: int | None
    height: int | None
    format: str | None
    phash: str | None


class DownloadResult(NamedTuple):
    """Outcome of making an iCloud file locally available."""

    needed_download: bool
    available: bool
    detail: str | None


def is_macos() -> bool:
    """True when running on macOS."""
    return sys.platform == "darwin"


def is_dataless(path: Path) -> bool:
    """True if the file is an iCloud placeholder: it has a size but no allocated blocks."""
    try:
        stat_result = path.stat()
    except OSError:
        return False
    blocks = getattr(stat_result, "st_blocks", None)
    if blocks is None:
        return False
    return stat_result.st_size > 0 and blocks == 0


def ensure_local(
    path: Path,
    timeout: float = ICLOUD_TIMEOUT_SECONDS,
    poll: float = ICLOUD_POLL_SECONDS,
) -> DownloadResult:
    """Materialise an evicted iCloud file with brctl, polling until it has local blocks.

    Reading a dataless file blocks rather than failing, so availability is judged from
    st_blocks and never from a speculative read.
    """
    if not path.exists():
        return DownloadResult(False, False, "file does not exist")
    if not is_dataless(path):
        return DownloadResult(False, True, None)
    if not is_macos():
        return DownloadResult(True, False, "dataless file on a non-macOS system")

    try:
        completed = subprocess.run(  # noqa: S603
            ["/usr/bin/brctl", "download", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DownloadResult(True, False, f"brctl download failed: {exc}")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not is_dataless(path):
            return DownloadResult(True, True, None)
        time.sleep(poll)

    detail = (completed.stderr or completed.stdout or "").strip()
    message = f"still evicted after {timeout:.0f} s"
    return DownloadResult(True, False, f"{message}: {detail}" if detail else message)


def sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_properties(path: Path) -> ImageProperties:
    """Dimensions, format and perceptual hash. Returns empty fields for unreadable images."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format
            phash = str(imagehash.phash(image.convert("RGB")))
    except (OSError, UnidentifiedImageError, ValueError):
        return ImageProperties(None, None, None, None)
    return ImageProperties(width, height, image_format, phash)


def hamming_distance(left: str, right: str) -> int:
    """Hamming distance between two hex perceptual hashes."""
    if not left or not right or len(left) != len(right):
        return PHASH_BITS
    return (int(left, 16) ^ int(right, 16)).bit_count()


def find_near_duplicate(
    phash: str | None,
    known: list[tuple[str, str]],
    threshold: int = NEAR_DUPLICATE_DISTANCE,
) -> str | None:
    """Hash of the closest catalogue entry within the threshold, if any."""
    if not phash:
        return None
    best_hash: str | None = None
    best_distance = threshold + 1
    for other_hash, other_phash in known:
        distance = hamming_distance(phash, other_phash)
        if distance <= threshold and distance < best_distance:
            best_hash, best_distance = other_hash, distance
    return best_hash


def unique_destination(path: Path) -> Path:
    """A destination that does not collide, appending -1, -2 and so on."""
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def place_file(source: Path, destination: Path, *, copy: bool) -> Path:
    """Move or copy a file to destination, creating parents and avoiding collisions."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(destination)
    if copy:
        shutil.copy2(source, destination)
    else:
        try:
            source.replace(destination)
        except OSError:
            # Crossing filesystems, which a rename cannot do.
            shutil.copy2(source, destination)
            source.unlink()
    return destination
