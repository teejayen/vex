"""Capture-date extraction, in the precedence order the spec sets out."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from PIL import ExifTags, Image, UnidentifiedImageError

from vex.naming import (
    CLEANSHOT_PATTERN,
    GENERIC_UNDERSCORE_PATTERN,
    MAC_PATTERN,
    strip_extensions,
)

if TYPE_CHECKING:
    import os

EXIF_IFD_TAG = 0x8769
_TAG_BY_NAME = {name: tag for tag, name in ExifTags.TAGS.items()}
DATETIME_ORIGINAL = _TAG_BY_NAME["DateTimeOriginal"]
DATETIME_DIGITIZED = _TAG_BY_NAME["DateTimeDigitized"]
DATETIME = _TAG_BY_NAME["DateTime"]
MODEL = _TAG_BY_NAME["Model"]

# xmp:CreateDate, photoshop:DateCreated, and the PNG "date:create" / "Creation Time" chunks.
_XMP_DATE_RE = re.compile(
    r"<(?:xmp:CreateDate|photoshop:DateCreated|exif:DateTimeOriginal)>"
    r"\s*([0-9T:\-\.\+ ]+?)\s*</",
    re.IGNORECASE,
)

PNG_DATE_KEYS = ("date:create", "Creation Time", "creation_time", "date:modify")

NOON_HOUR = 12


class CaptureInfo(NamedTuple):
    """A capture timestamp plus where it came from, and the device if known."""

    captured_at: str | None
    source: str | None
    device: str | None


def to_iso(value: datetime) -> str:
    """Seconds-precision local ISO 8601, without a timezone offset."""
    return value.replace(microsecond=0).isoformat()


def parse_exif_datetime(value: str) -> datetime | None:
    """Parse the EXIF 'YYYY:MM:DD HH:MM:SS' form."""
    text = str(value).strip().strip("\x00")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)  # noqa: DTZ007
        except ValueError:
            continue
    return None


def parse_iso_datetime(value: str) -> datetime | None:
    """Parse an ISO 8601 string, tolerating a trailing Z and an offset."""
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return parse_exif_datetime(text)
    return parsed.replace(tzinfo=None)


def from_exif(image: Image.Image) -> tuple[datetime | None, str | None]:
    """Read DateTimeOriginal (then Digitized, then DateTime) and the camera model."""
    try:
        exif = image.getexif()
    except (OSError, ValueError):
        return None, None
    if not exif:
        return None, None

    model = exif.get(MODEL)
    model = str(model).strip().strip("\x00") if model else None

    try:
        ifd = exif.get_ifd(EXIF_IFD_TAG)
    except (OSError, ValueError, KeyError):
        ifd = {}
    candidates = [str(ifd[t]) for t in (DATETIME_ORIGINAL, DATETIME_DIGITIZED) if ifd.get(t)]
    if exif.get(DATETIME):
        candidates.append(str(exif[DATETIME]))

    for candidate in candidates:
        parsed = parse_exif_datetime(candidate)
        if parsed:
            return parsed, model
    return None, model


def from_xmp_or_png_text(image: Image.Image) -> datetime | None:
    """Read xmp:CreateDate from XMP, or a PNG text chunk carrying a date."""
    xmp = image.info.get("XML:com.adobe.xmp") or image.info.get("xmp")
    if isinstance(xmp, bytes):
        xmp = xmp.decode("utf-8", errors="ignore")
    if isinstance(xmp, str):
        match = _XMP_DATE_RE.search(xmp)
        if match:
            parsed = parse_iso_datetime(match.group(1))
            if parsed:
                return parsed

    for key in PNG_DATE_KEYS:
        value = image.info.get(key)
        if value:
            parsed = parse_iso_datetime(str(value))
            if parsed:
                return parsed
    return None


def from_filename(name: str) -> datetime | None:
    """Parse a capture time out of a known screenshot filename pattern."""
    stem = strip_extensions(name)

    match = MAC_PATTERN.match(stem) or CLEANSHOT_PATTERN.match(stem)
    if match:
        parts = match.groupdict()
        hour = int(parts["hour"])
        ampm = (parts.get("ampm") or "").replace(".", "").lower()
        if ampm == "pm" and hour < NOON_HOUR:
            hour += NOON_HOUR
        elif ampm == "am" and hour == NOON_HOUR:
            hour = 0
        try:
            return datetime(  # noqa: DTZ001
                int(parts["year"]),
                int(parts["month"]),
                int(parts["day"]),
                hour,
                int(parts["minute"]),
                int(parts["second"]),
            )
        except ValueError:
            return None

    match = GENERIC_UNDERSCORE_PATTERN.match(stem)
    if match:
        parts = match.groupdict()
        if parts.get("year"):
            year, month, day, time_text = (
                parts["year"],
                parts["month"],
                parts["day"],
                parts["time"],
            )
        else:
            year, month, day, time_text = (
                parts["year2"],
                parts["month2"],
                parts["day2"],
                parts["time2"],
            )
        try:
            return datetime(  # noqa: DTZ001
                int(year),
                int(month),
                int(day),
                int(time_text[0:2]),
                int(time_text[2:4]),
                int(time_text[4:6]),
            )
        except ValueError:
            return None
    return None


def sidecar_path(path: Path) -> Path | None:
    """The <name>.json sidecar sitting next to an image, if there is one."""
    for candidate in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if candidate.is_file():
            return candidate
    return None


def read_sidecar(path: Path) -> dict | None:
    """Parse the sidecar next to an image. Returns None when there is nothing usable."""
    candidate = sidecar_path(path)
    if candidate is None:
        return None
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def from_sidecar(path: Path) -> datetime | None:
    """Read a date out of an osxphotos-style <name>.json sidecar next to the file."""
    for candidate in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        for key in ("date", "created", "date_created", "photoTakenTime", "creationDate"):
            value = data.get(key)
            if isinstance(value, dict):
                value = value.get("timestamp") or value.get("formatted")
            if not value:
                continue
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(float(value))  # noqa: DTZ006
            parsed = parse_iso_datetime(str(value))
            if parsed:
                return parsed
    return None


def from_stat(stat_result: os.stat_result) -> datetime:
    """Fall back to min(birthtime, mtime) on macOS, mtime elsewhere."""
    mtime = stat_result.st_mtime
    birthtime = getattr(stat_result, "st_birthtime", None)
    if sys.platform == "darwin" and birthtime:
        mtime = min(mtime, birthtime)
    return datetime.fromtimestamp(mtime)  # noqa: DTZ006


def extract(path: Path, stat_result: os.stat_result | None = None) -> CaptureInfo:
    """Work out when a screenshot was captured, and note which source answered."""
    path = Path(path)
    device: str | None = None

    try:
        with Image.open(path) as image:
            exif_dt, device = from_exif(image)
            if exif_dt:
                return CaptureInfo(to_iso(exif_dt), "exif", device)
            xmp_dt = from_xmp_or_png_text(image)
            if xmp_dt:
                return CaptureInfo(to_iso(xmp_dt), "xmp", device)
    except (OSError, UnidentifiedImageError, ValueError):
        pass

    filename_dt = from_filename(path.name)
    if filename_dt:
        return CaptureInfo(to_iso(filename_dt), "filename", device)

    sidecar_dt = from_sidecar(path)
    if sidecar_dt:
        return CaptureInfo(to_iso(sidecar_dt), "sidecar", device)

    stat_result = stat_result if stat_result is not None else path.stat()
    stat_source = (
        "birthtime"
        if sys.platform == "darwin" and getattr(stat_result, "st_birthtime", None)
        else "mtime"
    )
    return CaptureInfo(to_iso(from_stat(stat_result)), stat_source, device)
