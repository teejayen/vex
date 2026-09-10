"""Filename patterns, source detection and exFAT-safe naming."""

from __future__ import annotations

import re
from pathlib import Path

_STAMP_DIGITS = 14

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp", ".gif"})

# Characters exFAT and Windows will not accept in a filename.
ILLEGAL_CHARS = r'<>:"/\|?*'
_ILLEGAL_RE = re.compile(f"[{re.escape(ILLEGAL_CHARS)}]")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")

# Reserved device names on Windows, checked against the stem, case-insensitively.
WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# macOS: "Screenshot 2026-07-20 at 14.03.50.png", "Screen Shot 2020-01-02 at 3.04.05 pm.png",
# optionally with a " (2)" disambiguator.
MAC_PATTERN = re.compile(
    r"^Screen\s?[Ss]hot\s+"
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"\s+at\s+"
    r"(?P<hour>\d{1,2})\.(?P<minute>\d{2})\.(?P<second>\d{2})"
    r"(?:\s*(?P<ampm>[ap]\.?m\.?))?"
    r"(?:\s*\(\d+\))?$",
    re.IGNORECASE,
)

# Android / Windows style: "Screenshot_20-7-2026_140350_host.jpg", "Screenshot_20260720-140350".
GENERIC_UNDERSCORE_PATTERN = re.compile(
    r"^Screenshot[_-]"
    r"(?:"
    r"(?P<day>\d{1,2})-(?P<month>\d{1,2})-(?P<year>\d{4})_(?P<time>\d{6})"
    r"|"
    r"(?P<year2>\d{4})(?P<month2>\d{2})(?P<day2>\d{2})[-_](?P<time2>\d{6})"
    r")",
    re.IGNORECASE,
)

# Anything else beginning with "Screenshot" or "Screen Shot" counts as a screenshot by name.
SCREENSHOT_PREFIX = re.compile(r"^Screen\s?shot", re.IGNORECASE)

# iPhone camera-roll exports, including doubled extensions like IMG_8425.jpg.png
# and the " (2)" osxphotos appends when the original filename is already taken.
# iPhone numbers recycle, so that suffix is common rather than exotic.
IPHONE_PATTERN = re.compile(r"^IMG_\d{3,}(?:\s*\(\d+\))?(?:\.[A-Za-z0-9]+)*$", re.IGNORECASE)

# CleanShot X: "CleanShot 2026-07-20 at 14.03.50@2x.png".
CLEANSHOT_PATTERN = re.compile(
    r"^CleanShot\s+(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"\s+at\s+(?P<hour>\d{1,2})\.(?P<minute>\d{2})\.(?P<second>\d{2})",
    re.IGNORECASE,
)


def strip_extensions(name: str) -> str:
    """Drop every trailing image extension, so IMG_8425.jpg.png becomes IMG_8425."""
    stem = name
    while True:
        candidate = Path(stem)
        if candidate.suffix.lower() in IMAGE_EXTENSIONS:
            stem = candidate.stem
        else:
            return stem


def final_extension(name: str) -> str:
    """The real, last extension of a name, lowercased, including the dot."""
    return Path(name).suffix.lower()


def is_image(path: Path | str) -> bool:
    """True if the final extension is one we handle."""
    return final_extension(str(path)) in IMAGE_EXTENSIONS


def looks_like_screenshot(name: str) -> bool:
    """True if the filename matches any known screenshot naming convention."""
    stem = strip_extensions(name)
    return bool(
        MAC_PATTERN.match(stem)
        or GENERIC_UNDERSCORE_PATTERN.match(stem)
        or SCREENSHOT_PREFIX.match(stem)
        or IPHONE_PATTERN.match(stem)
        or CLEANSHOT_PATTERN.match(stem)
    )


def detect_source(name: str) -> str:
    """Guess the source from the filename: mac, iphone or other."""
    stem = strip_extensions(name)
    if MAC_PATTERN.match(stem) or CLEANSHOT_PATTERN.match(stem):
        return "mac"
    if IPHONE_PATTERN.match(stem):
        return "iphone"
    return "other"


def safe_filename(name: str, replacement: str = "_") -> str:
    """Make a filename safe for exFAT and Windows, and never empty."""
    cleaned = _CONTROL_RE.sub("", _ILLEGAL_RE.sub(replacement, name))
    cleaned = cleaned.strip()
    # No trailing dots or spaces, which Windows silently strips.
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        return "untitled"
    stem = Path(cleaned).stem
    if stem.upper() in WINDOWS_RESERVED:
        cleaned = f"{replacement}{cleaned}"
    return cleaned


def library_filename(captured_at: str, source: str, file_hash: str, extension: str) -> str:
    """Canonical library name: <YYYYMMDD-HHMMSS>_<source>_<hash8>.<ext>."""
    stamp = timestamp_slug(captured_at)
    extension = extension if extension.startswith(".") else f".{extension}"
    return safe_filename(f"{stamp}_{source}_{file_hash[:8]}{extension.lower()}")


def timestamp_slug(captured_at: str) -> str:
    """Turn an ISO 8601 timestamp into YYYYMMDD-HHMMSS."""
    digits = re.sub(r"\D", "", captured_at)[:_STAMP_DIGITS]
    if len(digits) < _STAMP_DIGITS:
        digits = digits.ljust(_STAMP_DIGITS, "0")
    return f"{digits[:8]}-{digits[8:14]}"


def library_relative_dir(captured_at: str) -> str:
    """The library/YYYY/MM directory for a capture timestamp."""
    slug = timestamp_slug(captured_at)
    return f"library/{slug[:4]}/{slug[4:6]}"
