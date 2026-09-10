"""Which library originals can go, once the note and the thumbnail hold their value.

Purging deletes only the original file. The thumbnail, the note and the catalogue
row always stay, so a purged screenshot is still searchable, still renders in its
note, and still blocks a re-ingest of the same bytes.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from vex import config, taxonomy

#: Junk is purgeable whatever its category says, so it is selectable by this name.
JUNK_SELECTOR = "junk"

#: Below this many characters of recognised text, the note is not worth much, so
#: the original stays.
DEFAULT_MIN_TEXT = 80

REASON_ALREADY_PURGED = "already purged"
REASON_MISSING = "library file already gone"
REASON_KEPT = "pinned with keep"
REASON_NO_THUMBNAIL = "no thumbnail"
REASON_UNCHECKED = "faces not checked"
REASON_FACES = "faces detected"
REASON_CATEGORY = "category not purgeable"
REASON_THIN_TEXT = "text too thin"

#: Reporting order, most interesting first.
REASON_ORDER: tuple[str, ...] = (
    REASON_FACES,
    REASON_THIN_TEXT,
    REASON_CATEGORY,
    REASON_UNCHECKED,
    REASON_KEPT,
    REASON_NO_THUMBNAIL,
    REASON_ALREADY_PURGED,
    REASON_MISSING,
)


class CategorySelectionError(ValueError):
    """Raised when --category names something that is never purgeable."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """Whether one row can be purged, and why not when it cannot."""

    eligible: bool
    reason: str | None = None


ELIGIBLE = Verdict(eligible=True)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A row whose original can be deleted."""

    hash: str
    path: str
    category: str
    bytes: int


def purgeable_categories() -> frozenset[str]:
    """Categories whose value survives in the note and the thumbnail.

    Marked ``purgeable = true`` in the taxonomy. Everything else keeps its original.
    """
    return taxonomy.active().purgeable_categories


def selectable_categories() -> tuple[str, ...]:
    """Every value --category will accept."""
    return (*sorted(purgeable_categories()), JUNK_SELECTOR)


def resolve_selection(categories: list[str] | None) -> frozenset[str] | None:
    """Validate a --category selection, returning None when nothing was asked for."""
    if not categories:
        return None
    chosen = {c.strip().lower() for c in categories if c.strip()}
    if not chosen:
        return None
    unknown = sorted(chosen - purgeable_categories() - {JUNK_SELECTOR})
    if unknown:
        allowed = ", ".join(selectable_categories())
        raise CategorySelectionError(f"Not purgeable: {', '.join(unknown)}. Choose from {allowed}.")
    return frozenset(chosen)


def _state_blocker(row: sqlite3.Row, *, thumb_exists: bool, file_exists: bool) -> str | None:
    """Reasons that have nothing to do with what the screenshot contains."""
    if row["purged_at"]:
        return REASON_ALREADY_PURGED
    if not file_exists:
        return REASON_MISSING
    if row["keep"]:
        return REASON_KEPT
    if not thumb_exists:
        # Purging the original with no thumbnail left would lose the image entirely.
        return REASON_NO_THUMBNAIL
    faces = row["faces"]
    if faces is None:
        # Never checked is not the same as no faces, so the original stays.
        return REASON_UNCHECKED
    return REASON_FACES if faces > 0 else None


def _content_blocker(
    row: sqlite3.Row, *, min_text: int, categories: frozenset[str] | None
) -> str | None:
    """Reasons drawn from the category and how much text the note carries."""
    if row["is_junk"]:
        # Junk needs no text to justify going; the thumbnail is more than enough.
        if categories is not None and JUNK_SELECTOR not in categories:
            return REASON_CATEGORY
        return None
    purgeable = purgeable_categories()
    allowed = purgeable if categories is None else (categories & purgeable)
    if (row["category"] or "") not in allowed:
        return REASON_CATEGORY
    if len(row["ocr_text"] or "") < min_text:
        return REASON_THIN_TEXT
    return None


def evaluate(
    row: sqlite3.Row,
    *,
    thumb_exists: bool,
    file_exists: bool,
    min_text: int = DEFAULT_MIN_TEXT,
    categories: frozenset[str] | None = None,
) -> Verdict:
    """Decide whether one row's original can be deleted.

    Every test has to pass. The checks run in reporting order, so a row stops at
    the first thing standing in its way and that is what gets counted.
    """
    reason = _state_blocker(row, thumb_exists=thumb_exists, file_exists=file_exists)
    if reason is None:
        reason = _content_blocker(row, min_text=min_text, categories=categories)
    return ELIGIBLE if reason is None else Verdict(eligible=False, reason=reason)


def plan(
    conn: sqlite3.Connection,
    root: Path,
    *,
    min_text: int = DEFAULT_MIN_TEXT,
    categories: frozenset[str] | None = None,
) -> tuple[list[Candidate], Counter[str]]:
    """Work out what would be purged, and tally why the rest would not be."""
    rows = conn.execute(
        "SELECT hash, path, category, bytes, ocr_text, faces, is_junk, keep, purged_at "
        "FROM screenshots ORDER BY captured_at"
    ).fetchall()

    candidates: list[Candidate] = []
    reasons: Counter[str] = Counter()
    for row in rows:
        verdict = evaluate(
            row,
            thumb_exists=config.thumb_path(root, row["hash"]).is_file(),
            file_exists=config.absolute_path(root, row["path"]).is_file(),
            min_text=min_text,
            categories=categories,
        )
        if verdict.eligible:
            candidates.append(
                Candidate(
                    hash=row["hash"],
                    path=row["path"],
                    category=JUNK_SELECTOR if row["is_junk"] else (row["category"] or "unknown"),
                    bytes=row["bytes"] or 0,
                )
            )
        elif verdict.reason is not None:
            reasons[verdict.reason] += 1
    return candidates, reasons


def by_category(candidates: list[Candidate]) -> list[tuple[str, int, int]]:
    """Candidates rolled up as (category, files, bytes), largest saving first."""
    files: Counter[str] = Counter()
    size: Counter[str] = Counter()
    for candidate in candidates:
        files[candidate.category] += 1
        size[candidate.category] += candidate.bytes
    return sorted(
        ((name, files[name], size[name]) for name in files),
        key=lambda item: (-item[2], item[0]),
    )


def ordered_reasons(reasons: Counter[str]) -> list[tuple[str, int]]:
    """Reason tallies in reporting order, skipping the ones that never fired."""
    return [(reason, reasons[reason]) for reason in REASON_ORDER if reasons.get(reason)]
