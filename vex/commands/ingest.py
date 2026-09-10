"""Bring screenshots into the library and catalogue them."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from vex import capture, config, db, faces, files, naming, settings, thumbs
from vex.console import console, summary_table

SOURCES = ("auto", "mac", "iphone", "other")
MAX_PROBLEMS_SHOWN = 20

#: Logged against a duplicate that was deleted rather than parked. The backlog
#: loop counts these to report what a year of re-exports actually cost.
DISCARDED_DETAIL = "re-export discarded"


@dataclass
class Counts:
    """Running totals for one ingest run."""

    added: int = 0
    face_checked: int = 0
    duplicates: int = 0
    discarded: int = 0
    near_duplicates: int = 0
    skipped: int = 0
    errors: int = 0
    downloaded: int = 0
    parked: int = 0
    problems: list[str] = field(default_factory=list)
    #: (name, why) for every file still sitting where it was found.
    left_behind: list[tuple[str, str]] = field(default_factory=list)


def discard_inbox_sidecar(root: Path, path: Path) -> None:
    """Remove the sidecar for an ingested file, but only one we wrote into the inbox."""
    sidecar = capture.sidecar_path(path)
    if sidecar is None:
        return
    try:
        sidecar.resolve().relative_to((root / "inbox").resolve())
    except ValueError:
        return  # Someone else's sidecar, sitting outside the inbox. Leave it alone.
    sidecar.unlink(missing_ok=True)


SKIPPED_DIR = "skipped"


def park_skipped(root: Path, path: Path) -> Path | None:
    """Move an inbox file we cannot use into ``skipped/``, sidecar and all.

    Left where it is, it is retried on every run forever, and worse, its sidecar
    goes on telling the next export that its asset has already been fetched. Moved
    out, the file is still there to look at and the asset is free to come again.
    Only ever inside our own inbox: a file of yours that we were pointed at stays
    exactly where you left it.
    """
    if not is_inside_inbox(root, path):
        return None
    parked = files.place_file(
        path, root / SKIPPED_DIR / naming.safe_filename(path.name), copy=False
    )
    sidecar = capture.sidecar_path(path)
    if sidecar is not None:
        files.place_file(
            sidecar, root / SKIPPED_DIR / naming.safe_filename(sidecar.name), copy=False
        )
    return parked


def remember_skipped(
    conn: sqlite3.Connection,
    photos_uuid: str | None,
    name: str,
    reason: str,
    run_at: str,
) -> None:
    """Note that a Photos asset has been set aside, so no export offers it again.

    Moving the file out of the inbox takes its sidecar with it, which frees the
    UUID: the next export sees an asset it has never fetched and fetches it. Then
    ingest sets it aside again. The two of them will do that indefinitely, and did.
    """
    if photos_uuid:
        db.record_skipped_uuid(conn, photos_uuid, name, reason, run_at)


#: How many names the left-behind summary spells out before it gives up counting.
MAX_NAMES_IN_SUMMARY = 10


def left_behind_summary(left_behind: list[tuple[str, str]]) -> str:
    """One line naming what was not taken in, and why.

    A file left in the inbox with a Photos sidecar beside it is invisible twice
    over: nothing moves it, and the sidecar tells the next export that asset has
    already been fetched. Writing the reason down is how that stops being silent.
    """
    named = ", ".join(f"{name} ({why})" for name, why in left_behind[:MAX_NAMES_IN_SUMMARY])
    if len(left_behind) > MAX_NAMES_IN_SUMMARY:
        named += f", and {len(left_behind) - MAX_NAMES_IN_SUMMARY} more"
    return f"{len(left_behind)} left in place: {named}"


def is_inside_inbox(root: Path, path: Path) -> bool:
    """True when a file sits under the root's own inbox, rather than somewhere of yours."""
    try:
        path.resolve().relative_to((root / "inbox").resolve())
    except ValueError:
        return False
    return True


def is_candidate(path: Path) -> bool:
    """True for files worth considering. JSON sidecars are metadata, not candidates."""
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() != ".json"


def gather_paths(
    paths: list[Path], root: Path, *, recursive: bool, mac_inbox: Path | None = None
) -> list[Path]:
    """Expand the given paths into a list of candidate files.

    With nothing named, that is everything under the root's own inbox plus the
    macOS drop zone, which lives outside the root so screenshots keep landing
    somewhere while an external drive is unplugged.
    """
    if not paths:
        inbox = root / "inbox"
        found = sorted(p for p in inbox.rglob("*") if is_candidate(p))
        if mac_inbox is not None and mac_inbox.is_dir() and not is_inside_inbox(root, mac_inbox):
            found += sorted(p for p in mac_inbox.rglob("*") if is_candidate(p))
        return found

    found: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            found.extend(p for p in sorted(iterator) if is_candidate(p))
    return found


def wants_file(
    path: Path,
    source_option: str,
    *,
    in_inbox: bool = False,
    from_photos: bool = False,
) -> tuple[bool, str | None]:
    """Decide whether to ingest a file. Returns (wanted, reason when not).

    The filename pattern is a guess for files found out in the world. It is not
    needed for anything in our own inbox, and it is actively wrong for a Photos
    export: Photos has already decided that asset is a screenshot, and the export
    can rename it to something the pattern rejects, like ``IMG_7390 (1).PNG``.
    """
    if not naming.is_image(path):
        return False, "not an image"
    if source_option != "auto" or in_inbox or from_photos:
        return True, None
    if not naming.looks_like_screenshot(path.name):
        return False, "filename does not match a screenshot pattern"
    return True, None


def inbox_source(path: Path, root: Path) -> str | None:
    """The source implied by which inbox subdirectory a file sits in."""
    try:
        relative = path.resolve().relative_to((root / "inbox").resolve())
    except ValueError:
        return None
    first = relative.parts[0] if relative.parts else ""
    return first if first in config.INBOX_SOURCES else None


def ingest_one(  # noqa: C901, PLR0912, PLR0915
    conn: sqlite3.Connection,
    root: Path,
    path: Path,
    source_option: str,
    run_at: str,
    counts: Counts,
    *,
    copy: bool,
    dry_run: bool,
) -> None:
    """Process a single candidate file, updating counts and the catalogue."""
    sidecar = capture.read_sidecar(path)
    photos_uuid = str(sidecar["photos_uuid"]) if sidecar and sidecar.get("photos_uuid") else None

    wanted, reason = wants_file(
        path,
        source_option,
        in_inbox=is_inside_inbox(root, path),
        from_photos=photos_uuid is not None,
    )
    if not wanted:
        counts.skipped += 1
        detail = reason or "not wanted"
        if not dry_run and park_skipped(root, path) is not None:
            counts.parked += 1
            detail += f", moved to {SKIPPED_DIR}/"
            remember_skipped(conn, photos_uuid, path.name, detail, run_at)
        else:
            counts.left_behind.append((path.name, detail))
        if not dry_run:
            db.log_ingest(conn, run_at, str(path), None, "skipped", detail)
        return

    download = files.ensure_local(path)
    if download.needed_download and download.available:
        counts.downloaded += 1
    if not download.available:
        counts.errors += 1
        detail = download.detail or "file is not available locally"
        counts.problems.append(f"{path.name}: {detail}")
        counts.left_behind.append((path.name, detail))
        if not dry_run:
            db.log_ingest(conn, run_at, str(path), None, "error", detail)
        return

    try:
        stat_result = path.stat()
        file_hash = files.sha256(path)
    except OSError as exc:
        counts.errors += 1
        counts.problems.append(f"{path.name}: {exc}")
        counts.left_behind.append((path.name, str(exc)))
        if not dry_run:
            db.log_ingest(conn, run_at, str(path), None, "error", str(exc))
        return

    existing = db.get_screenshot(conn, file_hash)
    if existing is not None:
        # A purged row still owns its hash, so re-ingesting the same bytes is a
        # duplicate. It must not resurrect the row or bring the original back.
        counts.duplicates += 1
        uuid_from_sidecar = photos_uuid
        # A Photos re-export of a row already held carries nothing the catalogue
        # does not have. Take the UUID off it so the next export skips the asset,
        # then delete the copy rather than parking gigabytes of identical bytes.
        # Only ever inside our own inbox, and never when asked to leave files alone.
        discard = uuid_from_sidecar is not None and not copy and is_inside_inbox(root, path)
        if not dry_run:
            detail = "hash already catalogued"
            if existing["purged_at"]:
                detail += ", original purged"
            if uuid_from_sidecar and not existing["photos_uuid"]:
                db.update_screenshot(conn, file_hash, {"photos_uuid": uuid_from_sidecar})
                detail += ", photos_uuid recorded"
            if discard:
                discard_inbox_sidecar(root, path)
                path.unlink(missing_ok=True)
                counts.discarded += 1
                detail += f", {DISCARDED_DETAIL}"
            elif not copy:
                destination = (
                    root / "duplicates" / f"{file_hash[:8]}_{naming.safe_filename(path.name)}"
                )
                files.place_file(path, destination, copy=False)
            db.log_ingest(conn, run_at, str(path), file_hash, "duplicate", detail)
        return

    properties = files.image_properties(path)
    if properties.width is None:
        counts.errors += 1
        detail = "could not read the image"
        counts.problems.append(f"{path.name}: {detail}")
        if not dry_run and park_skipped(root, path) is not None:
            counts.parked += 1
            detail += f", moved to {SKIPPED_DIR}/"
            remember_skipped(conn, photos_uuid, path.name, detail, run_at)
        else:
            counts.left_behind.append((path.name, detail))
        if not dry_run:
            db.log_ingest(conn, run_at, str(path), file_hash, "error", detail)
        return

    source = source_option
    if source == "auto":
        if photos_uuid:
            # It came out of Photos, whatever the export chose to call it.
            source = "iphone"
        else:
            # The filename is the stronger signal; the inbox drop zone breaks ties.
            detected = naming.detect_source(path.name)
            source = detected if detected != "other" else (inbox_source(path, root) or "other")

    info = capture.extract(path, stat_result)
    captured_at = info.captured_at or capture.to_iso(datetime.now())  # noqa: DTZ005
    device = info.device
    if not device and sidecar:
        device = sidecar.get("device") or None
    if not device:
        device = "mac" if source == "mac" else "iphone" if source == "iphone" else None

    near_duplicate = files.find_near_duplicate(properties.phash, db.all_phashes(conn))
    if near_duplicate:
        counts.near_duplicates += 1

    if dry_run:
        counts.added += 1
        return

    extension = naming.final_extension(path.name) or ".png"
    destination = (
        root
        / naming.library_relative_dir(captured_at)
        / naming.library_filename(captured_at, source, file_hash, extension)
    )
    placed = files.place_file(path, destination, copy=copy)

    db.insert_screenshot(
        conn,
        {
            "hash": file_hash,
            "phash": properties.phash,
            "path": config.relative_path(root, placed),
            "original_name": path.name,
            "original_path": str(path),
            "source": source,
            "captured_at": captured_at,
            "captured_at_source": info.source,
            "device": device,
            "width": properties.width,
            "height": properties.height,
            "bytes": stat_result.st_size,
            "format": properties.format,
            "ingested_at": run_at,
            "near_duplicate_of": near_duplicate,
            "photos_uuid": photos_uuid,
        },
    )
    discard_inbox_sidecar(root, path)
    thumbs.ensure(root, file_hash, placed)

    # Face counts are cheap here and save a separate pass later. Best effort only:
    # anything that goes wrong leaves faces null for 'vex faces' to pick up.
    found = faces.detect_best_effort(root, file_hash)
    if found is not None:
        db.update_screenshot(
            conn,
            file_hash,
            {"faces": found, "faces_at": capture.to_iso(datetime.now())},  # noqa: DTZ005
        )
        counts.face_checked += 1

    db.log_ingest(conn, run_at, str(path), file_hash, "added", config.relative_path(root, placed))
    counts.added += 1


def command(
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Files or directories to ingest. Defaults to everything under inbox/."),
    ] = None,
    source: Annotated[
        str,
        typer.Option("--source", help=f"One of {', '.join(SOURCES)}."),
    ] = "auto",
    copy: Annotated[
        bool,
        typer.Option("--copy", help="Copy instead of moving, leaving the originals in place."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report what would happen without changing anything."),
    ] = False,
    recursive: Annotated[
        bool,
        typer.Option("--recursive", help="Recurse into the given directories."),
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Ingest screenshots into the library."""
    if source not in SOURCES:
        console.print(f"[red]Unknown source '{source}'. Choose one of {', '.join(SOURCES)}.[/red]")
        raise typer.Exit(code=2)

    library_root = config.get_root(root)
    conn = db.connect(library_root)
    run_at = capture.to_iso(datetime.now())  # noqa: DTZ005
    counts = Counts()

    drop_zone = settings.mac_inbox(library_root)
    if not paths and not drop_zone.exists():
        # The zone macOS is pointed at has to exist before anything can land in it.
        drop_zone.mkdir(parents=True, exist_ok=True)
    candidates = gather_paths(
        list(paths or []), library_root, recursive=recursive, mac_inbox=drop_zone
    )
    if not candidates:
        console.print("Nothing to ingest.")
        return

    with console.status(f"Ingesting {len(candidates)} files...") as status:
        for index, path in enumerate(candidates, start=1):
            status.update(f"[{index}/{len(candidates)}] {path.name}")
            ingest_one(
                conn,
                library_root,
                path,
                source,
                run_at,
                counts,
                copy=copy,
                dry_run=dry_run,
            )

    title = "Ingest (dry run)" if dry_run else "Ingest"
    console.print(
        summary_table(
            title,
            [
                ("Added", str(counts.added)),
                ("Face-checked", str(counts.face_checked)),
                ("Duplicates", str(counts.duplicates)),
                ("Re-exports discarded", str(counts.discarded)),
                ("Near duplicates", str(counts.near_duplicates)),
                ("Skipped", str(counts.skipped)),
                (f"Moved to {SKIPPED_DIR}/", str(counts.parked)),
                ("Errors", str(counts.errors)),
                ("Downloaded from iCloud", str(counts.downloaded)),
            ],
        )
    )
    if counts.left_behind:
        summary = left_behind_summary(counts.left_behind)
        console.print(f"[yellow]{summary}[/yellow]")
        if not dry_run:
            db.log_ingest(conn, run_at, str(library_root), None, "left-behind", summary)

    for problem in counts.problems[:MAX_PROBLEMS_SHOWN]:
        console.print(f"[yellow]{problem}[/yellow]")
    if len(counts.problems) > MAX_PROBLEMS_SHOWN:
        remaining = len(counts.problems) - MAX_PROBLEMS_SHOWN
        console.print(f"[yellow]... and {remaining} more[/yellow]")
    conn.close()
