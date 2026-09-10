"""Export screenshots out of the macOS Photos library, and delete them from it."""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from vex import backlog as backlog_module
from vex import config, db, photos
from vex.console import console, summary_table

MAX_PROBLEMS_SHOWN = 20
PREVIEW_ROWS = 10

app = typer.Typer(
    name="photos",
    help="Work with the macOS Photos library.",
    no_args_is_help=True,
)


def _export_pending(pending: list, inbox: Path) -> tuple[int, list[str]]:
    """Export each pending photo, collecting failures rather than stopping."""
    exported_at = photos.now_iso()
    exported = 0
    problems: list[str] = []
    with console.status("Exporting...") as status:
        for index, photo in enumerate(pending, start=1):
            status.update(f"[{index}/{len(pending)}] {photo.original_filename}")
            try:
                item = photos.export_one(photo, inbox, exported_at)
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{photo.original_filename}: {exc}")
                continue
            if item is None:
                problems.append(f"{photo.original_filename}: nothing was exported")
                continue
            exported += 1
    return exported, problems


@app.command("export")
def export(
    year: Annotated[
        int | None, typer.Option("--year", help="Only export screenshots captured in this year.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Stop after this many new items.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be exported, changing nothing.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Export detected screenshots from Photos into inbox/iphone.

    Originals held only in iCloud are downloaded on the way through. Re-runs skip
    anything already exported or already catalogued, so this is safe to repeat.
    """
    try:
        photos.require_photos()
    except photos.PhotosUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    library_root = config.get_root(root)
    inbox = library_root / "inbox" / "iphone"

    # A dry run downloads nothing, so it stays useful for planning at any size.
    space = photos.build_check(library_root)
    if not dry_run and space.blocked:
        console.print(f"[red]{photos.low_disk_message(space)}[/red]")
        raise typer.Exit(code=1)

    conn = db.connect(library_root)
    catalogued = {
        row[0]
        for row in conn.execute(
            "SELECT photos_uuid FROM screenshots WHERE photos_uuid IS NOT NULL"
        ).fetchall()
    }
    # Assets ingest could not use. Offering one again only repeats the round trip.
    set_aside_before = db.skipped_uuids(conn)
    conn.close()

    with console.status("Reading the Photos library..."):
        selected, selector = photos.load_screenshots(year)
        found, set_aside = photos.split_by_type(selected)
    console.print(f"Selected by {selector}.")

    already = catalogued | photos.exported_uuids(inbox) | set_aside_before
    new_items = [photo for photo in found if photo.uuid not in already]
    pending = new_items[:limit] if limit else new_items

    console.print(
        summary_table(
            "Photos export (dry run)" if dry_run else "Photos export",
            [
                ("Detected screenshots", str(len(found))),
                ("Set aside, not an image we read", str(sum(set_aside.values()))),
                ("Set aside by an earlier ingest", str(len(set_aside_before))),
                ("Already exported or catalogued", str(len(found) - len(new_items))),
                ("New and not yet exported", str(len(new_items))),
                ("To export in this run", str(len(pending))),
            ],
        )
    )
    for label, count in sorted(set_aside.items()):
        console.print(f"  set aside: {count} {label}")

    result = photos.ExportResult(detected=len(found), new=len(new_items), pending=len(pending))
    if dry_run or not pending:
        if not pending:
            console.print("Nothing new to export.")
        print(result.format())
        return

    exported, problems = _export_pending(pending, inbox)
    failed = len(problems)
    result = replace(result, exported=exported, failed=failed)

    console.print(
        summary_table(
            "Exported",
            [("Written to inbox/iphone", str(exported)), ("Failed", str(failed))],
        )
    )
    for problem in problems[:MAX_PROBLEMS_SHOWN]:
        console.print(f"[yellow]{problem}[/yellow]")
    if len(problems) > MAX_PROBLEMS_SHOWN:
        console.print(f"[yellow]... and {len(problems) - MAX_PROBLEMS_SHOWN} more[/yellow]")
    if exported:
        console.print("Run 'vex ingest' to bring these into the library.")
    print(result.format())


def sort_for_deletion(root: Path, rows: list) -> tuple[list, list, list]:
    """Split catalogued rows into those Photos need not keep, and those it must.

    A purged row is not a missing one. Its original was deleted deliberately,
    because the note, the thumbnail and the catalogue row already hold what it was
    for, and that is exactly the point at which Photos no longer needs to be
    holding the only other copy. A row with neither an original nor a thumbnail is
    a different thing entirely, and is always refused.

    Returns (still has its original, purged but held by its thumbnail, refused).
    """
    with_original: list = []
    purged: list = []
    refused: list = []
    for row in rows:
        if config.absolute_path(root, row["path"]).is_file():
            with_original.append(row)
        elif row["purged_at"] and config.thumb_path(root, row["hash"]).is_file():
            purged.append(row)
        else:
            refused.append(row)
    return with_original, purged, refused


def run_deletion(conn: sqlite3.Connection, eligible: list, chunk: int) -> int | None:
    """Ask Photos to delete the eligible assets. Returns None if the person declined.

    Each request is one confirmation dialogue, so by default there is exactly one
    of both. Rows are marked as the requests succeed, which is what lets a rerun
    pick up where an interrupted one stopped.
    """
    by_uuid = {row["photos_uuid"]: row["hash"] for row in eligible}
    batches = photos.chunked(list(by_uuid), chunk)
    if len(batches) > 1:
        console.print(
            f"{len(batches)} requests, so macOS will ask you to confirm {len(batches)} times."
        )

    deleted = 0
    for index, batch in enumerate(batches, start=1):
        try:
            went = photos.delete_assets(batch)
        except photos.DeletionDeclinedError:
            console.print(
                "[yellow]Declined, nothing deleted.[/yellow] "
                f"{deleted} had already gone in earlier requests."
            )
            return None

        # The request succeeded, so Photos is no longer holding any of these,
        # including any it could not find because they had already gone.
        deleted_at = photos.now_iso()
        for uuid in batch:
            db.update_screenshot(conn, by_uuid[uuid], {"photos_deleted_at": deleted_at})
        deleted += went
        if len(batches) > 1:
            console.print(f"  request {index}: deleted {went} of {len(batch)} (total {deleted})")
    return deleted


@app.command("delete")
def delete(
    year: Annotated[
        int | None, typer.Option("--year", help="Only consider screenshots from this year.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="List what would be deleted. This is the default.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", help="Actually delete. Without this, nothing is touched.")
    ] = False,
    chunk: Annotated[
        int,
        typer.Option(
            "--chunk",
            help="Assets per PhotoKit request, and so per confirmation dialogue. 0 means one.",
        ),
    ] = photos.DEFAULT_DELETE_CHUNK,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Delete screenshots from Photos once they are safely in the library.

    Only rows this catalogue knows about are ever considered, and only when the
    library file is on disk and the text has been read out of it. Deleted items
    go to Photos' Recently Deleted, so there is a window to change your mind.
    """
    try:
        photos.require_photos()
    except photos.PhotosUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    library_root = config.get_root(root)
    conn = db.connect(library_root)

    sql = (
        "SELECT hash, photos_uuid, path, captured_at, purged_at FROM screenshots "
        "WHERE photos_uuid IS NOT NULL AND ocr_text IS NOT NULL "
        "AND photos_deleted_at IS NULL"
    )
    params: list[object] = []
    if year is not None:
        sql += " AND substr(captured_at, 1, 4) = ?"
        params.append(str(year))
    rows = conn.execute(sql + " ORDER BY captured_at", params).fetchall()

    with_original, purged, refused = sort_for_deletion(library_root, rows)
    eligible = with_original + purged

    console.print(
        summary_table(
            "Photos delete",
            [
                ("Catalogued with a Photos UUID and text", str(len(rows))),
                ("Original still on disk", str(len(with_original))),
                ("Purged, held by its note and thumbnail", str(len(purged))),
                ("No original and no thumbnail, so refused", str(len(refused))),
                ("Eligible for deletion", str(len(eligible))),
            ],
        )
    )
    if not eligible:
        console.print("Nothing to delete.")
        conn.close()
        return

    for row in eligible[:PREVIEW_ROWS]:
        console.print(f"  {row['captured_at']}  {row['photos_uuid']}  {row['path']}")
    if len(eligible) > PREVIEW_ROWS:
        console.print(f"  ... and {len(eligible) - PREVIEW_ROWS} more")

    if dry_run or not yes:
        console.print(
            "\n[yellow]Nothing was deleted.[/yellow] "
            "Re-run with --yes to move these to Photos' Recently Deleted."
        )
        conn.close()
        return

    try:
        deleted = run_deletion(conn, eligible, chunk)
    except photos.PhotosUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        conn.close()
        raise typer.Exit(code=1) from exc
    conn.close()
    if deleted is None:
        return
    console.print(
        f"\nMoved {deleted} screenshots to Photos' Recently Deleted. "
        "They stay there for 30 days before Photos removes them for good."
    )


@app.command("backlog")
def backlog(
    years: Annotated[
        str,
        typer.Option("--years", help="Comma-separated years to work through, oldest first."),
    ] = ",".join(str(y) for y in backlog_module.DEFAULT_YEARS),
    batch: Annotated[
        int, typer.Option("--batch", help="Items to export per batch.")
    ] = backlog_module.DEFAULT_BATCH,
    min_free_gb: Annotated[
        float,
        typer.Option("--min-free-gb", help="Stop cleanly when free space falls below this."),
    ] = backlog_module.DEFAULT_MIN_FREE_GB,
    max_batches: Annotated[
        int | None, typer.Option("--max-batches", help="Stop after this many batches.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan and change nothing.")
    ] = False,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Drain the Photos backlog overnight, a batch at a time.

    Each batch exports a year's screenshots, runs the whole pipeline over them,
    and purges the originals the notes have already captured. It stops when the
    years are drained, when free space runs low, or when it is asked to. Nothing
    is ever deleted from Photos: that stays a separate, deliberate command.
    """
    try:
        chosen = backlog_module.parse_years(years)
    except backlog_module.YearListError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc

    try:
        photos.require_photos()
    except photos.PhotosUnavailableError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    library_root = config.get_root(root)
    free_now = backlog_module.free_bytes(library_root) / backlog_module.BYTES_PER_GB

    if dry_run:
        _print_plan(library_root, chosen, batch, min_free_gb, free_now, max_batches)
        return

    if free_now < min_free_gb:
        console.print(
            f"[red]Only {free_now:.1f} GB free, below the {min_free_gb:.1f} GB floor. "
            "Nothing was started.[/red]"
        )
        raise typer.Exit(code=1)

    log = backlog_module.RunLog(backlog_module.log_path(library_root))
    console.print(f"Logging to {log.path}")
    run = backlog_module.BacklogRun(
        root=library_root,
        log=log,
        years=chosen,
        batch=batch,
        min_free_gb=min_free_gb,
        max_batches=max_batches,
    )
    with backlog_module.stop_on_signals(run.stop):
        summary = run.run()
    backlog_module.report(summary, log)


def _print_plan(
    root: Path,
    years: list[int],
    batch: int,
    min_free_gb: float,
    free_now: float,
    max_batches: int | None,
) -> None:
    """What the run would do, without touching anything."""
    console.print(
        summary_table(
            "Backlog plan",
            [
                ("Years", ", ".join(str(y) for y in years)),
                ("Batch size", str(batch)),
                ("Free space now", f"{free_now:.1f} GB"),
                ("Stop below", f"{min_free_gb:.1f} GB"),
                ("Batch limit", str(max_batches) if max_batches else "none"),
                ("Root", str(root)),
            ],
        )
    )

    with console.status("Reading the Photos library..."):
        plans = backlog_module.survey(root, years)

    table = Table(title="Still in Photos", title_justify="left")
    table.add_column("Year")
    table.add_column("Detected", justify="right")
    table.add_column("Already in the library", justify="right")
    table.add_column("Remaining", justify="right")
    table.add_column("Batches", justify="right")
    for plan in plans:
        table.add_row(
            str(plan.year),
            str(plan.detected),
            str(plan.already),
            str(plan.remaining),
            str(plan.batches(batch)),
        )
    remaining = sum(plan.remaining for plan in plans)
    table.add_section()
    table.add_row(
        "Total",
        str(sum(plan.detected for plan in plans)),
        str(sum(plan.already for plan in plans)),
        str(remaining),
        str(sum(plan.batches(batch) for plan in plans)),
        style="bold",
    )
    console.print(table)
    console.print(
        f"Dry run. {remaining} screenshots would be worked through in batches of {batch}. "
        "Nothing was exported, ingested or purged."
    )
