"""Command-line entry point."""

from __future__ import annotations

import typer

from vex import __version__, taxonomy
from vex.commands import (
    bakeoff,
    classify,
    faces,
    ingest,
    keep,
    notes,
    ocr,
    photos,
    purge,
    search,
    stats,
    thumbs,
    views,
)
from vex.commands import (
    run as run_command,
)
from vex.console import console

app = typer.Typer(
    name="vex",
    help="Consolidate, catalogue, OCR and classify screenshots.",
    no_args_is_help=True,
    add_completion=False,
)

# Every command module exposes a `command` callable; register them all here.
app.command("ingest")(ingest.command)
app.command("thumbs")(thumbs.command)
app.command("ocr")(ocr.command)
app.command("faces")(faces.command)
app.command("classify")(classify.command)
app.command("bakeoff")(bakeoff.command)
app.command("views")(views.command)
app.command("notes")(notes.command)
app.command("search")(search.command)
app.command("stats")(stats.command)
app.command("purge")(purge.command)
app.command("keep")(keep.keep_command)
app.command("unkeep")(keep.unkeep_command)
app.command("run")(run_command.command)

# Photos is a group of its own, since it reaches outside the library.
app.add_typer(photos.app, name="photos")


@app.command("version")
def version() -> None:
    """Print the version."""
    print(__version__)


def main() -> None:
    """Console script entry point.

    A broken ``taxonomy.toml`` is a configuration mistake rather than a crash, so it
    is reported as one line and a non-zero exit.
    """
    try:
        app()
    except taxonomy.TaxonomyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(2) from exc
