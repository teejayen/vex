"""Run the whole pipeline in order."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vex import config
from vex.commands import classify as classify_command
from vex.commands import ingest as ingest_command
from vex.commands import notes as notes_command
from vex.commands import ocr as ocr_command
from vex.commands import thumbs as thumbs_command
from vex.commands import views as views_command
from vex.console import console


def command(
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Ingest, thumbnail, read, classify and publish, in that order.

    This is the weekly run. It stops at the first step that fails, so a broken
    step never leaves later ones working from half-finished input.
    """
    library_root = config.get_root(root)

    steps: list[tuple[str, object]] = [
        ("ingest", lambda: ingest_command.command(root=library_root)),
        ("thumbs", lambda: thumbs_command.command(root=library_root)),
        ("ocr", lambda: ocr_command.command(root=library_root)),
        ("classify", lambda: classify_command.command(method="auto", root=library_root)),
        ("views", lambda: views_command.command(root=library_root)),
        ("notes", lambda: notes_command.command(root=library_root)),
    ]

    for number, (name, step) in enumerate(steps, start=1):
        console.rule(f"[bold]{number}/{len(steps)}  {name}")
        try:
            step()
        except typer.Exit as exc:
            if exc.exit_code:
                console.print(
                    f"[red]Stopped: '{name}' failed with exit code {exc.exit_code}.[/red]"
                )
                raise
            console.print(f"[yellow]'{name}' had nothing to do.[/yellow]")
        except Exception as exc:
            console.print(f"[red]Stopped: '{name}' failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc

    console.rule("[bold green]Done")
