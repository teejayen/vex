"""Compare candidate classification models over a sample of real screenshots."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from vex import bakeoff, config, db, openrouter
from vex import settings as settings_module
from vex.console import console

DEFAULT_SAMPLE = 20
TIER_CHOICES = ("both", "text", "vision")


def _parse_models(models: str | None) -> tuple[str, ...]:
    if not models:
        return bakeoff.DEFAULT_CANDIDATES
    chosen = tuple(part.strip() for part in models.split(",") if part.strip())
    return chosen or bakeoff.DEFAULT_CANDIDATES


def command(
    sample: Annotated[
        int, typer.Option("--sample", help="How many screenshots to test each model on.")
    ] = DEFAULT_SAMPLE,
    models: Annotated[
        str | None, typer.Option("--models", help="Comma-separated model ids to compare.")
    ] = None,
    tier: Annotated[
        str, typer.Option("--tier", help=f"One of {', '.join(TIER_CHOICES)}.")
    ] = "both",
    seed: Annotated[
        int | None, typer.Option("--seed", help="Seed the sample so a run can be repeated.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", help="Worker threads, 1 to 8.")
    ] = None,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Benchmark candidate models on real screenshots and record the comparison."""
    if tier not in TIER_CHOICES:
        console.print(f"[red]Unknown tier '{tier}'. Choose one of {', '.join(TIER_CHOICES)}.[/red]")
        raise typer.Exit(code=2)

    library_root = config.get_root(root)
    resolved = settings_module.resolve(library_root, concurrency=concurrency)
    candidates = _parse_models(models)
    tiers = bakeoff.TIERS if tier == "both" else (tier,)

    conn = db.connect(library_root)
    rows = bakeoff.sample_rows(conn, library_root, size=sample, seed=seed)
    if not rows:
        console.print(
            "[yellow]No rows with both OCR text and a thumbnail. "
            "Run ingest, thumbs and ocr first.[/yellow]"
        )
        conn.close()
        return

    try:
        client = openrouter.OpenRouterClient()
    except openrouter.MissingApiKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        conn.close()
        raise typer.Exit(code=1) from exc

    console.print(
        f"Comparing {len(candidates)} models over {len(rows)} screenshots in {len(tiers)} tier(s).",
        style="dim",
    )

    try:
        with console.status("Running the bakeoff...") as status:

            def progress(index: int, total: int, trial: bakeoff.Trial) -> None:
                status.update(f"[{index}/{total}] {trial.model} {trial.tier} {trial.hash[:8]}")

            result = bakeoff.run(
                conn,
                rows,
                models=candidates,
                client=client,
                tiers=tiers,
                concurrency=resolved.concurrency,
                progress=progress,
            )
    finally:
        client.close()
        conn.close()

    path = bakeoff.write_dump(library_root, result)
    console.print(_report_table(result))
    console.print(f"Full results: {path}", style="dim")
    console.print(
        "Agreement is measured against the majority category across all models, "
        "not against a hand-labelled truth set.",
        style="dim",
    )


def _report_table(result: bakeoff.BakeoffResult) -> Table:
    table = Table(title="Model bakeoff", title_justify="left")
    table.add_column("Model")
    table.add_column("Tier")
    table.add_column("OK", justify="right")
    table.add_column("Agreement", justify="right")
    table.add_column("Median latency", justify="right")
    table.add_column("Cost per image", justify="right")
    for size in bakeoff.PROJECTION_SIZES:
        table.add_column(f"{size:,} images", justify="right")

    for report in sorted(result.reports, key=lambda item: (item.tier, item.model)):
        agreement = "-" if report.agreement is None else f"{report.agreement * 100:.0f}%"
        row = [
            report.model,
            report.tier,
            f"{report.succeeded}/{report.attempted}",
            agreement,
            f"{report.median_latency_s:.2f}s",
            f"${report.mean_cost_usd:.5f}",
        ]
        row.extend(f"${report.projection(size):.2f}" for size in bakeoff.PROJECTION_SIZES)
        table.add_row(*row)
    return table
