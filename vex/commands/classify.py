"""Classify catalogued screenshots into categories, tags and captions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vex import classify, config, db, openrouter, taxonomy
from vex import settings as settings_module
from vex.console import console, counts_table, summary_table

METHODS = classify.METHODS

TIER_LABELS = {
    classify.METHOD_RULES: "Rules",
    classify.METHOD_TEXT: "Text model",
    classify.METHOD_VISION: "Vision model",
    "error": "Errors",
}

MAX_PROBLEMS_SHOWN = 10


def _needs_a_model(method: str) -> bool:
    return method != classify.METHOD_RULES


def command(
    method: Annotated[str, typer.Option("--method", help=f"One of {', '.join(METHODS)}.")] = "auto",
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only process this many rows.")
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Re-classify rows that already have a category.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report the tier split and estimated cost only.")
    ] = False,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the model id for both model tiers.")
    ] = None,
    concurrency: Annotated[
        int | None, typer.Option("--concurrency", help="Worker threads, 1 to 8.")
    ] = None,
    category: Annotated[
        list[str] | None,
        typer.Option(
            "--category",
            help="Only rows already in this category. Repeatable. Usually paired with --force.",
        ),
    ] = None,
    root: Annotated[Path | None, typer.Option("--root", help="Library root.")] = None,
) -> None:
    """Classify screenshots into categories, tags and captions."""
    if method not in METHODS:
        console.print(f"[red]Unknown method '{method}'. Choose one of {', '.join(METHODS)}.[/red]")
        raise typer.Exit(code=2)

    # The root has to be resolved first: it is what says which taxonomy is in force.
    library_root = config.get_root(root)
    names = taxonomy.active().names
    unknown = sorted(set(category or ()) - set(names))
    if unknown:
        console.print(
            f"[red]Unknown category {', '.join(unknown)}. Choose from {', '.join(names)}.[/red]"
        )
        raise typer.Exit(code=2)

    resolved = settings_module.resolve(library_root, model=model, concurrency=concurrency)
    conn = db.connect(library_root)
    rows = classify.select_rows(
        conn, library_root, force=force, limit=limit, method=method, categories=category
    )

    if not rows:
        console.print("Nothing to classify.")
        conn.close()
        return

    client = None
    if _needs_a_model(method):
        try:
            # A dry run only needs the public pricing list, so it works without a key.
            client = openrouter.OpenRouterClient(require_key=not dry_run)
        except openrouter.MissingApiKeyError as exc:
            console.print(f"[red]{exc}[/red]")
            conn.close()
            raise typer.Exit(code=1) from exc

    try:
        if dry_run:
            _report_estimate(
                classify.estimate(rows, method=method, settings=resolved, client=client)
            )
            return

        with console.status("Classifying...") as status:

            def progress(index: int, total: int, outcome: classify.RowOutcome) -> None:
                status.update(f"[{index}/{total}] {outcome.hash[:8]} -> {outcome.tier}")

            summary = classify.run(
                conn,
                rows,
                method=method,
                settings=resolved,
                client=client,
                progress=progress,
            )
        _report_run(summary, resolved)
    finally:
        if client is not None:
            client.close()
        conn.close()


def _report_estimate(estimate: classify.DryRunEstimate) -> None:
    rows = [(TIER_LABELS.get(tier, tier), str(count)) for tier, count in estimate.tiers.items()]
    rows.append(("Rows", str(sum(estimate.tiers.values()))))
    if estimate.priced:
        rows.append(("Estimated cost", f"${estimate.cost_usd:.4f} USD"))
    else:
        rows.append(("Estimated cost", "unknown, no pricing available"))
    console.print(summary_table("Classify (dry run)", rows))
    console.print(
        f"Text model: {estimate.text_model}    Vision model: {estimate.vision_model}",
        style="dim",
    )
    console.print(
        "Rows counted under the text tier may still escalate to vision when the model is unsure.",
        style="dim",
    )


def _report_run(summary: classify.RunSummary, resolved: settings_module.ClassifySettings) -> None:
    console.print(
        summary_table(
            "Classify",
            [
                ("Considered", str(summary.considered)),
                ("Classified", str(summary.classified)),
                ("Model calls", str(summary.calls)),
                ("Cache hits", str(summary.cache_hits)),
                ("Errors", str(summary.errors)),
                ("Cost", f"${summary.cost_usd:.4f} USD"),
            ],
        )
    )

    if summary.tiers:
        console.print(
            counts_table(
                "By tier",
                "Tier",
                [(TIER_LABELS.get(tier, tier), count) for tier, count in summary.tiers.items()],
            )
        )
    if summary.categories:
        console.print(counts_table("By category", "Category", summary.categories.most_common()))
    console.print(
        f"Text model: {resolved.text_model}    Vision model: {resolved.vision_model}",
        style="dim",
    )
    for file_hash, problem in summary.problems[:MAX_PROBLEMS_SHOWN]:
        console.print(f"[yellow]{file_hash[:8]}: {problem}[/yellow]")
    if len(summary.problems) > MAX_PROBLEMS_SHOWN:
        console.print(
            f"[yellow]... and {len(summary.problems) - MAX_PROBLEMS_SHOWN} more.[/yellow]"
        )
