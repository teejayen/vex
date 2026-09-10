"""The weekly pipeline command."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from vex.cli import app
from vex.commands import run as run_module

runner = CliRunner()

STEP_MODULES = (
    "ingest_command",
    "thumbs_command",
    "ocr_command",
    "classify_command",
    "views_command",
    "notes_command",
)


def record_steps(monkeypatch: pytest.MonkeyPatch, failing: str | None = None) -> list[str]:
    """Replace every pipeline step with a recorder, optionally failing at one."""
    order: list[str] = []

    def make(name: str):  # noqa: ANN202
        def step(**_: object) -> None:
            order.append(name)
            if name == failing:
                raise typer.Exit(code=1)

        return step

    for attribute in STEP_MODULES:
        name = attribute.removesuffix("_command")
        monkeypatch.setattr(getattr(run_module, attribute), "command", make(name))
    return order


def test_runs_every_step_in_order(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order = record_steps(monkeypatch)
    result = runner.invoke(app, ["run", "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert order == ["ingest", "thumbs", "ocr", "classify", "views", "notes"]


def test_stops_at_the_first_failing_step(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    order = record_steps(monkeypatch, failing="ocr")
    result = runner.invoke(app, ["run", "--root", str(root)])
    assert result.exit_code == 1
    assert order == ["ingest", "thumbs", "ocr"]
    assert "Stopped" in result.output
    assert "ocr" in result.output


def test_an_unexpected_error_also_stops_the_run(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []

    def boom(**_: object) -> None:
        order.append("views")
        raise RuntimeError("disk went away")

    record_steps(monkeypatch)
    monkeypatch.setattr(run_module.views_command, "command", boom)

    result = runner.invoke(app, ["run", "--root", str(root)])
    assert result.exit_code == 1
    assert "disk went away" in result.output
    assert "notes" not in order
