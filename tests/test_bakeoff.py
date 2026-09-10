"""Sampling, running and reporting the model bakeoff."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from tests.test_classify import (
    CLIENT_CLASS,
    LONG_TEXT,
    TEXT_MODEL,
    VISION_MODEL,
    add_row,
    reply,
)
from tests.test_openrouter import MODELS_PAYLOAD, completion_payload
from vex import bakeoff, db, openrouter
from vex.cli import app

runner = CliRunner()

MODELS = (TEXT_MODEL, VISION_MODEL)


def handler_for(verdicts: dict[str, str]) -> object:
    """A transport that answers with a fixed category per model."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        model = json.loads(request.content)["model"]
        return httpx.Response(200, json=completion_payload(reply(verdicts[model]), model=model))

    return handler


def make_client(verdicts: dict[str, str]) -> openrouter.OpenRouterClient:
    return CLIENT_CLASS(
        api_key="test-key",
        require_key=False,
        backoff=0.0,
        transport=httpx.MockTransport(handler_for(verdicts)),
    )


def test_sample_skips_rows_without_a_thumbnail(root: Path) -> None:
    add_row(root, file_hash="a" * 64, ocr_text=LONG_TEXT)
    add_row(root, file_hash="b" * 64, ocr_text=LONG_TEXT, thumbnail=False)
    add_row(root, file_hash="c" * 64, ocr_text="", thumbnail=True)

    conn = db.connect(root)
    sampled = bakeoff.sample_rows(conn, root, size=10, seed=1)
    conn.close()

    assert [row.hash for row in sampled] == ["a" * 64]


def test_sample_is_repeatable_with_a_seed(root: Path) -> None:
    for index in range(6):
        add_row(root, file_hash=str(index) * 64, ocr_text=LONG_TEXT)
    conn = db.connect(root)
    first = [row.hash for row in bakeoff.sample_rows(conn, root, size=3, seed=7)]
    second = [row.hash for row in bakeoff.sample_rows(conn, root, size=3, seed=7)]
    conn.close()
    assert first == second
    assert len(first) == 3


def test_text_only_models_are_not_asked_for_vision() -> None:
    client = make_client({TEXT_MODEL: "article", VISION_MODEL: "article"})
    assert bakeoff.usable_tiers(client, TEXT_MODEL, bakeoff.TIERS) == ["text"]
    assert bakeoff.usable_tiers(client, VISION_MODEL, bakeoff.TIERS) == ["text", "vision"]
    client.close()


def test_run_reports_agreement_latency_and_cost(root: Path) -> None:
    add_row(root, file_hash="a" * 64, ocr_text=LONG_TEXT)
    add_row(root, file_hash="b" * 64, ocr_text=LONG_TEXT)
    client = make_client({TEXT_MODEL: "article", VISION_MODEL: "meme"})
    conn = db.connect(root)
    rows = bakeoff.sample_rows(conn, root, size=10, seed=3)

    result = bakeoff.run(conn, rows, models=MODELS, client=client, concurrency=2)

    # The text model runs one tier, the vision model two, over two screenshots.
    assert len(result.trials) == 6
    by_key = {(report.model, report.tier): report for report in result.reports}
    assert set(by_key) == {
        (TEXT_MODEL, "text"),
        (VISION_MODEL, "text"),
        (VISION_MODEL, "vision"),
    }
    for report in result.reports:
        assert report.succeeded == 2
        assert report.mean_cost_usd > 0
        assert report.median_latency_s >= 0

    # Two of the three runs answered "meme", so that is the consensus.
    assert set(result.consensus.values()) == {"meme"}
    assert by_key[(TEXT_MODEL, "text")].agreement == 0.0
    assert by_key[(VISION_MODEL, "vision")].agreement == 1.0

    # Results are cached so a later classify run with the same model is free.
    assert db.get_cached_classification(conn, rows[0].hash, VISION_MODEL, "vision") is not None
    conn.close()
    client.close()


def test_failures_are_recorded_rather_than_raised(root: Path) -> None:
    add_row(root, file_hash="a" * 64, ocr_text=LONG_TEXT)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        return httpx.Response(400, text="no")

    client = CLIENT_CLASS(
        api_key="k", require_key=False, backoff=0.0, transport=httpx.MockTransport(handler)
    )
    conn = db.connect(root)
    rows = bakeoff.sample_rows(conn, root, size=1, seed=1)
    result = bakeoff.run(conn, rows, models=(TEXT_MODEL,), client=client, concurrency=1)
    conn.close()
    client.close()

    assert result.trials[0].error
    assert result.reports[0].succeeded == 0
    assert result.reports[0].agreement is None


def test_dump_is_written_under_the_root(root: Path) -> None:
    add_row(root, file_hash="a" * 64, ocr_text=LONG_TEXT)
    client = make_client({TEXT_MODEL: "article", VISION_MODEL: "article"})
    conn = db.connect(root)
    rows = bakeoff.sample_rows(conn, root, size=1, seed=1)
    result = bakeoff.run(conn, rows, models=(TEXT_MODEL,), client=client, concurrency=1)
    conn.close()
    client.close()

    path = bakeoff.write_dump(root, result)
    assert path.parent == root / "bakeoff"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["models"] == [TEXT_MODEL]
    assert payload["trials"][0]["category"] == "article"


def test_projection_scales_with_the_image_count() -> None:
    report = bakeoff.ModelReport(model="a/b", tier="text", mean_cost_usd=0.0002)
    assert report.projection(1_700) == pytest.approx(0.34)
    assert report.projection(15_000) == pytest.approx(3.0)


def test_bakeoff_command_needs_rows(root: Path) -> None:
    result = runner.invoke(app, ["bakeoff", "--root", str(root)])
    assert result.exit_code == 0
    assert "No rows" in result.output


def test_bakeoff_command_rejects_an_unknown_tier(root: Path) -> None:
    result = runner.invoke(app, ["bakeoff", "--tier", "smell", "--root", str(root)])
    assert result.exit_code == 2
    assert "Unknown tier" in result.output
