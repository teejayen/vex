"""Tier selection, caching, writing results and the classify command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from tests.conftest import make_image
from tests.test_openrouter import MODELS_PAYLOAD, completion_payload
from vex import classify, config, db, openrouter, settings, taxonomy
from vex.cli import app
from vex.commands import notes

runner = CliRunner()

TEXT_MODEL = "test/text"
VISION_MODEL = "test/vision"

LONG_TEXT = (
    "A long note about a product idea for the consulting business, written out "
    "properly so the text tier has something real to work with."
)

SETTINGS = settings.ClassifySettings(
    text_model=TEXT_MODEL, vision_model=VISION_MODEL, concurrency=2
)


def reply(category: str, confidence: float = 0.9, **extra: Any) -> str:
    payload = {
        "category": category,
        "tags": ["one", "two"],
        "caption": f"A screenshot about {category}.",
        "confidence": confidence,
        "is_junk": category == "junk",
        "use": None,
        "recipe": None,
    }
    payload.update(extra)
    return json.dumps(payload)


class Recorder:
    """A mocked transport that records which tier each request came from."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.requests: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        body = json.loads(request.content)
        model = body["model"]
        self.requests.append(model)
        return httpx.Response(200, json=completion_payload(self.replies[model], model=model))

    @property
    def calls(self) -> int:
        return len(self.requests)


#: Bound before any monkeypatching so the fixtures never patch themselves into a loop.
CLIENT_CLASS = openrouter.OpenRouterClient


def make_client(recorder: Recorder) -> openrouter.OpenRouterClient:
    return CLIENT_CLASS(
        api_key="test-key",
        require_key=False,
        backoff=0.0,
        transport=httpx.MockTransport(recorder),
    )


def add_row(
    root: Path,
    *,
    file_hash: str = "a" * 64,
    ocr_text: str = "",
    original_name: str = "Screenshot 2026-07-20 at 14.03.50.png",
    thumbnail: bool = True,
) -> classify.RowInput:
    """Insert a catalogue row and, by default, put a thumbnail beside it."""
    conn = db.connect(root)
    db.insert_screenshot(
        conn,
        {
            "hash": file_hash,
            "path": f"library/2026/07/{file_hash[:8]}.png",
            "original_name": original_name,
            "source": "mac",
            "captured_at": "2026-07-20T14:03:50",
            "width": 1920,
            "height": 1080,
            "ingested_at": "2026-07-21T09:00:00",
            "ocr_text": ocr_text,
        },
    )
    conn.close()
    if thumbnail:
        make_image(config.thumb_path(root, file_hash), "thumb")
    return classify.RowInput(
        hash=file_hash,
        original_name=original_name,
        source="mac",
        width=1920,
        height=1080,
        ocr_text=ocr_text,
        thumb=config.thumb_path(root, file_hash),
    )


# ---------------------------------------------------------------------------
# Tier selection
# ---------------------------------------------------------------------------


INVOICE_TEXT = "TAX INVOICE\nAmount due $1,320.00\nSubtotal $1,200.00\nDue date 14 August"


def test_auto_sends_a_confident_rules_hit_to_a_model_anyway(root: Path) -> None:
    # Keywords are good at spotting evidence and bad at reading it in context, so a
    # non-junk rules verdict is only ever a hint now.
    row = add_row(root, ocr_text=INVOICE_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("shopping", 0.95)})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert recorder.requests == [TEXT_MODEL]
    assert outcome.tier == classify.METHOD_TEXT
    assert outcome.classification.category == "shopping"
    client.close()


def test_auto_stops_only_on_confident_junk(root: Path) -> None:
    row = add_row(root, ocr_text="Swipe up to open. Face ID. Enter passcode.")
    recorder = Recorder({})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert outcome.tier == classify.METHOD_RULES
    assert outcome.classification.category == "junk"
    assert outcome.classification.is_junk
    assert outcome.classification.method == "rules"
    assert outcome.classification.model is None
    assert recorder.calls == 0
    client.close()


def test_the_rules_guess_is_passed_to_the_model_as_a_hint(root: Path) -> None:
    row = add_row(root, ocr_text=INVOICE_TEXT)
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion_payload(reply("receipt"), model=TEXT_MODEL))

    client = CLIENT_CLASS(
        api_key="k", require_key=False, backoff=0.0, transport=httpx.MockTransport(handler)
    )
    classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    prompt = bodies[0]["messages"][1]["content"]
    assert 'guessed "receipt"' in prompt
    assert "invoice" in prompt
    # The hint must not read as an instruction to agree with it.
    assert "often wrong" in prompt
    client.close()


def test_no_hint_is_offered_when_the_keywords_found_nothing() -> None:
    assert classify.hint_line(None) == ""


def test_auto_uses_the_text_tier_for_substantial_text(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("article", 0.88)})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert recorder.requests == [TEXT_MODEL]
    assert outcome.tier == classify.METHOD_TEXT
    assert outcome.classification.method == "text-model"
    assert outcome.classification.model == TEXT_MODEL
    client.close()


def test_auto_escalates_to_vision_when_the_text_tier_is_unsure(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("other", 0.4), VISION_MODEL: reply("meme", 0.95)})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert recorder.requests == [TEXT_MODEL, VISION_MODEL]
    assert outcome.tier == classify.METHOD_VISION
    assert outcome.classification.category == "meme"
    client.close()


def test_auto_skips_the_text_tier_when_there_is_barely_any_text(root: Path) -> None:
    row = add_row(root, ocr_text="Done")
    recorder = Recorder({VISION_MODEL: reply("meme")})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert recorder.requests == [VISION_MODEL]
    assert outcome.tier == classify.METHOD_VISION
    client.close()


def test_explicit_vision_method_never_calls_the_text_model(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({VISION_MODEL: reply("social-post")})
    client = make_client(recorder)

    outcome = classify.classify_row(
        row, method="vision", settings=SETTINGS, client=client, cache={}
    )

    assert recorder.requests == [VISION_MODEL]
    assert outcome.classification.category == "social-post"
    client.close()


def test_rules_method_never_calls_a_model(root: Path) -> None:
    row = add_row(root, ocr_text="something entirely unremarkable about the weather outside")
    outcome = classify.classify_row(row, method="rules", settings=SETTINGS, client=None, cache={})
    assert outcome.classification.category == "other"
    assert outcome.classification.confidence < 0.5


def test_missing_thumbnail_falls_back_to_the_text_result(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT, thumbnail=False)
    recorder = Recorder({TEXT_MODEL: reply("other", 0.3)})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert recorder.requests == [TEXT_MODEL]
    assert outcome.tier == classify.METHOD_TEXT
    assert outcome.error is None
    client.close()


def test_missing_thumbnail_with_no_text_is_an_error(root: Path) -> None:
    row = add_row(root, ocr_text="", thumbnail=False, file_hash="b" * 64)
    recorder = Recorder({})
    client = make_client(recorder)

    outcome = classify.classify_row(
        row, method="vision", settings=SETTINGS, client=client, cache={}
    )

    assert outcome.tier == "error"
    assert "thumbnail" in outcome.error
    assert recorder.calls == 0
    client.close()


# ---------------------------------------------------------------------------
# The structured payload
# ---------------------------------------------------------------------------

RECIPE_PAYLOAD = {
    "title": "Slow-cooked lamb shoulder",
    "ingredients": ["2 tbsp olive oil", "4 cloves garlic, finely chopped"],
    "method": ["Preheat the oven to 160C.", "Cook for 4 hours."],
    "serves": "6",
    "source": "Recipe Tin Eats",
}


def test_a_reuse_note_is_stored_for_the_use_note_categories(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    angle = "Use as the opening example for a post on why scoping calls stall."
    recorder = Recorder({TEXT_MODEL: reply("article", 0.9, use=angle)})
    client = make_client(recorder)
    conn = db.connect(root)

    rows = classify.select_rows(conn, root)
    classify.run(conn, rows, method="auto", settings=SETTINGS, client=client)

    stored = db.get_screenshot(conn, "a" * 64)
    assert db.load_json(stored["structured"]) == {"use": angle}
    conn.close()
    client.close()


@pytest.mark.parametrize("category", sorted(taxonomy.active().use_note_categories))
def test_every_use_note_category_keeps_its_angle(category: str) -> None:
    result = taxonomy.normalise_result(
        {"category": category, "use": "  Keep as a   prompt pattern. ", "recipe": None},
        method="text-model",
        model="test/model",
    )
    assert result.structured == {"use": "Keep as a prompt pattern."}


@pytest.mark.parametrize("use", [None, "", "   ", 42, "N/A"])
def test_an_unusable_reuse_note_is_not_stored(use: object) -> None:
    result = taxonomy.normalise_result(
        {"category": "article", "use": use}, method="text-model", model="test/model"
    )
    expected = {"use": "N/A"} if use == "N/A" else None
    assert result.structured == expected


def test_a_reuse_note_is_ignored_outside_the_four_categories() -> None:
    result = taxonomy.normalise_result(
        {"category": "shopping", "use": "Buy it later."},
        method="text-model",
        model="test/model",
    )
    assert result.structured is None


def test_a_recipe_is_stored_and_renders_in_a_note(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("recipe", 0.95, recipe=RECIPE_PAYLOAD)})
    client = make_client(recorder)
    conn = db.connect(root)

    rows = classify.select_rows(conn, root)
    classify.run(conn, rows, method="auto", settings=SETTINGS, client=client)

    stored = db.get_screenshot(conn, "a" * 64)
    assert db.load_json(stored["structured"]) == RECIPE_PAYLOAD

    note = notes.render(stored)
    assert "# Slow-cooked lamb shoulder" in note
    assert "## Ingredients" in note
    assert "- 2 tbsp olive oil" in note
    assert "## Method" in note
    assert "1. Preheat the oven to 160C." in note
    assert "2. Cook for 4 hours." in note
    assert "Serves: 6" in note
    assert "Source: Recipe Tin Eats" in note
    conn.close()
    client.close()


def test_a_reuse_note_renders_in_a_note(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    angle = "Keep as a prompt pattern for scoping calls."
    recorder = Recorder({TEXT_MODEL: reply("article", 0.9, use=angle)})
    client = make_client(recorder)
    conn = db.connect(root)

    classify.run(
        conn, classify.select_rows(conn, root), method="auto", settings=SETTINGS, client=client
    )
    note = notes.render(db.get_screenshot(conn, "a" * 64))

    assert "## Use" in note
    assert angle in note
    conn.close()
    client.close()


def test_the_old_nested_shape_is_still_read() -> None:
    result = taxonomy.normalise_result(
        {"category": "recipe", "structured": RECIPE_PAYLOAD},
        method="text-model",
        model="test/model",
    )
    assert result.structured == RECIPE_PAYLOAD


def test_a_reply_cached_under_the_old_schema_is_asked_again(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    stale = json.dumps(
        completion_payload(
            json.dumps(
                {
                    "category": "article",
                    "tags": [],
                    "caption": "old",
                    "confidence": 0.9,
                    "is_junk": False,
                    "structured": {},
                }
            ),
            model=TEXT_MODEL,
        )
    )
    angle = "Use as the hook for a post on deterministic pipelines."
    recorder = Recorder({TEXT_MODEL: reply("article", 0.9, use=angle)})
    client = make_client(recorder)

    outcome = classify.classify_row(
        row,
        method="auto",
        settings=SETTINGS,
        client=client,
        cache={(row.hash, TEXT_MODEL, "text"): stale},
    )

    assert outcome.cache_hits == 0
    assert recorder.calls == 1
    assert outcome.classification.structured == {"use": angle}
    client.close()


def test_a_reply_cached_under_the_current_schema_is_reused(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    angle = "Turn it into a worked example."
    fresh = json.dumps(completion_payload(reply("article", 0.9, use=angle), model=TEXT_MODEL))
    recorder = Recorder({})
    client = make_client(recorder)

    outcome = classify.classify_row(
        row,
        method="auto",
        settings=SETTINGS,
        client=client,
        cache={(row.hash, TEXT_MODEL, "text"): fresh},
    )

    assert outcome.cache_hits == 1
    assert recorder.calls == 0
    assert outcome.classification.structured == {"use": angle}
    client.close()


def test_the_schema_asks_for_use_and_recipe_by_name() -> None:
    schema = taxonomy.response_schema()
    assert "use" in schema["required"]
    assert "recipe" in schema["required"]
    # The old free-form field is what the models answered with an empty object.
    assert "structured" not in schema["properties"]
    assert schema["properties"]["use"]["description"]
    assert set(schema["properties"]["recipe"]["required"]) == set(taxonomy.RECIPE_STRUCTURED_KEYS)


# ---------------------------------------------------------------------------
# Replies that ran out of room
# ---------------------------------------------------------------------------

#: What a reply looks like when the model hit the output ceiling mid-caption.
CUT_OFF = (
    '{"category": "recipe", "tags": ["lamb", "slow-cook"], "caption": "Slow-cooked lamb should'
)


def truncating_handler(cut_off_calls: int, bodies: list[dict[str, Any]]) -> object:
    """Answers with a cut-off reply for the first N calls, then a complete one."""
    seen = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        body = json.loads(request.content)
        bodies.append(body)
        seen["count"] += 1
        if seen["count"] <= cut_off_calls:
            payload = completion_payload(CUT_OFF, model=body["model"], completion_tokens=1500)
            payload["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json=completion_payload(reply("recipe"), model=body["model"]))

    return handler


def test_a_cut_off_reply_is_retried_with_more_room(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    bodies: list[dict[str, Any]] = []
    client = CLIENT_CLASS(
        api_key="k",
        require_key=False,
        backoff=0.0,
        transport=httpx.MockTransport(truncating_handler(1, bodies)),
    )

    outcome = classify.classify_row(row, method="text", settings=SETTINGS, client=client, cache={})

    assert outcome.error is None
    assert outcome.classification.category == "recipe"
    assert outcome.calls == 2
    assert bodies[0]["max_tokens"] == classify.MAX_OUTPUT_TOKENS
    assert bodies[1]["max_tokens"] == classify.TRUNCATED_RETRY_OUTPUT_TOKENS
    assert bodies[1]["max_tokens"] > bodies[0]["max_tokens"]
    # The cut-off reply is never cached, only the one that parsed.
    assert len(outcome.cache_writes) == 1
    assert "Slow-cooked lamb should" not in outcome.cache_writes[0].response_json
    client.close()


def test_a_reply_cut_off_twice_is_reported_not_cached(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    bodies: list[dict[str, Any]] = []
    client = CLIENT_CLASS(
        api_key="k",
        require_key=False,
        backoff=0.0,
        transport=httpx.MockTransport(truncating_handler(99, bodies)),
    )

    outcome = classify.classify_row(row, method="text", settings=SETTINGS, client=client, cache={})

    assert outcome.tier == "error"
    assert "cut off" in outcome.error
    assert outcome.calls == 2
    assert outcome.cache_writes == []
    client.close()


def test_a_cut_off_vision_reply_is_retried_too(root: Path) -> None:
    row = add_row(root, ocr_text="ok")
    bodies: list[dict[str, Any]] = []
    client = CLIENT_CLASS(
        api_key="k",
        require_key=False,
        backoff=0.0,
        transport=httpx.MockTransport(truncating_handler(1, bodies)),
    )

    outcome = classify.classify_row(row, method="auto", settings=SETTINGS, client=client, cache={})

    assert outcome.tier == classify.METHOD_VISION
    assert outcome.classification.category == "recipe"
    assert [body["model"] for body in bodies] == [VISION_MODEL, VISION_MODEL]
    client.close()


def test_a_complete_reply_is_not_retried(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("article", 0.9)})
    client = make_client(recorder)

    outcome = classify.classify_row(row, method="text", settings=SETTINGS, client=client, cache={})

    assert outcome.calls == 1
    assert recorder.calls == 1
    client.close()


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_cache_hit_avoids_a_model_call(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    cached = json.dumps(completion_payload(reply("recipe", 0.91), model=TEXT_MODEL))
    recorder = Recorder({})
    client = make_client(recorder)

    outcome = classify.classify_row(
        row,
        method="auto",
        settings=SETTINGS,
        client=client,
        cache={(row.hash, TEXT_MODEL, "text"): cached},
    )

    assert recorder.calls == 0
    assert outcome.cache_hits == 1
    assert outcome.cost_usd == 0.0
    assert outcome.classification.category == "recipe"
    client.close()


def test_a_run_populates_the_cache_and_a_rerun_reads_it(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("article", 0.9)})
    client = make_client(recorder)
    conn = db.connect(root)

    rows = classify.select_rows(conn, root)
    classify.run(conn, rows, method="auto", settings=SETTINGS, client=client)
    assert recorder.calls == 1

    cached = db.get_cached_classification(conn, rows[0].hash, TEXT_MODEL, "text")
    assert cached is not None
    assert cached["input_tokens"] == 1000

    forced = classify.select_rows(conn, root, force=True)
    summary = classify.run(conn, forced, method="auto", settings=SETTINGS, client=client)
    assert recorder.calls == 1
    assert summary.cache_hits == 1
    conn.close()
    client.close()


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------


def test_run_writes_every_classification_column(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    structured = {"use": "Turn it into a post about scoping."}
    recorder = Recorder({TEXT_MODEL: reply("article", 0.93, structured=structured)})
    client = make_client(recorder)
    conn = db.connect(root)

    rows = classify.select_rows(conn, root)
    summary = classify.run(conn, rows, method="auto", settings=SETTINGS, client=client)

    stored = db.get_screenshot(conn, rows[0].hash)
    assert stored["category"] == "article"
    assert db.load_json(stored["tags"]) == ["one", "two"]
    assert stored["classify_method"] == "text-model"
    assert stored["classify_model"] == TEXT_MODEL
    assert stored["classify_confidence"] == pytest.approx(0.93)
    assert stored["classified_at"]
    assert db.load_json(stored["structured"]) == structured
    assert stored["is_junk"] == 0
    assert summary.classified == 1
    assert summary.cost_usd > 0
    conn.close()
    client.close()


def classified_as(root: Path, file_hash: str, **values: object) -> None:
    conn = db.connect(root)
    db.update_screenshot(conn, file_hash, values)
    conn.close()


def test_auto_revisits_every_rules_verdict_except_confident_junk(root: Path) -> None:
    # A weak rules verdict.
    add_row(root, file_hash="1" * 64, ocr_text=LONG_TEXT)
    classified_as(
        root, "1" * 64, category="other", classify_method="rules", classify_confidence=0.3
    )
    # Junk only because there was no text to read, so a model should still look.
    add_row(root, file_hash="2" * 64, ocr_text="")
    classified_as(
        root, "2" * 64, category="junk", classify_method="rules", classify_confidence=0.5, is_junk=1
    )
    # A confident keyword hit. Still only a hint, so it goes back to a model.
    add_row(root, file_hash="3" * 64, ocr_text=LONG_TEXT)
    classified_as(
        root, "3" * 64, category="receipt", classify_method="rules", classify_confidence=0.9
    )
    # Confident junk is the one thing the keywords settle on their own.
    add_row(root, file_hash="7" * 64, ocr_text="Swipe up to open")
    classified_as(
        root, "7" * 64, category="junk", classify_method="rules", classify_confidence=0.9, is_junk=1
    )
    # Anything a model or a person already decided is left alone.
    add_row(root, file_hash="4" * 64, ocr_text=LONG_TEXT)
    classified_as(
        root, "4" * 64, category="other", classify_method="text-model", classify_confidence=0.4
    )
    add_row(root, file_hash="5" * 64, ocr_text=LONG_TEXT)
    classified_as(
        root, "5" * 64, category="other", classify_method="manual", classify_confidence=1.0
    )
    # An untouched row is picked up either way.
    add_row(root, file_hash="6" * 64, ocr_text=LONG_TEXT)

    conn = db.connect(root)
    auto = {row.hash for row in classify.select_rows(conn, root, method="auto")}
    rules = {row.hash for row in classify.select_rows(conn, root, method="rules")}
    forced = {row.hash for row in classify.select_rows(conn, root, force=True)}
    conn.close()

    assert auto == {"1" * 64, "2" * 64, "3" * 64, "6" * 64}
    # The rules tier has nothing new to say about its own verdicts.
    assert rules == {"6" * 64}
    assert len(forced) == 7


def test_a_rules_row_with_no_confidence_recorded_is_revisited(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    classified_as(root, "a" * 64, category="recipe", classify_method="rules")
    conn = db.connect(root)
    assert [row.hash for row in classify.select_rows(conn, root, method="auto")] == ["a" * 64]
    conn.close()


def test_a_category_filter_narrows_the_selection(root: Path) -> None:
    add_row(root, file_hash="1" * 64, ocr_text=LONG_TEXT)
    classified_as(root, "1" * 64, category="other", classify_method="text-model")
    add_row(root, file_hash="2" * 64, ocr_text=LONG_TEXT)
    classified_as(root, "2" * 64, category="junk", classify_method="vision-model", is_junk=1)
    add_row(root, file_hash="3" * 64, ocr_text=LONG_TEXT)
    classified_as(root, "3" * 64, category="recipe", classify_method="text-model")
    # Never classified, so it is not in any category bucket.
    add_row(root, file_hash="4" * 64, ocr_text=LONG_TEXT)

    conn = db.connect(root)
    one = classify.select_rows(conn, root, force=True, categories=["other"])
    two = classify.select_rows(conn, root, force=True, categories=["other", "recipe"])
    # Without --force the filter still only sees rows that would have been selected.
    unforced = classify.select_rows(conn, root, categories=["other"])
    none_given = classify.select_rows(conn, root, force=True)
    conn.close()

    assert [row.hash for row in one] == ["1" * 64]
    assert {row.hash for row in two} == {"1" * 64, "3" * 64}
    assert unforced == []
    assert len(none_given) == 4


def test_a_vision_pass_over_other_keeps_a_second_other_verdict(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    classified_as(
        root, "a" * 64, category="other", classify_method="text-model", classify_confidence=0.5
    )
    recorder = Recorder({VISION_MODEL: reply("other", 0.85)})
    client = make_client(recorder)
    conn = db.connect(root)

    rows = classify.select_rows(conn, root, force=True, method="vision", categories=["other"])
    summary = classify.run(conn, rows, method="vision", settings=SETTINGS, client=client)

    assert recorder.requests == [VISION_MODEL]
    stored = db.get_screenshot(conn, "a" * 64)
    # Still "other", but now it is the vision tier saying so.
    assert stored["category"] == "other"
    assert stored["classify_method"] == "vision-model"
    assert stored["classify_model"] == VISION_MODEL
    assert stored["classify_confidence"] == pytest.approx(0.85)
    assert summary.classified == 1
    conn.close()
    client.close()


def test_the_command_rejects_an_unknown_category(root: Path) -> None:
    result = runner.invoke(app, ["classify", "--category", "sandwiches", "--root", str(root)])
    assert result.exit_code == 2
    assert "Unknown category sandwiches" in result.output


def test_the_command_passes_the_category_filter_through(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_row(root, file_hash="1" * 64, ocr_text=LONG_TEXT)
    classified_as(root, "1" * 64, category="other", classify_method="text-model")
    add_row(root, file_hash="2" * 64, ocr_text=LONG_TEXT)
    classified_as(root, "2" * 64, category="recipe", classify_method="text-model")
    recorder = Recorder({VISION_MODEL: reply("meme", 0.9)})
    use_test_models(monkeypatch, recorder)

    result = runner.invoke(
        app,
        ["classify", "--method", "vision", "--category", "other", "--force", "--root", str(root)],
    )

    assert result.exit_code == 0, result.output
    assert recorder.calls == 1
    conn = db.connect(root)
    assert db.get_screenshot(conn, "1" * 64)["category"] == "meme"
    assert db.get_screenshot(conn, "2" * 64)["category"] == "recipe"
    conn.close()


def test_run_leaves_already_classified_rows_alone(root: Path) -> None:
    row = add_row(root, ocr_text=LONG_TEXT)
    conn = db.connect(root)
    db.update_screenshot(conn, row.hash, {"category": "recipe"})

    assert classify.select_rows(conn, root) == []
    assert len(classify.select_rows(conn, root, force=True)) == 1
    conn.close()


def test_a_failing_row_does_not_stop_the_run(root: Path) -> None:
    add_row(root, ocr_text=LONG_TEXT, file_hash="c" * 64, original_name="doomed.png")
    add_row(root, ocr_text=LONG_TEXT, file_hash="d" * 64, original_name="fine.png")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS_PAYLOAD)
        if "doomed.png" in request.content.decode():
            return httpx.Response(400, text="bad request")
        return httpx.Response(200, json=completion_payload(reply("article"), model=TEXT_MODEL))

    client = CLIENT_CLASS(
        api_key="k", require_key=False, backoff=0.0, transport=httpx.MockTransport(handler)
    )
    conn = db.connect(root)
    rows = classify.select_rows(conn, root)
    summary = classify.run(conn, rows, method="text", settings=SETTINGS, client=client)

    assert summary.considered == 2
    assert summary.classified == 1
    assert summary.errors == 1
    conn.close()
    client.close()


# ---------------------------------------------------------------------------
# Dry run and the command
# ---------------------------------------------------------------------------


def test_estimate_splits_the_tiers_and_prices_them(root: Path) -> None:
    rules_row = add_row(
        root, file_hash="e" * 64, ocr_text="Swipe up to open. Face ID. Enter passcode."
    )
    # A confident keyword hit that is not junk still costs a model call now.
    text_row = add_row(root, file_hash="f" * 64, ocr_text=INVOICE_TEXT)
    another_text_row = add_row(root, file_hash="7" * 64, ocr_text=LONG_TEXT)
    vision_row = add_row(root, file_hash="0" * 64, ocr_text="ok")
    recorder = Recorder({})
    client = make_client(recorder)

    estimate = classify.estimate(
        [rules_row, text_row, another_text_row, vision_row],
        method="auto",
        settings=SETTINGS,
        client=client,
    )

    assert estimate.tiers[classify.METHOD_RULES] == 1
    assert estimate.tiers[classify.METHOD_TEXT] == 2
    assert estimate.tiers[classify.METHOD_VISION] == 1
    assert estimate.cost_usd > 0
    assert estimate.priced
    assert recorder.calls == 0
    client.close()


def use_test_models(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> None:
    """Point the command at the mocked transport and the fake model ids."""
    monkeypatch.setenv(settings.ENV_TEXT_MODEL, TEXT_MODEL)
    monkeypatch.setenv(settings.ENV_VISION_MODEL, VISION_MODEL)
    monkeypatch.setattr(openrouter, "OpenRouterClient", lambda **_: make_client(recorder))


def test_dry_run_reports_without_calling_a_model(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({})
    use_test_models(monkeypatch, recorder)

    result = runner.invoke(app, ["classify", "--dry-run", "--root", str(root)])

    assert result.exit_code == 0, result.output
    assert "dry run" in result.output
    assert "Estimated cost" in result.output
    assert recorder.calls == 0

    conn = db.connect(root)
    assert db.get_screenshot(conn, "a" * 64)["category"] is None
    conn.close()


def test_classify_command_writes_results(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    add_row(root, ocr_text=LONG_TEXT)
    recorder = Recorder({TEXT_MODEL: reply("article", 0.86)})
    use_test_models(monkeypatch, recorder)

    result = runner.invoke(app, ["classify", "--root", str(root)])

    assert result.exit_code == 0, result.output
    conn = db.connect(root)
    assert db.get_screenshot(conn, "a" * 64)["category"] == "article"
    conn.close()


def test_classify_command_rejects_an_unknown_method(root: Path) -> None:
    result = runner.invoke(app, ["classify", "--method", "psychic", "--root", str(root)])
    assert result.exit_code == 2
    assert "Unknown method" in result.output


def test_classify_command_says_when_there_is_nothing_to_do(root: Path) -> None:
    result = runner.invoke(app, ["classify", "--root", str(root)])
    assert result.exit_code == 0
    assert "Nothing to classify" in result.output


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_precedence(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(settings.ENV_TEXT_MODEL, raising=False)
    monkeypatch.delenv(settings.ENV_VISION_MODEL, raising=False)
    monkeypatch.setattr(settings.config, "load_env", lambda: None)

    default = settings.resolve(root)
    assert default.text_model == settings.DEFAULT_TEXT_MODEL
    assert default.concurrency == settings.DEFAULT_CONCURRENCY

    (root / settings.CONFIG_FILENAME).write_text(
        '[classify]\ntext_model = "from/file"\nconcurrency = 3\n', encoding="utf-8"
    )
    from_file = settings.resolve(root)
    assert from_file.text_model == "from/file"
    assert from_file.vision_model == settings.DEFAULT_VISION_MODEL
    assert from_file.concurrency == 3

    monkeypatch.setenv(settings.ENV_TEXT_MODEL, "from/env")
    assert settings.resolve(root).text_model == "from/env"

    override = settings.resolve(root, model="from/flag", concurrency=99)
    assert override.text_model == "from/flag"
    assert override.vision_model == "from/flag"
    assert override.concurrency == settings.MAX_CONCURRENCY


def test_rules_caption_uses_the_first_real_line_of_text(root: Path) -> None:
    ocr = (
        "  \n"
        "12:04\n"
        "Slow-cooked lamb shoulder\n"
        "Ingredients\n"
        "2 tbsp olive oil\n"
        "Preheat the oven to 160C and cook for 4 hours"
    )
    row = add_row(root, ocr_text=ocr)
    outcome = classify.classify_row(row, method="rules", settings=SETTINGS, client=None, cache={})
    assert outcome.classification.category == "recipe"
    assert outcome.classification.caption == "Slow-cooked lamb shoulder"


def test_rules_caption_trims_a_very_long_first_line(root: Path) -> None:
    row = add_row(root, ocr_text="TAX INVOICE " + "for the plumber " * 20 + "\nAmount due $10")
    outcome = classify.classify_row(row, method="rules", settings=SETTINGS, client=None, cache={})
    assert len(outcome.classification.caption) <= taxonomy.MAX_CAPTION_CHARS
    assert outcome.classification.caption.startswith("TAX INVOICE")


def test_rules_caption_falls_back_to_the_matched_keywords(root: Path) -> None:
    # Short lines that never reach caption length, but still carry enough keywords.
    row = add_row(root, ocr_text="invoice\npaid\nbalance\ncart\n$12.00")
    outcome = classify.classify_row(row, method="rules", settings=SETTINGS, client=None, cache={})
    assert outcome.classification.category == "receipt"
    assert outcome.classification.caption.startswith("Matched on")


def test_rules_caption_when_there_is_no_text(root: Path) -> None:
    row = add_row(root, ocr_text="")
    outcome = classify.classify_row(row, method="rules", settings=SETTINGS, client=None, cache={})
    assert outcome.classification.caption == "No text found."


def test_prompts_carry_the_context_the_models_need(root: Path) -> None:
    row = add_row(root, ocr_text="Invoice 10428")
    prompt = classify.text_prompt(row)
    assert "1920x1080" in prompt
    assert "Invoice 10428" in prompt
    assert row.original_name in prompt
    assert "landscape" in classify.vision_prompt(row)
    assert taxonomy.active().names[0] in classify.system_prompt()
