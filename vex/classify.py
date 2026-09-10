"""Classification: the rules tier, the two model tiers, and the run that drives them."""

from __future__ import annotations

import base64
import sqlite3
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vex import config, db, openrouter, taxonomy
from vex.openrouter import OpenRouterClient
from vex.settings import ClassifySettings
from vex.taxonomy import Classification

METHODS = ("auto", "rules", "text", "vision")

MODE_TEXT = "text"
MODE_VISION = "vision"

METHOD_RULES = "rules"
METHOD_TEXT = "text-model"
METHOD_VISION = "vision-model"

#: A rules verdict this confident is trusted, but only for junk. Keyword evidence is
#: good at spotting a lock screen and bad at telling a recipe from a post that says
#: "recipe", so every other verdict is only ever a hint to the model.
RULES_ACCEPT_CONFIDENCE = 0.8
#: A text-tier verdict below this is re-checked by the vision tier in ``auto``.
TEXT_ESCALATE_CONFIDENCE = 0.7

MAX_OCR_CHARS_TEXT = 6000
MAX_OCR_CHARS_VISION = 800

#: Output ceiling for a classification reply. A recipe with a full ingredient list and
#: method runs long, so there is deliberate headroom here.
MAX_OUTPUT_TOKENS = 1500
#: What to allow on the one retry after a reply was cut off mid-object.
TRUNCATED_RETRY_OUTPUT_TOKENS = 4000

#: Rough token estimates, only ever used for the dry-run costing.
CHARS_PER_TOKEN = 4
THUMBNAIL_IMAGE_TOKENS = 260
OUTPUT_TOKENS_ESTIMATE = 180


def system_prompt(taxonomy_in_force: taxonomy.Taxonomy | None = None) -> str:
    """The system prompt, built from whichever taxonomy is in force."""
    resolved = taxonomy_in_force or taxonomy.active()
    return f"""{resolved.context}

Give every screenshot exactly one primary category from this list:
{taxonomy.category_list_for_prompt(resolved)}

Rules:
- Pick the single best category. Use "other" only when nothing else fits, and "junk" for
  captures with no lasting value.
- tags: up to {taxonomy.MAX_TAGS} short lowercase keywords, no hashes, most specific first.
- caption: one plain sentence, at most {taxonomy.MAX_CAPTION_CHARS} characters, saying what the
  screenshot shows. No preamble, no "this screenshot shows".
- confidence: 0 to 1, your honest read. Be low when the evidence is thin.
- is_junk: true when the capture is blank, accidental, a lock screen, a loading state or an
  unusable partial capture.
{taxonomy.structured_rules_for_prompt(resolved)}
- Australian English. No emojis.

Reply with a single JSON object and nothing else:
{{"category": string, "tags": [string], "caption": string, "confidence": number,
 "is_junk": boolean, "use": string or null, "recipe": object or null}}"""


@dataclass(frozen=True, slots=True)
class RowInput:
    """The catalogue fields the classifier needs for one screenshot."""

    hash: str
    original_name: str
    source: str
    width: int | None
    height: int | None
    ocr_text: str
    thumb: Path

    @property
    def text(self) -> str:
        """OCR text, trimmed."""
        return self.ocr_text.strip()

    def has_thumbnail(self) -> bool:
        """True when a thumbnail exists on disk for the vision tier."""
        return self.thumb.is_file()


@dataclass(frozen=True, slots=True)
class CacheWrite:
    """A raw model response waiting to be written to classify_cache."""

    hash: str
    model: str
    mode: str
    response_json: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    created_at: str


@dataclass(slots=True)
class RowOutcome:
    """What happened to one row."""

    hash: str
    tier: str
    classification: Classification | None = None
    cache_writes: list[CacheWrite] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    error: str | None = None


@dataclass(slots=True)
class RunSummary:
    """Totals for a whole classify run."""

    considered: int = 0
    classified: int = 0
    errors: int = 0
    cache_hits: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    tiers: Counter[str] = field(default_factory=Counter)
    categories: Counter[str] = field(default_factory=Counter)
    problems: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class DryRunEstimate:
    """What a run would do, without doing any of it."""

    tiers: Counter[str] = field(default_factory=Counter)
    cost_usd: float = 0.0
    priced: bool = True
    text_model: str = ""
    vision_model: str = ""


# ---------------------------------------------------------------------------
# Row selection
# ---------------------------------------------------------------------------


#: A rules verdict is a hint, not an answer, so a run that can reach a model treats one
#: as unfinished work. Confident junk is the exception: the keywords settle that alone.
UNFINISHED_RULES_SQL = (
    "(classify_method = 'rules' AND NOT (is_junk = 1 AND classify_confidence >= ?))"
)


def select_rows(
    conn: sqlite3.Connection,
    root: Path,
    *,
    force: bool = False,
    limit: int | None = None,
    method: str = "auto",
    categories: Sequence[str] | None = None,
) -> list[RowInput]:
    """Rows that still need work, or every row when forced.

    A run that can reach a model also picks up every row an earlier rules-only run left
    behind, because keyword evidence is a hint rather than a verdict. Model and manual
    results are left alone, and so is anything the keywords confidently called junk.

    ``categories`` narrows whatever that selects to the categories named, which is how a
    second pass over one bucket is run: ``--category other --force``.
    """
    sql = "SELECT hash, original_name, source, width, height, ocr_text FROM screenshots"
    params: list[object] = []
    clauses: list[str] = []
    if not force:
        unfinished = "category IS NULL"
        if method != METHOD_RULES:
            unfinished += f" OR {UNFINISHED_RULES_SQL}"
            params.append(RULES_ACCEPT_CONFIDENCE)
        clauses.append(f"({unfinished})")
    if categories:
        placeholders = ", ".join("?" for _ in categories)
        clauses.append(f"category IN ({placeholders})")
        params.extend(categories)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY captured_at"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [
        RowInput(
            hash=row["hash"],
            original_name=row["original_name"] or "",
            source=row["source"] or "",
            width=row["width"],
            height=row["height"],
            ocr_text=row["ocr_text"] or "",
            thumb=config.thumb_path(root, row["hash"]),
        )
        for row in conn.execute(sql, params).fetchall()
    ]


def load_cache(
    conn: sqlite3.Connection, rows: list[RowInput], models: tuple[str, ...]
) -> dict[tuple[str, str, str], str]:
    """Pre-load classify_cache for these rows so the workers never touch SQLite."""
    if not rows or not models:
        return {}
    cache: dict[tuple[str, str, str], str] = {}
    hashes = [row.hash for row in rows]
    chunk = 400
    for start in range(0, len(hashes), chunk):
        window = hashes[start : start + chunk]
        placeholders = ", ".join("?" for _ in window)
        model_places = ", ".join("?" for _ in models)
        sql = (
            "SELECT hash, model, mode, response_json FROM classify_cache "  # noqa: S608
            f"WHERE hash IN ({placeholders}) AND model IN ({model_places})"
        )
        for row in conn.execute(sql, [*window, *models]).fetchall():
            cache[(row["hash"], row["model"], row["mode"])] = row["response_json"]
    return cache


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _dimensions(row: RowInput) -> str:
    if row.width and row.height:
        orientation = "portrait" if row.height > row.width else "landscape"
        return f"{row.width}x{row.height} ({orientation})"
    return "unknown"


def hint_line(match: taxonomy.RuleMatch | None) -> str:
    """The rules guess, offered to the model as one weak signal among others."""
    if match is None:
        return ""
    matched = ", ".join(match.matched[:4])
    evidence = f" on the words: {matched}" if matched else ""
    return (
        f'\nA keyword pass guessed "{match.category}"{evidence}. It matches single words '
        "without reading them in context and is often wrong, so treat it as one weak "
        "signal and ignore it when the screenshot says otherwise."
    )


def text_prompt(row: RowInput, hint: taxonomy.RuleMatch | None = None) -> str:
    """The user message for the text tier."""
    text = row.text[:MAX_OCR_CHARS_TEXT]
    return (
        f"Original filename: {row.original_name}\n"
        f"Source device: {row.source or 'unknown'}\n"
        f"Dimensions: {_dimensions(row)}\n\n"
        f"Text recognised in the screenshot:\n---\n{text}\n---\n"
        f"{hint_line(hint)}\n"
        "Classify the screenshot from this text."
    )


def vision_prompt(row: RowInput, hint: taxonomy.RuleMatch | None = None) -> str:
    """The user message for the vision tier."""
    lines = [
        f"Original filename: {row.original_name}",
        f"Source device: {row.source or 'unknown'}",
        f"Dimensions: {_dimensions(row)}",
    ]
    if row.text:
        excerpt = row.text[:MAX_OCR_CHARS_VISION]
        lines.append(f"\nText recognised in the screenshot (may be partial):\n{excerpt}")
    hint_text = hint_line(hint)
    if hint_text:
        lines.append(hint_text)
    lines.append("\nClassify the screenshot in the image.")
    return "\n".join(lines)


def encode_thumbnail(path: Path) -> str:
    """Base64 the thumbnail JPEG for an inline image message."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()  # noqa: DTZ005


def _from_cached(
    response_json: str, *, method: str, model: str
) -> tuple[Classification, dict[str, Any]] | None:
    """A cached reply, or None when it cannot be used and the model must be asked again."""
    payload = db.load_json(response_json)
    if not isinstance(payload, dict):
        return None
    try:
        content = openrouter.message_content(payload)
        parsed = openrouter.parse_json_object(content)
    except openrouter.OpenRouterError:
        return None
    if not taxonomy.is_current_schema(parsed):
        # Cached under the old nested shape, which never carried a reuse note or a
        # recipe. Ask again rather than serve a reply the current schema would beat.
        return None
    return taxonomy.normalise_result(parsed, method=method, model=model), payload


def _messages_for(
    row: RowInput, mode: str, hint: taxonomy.RuleMatch | None = None
) -> tuple[list[dict[str, Any]], int]:
    """The chat messages for one tier, and how many images they carry."""
    if mode == MODE_TEXT:
        return [
            openrouter.text_message("system", system_prompt()),
            openrouter.text_message("user", text_prompt(row, hint)),
        ], 0
    return [
        openrouter.text_message("system", system_prompt()),
        openrouter.image_message(vision_prompt(row, hint), encode_thumbnail(row.thumb)),
    ], 1


def _complete_with_room(
    client: OpenRouterClient,
    model: str,
    messages: list[dict[str, Any]],
    *,
    images: int,
    outcome: RowOutcome,
) -> openrouter.Completion:
    """Ask for the classification, and ask again with more room if the reply was cut off.

    A reply that stops mid-object cannot be parsed at all, so the retry is worth its cost.
    """
    completion = client.complete(
        model,
        messages,
        json_schema=taxonomy.response_format(),
        images=images,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    outcome.calls += 1
    outcome.cost_usd += completion.cost_usd
    if not completion.truncated:
        return completion

    completion = client.complete(
        model,
        messages,
        json_schema=taxonomy.response_format(),
        images=images,
        max_tokens=TRUNCATED_RETRY_OUTPUT_TOKENS,
    )
    outcome.calls += 1
    outcome.cost_usd += completion.cost_usd
    if completion.truncated:
        raise openrouter.ResponseFormatError(
            f"{model} was still cut off at {TRUNCATED_RETRY_OUTPUT_TOKENS} output tokens."
        )
    return completion


def call_tier(
    client: OpenRouterClient,
    row: RowInput,
    *,
    mode: str,
    model: str,
    cache: dict[tuple[str, str, str], str],
    outcome: RowOutcome,
    hint: taxonomy.RuleMatch | None = None,
) -> Classification:
    """Run one model tier for one row, using the cache when it already has the answer."""
    method = METHOD_TEXT if mode == MODE_TEXT else METHOD_VISION
    cached = cache.get((row.hash, model, mode))
    if cached:
        recovered = _from_cached(cached, method=method, model=model)
        if recovered:
            outcome.cache_hits += 1
            return recovered[0]

    messages, images = _messages_for(row, mode, hint)
    completion = _complete_with_room(client, model, messages, images=images, outcome=outcome)
    result = taxonomy.normalise_result(completion.json_payload(), method=method, model=model)

    # Cached only once it has parsed, so a cut-off reply never poisons the cache.
    outcome.cache_writes.append(
        CacheWrite(
            hash=row.hash,
            model=model,
            mode=mode,
            response_json=db.dump_json(completion.raw) or "{}",
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            created_at=_now(),
        )
    )
    return result


def _rules_caption(match: taxonomy.RuleMatch, text: str) -> str:
    """The first real line of OCR text if there is one, else the keywords that matched."""
    line = taxonomy.first_meaningful_line(text)
    if line:
        return line
    if match.matched:
        return taxonomy.normalise_caption(f"Matched on {', '.join(match.matched[:3])}.")
    return "No text found."


def _from_rules(match: taxonomy.RuleMatch, text: str = "") -> Classification:
    return Classification(
        category=match.category,
        tags=match.tags,
        caption=_rules_caption(match, text),
        confidence=match.confidence,
        is_junk=match.is_junk,
        structured=None,
        method=METHOD_RULES,
        model=None,
    )


def settles_without_a_model(match: taxonomy.RuleMatch | None) -> bool:
    """True when the keywords are enough on their own.

    Only a confident junk verdict qualifies. A lock screen or a "swipe up to open" is
    unambiguous; everything else the keywords find is a hint for a model to confirm.
    """
    return bool(match and match.is_junk and match.confidence >= RULES_ACCEPT_CONFIDENCE)


def plan_tier(row: RowInput, method: str) -> str:
    """Which tier this row would land in, without calling anything."""
    if method == METHOD_RULES:
        return METHOD_RULES
    if method == MODE_TEXT:
        return METHOD_TEXT
    if method == MODE_VISION:
        return METHOD_VISION
    match = taxonomy.apply_rules(row.ocr_text, row.original_name)
    if settles_without_a_model(match):
        return METHOD_RULES
    if len(row.text) >= taxonomy.THIN_TEXT_CHARS:
        # It may still escalate to vision, but the text call is the one we can count on.
        return METHOD_TEXT
    return METHOD_VISION


def _no_rule_match() -> Classification:
    """What the rules tier returns when nothing matched and there is no model to ask."""
    return Classification(
        category="other",
        tags=[],
        caption="No keyword rule matched.",
        confidence=0.3,
        is_junk=False,
        method=METHOD_RULES,
    )


def _text_tier(
    row: RowInput,
    settings: ClassifySettings,
    client: OpenRouterClient,
    cache: dict[tuple[str, str, str], str],
    outcome: RowOutcome,
    hint: taxonomy.RuleMatch | None = None,
) -> Classification | None:
    try:
        return call_tier(
            client,
            row,
            mode=MODE_TEXT,
            model=settings.text_model,
            cache=cache,
            outcome=outcome,
            hint=hint,
        )
    except (openrouter.OpenRouterError, OSError) as exc:
        outcome.error = f"text tier: {exc}"
        return None


def _vision_tier(
    row: RowInput,
    settings: ClassifySettings,
    client: OpenRouterClient,
    cache: dict[tuple[str, str, str], str],
    outcome: RowOutcome,
    hint: taxonomy.RuleMatch | None = None,
) -> Classification | None:
    if not row.has_thumbnail():
        outcome.error = "No thumbnail on disk for the vision tier."
        return None
    try:
        return call_tier(
            client,
            row,
            mode=MODE_VISION,
            model=settings.vision_model,
            cache=cache,
            outcome=outcome,
            hint=hint,
        )
    except (openrouter.OpenRouterError, OSError) as exc:
        outcome.error = f"vision tier: {exc}"
        return None


def _settle(
    outcome: RowOutcome,
    text_result: Classification | None,
    vision_result: Classification | None,
) -> RowOutcome:
    """Keep whichever tier is more sure of itself, or report that neither answered."""
    candidates = [
        (tier, result)
        for tier, result in ((METHOD_VISION, vision_result), (METHOD_TEXT, text_result))
        if result is not None
    ]
    if not candidates:
        outcome.tier = "error"
        outcome.error = outcome.error or "No tier produced a classification."
        return outcome
    outcome.tier, outcome.classification = max(candidates, key=lambda item: item[1].confidence)
    outcome.error = None
    return outcome


def classify_row(
    row: RowInput,
    *,
    method: str,
    settings: ClassifySettings,
    client: OpenRouterClient | None,
    cache: dict[tuple[str, str, str], str],
) -> RowOutcome:
    """Take one row through the tiers, cheapest first, and return the verdict."""
    outcome = RowOutcome(hash=row.hash, tier=METHOD_RULES)
    match = taxonomy.apply_rules(row.ocr_text, row.original_name)

    if method == METHOD_RULES:
        outcome.classification = _from_rules(match, row.ocr_text) if match else _no_rule_match()
        return outcome

    if method == "auto" and settles_without_a_model(match):
        outcome.classification = _from_rules(match, row.ocr_text)
        return outcome

    if client is None:
        outcome.tier = "error"
        outcome.error = "No OpenRouter client available."
        return outcome

    text_result: Classification | None = None
    if method == MODE_TEXT or (method == "auto" and len(row.text) >= taxonomy.THIN_TEXT_CHARS):
        text_result = _text_tier(row, settings, client, cache, outcome, match)
        settled = text_result is not None and text_result.confidence >= TEXT_ESCALATE_CONFIDENCE
        if method == MODE_TEXT or settled:
            return _settle(outcome, text_result, None)

    vision_result = _vision_tier(row, settings, client, cache, outcome, match)
    return _settle(outcome, text_result, vision_result)


# ---------------------------------------------------------------------------
# Writing results
# ---------------------------------------------------------------------------


def write_outcome(conn: sqlite3.Connection, outcome: RowOutcome) -> None:
    """Persist the cache entries and the catalogue columns for one row."""
    for entry in outcome.cache_writes:
        db.cache_classification(
            conn,
            entry.hash,
            entry.model,
            entry.mode,
            entry.response_json,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cost_usd=entry.cost_usd,
            created_at=entry.created_at,
        )
    result = outcome.classification
    if result is None:
        return
    db.update_screenshot(
        conn,
        outcome.hash,
        {
            "category": result.category,
            "tags": db.dump_json(result.tags),
            "caption": result.caption,
            "classify_confidence": result.confidence,
            "classify_method": result.method,
            "classify_model": result.model,
            "classified_at": _now(),
            "structured": db.dump_json(result.structured),
            "is_junk": int(result.is_junk),
        },
    )


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """A rough token count, good enough for a cost estimate."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def estimate(
    rows: list[RowInput],
    *,
    method: str,
    settings: ClassifySettings,
    client: OpenRouterClient | None,
) -> DryRunEstimate:
    """Count the tier split and price it, without calling any model."""
    result = DryRunEstimate(text_model=settings.text_model, vision_model=settings.vision_model)
    overhead = estimate_tokens(system_prompt()) + 60

    text_pricing = client.pricing_for(settings.text_model) if client else None
    vision_pricing = client.pricing_for(settings.vision_model) if client else None
    result.priced = bool(text_pricing and text_pricing.prompt) or bool(
        vision_pricing and vision_pricing.prompt
    )

    for row in rows:
        tier = plan_tier(row, method)
        result.tiers[tier] += 1
        if tier == METHOD_TEXT and text_pricing:
            tokens = overhead + estimate_tokens(row.text[:MAX_OCR_CHARS_TEXT])
            result.cost_usd += text_pricing.cost(tokens, OUTPUT_TOKENS_ESTIMATE)
        elif tier == METHOD_VISION and vision_pricing:
            tokens = overhead + THUMBNAIL_IMAGE_TOKENS
            result.cost_usd += vision_pricing.cost(tokens, OUTPUT_TOKENS_ESTIMATE, images=1)
    return result


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(
    conn: sqlite3.Connection,
    rows: list[RowInput],
    *,
    method: str,
    settings: ClassifySettings,
    client: OpenRouterClient | None,
    progress: Any = None,
) -> RunSummary:
    """Classify every row, writing results on the calling thread as they land."""
    summary = RunSummary(considered=len(rows))
    if not rows:
        return summary

    models = tuple({settings.text_model, settings.vision_model})
    cache = load_cache(conn, rows, models)

    def work(row: RowInput) -> RowOutcome:
        try:
            return classify_row(row, method=method, settings=settings, client=client, cache=cache)
        except Exception as exc:  # noqa: BLE001 - one bad row must not stop the run
            return RowOutcome(hash=row.hash, tier="error", error=str(exc))

    workers = max(1, min(settings.concurrency, len(rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, outcome in enumerate(pool.map(work, rows), start=1):
            summary.calls += outcome.calls
            summary.cache_hits += outcome.cache_hits
            summary.cost_usd += outcome.cost_usd
            summary.tiers[outcome.tier] += 1
            if outcome.classification is not None:
                summary.classified += 1
                summary.categories[outcome.classification.category] += 1
            if outcome.error:
                summary.errors += 1
                summary.problems.append((outcome.hash, outcome.error))
            write_outcome(conn, outcome)
            if progress is not None:
                progress(index, len(rows), outcome)
    return summary
