"""The classification taxonomy, the rules tier and the JSON contract for the models.

The taxonomy is data rather than code. Categories, their descriptions, the keyword
sets the rules tier scores against, the structured shape each category carries and
whether its originals may be purged all come from ``taxonomy.toml`` in the library
root. Without that file a generic built-in default applies, so the tool works out of
the box and can then be bent to whatever a particular library is full of.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any

TAXONOMY_FILENAME = "taxonomy.toml"

#: The file shipped with the package, used when a root has no taxonomy of its own.
DEFAULT_TAXONOMY_RESOURCE = "default_taxonomy.toml"

MAX_TAGS = 6
MAX_CAPTION_CHARS = 120

# ---------------------------------------------------------------------------
# Structured shapes
# ---------------------------------------------------------------------------

#: A one-sentence note on how the owner could reuse what is on screen.
SHAPE_USE_NOTE = "use-note"
#: A transcribed recipe: title, ingredients, method, serves, source.
SHAPE_RECIPE = "recipe"

SHAPES: tuple[str, ...] = (SHAPE_USE_NOTE, SHAPE_RECIPE)

RECIPE_STRUCTURED_KEYS = ("title", "ingredients", "method", "serves", "source")

STRUCTURED_SHAPES: dict[str, str] = {
    SHAPE_RECIPE: (
        '{"title": string, "ingredients": [string], "method": [string], '
        '"serves": string or null, "source": string or null}'
    ),
    SHAPE_USE_NOTE: '{"use": string} - one sentence on how this could be reused, or null',
}


# ---------------------------------------------------------------------------
# The taxonomy itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rule:
    """Keyword evidence for one category.

    ``strong`` phrases are close to decisive on their own; ``weak`` ones only count
    together. Everything is matched case-insensitively against the OCR text plus the
    original filename.
    """

    strong: tuple[str, ...] = ()
    weak: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default=())

    def is_empty(self) -> bool:
        """True when this category offers the rules tier nothing to match on."""
        return not (self.strong or self.weak)


@dataclass(frozen=True, slots=True)
class Category:
    """One category: what it means, what it carries, and whether it can be purged."""

    name: str
    description: str
    structured: str | None = None
    purgeable: bool = False
    rule: Rule = Rule()


#: The opening line of the system prompt when a taxonomy does not set its own.
DEFAULT_CONTEXT = "You file screenshots for the person who took them."


@dataclass(frozen=True)
class Taxonomy:
    """An ordered set of categories, with the views the rest of the tool asks for."""

    categories: tuple[Category, ...]
    #: One sentence telling the model whose library this is. It opens the system prompt.
    context: str = DEFAULT_CONTEXT

    @cached_property
    def by_name(self) -> dict[str, Category]:
        """Every category, keyed by name."""
        return {category.name: category for category in self.categories}

    @cached_property
    def names(self) -> tuple[str, ...]:
        """Category names, in the order they are offered to the model."""
        return tuple(category.name for category in self.categories)

    @cached_property
    def descriptions(self) -> dict[str, str]:
        """Category descriptions, keyed by name, in order."""
        return {category.name: category.description for category in self.categories}

    @cached_property
    def rules(self) -> dict[str, Rule]:
        """The keyword rules, for the categories that have any."""
        return {c.name: c.rule for c in self.categories if not c.rule.is_empty()}

    @cached_property
    def use_note_categories(self) -> frozenset[str]:
        """Categories whose structured payload is a reuse note."""
        return frozenset(c.name for c in self.categories if c.structured == SHAPE_USE_NOTE)

    @cached_property
    def recipe_categories(self) -> frozenset[str]:
        """Categories whose structured payload is a transcribed recipe."""
        return frozenset(c.name for c in self.categories if c.structured == SHAPE_RECIPE)

    @cached_property
    def purgeable_categories(self) -> frozenset[str]:
        """Categories whose originals may be deleted once the note and thumbnail exist."""
        return frozenset(c.name for c in self.categories if c.purgeable)

    def shape(self, category: str) -> str | None:
        """The structured shape for a category, or None when it carries nothing."""
        found = self.by_name.get(category)
        return found.structured if found else None


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


class TaxonomyError(ValueError):
    """Raised when a taxonomy file cannot be read or does not make sense."""


#: Category names become directory names under notes/ and views/, so they are slugs.
_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

#: Every taxonomy needs somewhere to put a worthless capture and somewhere to put
#: one that fits nothing else. Both are named in the prompt and in the fallbacks.
REQUIRED_CATEGORIES = ("junk", "other")

_CATEGORY_KEYS = frozenset(
    {"name", "description", "structured", "purgeable", "strong", "weak", "tags"}
)


def _fail(source: str, problem: str) -> TaxonomyError:
    return TaxonomyError(f"{source}: {problem}")


def _string_tuple(value: Any, *, source: str, where: str, strip: bool = False) -> tuple[str, ...]:
    """A list of TOML strings, lowercased, with the blanks dropped.

    Keyword phrases keep their surrounding whitespace: ``"def "`` with the trailing
    space is what stops it matching "default", so stripping it would quietly widen
    the rule. Tags are stripped, since a tag with an edge space is only ever a typo.
    """
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _fail(source, f"{where} must be a list of strings")
    items = (item.strip() if strip else item for item in value if item.strip())
    return tuple(item.lower() for item in items)


def _parse_category(raw: Any, *, source: str, index: int) -> Category:
    where = f"category {index}"
    if not isinstance(raw, Mapping):
        raise _fail(source, f"{where} must be a table")

    unknown = sorted(set(raw) - _CATEGORY_KEYS)
    if unknown:
        allowed = ", ".join(sorted(_CATEGORY_KEYS))
        raise _fail(source, f"{where} has unknown keys {', '.join(unknown)}. Allowed: {allowed}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _fail(source, f"{where} needs a non-empty name")
    name = name.strip()
    if not _NAME_PATTERN.match(name):
        raise _fail(
            source,
            f"category name '{name}' must be lowercase letters, digits and single hyphens",
        )

    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        raise _fail(source, f"category '{name}' needs a non-empty description")

    structured = raw.get("structured")
    if structured is not None and structured not in SHAPES:
        raise _fail(
            source,
            f"category '{name}' has structured '{structured}'. Choose from {', '.join(SHAPES)}",
        )

    purgeable = raw.get("purgeable", False)
    if not isinstance(purgeable, bool):
        raise _fail(source, f"category '{name}' has a non-boolean purgeable")

    return Category(
        name=name,
        description=re.sub(r"\s+", " ", description).strip(),
        structured=structured,
        purgeable=purgeable,
        rule=Rule(
            strong=_string_tuple(raw.get("strong"), source=source, where=f"'{name}' strong"),
            weak=_string_tuple(raw.get("weak"), source=source, where=f"'{name}' weak"),
            tags=_string_tuple(raw.get("tags"), source=source, where=f"'{name}' tags", strip=True),
        ),
    )


def parse(data: Mapping[str, Any], *, source: str = TAXONOMY_FILENAME) -> Taxonomy:
    """Build a Taxonomy from parsed TOML, with a readable error when it is wrong."""
    raw_categories = data.get("category")
    if not isinstance(raw_categories, Sequence) or isinstance(raw_categories, str | bytes):
        raise _fail(source, "needs an array of [[category]] tables")
    if not raw_categories:
        raise _fail(source, "needs at least one [[category]] table")

    categories: list[Category] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_categories, start=1):
        category = _parse_category(raw, source=source, index=index)
        if category.name in seen:
            raise _fail(source, f"category '{category.name}' is defined twice")
        seen.add(category.name)
        categories.append(category)

    context = data.get("context", DEFAULT_CONTEXT)
    if not isinstance(context, str) or not context.strip():
        raise _fail(source, "context must be a non-empty string when it is set")

    missing = [name for name in REQUIRED_CATEGORIES if name not in seen]
    if missing:
        noun = "category" if len(missing) == 1 else "categories"
        raise _fail(source, f"missing the required {noun} {', '.join(missing)}")

    return Taxonomy(categories=tuple(categories), context=re.sub(r"\s+", " ", context).strip())


def load_file(path: Path) -> Taxonomy:
    """Read and validate one taxonomy file."""
    try:
        with Path(path).open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise _fail(str(path), f"cannot be read: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise _fail(str(path), f"is not valid TOML: {exc}") from exc
    return parse(data, source=str(path))


def _load_default() -> Taxonomy:
    text = resources.files("vex").joinpath(DEFAULT_TAXONOMY_RESOURCE).read_text("utf-8")
    return parse(tomllib.loads(text), source="the built-in taxonomy")


DEFAULT = _load_default()


def path_for_root(root: Path | str) -> Path:
    """Where a library keeps its taxonomy."""
    return Path(root) / TAXONOMY_FILENAME


def for_root(root: Path | str) -> Taxonomy:
    """The taxonomy for a library root: its own file when there is one, else the default."""
    path = path_for_root(root)
    return load_file(path) if path.is_file() else DEFAULT


_active: Taxonomy = DEFAULT


def active() -> Taxonomy:
    """The taxonomy currently in force."""
    return _active


def use(taxonomy: Taxonomy) -> Taxonomy:
    """Make a taxonomy the active one. Returns it, for convenience."""
    global _active  # noqa: PLW0603
    _active = taxonomy
    return _active


def load(root: Path | str) -> Taxonomy:
    """Load the taxonomy for a root and make it the active one."""
    return use(for_root(root))


def reset() -> None:
    """Go back to the built-in default."""
    use(DEFAULT)


def _resolve(taxonomy: Taxonomy | None) -> Taxonomy:
    return active() if taxonomy is None else taxonomy


# ---------------------------------------------------------------------------
# The JSON contract handed to the models
# ---------------------------------------------------------------------------


RECIPE_SCHEMA: dict[str, Any] = {
    "type": ["object", "null"],
    "description": (
        "Only for the recipe category, null otherwise. Transcribe what the screenshot "
        "actually shows; do not invent quantities or steps that are not there."
    ),
    "properties": {
        "title": {"type": ["string", "null"], "description": "The name of the dish."},
        "ingredients": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per ingredient, quantity included.",
        },
        "method": {
            "type": "array",
            "items": {"type": "string"},
            "description": "One entry per step, in order.",
        },
        "serves": {"type": ["string", "null"]},
        "source": {"type": ["string", "null"], "description": "Where the recipe came from."},
    },
    "required": ["title", "ingredients", "method", "serves", "source"],
    "additionalProperties": False,
}


def use_description(taxonomy: Taxonomy | None = None) -> str:
    """The schema description for the ``use`` field, naming the categories that need it."""
    names = sorted(_resolve(taxonomy).use_note_categories)
    if not names:
        return "Not used by this taxonomy. Always null."
    return (
        "REQUIRED for the "
        + ", ".join(names)
        + " categories: one full sentence on how the owner of the library could reuse "
        "this, as a concrete "
        "angle rather than a summary. For example 'Use as the opening example for a post on "
        "why integration projects stall' or 'Keep as a prompt pattern for scoping calls'. "
        "Null for every other category."
    )


def response_schema(taxonomy: Taxonomy | None = None) -> dict[str, Any]:
    """The JSON schema a classification reply has to satisfy."""
    resolved = _resolve(taxonomy)
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": list(resolved.names)},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_TAGS,
            },
            "caption": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "is_junk": {"type": "boolean"},
            "use": {"type": ["string", "null"], "description": use_description(resolved)},
            "recipe": RECIPE_SCHEMA,
        },
        "required": ["category", "tags", "caption", "confidence", "is_junk", "use", "recipe"],
        "additionalProperties": False,
    }


def response_format(taxonomy: Taxonomy | None = None) -> dict[str, Any]:
    """The ``response_format`` block for a structured-output request."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "screenshot_classification",
            "strict": True,
            "schema": response_schema(taxonomy),
        },
    }


#: Keys that only exist under the current schema. A cached reply without either of them
#: predates the flattened shape and cannot carry a reuse note or a recipe.
CURRENT_SCHEMA_KEYS = ("use", "recipe")


def is_current_schema(payload: dict[str, Any]) -> bool:
    """True when a parsed reply was produced under the current response schema."""
    return any(key in payload for key in CURRENT_SCHEMA_KEYS)


def category_list_for_prompt(taxonomy: Taxonomy | None = None) -> str:
    """The taxonomy rendered as a bulleted list for a system prompt."""
    resolved = _resolve(taxonomy)
    return "\n".join(f"- {name}: {text}" for name, text in resolved.descriptions.items())


def structured_rules_for_prompt(taxonomy: Taxonomy | None = None) -> str:
    """How to fill in the ``use`` and ``recipe`` fields, for a system prompt."""
    resolved = _resolve(taxonomy)
    lines: list[str] = []
    use_note = ", ".join(sorted(resolved.use_note_categories))
    if use_note:
        lines.append(
            f"- use: a full sentence, REQUIRED whenever the category is {use_note}. Say how this\n"
            "  could be reused: a post angle, a product idea, a prompt worth keeping. Write a\n"
            "  concrete angle, not a summary of what is on screen. Never leave it null for those\n"
            '  categories, and never write "N/A". Null for every other category.'
        )
    else:
        lines.append("- use: always null; no category in this taxonomy carries a reuse note.")

    recipes = ", ".join(sorted(resolved.recipe_categories))
    if recipes:
        lines.append(
            f"- recipe: {STRUCTURED_SHAPES[SHAPE_RECIPE]}\n"
            f"  Filled in only when the category is {recipes}, null otherwise. Transcribe what is\n"
            "  on screen rather than inventing steps."
        )
    else:
        lines.append("- recipe: always null; no category in this taxonomy transcribes a recipe.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Classification:
    """A normalised classification, whichever tier produced it."""

    category: str
    tags: list[str]
    caption: str
    confidence: float
    is_junk: bool
    structured: dict[str, Any] | None = None
    method: str = "rules"
    model: str | None = None


def _clean_tag(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    tag = re.sub(r"\s+", " ", value).strip().strip("#").lower()
    return tag[:40] or None


def normalise_tags(values: Any) -> list[str]:
    """Coerce whatever the model returned into at most six clean, unique tags."""
    if isinstance(values, str):
        values = re.split(r"[,;]", values)
    if not isinstance(values, list):
        return []
    seen: list[str] = []
    for raw in values:
        tag = _clean_tag(raw)
        if tag and tag not in seen:
            seen.append(tag)
        if len(seen) == MAX_TAGS:
            break
    return seen


def normalise_caption(value: Any) -> str:
    """One line, no longer than the caption limit."""
    if not isinstance(value, str):
        return ""
    caption = re.sub(r"\s+", " ", value).strip()
    if len(caption) <= MAX_CAPTION_CHARS:
        return caption
    return caption[: MAX_CAPTION_CHARS - 1].rstrip() + "…"


#: An OCR line needs this much substance before it will do as a caption.
MIN_CAPTION_LINE_CHARS = 10
MIN_CAPTION_LINE_LETTERS = 4

_LETTERS = re.compile(r"[^\W\d_]")


def first_meaningful_line(text: str | None) -> str:
    """The first OCR line with enough substance to caption the screenshot.

    Screenshots usually lead with their own title, so the top line is the closest
    thing to a free caption the rules tier can offer.
    """
    for raw in (text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if (
            len(line) >= MIN_CAPTION_LINE_CHARS
            and len(_LETTERS.findall(line)) >= MIN_CAPTION_LINE_LETTERS
        ):
            return normalise_caption(line)
    return ""


def normalise_confidence(value: Any) -> float:
    """Clamp to 0-1, defaulting to 0.5 when the model sent something unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(number):
        return 0.5
    return max(0.0, min(1.0, number))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [line.strip(" -*\t") for line in value.splitlines()]
        return [part for part in parts if part]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def structured_from_payload(
    category: str, payload: dict[str, Any], taxonomy: Taxonomy | None = None
) -> dict[str, Any] | None:
    """Assemble the stored ``structured`` value from a model reply.

    Models are asked for ``use`` and ``recipe`` as top-level fields, because a schema
    field with a description is answered far more reliably than a free-form nested
    object. Replies cached under the older nested shape are still read.
    """
    resolved = _resolve(taxonomy)
    legacy = payload.get("structured")
    shape = resolved.shape(category)
    if shape == SHAPE_RECIPE:
        recipe = payload.get("recipe")
        if not isinstance(recipe, dict):
            recipe = legacy
        return normalise_structured(category, recipe, resolved)
    if shape == SHAPE_USE_NOTE:
        use_note = payload.get("use")
        if not isinstance(use_note, str) and isinstance(legacy, dict):
            use_note = legacy.get("use")
        return normalise_structured(category, {"use": use_note}, resolved)
    return None


def normalise_structured(
    category: str, value: Any, taxonomy: Taxonomy | None = None
) -> dict[str, Any] | None:
    """Keep only the structured shape the taxonomy defines for this category."""
    if not isinstance(value, dict):
        return None
    shape = _resolve(taxonomy).shape(category)
    if shape == SHAPE_RECIPE:
        title = value.get("title")
        ingredients = _string_list(value.get("ingredients"))
        method = _string_list(value.get("method"))
        if not (title or ingredients or method):
            return None
        return {
            "title": str(title).strip() if isinstance(title, str) else None,
            "ingredients": ingredients,
            "method": method,
            "serves": str(value["serves"]).strip()
            if isinstance(value.get("serves"), str)
            else None,
            "source": str(value["source"]).strip()
            if isinstance(value.get("source"), str)
            else None,
        }
    if shape == SHAPE_USE_NOTE:
        use_note = value.get("use")
        if isinstance(use_note, str) and use_note.strip():
            return {"use": re.sub(r"\s+", " ", use_note).strip()}
        return None
    return None


def normalise_result(
    payload: dict[str, Any],
    *,
    method: str,
    model: str | None,
    taxonomy: Taxonomy | None = None,
) -> Classification:
    """Turn a raw model payload into a Classification the catalogue can store."""
    resolved = _resolve(taxonomy)
    raw_category = payload.get("category")
    category = raw_category.strip().lower() if isinstance(raw_category, str) else ""
    if category not in resolved.by_name:
        category = "other"
    is_junk = bool(payload.get("is_junk")) or category == "junk"
    if is_junk and category == "other":
        category = "junk"
    return Classification(
        category=category,
        tags=normalise_tags(payload.get("tags")),
        caption=normalise_caption(payload.get("caption")),
        confidence=normalise_confidence(payload.get("confidence")),
        is_junk=is_junk,
        structured=structured_from_payload(category, payload, resolved),
        method=method,
        model=model,
    )


# ---------------------------------------------------------------------------
# The rules tier
# ---------------------------------------------------------------------------


STRONG_WEIGHT = 3
WEAK_WEIGHT = 1

#: A rules verdict needs this much evidence before it is offered at all.
MIN_SCORE = 3
#: ... and this much daylight over the runner-up.
MIN_MARGIN = 2
#: Score at which a rules verdict is treated as decisive.
HIGH_SCORE = 6

#: OCR text shorter than this is treated as no usable text at all.
BLANK_TEXT_CHARS = 5
#: Below this the text tier has too little to work with, so vision takes over.
THIN_TEXT_CHARS = 40


@dataclass(frozen=True, slots=True)
class RuleMatch:
    """What the rules tier concluded, and the evidence behind it."""

    category: str
    confidence: float
    tags: list[str]
    matched: list[str]
    is_junk: bool


_WORDY = re.compile(r"[a-z0-9]")


def _hits(haystack: str, phrases: tuple[str, ...]) -> list[str]:
    return [phrase for phrase in phrases if phrase in haystack]


def score_categories(
    text: str, original_name: str = "", taxonomy: Taxonomy | None = None
) -> dict[str, tuple[int, list[str]]]:
    """Score every category against the OCR text and filename."""
    haystack = f"{text}\n{original_name}".lower()
    scores: dict[str, tuple[int, list[str]]] = {}
    for category, rule in _resolve(taxonomy).rules.items():
        strong = _hits(haystack, rule.strong)
        weak = _hits(haystack, rule.weak)
        score = STRONG_WEIGHT * len(strong) + WEAK_WEIGHT * len(weak)
        if score:
            scores[category] = (score, [*strong, *weak])
    return scores


def apply_rules(
    text: str | None, original_name: str = "", taxonomy: Taxonomy | None = None
) -> RuleMatch | None:
    """Classify from keywords alone, or return None when the evidence is thin.

    Deliberately conservative: it is better to fall through to a model than to
    file a screenshot under the wrong category for free.
    """
    resolved = _resolve(taxonomy)
    cleaned = (text or "").strip()
    if len(_WORDY.findall(cleaned.lower())) < BLANK_TEXT_CHARS:
        # Nothing readable. Probably junk, but not confidently enough to stop here.
        return RuleMatch(
            category="junk",
            confidence=0.5,
            tags=["junk", "no-text"],
            matched=[],
            is_junk=True,
        )

    scores = score_categories(cleaned, original_name, resolved)
    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
    category, (score, matched) = ranked[0]
    runner_up = ranked[1][1][0] if len(ranked) > 1 else 0
    if score < MIN_SCORE or score - runner_up < MIN_MARGIN:
        return None

    confidence = 0.9 if score >= HIGH_SCORE else 0.8
    tags = normalise_tags([*resolved.rules[category].tags, *matched])
    return RuleMatch(
        category=category,
        confidence=confidence,
        tags=tags,
        matched=matched,
        is_junk=category == "junk",
    )
