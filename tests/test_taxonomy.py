"""The taxonomy loader, the rules tier and result normalisation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from vex import config, taxonomy

RECIPE_TEXT = """
Slow-cooked lamb shoulder
Ingredients
2 tbsp olive oil
4 cloves garlic, finely chopped
Method
Preheat the oven to 160C. Cook for 4 hours in a covered saucepan.
Serves 6
"""

RECEIPT_TEXT = """
TAX INVOICE
Invoice 10428
Subtotal $1,200.00
Amount due $1,320.00
Due date 14 August 2026
"""

CODE_TEXT = """
Traceback (most recent call last):
  File "app.py", line 42, in main
    raise ValueError("bad state")
SyntaxError: invalid syntax
$ npm install --save-dev ruff
"""

SOCIAL_POST_TEXT = """
LinkedIn
Ada Lovelace - 1st
See more
128 reactions - 14 comments - 3 reposts
2,410 impressions
"""

SCHOOL_TEXT = """
Riverbank Primary School newsletter
Term 3 assembly is on Friday.
Please return the permission form to the teacher.
Tuckshop orders close Thursday.
"""

#: A small taxonomy of the kind a library owner would actually write.
CUSTOM_TOML = textwrap.dedent(
    """
    context = "You file screenshots for a primary school office."

    [[category]]
    name = "school"
    description = "Newsletters, portals, permission forms and notes from teachers."
    structured = "use-note"
    purgeable = true
    strong = ["school newsletter", "permission form", "tuckshop", "parent portal"]
    weak = ["assembly", "teacher", "term 3", "uniform"]
    tags = ["school"]

    [[category]]
    name = "junk"
    description = "Blank captures, lock screens, loading states."
    strong = ["enter passcode"]
    tags = ["junk"]

    [[category]]
    name = "other"
    description = "Worth keeping, fits nothing else."
    """
)


@pytest.fixture(autouse=True)
def _default_taxonomy() -> None:
    """Every test starts from the built-in taxonomy, whatever the last one loaded."""
    taxonomy.reset()


# ---------------------------------------------------------------------------
# The rules tier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (RECIPE_TEXT, "recipe"),
        (RECEIPT_TEXT, "receipt"),
        (CODE_TEXT, "code"),
        (SOCIAL_POST_TEXT, "social-post"),
    ],
)
def test_rules_classify_known_shapes(text: str, expected: str) -> None:
    match = taxonomy.apply_rules(text, "Screenshot 2026-07-20 at 14.03.50.png")
    assert match is not None
    assert match.category == expected
    assert match.confidence >= 0.8
    assert match.tags


def test_rules_return_junk_for_blank_text() -> None:
    match = taxonomy.apply_rules("   ", "IMG_8425.PNG")
    assert match is not None
    assert match.category == "junk"
    assert match.is_junk
    # Not confident enough to stop there: auto should still send it to the vision tier.
    assert match.confidence < 0.8


def test_rules_decline_when_evidence_is_thin() -> None:
    assert (
        taxonomy.apply_rules("A photograph of the back garden on a sunny afternoon", "a.png")
        is None
    )


def test_rules_decline_when_two_categories_tie() -> None:
    # Three weak hits each for shopping and map, so neither wins by the required margin.
    text = "order the delivery into the cart, then the hotel route and flight"
    assert taxonomy.apply_rules(text, "x.png") is None


def test_lock_screen_text_is_confidently_junk() -> None:
    match = taxonomy.apply_rules("Swipe up to open. Face ID. Enter passcode.", "x.png")
    assert match is not None
    assert match.category == "junk"
    assert match.confidence >= 0.8


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------


def test_normalise_result_caps_tags_and_caption() -> None:
    result = taxonomy.normalise_result(
        {
            "category": "Recipe",
            "tags": ["#Lamb", "lamb", "Slow Cook", "a", "b", "c", "d", "e"],
            "caption": "x" * 300,
            "confidence": 1.4,
            "is_junk": False,
            "structured": {"title": "Lamb", "ingredients": ["oil"], "method": ["cook"]},
        },
        method="text-model",
        model="test/model",
    )
    assert result.category == "recipe"
    assert len(result.tags) == taxonomy.MAX_TAGS
    assert result.tags[0] == "lamb"
    assert len(result.caption) <= taxonomy.MAX_CAPTION_CHARS
    assert result.confidence == 1.0
    assert result.structured["serves"] is None


def test_normalise_result_falls_back_to_other() -> None:
    result = taxonomy.normalise_result(
        {"category": "spreadsheets", "confidence": "nonsense"},
        method="vision-model",
        model="test/model",
    )
    assert result.category == "other"
    assert result.confidence == 0.5
    assert result.tags == []
    assert result.structured is None


def test_structured_is_dropped_for_categories_without_a_shape() -> None:
    result = taxonomy.normalise_result(
        {"category": "shopping", "structured": {"use": "buy it"}},
        method="text-model",
        model="test/model",
    )
    assert result.structured is None


def test_use_note_is_kept_for_a_use_note_category() -> None:
    result = taxonomy.normalise_result(
        {"category": "article", "structured": {"use": "  A post   angle "}},
        method="text-model",
        model="test/model",
    )
    assert result.structured == {"use": "A post angle"}


def test_is_junk_flag_moves_other_to_junk() -> None:
    result = taxonomy.normalise_result(
        {"category": "other", "is_junk": True},
        method="vision-model",
        model="test/model",
    )
    assert result.category == "junk"
    assert result.is_junk


def test_prompt_helpers_cover_every_category() -> None:
    rendered = taxonomy.category_list_for_prompt()
    for category in taxonomy.active().names:
        assert f"- {category}:" in rendered
    assert "recipe:" in taxonomy.structured_rules_for_prompt()


def test_response_schema_offers_exactly_the_taxonomy() -> None:
    schema = taxonomy.response_schema()
    assert schema["properties"]["category"]["enum"] == list(taxonomy.active().names)


# ---------------------------------------------------------------------------
# Loading a taxonomy of your own
# ---------------------------------------------------------------------------


def test_default_taxonomy_has_the_required_categories() -> None:
    assert "junk" in taxonomy.DEFAULT.by_name
    assert "other" in taxonomy.DEFAULT.by_name
    assert taxonomy.DEFAULT.recipe_categories == {"recipe"}
    assert taxonomy.DEFAULT.use_note_categories == {"article", "social-post"}


def test_a_root_without_a_taxonomy_file_gets_the_default(tmp_path: Path) -> None:
    assert taxonomy.for_root(tmp_path) is taxonomy.DEFAULT


def test_a_root_taxonomy_replaces_the_default(tmp_path: Path) -> None:
    (tmp_path / taxonomy.TAXONOMY_FILENAME).write_text(CUSTOM_TOML, encoding="utf-8")
    loaded = taxonomy.for_root(tmp_path)
    assert loaded.names == ("school", "junk", "other")
    assert loaded.use_note_categories == {"school"}
    assert loaded.purgeable_categories == {"school"}
    assert loaded.context.startswith("You file screenshots for a primary school office")


def test_get_root_loads_the_taxonomy_beside_the_catalogue(tmp_path: Path) -> None:
    (tmp_path / taxonomy.TAXONOMY_FILENAME).write_text(CUSTOM_TOML, encoding="utf-8")
    config.get_root(tmp_path)
    match = taxonomy.apply_rules(SCHOOL_TEXT, "IMG_0001.PNG")
    assert match is not None
    assert match.category == "school"


def test_categories_the_taxonomy_does_not_define_fall_back_to_other(tmp_path: Path) -> None:
    (tmp_path / taxonomy.TAXONOMY_FILENAME).write_text(CUSTOM_TOML, encoding="utf-8")
    config.get_root(tmp_path)
    result = taxonomy.normalise_result(
        {"category": "recipe"}, method="text-model", model="test/model"
    )
    assert result.category == "other"


@pytest.mark.parametrize(
    ("toml", "expected"),
    [
        ("", "needs an array of [[category]] tables"),
        ("category = []", "needs at least one [[category]] table"),
        ('[[category]]\nname = ""\n', "needs a non-empty name"),
        ('[[category]]\nname = "Junk Drawer"\n', "must be lowercase letters"),
        ('[[category]]\nname = "junk"\n', "needs a non-empty description"),
        ('[[category]]\nname = "junk"\ndescription = "x"\nnonsense = 1\n', "unknown keys nonsense"),
        (
            '[[category]]\nname = "junk"\ndescription = "x"\nstructured = "poem"\n',
            "Choose from use-note, recipe",
        ),
        (
            '[[category]]\nname = "junk"\ndescription = "x"\npurgeable = "yes"\n',
            "non-boolean purgeable",
        ),
        (
            '[[category]]\nname = "junk"\ndescription = "x"\nstrong = "invoice"\n',
            "must be a list of strings",
        ),
        ('[[category]]\nname = "junk"\ndescription = "x"\n', "missing the required category other"),
        (
            (
                '[[category]]\nname = "junk"\ndescription = "x"\n'
                '[[category]]\nname = "junk"\ndescription = "y"\n'
            ),
            "is defined twice",
        ),
        (
            (
                'context = ""\n[[category]]\nname = "junk"\ndescription = "x"\n'
                '[[category]]\nname = "other"\ndescription = "y"\n'
            ),
            "context must be a non-empty string",
        ),
    ],
)
def test_a_broken_taxonomy_says_what_is_wrong(tmp_path: Path, toml: str, expected: str) -> None:
    path = tmp_path / taxonomy.TAXONOMY_FILENAME
    path.write_text(toml, encoding="utf-8")
    with pytest.raises(taxonomy.TaxonomyError, match=expected.replace("[", r"\[")):
        taxonomy.for_root(tmp_path)


def test_invalid_toml_is_reported_as_such(tmp_path: Path) -> None:
    (tmp_path / taxonomy.TAXONOMY_FILENAME).write_text("[[category\n", encoding="utf-8")
    with pytest.raises(taxonomy.TaxonomyError, match="is not valid TOML"):
        taxonomy.for_root(tmp_path)
