"""``_normalise_stringified_json`` re-parses stringified JSON in non-string slots.

The motivating case is Gemini soft-enforcement on nested object types: a
``list[Item]`` slot occasionally receives ``["{...}", "{...}"]`` instead
of ``[{...}, {...}]``. Without normalisation, Pydantic rejects the
strings and stitcher pays a patch-turn round-trip. With normalisation,
the strings are re-parsed in place and validation passes on the first
attempt.

Safety property: only fires when the slot's annotation is a structural
type and the parsed shape matches. Slots typed ``str`` (or
``Union[str, ...]``) are left alone even if the string content looks
like JSON.
"""
from __future__ import annotations

from typing import Optional, Union

import pytest
from pydantic import BaseModel

from stitcher import Extractor
from stitcher.extractor import (
    JsonPatchResponse,
    _normalise_stringified_json,
    _try_parse_json_string,
)


# Module mark only applies to async tests; sync helper tests carry no mark.


class Item(BaseModel):
    name: str
    value: int


class Container(BaseModel):
    items: list[Item]
    name: str


# ---------- helper unit tests (no LLM) ----------


def test_normalises_stringified_object_in_list():
    parsed = {
        "items": ['{"name": "x", "value": 1}', '{"name": "y", "value": 2}'],
        "name": "c",
    }
    result = _normalise_stringified_json(parsed, Container)
    assert result == {
        "items": [{"name": "x", "value": 1}, {"name": "y", "value": 2}],
        "name": "c",
    }


def test_preserves_str_slot_even_if_content_looks_like_json():
    """A field annotated `str` is NOT touched even if its content is
    JSON-shaped \u2014 the schema explicitly allows strings here."""
    parsed = {"items": [], "name": '{"this": "is just a string"}'}
    result = _normalise_stringified_json(parsed, Container)
    assert result["name"] == '{"this": "is just a string"}'


def test_preserves_invalid_json_string():
    """A string starting with `{` that isn't valid JSON is left alone \u2014
    Pydantic will raise the original error and the patch loop handles it."""
    parsed = {"items": ["{not valid json"], "name": "c"}
    result = _normalise_stringified_json(parsed, Container)
    assert result["items"] == ["{not valid json"]


def test_preserves_string_in_union_when_str_is_a_member():
    """Union[str, Item] with a string value: the schema explicitly allows
    strings, so we don't try to coerce."""

    class WithUnion(BaseModel):
        payload: Union[str, Item]

    parsed = {"payload": '{"name": "x", "value": 1}'}
    result = _normalise_stringified_json(parsed, WithUnion)
    assert result["payload"] == '{"name": "x", "value": 1}'


def test_handles_optional_list_of_models():
    """Optional[list[Item]] with stringified items normalises correctly."""

    class WithOptional(BaseModel):
        items: Optional[list[Item]] = None

    parsed = {"items": ['{"name": "x", "value": 1}']}
    result = _normalise_stringified_json(parsed, WithOptional)
    assert result == {"items": [{"name": "x", "value": 1}]}


def test_recurses_into_nested_models():
    """Stringified JSON deep inside nested models is normalised."""

    class Inner(BaseModel):
        deep: list[Item]

    class Outer(BaseModel):
        inner: Inner

    parsed = {"inner": {"deep": ['{"name": "x", "value": 1}']}}
    result = _normalise_stringified_json(parsed, Outer)
    assert result == {"inner": {"deep": [{"name": "x", "value": 1}]}}


def test_handles_stringified_list_in_list_slot():
    """A whole list field arriving as a string `'[...]'` is re-parsed."""

    class Box(BaseModel):
        items: list[Item]

    parsed = {"items": '[{"name": "x", "value": 1}, {"name": "y", "value": 2}]'}
    result = _normalise_stringified_json(parsed, Box)
    assert result["items"] == [
        {"name": "x", "value": 1},
        {"name": "y", "value": 2},
    ]


def test_idempotent_on_already_normal_input():
    """Running normalise on an already-clean dict is a no-op."""
    parsed = {"items": [{"name": "x", "value": 1}], "name": "c"}
    result = _normalise_stringified_json(parsed, Container)
    assert result == parsed


def test_preserves_unknown_keys():
    """Extra keys not in the model schema are kept as-is so Pydantic's
    own extra-field handling decides what to do with them."""
    parsed = {"items": [], "name": "c", "extra_field": "kept"}
    result = _normalise_stringified_json(parsed, Container)
    assert result["extra_field"] == "kept"


def test_try_parse_json_string_rejects_non_container():
    """Helper rejects a string that parses to a scalar even if it's valid JSON
    \u2014 we only re-parse for object/array slots."""
    assert _try_parse_json_string("42", dict) is None
    assert _try_parse_json_string('"hello"', list) is None


def test_try_parse_json_string_requires_correct_prefix():
    """Helper requires `{` for dict and `[` for list \u2014 keeps it cheap."""
    assert _try_parse_json_string('[1, 2]', dict) is None
    assert _try_parse_json_string('{"a": 1}', list) is None


# ---------- integration tests (with FakeLLM, no patch turn needed) ----------


@pytest.mark.asyncio
async def test_initial_extract_with_stringified_objects_skips_patch_turn(fake_llm):
    """The motivating case end-to-end: model returns stringified-JSON in a
    list slot; normalisation fixes it before validation; patch loop never
    fires."""

    class Box(BaseModel):
        items: list[Item]

    fake_llm.set_scripts(
        initial=[{"items": ['{"name": "alice", "value": 30}', '{"name": "bob", "value": 25}']}],
        patch=[],   # MUST not be reached
    )
    extractor = Extractor(fake_llm, Box)

    result = await extractor.ainvoke([])

    assert result.value == Box(items=[Item(name="alice", value=30), Item(name="bob", value=25)])
    assert result.attempts == 1
    assert fake_llm.patch_runnable.calls == []


@pytest.mark.asyncio
async def test_aupdate_seeded_with_stringified_objects_normalises(fake_llm):
    """aupdate's seed dict is also passed through normalisation \u2014 a caller
    can pass a partially-stringified dict from elsewhere and have it
    handled the same way as fresh LLM output."""

    class Box(BaseModel):
        items: list[Item]

    fake_llm.set_scripts(
        initial=[],
        patch=[
            JsonPatchResponse(
                reasoning="(test)",
                operations=[{"op": "replace", "path": "/items/0/value", "value": 99}],
            )
        ],
    )
    extractor = Extractor(fake_llm, Box)

    # The seed has stringified objects; normalisation makes the FIRST patch
    # turn see a well-shaped dict, so the model's patch can target the
    # correct fields.
    await extractor.aupdate(
        existing={"items": ['{"name": "carol", "value": 30}']},
        messages=[],
    )

    # Verify the patch turn's AIMessage carried the NORMALISED prev_dict,
    # not the stringified form (the model's JSON Pointer paths target the
    # AIMessage's structure; if it were stringified the patch would fail).
    sent_input = fake_llm.patch_runnable.calls[0]["input"]
    prev_msg = sent_input[-2]
    assert prev_msg.__class__.__name__ == "AIMessage"
    assert '"name": "carol"' in prev_msg.content
    assert '"value": 30' in prev_msg.content
    # The stringified form should NOT appear in the AIMessage:
    assert '"{\\"name\\"' not in prev_msg.content
