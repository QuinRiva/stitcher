"""Validation errors render with real newlines and JSON Pointer paths.

The previous renderer used ``json.dumps(errors)`` which escaped every ``\\n``
in user-supplied validator messages into literal ``\\n`` text — turning
multi-line messages (markdown, examples, hints) into unreadable single-line
walls. The model couldn't parse the structure.

The new renderer:
- Uses real newlines (no JSON escaping)
- Renders ``loc`` as a JSON Pointer the model can copy verbatim into a patch op
- Uses ``[1]`` / ``[2]`` numbering for cross-reference in the model's reasoning
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, model_validator

from stitcher import Extractor
from stitcher.extractor import (
    JsonPatchResponse,
    _build_patch_prompt,
    _format_validation_errors,
    _loc_to_json_pointer,
)


# ---------- unit tests for the helpers ----------


def test_loc_to_json_pointer_simple():
    assert _loc_to_json_pointer(("field_adjudications", 3)) == "/field_adjudications/3"


def test_loc_to_json_pointer_root_is_empty_string():
    assert _loc_to_json_pointer(()) == ""


def test_loc_to_json_pointer_escapes_special_chars():
    """RFC 6901: ``~`` becomes ``~0`` and ``/`` becomes ``~1``."""
    assert _loc_to_json_pointer(("a/b", "c~d")) == "/a~1b/c~0d"


def test_format_validation_errors_preserves_multiline_messages():
    """The whole point of the change: real newlines, not literal ``\\n`` text."""
    errs = [
        {
            "loc": ("field_adjudications", 3),
            "msg": "Value error, **Header**\n\nFollowed by:\n- Bullet one\n- Bullet two",
            "type": "value_error",
        }
    ]
    out = _format_validation_errors(errs)
    # Real newlines, not literal `\n` characters
    assert "\n\n" in out
    assert "\\n" not in out
    # Markdown content is preserved verbatim
    assert "**Header**" in out
    assert "- Bullet one" in out


def test_format_validation_errors_uses_bracket_numbering():
    """Headers use ``[1]`` / ``[2]`` so the model can cross-reference in reasoning."""
    errs = [
        {"loc": ("a",), "msg": "first", "type": "value_error"},
        {"loc": ("b",), "msg": "second", "type": "value_error"},
    ]
    out = _format_validation_errors(errs)
    assert out.startswith("[1]")
    assert "[2]" in out
    # Path comes from loc, type from the dict
    assert "/a" in out
    assert "/b" in out


def test_format_validation_errors_empty_list():
    """Defensive guard for empty errors() — should not happen in practice but
    stay polite if it does."""
    assert _format_validation_errors([]) == "(no specific errors reported)"


def test_format_validation_errors_root_loc_shows_root_label():
    """An empty loc renders as ``(root)`` rather than an empty path."""
    errs = [{"loc": (), "msg": "wrong shape at root", "type": "value_error"}]
    out = _format_validation_errors(errs)
    assert "(root)" in out


# ---------- integration: prompt builder uses the new format ----------


def test_build_patch_prompt_no_previous_block():
    """The patch prompt does NOT embed prev_dict (it lives in the AIMessage above)."""
    prompt = _build_patch_prompt(
        error=None,
        schema_json={"type": "object"},
    )
    assert "<previous>" not in prompt
    assert "<schema>" in prompt
    assert "the assistant message above" in prompt


def test_build_patch_prompt_repair_uses_new_error_format():
    """When error is set, the prompt uses the bracketed, real-newline format."""
    from pydantic import BaseModel, ValidationError

    class S(BaseModel):
        age: int

    try:
        S.model_validate({"age": "thirty"})
    except ValidationError as e:
        prompt = _build_patch_prompt(error=e, schema_json={"type": "object"})

    assert "Your previous JSON output failed validation" in prompt
    assert "[1]" in prompt
    assert "/age" in prompt
    # Confirm we're NOT using json.dumps — no stray quoted keys
    assert '"loc":' not in prompt
    assert '"msg":' not in prompt


# ---------- integration: end-to-end with FakeLLM ----------


@pytest.mark.asyncio
async def test_multiline_validator_message_arrives_with_real_newlines(fake_llm):
    """End-to-end: a validator that raises a multi-line ValueError produces
    a patch prompt where the message renders with real newlines, not literal
    ``\\n`` text."""

    class Person(BaseModel):
        age: int

        @model_validator(mode="after")
        def _check(self):
            if self.age < 0:
                raise ValueError(
                    "**Multi-line error**\n\nLine two\nLine three"
                )
            return self

    fake_llm.set_scripts(
        initial=[{"age": -1}],
        patch=[
            JsonPatchResponse(
                reasoning="(test)",
                operations=[{"op": "replace", "path": "/age", "value": 7}],
            )
        ],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([])

    patch_prompt = fake_llm.patch_runnable.calls[0]["input"][-1].content
    # The rendered error preserves real newlines from the validator
    assert "**Multi-line error**\n\nLine two\nLine three" in patch_prompt
    # And the path is in JSON Pointer form (root because model_validator(mode="after"))
    assert "[1] path (root)" in patch_prompt
