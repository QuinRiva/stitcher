"""Parse-failure semantics on both extract paths.

Stitcher binds the schema directly and self-parses the model's JSON content
(``_parse_content``): a JSON-decode failure (initial) or a failure to
validate into ``JsonPatchResponse`` (patch) surfaces as ``parsing_error``.
Stitcher's handling depends on which call hit the error:

- **Initial extract** (``ainvoke`` only \u2014 ``aupdate`` never enters this
  branch): treated as a hard failure of the seed. Symmetric with a
  catastrophic-weight ``ValidationError``: reset the happy path, consume an
  attempt, re-extract. If ``allow_re_extract`` is somehow disabled here, the
  error is raised. ``was_re_extracted`` flips to True even though the trigger
  was a parse failure, not a weight overflow \u2014 the public meaning is "we
  threw away an extract and tried again," which both paths satisfy.

- **Patch turn**: treated as a patch-protocol failure, structurally identical
  to ``jsonpatch.apply`` rejecting bad ops. Fed back to the next iteration
  as a corrective ``ValueError`` ("your response didn't match the required
  shape"), consuming an attempt. The malformed AIMessage stays on
  ``raw_messages`` so the caller can see what the model emitted.

These tests script an ``AIMessage`` with non-JSON ``content`` to drive the
parse failure on both paths.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, model_validator

from stitcher import Extractor
from stitcher.extractor import JsonPatchResponse


pytestmark = pytest.mark.asyncio


class Person(BaseModel):
    name: str
    age: int

    @model_validator(mode="after")
    def _positive(self):
        if self.age < 0:
            raise ValueError("age must be non-negative")
        return self


def _parse_error_envelope(content: str = "garbage") -> AIMessage:
    """An ``AIMessage`` whose non-JSON ``content`` makes stitcher's
    ``_parse_content`` raise on ``json.loads`` — the parse-failure path."""
    return AIMessage(content=content)


async def test_initial_parse_error_triggers_re_extract(fake_llm):
    """A parsing_error on the initial extract resets and re-extracts on the
    next attempt; the discarded message stays on raw_messages, was_re_extracted
    is True, and attempts counts both."""
    fake_llm.set_scripts(
        initial=[
            _parse_error_envelope(),                # attempt 1: parse-fails
            {"name": "Alice", "age": 30},           # attempt 2: clean
        ],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    result = await extractor.ainvoke([])

    assert result.value == Person(name="Alice", age=30)
    assert result.attempts == 2
    assert result.was_re_extracted is True
    # Both AIMessages on raw_messages \u2014 the discarded garbage and the clean one.
    ai_messages = [m for m in result.raw_messages if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    assert ai_messages[0].content == "garbage"


async def test_initial_parse_error_exhausts_attempts(fake_llm):
    """Persistent parse failures eventually exhaust max_attempts and raise."""
    fake_llm.set_scripts(
        initial=[_parse_error_envelope() for _ in range(3)],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=3)

    with pytest.raises(RuntimeError, match="exhausted 3 attempts"):
        await extractor.ainvoke([])


async def test_patch_parse_error_is_fed_back(fake_llm):
    """A parsing_error on the patch turn is folded into the patch-apply-failure
    path: consume an attempt, surface a corrective message, retry. The bad
    AIMessage stays on raw_messages."""
    fake_llm.set_scripts(
        initial=[{"name": "Alice", "age": -1}],     # forces a patch
        patch=[
            _parse_error_envelope("not json"),       # attempt 2: malformed patch response
            JsonPatchResponse(                       # attempt 3: valid
                reasoning="(test)",
                operations=[{"op": "replace", "path": "/age", "value": 30}],
            ),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    result = await extractor.ainvoke([])

    assert result.value == Person(name="Alice", age=30)
    assert result.attempts == 3
    assert result.was_re_extracted is False  # parse error was on patch, not initial
    # Bad patch response is preserved for debugging
    ai_msgs = [m for m in result.raw_messages if isinstance(m, AIMessage)]
    assert any(m.content == "not json" for m in ai_msgs)


async def test_aupdate_patch_parse_error_is_fed_back(fake_llm):
    """Same patch-side behaviour under aupdate: parse error consumes an attempt
    and the loop retries (aupdate has no re-extract path, so the patch-feedback
    path is the only recovery)."""
    fake_llm.set_scripts(
        initial=[],
        patch=[
            _parse_error_envelope(),
            JsonPatchResponse(
                reasoning="(test)",
                operations=[{"op": "replace", "path": "/age", "value": 31}],
            ),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    result = await extractor.aupdate(
        existing=Person(name="Alice", age=30),
        messages=[],
    )

    assert result.value == Person(name="Alice", age=31)
    assert result.attempts == 2
    assert result.was_re_extracted is False
