"""Behaviour 3 — ``aupdate(initial_error=...)`` seeds the first patch turn.

In update mode the loop is seeded with an existing document, so the base loop's
first patch turn carries no validator header. ``initial_error`` delivers an
initial adjudication failure through that per-turn patch channel instead of the
system channel / ``original_messages``: it frames turn one with the same
``## Validation Errors`` prefix a validator failure would, and is REPLACED by
the real current error every subsequent turn (so nothing accumulates). With no
``initial_error`` the first patch prompt is byte-identical to today's.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, model_validator

from stitcher import Extractor
from stitcher.extractor import JsonPatchResponse, _build_patch_prompt


pytestmark = pytest.mark.asyncio


class Person(BaseModel):
    name: str
    age: int

    @model_validator(mode="after")
    def _positive(self):
        if self.age < 0:
            raise ValueError("age must be non-negative")
        return self


async def test_initial_error_frames_turn_one(fake_llm):
    """Turn one carries the ## Validation Errors prefix around the seeded
    error, matching _build_patch_prompt(initial_error, schema)."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(t)", operations=[{"op": "replace", "path": "/age", "value": 31}])],
    )
    ext = Extractor(fake_llm, Person)
    seed = ValueError("SEED-ERR: unmet_roles flat list")

    await ext.aupdate(existing={"name": "A", "age": 30}, messages=[], initial_error=seed)

    turn1 = fake_llm.patch_runnable.calls[0]["input"][-1].content
    assert "## Validation Errors" in turn1
    assert "SEED-ERR: unmet_roles flat list" in turn1
    # Exactly what the shared prompt builder would produce for that error.
    assert turn1 == _build_patch_prompt(seed, ext._schema_json)


async def test_initial_error_replaced_on_subsequent_turn(fake_llm):
    """Turn two carries the REAL current validation error, not the seed —
    the seed fires exactly once and never accumulates."""
    fake_llm.set_scripts(
        initial=[],
        patch=[
            # Turn 1 applies but leaves age invalid → forces a repair turn.
            JsonPatchResponse(reasoning="(t)", operations=[{"op": "replace", "path": "/age", "value": -1}]),
            # Turn 2 fixes it.
            JsonPatchResponse(reasoning="(t)", operations=[{"op": "replace", "path": "/age", "value": 7}]),
        ],
    )
    ext = Extractor(fake_llm, Person, max_attempts=5)

    result = await ext.aupdate(
        existing={"name": "A", "age": 30},
        messages=[],
        initial_error=ValueError("SEED-ERR"),
    )
    assert result.value == Person(name="A", age=7)

    turn1 = fake_llm.patch_runnable.calls[0]["input"][-1].content
    turn2 = fake_llm.patch_runnable.calls[1]["input"][-1].content
    assert "SEED-ERR" in turn1
    assert "SEED-ERR" not in turn2  # replaced by the real current error
    assert "age must be non-negative" in turn2


async def test_no_initial_error_is_byte_identical(fake_llm):
    """Without initial_error the first patch prompt is the headerless prompt
    (no ## Validation Errors), byte-identical to the shared builder's output."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(t)", operations=[{"op": "replace", "path": "/age", "value": 31}])],
    )
    ext = Extractor(fake_llm, Person)

    await ext.aupdate(existing={"name": "A", "age": 30}, messages=[])

    turn1 = fake_llm.patch_runnable.calls[0]["input"][-1].content
    assert "## Validation Errors" not in turn1
    assert turn1 == _build_patch_prompt(None, ext._schema_json)
