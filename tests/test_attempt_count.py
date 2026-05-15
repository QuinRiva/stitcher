"""``attempt_count`` is auto-injected into validation_context on every pass.

Mirrors trustcall's contract: validators can read ``info.context["attempt_count"]``
to implement first-attempt-strict / later-lenient patterns (e.g. an
adjudicator that pushes back on the LLM once but accepts its judgment on
retry). The key starts at 1 on the first validation and increments by one
per validation attempt.

Precedence on key collision: user-supplied ``attempt_count`` wins, matching
trustcall. This makes it possible to simulate "I'm on attempt N" in tests
without driving the loop that far.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationInfo, model_validator

from stitcher import Extractor
from stitcher.extractor import JsonPatchResponse


pytestmark = pytest.mark.asyncio


def _make_recording_schema(seen: list[int]) -> type[BaseModel]:
    """Build a Pydantic model whose validator records the attempt_count it
    saw and only passes once age >= 0."""

    class Recording(BaseModel):
        age: int

        @model_validator(mode="after")
        def _record(self, info: ValidationInfo):
            seen.append(info.context["attempt_count"])
            if self.age < 0:
                raise ValueError("age must be non-negative")
            return self

    return Recording


async def test_attempt_count_increments_across_patch_turns(fake_llm):
    """Each model_validate sees an attempt_count one higher than the last."""
    seen: list[int] = []
    Recording = _make_recording_schema(seen)

    fake_llm.set_scripts(
        initial=[{"age": -5}],   # attempt 1: invalid
        patch=[
            JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": -1}]),  # attempt 2: still invalid
            JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": 7}]),   # attempt 3: passes
        ],
    )
    extractor = Extractor(fake_llm, Recording, max_attempts=5)

    result = await extractor.ainvoke([])
    assert result.value.age == 7
    assert seen == [1, 2, 3]


async def test_attempt_count_starts_at_one_for_aupdate(fake_llm):
    """aupdate's first validation sees attempt_count=1 (the patched seed)."""
    seen: list[int] = []
    Recording = _make_recording_schema(seen)

    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": 25}])],
    )
    extractor = Extractor(fake_llm, Recording)

    await extractor.aupdate(existing={"age": 30}, messages=[])
    assert seen == [1]


async def test_attempt_count_user_override_wins(fake_llm):
    """User-supplied attempt_count overrides stitcher's loop counter."""
    seen: list[int] = []
    Recording = _make_recording_schema(seen)

    fake_llm.set_scripts(initial=[{"age": 10}], patch=[])
    extractor = Extractor(fake_llm, Recording)

    await extractor.ainvoke([], validation_context={"attempt_count": 99})
    assert seen == [99]


async def test_attempt_count_coexists_with_user_validation_context(fake_llm):
    """User keys are merged alongside attempt_count, both visible to the validator."""
    seen_contexts: list[dict] = []

    class S(BaseModel):
        x: int

        @model_validator(mode="after")
        def _record(self, info: ValidationInfo):
            seen_contexts.append(dict(info.context))
            return self

    fake_llm.set_scripts(initial=[{"x": 1}], patch=[])
    extractor = Extractor(fake_llm, S)

    await extractor.ainvoke([], validation_context={"trace_id": "abc-123"})
    assert seen_contexts == [{"attempt_count": 1, "trace_id": "abc-123"}]
