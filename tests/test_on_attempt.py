"""``on_attempt`` callback fires once per validation attempt.

Mirrors trustcall's ``on_attempt`` hook (with ``ai_message`` deliberately
omitted — see ``AttemptInfo`` docstring). Lets callers wide-log
per-attempt classifications (validation_failure vs patch_apply_failure vs
success) for observability — the canonical use case is a downstream
adjudicator that classifies failures across many attempts.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, model_validator

from pydantic import ValidationError

from stitcher import AttemptInfo, Extractor
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


async def test_on_attempt_fires_once_on_success(fake_llm):
    """One LLM call, validation passes \u2192 one on_attempt with is_success=True."""
    seen: list[AttemptInfo] = []
    fake_llm.set_scripts(initial=[{"name": "Alice", "age": 30}], patch=[])
    extractor = Extractor(fake_llm, Person, on_attempt=seen.append)

    result = await extractor.ainvoke([])

    assert result.value == Person(name="Alice", age=30)
    assert len(seen) == 1
    info = seen[0]
    assert info.attempt_number == 1
    assert info.parsed == {"name": "Alice", "age": 30}
    assert info.error is None
    assert info.is_success is True


async def test_on_attempt_fires_per_validation_attempt(fake_llm):
    """Initial extract + N patch turns \u2192 N+1 callbacks with monotone attempt_number."""
    seen: list[AttemptInfo] = []
    fake_llm.set_scripts(
        initial=[{"name": "Bob", "age": -5}],   # attempt 1: invalid
        patch=[
            JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": -1}]),  # attempt 2: invalid
            JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": 7}]),   # attempt 3: success
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5, on_attempt=seen.append)

    result = await extractor.ainvoke([])

    assert result.value == Person(name="Bob", age=7)
    assert [info.attempt_number for info in seen] == [1, 2, 3]
    assert [info.is_success for info in seen] == [False, False, True]
    # parsed reflects the dict that WAS validated on each attempt
    assert seen[0].parsed == {"name": "Bob", "age": -5}
    assert seen[1].parsed == {"name": "Bob", "age": -1}
    assert seen[2].parsed == {"name": "Bob", "age": 7}


async def test_on_attempt_validation_failure_passes_raw_exception(fake_llm):
    """On validation failure, error is the raw ValidationError — callers can
    introspect via .errors() to classify (type, loc, ctx['error'])."""
    seen: list[AttemptInfo] = []
    fake_llm.set_scripts(
        initial=[{"name": "Carol", "age": -5}],
        patch=[JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": 18}])],
    )
    extractor = Extractor(fake_llm, Person, on_attempt=seen.append)

    await extractor.ainvoke([])

    assert seen[0].is_success is False
    assert isinstance(seen[0].error, ValidationError)
    # Caller can introspect the structure they need:
    errs = seen[0].error.errors()
    assert len(errs) == 1
    assert errs[0]["type"] == "value_error"
    assert "non-negative" in errs[0]["msg"]


async def test_on_attempt_fires_for_patch_apply_failure(fake_llm):
    """Patch couldn't be applied (bad pointer) \u2192 on_attempt fires with parsed=None."""
    seen: list[AttemptInfo] = []
    fake_llm.set_scripts(
        initial=[{"name": "Dave", "age": -5}],
        patch=[
            # First patch: pointer doesn't exist \u2192 jsonpatch raises
            JsonPatchResponse(operations=[{"op": "replace", "path": "/nonexistent/field", "value": 1}]),
            # Second patch: valid recovery
            JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": 25}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5, on_attempt=seen.append)

    result = await extractor.ainvoke([])

    assert result.value.age == 25
    # 3 attempts: initial validate-fail, patch-apply-fail, patch-success
    assert [info.attempt_number for info in seen] == [1, 2, 3]
    assert [info.is_success for info in seen] == [False, False, True]
    # Attempt 2 is the patch-apply failure: parsed is None, error is a ValueError
    # (synthesized from the jsonpatch exception) — NOT a ValidationError. Callers
    # classifying failures should isinstance-discriminate to distinguish patch-apply
    # failures from Pydantic validation failures.
    assert seen[1].parsed is None
    assert isinstance(seen[1].error, ValueError)
    assert not isinstance(seen[1].error, ValidationError)
    assert "could not be applied" in str(seen[1].error).lower()


async def test_on_attempt_supports_async_callback(fake_llm):
    """An async on_attempt callable is awaited."""
    seen: list[AttemptInfo] = []

    async def async_handler(info: AttemptInfo) -> None:
        seen.append(info)

    fake_llm.set_scripts(initial=[{"name": "Eve", "age": 22}], patch=[])
    extractor = Extractor(fake_llm, Person, on_attempt=async_handler)

    await extractor.ainvoke([])
    assert len(seen) == 1
    assert seen[0].is_success is True


async def test_on_attempt_fires_for_aupdate(fake_llm):
    """The hook fires from aupdate's patch loop too (same _patch_loop)."""
    seen: list[AttemptInfo] = []
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(operations=[{"op": "replace", "path": "/age", "value": 99}])],
    )
    extractor = Extractor(fake_llm, Person, on_attempt=seen.append)

    await extractor.aupdate(existing={"name": "Frank", "age": 40}, messages=[])

    assert len(seen) == 1
    assert seen[0].attempt_number == 1
    assert seen[0].is_success is True
    assert seen[0].parsed == {"name": "Frank", "age": 99}


async def test_on_attempt_unset_no_overhead(fake_llm):
    """No on_attempt configured \u2192 the loop runs as before with no errors."""
    fake_llm.set_scripts(initial=[{"name": "Gina", "age": 50}], patch=[])
    extractor = Extractor(fake_llm, Person)  # no on_attempt

    result = await extractor.ainvoke([])
    assert result.value.age == 50
