"""``run_name`` kwarg threads through every stitcher LLM call.

When set, both the initial extract and each patch turn use a name derived
from the user-provided ``run_name``. When unset, stitcher falls back to
its default names. This matters for langfuse trace differentiation when
the same Extractor pattern is invoked from multiple call sites.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, model_validator

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


async def test_run_name_threads_to_initial_call(fake_llm):
    """``run_name='X'`` causes the initial extract to be named ``X.initial``."""
    fake_llm.set_scripts(
        initial=[{"name": "Alice", "age": 30}],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person)

    result = await extractor.ainvoke([], run_name="my-pipeline.extract_person")

    assert result.value == Person(name="Alice", age=30)
    assert len(fake_llm.initial_runnable.calls) == 1
    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "my-pipeline.extract_person"


async def test_run_name_threads_to_patch_calls(fake_llm):
    """Each patch turn is named ``{run_name}.patch``, regardless of how many."""
    fake_llm.set_scripts(
        initial=[{"name": "Bob", "age": -5}],   # fails validation
        patch=[
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -1}]),  # still fails
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 7}]),   # passes
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    result = await extractor.ainvoke([], run_name="batch_42")

    assert result.value == Person(name="Bob", age=7)
    assert result.attempts == 3  # initial + 2 patches
    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "batch_42"
    assert len(fake_llm.patch_runnable.calls) == 2
    for call in fake_llm.patch_runnable.calls:
        assert call["config"]["run_name"] == "batch_42.patch"


async def test_run_name_default_when_unset(fake_llm):
    """When ``run_name`` is not provided, the base name defaults to
    ``stitcher``: initial extract is ``stitcher`` (bare), patch turns are
    ``stitcher.patch``."""
    fake_llm.set_scripts(
        initial=[{"name": "Carol", "age": -1}],
        patch=[
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 99}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=3)

    result = await extractor.ainvoke([])

    assert result.value == Person(name="Carol", age=99)
    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "stitcher"
    assert fake_llm.patch_runnable.calls[0]["config"]["run_name"] == "stitcher.patch"
