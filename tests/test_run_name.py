"""``run_name`` kwarg names the parent extraction run; child LLM calls are
fixed-name (``initial`` / ``patch``) within the parent's trace.

Stitcher wraps each ``ainvoke``/``aupdate`` call in a ``RunnableLambda``
so LangChain's callback machinery establishes a parent run that the
initial extract and any patch turns become children of. The user's
``run_name`` becomes the parent's name; children are always ``initial``
and ``patch``. This ensures one Langfuse trace per ainvoke (rather than
N separate traces, which is what stitcher used to produce).
"""
from __future__ import annotations

import pytest
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


async def test_run_name_becomes_parent_name_initial_child_is_named_initial(
    fake_llm, capturing_callback
):
    """``run_name='X'`` names the parent run ``X``; the initial extract child is named ``initial``."""
    fake_llm.set_scripts(
        initial=[{"name": "Alice", "age": 30}],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person)

    result = await extractor.ainvoke(
        [], run_name="my-pipeline.extract_person", callbacks=[capturing_callback]
    )

    assert result.value == Person(name="Alice", age=30)
    # Parent run carries the user's run_name; it has no parent of its own.
    parent = next(
        e for e in capturing_callback.events if e["name"] == "my-pipeline.extract_person"
    )
    assert parent["parent_run_id"] is None
    # Inner extract is the fixed child name "initial" — names are
    # differentiators within the parent trace, not standalone trace names.
    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "initial"


async def test_patch_turns_are_named_patch_within_parent(fake_llm, capturing_callback):
    """Patch turns are children named ``patch`` (regardless of how many);
    the user's ``run_name`` lives on the parent."""
    fake_llm.set_scripts(
        initial=[{"name": "Bob", "age": -5}],
        patch=[
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -1}]),
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 7}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    result = await extractor.ainvoke(
        [], run_name="batch_42", callbacks=[capturing_callback]
    )

    assert result.value == Person(name="Bob", age=7)
    assert result.attempts == 3
    parent = next(e for e in capturing_callback.events if e["name"] == "batch_42")
    assert parent["parent_run_id"] is None
    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "initial"
    assert len(fake_llm.patch_runnable.calls) == 2
    for call in fake_llm.patch_runnable.calls:
        assert call["config"]["run_name"] == "patch"


async def test_run_name_default_when_unset(fake_llm, capturing_callback):
    """When ``run_name`` is not provided, parent defaults to ``stitcher``
    and children remain ``initial`` / ``patch``."""
    fake_llm.set_scripts(
        initial=[{"name": "Carol", "age": -1}],
        patch=[
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 99}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=3)

    result = await extractor.ainvoke([], callbacks=[capturing_callback])

    assert result.value == Person(name="Carol", age=99)
    parent = next(e for e in capturing_callback.events if e["name"] == "stitcher")
    assert parent["parent_run_id"] is None
    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "initial"
    assert fake_llm.patch_runnable.calls[0]["config"]["run_name"] == "patch"
