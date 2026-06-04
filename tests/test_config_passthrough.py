"""Arbitrary LangChain ``RunnableConfig`` fields pass through via ``**config``.

Stitcher has explicit kwargs for the two things it does itself (``run_name``
mangling for ``.initial``/``.patch``; ``validation_context`` plumbing into
Pydantic). Everything else is forwarded verbatim to LangChain's
``RunnableConfig`` on each LLM call \u2014 callbacks, tags, metadata,
configurable, recursion_limit, etc.

These tests verify the forwarding for the canonical observability fields
(tags, metadata) and confirm the same config reaches both the initial
extract and every patch turn.
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


async def test_tags_forwarded_to_initial_and_patch_turns(fake_llm):
    """``tags=[...]`` reaches both the initial extract and every patch turn."""
    fake_llm.set_scripts(
        initial=[{"name": "Alice", "age": -1}],   # forces a patch turn
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 30}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([], tags=["batch_42", "experiment_x"])

    # User tags reach both child calls verbatim. Stitcher no longer injects
    # any tags of its own (the ``langsmith:hidden`` trace hack is gone now
    # that each LLM call is a single, visible generation span).
    initial_tags = fake_llm.initial_runnable.calls[0]["config"]["tags"]
    patch_tags = fake_llm.patch_runnable.calls[0]["config"]["tags"]
    assert initial_tags == ["batch_42", "experiment_x"]
    assert patch_tags == ["batch_42", "experiment_x"]
    assert "langsmith:hidden" not in initial_tags
    assert "langsmith:hidden" not in patch_tags


async def test_metadata_forwarded_to_initial_and_patch_turns(fake_llm):
    """``metadata={...}`` reaches both the initial extract and every patch turn."""
    fake_llm.set_scripts(
        initial=[{"name": "Bob", "age": -1}],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 7}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke(
        [],
        metadata={"trace_id": "abc-123", "input_doc_id": "doc-7"},
    )

    expected = {"trace_id": "abc-123", "input_doc_id": "doc-7"}
    assert fake_llm.initial_runnable.calls[0]["config"]["metadata"] == expected
    assert fake_llm.patch_runnable.calls[0]["config"]["metadata"] == expected


async def test_arbitrary_config_kwarg_forwarded(fake_llm):
    """A field stitcher knows nothing about (e.g. ``configurable``) still flows through."""
    fake_llm.set_scripts(initial=[{"name": "Carol", "age": 30}], patch=[])
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([], configurable={"prompt_variant": "v3"})

    assert fake_llm.initial_runnable.calls[0]["config"]["configurable"] == {"prompt_variant": "v3"}


async def test_run_name_on_parent_children_named_initial_and_patch(fake_llm, capturing_callback):
    """User's ``run_name`` is the parent run's name; children are uniformly
    ``initial`` and ``patch``. Tags/metadata flow to both children via
    LangChain context inheritance."""
    fake_llm.set_scripts(
        initial=[{"name": "Dave", "age": -1}],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 18}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke(
        [],
        run_name="batch_99",
        tags=["alpha"],
        metadata={"k": "v"},
        callbacks=[capturing_callback],
    )

    parent = next(e for e in capturing_callback.events if e["name"] == "batch_99")
    assert parent["parent_run_id"] is None
    initial_cfg = fake_llm.initial_runnable.calls[0]["config"]
    patch_cfg = fake_llm.patch_runnable.calls[0]["config"]
    assert initial_cfg["run_name"] == "initial"
    assert patch_cfg["run_name"] == "patch"
    # User tags reach both verbatim; stitcher injects none of its own.
    assert initial_cfg["tags"] == ["alpha"] == patch_cfg["tags"]
    assert initial_cfg["metadata"] == {"k": "v"} == patch_cfg["metadata"]


async def test_aupdate_also_forwards_config(fake_llm):
    """``aupdate`` forwards ``**config`` to its patch turns too."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 50}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.aupdate(
        existing={"name": "Eve", "age": 40},
        messages=[],
        tags=["update_flow"],
        metadata={"actor": "system"},
    )

    cfg = fake_llm.patch_runnable.calls[0]["config"]
    assert cfg["tags"] == ["update_flow"]
    assert cfg["metadata"] == {"actor": "system"}


async def test_callbacks_reach_inner_calls(fake_llm, capturing_callback):
    """User-supplied callbacks are attached to the parent and inherited by
    inner calls via LangChain's context propagation — the callback receives
    on_chain_start events for the parent run."""
    fake_llm.set_scripts(initial=[{"name": "Frank", "age": 22}], patch=[])
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([], callbacks=[capturing_callback], run_name="X")

    # Parent chain start was observed (the user's callback fires for the
    # RunnableLambda parent that wraps the patch loop).
    assert any(
        e["kind"] == "chain" and e["name"] == "X" for e in capturing_callback.events
    )
