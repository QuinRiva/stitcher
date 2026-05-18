"""Stitcher tags the ``with_structured_output`` plumbing as ``langsmith:hidden``.

The Langfuse callback handler (per langfuse-python PR #1077) honors
``langsmith:hidden`` by demoting tagged chain/tool/retriever spans to
``level="DEBUG"`` — which the Langfuse trace UI hides by default. The LLM
generation path is structurally exempt from the demotion, so the actual
model call stays at DEFAULT and visible.

Stitcher applies this tag to the ``with_structured_output(include_raw=True)``
chains in ``Extractor.__init__`` so the noisy LangChain plumbing (an
unavoidable consequence of asking for the raw AIMessage back —
RunnableSequence / RunnableParallel<raw> / RunnableWithFallbacks /
RunnableAssign / JsonOutputParser / internal RunnableLambdas) collapses to
a single "N hidden observations" hint in the trace tree.

The user-meaningful ``initial`` / ``patch`` labels stay visible because
they sit on outer ``RunnableLambda`` wrappers (``_invoke_initial`` /
``_invoke_patch``) that are NOT tagged hidden.

These tests assert the contract end-to-end via the FakeLLM, which sees the
merged config including context-var-propagated tags.
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


async def test_langsmith_hidden_tag_reaches_initial_extract(fake_llm):
    """The tag propagates through with_structured_output's pipeline to the LLM call."""
    fake_llm.set_scripts(initial=[{"name": "Alice", "age": 30}], patch=[])
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([])

    initial_tags = fake_llm.initial_runnable.calls[0]["config"]["tags"]
    assert "langsmith:hidden" in initial_tags


async def test_langsmith_hidden_tag_reaches_patch_turn(fake_llm):
    """Patch chain is independently tagged; tag survives even with no user-supplied tags."""
    fake_llm.set_scripts(
        initial=[{"name": "Bob", "age": -1}],
        patch=[JsonPatchResponse(
            reasoning="(test)",
            operations=[{"op": "replace", "path": "/age", "value": 30}],
        )],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([])

    patch_tags = fake_llm.patch_runnable.calls[0]["config"]["tags"]
    assert "langsmith:hidden" in patch_tags


async def test_initial_and_patch_labels_still_present(fake_llm):
    """The thin RunnableLambda wrappers preserve the ``initial`` and ``patch``
    labels on visible (DEFAULT-level) spans — without them, Langfuse would
    drop the labels into DEBUG alongside the rest of the plumbing.

    Verified by checking the run_name reached the FakeLLM (the inner call still
    carries the label via the explicit ``config=`` passed by the wrappers)."""
    fake_llm.set_scripts(
        initial=[{"name": "Carol", "age": -1}],
        patch=[JsonPatchResponse(
            reasoning="(test)",
            operations=[{"op": "replace", "path": "/age", "value": 25}],
        )],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([])

    assert fake_llm.initial_runnable.calls[0]["config"]["run_name"] == "initial"
    assert fake_llm.patch_runnable.calls[0]["config"]["run_name"] == "patch"


async def test_aupdate_patch_also_tagged_hidden(fake_llm):
    """aupdate's patch path uses the same tagged chain — tag must reach there too."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(
            reasoning="(test)",
            operations=[{"op": "replace", "path": "/age", "value": 31}],
        )],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.aupdate(
        existing=Person(name="Dave", age=30),
        messages=[],
    )

    patch_tags = fake_llm.patch_runnable.calls[0]["config"]["tags"]
    assert "langsmith:hidden" in patch_tags


async def test_user_tags_merge_with_langsmith_hidden(fake_llm):
    """User-supplied tags appear alongside ``langsmith:hidden``, not in place of it.
    Locks in the merge semantics so a future LangChain change to tag-merging
    behavior would surface here before silently breaking Langfuse trace cleanup."""
    fake_llm.set_scripts(initial=[{"name": "Eve", "age": 30}], patch=[])
    extractor = Extractor(fake_llm, Person)

    await extractor.ainvoke([], tags=["mine_a", "mine_b"])

    tags = fake_llm.initial_runnable.calls[0]["config"]["tags"]
    assert set(tags) >= {"langsmith:hidden", "mine_a", "mine_b"}
